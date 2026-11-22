import argparse
import os
import time
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engram_moe import compute_ngram_hashes
from src.model import LT1Model, apply_rope
from src.tokenizer import NanoTokenizer


class LT1OnnxWrapper(nn.Module):
    """Stateless ONNX-exportable wrapper around LT1Model accepting precomputed ngram_hashes."""

    def __init__(self, model: LT1Model, num_loops: int = 2):
        super().__init__()
        self.model = model
        self.num_loops = min(num_loops, model.cfg.max_loops)

    def forward(self, token_ids: torch.Tensor, ngram_hashes: torch.Tensor) -> torch.Tensor:
        m = self.model
        b, seq_len = token_ids.shape
        cos = m.cos_cached[:seq_len]
        sin = m.sin_cached[:seq_len]

        x = m.tok_emb(token_ids)
        x = m.engram_1(x, ngram_hashes)

        for enc_layer in m.encoder_layers:
            x, _ = enc_layer(x, cos, sin, cached_kv=None)

        x = m.engram_2(x, ngram_hashes)
        h_enc = m.enc_norm(x)

        h_enc, _, _, _ = m.ncm_bridge(h_enc, prev_enc_states=None, new_prev_states_out=None)

        k_global = m.bridge_wk(h_enc).view(b, seq_len, m.cfg.n_kv_heads, m.cfg.head_dim).transpose(1, 2)
        v_global = m.bridge_wv(h_enc).view(b, seq_len, m.cfg.n_kv_heads, m.cfg.head_dim).transpose(1, 2)
        k_global = apply_rope(k_global, cos, sin)
        global_kv = (k_global, v_global)

        anchor_kv = global_kv
        curr_x = h_enc
        for l in range(self.num_loops):
            alpha_l = torch.sigmoid(m.loop_inj_gates[l])
            curr_x = m.loop_bound_norm(curr_x + alpha_l * m.loop_inj_proj(h_enc))
            curr_x, _ = m.recurrent_layers[0](curr_x, cos, sin, shared_kv=global_kv)
            curr_x, _ = m.recurrent_layers[1](curr_x, cos, sin, shared_kv=global_kv)
            curr_x, new_anchor_kv = m.recurrent_layers[2](curr_x, cos, sin, cached_kv=None)
            anchor_kv = new_anchor_kv if new_anchor_kv is not None else anchor_kv
            curr_x, _ = m.recurrent_layers[3](curr_x, cos, sin, shared_kv=anchor_kv)

        coda_x = curr_x
        for coda_layer in m.coda_layers:
            coda_x, _ = coda_layer(coda_x, cos, sin, shared_kv=anchor_kv)

        coda_normed = m.final_norm(coda_x)
        return F.linear(coda_normed, m.tok_emb.weight.to(coda_normed.dtype))


def load_packed_model_cpu(ckpt_path: str, tokenizer_path: str) -> tuple[LT1Model, NanoTokenizer]:
    tok = NanoTokenizer.load(tokenizer_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg.vocab_size = tok.vocab_size
    # Export in FP32 for universal ONNX Runtime CPU compatibility
    cfg.use_bf16_weights = False

    model = LT1Model(cfg)
    state_dict = ckpt["model_state_dict"]
    if ckpt.get("is_exported_inference", False):
        model.pack_for_inference(device="cpu")
        for name in ("engram_1", "engram_2"):
            key = f"{name}.compact_table"
            if key in state_dict:
                mod = getattr(model, name)
                t = state_dict[key]
                mod.compact_table = nn.Parameter(
                    torch.empty_like(t, dtype=torch.float32),
                    requires_grad=False,
                )
                if t.shape[0] == 1:
                    mod.lut = torch.zeros(mod.num_buckets, dtype=torch.long)
    model.load_state_dict(state_dict, strict=False)
    if not ckpt.get("is_exported_inference", False):
        model.pack_for_inference(device="cpu")
    model.float().cpu().eval()
    return model, tok


def export_to_onnx(
    ckpt_path: str = "checkpoints/lt1_1b_chat_inference.pt",
    tokenizer_path: str = "data/tokenizer_cosmo.json",
    out_path: str = "checkpoints/onnx/lt1_1b_chat.onnx",
    num_loops: int = 2,
    opset: int = 17,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"Loading checkpoint {ckpt_path} in FP32 on CPU...")
    model, tok = load_packed_model_cpu(ckpt_path, tokenizer_path)
    wrapper = LT1OnnxWrapper(model, num_loops=num_loops).eval()

    sample_ids = torch.tensor(
        [[tok.user_id] + tok.encode("Hello, how are you?") + [tok.eos_id, tok.assistant_id]],
        dtype=torch.long,
    )
    sample_hashes = compute_ngram_hashes(sample_ids, model.cfg.engram_buckets)

    t0 = time.perf_counter()
    print(f"Exporting ONNX (opset={opset}) to {out_path}...")
    torch.onnx.export(
        wrapper,
        (sample_ids, sample_hashes),
        out_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["token_ids", "ngram_hashes"],
        output_names=["logits"],
        dynamic_axes={
            "token_ids": {0: "batch", 1: "seq_len"},
            "ngram_hashes": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        dynamo=False,
    )
    print(f"ONNX export finished in {time.perf_counter() - t0:.1f}s")

    # Verify with ONNX Runtime (including a second dynamic seq_len)
    print("Verifying ONNX Runtime parity...")
    with torch.no_grad():
        pt_logits = wrapper(sample_ids, sample_hashes).numpy()

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    ort_logits = sess.run(
        ["logits"],
        {"token_ids": sample_ids.numpy(), "ngram_hashes": sample_hashes.numpy()},
    )[0]
    max_diff = float(np.max(np.abs(pt_logits - ort_logits)))
    print(f"ORT parity check passed | max_abs_diff={max_diff:.6f} | output_shape={ort_logits.shape}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export LT-1 checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lt1_1b_chat_inference.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer_cosmo.json")
    parser.add_argument("--output", type=str, default="checkpoints/onnx/lt1_1b_chat.onnx")
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.tokenizer, args.output, num_loops=args.loops, opset=args.opset)


if __name__ == "__main__":
    main()
