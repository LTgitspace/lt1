import os
import time
import argparse
import torch
from src.config import base_config, LT1Config
from src.model import LT1Model
from src.tokenizer import NanoTokenizer
from src.generate import generate_text


def export_checkpoint(
    ckpt_path: str,
    out_path: str,
    tokenizer_path: str = "data/tokenizer_cosmo.json",
    device: str | None = None,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg: LT1Config = ckpt.get("config", base_config())
    model = LT1Model(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    raw_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
    print(f"Original checkpoint size: {raw_size_mb:.2f} MB")

    tok = None
    if os.path.exists(tokenizer_path):
        tok = NanoTokenizer.load(tokenizer_path)
    elif os.path.exists("data/tokenizer.json"):
        tok = NanoTokenizer.load("data/tokenizer.json")

    # Benchmark raw inference if tokenizer available
    pre_speed = None
    if tok is not None:
        model.to(device)
        model.eval()
        t0 = time.perf_counter()
        _ = generate_text(model, tok, "Once upon a time", max_new_tokens=25, num_loops=2, device=device)
        dt = time.perf_counter() - t0
        pre_speed = 25.0 / max(1e-5, dt)
        print(f"Pre-export inference speed: {pre_speed:.1f} tok/s")

    # Pack model for zero-sync inference
    print("Packing model for zero-sync inference (3D BMM MoE + Fast-Path NCM + LUT Engram)...")
    model.pack_for_inference(device=device)

    # Benchmark post-export inference
    post_speed = None
    if tok is not None:
        t0 = time.perf_counter()
        _ = generate_text(model, tok, "Once upon a time", max_new_tokens=25, num_loops=2, device=device)
        dt = time.perf_counter() - t0
        post_speed = 25.0 / max(1e-5, dt)
        speedup = (post_speed / max(1e-5, pre_speed)) if pre_speed else 1.0
        print(f"Post-export inference speed: {post_speed:.1f} tok/s ({speedup:.2f}x speedup)")

    # Save compact exported artifact
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    export_payload = {
        "config": cfg,
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "is_exported_inference": True,
        "step": ckpt.get("step", 0),
        "step_loss": ckpt.get("step_loss", 0.0),
    }
    torch.save(export_payload, out_path)
    export_size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"Exported checkpoint saved to: {out_path} ({export_size_mb:.2f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="LT-1 Inference Optimization & Export CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Path to input checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/latest_inference.pt", help="Path to exported checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer_cosmo.json", help="Path to tokenizer")
    args = parser.parse_args()

    export_checkpoint(args.checkpoint, args.output, tokenizer_path=args.tokenizer)


if __name__ == "__main__":
    main()
