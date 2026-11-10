import argparse
import os
import torch
import torch.nn.functional as F
from src.model import LT1Model, CEDKVCache
from src.tokenizer import NanoTokenizer
from src.config import base_config


def sample_next_token(logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> int:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1).item()

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        probs = probs.masked_fill(indices_to_remove, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    next_tok = torch.multinomial(probs, num_samples=1)
    return next_tok.item()


@torch.no_grad()
def generate_text(
    model: LT1Model,
    tokenizer: NanoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    num_loops: int = 2,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str | None = None,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id]

    max_ctx = model.cfg.max_seq_len
    if len(prompt_ids) >= max_ctx:
        prompt_ids = prompt_ids[-(max_ctx - 1):]
    max_steps = min(max_new_tokens, max_ctx - len(prompt_ids))

    tokens = list(prompt_ids)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    cache = CEDKVCache(model.cfg.n_encoder_layers)
    amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    with torch.amp.autocast(device_type=device if device == "cuda" else "cpu", dtype=amp_dtype):
        # 1. Prefill all prompt tokens except the last one
        if x.shape[1] > 1:
            model(x[:, :-1], num_loops=num_loops, kv_cache=cache)
            curr_token = x[:, -1:]
        else:
            curr_token = x

        # 2. Autoregressive token generation loop
        for _ in range(max_steps):
            logits, _ = model(curr_token, num_loops=num_loops, kv_cache=cache)
            next_id = sample_next_token(logits[0, -1, :], temperature=temperature, top_p=top_p)
            tokens.append(next_id)
            if next_id == tokenizer.eos_id:
                break
            curr_token = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return tokenizer.decode(tokens)


def main():
    import time
    parser = argparse.ArgumentParser(description="LT-1 Text Generation CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
    parser.add_argument("--loops", type=int, default=2, choices=[1, 2, 4], help="Dory elastic loop count")
    parser.add_argument("--max-tokens", type=int, default=80, help="Maximum generated tokens")
    parser.add_argument("--temp", "--temperature", dest="temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top_p")
    parser.add_argument("--interactive", action="store_true", help="Run interactive playground session")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = NanoTokenizer.load(args.tokenizer)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = ckpt.get("config", base_config())
        cfg.vocab_size = tok.vocab_size
        model = LT1Model(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        step_num = ckpt.get("step", "unknown")
        step_loss = ckpt.get("step_loss", 0.0)
        print(f"[Loaded] {args.checkpoint} (Step {step_num}, Loss {step_loss:.4f}) on {device}")
    else:
        cfg = base_config()
        cfg.vocab_size = tok.vocab_size
        model = LT1Model(cfg).to(device)
        print(f"[Warning] Checkpoint {args.checkpoint} not found, using random weights.")

    if args.interactive:
        loops = args.loops
        temp = args.temp
        max_tokens = args.max_tokens
        print("======================================================")
        print("         LT-1 Interactive Generation Playground       ")
        print("======================================================")
        print("Commands: /loops [1|2|4] | /temp [0.1-1.5] | /quit")
        print("------------------------------------------------------")
        while True:
            try:
                user_in = input(f"\n[k={loops}, T={temp}] Prompt > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not user_in:
                user_in = "Once upon a time"
            if user_in.lower() in ("/quit", "/exit", "exit", "quit"):
                break
            if user_in.startswith("/loops"):
                parts = user_in.split()
                if len(parts) > 1 and parts[1] in ("1", "2", "4"):
                    loops = int(parts[1])
                    print(f"Recurrent loops set to k={loops}")
                else:
                    print("Usage: /loops 1 | /loops 2 | /loops 4")
                continue
            if user_in.startswith("/temp"):
                parts = user_in.split()
                if len(parts) > 1:
                    try:
                        temp = float(parts[1])
                        print(f"Temperature set to {temp}")
                    except ValueError:
                        print("Invalid temperature float.")
                continue

            t0 = time.perf_counter()
            output = generate_text(
                model, tok, user_in,
                max_new_tokens=max_tokens,
                num_loops=loops,
                temperature=temp,
                top_p=args.top_p,
                device=device,
            )
            dt = time.perf_counter() - t0
            gen_tokens = len(tok.encode(output)) - len(tok.encode(user_in))
            speed = max(1, gen_tokens) / max(1e-5, dt)
            print("------------------------------------------------------")
            print(output)
            print("------------------------------------------------------")
            print(f"[Generated {gen_tokens} tokens in {dt:.2f}s | {speed:.1f} tok/s | Loops k={loops}]")
    else:
        t0 = time.perf_counter()
        output = generate_text(
            model, tok, args.prompt,
            max_new_tokens=args.max_tokens,
            num_loops=args.loops,
            temperature=args.temp,
            top_p=args.top_p,
            device=device,
        )
        dt = time.perf_counter() - t0
        print("------------------------------------------------------")
        print(output)
        print("------------------------------------------------------")
        print(f"[Completed in {dt:.2f}s | Loops k={args.loops}]")



if __name__ == "__main__":
    main()
