import argparse
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

    tokens = list(prompt_ids)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    cache = CEDKVCache(model.cfg.n_encoder_layers)

    # 1. Prefill all prompt tokens except the last one
    if x.shape[1] > 1:
        model(x[:, :-1], num_loops=num_loops, kv_cache=cache)
        curr_token = x[:, -1:]
    else:
        curr_token = x

    # 2. Autoregressive token generation loop
    for _ in range(max_new_tokens):
        logits, _ = model(curr_token, num_loops=num_loops, kv_cache=cache)
        next_id = sample_next_token(logits[0, -1, :], temperature=temperature, top_p=top_p)
        tokens.append(next_id)
        if next_id == tokenizer.eos_id:
            break
        curr_token = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return tokenizer.decode(tokens)


def main():
    parser = argparse.ArgumentParser(description="LT-1 Text Generation CLI")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
    parser.add_argument("--loops", type=int, default=2, choices=[1, 2, 4], help="Dory elastic loop count")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum generated tokens")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    cfg = base_config()
    model = LT1Model(cfg)
    # Default fallback demo tokenizer
    tok = NanoTokenizer.train_from_iterator(["Once upon a time in a small kingdom."], vocab_size=cfg.vocab_size)
    output = generate_text(model, tok, args.prompt, max_new_tokens=args.max_tokens, num_loops=args.loops, temperature=args.temp)
    print("LT-1 Output:", output)


if __name__ == "__main__":
    main()
