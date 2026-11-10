import argparse
import os
import time
import math
import numpy as np
import torch
from src.config import base_config, tiny_smoke_config, LT1Config
from src.model import LT1Model
from src.optim import TriOptimizer
from src.tokenizer import NanoTokenizer
from src.data import MemmapDataset
from src.generate import generate_text
from src.ncm import ParallelMultiscaleNCM


SMOKE_TEXTS = [
    "Once upon a time, there was a little robot named Dolores.",
    "The robot loved to brew coffee for all the villagers in the morning.",
    "Every day, the steam boiler warmed up to ninety-three degrees Celsius.",
    "The flow rate sensor measured five hundred milliliters per minute.",
    "The coffee machine platform operated reliably across all kiosks in town.",
    "A distributed system must handle network partitions with grace.",
    "Virtual threads on JDK 25 provide high concurrency with clean blocking syntax.",
    "The robotic cobot moved smoothly to pick up the ceramic cup.",
    "N-gram conditional memory allows fast lexical lookup with zero matrix FLOPs.",
    "Next-concept modeling generates coarse-to-fine representations in parallel.",
] * 200


def get_cosine_lr_scale(step: int, total_steps: int, warmup_steps: int = 10, min_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


def run_smoke_training(steps: int = 50, batch_size: int = 4, seq_len: int = 64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting LT-1 Smoke Verification on device: {device}")

    data_dir = "C:\\github\\lt1\\data"
    os.makedirs(data_dir, exist_ok=True)
    tok_path = os.path.join(data_dir, "smoke_tokenizer.json")
    bin_path = os.path.join(data_dir, "smoke_train.bin")

    # 1. Train or load tokenizer
    if not os.path.exists(tok_path):
        print("Training BPE tokenizer on smoke corpus...")
        tok = NanoTokenizer.train_from_iterator(SMOKE_TEXTS, vocab_size=512, save_path=tok_path)
    else:
        tok = NanoTokenizer.load(tok_path)

    # 2. Pack smoke binary shard
    if not os.path.exists(bin_path):
        print("Packing uint16 binary shard...")
        ids = []
        for line in SMOKE_TEXTS:
            ids.extend(tok.encode(line) + [tok.eos_id])
        np.array(ids, dtype=np.uint16).tofile(bin_path)

    # 3. Model & TriOptimizer setup
    cfg = tiny_smoke_config()
    cfg.vocab_size = tok.vocab_size
    cfg.max_seq_len = seq_len
    model = LT1Model(cfg).to(device)
    optimizer = TriOptimizer(model, muon_lr=0.02, adamw_lr=3e-3, sparse_lr=1e-2)
    dataset = MemmapDataset(bin_path, seq_len=seq_len)

    counts = model.count_parameters()
    print(f"Model parameters: {counts['total']:,} total ({counts['dense_backbone']:,} dense + {counts['sparse_engram']:,} sparse Engram)")

    # 4. Training loop
    model.train()
    start_time = time.time()
    for step in range(steps):
        t0 = time.time()
        x, y = dataset.get_batch(batch_size, device=device)

        # Learning rate schedule
        lr_scale = get_cosine_lr_scale(step, steps, warmup_steps=5)
        optimizer.set_lr_scale(lr_scale)

        optimizer.zero_grad()
        # Randomize loops in {1, 2} to train elastic depth
        loops = 1 if (step % 2 == 0) else 2

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
            logits, loss = model(x, targets=y, num_loops=loops)

        loss.backward()
        optimizer.step(max_norm=1.0)

        step_time = time.time() - t0
        toks_per_sec = (batch_size * seq_len) / max(1e-5, step_time)

        if (step + 1) % 10 == 0 or step == 0:
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
            print(f"Step {step+1:03d}/{steps:03d} | Loss: {loss.item():.4f} | Throughput: {toks_per_sec:.0f} tok/s | Peak VRAM: {vram_mb:.1f} MB | Loops: {loops}")

    total_time = time.time() - start_time
    print(f"Smoke training complete in {total_time:.1f}s!")

    # 5. Elastic Loop Inference verification
    print("\n--- Testing Elastic Recurrent Loop Generation ---")
    for loop_k in (1, 2, 4):
        gen = generate_text(model, tok, "Once upon a time", max_new_tokens=25, num_loops=loop_k, temperature=0.7)
        print(f"[Loop k={loop_k}] -> {gen}")


def main():
    parser = argparse.ArgumentParser(description="LT-1 Training & Benchmark Entrypoint")
    parser.add_argument("--stage", type=str, default="smoke", choices=["smoke", "pretrain", "sft", "ncm-bench"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    if args.stage == "smoke":
        run_smoke_training(steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len)
    elif args.stage == "ncm-bench":
        print("Running NCM Multiscale Benchmark...")
        cfg = tiny_smoke_config()
        ncm = ParallelMultiscaleNCM(cfg, block_len=16, scales=(4, 8, 16))
        ctx = torch.randn(2, cfg.d_model)
        logits, passes = ncm.parallel_generate(ctx)
        print(f"NCM parallel generation: 16 tokens generated in {passes} passes! Output shape: {logits.shape}")


if __name__ == "__main__":
    main()
