"""
LT-1 (1.00B) Extended Pre-training & Conversational SFT Runner on Kaggle GPU.
Handles repository cloning, dataset discovery/symlinking, execution, and artifact tracking.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
import torch

def main():
    print("=" * 60)
    print("=== LT-1 (1.00B) Kaggle GPU Training Runner ===")
    print("=" * 60)

    # 1. Environment & Hardware Diagnostics
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Active GPU: {device_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("[WARNING] CUDA not detected! Check GPU accelerator setting in Kaggle.")

    # 2. Clone or Update LT-1 Repository
    working_dir = Path("/kaggle/working")
    repo_dir = working_dir / "lt1"

    if not repo_dir.exists():
        print(f"Cloning LT-1 from GitHub into {repo_dir}...")
        subprocess.run(["git", "clone", "https://github.com/LTgitspace/lt1.git", str(repo_dir)], check=True)
    else:
        print(f"Repository already exists at {repo_dir}. Pulling latest...")
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=False)

    os.chdir(str(repo_dir))
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    # 3. Locate and Symlink Kaggle Input Dataset Files
    input_root = Path("/kaggle/input")
    data_dir = repo_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    found_train_bin = None
    for p in input_root.rglob("train_mix.bin"):
        found_train_bin = p
        break

    if not found_train_bin:
        print("[ERROR] train_mix.bin could not be found under /kaggle/input!")
        sys.exit(1)

    src_dir = found_train_bin.parent
    print(f"Located Kaggle dataset files in: {src_dir}")

    target_files = ["tokenizer_cosmo.json", "tokenizer.json", "train_mix.bin", "sft_mix.pt"]
    for fname in target_files:
        s = src_dir / fname
        d = data_dir / fname
        if s.exists():
            if d.exists() or d.is_symlink():
                d.unlink()
            print(f"Symlinking: {s.name} -> data/{d.name}")
            d.symlink_to(s)

    # 4. Prepare Output Checkpoints Directory in /kaggle/working
    ckpt_dir = repo_dir / "checkpoints_1b"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 5. Execute Extended Pre-training (25,000 steps) chained into SFT (1,000 steps)
    print("\n" + "=" * 60)
    print("=== Launching Extended Pre-training (25,000 Steps) ===")
    print("Batch: 8 | SeqLen: 256 | GradAccum: 4 | EffectiveBatch: 32")
    print("=" * 60 + "\n")

    cmd = [
        sys.executable, "-u", "-m", "src.train",
        "--stage", "pretrain",
        "--scale", "1b",
        "--steps", "25000",
        "--batch-size", "8",
        "--seq-len", "256",
        "--grad-accum-steps", "4",
        "--loops", "2",
        "--save-interval", "2500",
        "--eval-interval", "500",
        "--data-bin", "data/train_mix.bin",
        "--tokenizer-path", "data/tokenizer_cosmo.json",
        "--checkpoint-dir", "checkpoints_1b",
        "--log-file", "train_1b_kaggle.log",
        "--chain-sft",
        "--sft-pt", "data/sft_mix.pt",
        "--sft-steps", "1000",
    ]

    print("Executing command:\n" + " ".join(cmd) + "\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)

    proc.stdout.close()
    ret = proc.wait()

    print("\n" + "=" * 60)
    print(f"Execution finished with return code {ret}")
    if ret == 0:
        print("Checkpoints saved to /kaggle/working/lt1/checkpoints_1b")
    print("=" * 60)
    sys.exit(ret)

if __name__ == "__main__":
    main()
