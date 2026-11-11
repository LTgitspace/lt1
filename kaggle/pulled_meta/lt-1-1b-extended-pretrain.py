"""
LT-1 (1.00B) Extended Pre-training & Conversational SFT Runner on Kaggle GPU.
Supports 100% offline execution via bundled lt1_src/ in the attached Kaggle dataset,
with GitHub clone fallback when outbound internet is active.
"""
import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path
import torch


def main():
    print("=" * 60)
    print("=== LT-1 (1.00B) Kaggle GPU Training Runner ===")
    print("=" * 60)

    # 1. Environment & Hardware Diagnostics
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    subprocess.run(["nvidia-smi"], check=False)
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            device_name = torch.cuda.get_device_name(i)
            vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {device_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("[ERROR] CUDA is NOT available in this container (PyTorch CPU build detected).")
        print("Why this happens on Kaggle even with enable_gpu=true:")
        print("  1. Your Kaggle account must be Phone Verified at https://www.kaggle.com/settings")
        print("     to unlock GPU accelerators and outbound Internet in Kernels.")
        print("  2. Or open https://www.kaggle.com/code/ltnanhtun/lt-1-1b-extended-pretrain/edit")
        print("     and select Accelerator -> GPU T4 x2 / P100 in the right-hand Session Options panel.")
        sys.exit(1)

    # 2. Locate Kaggle Input Dataset
    input_root = Path("/kaggle/input")
    found_train_bin = None
    for p in input_root.rglob("train_mix.bin"):
        found_train_bin = p
        break

    if not found_train_bin:
        print("[ERROR] train_mix.bin could not be found under /kaggle/input!")
        sys.exit(1)

    src_dir = found_train_bin.parent
    print(f"Located Kaggle dataset files in: {src_dir}")

    # 3. Setup Working Codebase (Prefer offline lt1_src/ or lt1_src.zip from dataset, fallback to git clone)
    working_dir = Path("/kaggle/working")
    repo_dir = working_dir / "lt1"
    unpacked_src = src_dir / "lt1_src"
    src_zip = src_dir / "lt1_src.zip"

    if unpacked_src.exists():
        print(f"Copying bundled source code from {unpacked_src} into {repo_dir}...")
        shutil.copytree(unpacked_src, repo_dir, dirs_exist_ok=True)
    elif src_zip.exists():
        print(f"Extracting bundled source code from {src_zip} into {repo_dir}...")
        repo_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src_zip, "r") as zf:
            zf.extractall(repo_dir)
    elif not repo_dir.exists():
        print(f"Cloning LT-1 from GitHub into {repo_dir}...")
        subprocess.run(["git", "clone", "https://github.com/LTgitspace/lt1.git", str(repo_dir)], check=True)

    os.chdir(str(repo_dir))
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    # 4. Symlink Dataset Files into repo_dir/data
    data_dir = repo_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    target_files = ["tokenizer_cosmo.json", "tokenizer.json", "train_mix.bin", "sft_mix.pt"]
    for fname in target_files:
        s = src_dir / fname
        d = data_dir / fname
        if s.exists():
            if d.exists() or d.is_symlink():
                d.unlink()
            print(f"Symlinking: {s.name} -> data/{d.name}")
            d.symlink_to(s)

    # 5. Prepare Output Checkpoints Directory in /kaggle/working
    ckpt_dir = repo_dir / "checkpoints_1b"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 6. Execute Extended Pre-training (25,000 steps) chained into SFT (1,000 steps)
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
