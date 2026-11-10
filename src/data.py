import os
import numpy as np
import torch
from src.tokenizer import NanoTokenizer


class MemmapDataset:
    """
    Zero-copy memory-mapped dataset reading packed uint16 binary token arrays.
    """
    def __init__(self, bin_path: str, seq_len: int):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.data)
        assert self.num_tokens > seq_len + 1, f"File {bin_path} has {self.num_tokens} tokens, need > {seq_len + 1}"

    def get_batch(self, batch_size: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        max_idx = self.num_tokens - self.seq_len - 1
        start_indices = np.random.randint(0, max_idx + 1, size=batch_size)

        x_list = []
        y_list = []
        for idx in start_indices:
            chunk = self.data[idx : idx + self.seq_len + 1].astype(np.int64)
            x_list.append(chunk[:-1])
            y_list.append(chunk[1:])

        x = torch.from_numpy(np.stack(x_list)).to(device)
        y = torch.from_numpy(np.stack(y_list)).to(device)
        return x, y


def encode_sft_turn(
    tokenizer: NanoTokenizer,
    user_text: str,
    assistant_text: str,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes a (user, assistant) turn with user prompt targets masked to -100.
    Guarantees >= 1 unmasked assistant token even under heavy prompt truncation.
    """
    user_ids = [tokenizer.user_id] + tokenizer.encode(user_text) + [tokenizer.eos_id]
    asst_ids = [tokenizer.assistant_id] + tokenizer.encode(assistant_text) + [tokenizer.eos_id]

    # Ensure assistant text has at least 1 token budget
    if len(asst_ids) >= max_seq_len:
        asst_ids = asst_ids[:max_seq_len - 1] + [tokenizer.eos_id]

    max_user_len = max_seq_len - len(asst_ids)
    if len(user_ids) > max_user_len:
        # Left-truncate prompt to fit context budget
        user_ids = [tokenizer.user_id] + user_ids[-max_user_len + 1:]

    full_ids = user_ids + asst_ids
    target_ids = [-100] * len(user_ids) + asst_ids

    # Pad if shorter than max_seq_len
    if len(full_ids) < max_seq_len:
        pad_len = max_seq_len - len(full_ids)
        full_ids = full_ids + [tokenizer.pad_id] * pad_len
        target_ids = target_ids + [-100] * pad_len

    return (
        torch.tensor(full_ids[:max_seq_len], dtype=torch.long),
        torch.tensor(target_ids[:max_seq_len], dtype=torch.long),
    )


def prepare_dataset(
    dataset_name: str = "roneneldan/TinyStories",
    subset: str | None = None,
    split: str = "train",
    output_bin: str = "data/train.bin",
    tokenizer_path: str = "data/tokenizer.json",
    max_samples: int = 25000,
    vocab_size: int = 4096,
) -> tuple[str, str]:
    """
    Streams a dataset from HuggingFace, trains a byte-level BPE tokenizer (if missing),
    and packs token IDs into a memory-mapped uint16 binary shard.
    """
    from datasets import load_dataset

    os.makedirs(os.path.dirname(os.path.abspath(output_bin)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(tokenizer_path)), exist_ok=True)

    print(f"Loading stream from {dataset_name} (subset={subset}, split={split})...")
    if subset:
        ds = load_dataset(dataset_name, subset, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=True)

    texts = []
    print(f"Streaming {max_samples:,} samples for tokenizer training & packing...")
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        text = item.get("text", "")
        if text and len(text.strip()) > 10:
            texts.append(text.strip())

    if not os.path.exists(tokenizer_path):
        print(f"Training byte-level BPE tokenizer (vocab_size={vocab_size})...")
        tok = NanoTokenizer.train_from_iterator(texts, vocab_size=vocab_size, save_path=tokenizer_path)
    else:
        print(f"Loading existing tokenizer from {tokenizer_path}...")
        tok = NanoTokenizer.load(tokenizer_path)

    print(f"Tokenizing {len(texts):,} texts into uint16 tokens...")
    all_token_ids = []
    for text in texts:
        tokens = tok.encode(text) + [tok.eos_id]
        all_token_ids.extend(tokens)

    print(f"Packing {len(all_token_ids):,} tokens to {output_bin}...")
    np.array(all_token_ids, dtype=np.uint16).tofile(output_bin)
    print(f"Dataset preparation complete: {output_bin} ({len(all_token_ids):,} tokens, {os.path.getsize(output_bin) / (1024**2):.2f} MB)")
    return output_bin, tokenizer_path

