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
