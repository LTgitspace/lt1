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
    Encodes a (user, assistant) turn with prompt masking (-100) and causal 1-token target shift.
    For every unmasked position i, target_ids[i] predicts full_seq[i + 1].
    Guarantees >= 1 unmasked assistant token even under heavy prompt truncation.
    """
    user_ids = [tokenizer.user_id] + tokenizer.encode(user_text) + [tokenizer.eos_id]
    asst_ids = [tokenizer.assistant_id] + tokenizer.encode(assistant_text) + [tokenizer.eos_id]

    total_budget = max_seq_len + 1
    # Ensure assistant text has at least 2 tokens and user prompt keeps >= 1 context token
    if len(asst_ids) >= total_budget:
        asst_ids = asst_ids[:total_budget - 2] + [tokenizer.eos_id]

    max_user_len = total_budget - len(asst_ids)
    if len(user_ids) > max_user_len:
        # Left-truncate prompt to fit context budget
        user_ids = [tokenizer.user_id] + user_ids[-(max_user_len - 1):] if max_user_len > 1 else [tokenizer.user_id]

    full_seq = user_ids + asst_ids
    raw_targets = [-100] * len(user_ids) + asst_ids

    # Shift for next-token prediction: x[i] predicts full_seq[i + 1]
    inp_ids = full_seq[:-1]
    tgt_ids = raw_targets[1:]

    # Pad if shorter than max_seq_len
    if len(inp_ids) < max_seq_len:
        pad_len = max_seq_len - len(inp_ids)
        inp_ids = inp_ids + [tokenizer.pad_id] * pad_len
        tgt_ids = tgt_ids + [-100] * pad_len

    return (
        torch.tensor(inp_ids[:max_seq_len], dtype=torch.long),
        torch.tensor(tgt_ids[:max_seq_len], dtype=torch.long),
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


class SFTDataset:
    """
    Tensor dataset for conversational instruction tuning with prompt masking.
    """
    def __init__(self, pt_path: str):
        data = torch.load(pt_path, weights_only=True)
        self.input_ids = data["input_ids"]
        self.target_ids = data["target_ids"]
        self.num_samples = len(self.input_ids)

    def __len__(self) -> int:
        return self.num_samples

    def get_batch(self, batch_size: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.num_samples, (batch_size,))
        x = self.input_ids[indices].to(device)
        y = self.target_ids[indices].to(device)
        return x, y


def prepare_multidomain_datasets(
    cosmo_bin: str = "data/train_cosmo.bin",
    tokenizer_path: str = "data/tokenizer_cosmo.json",
    out_train_bin: str = "data/train_mix.bin",
    out_sft_pt: str = "data/sft_mix.pt",
    max_finance_samples: int = 20000,
    max_daily_samples: int = 15000,
    max_sft_turns: int = 20000,
    seq_len: int = 256,
) -> tuple[str, str]:
    """
    Blends General Knowledge (Cosmopedia v2) with Finance (Finance-Alpaca) and
    Daily Conversation (DialogSum + UltraChat) for 1B pre-training and conversational SFT.
    """
    from datasets import load_dataset
    import random

    os.makedirs(os.path.dirname(os.path.abspath(out_train_bin)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_sft_pt)), exist_ok=True)

    tok = NanoTokenizer.load(tokenizer_path)

    # 1. Stream Finance & Markets (gbharti/finance-alpaca)
    print(f"Streaming Finance dataset: gbharti/finance-alpaca ({max_finance_samples:,} samples)...")
    ds_fin = load_dataset("gbharti/finance-alpaca", split="train", streaming=True)
    fin_texts = []
    fin_sft_pairs = []
    for i, row in enumerate(ds_fin):
        if i >= max_finance_samples:
            break
        instr = row.get("instruction", "").strip()
        inp = row.get("input", "").strip()
        out = row.get("output", "").strip()
        if not out:
            continue
        u_text = f"{instr}\n{inp}".strip() if inp else instr
        if u_text and len(out) > 5:
            fin_texts.append(f"Financial Knowledge: {u_text}\nAnswer: {out}")
            fin_sft_pairs.append((u_text, out))

    # 2. Stream Daily Conversation (knkarthick/dialogsum + HuggingFaceH4/ultrachat_200k)
    print(f"Streaming Daily Conversation: knkarthick/dialogsum & ultrachat_200k ({max_daily_samples:,} samples)...")
    ds_diag = load_dataset("knkarthick/dialogsum", split="train", streaming=True)
    daily_texts = []
    daily_sft_pairs = []
    for i, row in enumerate(ds_diag):
        if i >= max_daily_samples // 2:
            break
        dial = row.get("dialogue", "").strip()
        summ = row.get("summary", "").strip()
        if dial and len(dial) > 20:
            daily_texts.append(f"Dialogue:\n{dial}")
            # Pair summary or conversation turns
            lines = [l.strip() for l in dial.split("\n") if ":" in l]
            if len(lines) >= 2:
                speaker1, msg1 = lines[0].split(":", 1)
                speaker2, msg2 = lines[1].split(":", 1)
                daily_sft_pairs.append((msg1.strip(), msg2.strip()))

    ds_ultra = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    for i, row in enumerate(ds_ultra):
        if i >= max_daily_samples // 2:
            break
        msgs = row.get("messages", [])
        if len(msgs) >= 2:
            u_msg = msgs[0].get("content", "").strip()
            a_msg = msgs[1].get("content", "").strip()
            if u_msg and a_msg:
                daily_texts.append(f"Conversation:\nUser: {u_msg}\nAssistant: {a_msg}")
                daily_sft_pairs.append((u_msg, a_msg))

    # 3. Interleave and Pack Pre-Training Shard (data/train_mix.bin)
    print(f"Tokenizing new Finance ({len(fin_texts):,}) and Daily Conversation ({len(daily_texts):,}) texts...")
    new_tokens = []
    for t in fin_texts + daily_texts:
        new_tokens.extend(tok.encode(t) + [tok.eos_id])

    print(f"Loading Cosmopedia v2 tokens from {cosmo_bin}...")
    cosmo_tokens = np.fromfile(cosmo_bin, dtype=np.uint16)
    print(f"Cosmopedia: {len(cosmo_tokens):,} tokens | Finance & Chat: {len(new_tokens):,} tokens")

    # Split into chunks of 1024 tokens and interleave for uniform domain representation
    chunk_sz = 1024
    chunks = []
    for i in range(0, len(cosmo_tokens), chunk_sz):
        chunks.append(cosmo_tokens[i : i + chunk_sz])
    new_arr = np.array(new_tokens, dtype=np.uint16)
    for i in range(0, len(new_arr), chunk_sz):
        chunks.append(new_arr[i : i + chunk_sz])

    random.seed(42)
    random.shuffle(chunks)
    mixed_arr = np.concatenate(chunks)
    mixed_arr.tofile(out_train_bin)
    print(f"Packed multi-domain pre-training shard: {out_train_bin} ({len(mixed_arr):,} tokens, {os.path.getsize(out_train_bin) / (1024**2):.2f} MB)")

    # 4. Build and Pack Conversational SFT Shard (data/sft_mix.pt)
    print("Building balanced Conversational SFT shard (data/sft_mix.pt)...")
    sft_pairs = fin_sft_pairs[: max_sft_turns // 2] + daily_sft_pairs[: max_sft_turns // 2]
    random.shuffle(sft_pairs)
    sft_pairs = sft_pairs[:max_sft_turns]

    in_list = []
    tgt_list = []
    for u, a in sft_pairs:
        x, y = encode_sft_turn(tok, u, a, max_seq_len=seq_len)
        in_list.append(x)
        tgt_list.append(y)

    sft_payload = {
        "input_ids": torch.stack(in_list),
        "target_ids": torch.stack(tgt_list),
    }
    torch.save(sft_payload, out_sft_pt)
    print(f"Packed conversational SFT shard: {out_sft_pt} ({len(in_list):,} turns, {os.path.getsize(out_sft_pt) / (1024**2):.2f} MB)")
    return out_train_bin, out_sft_pt


