import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
from src.config import tiny_smoke_config
from src.tokenizer import NanoTokenizer
from src.data import MemmapDataset, encode_sft_turn
from src.model import LT1Model
from src.optim import TriOptimizer
from src.generate import generate_text


class TestDataAndPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sft_masking_never_all_ignored(self):
        texts = ["Hello world! This is a simple test sentence for BPE training."] * 50
        tok_path = os.path.join(self.test_dir, "tokenizer.json")
        tok = NanoTokenizer.train_from_iterator(texts, vocab_size=512, save_path=tok_path)

        long_user = "Hello " * 200
        assistant = "Short answer."
        inp, tgt = encode_sft_turn(tok, long_user, assistant, max_seq_len=32)
        self.assertEqual(inp.shape, (32,))
        self.assertEqual(tgt.shape, (32,))
        self.assertGreaterEqual((tgt != -100).sum().item(), 1)

        # Verify causal 1-token next-token shift on normal turns
        inp2, tgt2 = encode_sft_turn(tok, "Hello world", "Short answer", max_seq_len=32)
        for i in range(31):
            if tgt2[i].item() != -100 and inp2[i + 1].item() != tok.pad_id:
                self.assertEqual(tgt2[i].item(), inp2[i + 1].item())

    def test_end_to_end_overfit_convergence_and_generation(self):
        torch.manual_seed(1337)
        cfg = tiny_smoke_config()
        cfg.vocab_size = 512
        texts = ["The quick brown fox jumps over the lazy dog."] * 100
        tok_path = os.path.join(self.test_dir, "tok.json")
        tok = NanoTokenizer.train_from_iterator(texts, vocab_size=512, save_path=tok_path)

        ids = []
        for t in texts:
            ids.extend(tok.encode(t) + [tok.eos_id])
        bin_path = os.path.join(self.test_dir, "train.bin")
        np.array(ids, dtype=np.uint16).tofile(bin_path)

        ds = MemmapDataset(bin_path, seq_len=16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LT1Model(cfg).to(device)
        opt = TriOptimizer(model, muon_lr=0.02, adamw_lr=3e-3, sparse_lr=1e-2)

        x, y = ds.get_batch(4, device)
        _, initial_loss = model(x, targets=y, num_loops=2)
        for step in range(25):
            opt.zero_grad()
            loops = 1 if (step % 2 == 0) else 2
            _, loss = model(x, targets=y, num_loops=loops)
            loss.backward()
            opt.step()

        self.assertLess(loss.item(), initial_loss.item() * 0.6)
        out_str = generate_text(model, tok, "The quick", max_new_tokens=6, num_loops=4, temperature=0.0)
        self.assertGreater(len(out_str), len("The quick"))


if __name__ == "__main__":
    unittest.main()
