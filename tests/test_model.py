import unittest
import torch
from src.config import tiny_smoke_config
from src.model import NanoCEDModel, CEDKVCache


class TestModel(unittest.TestCase):
    def test_strict_causality_no_future_leakage(self):
        torch.manual_seed(42)
        cfg = tiny_smoke_config()
        model = NanoCEDModel(cfg).eval()

        seq1 = torch.randint(0, cfg.vocab_size, (1, 12))
        seq2 = seq1.clone()
        seq2[0, 7:] = torch.randint(0, cfg.vocab_size, (5,)) # Mutate future tokens 7..11

        for loops in (1, 2, 4):
            with torch.no_grad():
                logits1, _ = model(seq1, num_loops=loops)
                logits2, _ = model(seq2, num_loops=loops)
            # Positions 0..6 must be 100% identical with zero future leakage across CED bridge or Dory loops
            self.assertTrue(
                torch.allclose(logits1[:, :7, :], logits2[:, :7, :], atol=1e-5),
                f"Causal leak detected at num_loops={loops}",
            )

    def test_last_loop_kv_cache_shapes_and_consistency(self):
        torch.manual_seed(42)
        cfg = tiny_smoke_config()
        model = NanoCEDModel(cfg).eval()
        seq = torch.randint(0, cfg.vocab_size, (1, 8))

        # Test loops=1 for exact parallel vs step-by-step equivalence
        with torch.no_grad():
            full_logits1, _ = model(seq, num_loops=1)
            cache1 = CEDKVCache(cfg.n_encoder_layers)
            prefill_logits1, _ = model(seq[:, :5], num_loops=1, kv_cache=cache1)
            self.assertEqual(cache1.seq_len, 5)
            self.assertEqual(cache1.anchor_kv[0].shape[2], 5)

            step_logits1 = []
            for t in range(5, 8):
                lg, _ = model(seq[:, t:t+1], num_loops=1, kv_cache=cache1)
                step_logits1.append(lg)
            decoded_logits1 = torch.cat([prefill_logits1] + step_logits1, dim=1)
            self.assertTrue(torch.allclose(full_logits1, decoded_logits1, atol=1e-4))

        # Test loops=2 for Last-Loop KV cache shape invariants (flat cache across loops)
        with torch.no_grad():
            full_logits2, _ = model(seq, num_loops=2)
            cache2 = CEDKVCache(cfg.n_encoder_layers)
            prefill_logits2, _ = model(seq[:, :5], num_loops=2, kv_cache=cache2)
            # Prefill matches full sequence exactly
            self.assertTrue(torch.allclose(full_logits2[:, :5], prefill_logits2, atol=1e-4))
            self.assertEqual(cache2.seq_len, 5)
            # Last-loop KV must have length exactly equal to 5, NOT 5 * num_loops (10)!
            self.assertEqual(cache2.anchor_kv[0].shape[2], 5)

            for t in range(5, 8):
                model(seq[:, t:t+1], num_loops=2, kv_cache=cache2)
            # Final KV cache length must be exactly total sequence length (8), NOT 16!
            self.assertEqual(cache2.anchor_kv[0].shape[2], 8)

    def test_model_parameter_counts(self):
        cfg = tiny_smoke_config()
        model = NanoCEDModel(cfg)
        counts = model.count_parameters()
        self.assertIn("total", counts)
        self.assertIn("sparse_engram", counts)
        self.assertIn("dense_backbone", counts)
        self.assertEqual(counts["total"], counts["sparse_engram"] + counts["dense_backbone"])


if __name__ == "__main__":
    unittest.main()
