import unittest
import torch
from src.config import tiny_smoke_config
from src.ncm import EMACodebook, CausalMultiscaleRVQ, ParallelMultiscaleNCM, causal_left_window_pool


class TestNCM(unittest.TestCase):
    def test_gate1_codebook_dead_code_restart_and_perplexity(self):
        # Initialize codebook with tiny dead threshold
        cb = EMACodebook(num_codes=32, code_dim=16, decay=0.9, dead_threshold=2.0)
        cb.train()

        # Simulate collapsed data: only a single repeated vector
        collapsed_input = torch.ones(64, 16)
        _, _, _, m1 = cb(collapsed_input)

        # Now feed diverse vectors: dead-code restart should activate and revive unused codes
        diverse_input = torch.randn(128, 16)
        for _ in range(5):
            _, _, _, m2 = cb(diverse_input)

        # Active ratio must recover significantly
        self.assertGreater(m2["active_ratio"], 0.3)

    def test_gate2_monotonic_multiscale_residual_reduction(self):
        cfg = tiny_smoke_config()
        rvq = CausalMultiscaleRVQ(cfg)
        rvq.eval()

        h_enc = torch.randn(2, 16, cfg.d_model)
        residual = h_enc.clone()
        res_norms = []

        with torch.no_grad():
            for scale_w, cb in zip(rvq.scales, rvq.codebooks):
                pooled = causal_left_window_pool(residual, scale_w)
                q_s, _, _, _ = cb(pooled)
                residual = residual - q_s
                res_norms.append(residual.norm().item())

        # Norm of residual must drop monotonically after quantizing each scale
        for i in range(len(res_norms) - 1):
            self.assertLessEqual(res_norms[i+1], res_norms[i] + 1e-4)

    def test_gate3_causal_pooling_strict_zero_future_leakage(self):
        torch.manual_seed(42)
        x1 = torch.randn(1, 16, 32)
        x2 = x1.clone()
        # Mutate future tokens 8..15
        x2[0, 8:, :] = torch.randn(8, 32)

        for w in (1, 2, 4):
            p1 = causal_left_window_pool(x1, window_size=w)
            p2 = causal_left_window_pool(x2, window_size=w)
            # Positions 0..7 must remain 100% identical
            self.assertTrue(torch.allclose(p1[:, :8, :], p2[:, :8, :], atol=1e-6))

    def test_gate4_parallel_multiscale_ncm_generation_pass_count(self):
        cfg = tiny_smoke_config()
        ncm = ParallelMultiscaleNCM(cfg, block_len=16, scales=(4, 8, 16))
        ctx = torch.randn(2, cfg.d_model)

        logits, passes = ncm.parallel_generate(ctx)
        self.assertEqual(logits.shape, (2, 16, cfg.vocab_size))
        # Must generate 16 tokens in exactly 3 forward passes (not 16)
        self.assertEqual(passes, 3)


if __name__ == "__main__":
    unittest.main()
