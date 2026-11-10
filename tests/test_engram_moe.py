import unittest
import torch
from src.config import tiny_smoke_config
from src.engram_moe import compute_ngram_hashes, EngramModule, DeepSeekMoE


class TestEngramMoE(unittest.TestCase):
    def test_engram_single_token_and_context_hashing(self):
        cfg = tiny_smoke_config()
        seq = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        full_hashes = compute_ngram_hashes(seq, cfg.engram_buckets)
        self.assertEqual(full_hashes.shape, (1, 4, 2))

        # Single-token step with prev_tokens=[20, 30] and curr=[40] must match position 3 of full sequence
        step_hash = compute_ngram_hashes(seq[:, 3:4], cfg.engram_buckets, prev_tokens=seq[:, 1:3])
        self.assertTrue(torch.equal(full_hashes[:, 3:4, :], step_hash))

        mod = EngramModule(cfg)
        x = torch.randn(1, 4, cfg.d_model, requires_grad=True)
        out = mod(x, full_hashes)
        out.sum().backward()
        self.assertTrue(mod.table.weight.grad.is_sparse)
        self.assertEqual(mod.table.weight.grad.dtype, torch.float32)

    def test_deepseek_moe_selection_only_bias_and_gradient_purity(self):
        cfg = tiny_smoke_config()
        moe = DeepSeekMoE(cfg, top_k=2)
        moe.train()
        # 15 tokens * 2 = 30 slots, cannot be evenly divided across 4 experts (guarantees imbalance)
        x = torch.randn(1, 15, cfg.d_model, requires_grad=True)
        bias_before = moe.expert_bias.clone()
        out = moe(x)
        out.sum().backward()
        self.assertFalse(torch.equal(moe.expert_bias, bias_before))
        self.assertIsNotNone(moe.router.weight.grad)

        # Verify selection-only bias: small perturbation to expert_bias that does NOT change top-k winner order
        # must produce 0.0 change in forward output
        moe.eval()
        with torch.no_grad():
            moe.expert_bias.zero_()
            out1 = moe(x)
            moe.expert_bias.add_(1e-5)
            out2 = moe(x)
            self.assertTrue(torch.allclose(out1, out2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
