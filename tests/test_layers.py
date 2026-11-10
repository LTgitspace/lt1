import unittest
import torch
from src.config import LT1Config, base_config, tiny_smoke_config
from src.layers import RMSNorm, BitLinear, make_linear, precompute_rope_cos_sin, apply_rope


class TestLayers(unittest.TestCase):
    def test_configs_and_primitives(self):
        cfg = base_config()
        self.assertEqual(cfg.vocab_size, 16384)
        self.assertEqual(cfg.n_encoder_layers, 4)
        self.assertEqual(cfg.n_recurrent_layers, 4)
        self.assertEqual(cfg.n_coda_layers, 2)
        self.assertEqual(cfg.d_model, 384)
        self.assertEqual(cfg.n_heads, 6)
        self.assertEqual(cfg.n_kv_heads, 2)
        self.assertEqual(cfg.head_dim, 64)
        self.assertEqual(cfg.engram_buckets, 65536)

        # RMSNorm check
        x = torch.randn(2, 8, 64, requires_grad=True)
        norm = RMSNorm(64)
        y = norm(x)
        self.assertEqual(y.shape, (2, 8, 64))

        # BitLinear check
        bl = BitLinear(64, 32, bias=False)
        out = bl(y)
        out.sum().backward()
        self.assertEqual(out.shape, (2, 8, 32))
        self.assertIsNotNone(bl.weight.grad)
        self.assertGreater(bl.weight.grad.abs().sum().item(), 0)

        # RoPE norm preservation check
        cos, sin = precompute_rope_cos_sin(64, 16)
        q = torch.randn(2, 4, 8, 64)
        q_rot = apply_rope(q, cos[:8], sin[:8])
        self.assertEqual(q_rot.shape, q.shape)
        self.assertTrue(torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4))

    def test_make_linear_factory(self):
        dense = make_linear(32, 16, use_bitlinear=False)
        self.assertIsInstance(dense, torch.nn.Linear)
        bit = make_linear(32, 16, use_bitlinear=True)
        self.assertIsInstance(bit, BitLinear)


if __name__ == "__main__":
    unittest.main()
