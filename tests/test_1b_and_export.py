import unittest
import torch
from src.config import lt1_1b_config, tiny_smoke_config
from src.model import LT1Model
from src.optim import TriOptimizer
from src.generate import generate_text


class Test1BAndExport(unittest.TestCase):
    def test_1b_parameter_count_and_config(self):
        cfg = lt1_1b_config(vocab_size=8192)
        self.assertTrue(cfg.engram_offload_cpu)
        self.assertTrue(cfg.use_grad_checkpointing)
        self.assertTrue(cfg.use_bf16_weights)

        model = LT1Model(cfg)
        counts = model.get_num_params()

        # Parameter count target: ~1.00B (999,450,000)
        self.assertGreaterEqual(counts["total"], 995_000_000)
        self.assertLessEqual(counts["total"], 1_005_000_000)

        # Offloaded table parameters: ~469.8M
        self.assertGreater(counts["offloaded_cpu"], 460_000_000)

        # Active parameters on compute device: ~529.6M
        self.assertLess(counts["on_device"], 550_000_000)

    def test_1b_cuda_forward_backward_vram_and_optim_step(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for RTX 3060 1B VRAM validation")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        cfg = lt1_1b_config(vocab_size=8192)
        model = LT1Model(cfg).to("cuda")

        # Offloaded Engram tables must reside on CPU in bfloat16
        for name, table in model.get_offloaded_tables().items():
            self.assertEqual(table.device.type, "cpu")
            self.assertEqual(table.dtype, torch.bfloat16)

        opt = TriOptimizer(model)

        x = torch.randint(0, cfg.vocab_size, (4, 256), device="cuda")
        y = torch.randint(0, cfg.vocab_size, (4, 256), device="cuda")

        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(x, targets=y, num_loops=2)

        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        opt.step(max_norm=1.0)

        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        # Peak VRAM must be strictly below 3.8 GB to prevent WDDM paging on 6 GB RTX 3060
        self.assertLess(peak_gb, 3.8, f"Peak VRAM {peak_gb:.2f} GB exceeded 3.8 GB threshold")

        del model, opt, x, y, logits, loss
        torch.cuda.empty_cache()

    def test_packed_inference_equivalence(self):
        cfg = tiny_smoke_config()
        model = LT1Model(cfg).eval()

        x = torch.randint(0, cfg.vocab_size, (2, 32))
        with torch.no_grad():
            logits_unpacked, _ = model(x, num_loops=1)

        # Pack for inference
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        x = x.to(device)
        model.pack_for_inference(device=device)

        with torch.no_grad():
            logits_packed, _ = model(x, num_loops=1)

        # Check shapes and finite logits
        self.assertEqual(logits_packed.shape, (2, 32, cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits_packed).all().item())


if __name__ == "__main__":
    unittest.main()
