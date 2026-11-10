import unittest
import torch
from src.config import base_config, tiny_smoke_config
from src.model import NanoCEDModel
from src.optim import TriOptimizer


class TestOptimAndVRAM(unittest.TestCase):
    def test_tri_optimizer_parameter_partitioning(self):
        cfg = tiny_smoke_config()
        model = NanoCEDModel(cfg)
        opt = TriOptimizer(model)

        all_model_params = {id(p) for p in model.parameters() if p.requires_grad}
        muon_params = {id(p) for g in opt.muon.param_groups for p in g["params"]} if opt.muon else set()
        adamw_params = {id(p) for g in opt.adamw.param_groups for p in g["params"]} if opt.adamw else set()
        sparse_params = {id(p) for g in opt.sparse_adam.param_groups for p in g["params"]} if opt.sparse_adam else set()

        self.assertTrue(muon_params.isdisjoint(adamw_params))
        self.assertTrue(muon_params.isdisjoint(sparse_params))
        self.assertTrue(adamw_params.isdisjoint(sparse_params))
        self.assertEqual(muon_params | adamw_params | sparse_params, all_model_params)
        self.assertIn(id(model.tok_emb.weight), adamw_params)

    def test_tri_optimizer_bf16_autocast_step_and_vram_ceiling(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for RTX 3060 VRAM test")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        cfg = base_config()
        model = NanoCEDModel(cfg).cuda()
        opt = TriOptimizer(model)

        x = torch.randint(0, cfg.vocab_size, (8, 512), device="cuda")
        y = torch.randint(0, cfg.vocab_size, (8, 512), device="cuda")

        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y, num_loops=2)
        loss.backward()
        opt.step()

        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        # Assert strictly below 3.2 GB ceiling on the 6 GB RTX 3060 (leaving > 2.8 GB headroom)
        self.assertLess(peak_gb, 3.2, f"Peak VRAM {peak_gb:.2f} GB exceeded 3.2 GB ceiling on 6GB GPU")


if __name__ == "__main__":
    unittest.main()
