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
        self.assertIn(id(model.ncm_bridge.next_concept_head.weight), adamw_params)

    def test_muon_decoupled_weight_decay_and_fp32_adamw_no_collapse(self):
        cfg = tiny_smoke_config()
        cfg.use_bf16_weights = True
        cfg.use_fp32_master_weights = True
        model = NanoCEDModel(cfg)
        opt = TriOptimizer(model, muon_lr=0.015, adamw_lr=4e-4, weight_decay=0.01)

        # Verify AdamW-updated norms/embeddings/routers stay in fp32
        self.assertEqual(model.tok_emb.weight.dtype, torch.float32)
        self.assertEqual(model.final_norm.weight.dtype, torch.float32)
        self.assertEqual(model.ncm_bridge.next_concept_head.weight.dtype, torch.float32)

        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))

        # Step 0 CE loss must be near ln(vocab_size) ~ 6.24, not >100
        _, _ = model(x, targets=y, num_loops=1)
        self.assertLess(model.last_loss_dict["ce_loss"], 7.5)

        init_norm = model.final_norm.weight.clone()
        for _ in range(15):
            opt.zero_grad()
            _, loss = model(x, targets=y, num_loops=1)
            loss.backward()
            opt.step(max_norm=1.0)

        # Routed experts must NOT collapse to 0.0000 and RMSNorm must actually update
        routed_std = model.encoder_layers[0].moe.routed_experts[0].w_gate.weight.float().std().item()
        self.assertGreater(routed_std, 0.005, "Routed expert weights collapsed under Muon!")
        self.assertFalse(torch.allclose(init_norm, model.final_norm.weight), "RMSNorm weights froze under BF16!")

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
