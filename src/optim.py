import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW, SparseAdam
from src.model import LT1Model


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz quintic iteration for orthogonalizing 2D weight gradients in Muon.
    Calculated in bfloat16 for fast tensor-core execution.
    """
    assert len(G.shape) == 2, "Muon orthogonalization requires 2D matrices"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz.
    Uses a single momentum buffer per 2D weight matrix (half the memory of AdamW).
    """
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if weight_decay != 0:
                    g = g.add(p, alpha=weight_decay)

                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                u_orth = zeropower_via_newtonschulz5(update, steps=ns_steps)
                # Aspect ratio scaling
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(u_orth, alpha=-lr * scale)


class TriOptimizer:
    """
    Disjoint Tri-Optimizer orchestrator:
    1. SparseAdam: updates sparse EngramModule tables in fp32.
    2. Muon: updates 2D hidden projection matrices with Newton-Schulz orthogonalization.
    3. AdamW: updates 1D norms, router weights, loop gates, and tied token embeddings.
    """
    def __init__(
        self,
        model: LT1Model,
        muon_lr: float = 0.02,
        adamw_lr: float = 3e-4,
        sparse_lr: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        self.model = model
        self.base_lrs = {"muon": muon_lr, "adamw": adamw_lr, "sparse": sparse_lr}

        sparse_params = []
        muon_params = []
        adamw_params = []

        # Identify sparse engram table IDs
        sparse_ids = {
            id(model.engram_1.table.weight),
            id(model.engram_2.table.weight),
        }

        # Excluded from Muon (must use AdamW)
        excluded_from_muon = {
            id(model.tok_emb.weight),
            id(model.loop_inj_gates),
        }
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.out_features == model.cfg.moe_n_routed_experts:
                excluded_from_muon.add(id(m.weight))

        for p in model.parameters():
            if not p.requires_grad:
                continue
            p_id = id(p)
            if p_id in sparse_ids:
                sparse_params.append(p)
            elif p.ndim == 2 and p_id not in excluded_from_muon:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        self.dense_params = muon_params + adamw_params

        self.sparse_adam = SparseAdam(sparse_params, lr=sparse_lr) if sparse_params else None
        self.muon = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay) if muon_params else None
        self.adamw = AdamW(adamw_params, lr=adamw_lr, betas=(0.9, 0.95), weight_decay=weight_decay) if adamw_params else None

    def zero_grad(self):
        if self.sparse_adam:
            self.sparse_adam.zero_grad()
        if self.muon:
            self.muon.zero_grad()
        if self.adamw:
            self.adamw.zero_grad()

    def step(self, max_norm: float | None = 1.0):
        # Clip only dense parameters (CUDA sparse tensors do not support clip_grad_norm_)
        if max_norm is not None and self.dense_params:
            torch.nn.utils.clip_grad_norm_(self.dense_params, max_norm)

        if self.sparse_adam:
            self.sparse_adam.step()
        if self.muon:
            self.muon.step()
        if self.adamw:
            self.adamw.step()

    def set_lr_scale(self, scale: float):
        if self.muon:
            for g in self.muon.param_groups:
                g["lr"] = self.base_lrs["muon"] * scale
        if self.adamw:
            for g in self.adamw.param_groups:
                g["lr"] = self.base_lrs["adamw"] * scale
        if self.sparse_adam:
            for g in self.sparse_adam.param_groups:
                g["lr"] = self.base_lrs["sparse"] * scale
