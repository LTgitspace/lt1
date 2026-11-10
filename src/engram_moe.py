import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import LT1Config
from src.layers import RMSNorm, make_linear


def compute_ngram_hashes(
    token_ids: torch.Tensor,
    num_buckets: int,
    prev_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes 2-gram and 3-gram XOR-prime hashes into [0, num_buckets - 1].
    token_ids: [B, T]
    prev_tokens: [B, P] (optional preceding tokens from cache, P <= 2)
    Returns: [B, T, 2] int64 tensor containing (h_2gram, h_3gram) for each token position.
    """
    mask = num_buckets - 1

    if prev_tokens is not None and prev_tokens.numel() > 0:
        full_tokens = torch.cat([prev_tokens, token_ids], dim=1)
    else:
        full_tokens = token_ids

    t_len = token_ids.shape[1]
    # Ensure full_tokens has at least 2 tokens of prefix padding for 2-gram and 3-gram lookups
    if full_tokens.shape[1] < t_len + 2:
        pad_needed = (t_len + 2) - full_tokens.shape[1]
        full_tokens = F.pad(full_tokens, (pad_needed, 0), value=0)

    t_0 = full_tokens[:, -t_len:]
    t_prev1 = full_tokens[:, -t_len - 1 : -1]
    t_prev2 = full_tokens[:, -t_len - 2 : -2]

    # 64-bit integer XOR-prime hashing
    p1 = 0x9E3779B1
    p2 = 0x85EBCA77
    p3 = 0xC2B2AE3D
    p4 = 0x27D4EB2F
    p5 = 0x165667B1

    h_2gram = ((t_prev1 * p1) ^ (t_0 * p2)) & mask
    h_3gram = ((t_prev2 * p3) ^ (t_prev1 * p4) ^ (t_0 * p5)) & mask

    return torch.stack([h_2gram, h_3gram], dim=-1)


class _OffloadedEngramLookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ngram_hashes: torch.Tensor, table_weight: torch.Tensor, module: "EngramModule") -> torch.Tensor:
        # ngram_hashes: [B, T, 2] on GPU
        device = ngram_hashes.device
        hashes_cpu = ngram_hashes.detach().to(device="cpu", dtype=torch.long)
        # Gather rows on CPU in bfloat16
        gathered = table_weight[hashes_cpu] # [B, T, 2, dim]
        mem_cpu = gathered.sum(dim=-2)      # [B, T, dim]
        mem_gpu = mem_cpu.to(device=device, dtype=torch.bfloat16)
        ctx.hashes_cpu = hashes_cpu
        ctx.module = module
        return mem_gpu

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        module = ctx.module
        if module.training:
            grad_cpu = grad_output.detach().to(device="cpu", dtype=torch.float32)
            module.queue_sparse_grad(ctx.hashes_cpu, grad_cpu)
        return None, None, None


class EngramModule(nn.Module):
    """
    N-gram conditional memory module with sparse embedding table and context gating.
    Supports GPU fp32 sparse embedding lookup or CPU-offloaded Factored Row-RMS bfloat16 table.
    """
    def __init__(self, cfg: LT1Config):
        super().__init__()
        self.num_buckets = cfg.engram_buckets
        self.dim = cfg.engram_dim
        self.d_model = cfg.d_model
        self.offload_cpu = cfg.engram_offload_cpu
        self.is_packed_inference = False

        if self.offload_cpu:
            # CPU-resident bfloat16 table zero-initialized to avoid random noise on unseen n-grams
            self.register_buffer(
                "table_weight",
                torch.zeros(self.num_buckets, self.dim, dtype=torch.bfloat16, device="cpu"),
                persistent=True,
            )
            self.row_ms = torch.zeros(self.num_buckets, 1, dtype=torch.float32, device="cpu")
            self.pending_grads: list[tuple[torch.Tensor, torch.Tensor]] = []
            self.table = None
        else:
            self.table = nn.Embedding(self.num_buckets, self.dim, sparse=True)
            self.table_weight = None

        self.w_proj = make_linear(self.dim, self.d_model, bias=False, use_bitlinear=cfg.use_bitlinear)
        self.w_gate = make_linear(self.d_model, self.d_model, bias=True, use_bitlinear=cfg.use_bitlinear)
        self.rms_norm = RMSNorm(self.d_model, eps=cfg.rms_norm_eps)

    def _apply(self, fn, recurse=True):
        if self.offload_cpu and hasattr(self, "table_weight") and self.table_weight is not None:
            prev_table = self.table_weight
            res = super()._apply(fn, recurse=recurse)
            self._buffers["table_weight"] = prev_table
            return res
        return super()._apply(fn, recurse=recurse)

    def queue_sparse_grad(self, hashes_cpu: torch.Tensor, grad_cpu: torch.Tensor):
        self.pending_grads.append((hashes_cpu, grad_cpu))

    def step_sparse_rms(self, lr: float, beta2: float = 0.99, eps: float = 1e-6):
        if not self.pending_grads:
            return
        # Coalesce gradients on CPU
        accum: dict[int, torch.Tensor] = {}
        for hashes_cpu, grad_cpu in self.pending_grads:
            # hashes_cpu: [B, T, 2], grad_cpu: [B, T, dim]
            h_flat = hashes_cpu.reshape(-1) # [2*B*T]
            # Each token's gradient applies to both its 2-gram and 3-gram bucket
            g_rep = grad_cpu.repeat_interleave(2, dim=1).reshape(-1, self.dim) # [2*B*T, dim]
            unique_h, inv = torch.unique(h_flat, return_inverse=True)
            g_sum = torch.zeros(len(unique_h), self.dim, dtype=torch.float32)
            g_sum.index_add_(0, inv, g_rep)
            for idx_t, g_vec in zip(unique_h.tolist(), g_sum):
                if idx_t in accum:
                    accum[idx_t].add_(g_vec)
                else:
                    accum[idx_t] = g_vec.clone()

        self.pending_grads.clear()

        # Apply Factored Row-RMS update in-place on CPU
        active_indices = torch.tensor(list(accum.keys()), dtype=torch.long)
        if len(active_indices) == 0:
            return
        active_grads = torch.stack(list(accum.values())) # [K, dim]

        row_grad_sq = active_grads.pow(2).mean(dim=-1, keepdim=True) # [K, 1]
        self.row_ms[active_indices] = (
            beta2 * self.row_ms[active_indices] + (1.0 - beta2) * row_grad_sq
        )
        rms = torch.sqrt(self.row_ms[active_indices]) + eps
        update = active_grads / rms
        self.table_weight[active_indices] = (
            self.table_weight[active_indices].float() - lr * update
        ).to(torch.bfloat16)

    def pack_for_inference(self, device: torch.device | str = "cuda"):
        if self.is_packed_inference:
            return
        if self.offload_cpu and self.table_weight is not None:
            # Deduplicate untouched zero rows into row 0
            with torch.no_grad():
                active_mask = (self.table_weight.abs().sum(dim=-1) > 0) | (self.row_ms.squeeze(-1) > 0)
                if active_mask.any():
                    compact_rows = self.table_weight[active_mask]
                    compact_table = torch.cat([torch.zeros(1, self.dim, dtype=torch.bfloat16), compact_rows], dim=0)
                    lut = torch.zeros(self.num_buckets, dtype=torch.int32)
                    lut[active_mask] = torch.arange(1, active_mask.sum().item() + 1, dtype=torch.int32)
                else:
                    compact_table = torch.zeros(1, self.dim, dtype=torch.bfloat16)
                    lut = torch.zeros(self.num_buckets, dtype=torch.int32)

            self.compact_table = nn.Parameter(compact_table.to(device=device), requires_grad=False)
            self.lut = lut.to(device=device)
            self.table_weight = None
            self.row_ms = None
        elif self.table is not None:
            # Direct GPU embedding
            compact_table = self.table.weight.data.to(device=device, dtype=torch.bfloat16)
            lut = torch.arange(self.num_buckets, dtype=torch.int32, device=device)
            self.compact_table = nn.Parameter(compact_table, requires_grad=False)
            self.lut = lut
            self.table = None

        self.is_packed_inference = True

    def forward(self, x: torch.Tensor, ngram_hashes: torch.Tensor) -> torch.Tensor:
        if self.is_packed_inference:
            # 100% on GPU, zero PCIe transfer, O(1) indexed
            indices = self.lut[ngram_hashes]
            mem = self.compact_table[indices].sum(dim=-2)
        elif self.offload_cpu:
            mem = _OffloadedEngramLookup.apply(ngram_hashes, self.table_weight, self)
        else:
            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                mem = self.table(ngram_hashes).sum(dim=-2)

        mem = mem.to(x.dtype)
        norm_x = self.rms_norm(x)
        gate = torch.sigmoid(self.w_gate(norm_x))
        return x + gate * self.w_proj(mem)



class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, use_bitlinear: bool = False):
        super().__init__()
        self.w_gate = make_linear(d_model, d_ff, bias=False, use_bitlinear=use_bitlinear)
        self.w_up = make_linear(d_model, d_ff, bias=False, use_bitlinear=use_bitlinear)
        self.w_down = make_linear(d_ff, d_model, bias=False, use_bitlinear=use_bitlinear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DeepSeekMoE(nn.Module):
    """
    Fine-Grained Mixture-of-Experts with 1 Shared Expert + N Routed Experts
    and Auxiliary-Loss-Free Dynamic Bias Load Balancing.
    """
    def __init__(self, cfg: LT1Config, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.n_routed = cfg.moe_n_routed_experts
        self.bias_lr = cfg.moe_bias_lr

        self.shared_expert = SwiGLUExpert(cfg.d_model, cfg.moe_d_ff, cfg.use_bitlinear)
        self.routed_experts = nn.ModuleList(
            [SwiGLUExpert(cfg.d_model, cfg.moe_d_ff, cfg.use_bitlinear) for _ in range(self.n_routed)]
        )
        self.router = nn.Linear(cfg.d_model, self.n_routed, bias=False)
        self.register_buffer("expert_bias", torch.zeros(self.n_routed))
        self.register_buffer("token_counts", torch.zeros(self.n_routed), persistent=False)
        self.is_packed_inference = False

    def pack_for_inference(self, device: torch.device | str = "cuda"):
        if getattr(self, "is_packed_inference", False):
            return
        with torch.no_grad():
            w_gates = torch.stack([e.w_gate.weight.data for e in self.routed_experts]).to(device=device, dtype=torch.bfloat16)
            w_ups = torch.stack([e.w_up.weight.data for e in self.routed_experts]).to(device=device, dtype=torch.bfloat16)
            w_downs = torch.stack([e.w_down.weight.data for e in self.routed_experts]).to(device=device, dtype=torch.bfloat16)
        self.w_gate_packed = nn.Parameter(w_gates, requires_grad=False)
        self.w_up_packed = nn.Parameter(w_ups, requires_grad=False)
        self.w_down_packed = nn.Parameter(w_downs, requires_grad=False)
        self.is_packed_inference = True

    def step_bias(self):
        if not self.training:
            return
        with torch.no_grad():
            if self.token_counts.sum() > 0:
                avg_count = self.token_counts.sum() / self.n_routed
                self.expert_bias.add_(self.bias_lr * torch.sign(avg_count - self.token_counts))
                self.expert_bias.clamp_(-2.0, 2.0)
                self.token_counts.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        num_tokens = x_flat.shape[0]

        # 1. Shared expert executes on all tokens
        out_shared = self.shared_expert(x_flat)

        # 2. Router affinity scores
        logits = self.router(x_flat)
        scores = torch.sigmoid(logits.float()) # [N, n_routed]

        # Selection uses biased scores to balance load
        _, topk_idx = torch.topk(scores + self.expert_bias, k=self.top_k, dim=-1) # [N, top_k]

        # Gating weights use clean UNBIASED scores for pure gradient flow
        topk_weights = torch.gather(scores, dim=-1, index=topk_idx)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 3. Execute routed experts: Zero-sync fused BMM fast path during inference
        if getattr(self, "is_packed_inference", False) and num_tokens <= 64:
            flat_idx = topk_idx.reshape(-1) # [N * top_k]
            wg = self.w_gate_packed[flat_idx] # [K, d_ff, d_model]
            wu = self.w_up_packed[flat_idx]   # [K, d_ff, d_model]
            wd = self.w_down_packed[flat_idx] # [K, d_model, d_ff]

            x_rep = x_flat.repeat_interleave(self.top_k, dim=0).unsqueeze(-1).to(torch.bfloat16) # [K, d_model, 1]
            gate = F.silu(torch.bmm(wg, x_rep))
            up = torch.bmm(wu, x_rep)
            h = gate * up
            out_exp = torch.bmm(wd, h).squeeze(-1).view(num_tokens, self.top_k, -1)
            out_routed = (out_exp * topk_weights.to(out_exp.dtype).unsqueeze(-1)).sum(dim=1).to(x.dtype)
        else:
            out_routed = torch.zeros_like(x_flat)
            topk_weights_typed = topk_weights.to(x.dtype)
            for e_idx, expert in enumerate(self.routed_experts):
                mask = (topk_idx == e_idx)
                if mask.any():
                    token_indices, k_slot = torch.where(mask)
                    x_sub = x_flat[token_indices]
                    w_sub = topk_weights_typed[token_indices, k_slot].unsqueeze(-1)
                    expert_out = expert(x_sub) * w_sub
                    out_routed.index_add_(0, token_indices, expert_out)

        # 4. Aux-loss-free load statistics accumulation during training
        if self.training:
            with torch.no_grad():
                counts = torch.bincount(topk_idx.reshape(-1), minlength=self.n_routed).float()
                self.token_counts.add_(counts)

        total_out = out_shared + out_routed
        return total_out.reshape(orig_shape)

