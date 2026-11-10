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


class EngramModule(nn.Module):
    """
    N-gram conditional memory module with sparse embedding table and context gating.
    Uses fp32 for the sparse embedding lookup to ensure SparseAdam receives fp32 gradients.
    """
    def __init__(self, cfg: LT1Config):
        super().__init__()
        self.num_buckets = cfg.engram_buckets
        self.dim = cfg.engram_dim
        self.d_model = cfg.d_model

        self.table = nn.Embedding(self.num_buckets, self.dim, sparse=True)
        self.w_proj = make_linear(self.dim, self.d_model, bias=False, use_bitlinear=cfg.use_bitlinear)
        self.w_gate = make_linear(self.d_model, self.d_model, bias=True, use_bitlinear=cfg.use_bitlinear)
        self.rms_norm = RMSNorm(self.d_model, eps=cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, ngram_hashes: torch.Tensor) -> torch.Tensor:
        # Lookup in fp32 for sparse gradient stability
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            # ngram_hashes: [B, T, 2]
            mem = self.table(ngram_hashes).sum(dim=-2) # [B, T, engram_dim]

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
        topk_weights = topk_weights.to(x.dtype)

        # 3. Execute routed experts
        out_routed = torch.zeros_like(x_flat)
        for e_idx, expert in enumerate(self.routed_experts):
            # Find tokens routed to this expert
            mask = (topk_idx == e_idx)
            if mask.any():
                token_indices, k_slot = torch.where(mask)
                x_sub = x_flat[token_indices]
                w_sub = topk_weights[token_indices, k_slot].unsqueeze(-1)
                expert_out = expert(x_sub) * w_sub
                out_routed.index_add_(0, token_indices, expert_out)

        # 4. Aux-loss-free dynamic bias update during training
        if self.training:
            with torch.no_grad():
                counts = torch.bincount(topk_idx.reshape(-1), minlength=self.n_routed).float()
                avg_count = counts.sum() / self.n_routed
                self.expert_bias.add_(self.bias_lr * torch.sign(avg_count - counts))
                self.expert_bias.clamp_(-2.0, 2.0)

        total_out = out_shared + out_routed
        return total_out.reshape(orig_shape)
