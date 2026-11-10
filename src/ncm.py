import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import LT1Config


class EMACodebook(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average (EMA) codebook updates,
    commitment loss, and dead-code restarting.
    """
    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        decay: float = 0.99,
        dead_threshold: float = 0.5,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.dead_threshold = dead_threshold
        self.commitment_weight = commitment_weight

        # Codebook weights and tracking statistics
        embed = torch.randn(num_codes, code_dim)
        embed = F.normalize(embed, dim=-1)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.ones(num_codes))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        orig_shape = z.shape
        z_flat = z.reshape(-1, self.code_dim).float()

        # Compute squared Euclidean distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z @ e.T
        z_sq = z_flat.pow(2).sum(dim=-1, keepdim=True)
        e_sq = self.embedding.pow(2).sum(dim=-1, keepdim=True).T
        dist = z_sq + e_sq - 2.0 * (z_flat @ self.embedding.T)

        indices = torch.argmin(dist, dim=-1)
        z_q_flat = self.embedding[indices]
        z_q = z_flat + (z_q_flat - z_flat).detach() # Straight-Through Estimator

        # Commitment loss
        commit_loss = self.commitment_weight * F.mse_loss(z_flat, z_q_flat.detach())

        # EMA codebook update & dead-code restart during training
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.num_codes).float()
                curr_counts = one_hot.sum(dim=0)
                curr_embed_sum = one_hot.T @ z_flat

                self.cluster_size.mul_(self.decay).add_(curr_counts, alpha=1.0 - self.decay)
                self.embed_avg.mul_(self.decay).add_(curr_embed_sum, alpha=1.0 - self.decay)

                # Laplace smoothing
                n = self.cluster_size.sum()
                smoothed = (self.cluster_size + 1e-5) / (n + self.num_codes * 1e-5) * n
                updated_embed = self.embed_avg / smoothed.unsqueeze(1)
                self.embedding.copy_(updated_embed)

                # Dead code restart from active batch latents
                dead_mask = self.cluster_size < self.dead_threshold
                if dead_mask.any() and z_flat.shape[0] > 0:
                    num_dead = dead_mask.sum().item()
                    rand_idx = torch.randint(0, z_flat.shape[0], (num_dead,), device=z.device)
                    replacement = z_flat[rand_idx]
                    self.embedding[dead_mask] = replacement
                    self.embed_avg[dead_mask] = replacement
                    self.cluster_size[dead_mask] = 1.0

        # Compute codebook perplexity metrics
        with torch.no_grad():
            counts = torch.bincount(indices, minlength=self.num_codes).float()
            p = counts / (counts.sum() + 1e-6)
            entropy = -torch.sum(p * torch.log(p + 1e-8))
            perplexity = torch.exp(entropy)
            active_ratio = perplexity / self.num_codes

        metrics = {
            "perplexity": perplexity.item(),
            "active_ratio": active_ratio.item(),
        }

        z_q = z_q.to(z.dtype).reshape(orig_shape)
        indices = indices.reshape(orig_shape[:-1])
        return z_q, indices, commit_loss, metrics


def causal_left_window_pool(
    x: torch.Tensor,
    window_size: int,
    prev_states: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes strictly causal left-window average pooling over sequence length.
    At position t, averages over [max(0, t - window_size + 1) .. t].
    Ensures zero future-token leakage (j <= i always holds).
    """
    if window_size <= 1:
        return x

    b, t, d = x.shape
    if prev_states is not None and prev_states.numel() > 0:
        p = prev_states[:, -(window_size - 1):, :]
        full_x = torch.cat([p, x], dim=1)
    else:
        full_x = x

    # Ensure full_x has length at least t + window_size - 1 by padding on the left
    pad_needed = (t + window_size - 1) - full_x.shape[1]
    if pad_needed > 0:
        left_pad = full_x[:, 0:1, :].repeat(1, pad_needed, 1)
        full_x = torch.cat([left_pad, full_x], dim=1)

    # full_x now has length exactly t + window_size - 1
    windows = full_x.unfold(dimension=1, size=window_size, step=1) # [b, t, d, window_size]
    return windows.mean(dim=-1)


class CausalMultiscaleRVQ(nn.Module):
    """
    Causal Multiscale Residual Vector Quantizer for the CED-NCM Bridge.
    Quantizes coarse-to-fine left-window concepts with zero future leakage.
    """
    def __init__(self, cfg: LT1Config):
        super().__init__()
        self.d_model = cfg.d_model
        self.scales = cfg.ncm_scales # e.g. (4, 2, 1)
        self.codebooks = nn.ModuleList([
            EMACodebook(cfg.ncm_codebook_size, cfg.d_model) for _ in self.scales
        ])
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_model)
        self.next_concept_head = nn.Linear(cfg.d_model, cfg.ncm_codebook_size)

    def forward(
        self,
        h_enc: torch.Tensor,
        prev_enc_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        h_enc: [B, T, D]
        Returns: (h_out, commit_loss, next_concept_loss, metrics_dict)
        """
        residual = h_enc
        quantized_accum = torch.zeros_like(h_enc)
        total_commit_loss = torch.tensor(0.0, device=h_enc.device, dtype=h_enc.dtype)
        scale_metrics = {}
        first_scale_indices = None

        for s_idx, (scale_w, codebook) in enumerate(zip(self.scales, self.codebooks)):
            # Causal left-window pool for this scale
            pooled = causal_left_window_pool(residual, scale_w, prev_enc_states)
            q_s, idx_s, c_loss, m = codebook(pooled)

            if s_idx == 0:
                first_scale_indices = idx_s

            residual = residual - q_s
            quantized_accum = quantized_accum + q_s
            total_commit_loss = total_commit_loss + c_loss
            scale_metrics[f"scale_{scale_w}_active_ratio"] = m["active_ratio"]

        # Context-gated injection into encoder hidden states
        gate = torch.sigmoid(self.w_gate(h_enc))
        h_out = h_enc + gate * quantized_accum

        # Auxiliary next-concept prediction loss (predicting coarse concept code at span t + 4)
        next_concept_loss = torch.tensor(0.0, device=h_enc.device, dtype=h_enc.dtype)
        if first_scale_indices is not None and h_enc.shape[1] > 4:
            pred_logits = self.next_concept_head(h_enc[:, :-4]) # [B, T - 4, num_codes]
            targets = first_scale_indices[:, 4:]                 # [B, T - 4]
            next_concept_loss = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.shape[-1]),
                targets.reshape(-1),
            )

        return h_out, total_commit_loss, next_concept_loss, scale_metrics


class ParallelMultiscaleNCM(nn.Module):
    """
    Standalone Two-Stage NCM Generator demonstrating parallel level-by-level generation.
    Generates a 16-token sequence in exactly 3 coarse-to-fine parallel scale passes.
    """
    def __init__(self, cfg: LT1Config, block_len: int = 16, scales: tuple[int, ...] = (4, 8, 16)):
        super().__init__()
        self.block_len = block_len
        self.scales = scales
        self.d_model = cfg.d_model
        self.codebooks = nn.ModuleList([
            EMACodebook(cfg.ncm_codebook_size, cfg.d_model) for _ in scales
        ])
        # Level predictor predicts the next discrete scale representation in parallel
        self.level_predictors = nn.ModuleList([
            nn.Linear(cfg.d_model, s_len * cfg.d_model) for s_len in scales
        ])
        self.decoder = nn.Linear(cfg.d_model, cfg.vocab_size)

    def parallel_generate(self, context_vec: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Generates tokens for the block in len(scales) parallel passes.
        Returns: (logits [B, block_len, vocab_size], num_forward_passes)
        """
        b = context_vec.shape[0]
        accum = torch.zeros(b, self.block_len, self.d_model, device=context_vec.device, dtype=context_vec.dtype)
        passes = 0

        current_cond = context_vec
        for s_idx, (s_len, codebook, predictor) in enumerate(zip(self.scales, self.codebooks, self.level_predictors)):
            # Predict all codes of scale s in one parallel pass
            pred_flat = predictor(current_cond).reshape(b, s_len, self.d_model)
            q_s, _, _, _ = codebook(pred_flat)
            passes += 1

            # Upsample scale representations to full block_len
            factor = self.block_len // s_len
            q_upsampled = q_s.repeat_interleave(factor, dim=1)
            accum = accum + q_upsampled
            current_cond = accum.mean(dim=1)

        logits = self.decoder(accum)
        return logits, passes
