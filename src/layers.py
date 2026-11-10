import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (normed * self.weight.float()).to(orig_dtype)


def precompute_rope_cos_sin(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes cos and sin tensors for Rotary Positional Embeddings.
    Returns cos and sin of shape [max_seq_len, head_dim].
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(t, freqs)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Positional Embedding to query or key tensor.
    x shape: [B, H, T, D] or [B, T, H, D] or [B, T, D].
    cos, sin shape: [T, D] or broadcastable.
    """
    # Rotate interleaved dimensions: (-x1, x0, -x3, x2, ...)
    x_rot = torch.empty_like(x)
    x_rot[..., 0::2] = -x[..., 1::2]
    x_rot[..., 1::2] = x[..., 0::2]

    # Ensure cos and sin match x dimensions for broadcasting
    while cos.ndim < x.ndim:
        # If x is [B, H, T, D] and cos is [T, D], insert dims: [1, 1, T, D]
        if x.ndim == 4 and cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

    return (x * cos) + (x_rot * sin)


class BitLinear(nn.Module):
    """
    1.58-bit ternary weight linear layer with Straight-Through Estimator (STE).
    Weights are quantized to {-1, 0, +1} on forward pass; activations to 8-bit.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Weight quantization to ternary {-1, 0, +1}
        gamma = self.weight.abs().mean().clamp(min=1e-5)
        w_scaled = self.weight / gamma
        w_quant = torch.clamp(torch.round(w_scaled), -1.0, 1.0) * gamma
        w_effective = self.weight + (w_quant - self.weight).detach()

        # Activation quantization to 8-bit
        eta = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        q_b = 127.0 / eta
        x_quant = torch.clamp(torch.round(x * q_b), -128.0, 127.0) / q_b
        x_effective = x + (x_quant - x).detach()

        return F.linear(x_effective, w_effective.to(orig_dtype), self.bias)


def make_linear(
    in_features: int,
    out_features: int,
    bias: bool = False,
    use_bitlinear: bool = False,
) -> nn.Module:
    if use_bitlinear:
        return BitLinear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)
