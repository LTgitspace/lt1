from dataclasses import dataclass, field


@dataclass
class LT1Config:
    vocab_size: int = 16384
    max_seq_len: int = 512
    d_model: int = 384
    n_encoder_layers: int = 4
    n_recurrent_layers: int = 4
    n_coda_layers: int = 2
    max_loops: int = 4
    default_loops: int = 2
    n_heads: int = 6
    n_kv_heads: int = 2
    head_dim: int = 64
    moe_n_routed_experts: int = 8
    moe_n_shared_experts: int = 1
    moe_top_k_encoder: int = 1
    moe_top_k_decoder: int = 2
    moe_d_ff: int = 384
    moe_bias_lr: float = 0.001
    engram_buckets: int = 65536
    engram_dim: int = 128
    ncm_codebook_size: int = 512
    ncm_scales: tuple[int, ...] = (4, 2, 1)
    use_bitlinear: bool = False
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.head_dim == self.d_model // self.n_heads, "head_dim must equal d_model // n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads for GQA"


def base_config() -> LT1Config:
    """Returns standard LT-1 configuration optimized for RTX 3060 6GB Laptop GPU."""
    return LT1Config()


def tiny_smoke_config() -> LT1Config:
    """Returns lightweight configuration for sub-second CPU/GPU testing."""
    return LT1Config(
        vocab_size=512,
        max_seq_len=64,
        d_model=128,
        n_encoder_layers=2,
        n_recurrent_layers=2,
        n_coda_layers=1,
        max_loops=4,
        default_loops=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        moe_n_routed_experts=4,
        moe_n_shared_experts=1,
        moe_top_k_encoder=1,
        moe_top_k_decoder=2,
        moe_d_ff=128,
        engram_buckets=4096,
        engram_dim=64,
        ncm_codebook_size=64,
        ncm_scales=(4, 2, 1),
    )
