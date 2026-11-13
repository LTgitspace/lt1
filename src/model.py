import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import LT1Config
from src.layers import RMSNorm, precompute_rope_cos_sin, apply_rope, make_linear
from src.engram_moe import compute_ngram_hashes, EngramModule, DeepSeekMoE
from src.ncm import CausalMultiscaleRVQ


class CEDKVCache:
    """
    Key-Value cache for LT-1 with Last-Loop KV caching.
    Stores encoder layer KVs, single CED bridge KV, and single Recurrent anchor KV.
    """
    def __init__(self, n_encoder_layers: int):
        self.n_encoder_layers = n_encoder_layers
        self.encoder_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_encoder_layers
        self.bridge_kv: tuple[torch.Tensor, torch.Tensor] | None = None
        self.anchor_kv: tuple[torch.Tensor, torch.Tensor] | None = None
        self.prev_tokens: torch.Tensor | None = None
        self.prev_enc_states: list[torch.Tensor] | None = None
        self.seq_len: int = 0

    def reset(self):
        self.encoder_kv = [None] * self.n_encoder_layers
        self.bridge_kv = None
        self.anchor_kv = None
        self.prev_tokens = None
        self.prev_enc_states = None
        self.seq_len = 0


class CEDAttention(nn.Module):
    """
    Grouped-Query Attention supporting Full mode (projects Q, K, V) and
    Reuse mode (projects Q only and reuses pre-projected shared K and V).
    """
    def __init__(self, cfg: LT1Config, mode: str = "full"):
        super().__init__()
        self.mode = mode
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.repeat = self.n_heads // self.n_kv_heads

        self.w_q = make_linear(self.d_model, self.n_heads * self.head_dim, bias=False, use_bitlinear=cfg.use_bitlinear)
        if self.mode == "full":
            self.w_k = make_linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False, use_bitlinear=cfg.use_bitlinear)
            self.w_v = make_linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False, use_bitlinear=cfg.use_bitlinear)
        else:
            self.w_k = None
            self.w_v = None

        self.w_o = make_linear(self.n_heads * self.head_dim, self.d_model, bias=False, use_bitlinear=cfg.use_bitlinear)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        shared_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        b, q_len, _ = x.shape

        # 1. Project Query and apply RoPE
        q = self.w_q(x).view(b, q_len, self.n_heads, self.head_dim).transpose(1, 2) # [B, n_heads, q_len, head_dim]
        q = apply_rope(q, cos, sin)

        # 2. Key and Value resolution
        if self.mode == "full":
            k = self.w_k(x).view(b, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2) # [B, n_kv_heads, q_len, head_dim]
            v = self.w_v(x).view(b, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            k = apply_rope(k, cos, sin)

            if cached_kv is not None and cached_kv[0] is not None:
                past_k, past_v = cached_kv
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            new_kv = (k, v)
        else:
            assert shared_kv is not None, "shared_kv must be provided in reuse mode"
            k, v = shared_kv
            new_kv = None

        # Repeat KV heads for Grouped-Query Attention if needed
        kv_len = k.shape[2]
        if self.repeat > 1:
            k_expanded = k.repeat_interleave(self.repeat, dim=1)
            v_expanded = v.repeat_interleave(self.repeat, dim=1)
        else:
            k_expanded = k
            v_expanded = v

        # Determine causal masking
        if q_len == 1 and kv_len > 1:
            is_causal = False
            attn_mask = None
        elif q_len == kv_len:
            is_causal = True
            attn_mask = None
        else:
            is_causal = False
            # Bottom-right diagonal causal mask for chunked decoding
            row_idx = torch.arange(q_len, device=x.device).unsqueeze(1) + (kv_len - q_len)
            col_idx = torch.arange(kv_len, device=x.device).unsqueeze(0)
            attn_mask = (col_idx <= row_idx).unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(b, q_len, self.n_heads * self.head_dim)
        return self.w_o(out), new_kv


class CEDBlock(nn.Module):
    def __init__(self, cfg: LT1Config, attn_mode: str, top_k: int):
        super().__init__()
        self.rms_norm_1 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = CEDAttention(cfg, mode=attn_mode)
        self.rms_norm_2 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.moe = DeepSeekMoE(cfg, top_k=top_k)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        shared_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, new_kv = self.attn(
            self.rms_norm_1(x),
            cos, sin,
            shared_kv=shared_kv,
            cached_kv=cached_kv,
        )
        x = x + attn_out
        x = x + self.moe(self.rms_norm_2(x))
        return x, new_kv


class LT1Model(nn.Module):
    """
    LT-1 Small Language Model:
    - Stage 1: Causal Encoder Prelude (4 Layers, Top-1 MoE) + Engram Modules
    - CED-NCM Bridge: Causal Multiscale Residual VQ Concept Bottleneck
    - CED Global KV Bridge: Shared (K_global, V_global)
    - Stage 2: Dory Shared Recurrent Core (4 Layers x k Loops, Top-2 MoE) + Last-Loop KV Cache
    - Stage 3: Output Coda (2 Layers, Top-2 MoE) + Tied LM Head
    """
    def __init__(self, cfg: LT1Config):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model

        # Token embedding and RoPE precomputations
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        cos, sin = precompute_rope_cos_sin(cfg.head_dim, cfg.max_seq_len, theta=cfg.rope_theta)
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

        # Engram modules (Post-Embedding and Post-Encoder)
        self.engram_1 = EngramModule(cfg)
        self.engram_2 = EngramModule(cfg)

        # Stage 1: Causal Encoder Prelude (4 physical layers, Top-1 MoE)
        self.encoder_layers = nn.ModuleList([
            CEDBlock(cfg, attn_mode="full", top_k=cfg.moe_top_k_encoder)
            for _ in range(cfg.n_encoder_layers)
        ])
        self.enc_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)

        # CED-NCM Bridge (Causal Multiscale RVQ)
        self.ncm_bridge = CausalMultiscaleRVQ(cfg)

        # CED Global KV Bridge projections
        self.bridge_wk = make_linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False, use_bitlinear=cfg.use_bitlinear)
        self.bridge_wv = make_linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False, use_bitlinear=cfg.use_bitlinear)

        # Stage 2: Dory Shared Recurrent Core (4 physical layers)
        # Layer 0: Reuse (K_global, V_global)
        # Layer 1: Reuse (K_global, V_global)
        # Layer 2: Full (refreshes Anchor KV with Last-Loop KV cache commit)
        # Layer 3: Reuse Anchor KV
        self.recurrent_layers = nn.ModuleList([
            CEDBlock(cfg, attn_mode="reuse", top_k=cfg.moe_top_k_decoder),
            CEDBlock(cfg, attn_mode="reuse", top_k=cfg.moe_top_k_decoder),
            CEDBlock(cfg, attn_mode="full",  top_k=cfg.moe_top_k_decoder),
            CEDBlock(cfg, attn_mode="reuse", top_k=cfg.moe_top_k_decoder),
        ])
        self.loop_inj_gates = nn.Parameter(torch.zeros(cfg.max_loops))
        self.loop_inj_proj = make_linear(cfg.d_model, cfg.d_model, bias=False, use_bitlinear=cfg.use_bitlinear)
        self.loop_bound_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)

        # Stage 3: Output Coda (2 physical layers, reuse Anchor KV)
        self.coda_layers = nn.ModuleList([
            CEDBlock(cfg, attn_mode="reuse", top_k=cfg.moe_top_k_decoder)
            for _ in range(cfg.n_coda_layers)
        ])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.is_packed_inference = False
        self.last_loss_dict: dict[str, float] = {}

        if cfg.use_bf16_weights:
            # Cast large 2D linear weights to bfloat16 to cut VRAM by 50%,
            # while keeping AdamW-updated 1D norms, tok_emb, routers, and classifier heads in float32
            # so small AdamW updates do not underflow bfloat16's 7-bit mantissa.
            self.eng_proj_norm_to_bf16()
            self.encoder_layers.bfloat16()
            self.ncm_bridge.bfloat16()
            self.bridge_wk.bfloat16()
            self.bridge_wv.bfloat16()
            self.recurrent_layers.bfloat16()
            self.loop_inj_proj.bfloat16()
            self.coda_layers.bfloat16()
            self._keep_adamw_params_fp32()

    def _keep_adamw_params_fp32(self):
        self.tok_emb.float()
        self.enc_norm.float()
        self.loop_bound_norm.float()
        self.final_norm.float()
        self.ncm_bridge.next_concept_head.float()
        for m in self.modules():
            if isinstance(m, RMSNorm):
                m.float()
            elif hasattr(m, "router") and isinstance(m.router, nn.Linear):
                m.router.float()

    def eng_proj_norm_to_bf16(self):
        for eng in (self.engram_1, self.engram_2):
            eng.w_proj.bfloat16()
            eng.w_gate.bfloat16()

    def pack_for_inference(self, device: torch.device | str = "cuda"):
        self.eval()
        self.engram_1.pack_for_inference(device)
        self.engram_2.pack_for_inference(device)
        self.ncm_bridge.pack_for_inference(device)
        for layer in list(self.encoder_layers) + list(self.recurrent_layers) + list(self.coda_layers):
            layer.moe.pack_for_inference(device)
            layer.to(device=device, dtype=torch.bfloat16)
        self.to(device=device, dtype=torch.bfloat16)
        self.is_packed_inference = True
        return self

    def count_parameters(self) -> dict[str, int]:
        total_params = sum(p.numel() for p in self.parameters())
        sparse_params = 0
        for eng in (self.engram_1, self.engram_2):
            if hasattr(eng, "table_weight") and eng.table_weight is not None:
                sparse_params += eng.table_weight.numel()
            elif hasattr(eng, "compact_table") and eng.compact_table is not None:
                sparse_params += eng.compact_table.numel()
            elif hasattr(eng, "table") and eng.table is not None:
                sparse_params += eng.table.weight.numel()

        if self.cfg.engram_offload_cpu and sparse_params > 0:
            total_params += sparse_params

        dense_params = total_params - sparse_params
        return {
            "total": total_params,
            "sparse_engram": sparse_params,
            "dense_backbone": dense_params,
        }

    def get_num_params(self) -> dict[str, int]:
        counts = self.count_parameters()
        return {
            "total": counts["total"],
            "sparse_engram": counts["sparse_engram"],
            "dense_backbone": counts["dense_backbone"],
            "offloaded_cpu": counts["sparse_engram"] if self.cfg.engram_offload_cpu else 0,
            "on_device": counts["dense_backbone"] if self.cfg.engram_offload_cpu else counts["total"],
        }

    def get_offloaded_tables(self) -> dict[str, torch.Tensor]:
        tables = {}
        for idx, eng in enumerate((self.engram_1, self.engram_2), start=1):
            if hasattr(eng, "table_weight") and eng.table_weight is not None:
                tables[f"engram_{idx}"] = eng.table_weight
        return tables

    def step_moe_biases(self):
        for m in self.modules():
            if hasattr(m, "step_bias"):
                m.step_bias()

    def _run_block(self, layer, x, cos, sin, shared_kv=None, cached_kv=None):
        if self.training and self.cfg.use_grad_checkpointing and cached_kv is None:
            import torch.utils.checkpoint as cp
            return cp.checkpoint(layer, x, cos, sin, shared_kv, cached_kv, use_reentrant=False)
        return layer(x, cos, sin, shared_kv=shared_kv, cached_kv=cached_kv)


    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        num_loops: int | None = None,
        kv_cache: CEDKVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, seq_len = token_ids.shape
        loops = num_loops or self.cfg.default_loops
        loops = min(loops, self.cfg.max_loops)

        # Resolve positional range and RoPE cos/sin slices
        start_pos = kv_cache.seq_len if kv_cache is not None else 0
        end_pos = start_pos + seq_len
        cos = self.cos_cached[start_pos:end_pos]
        sin = self.sin_cached[start_pos:end_pos]

        # Compute N-gram hashes
        prev_toks = kv_cache.prev_tokens if kv_cache is not None else None
        ngram_hashes = compute_ngram_hashes(token_ids, self.cfg.engram_buckets, prev_tokens=prev_toks)

        # 1. Embedding + Engram 1
        x = self.tok_emb(token_ids)
        if self.cfg.use_bf16_weights:
            x = x.to(torch.bfloat16)
        x = self.engram_1(x, ngram_hashes)

        # 2. Stage 1: Causal Encoder Prelude
        for l_idx, enc_layer in enumerate(self.encoder_layers):
            cached = kv_cache.encoder_kv[l_idx] if kv_cache is not None else None
            x, new_kv = self._run_block(enc_layer, x, cos, sin, cached_kv=cached)
            if kv_cache is not None:
                kv_cache.encoder_kv[l_idx] = new_kv

        # Engram 2 + Post-Encoder Normalization -> H_enc
        x = self.engram_2(x, ngram_hashes)
        h_enc = self.enc_norm(x)

        # 3. CED-NCM Bridge (Causal Multiscale RVQ)
        prev_enc = kv_cache.prev_enc_states if kv_cache is not None else None
        new_prev_enc = [] if kv_cache is not None else None
        h_enc, commit_loss, next_concept_loss, _ = self.ncm_bridge(
            h_enc, prev_enc_states=prev_enc, new_prev_states_out=new_prev_enc
        )

        # 4. CED Global KV Bridge projection
        if kv_cache is not None and kv_cache.bridge_kv is not None:
            # Incremental projection
            k_new = self.bridge_wk(h_enc).view(b, seq_len, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
            v_new = self.bridge_wv(h_enc).view(b, seq_len, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
            k_new = apply_rope(k_new, cos, sin)
            k_global = torch.cat([kv_cache.bridge_kv[0], k_new], dim=2)
            v_global = torch.cat([kv_cache.bridge_kv[1], v_new], dim=2)
            kv_cache.bridge_kv = (k_global, v_global)
        else:
            k_global = self.bridge_wk(h_enc).view(b, seq_len, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
            v_global = self.bridge_wv(h_enc).view(b, seq_len, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
            k_global = apply_rope(k_global, cos, sin)
            if kv_cache is not None:
                kv_cache.bridge_kv = (k_global, v_global)

        global_kv = (k_global, v_global)

        # 5. Stage 2: Dory Shared Recurrent Core
        # Initialize anchor KV with global KV for early loops
        anchor_kv = global_kv
        curr_x = h_enc

        for l in range(loops):
            is_last = (l == loops - 1)
            alpha_l = torch.sigmoid(self.loop_inj_gates[l])
            # Bounded loop normalization with anchor re-injection
            curr_x = self.loop_bound_norm(curr_x + alpha_l * self.loop_inj_proj(h_enc))

            # Recurrent Layer 0: Reuse global_kv
            curr_x, _ = self._run_block(self.recurrent_layers[0], curr_x, cos, sin, shared_kv=global_kv)

            # Recurrent Layer 1: Reuse global_kv
            curr_x, _ = self._run_block(self.recurrent_layers[1], curr_x, cos, sin, shared_kv=global_kv)

            # Recurrent Layer 2: Refresh Anchor KV (Last-Loop KV caching)
            cached_anchor = kv_cache.anchor_kv if kv_cache is not None else None
            curr_x, new_anchor_kv = self._run_block(
                self.recurrent_layers[2], curr_x, cos, sin,
                cached_kv=cached_anchor,
            )
            anchor_kv = new_anchor_kv if new_anchor_kv is not None else anchor_kv
            if kv_cache is not None and is_last:
                kv_cache.anchor_kv = new_anchor_kv

            # Recurrent Layer 3: Reuse anchor_kv
            curr_x, _ = self._run_block(self.recurrent_layers[3], curr_x, cos, sin, shared_kv=anchor_kv)

        # 6. Stage 3: Output Coda Block
        coda_x = curr_x
        for coda_layer in self.coda_layers:
            coda_x, _ = self._run_block(coda_layer, coda_x, cos, sin, shared_kv=anchor_kv)


        # 7. Final Norm & Tied LM Head
        coda_normed = self.final_norm(coda_x)
        logits = F.linear(coda_normed, self.tok_emb.weight.to(coda_normed.dtype))

        # Update cache state if provided
        if kv_cache is not None:
            kv_cache.seq_len = end_pos
            if kv_cache.prev_tokens is None:
                kv_cache.prev_tokens = token_ids[:, -2:].clone()
            else:
                kv_cache.prev_tokens = torch.cat([kv_cache.prev_tokens, token_ids], dim=1)[:, -2:].clone()
            kv_cache.prev_enc_states = new_prev_enc

        # 8. Compute total loss if targets given
        total_loss = None
        if targets is not None:
            vocab_s = self.cfg.vocab_size
            ce_loss = F.cross_entropy(logits.view(-1, vocab_s), targets.view(-1), ignore_index=-100)
            total_loss = ce_loss + 0.1 * commit_loss + 0.1 * next_concept_loss
            self.last_loss_dict = {
                "ce_loss": ce_loss.item(),
                "commit_loss": commit_loss.item(),
                "next_concept_loss": next_concept_loss.item(),
            }

        return logits, total_loss


# Alias for backward compatibility with design specification
NanoCEDModel = LT1Model
