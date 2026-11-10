# LT-1: Next-Generation Small Language Model Architecture

LT-1 is a clean-sheet, high-efficiency small language model architecture built from scratch in pure PyTorch. It integrates five state-of-the-art innovations designed for maximal inference-time elasticity, deep factual recall, and sub-3.2 GB training on an NVIDIA GeForce RTX 3060 6GB Laptop GPU.

## Key Architectural Innovations

1. **N-Gram Engram Conditional Memory:**
   Deterministic hash-projected fp32 sparse embedding table with gated residual injection across 2-gram, 3-gram, and 4-gram prefix tokens. Offloads static multi-word memorization from feed-forward parameters into an O(1) hash table trained with PyTorch SparseAdam.

2. **Causal Encoder-Decoder (CED) Global KV Bridge:**
   Asymmetric architecture featuring a deep Causal Encoder (rich bidirectional-like prefix representations with full GQA) and a lightweight Causal Decoder that reuses the Encoder's cached Key and Value projections via a persistent Global KV Bridge.

3. **NVIDIA Dory Elastic Recurrent Loops:**
   Dynamically unrolled recurrent reasoning loop (k in {1, 2, 4}) inside the Decoder. Early loops re-inject the static H_enc anchor to prevent semantic drift. Employs Last-Loop KV caching during autoregressive generation to ensure O(1) inference memory regardless of recurrence depth.

4. **Fine-Grained DeepSeekMoE:**
   1 Shared Expert + 8 Routed SwiGLU Experts per layer with top-2 routing and selection-only dynamic bias adjustment to prevent routing collapse without injecting noisy loss terms into the gradient landscape.

5. **Next-Concept Modeling (NCM) Multiscale RVQ:**
   Hierarchical residual vector quantizer operating causally over the CED Bridge across three temporal resolutions (scales 4, 2, 1). Trained with exponential moving average (EMA) codebook updates, dead-code restarts, and multi-scale auxiliary cross-entropy loss. Enables parallel multi-scale speculation during generation.

6. **Tri-Optimizer System:**
   - SparseAdam for fp32 Engram hash tables.
   - Muon with 5th-order Newton-Schulz orthogonalization for 2D weight matrices (d_out >= d_in >= 128).
   - AdamW with bfloat16 autocast for norms, embeddings, routers, and 1D parameters.

## Directory Layout

```
C:\github\lt1
├── .gitignore
├── README.md
├── src
│   ├── __init__.py
│   ├── config.py         # LT1Config, base_config (65.2M), tiny_smoke_config (1.4M)
│   ├── layers.py         # RMSNorm, RoPE, BitLinear 1.58-bit ternary quantized linear
│   ├── engram_moe.py     # EngramModule, SwiGLU, DeepSeekMoE with selection-only bias
│   ├── ncm.py            # EMACodebook, CausalMultiscaleRVQ, ParallelMultiscaleNCM
│   ├── model.py          # CEDKVCache, CEDAttention, CEDBlock, LT1Model
│   ├── optim.py          # Muon (Newton-Schulz5), TriOptimizer
│   ├── tokenizer.py      # NanoTokenizer (HF tokenizers byte-level BPE wrapper)
│   ├── data.py           # MemmapDataset (uint16 binary shards), SFT conversational encoder
│   ├── train.py          # Unified CLI for smoke, pretrain, SFT, and NCM benchmarking
│   └── generate.py       # Autoregressive generation with elastic loops
├── tests
│   ├── __init__.py
│   ├── test_layers.py
│   ├── test_engram_moe.py
│   ├── test_ncm.py
│   ├── test_model.py
│   ├── test_optim_and_vram.py
│   └── test_data_and_pipeline.py
└── data
    └── smoke_tokenizer.json
```

## Hardware Verification & Memory Profile

- Target GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6,144 MiB VRAM).
- Precision: bfloat16 mixed precision with fp32 master weights.
- Peak VRAM during base_config training (B=8, T=512, k=2): 2.81 GB (well below the 3.2 GB safety threshold to avoid Windows 11 WDDM paging).
- Smoke config peak VRAM: 74.7 MB.

## Quick Start & Verification

### Run Unit Test Suite
```powershell
python -m unittest discover -s C:\github\lt1\tests -t C:\github\lt1 -v
```

### Run Smoke Pre-Training
```powershell
python -m src.train --stage smoke --steps 50 --batch-size 4 --seq-len 64
```

### Benchmark Next-Concept Modeling Parallel Inference
```powershell
python -m src.train --stage ncm-bench
```

### Generate Text with Elastic Recurrence
```powershell
python -m src.generate --checkpoint checkpoints/smoke_step_50.pt --prompt "Once upon a time" --loops 2 --max-tokens 32
```
