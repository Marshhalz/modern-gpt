# modern-gpt

> A small GPT, written from scratch, progressively modernised from the original 2017 Transformer toward the LLaMA 3 architecture — one architectural change per commit, each measured against the baseline.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-pytest-green)](tests/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## What this is

A decoder-only Transformer that starts as **vanilla GPT-2** and progressively becomes **LLaMA 3** — but small enough to train on a single laptop GPU. Each modern component (RMSNorm, RoPE, SwiGLU, GQA, FlashAttention, KV cache, speculative decoding) is added in its own commit with a clean diff and a measured benchmark against the baseline.

The goal is not to ship state-of-the-art performance — it is to **understand and reproduce** the architectural decisions that distinguish modern open-weight LLMs.

---

## Quick start

```bash
git clone https://github.com/Marshhalz/modern-gpt.git
cd modern-gpt

# Install with uv (recommended)
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

```python
from modern_gpt import GPT, GPTConfig
import torch

cfg   = GPTConfig()                 # ~800K params at default settings
model = GPT(cfg)

# Forward pass
idx          = torch.zeros((1, 8), dtype=torch.long)
logits, _    = model(idx)            # (1, 8, vocab_size)

# Sampling
generated    = model.generate(idx, max_new_tokens=100)
```

Run the tests:

```bash
pytest
```

---

## Training

Reproducible end-to-end with:

```bash
python scripts/train.py
```

| Setting          | Value                                       |
|------------------|---------------------------------------------|
| Dataset          | Tiny Shakespeare (~1 MB, char-level)        |
| Train / val      | 90 / 10                                     |
| Batch size       | 64                                          |
| Iterations       | 5 000                                       |
| Optimiser        | AdamW (lr = 3 × 10⁻⁴, no schedule)           |
| Gradient clip    | None                                        |
| Mixed precision  | No (fp32)                                   |
| Seed             | 1337                                        |
| Hardware         | NVIDIA RTX 3060 Laptop GPU (6 GB VRAM)      |

Full per-step loss trajectory and analysis: [`benchmarks/baseline.md`](benchmarks/baseline.md).

---

## Roadmap

Each row below is one commit.  The **Val loss** column is the bar each
modernisation must beat (or — for efficiency-oriented changes — match
while delivering a speed-up).

| # | Change | Status | Train loss | Val loss | Notes |
|---|--------|--------|-----------:|---------:|-------|
| 0 | Baseline GPT-2 (LayerNorm, learned PE, ReLU FFN, MHA) | ✅ | 1.3548 | 1.5787 | Reference point — 817K params, ~10 min |
| 1 | `LayerNorm` → **RMSNorm** | ✅ | **1.3570** | **1.5858** | −1 152 params, ~9 % faster — see [`benchmarks/rmsnorm.md`](benchmarks/rmsnorm.md) |
| 2 | learned PE → **Rotary Position Embeddings (RoPE)** | ✅ | **1.3258** | **1.5474** | −8 192 params, ~30 % slower — first quality gain above baseline — see [`benchmarks/rope.md`](benchmarks/rope.md) |
| 3 | ReLU FFN → **SwiGLU** | ⬜ | – | – | Gated activation (Shazeer, 2020) |
| 4 | MHA → **Grouped-Query Attention (GQA)** | ⬜ | – | – | Memory-efficient inference (LLaMA 2/3) |
| 5 | **QK-Norm** (cosine attention) | ⬜ | – | – | Connects attention to RKHS kernel theory |
| 6 | Naive attention → **`F.scaled_dot_product_attention`** | ⬜ | – | – | FlashAttention via PyTorch fused kernel |
| 7 | **KV cache** for inference | ⬜ | – | – | $O(T)$ per step instead of $O(T^2)$ |
| 8 | **Speculative decoding** | ⬜ | – | – | Draft+verify, 2–3× faster sampling |

---

## Architecture overview (current)

```
tokens (B, T)
   │
   ├── token_embedding (vocab_size, n_embd)
   └── position_embedding (block_size, n_embd)          ← learned, replaced by RoPE in step 2
   │
   ▼  (B, T, n_embd)
   │
┌──┴─────────────────────────────────────────┐
│  Block × n_layer                            │
│    ├── LayerNorm                            │  ← replaced by RMSNorm in step 1
│    ├── Multi-Head Attention                 │  ← becomes GQA in step 4, FlashAttn in step 6
│    ├── + residual                           │
│    ├── LayerNorm                            │
│    ├── FeedForward (ReLU, 4× expansion)     │  ← becomes SwiGLU in step 3
│    └── + residual                           │
└──┬─────────────────────────────────────────┘
   │
   ▼
LayerNorm (final)                                       ← also becomes RMSNorm in step 1
   │
   ▼
lm_head: Linear(n_embd, vocab_size)
   │
   ▼
logits
```

---

## Why this project

Most "build GPT from scratch" projects stop at the 2017 Transformer. Every interesting innovation since 2019 — rotary embeddings, gated activations, grouped-query attention, FlashAttention, speculative decoding — is left out, either because they are seen as too specialised or because no single tutorial covers them all.

This repository is the bridge. It begins where Karpathy's tutorial ends and walks through the architectural decisions that make a 2025-era LLM different from a 2017-era one — each one a single, isolated commit you can read and reproduce.

---

## References

| Paper | Used for |
|-------|----------|
| Vaswani et al. (2017) — *Attention Is All You Need* | Baseline Transformer |
| Su et al. (2021) — *RoFormer: Enhanced Transformer with Rotary Position Embedding* | RoPE |
| Shazeer (2020) — *GLU Variants Improve Transformer* | SwiGLU |
| Ainslie et al. (2023) — *GQA: Training Generalized Multi-Query Transformer Models* | Grouped-Query Attention |
| Dao et al. (2022) — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* | FlashAttention |
| Leviathan et al. (2023) — *Fast Inference from Transformers via Speculative Decoding* | Speculative decoding |
| Touvron et al. (2024) — *The Llama 3 Herd of Models* | Reference for modern architecture choices |
| Hoffmann et al. (2022) — *Training Compute-Optimal Large Language Models* | Scaling laws |

---

## License

MIT — see [`LICENSE`](LICENSE).
