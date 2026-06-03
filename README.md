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
| 2 | learned PE → **Rotary Position Embeddings (RoPE)** | ✅ | **1.3258** | **1.5474** | −8 192 params, ~30 % slower than RMSNorm (rotation in hot path) — first quality gain above baseline — see [`benchmarks/rope.md`](benchmarks/rope.md) |
| 3 | ReLU FFN → **SwiGLU** | ✅ | **1.2894** | **1.5281** | +1 536 params, ~10 % slower — best val loss so far — see [`benchmarks/swiglu.md`](benchmarks/swiglu.md) |
| 4 | MHA → **Grouped-Query Attention (GQA)** | ✅ | **1.2993** | **1.5350** | −65 536 params, 4→2 KV heads (50 % smaller KV cache) — see [`benchmarks/gqa.md`](benchmarks/gqa.md) |
| 5 | **QK-Norm** (cosine attention) | ✅ | **1.2932** | **1.5303** | +260 params — bounds attention logits; cosine kernel — see [`benchmarks/qknorm.md`](benchmarks/qknorm.md) |
| 6 | Naive attention → **`F.scaled_dot_product_attention`** | ⬜ | – | – | FlashAttention via PyTorch fused kernel |
| 7 | **KV cache** for inference | ⬜ | – | – | $O(T)$ per step instead of $O(T^2)$ |
| 8 | **Speculative decoding** | ⬜ | – | – | Draft+verify, 2–3× faster sampling |

---

## Architecture overview (current)

```
tokens (B, T)
   │
   └── token_embedding (vocab_size, n_embd)             ← position handled by RoPE inside attention
   │
   ▼  (B, T, n_embd)
   │
┌──┴─────────────────────────────────────────┐
│  Block × n_layer                            │
│    ├── RMSNorm                              │  ✅ step 1
│    ├── Grouped-Query Attention + RoPE + QK-Norm │  ✅ steps 2, 4, 5 — FlashAttn in step 6
│    ├── + residual                           │
│    ├── RMSNorm                              │
│    ├── SwiGLU (gated, 8/3× expansion)       │  ✅ step 3
│    └── + residual                           │
└──┬─────────────────────────────────────────┘
   │
   ▼
RMSNorm (final)                                         ✅ step 1
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
