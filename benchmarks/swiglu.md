# Phase 2.3 — ReLU FFN → SwiGLU

Replaces the original two-matrix ReLU feed-forward network with a three-matrix
**SwiGLU** gated FFN (Shazeer, 2020) — the feed-forward variant used in
LLaMA 1/2/3, Mistral, Qwen, and DeepSeek.

---

## What changed

| | RoPE baseline (prev. commit) | This commit (SwiGLU) |
|---|---|---|
| FFN form     | `W_down · ReLU(W_up · x)`            | `W_down · [SiLU(W_gate · x) ⊙ (W_up · x)]` |
| Matrices     | 2 (`W_up`, `W_down`)                 | 3 (`W_gate`, `W_up`, `W_down`)             |
| Activation   | ReLU (hard cutoff, dead neurons)     | SiLU / Swish (smooth, no dead neurons)     |
| Gating       | none                                 | element-wise gate controls information flow |
| Hidden dim   | 4 × n_embd = 512                     | ⌈2/3 · 4 · n_embd⌉₈ = 344 (LLaMA rule)     |
| Bias         | yes                                  | no (LLaMA convention)                      |

New file: `src/modern_gpt/ffn.py` (`SwiGLU`).
Changed: `model.py` (`Block` uses `SwiGLU`, old `FeedForward` removed),
`config.py` (`ffn_hidden_dim` property).

---

## Results

Identical hyperparameters, identical seed (1337), same hardware
(RTX 3060 Laptop, 6 GB):

| Metric            | Baseline | RMSNorm | RoPE    | **SwiGLU (this)** | Δ vs RoPE |
|-------------------|---------:|--------:|--------:|------------------:|----------:|
| Final train loss  | 1.3548   | 1.3570  | 1.3258  | **1.2894**        | −0.0364   |
| Final val loss    | 1.5787   | 1.5858  | 1.5474  | **1.5281**        | **−0.0193** |
| Total parameters  | 816 705  | 815 553 | 807 361 | **808 897**       | +1 536    |
| Train time (5k)   | ~600 s   | ~549 s  | ~713 s  | **784.1 s**       | ~10 % slower |

**Reading:**

- **Best val loss in the series so far (1.5281).** SwiGLU beats RoPE on both
  train and val loss — the gated activation is a genuine quality win, not just
  a wash.
- **+1 536 parameters** — exactly `4 layers × 384`.  Per block, SwiGLU's three
  344-wide matrices (132 096 params) cost 384 more than the two 512-wide ReLU
  matrices plus biases (131 712).  The 2/3 hidden-dim rule keeps this delta
  tiny by design.
- **~10 % slower than RoPE.** Three matmuls + a SiLU + an element-wise multiply
  per block, versus two matmuls + a ReLU.  More FLOPs for more quality.

---

## Sanity checks

- All tests in `tests/test_model.py` continue to pass.
- New tests in `tests/test_ffn.py` verify:
  - output shape preserved;
  - exactly three bias-free linear layers;
  - hidden dim follows the 2/3 rule (344) and is a multiple of 8;
  - **gating works**: zeroing `W_gate` forces `SiLU(0)·up = 0`, so the whole
    FFN output collapses to zero — proof the gate genuinely controls flow.
- Parameter delta matches theory exactly: per block
  `3·(128·344) − (2·128·512 + 512 + 128) = 132 096 − 131 712 = 384`,
  times 4 layers = 1 536.

---

## Why this works

**SiLU vs ReLU.** ReLU sends every negative pre-activation to exactly 0, with
zero gradient — a unit pushed negative can never recover ("dead neuron").
SiLU, `z·σ(z)`, is smooth and slightly negative for small negative inputs, so
gradients always flow.

**Gating.** The decisive ingredient is not the activation but the element-wise
product `SiLU(W_gate·x) ⊙ (W_up·x)`.  The gate branch learns, per hidden unit,
a multiplier that suppresses (≈0), passes (≈1), or amplifies (>1) the content
branch.  This is a multiplicative, input-dependent interaction that a plain
ReLU MLP cannot represent without far more depth.  Shazeer's 2020 ablation
found GLU variants consistently improved Transformer quality at equal
parameter budgets; every major open-weight LLM since LLaMA has adopted SwiGLU.

---

## References

- Shazeer, N. (2020). *GLU Variants Improve Transformer.* https://arxiv.org/abs/2002.05202
- Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* Establishes SwiGLU with the 2/3 hidden-dim rule as the modern default.
- Hendrycks, D. & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs).* Background on smooth activations related to SiLU/Swish.
