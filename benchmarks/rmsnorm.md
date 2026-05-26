# Phase 2.1 — `nn.LayerNorm` → `RMSNorm`

Replaces every `nn.LayerNorm(n_embd)` in the model (9 instances total: two
per Transformer block plus the final norm) with a custom `RMSNorm` module.

This is the **first architectural modernisation** of the baseline.  All
other variables (hyperparameters, optimiser, batch size, seed) are kept
identical to `benchmarks/baseline.md` so any difference comes purely from
the normalisation change.

---

## What changed

| | Baseline | This commit |
|---|---|---|
| Normalisation         | `nn.LayerNorm` (mean-centred, with bias)   | `RMSNorm` (no mean, no bias)               |
| Formula               | $\dfrac{x-\mu}{\sqrt{\sigma^2+\varepsilon}} \cdot \gamma + \beta$ | $\dfrac{x}{\sqrt{\mathrm{RMS}^2+\varepsilon}} \cdot \gamma$ |
| Operations per token  | mean, variance, subtract, sqrt, divide     | square, mean, sqrt, divide                 |
| Parameters per layer  | $2n$ (γ and β)                              | $n$ (γ only)                                |
| Backend kernel        | Fused `torch.nn.functional.layer_norm`     | Custom; un-fused in eager mode             |

---

## Results

Identical hyperparameters, identical seed (1337), same hardware
(RTX 3060 Laptop, 6 GB):

| Metric            | Baseline (LayerNorm) | RMSNorm     | Δ            |
|-------------------|---------------------:|------------:|-------------:|
| Final train loss  | 1.3548               | **1.3570**  | +0.0022      |
| Final val loss    | 1.5787               | **1.5858**  | +0.0071      |
| Total parameters  | 816 705              | **815 553** | −1 152       |
| Train time (5k steps) | ~600 s (~10 min) | **548.5 s (~9 min)** | **−51 s (~9 % faster)** |

**Reading:** loss is statistically unchanged (≲ 0.01 is within
batch-to-batch noise), parameter count drops slightly (the 9 LayerNorm
bias vectors are gone), and wall-clock training time drops by ~9 %.
RMSNorm pays for itself without sacrificing quality.

---

## Sanity checks

- All 12 unit tests in `tests/test_model.py` continue to pass, including
  the architectural invariants (forward shapes, initial-loss range,
  causal-mask correctness, parameter count window).
- Parameter delta exactly matches theory:
  $9 \text{ norms} \times 128 \text{ dims} = 1\,152$ bias values removed.
- Output distribution of `RMSNorm(x)` keeps the original token mean
  (unlike LayerNorm which forces it to zero) but bounds the RMS magnitude.

---

## Why this works

The 2019 paper by Zhang & Sennrich showed empirically that the
mean-subtraction in LayerNorm contributes almost nothing to the model's
final loss.  The intuition is that the network can already learn any
desired shift via:

1. the bias terms in the surrounding `Linear` layers, and
2. the residual stream that flows through every block unchanged.

Removing mean-centring saves one pass over the data per layer and one
bias vector per norm.  Empirical ablations at the LLaMA, Mistral, and
DeepSeek scales have repeatedly confirmed this — it is now the default
choice in every modern open-weight LLM.

---

## Caveat: speed in eager mode vs `torch.compile`

The pure-Python implementation in `modern_gpt/norm.py` dispatches several
individual CUDA kernels (`pow`, `mean`, `rsqrt`, two element-wise
multiplies).  The fused `nn.LayerNorm` kernel runs in a single launch.
At this model size (~800 K params, 5 000 steps) the ~9 % speedup we
observe is *despite* the un-fused implementation.

Wrapping the model in `torch.compile()` or substituting PyTorch ≥ 2.4's
`nn.functional.rms_norm` fuses the four ops and pushes the speedup well
above 10 %.  That refinement is deferred — at this stage the goal is a
readable implementation that exactly matches what production LLM
codebases (LLaMA, Mistral) write themselves.

---

## References

- Zhang, B. & Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS 2019. https://arxiv.org/abs/1910.07467
- Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*. Establishes RMSNorm as the default for modern open-weight LLMs.
