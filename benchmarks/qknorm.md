# Phase 2.5 — QK-Norm (cosine attention)

Normalises the query and key vectors (per head) before the attention dot
product, and replaces the fixed `1/sqrt(head_size)` scale with a **learnable
temperature**.  After normalisation the dot product is a cosine similarity, so
attention logits are bounded regardless of how large q/k grow during training.

Added inside `GroupedQueryAttention` (`attention.py`) — it is two `RMSNorm`
layers and one scalar parameter, applied after RoPE and before the score.

References: Henry et al. (2020), *Query-Key Normalization for Transformers*;
Dehghani et al. (2023), *Scaling ViT to 22B* (QK-Norm at scale).

---

## What changed

| | GQA baseline (prev. commit) | This commit (QK-Norm) |
|---|---|---|
| Score | `q · kᵀ / sqrt(head_size)`            | `q̂ · k̂ᵀ · τ` (q̂, k̂ RMS-normed)   |
| Scale | fixed `head_size^-0.5`               | learnable temperature `τ`            |
| Logit range | grows with ‖q‖‖k‖             | bounded (cosine similarity)          |
| Kernel view | unbounded inner product       | normalised **cosine kernel** on the sphere |

New per layer: `q_norm` (head_size γ), `k_norm` (head_size γ), `scale` (1) =
`2·head_size + 1` params.

---

## Results

Identical hyperparameters, identical seed (1337), same hardware
(RTX 3060 Laptop, 6 GB):

| Metric            | SwiGLU  | GQA     | **QK-Norm (this)** | Δ vs GQA |
|-------------------|--------:|--------:|-------------------:|---------:|
| Final train loss  | 1.2894  | 1.2993  | **1.2932**         | −0.0061  |
| Final val loss    | 1.5281  | 1.5350  | **1.5303**         | −0.0047  |
| Total parameters  | 808 897 | 743 361 | **743 621**        | +260     |
| Train time (5k)   | ~784 s  | ~386 s  | **571.4 s**        | ~48 % slower |

**Reading:**

- **Both losses improve slightly** — val 1.5303, second-best of the series and
  effectively tied with SwiGLU's 1.5281. On a toy model this is small; the real
  value is stability, which a 5k-step run cannot stress.
- **+260 params** — `4 layers × (32 + 32 + 1)`. Negligible.
- **~48 % slower** — two extra un-fused `RMSNorm` passes (on q and on k) per
  attention, in eager mode. `torch.compile` / FlashAttention (Phase 2.6) fuse
  these away.

---

## Why it matters (beyond the toy numbers)

Standard attention assumes q and k have ~unit variance per dimension, which the
`1/sqrt(d)` scale corrects for.  But nothing *keeps* them there: during training
the q/k projections can grow, inflating logits until softmax saturates — one
token gets ~all the attention and the gradient through the attention weights
vanishes.  This is a documented instability at large model size and long
training (it is exactly why Dehghani et al. needed QK-Norm to train a 22B ViT).

Normalising q and k bounds the logits to `[-1, 1] · τ` permanently, and the
learnable temperature lets the model *choose* its attention sharpness instead of
inheriting it from a fixed architectural constant.

---

## The kernel-theory view (RKHS seminar bridge)

After normalisation the attention score is the **cosine (angular) kernel**

$$k_\text{cos}(q,k) = \frac{q \cdot k}{\lVert q\rVert\,\lVert k\rVert} = \hat q \cdot \hat k \in [-1, 1],$$

a positive-definite kernel on the unit sphere $S^{d-1}$.  Combined with RoPE —
which makes the score **stationary** (shift-invariant) in position — modern
attention is a composition of two classical kernel-theoretic primitives:

| Component | Kernel property |
|-----------|-----------------|
| RoPE      | stationary kernel in position (Bochner) |
| QK-Norm   | normalised cosine kernel in content space |
| Softmax   | row-stochastic normalisation of the kernel matrix |

This is the connection developed for the RKHS seminar.

---

## Sanity checks

- All existing tests pass; parameter range test updated (~744K).
- New tests in `tests/test_attention.py`:
  - `q_norm`/`k_norm` are per-head (`head_size`) and `scale` is a learnable
    parameter;
  - **logit bounding**: scaling the input by 1000× does not collapse softmax
    entropy — the saturation failure mode QK-Norm prevents.

---

## References

- Henry, A., Dachapally, P. R., Pawar, S., & Chen, Y. (2020). *Query-Key Normalization for Transformers.* https://arxiv.org/abs/2010.04245
- Dehghani et al. (2023). *Scaling Vision Transformers to 22 Billion Parameters.* (QK-Norm used to stabilise training at scale.) https://arxiv.org/abs/2302.05442
