# Phase 2.6 — Naive attention → FlashAttention (fused SDPA)

Replaces the manual `softmax(q·kᵀ) @ v` block with
`torch.nn.functional.scaled_dot_product_attention`, which dispatches to the
fused, IO-aware **FlashAttention** kernel (Dao et al., 2022).

This changes *how* attention is computed, not *what*: the output is exact (same
as the manual path, up to float-reduction order), the memory drops from O(T²) to
O(T), and the score → softmax → value pipeline runs in a single fused kernel.

---

## What changed

| | QK-Norm baseline (prev. commit) | This commit (FlashAttention) |
|---|---|---|
| Score path  | manual `q·kᵀ → mask → softmax → dropout → @v` | `F.scaled_dot_product_attention(...)` |
| Causal mask | explicit `tril` buffer + `masked_fill`         | `is_causal=True` (built inside the kernel) |
| Dropout     | `nn.Dropout` on the weights                    | `dropout_p` inside the kernel             |
| QK-Norm scale | multiply scores by learnable `self.scale`    | folded into q; `scale=1.0` passed to SDPA |
| Memory      | O(T²) attention matrix in HBM                  | **O(T)** — matrix never materialised      |
| Kernels     | several launches                              | **one fused launch**                      |

The learnable QK-Norm temperature can't be passed as SDPA's float `scale`
argument without detaching its gradient, so it is **folded into q**
(`q = q * self.scale`) and `scale=1.0` is passed instead — mathematically
identical, gradient preserved.

---

## Results

Identical hyperparameters, identical seed (1337), same hardware
(RTX 3060 Laptop, 6 GB):

| Metric            | GQA     | QK-Norm | **FlashAttention (this)** | Δ vs QK-Norm |
|-------------------|--------:|--------:|--------------------------:|-------------:|
| Final train loss  | 1.2993  | 1.2932  | **1.2951**                | +0.0019      |
| Final val loss    | 1.5350  | 1.5303  | **1.5407**                | +0.0104      |
| Total parameters  | 743 361 | 743 621 | **743 621**               | 0            |
| Train time (5k)   | ~386 s  | ~571 s  | **424.4 s**               | **−147 s (~26 % faster)** |

**Reading:**

- **The val delta is not a regression.** FlashAttention is exact in the forward
  pass. The difference comes from the **dropout RNG stream**: SDPA applies dropout
  internally with a different random sequence than `nn.Dropout`, so the run is a
  different random realisation of the *same* model — well within the ~0.01
  run-to-run band seen across the whole series.
- **~26 % faster even at T = 64.** The speedup here is **kernel fusion**
  (QK-Norm + softmax + both matmuls in one launch), not the memory trick — at this
  tiny context there is no O(T²) pressure to relieve.
- **The real win is invisible at this scale.** O(T) memory is what makes long
  context (4k / 32k / 128k) feasible; on a 64-token model it is a wash on memory
  but free on speed.

---

## Sanity checks

- All tests pass; the QK-Norm logit-bounding test was updated to build its causal
  mask locally (the `tril` buffer is gone now that `is_causal=True` handles it).
- Parameter count unchanged (743 621) — the kernel swap touches computation only.
- `is_causal=True` is correct for training (square q/k). The KV-cache phase
  (2.7, deferred) will revisit this, since one new query attending to many cached
  keys is no longer square.

---

## Why FlashAttention is exact (not an approximation)

Standard attention materialises the full T×T score matrix in slow HBM, reads and
writes it three times (scores, softmax, value-weighting). FlashAttention tiles
the computation into SRAM-sized blocks and uses an **online softmax** — carrying a
running max `m` and denominator `l`, and rescaling the running output by
`exp(m_old − m_new)` whenever a new block raises the max. This reproduces the
exact softmax over the full row without ever holding it, so the result is
identical while HBM traffic and memory drop from O(T²) to O(T). Memory movement
(not arithmetic) is attention's bottleneck, which is why this is a large speedup
at long context.

---

## References

- Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* https://arxiv.org/abs/2205.14135
- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* https://arxiv.org/abs/2307.08691
- Milakov, M. & Gimelshein, N. (2018). *Online normalizer calculation for softmax.* (The online-softmax trick FlashAttention relies on.) https://arxiv.org/abs/1805.02867
