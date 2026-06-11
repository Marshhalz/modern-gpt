# Phase 3.1 — Scaling Laws: modern-gpt on WikiText-103

Trains the full modern-gpt stack (RMSNorm · RoPE · SwiGLU · GQA · QK-Norm · FlashAttention)
at five sizes, fits a power-law `L(N) = a·N^(−b) + c` on the four smallest, and predicts the
fifth. The prediction error is the result — it shows that small runs forecast large-scale behaviour.

---

## Fitted law

```
L(N) = 4.306 · N^(−0.1191) + 1.690
```

| Coefficient | Value | Meaning |
|---|---|---|
| `a = 4.306` | scale | steepness of the curve |
| `b = 0.1191` | exponent | rate of improvement per 10× in params |
| `c = 1.690` | floor | irreducible loss — entropy of the data/tokenizer |

---

## Size sweep results

Fixed token budget: **25 M tokens per model**, WikiText-103, BPE vocab 4 096, block size 256.

| Layers | Width | N (non-embed) | N (total) | Val loss | Time |
|-------:|------:|--------------:|----------:|---------:|-----:|
| 2 | 128 | 363 650 | 1 416 322 | 2.6263 | 127 s |
| 3 | 192 | 1 181 955 | 2 758 915 | 2.5066 | 167 s |
| 4 | 256 | 2 903 812 | 5 005 060 | 2.4177 | 210 s |
| 5 | 320 | 5 343 365 | 7 968 901 | 2.3720 | 282 s |
| **6** | **384** | **9 445 254** | **12 595 078** | **2.3171** | **381 s** |

The largest model (L=6, bold) was **held out** — not used to fit the curve.

---

## Prediction result

| | Val loss |
|---|---|
| Predicted (from 4 small models) | **2.3256** |
| Actual (measured) | **2.3171** |
| **Prediction error** | **0.0085 (0.37%)** |

Four models predicted a 2.6× larger model's loss to within 0.37%. The held-out star and
predicted × land nearly on top of each other on the log-scale plot.

---

## Reading the exponent

`b = 0.1191` is steeper than Kaplan et al.'s `b ≈ 0.076`. This is expected:

- Kaplan measured on models spanning 10⁷–10¹⁰ parameters with hundreds of billions of tokens.
- Our range is 3.6×10⁵–9.4×10⁶ — the small-model regime where each added parameter improves
  loss more per unit than at larger scale.
- At our scale the curve hasn't yet flattened into the shallower power law that emerges at
  frontier scale. The exponent will drift toward Kaplan's value as models and data grow.

The floor `c = 1.690` is the entropy the tokenizer cannot compress further — no model size
removes it.

---

## Methodology (what makes this honest)

1. **Non-embedding parameters for N.** Embedding (`tok`) and `lm_head` scale with vocab size,
   not model depth/width, and distort the curve. Stripped following Kaplan and Hoffmann.
2. **WikiText-103, not Shakespeare.** Shakespeare (1 MB) lets big models memorize → measures
   the data wall, not scaling. WikiText-103 (~38 M tokens after BPE) prevents memorization
   at our scale.
3. **Fixed token budget.** Each size sees exactly 25 M tokens so the only variable is size.
   Honest caveat: the largest model is mildly data-limited at a fixed budget; raising
   `TRAIN_TOKENS` would push it slightly lower and steepen the held-out prediction.
4. **Word-frequency BPE.** The notebook-08 BPE algorithm, run on unique words weighted by
   frequency instead of the raw corpus — same merges, ~1000× faster on 80 MB of text.

---

## Caveats

- Single run, no LR schedule, no multiple seeds — wide error bars. A real study repeats
  and reports variance across seeds.
- Five points is the minimum for a credible fit. More sizes would tighten the curve.
- The fixed budget makes the largest model slightly data-limited; its actual loss would be
  lower with more tokens, making the curve cleaner.

---

## References

- Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models.* https://arxiv.org/abs/2001.08361
- Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models (Chinchilla).* https://arxiv.org/abs/2203.15556
