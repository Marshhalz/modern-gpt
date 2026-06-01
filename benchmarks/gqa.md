# Phase 2.4 — MHA → Grouped-Query Attention (GQA)

Replaces multi-head attention with **Grouped-Query Attention** (Ainslie et al.,
2023): the `n_head` query heads now share only `n_kv_head` key/value heads.
At the default config that is **4 query heads, 2 K/V heads** — the variant used
in LLaMA 2/3, Mistral, and Qwen.

This is the first change motivated by **inference efficiency** rather than
quality. The loss should stay essentially unchanged; the win is a smaller KV
cache, which we will exploit in Phase 2.7.

---

## What changed

| | SwiGLU baseline (prev. commit) | This commit (GQA) |
|---|---|---|
| Query heads        | 4                                  | 4                                  |
| Key/value heads    | 4                                  | **2** (each shared by 2 queries)   |
| K/V projection size| `n_embd → n_head·head_size` (128)  | `n_embd → n_kv_head·head_size` (64)|
| Implementation     | Python loop over per-head modules  | single batched tensor op           |
| KV cache (future)  | 4 heads stored                     | **2 heads stored (50 % smaller)**  |

New file: `src/modern_gpt/attention.py` (`GroupedQueryAttention`).
Removed: per-head `Head` and `MultiHeadAttention` classes from `model.py`.
Changed: `config.py` adds `n_kv_head` and the `n_rep` property.

---

## Results

Identical hyperparameters, identical seed (1337), same hardware
(RTX 3060 Laptop, 6 GB):

| Metric            | RoPE    | SwiGLU  | **GQA (this)** | Δ vs SwiGLU |
|-------------------|--------:|--------:|---------------:|------------:|
| Final train loss  | 1.3258  | 1.2894  | **1.2993**     | +0.0099     |
| Final val loss    | 1.5474  | 1.5281  | **1.5350**     | +0.0069     |
| Total parameters  | 807 361 | 808 897 | **743 361**    | −65 536     |
| Train time (5k)   | ~713 s  | ~784 s  | **385.7 s**    | ~2× faster  |

**Reading:**

- **Quality cost is negligible** — val loss rises by 0.0069, well within
  run-to-run noise. That is the expected, acceptable trade: GQA buys cheaper
  inference for ~no quality loss.
- **−65 536 parameters** — `k_proj` and `v_proj` each shrink from 4 to 2 heads.
  Per layer that removes `2 · (128·128 − 128·64) = 16 384`; over 4 layers,
  65 536.
- **KV cache halved (the real point).** With 2 K/V heads instead of 4, the
  per-token cache at inference is 50 % smaller. Not visible during training —
  it shows up in Phase 2.7 when we add the cache and generate long sequences.

---

## Honest note on the 2× speedup

The ~2× wall-clock drop (784 s → 386 s) is **mostly from the vectorised
rewrite, not from GQA itself.** The previous attention looped over heads in
Python (`torch.cat([h(x) for h in self.heads])`), launching many small ops; the
new `GroupedQueryAttention` computes all heads in one batched
`(B, n_head, T, head_size)` matmul. Reducing K/V heads from 4 to 2 contributes a
little, but the dominant factor is removing the Python-level head loop.

Reported this way deliberately — the parameter saving and KV-cache reduction
are the genuine GQA effects; the speedup is largely an implementation cleanup
that any production codebase already does.

---

## Sanity checks

- All tests in `tests/test_model.py` continue to pass (forward shapes, causal
  masking, generation, parameter range).
- New tests in `tests/test_attention.py` verify:
  - `n_rep = n_head // n_kv_head` (1 = MHA, n_head = MQA);
  - non-divisible `n_kv_head` raises;
  - `k_proj`/`v_proj` are narrower than `q_proj`, all bias-free;
  - **causality**: perturbing the last token leaves earlier outputs unchanged;
  - GQA has strictly fewer params than the equivalent MHA (`n_kv_head == n_head`).

---

## Why this works

A query head asks "what am I looking for?"; a key/value head answers "here is
what I offer, and here is the content." Empirically, many query heads can share
the same K/V projections with little quality loss — the queries still
differentiate what each head attends to. Ainslie et al. (2023) showed GQA
recovers almost all of MHA's quality at a fraction of the KV-cache cost, and it
interpolates smoothly between MHA (`n_kv_head = n_head`) and MQA
(`n_kv_head = 1`). LLaMA 3 70B uses 64 query heads with only 8 K/V heads — an 8×
cache reduction that makes long-context serving feasible.

---

## References

- Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* https://arxiv.org/abs/2305.13245
- Shazeer, N. (2019). *Fast Transformer Decoding: One Write-Head is All You Need.* (MQA, the n_kv_head = 1 extreme.) https://arxiv.org/abs/1911.02150
- Touvron et al. (2024). *The Llama 3 Herd of Models.* Uses GQA with 8 K/V heads.
