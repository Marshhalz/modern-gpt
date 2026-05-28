# Phase 2.2 — Learned PE → Rotary Position Embeddings (RoPE)

Replaces the `position_embedding_table` (`nn.Embedding(block_size, n_embd)`)
with **RoPE** — a parameter-free rotation applied directly to the query and key
vectors inside every attention head.

This is the position encoding used in LLaMA 1/2/3, Mistral, Qwen, DeepSeek,
and almost every modern open-weight LLM since 2022.

---

## What changed

| | RMSNorm baseline (prev. commit) | This commit (RoPE) |
|---|---|---|
| Position encoding      | Learned table `(block_size × n_embd)` | RoPE: rotate q and k by angle ∝ position |
| Learned position params | 64 × 128 = **8 192** | **0** |
| Where position lives   | Added to token embedding at input | Inside attention: `q = apply_rotary(q, cos, sin)` |
| Max sequence length    | Hard-capped at `block_size=64`  | Unlimited — rotation is defined for any integer position |
| Relative-position bias | None — absolute positions only  | Automatic — cos/sin identity makes absolute positions cancel in dot product |

New files: `src/modern_gpt/rope.py` (`RotaryEmbedding`, `apply_rotary`, `rotate_half`).  
Changed files: `src/modern_gpt/model.py` — `Head`, `MultiHeadAttention`, `Block` thread `cos, sin`; `GPT` removes `position_embedding_table`, changes `blocks` from `Sequential` → `ModuleList`.

---

## Results

Identical hyperparameters, identical seed (1337), same hardware
(RTX 3060 Laptop, 6 GB):

| Metric            | Baseline (LayerNorm) | RMSNorm     | **RoPE (this)** | Δ vs RMSNorm    |
|-------------------|---------------------:|------------:|----------------:|----------------:|
| Final train loss  | 1.3548               | 1.3570      | **1.3258**      | −0.0312         |
| Final val loss    | 1.5787               | 1.5858      | **1.5474**      | **−0.0384**     |
| Total parameters  | 816 705              | 815 553     | **807 361**     | −8 192          |
| Train time (5k steps) | ~600 s (~10 min) | ~549 s (~9 min) | **712.6 s (~12 min)** | +163 s (~30 % slower) |

**Reading:**

- **Quality improves meaningfully.** Val loss drops by 0.0384 vs RMSNorm and by
  0.0313 vs the original baseline — the largest quality gain of any phase so far,
  and the first change to beat the baseline on val loss.
- **8 192 fewer learned parameters.** The position table is gone entirely.
- **~30 % slower in eager mode.** Each forward pass now calls `apply_rotary` on q
  and k for every head in every layer (4 heads × 4 layers = 16 rotations per step).
  This is pure kernel-launch overhead in un-fused eager PyTorch.

---

## Sanity checks

- All existing tests in `tests/test_model.py` continue to pass — forward shapes,
  causal masking, generation, parameter count range.
- New tests in `tests/test_rope.py` verify:
  - `rotate_half` and `apply_rotary` are mathematically correct.
  - `RotaryEmbedding` has zero learnable parameters.
  - **Relative-position property:** `rotated_dot(5,3) == rotated_dot(10,8) == rotated_dot(20,18)`
    to within 1e-5 — the fundamental RoPE guarantee demonstrated numerically.
- Parameter delta exactly matches theory: `block_size × n_embd = 64 × 128 = 8 192` params removed.

---

## Why the quality improves

The learned PE table assigns each absolute slot an unrelated vector.  The model
must *learn* to extract relative distance from pairs of unrelated absolute
embeddings — an indirect route.

RoPE makes relative distance a structural property of the dot product.  For any
query at position $m$ and key at position $n$:

$$q_m^\top k_n = f(m - n)$$

The model does not need to learn this relationship — it is baked in by the
rotation.  The weights are free to focus on *what* to attend to, not *how to
encode distance*.

This is also why RoPE extrapolates to longer sequences: the rotation formula
$m \cdot \theta_i$ is well-defined for any integer $m$, not just the 64 slots
seen during training.

---

## Why it is slower

The current implementation is un-fused eager PyTorch.  `apply_rotary` dispatches
three separate CUDA kernels per call (`multiply`, `rotate_half`, `multiply+add`).
At 16 calls per forward pass (4 heads × 4 layers), that is 48 extra kernel
launches per training step.

Two paths to eliminate this overhead (deferred):
1. **`torch.compile()`** — fuses the three ops into one or two kernel launches.
2. **`F.scaled_dot_product_attention`** (Phase 2.6) — FlashAttention via PyTorch's
   fused kernel subsumes the rotation overhead entirely.

---

## The RKHS connection

RoPE makes the attention kernel shift-invariant:
$$k_\text{pos}(m, n) = g(m - n)$$

By **Bochner's theorem**, every continuous shift-invariant positive-definite
kernel is the Fourier transform of a non-negative measure.  RoPE's frequencies
$\theta_i = \text{base}^{-2i/d}$ are a discrete set of Fourier basis frequencies —
RoPE is doing **Fourier feature approximation on position**, constructing a
shift-invariant kernel without explicitly naming it.

The same shift-invariance axiom defines the **RBF/Gaussian kernel** studied in
classical kernel theory (Shawe-Taylor & Cristianini, Ch. 3).  RoPE imports this
structural assumption from the kernel literature into the attention mechanism.

---

## References

- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* https://arxiv.org/abs/2104.09864
- Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* Establishes RoPE as the default for modern open-weight LLMs.
- Bochner, S. (1933). Monotone Funktionen, Stieltjes Integrale und harmonische Analyse. — The classical theorem connecting shift-invariant kernels and Fourier transforms.
