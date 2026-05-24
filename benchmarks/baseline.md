# Baseline benchmark — vanilla GPT-2 architecture

This document records the exact configuration, training procedure, and
results for the **Phase 1 baseline** model.  Every subsequent architectural
change in this repository is benchmarked against the numbers below.

---

## Architecture

Vanilla decoder-only Transformer (GPT-2 style):

| Component             | Choice                                    |
|-----------------------|-------------------------------------------|
| Tokenisation          | Character-level (no BPE)                  |
| Position embeddings   | Learned (`nn.Embedding(block_size, n_embd)`) |
| Normalisation         | `nn.LayerNorm`, pre-norm placement        |
| Attention             | Standard Multi-Head Self-Attention (4 heads) |
| Attention mask        | Causal (lower-triangular)                 |
| FFN activation        | ReLU, 4× expansion                        |
| Residuals             | Around each sub-layer (pre-norm)          |
| Weight init           | `N(0, 0.02)` for Linear and Embedding     |
| Output head           | Untied `Linear(n_embd, vocab_size)`       |

### Hyperparameters (`GPTConfig`)

| Name         | Value | Meaning                                  |
|--------------|-------|------------------------------------------|
| `vocab_size` | 65    | Unique characters in Tiny Shakespeare    |
| `block_size` | 64    | Context window length                    |
| `n_embd`     | 128   | Residual stream / embedding width        |
| `n_head`     | 4     | Number of attention heads                |
| `n_layer`    | 4     | Number of Transformer blocks             |
| `head_size`  | 32    | Derived as `n_embd // n_head`            |
| `dropout`    | 0.1   | Applied after attention and FFN          |

**Total trainable parameters:** **816,705**

---

## Training setup

| Item               | Value                                                |
|--------------------|------------------------------------------------------|
| Dataset            | Tiny Shakespeare (~1 MB), 90/10 train/val split      |
| Batch size         | 64                                                   |
| Iterations         | 5 000                                                |
| Optimiser          | AdamW (default β₁=0.9, β₂=0.999, weight_decay=0.01)  |
| Learning rate      | 3 × 10⁻⁴                                              |
| LR schedule        | Constant (no warmup, no decay)                       |
| Gradient clipping  | None                                                 |
| Mixed precision    | No (fp32)                                            |
| Eval interval      | Every 500 steps                                      |
| Eval batches       | 200 per split, averaged                              |
| Seed               | 1337                                                 |
| Hardware           | NVIDIA RTX 3060 Laptop GPU, 6 GB VRAM                |

Reproducible with `python scripts/train.py`.

---

## Results

### Loss trajectory

| Step  | Train loss | Val loss | Train–val gap |
|------:|-----------:|---------:|--------------:|
|    0  | 3.9241     | 3.9261   | -0.002        |
|  500  | 1.9913     | 2.0543   | -0.063        |
| 1000  | 1.6700     | 1.8359   | -0.166        |
| 1500  | 1.5573     | 1.7453   | -0.188        |
| 2000  | 1.5013     | 1.6938   | -0.193        |
| 2500  | 1.4577     | 1.6606   | -0.203        |
| 3000  | 1.4184     | 1.6242   | -0.206        |
| 3500  | 1.4034     | 1.6150   | -0.212        |
| 4000  | 1.3837     | 1.5970   | -0.213        |
| 4500  | 1.3665     | 1.5827   | -0.217        |
| 4999  | **1.3548** | **1.5787** | -0.224      |

### Sanity checks

- **Initial loss ≈ ln(vocab_size).**  Expected `ln(65) ≈ 4.174`; observed `3.92`. ✔
- **Loss drops fast in first 500 steps** then continues to decrease monotonically. ✔
- **Train–val gap grows slowly** (-0.06 at step 500 → -0.22 at step 5 000), indicating mild but non-pathological overfitting on 1 MB of text. ✔

### Headline numbers

| Metric           | Value     |
|------------------|-----------|
| Final train loss | **1.3548** |
| Final val loss   | **1.5787** |
| Loss reduction   | 4.17 → 1.58 (62 % relative reduction) |
| Train time       | ~4 min on RTX 3060 (6 GB)             |

These numbers define the bar that every Phase 2 modification must beat
(or, in the case of efficiency-oriented changes such as FlashAttention,
match while delivering a measurable speed-up).

---

## Notes on the train/val gap

The growing gap is expected: with only 1 MB of training text and ~800 K
parameters, the model is overparameterised relative to the dataset and
will eventually memorise.  Dropout 0.1 slows but does not eliminate
this.  Two later commits will help:

- **SwiGLU** (Phase 2.3) slightly reduces overfitting empirically vs. ReLU
- **Bigger context window with FlashAttention** (Phase 2.6) lets the
  model use more data per gradient step, improving sample efficiency

For comparison purposes the baseline is "frozen" — it will not be
re-tuned to look better.
