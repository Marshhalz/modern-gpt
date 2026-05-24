"""
Reproducible training script for the modern-gpt baseline.

Run:
    python scripts/train.py

What it does:
    1. Downloads Tiny Shakespeare (~1 MB) into ./data if missing
    2. Builds a character-level vocabulary
    3. Trains the default GPTConfig model for `MAX_ITERS` steps
    4. Logs train/val loss every `EVAL_INTERVAL` steps
    5. Saves the checkpoint (+ hyperparameters + final losses) to ./checkpoints

This is the exact script used to produce the numbers reported in
benchmarks/baseline.md and the README.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

import torch

from modern_gpt import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 1337

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_URL  = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_PATH = Path("data/tinyshakespeare.txt")
CKPT_PATH = Path("checkpoints/baseline.pt")

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
MAX_ITERS     = 5000
BATCH_SIZE    = 64
LEARNING_RATE = 3e-4         # AdamW, no warmup, no schedule (kept simple)
EVAL_INTERVAL = 500
EVAL_ITERS    = 200          # batches averaged for each loss estimate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_dataset() -> str:
    """Download Tiny Shakespeare if not present, return raw text."""
    DATA_PATH.parent.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Downloading {DATA_URL} -> {DATA_PATH}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


def build_tokenizer(text: str) -> tuple[int, dict[str, int]]:
    chars      = sorted(set(text))
    vocab_size = len(chars)
    stoi       = {ch: i for i, ch in enumerate(chars)}
    return vocab_size, stoi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- data -----------------------------------------------------------
    text = download_dataset()
    vocab_size, stoi = build_tokenizer(text)
    encode = lambda s: [stoi[c] for c in s]
    data       = torch.tensor(encode(text), dtype=torch.long)
    n          = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]
    print(f"Dataset: {len(text):,} chars | vocab: {vocab_size} | "
          f"train: {len(train_data):,} | val: {len(val_data):,}")

    # ---- model ----------------------------------------------------------
    cfg = GPTConfig(vocab_size=vocab_size)
    model = GPT(cfg).to(device)
    print(f"Model:   {model.num_parameters():,} parameters")
    print(f"Config:  {cfg}")

    # ---- training utilities --------------------------------------------
    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        d  = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - cfg.block_size, (BATCH_SIZE,))
        x  = torch.stack([d[i : i + cfg.block_size]         for i in ix])
        y  = torch.stack([d[i + 1 : i + cfg.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        model.eval()
        out: dict[str, float] = {}
        for split in ["train", "val"]:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # ---- train loop ----------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history: list[dict] = []
    t0 = time.time()

    print(f"\nTraining for {MAX_ITERS} iterations (eval every {EVAL_INTERVAL})...")
    for step in range(MAX_ITERS):
        xb, yb  = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses = estimate_loss()
            elapsed = time.time() - t0
            print(
                f"step {step:5d} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"{elapsed:6.1f}s"
            )
            history.append({"step": step, **losses, "elapsed_s": elapsed})

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.1f}s.")

    # ---- save checkpoint -----------------------------------------------
    CKPT_PATH.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config":           asdict(cfg),
            "history":          history,
            "final_train_loss": history[-1]["train"],
            "final_val_loss":   history[-1]["val"],
            "total_seconds":    total_time,
            "seed":             SEED,
            "hyperparameters": {
                "max_iters":     MAX_ITERS,
                "batch_size":    BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "optimizer":     "AdamW",
            },
        },
        CKPT_PATH,
    )
    print(f"Saved checkpoint to {CKPT_PATH}")


if __name__ == "__main__":
    main()
