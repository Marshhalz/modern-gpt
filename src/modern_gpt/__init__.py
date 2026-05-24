"""modern-gpt — a small GPT progressively upgraded toward the LLaMA 3 architecture."""

from .config import GPTConfig
from .model import (
    GPT,
    Block,
    FeedForward,
    Head,
    MultiHeadAttention,
)

__version__ = "0.1.0"

__all__ = [
    "GPTConfig",
    "GPT",
    "Block",
    "Head",
    "MultiHeadAttention",
    "FeedForward",
]
