"""
config.py

Stores the hyperparameter configurations for the Scaling Laws study.
These architectures are designed to span 3 orders of magnitude in parameter count
(1M to 100M) as required by the project instructions.

Estimates based on standard Transformer formula: P ≈ 12 * n_layer * d_model^2
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int = 256
    vocab_size: int = 128
    dropout: float = 0.1




GPT_TINY = ModelConfig(
    n_layer=4,
    n_head=4,
    n_embd=128
)


GPT_SMALL = ModelConfig(
    n_layer=6,
    n_head=8,
    n_embd=256
)


GPT_MEDIUM = ModelConfig(
    n_layer=12,
    n_head=6,
    n_embd=384
)


GPT_LARGE = ModelConfig(
    n_layer=16,
    n_head=8,
    n_embd=512
)



GPT_XL = ModelConfig(
    n_layer=16,
    n_head=12,
    n_embd=768
)


MODEL_CONFIGS = {
    "tiny": GPT_TINY,
    "small": GPT_SMALL,
    "medium": GPT_MEDIUM,
    "large": GPT_LARGE,
    "xl": GPT_XL
}
