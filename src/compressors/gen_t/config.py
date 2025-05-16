from dataclasses import dataclass
from typing import Tuple


@dataclass
class GenTConfig:
    chain_length: int = 3
    epochs: int = 100
    batch_size: int = 500
    discriminator_dim: Tuple[int, ...] = (128,)
    generator_dim: Tuple[int, ...] = (128,)
