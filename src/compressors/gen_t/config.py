from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass
class GenTConfig:
    chain_length: int = 3
    epochs: int = 5
    batch_size: int = 500
    discriminator_dim: Tuple[int, ...] = (64,)
    generator_dim: Tuple[int, ...] = (64,)
    span_cnt_threshold: int = 10

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the dataclass to a dictionary suitable for msgpack serialization.
        This handles any custom serialization logic for complex types.
        """
        data = asdict(self)

        # Handle any special conversions if needed
        # For example, ensure tuples are converted to lists for msgpack compatibility
        data["discriminator_dim"] = list(self.discriminator_dim)
        data["generator_dim"] = list(self.generator_dim)

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenTConfig":
        """
        Create a GenTConfig instance from a dictionary.
        This handles any custom deserialization logic needed.
        """
        # Handle any special conversions from msgpack-compatible types back to Python types
        if "discriminator_dim" in data and isinstance(data["discriminator_dim"], list):
            data["discriminator_dim"] = tuple(data["discriminator_dim"])

        if "generator_dim" in data and isinstance(data["generator_dim"], list):
            data["generator_dim"] = tuple(data["generator_dim"])

        return cls(**data)
