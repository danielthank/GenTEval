from dataclasses import dataclass
from enum import Enum


class RootModel(Enum):
    VAE = "VAE"
    MLP = "MLP"
    TABLE = "TABLE"


@dataclass
class MarkovGenTConfig:
    # VAE for start_time
    start_time_latent_dim: int = 16
    start_time_hidden_dim: int = 64
    start_time_epochs: int = 20

    # Markov chain for graph
    markov_order: int = 1
    max_depth: int = 10
    max_children: int = 5000

    # Root span duration models
    root_latent_dim: int = 32
    root_hidden_dim: int = 128
    root_epochs: int = 500

    # Metadata VAE/NN
    metadata_latent_dim: int = 32
    metadata_hidden_dim: int = 128
    metadata_epochs: int = 10

    # General
    batch_size: int = 64
    learning_rate: float = 0.001

    # Compression options
    save_decoders_only: bool = True

    # Model selection
    root_model: RootModel = RootModel.TABLE

    @classmethod
    def from_dict(cls, config_dict):
        # Create a copy to avoid modifying the original
        config_dict = config_dict.copy()
        
        # Convert root_model string to enum if needed
        if "root_model" in config_dict and isinstance(config_dict["root_model"], str):
            config_dict["root_model"] = RootModel(config_dict["root_model"])
            
        return cls(**config_dict)

    def to_dict(self):
        result = self.__dict__.copy()
        # Convert enum to string for serialization
        if "root_model" in result:
            result["root_model"] = result["root_model"].value
        return result
