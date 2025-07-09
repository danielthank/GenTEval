from dataclasses import dataclass


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
    use_root_mlp: bool = True  # If True, use MLP for root duration, else use VAE

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def to_dict(self):
        return self.__dict__
