from dataclasses import dataclass
from enum import Enum


class RootModel(Enum):
    VAE = "VAE"
    MLP = "MLP"
    TABLE = "TABLE"


class MetadataModel(Enum):
    VAE = "VAE"


class MrfEdgeApproach(Enum):
    BASIC = "BASIC"
    PREV_SIBLING_AWARE = "PREV_SIBLING_AWARE"


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
    metadata_latent_dim: int = 4
    metadata_hidden_dim: int = 16
    metadata_epochs: int = 500
    early_stopping_patience: int = 10
    num_beta_components: int = 5

    # Flow-based prior for metadata VAE
    use_flow_prior: bool = False
    prior_flow_layers: int = 4
    prior_flow_hidden_dim: int = 16

    # Beta scheduling for VAE
    beta: float = 10

    # General
    batch_size: int = 256
    learning_rate: float = 0.001

    # Compression options
    save_decoders_only: bool = True

    # Model selection
    mrf_edge_approach: MrfEdgeApproach = MrfEdgeApproach.BASIC
    root_model: RootModel = RootModel.TABLE
    metadata_model: MetadataModel = MetadataModel.VAE

    @classmethod
    def from_dict(cls, config_dict):
        # Create a copy to avoid modifying the original
        config_dict = config_dict.copy()

        # Convert root_model string to enum if needed
        if "root_model" in config_dict and isinstance(config_dict["root_model"], str):
            config_dict["root_model"] = RootModel(config_dict["root_model"])

        # Convert metadata_model string to enum if needed
        if "metadata_model" in config_dict and isinstance(
            config_dict["metadata_model"], str
        ):
            config_dict["metadata_model"] = MetadataModel(config_dict["metadata_model"])

        # Convert mrf_edge_approach string to enum if needed
        if "mrf_edge_approach" in config_dict and isinstance(
            config_dict["mrf_edge_approach"], str
        ):
            config_dict["mrf_edge_approach"] = MrfEdgeApproach(
                config_dict["mrf_edge_approach"]
            )

        return cls(**config_dict)

    def to_dict(self):
        result = self.__dict__.copy()
        # Convert enum to string for serialization
        if "root_model" in result:
            result["root_model"] = result["root_model"].value
        if "metadata_model" in result:
            result["metadata_model"] = result["metadata_model"].value
        if "mrf_edge_approach" in result:
            result["mrf_edge_approach"] = result["mrf_edge_approach"].value
        return result
