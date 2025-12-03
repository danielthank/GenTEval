from dataclasses import dataclass
from enum import Enum


class TopologyModelType(Enum):
    """Topology model variants."""

    MRF = "MRF"  # Markov Random Field approach


class RootDurationModelType(Enum):
    """Root duration model variants."""

    GMM = "GMM"  # Gaussian Mixture Model


class MetadataVAEModelType(Enum):
    """MetadataVAE model variants."""

    METADATA_VAE = "METADATA_VAE"  # Neural network VAE for joint gap/duration modeling


@dataclass
class SimpleGenTConfig:
    """Configuration for Simple GenT algorithm."""

    # Time bucketing configuration
    time_bucket_duration_us: int = 1 * 60 * 1000000

    # Model selection
    topology_model: TopologyModelType = TopologyModelType.MRF
    root_duration_model: RootDurationModelType = RootDurationModelType.GMM
    metadata_vae_model: MetadataVAEModelType = MetadataVAEModelType.METADATA_VAE

    # Topology model parameters
    max_depth: int = 100
    max_children: int = 5000

    # GMM parameters for root duration
    max_gmm_components: int = 3
    min_samples_for_gmm: int = 2

    # MetadataVAE parameters
    metadata_hidden_dim: int = 16
    metadata_latent_dim: int = 8
    use_flow_prior: bool = False
    prior_flow_layers: int = 1
    prior_flow_hidden_dim: int = 16
    num_beta_components: int = 5
    learning_rate: float = 1e-3
    metadata_epochs: int = 500
    batch_size: int = 1024
    beta: float = 10.0  # KL divergence weight
    early_stopping_patience: int = 10
    sequential_training_lr_factor: float = 0.5  # LR reduction for sequential training

    # Focal loss for status code classification (experimental)
    use_focal_loss: bool = True
    focal_loss_gamma: float = 2.0

    # Reject sampling parameters
    reject_sampling_max_attempts: int = 10
    reject_sampling_enabled: bool = True

    # Stratified sampling (separate normal/error trace generation)
    stratified_sampling: bool = False

    # General parameters
    random_seed: int = 42

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SimpleGenTConfig":
        """Create config from dictionary, handling enum conversions."""
        config_dict = config_dict.copy()

        # Convert string enums back to enum objects
        if "topology_model" in config_dict and isinstance(
            config_dict["topology_model"], str
        ):
            config_dict["topology_model"] = TopologyModelType(
                config_dict["topology_model"]
            )

        if "root_duration_model" in config_dict and isinstance(
            config_dict["root_duration_model"], str
        ):
            config_dict["root_duration_model"] = RootDurationModelType(
                config_dict["root_duration_model"]
            )

        if "metadata_vae_model" in config_dict and isinstance(
            config_dict["metadata_vae_model"], str
        ):
            config_dict["metadata_vae_model"] = MetadataVAEModelType(
                config_dict["metadata_vae_model"]
            )

        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary, handling enum serialization."""
        result = self.__dict__.copy()

        # Convert enums to strings for serialization
        if "topology_model" in result:
            result["topology_model"] = result["topology_model"].value
        if "root_duration_model" in result:
            result["root_duration_model"] = result["root_duration_model"].value
        if "metadata_vae_model" in result:
            result["metadata_vae_model"] = result["metadata_vae_model"].value

        return result
