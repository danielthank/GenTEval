import logging
import time

import numpy as np
import torch
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.distributions import Beta, Categorical, Normal
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from genteval.compressors import CompressedDataset, SerializationFormat


class CouplingLayer(nn.Module):
    """Coupling layer for normalizing flows (RealNVP style)."""

    def __init__(
        self, dim, hidden_dim=64, mask_type="checkerboard", conditioning_dim=0
    ):
        super().__init__()
        self.dim = dim
        self.conditioning_dim = conditioning_dim

        # Create mask
        if mask_type == "checkerboard":
            self.register_buffer("mask", torch.arange(dim) % 2)
        else:  # split
            mask = torch.zeros(dim)
            mask[dim // 2 :] = 1
            self.register_buffer("mask", mask)

        # Input dimension includes masked variables + conditioning
        input_dim = dim + conditioning_dim

        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh(),
        )

        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x, conditioning=None, reverse=False):
        """Forward pass through coupling layer."""
        masked_x = x * self.mask

        # Concatenate masked input with conditioning
        if conditioning is not None and self.conditioning_dim > 0:
            net_input = torch.cat([masked_x, conditioning], dim=-1)
        else:
            net_input = masked_x

        scale = self.scale_net(net_input)
        translate = self.translate_net(net_input)

        if reverse:
            # Inverse transformation
            y = (x - translate * (1 - self.mask)) / torch.exp(scale * (1 - self.mask))
            log_det = -torch.sum(scale * (1 - self.mask), dim=-1)
        else:
            # Forward transformation
            y = x * torch.exp(scale * (1 - self.mask)) + translate * (1 - self.mask)
            log_det = torch.sum(scale * (1 - self.mask), dim=-1)

        return y, log_det


class NormalizingFlow(nn.Module):
    """Normalizing flow for flow-based prior."""

    def __init__(self, dim, num_layers=4, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        # Create coupling layers with alternating masks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            mask_type = "checkerboard" if i % 2 == 0 else "split"
            self.layers.append(CouplingLayer(dim, hidden_dim, mask_type))

    def forward(self, z, reverse=False):
        """Transform samples through the flow."""
        log_det_total = torch.zeros(z.shape[0], device=z.device)

        if reverse:
            # Go through layers in reverse order
            for layer in reversed(self.layers):
                z, log_det = layer(z, reverse=True)
                log_det_total += log_det
        else:
            # Go through layers in forward order
            for layer in self.layers:
                z, log_det = layer(z, reverse=False)
                log_det_total += log_det

        return z, log_det_total

    def log_prob(self, z):
        """Compute log probability of samples under the flow."""
        # Transform to base distribution
        z0, log_det = self.forward(z, reverse=True)

        # Base distribution is standard normal
        base_log_prob = Normal(0, 1).log_prob(z0).sum(dim=-1)

        # Apply change of variables
        return base_log_prob + log_det

    def sample(self, batch_size, device):
        """Sample from the flow-based prior."""
        # Sample from base distribution (standard normal)
        z0 = torch.randn(batch_size, self.dim, device=device)

        # Transform through the flow
        z, _ = self.forward(z0, reverse=False)

        return z


class MetadataVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        use_flow_prior: bool = False,
        prior_flow_layers: int = 4,
        prior_flow_hidden_dim: int = 64,
        num_beta_components: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.embed_dim = self.hidden_dim
        self.use_flow_prior = use_flow_prior
        self.num_beta_components = num_beta_components

        # Embedding for node names
        self.node_embedding = nn.Embedding(vocab_size, self.embed_dim)

        input_dim = (
            3 + 2 * self.embed_dim
        )  # 3 numerical (start_time, duration, child_idx) + 2 embeddings

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Flow-based prior
        if self.use_flow_prior:
            self.flow_prior = NormalizingFlow(
                latent_dim,
                num_layers=prior_flow_layers,
                hidden_dim=prior_flow_hidden_dim,
            )

        # Decoder: mixture Beta likelihood
        # Output: [gap_from_parent_ratio, x_scale, π_1...π_K, α_1...α_K, β_1...β_K]
        decoder_input_dim = latent_dim + input_dim
        beta_output_dim = (
            2 + 3 * num_beta_components
        )  # gap, x_scale, K mixture weights, K alphas, K betas
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, beta_output_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _sample_mixture_beta(
        self,
        mixture_weights: torch.Tensor,
        alphas: torch.Tensor,
        betas: torch.Tensor,
        x_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from mixture of Beta distributions with shared scale.

        Args:
            mixture_weights: (batch_size, K) mixture probabilities
            alphas: (batch_size, K) alpha parameters
            betas: (batch_size, K) beta parameters
            x_scale: (batch_size,) scaling factor

        Returns:
            child_duration_ratio: (batch_size,) sampled values
        """
        batch_size = mixture_weights.shape[0]
        K = mixture_weights.shape[1]

        # Sample component indices from categorical distribution
        component_dist = Categorical(mixture_weights)
        components = component_dist.sample()  # (batch_size,)

        # Gather selected alpha and beta parameters
        selected_alphas = alphas.gather(1, components.unsqueeze(1)).squeeze(
            1
        )  # (batch_size,)
        selected_betas = betas.gather(1, components.unsqueeze(1)).squeeze(
            1
        )  # (batch_size,)

        # Sample from selected Beta distributions
        beta_dists = Beta(selected_alphas, selected_betas)
        beta_samples = beta_dists.sample()  # (batch_size,)

        # Scale by x_scale
        return x_scale * beta_samples

    def _mixture_beta_log_prob(
        self,
        target: torch.Tensor,
        mixture_weights: torch.Tensor,
        alphas: torch.Tensor,
        betas: torch.Tensor,
        x_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood for mixture of Beta distributions.

        Args:
            target: (batch_size,) target values
            mixture_weights: (batch_size, K) mixture probabilities
            alphas: (batch_size, K) alpha parameters
            betas: (batch_size, K) beta parameters
            x_scale: (batch_size,) scaling factor

        Returns:
            Negative log-likelihood sum
        """
        batch_size = target.shape[0]
        K = mixture_weights.shape[1]

        # Unscale target: target / x_scale
        unscaled_target = target / (x_scale + 1e-8)
        unscaled_target_clamped = torch.clamp(unscaled_target, 1e-6, 1 - 1e-6)

        # Compute log-probability for each component
        log_probs = []
        for k in range(K):
            beta_dist = Beta(alphas[:, k], betas[:, k])
            log_prob_k = beta_dist.log_prob(unscaled_target_clamped)
            log_probs.append(log_prob_k)

        log_probs = torch.stack(log_probs, dim=1)  # (batch_size, K)

        # Weighted log-probabilities
        weighted_log_probs = log_probs + torch.log(mixture_weights + 1e-8)

        # Log-sum-exp for mixture
        mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)  # (batch_size,)

        # Add Jacobian for scaling transformation
        jacobian_log_det = torch.log(x_scale + 1e-8)
        total_log_prob = mixture_log_prob + jacobian_log_det

        return -total_log_prob.sum()

    def decode(
        self, z: torch.Tensor, conditioning: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        # Concatenate latent vector with conditioning information
        decoder_input = torch.cat([z, conditioning], dim=1)

        # Beta mixture likelihood
        output = self.decoder(decoder_input)
        K = self.num_beta_components

        # Parse output into mixture components
        gap_from_parent_ratio = torch.sigmoid(output[:, 0])  # [0, 1]
        x_scale = torch.sigmoid(output[:, 1])  # [0, 1] scaling factor

        # Mixture weights: softmax to ensure they sum to 1
        mixture_weights = torch.softmax(output[:, 2 : 2 + K], dim=1)  # (batch_size, K)

        # Alpha and beta parameters: softplus to ensure positive
        alphas = (
            torch.nn.functional.softplus(output[:, 2 + K : 2 + 2 * K]) + 1e-6
        )  # (batch_size, K)
        betas = (
            torch.nn.functional.softplus(output[:, 2 + 2 * K : 2 + 3 * K]) + 1e-6
        )  # (batch_size, K)

        # Sample child_duration_ratio from mixture of scaled Beta distributions
        child_duration_ratio = self._sample_mixture_beta(
            mixture_weights, alphas, betas, x_scale
        )

        return (
            gap_from_parent_ratio,
            x_scale,
            mixture_weights,
            alphas,
            betas,
            child_duration_ratio,
        )

    def forward(
        self,
        parent_start_time: torch.Tensor,
        parent_duration: torch.Tensor,
        child_idx: torch.Tensor,
        parent_node_idx: torch.Tensor,
        child_node_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            parent_start_time: Tensor of shape (batch_size,)
            parent_duration: Tensor of shape (batch_size,)
            child_idx: Tensor of shape (batch_size,) - index of child among siblings (ordered by startTime)
            parent_node_idx: Tensor of shape (batch_size,) - node name indices
            child_node_idx: Tensor of shape (batch_size,) - node name indices

        Returns:
            Tuple of (reconstruction, x_scale, alpha, beta, mu, logvar, z)
            reconstruction contains [gap_from_parent_ratio, child_duration_ratio] in [0,1]
            x_scale is the scaling factor for the beta distribution
            alpha, beta are the beta distribution parameters for child_duration_ratio
            z is the latent sample
        """
        # Get embeddings
        parent_emb = self.node_embedding(parent_node_idx)  # (batch_size, 32)
        child_emb = self.node_embedding(child_node_idx)  # (batch_size, 32)

        # Concatenate all inputs
        x = torch.cat(
            [
                parent_start_time.unsqueeze(1),
                parent_duration.unsqueeze(1),
                child_idx.unsqueeze(1),
                parent_emb,
                child_emb,
            ],
            dim=1,
        )  # (batch_size, 3 + 2 * embed)

        # VAE forward pass
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decode_output = self.decode(z, x)

        (
            gap_from_parent_ratio,
            x_scale,
            mixture_weights,
            alphas,
            betas,
            child_duration_ratio,
        ) = decode_output
        # Combine outputs for reconstruction
        recon = torch.stack([gap_from_parent_ratio, child_duration_ratio], dim=1)
        return recon, x_scale, mixture_weights, alphas, betas, mu, logvar, z

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        x_scale: torch.Tensor,
        mixture_weights: torch.Tensor,
        alphas: torch.Tensor,
        betas: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Split reconstruction and targets
        gap_target = x[:, 0]
        child_duration_target = x[:, 1]
        target_outputs = torch.stack([gap_target, child_duration_target], dim=1)

        # Beta mixture likelihood
        gap_recon = recon_x[:, 0]
        child_duration_recon = recon_x[:, 1]

        # MSE loss for gap_from_parent_ratio
        gap_loss = functional.mse_loss(gap_recon, gap_target, reduction="sum")

        # Mixture Beta distribution negative log-likelihood for child_duration_ratio
        mixture_beta_nll = self._mixture_beta_log_prob(
            child_duration_target, mixture_weights, alphas, betas, x_scale
        )

        # Total reconstruction loss
        recon_loss = gap_loss + mixture_beta_nll

        # KL divergence loss
        if self.use_flow_prior:
            # For flow-based prior: KL[q(z|x) || p_flow(z)]
            # q(z|x) log prob
            posterior_log_prob = (
                Normal(mu, torch.exp(0.5 * logvar)).log_prob(z).sum(dim=-1)
            )
            # p_flow(z) log prob
            prior_log_prob = self.flow_prior.log_prob(z)
            # KL divergence
            kl_loss = (posterior_log_prob - prior_log_prob).sum()
        else:
            # Standard VAE KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + kl_loss * beta
        return total_loss, recon_loss, kl_loss

    def sample(
        self,
        parent_start_time: torch.Tensor,
        parent_duration: torch.Tensor,
        child_idx: torch.Tensor,
        parent_node_idx: torch.Tensor,
        child_node_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate samples by sampling from the prior distribution (no encoder needed).

        Args:
            parent_start_time: Tensor of shape (batch_size,)
            parent_duration: Tensor of shape (batch_size,)
            child_idx: Tensor of shape (batch_size,) - index of child among siblings
            parent_node_idx: Tensor of shape (batch_size,) - node name indices
            child_node_idx: Tensor of shape (batch_size,) - node name indices

        Returns:
            Tensor of shape (batch_size, 2) containing sampled outputs
        """
        batch_size = parent_start_time.shape[0]

        # Get embeddings and create conditioning vector
        parent_emb = self.node_embedding(parent_node_idx)  # (batch_size, 32)
        child_emb = self.node_embedding(child_node_idx)  # (batch_size, 32)

        conditioning = torch.cat(
            [
                parent_start_time.unsqueeze(1),
                parent_duration.unsqueeze(1),
                child_idx.unsqueeze(1),
                parent_emb,
                child_emb,
            ],
            dim=1,
        )  # (batch_size, 67)

        # Sample from prior distribution
        if self.use_flow_prior:
            z = self.flow_prior.sample(batch_size, conditioning.device)
        else:
            z = torch.randn(batch_size, self.latent_dim, device=conditioning.device)

        # Decode with conditioning
        decode_output = self.decode(z, conditioning)

        (
            gap_from_parent_ratio,
            x_scale,
            mixture_weights,
            alphas,
            betas,
            child_duration_ratio,
        ) = decode_output

        # Return combined output
        return torch.stack([gap_from_parent_ratio, child_duration_ratio], dim=1)


class MetadataSynthesizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.node_encoder = LabelEncoder()
        self.model = None
        self.optimizer = None
        self.start_time_scaler = {"mean": 0, "std": 1}
        self.duration_scaler = {"mean": 0, "std": 1}
        self.is_fitted = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def _prepare_training_data(self, traces: list) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data from traces."""
        training_inputs = []
        training_targets = []

        # Collect all node names for encoding
        all_node_names = set()
        for trace in traces:
            for span_data in trace.spans.values():
                node_name = span_data["nodeName"]
                all_node_names.add(node_name)

        # Fit node encoder
        self.logger.info(
            "Fitting node encoder with %d unique node names", len(all_node_names)
        )
        self.node_encoder.fit(list(all_node_names))

        # Collect training data
        all_start_times = []
        all_durations = []

        # First pass: collect all training examples with node names as strings
        raw_training_data = []

        self.logger.info("Collecting training data from traces")
        for trace in tqdm(traces, desc="Processing traces for metadata training"):
            try:
                # Build parent-child relationships and sort children by startTime
                parent_children = {}
                for span_data in trace.spans.values():
                    parent_id = span_data["parentSpanId"]
                    if parent_id and parent_id in trace.spans:
                        if parent_id not in parent_children:
                            parent_children[parent_id] = []
                        parent_children[parent_id].append(span_data)

                # Sort children by startTime for each parent
                for parent_id, children in parent_children.items():
                    children.sort(key=lambda x: x["startTime"])

                    parent_data = trace.spans[parent_id]

                    for child_idx, span_data in enumerate(children):
                        # Get features
                        parent_start_time = parent_data["startTime"]
                        parent_duration = parent_data["duration"]
                        parent_node = parent_data["nodeName"]
                        child_node = span_data["nodeName"]

                        child_start_time = span_data["startTime"]
                        child_duration = span_data["duration"]
                        gap_from_parent = child_start_time - parent_start_time

                        # Skip invalid data
                        if parent_duration <= 0 or child_duration <= 0:
                            continue

                        # Calculate ratios and ensure they're bounded [0, 1]
                        gap_from_parent_ratio = max(
                            0, min(1, gap_from_parent / parent_duration)
                        )
                        child_duration_ratio = max(
                            0, min(1, child_duration / parent_duration)
                        )

                        # Normalize child_idx by total number of children
                        normalized_child_idx = child_idx / len(children)

                        # Store raw data (with string node names and normalized child_idx)
                        raw_training_data.append(
                            {
                                "parent_start_time": parent_start_time,
                                "parent_duration": parent_duration,
                                "normalized_child_idx": normalized_child_idx,
                                "parent_node": parent_node,
                                "child_node": child_node,
                                "gap_from_parent_ratio": gap_from_parent_ratio,
                                "child_duration_ratio": child_duration_ratio,
                            }
                        )

                        # Collect for scaling (only need to scale inputs)
                        all_start_times.append(parent_start_time)
                        all_durations.append(parent_duration)

            except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
                self.logger.warning(f"Error processing trace: {e}")
                continue

        if not raw_training_data:
            raise ValueError("No valid training data found")

        # Second pass: batch transform all node names
        self.logger.info(
            "Encoding node names for %d training examples", len(raw_training_data)
        )
        all_parent_nodes = [item["parent_node"] for item in raw_training_data]
        all_child_nodes = [item["child_node"] for item in raw_training_data]

        parent_node_indices = self.node_encoder.transform(all_parent_nodes)
        child_node_indices = self.node_encoder.transform(all_child_nodes)

        # Third pass: create final training arrays using vectorized operations
        self.logger.info(
            "Creating training tensors for %d examples", len(raw_training_data)
        )

        # Pre-allocate arrays for much faster performance
        num_examples = len(raw_training_data)
        training_inputs = np.zeros((num_examples, 5))  # Added normalized_child_idx
        training_targets = np.zeros((num_examples, 2))

        # Extract all data at once using list comprehensions (vectorized)
        training_inputs[:, 0] = [
            item["parent_start_time"] for item in raw_training_data
        ]
        training_inputs[:, 1] = [item["parent_duration"] for item in raw_training_data]
        training_inputs[:, 2] = [
            item["normalized_child_idx"] for item in raw_training_data
        ]
        training_inputs[:, 3] = parent_node_indices
        training_inputs[:, 4] = child_node_indices

        training_targets[:, 0] = [
            item["gap_from_parent_ratio"] for item in raw_training_data
        ]
        training_targets[:, 1] = [
            item["child_duration_ratio"] for item in raw_training_data
        ]

        # Compute scaling parameters
        self.start_time_scaler = {
            "mean": np.mean(all_start_times),
            "std": np.std(all_start_times) + 1e-8,
        }
        self.duration_scaler = {
            "mean": np.mean(all_durations),
            "std": np.std(all_durations) + 1e-8,
        }

        # Apply scaling to inputs only (targets are already bounded ratios [0,1])
        # Scale inputs (but not child_idx which is already an index)
        training_inputs[:, 0] = (
            training_inputs[:, 0] - self.start_time_scaler["mean"]
        ) / self.start_time_scaler["std"]
        training_inputs[:, 1] = (
            training_inputs[:, 1] - self.duration_scaler["mean"]
        ) / self.duration_scaler["std"]

        # No scaling needed for targets (ratios are already [0,1])

        return training_inputs, training_targets

    def _evaluate_model(self, val_loader, beta=1.0):
        """Evaluate model on validation set and return average loss."""
        self.model.eval()
        total_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                # Move to device
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                # Extract features
                parent_start_time = batch_inputs[:, 0]
                parent_duration = batch_inputs[:, 1]
                normalized_child_idx = batch_inputs[:, 2]
                parent_node_idx = batch_inputs[:, 3].long()
                child_node_idx = batch_inputs[:, 4].long()

                model_output = self.model(
                    parent_start_time,
                    parent_duration,
                    normalized_child_idx,
                    parent_node_idx,
                    child_node_idx,
                )

                # Unpack model output
                recon, x_scale, mixture_weights, alphas, betas, mu, logvar, z = (
                    model_output
                )
                val_loss, _, _ = self.model.loss_function(
                    recon,
                    batch_targets,
                    x_scale,
                    mixture_weights,
                    alphas,
                    betas,
                    mu,
                    logvar,
                    z,
                    beta,
                )

                total_val_loss += val_loss.item()
                num_val_batches += 1

        return total_val_loss / max(num_val_batches, 1)

    def fit(self, traces: list):
        """Train the metadata synthesis model with train/val split and early stopping."""
        self.logger.info("Training Metadata Neural Network")

        # Prepare data
        inputs, targets = self._prepare_training_data(traces)

        # Train/validation split (80/20)
        train_inputs, val_inputs, train_targets, val_targets = train_test_split(
            inputs, targets, test_size=0.2, random_state=42
        )

        self.logger.info(
            f"Training with {len(train_inputs)} examples, validating with {len(val_inputs)} examples"
        )

        # Initialize model
        vocab_size = len(self.node_encoder.classes_)
        self.model = MetadataVAE(
            vocab_size,
            self.config.metadata_hidden_dim,
            self.config.metadata_latent_dim,
            self.config.use_flow_prior,
            self.config.prior_flow_layers,
            self.config.prior_flow_hidden_dim,
            self.config.num_beta_components,
        )
        self.model.to(self.device)  # Move model to GPU
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_inputs), torch.FloatTensor(train_targets)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(val_inputs), torch.FloatTensor(val_targets)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        # Early stopping parameters
        best_val_loss = float("inf")
        patience = self.config.early_stopping_patience
        patience_counter = 0
        best_model_state = None

        # Training loop
        self.model.train()
        pbar = tqdm(range(self.config.metadata_epochs), desc="Training Metadata NN")
        for epoch in pbar:
            # Get current beta value for this epoch
            current_beta = self.config.beta

            self.model.train()
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            num_batches = 0

            for batch_inputs, batch_targets in train_loader:
                # Move to device
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                # Extract features
                parent_start_time = batch_inputs[:, 0]
                parent_duration = batch_inputs[:, 1]
                normalized_child_idx = batch_inputs[:, 2]
                parent_node_idx = batch_inputs[:, 3].long()
                child_node_idx = batch_inputs[:, 4].long()

                self.optimizer.zero_grad()
                model_output = self.model(
                    parent_start_time,
                    parent_duration,
                    normalized_child_idx,
                    parent_node_idx,
                    child_node_idx,
                )

                # Unpack model output
                recon, x_scale, mixture_weights, alphas, betas, mu, logvar, z = (
                    model_output
                )
                total_loss, recon_loss, kl_loss = self.model.loss_function(
                    recon,
                    batch_targets,
                    x_scale,
                    mixture_weights,
                    alphas,
                    betas,
                    mu,
                    logvar,
                    z,
                    current_beta,
                )
                total_loss.backward()
                self.optimizer.step()

                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                num_batches += 1

            # Calculate training losses
            avg_total_loss = epoch_total_loss / max(num_batches, 1)
            avg_recon_loss = epoch_recon_loss / max(num_batches, 1)
            avg_kl_loss = epoch_kl_loss / max(num_batches, 1)

            # Evaluate on validation set
            val_loss = self._evaluate_model(val_loader, current_beta)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # Update progress bar with current losses
            pbar.set_postfix(
                {
                    "Train": f"{avg_total_loss:.4f}",
                    "Val": f"{val_loss:.4f}",
                    "Recon": f"{avg_recon_loss:.4f}",
                    "KL": f"{avg_kl_loss:.4f}",
                    "Beta": f"{current_beta:.4f}",
                    "Patience": f"{patience_counter}/{patience}",
                }
            )

            if wandb.run is not None:
                wandb.log(
                    {
                        "metadata_vae_train_loss": avg_total_loss,
                        "metadata_vae_val_loss": val_loss,
                        "metadata_vae_recon_loss": avg_recon_loss,
                        "metadata_vae_kl_loss": avg_kl_loss,
                        "metadata_vae_beta": current_beta,
                        "metadata_vae_epoch": epoch,
                    }
                )

            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Load best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(
                f"Restored best model with validation loss: {best_val_loss:.4f}"
            )

        self.is_fitted = True

    def synthesize_metadata_batch(
        self,
        parent_start_times: list[float],
        parent_durations: list[float],
        child_indices: list[int],
        parent_nodes: list[str],
        child_nodes: list[str],
    ) -> list[tuple[float, float]]:
        """
        Generate child start_times and durations for multiple parent-child pairs at once.

        Args:
            parent_start_times: List of parent start times
            parent_durations: List of parent durations
            child_indices: List of child indices (0-based index among siblings, ordered by startTime)
            parent_nodes: List of parent node names
            child_nodes: List of child node names

        Returns:
            List of tuples (child_start_time, child_duration).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before synthesis")

        if not (
            len(parent_start_times)
            == len(parent_durations)
            == len(child_indices)
            == len(parent_nodes)
            == len(child_nodes)
        ):
            raise ValueError("All input lists must have the same length")

        if not parent_start_times:
            return []

        self.model.eval()
        with torch.no_grad():
            batch_size = len(parent_start_times)

            # DEBUG: Start timing
            start_time = time.perf_counter()
            self.logger.debug(
                f"[TIMING] Starting synthesize_metadata_batch with batch_size={batch_size}"
            )

            # Encode all node names in batch
            node_encoding_start = time.perf_counter()
            try:
                parent_node_indices = self.node_encoder.transform(parent_nodes)
            except ValueError:
                parent_node_indices = []
                for parent_node in parent_nodes:
                    try:
                        parent_node_idx = self.node_encoder.transform([parent_node])[0]
                    except ValueError:
                        rng = np.random.default_rng()
                        parent_node_idx = rng.integers(
                            0, len(self.node_encoder.classes_)
                        )
                    parent_node_indices.append(parent_node_idx)
                parent_node_indices = np.array(parent_node_indices)

            try:
                child_node_indices = self.node_encoder.transform(child_nodes)
            except ValueError:
                child_node_indices = []
                for child_node in child_nodes:
                    try:
                        child_node_idx = self.node_encoder.transform([child_node])[0]
                    except ValueError:
                        rng = np.random.default_rng()
                        child_node_idx = rng.integers(
                            0, len(self.node_encoder.classes_)
                        )
                    child_node_indices.append(child_node_idx)
                child_node_indices = np.array(child_node_indices)

            node_encoding_time = time.perf_counter() - node_encoding_start
            self.logger.debug(f"[TIMING] Node encoding took: {node_encoding_time:.4f}s")

            # Scale all inputs
            scaling_start = time.perf_counter()
            scaled_start_times = [
                (start_time - self.start_time_scaler["mean"])
                / self.start_time_scaler["std"]
                for start_time in parent_start_times
            ]
            scaled_durations = [
                (duration - self.duration_scaler["mean"]) / self.duration_scaler["std"]
                for duration in parent_durations
            ]

            scaling_time = time.perf_counter() - scaling_start
            self.logger.debug(f"[TIMING] Input scaling took: {scaling_time:.4f}s")

            # Prepare batch tensors and move to device
            tensor_start = time.perf_counter()
            parent_start_tensor = torch.FloatTensor(scaled_start_times).to(self.device)
            parent_duration_tensor = torch.FloatTensor(scaled_durations).to(self.device)
            normalized_child_idx_tensor = torch.FloatTensor(child_indices).to(
                self.device
            )
            parent_node_tensor = torch.LongTensor(parent_node_indices).to(self.device)
            child_node_tensor = torch.LongTensor(child_node_indices).to(self.device)

            tensor_time = time.perf_counter() - tensor_start
            self.logger.debug(
                f"[TIMING] Tensor creation and device transfer took: {tensor_time:.4f}s"
            )

            # Sample using VAE in batch (no encoder needed for generation)
            inference_start = time.perf_counter()
            samples = self.model.sample(
                parent_start_tensor,
                parent_duration_tensor,
                normalized_child_idx_tensor,
                parent_node_tensor,
                child_node_tensor,
            )

            inference_time = time.perf_counter() - inference_start
            self.logger.debug(f"[TIMING] Model inference took: {inference_time:.4f}s")

            # Process batch results
            processing_start = time.perf_counter()
            results = []
            for i in range(batch_size):
                # Get ratio outputs (already bounded [0,1] by sigmoid)
                gap_from_parent_ratio = samples[i, 0].item()
                child_duration_ratio = samples[i, 1].item()

                # Convert ratios back to absolute values
                gap_from_parent = gap_from_parent_ratio * parent_durations[i]
                child_duration = child_duration_ratio * parent_durations[i]

                # Compute child start time
                child_start_time = parent_start_times[i] + max(0, gap_from_parent)
                child_duration = max(1, child_duration)

                results.append((child_start_time, child_duration))

            processing_time = time.perf_counter() - processing_start
            total_time = time.perf_counter() - start_time
            self.logger.debug(
                f"[TIMING] Result processing took: {processing_time:.4f}s"
            )
            self.logger.debug(
                f"[TIMING] Total synthesize_metadata_batch took: {total_time:.4f}s"
            )

            return results

    def save_state_dict(
        self, compressed_data: CompressedDataset, decoder_only: bool = False
    ):
        """Save state dictionary with optional decoder-only mode."""

        # Prepare common data to save

        compressed_data.add(
            "metadata_synthesizer",
            CompressedDataset(
                data={
                    "node_encoder": (
                        self.node_encoder,
                        SerializationFormat.CLOUDPICKLE,
                    ),
                    "start_time_scaler": (
                        self.start_time_scaler,
                        SerializationFormat.MSGPACK,
                    ),
                    "duration_scaler": (
                        self.duration_scaler,
                        SerializationFormat.MSGPACK,
                    ),
                    "is_fitted": (self.is_fitted, SerializationFormat.MSGPACK),
                    "vocab_size": (
                        len(self.node_encoder.classes_)
                        if hasattr(self.node_encoder, "classes_")
                        else 0,
                        SerializationFormat.MSGPACK,
                    ),
                }
            ),
            SerializationFormat.CLOUDPICKLE,
        )

        if decoder_only:
            metadata_vae = {
                k: v
                for k, v in self.model.state_dict().items()
                if k.startswith(
                    ("decoder", "node_embedding")
                )  # Keep embeddings for conditioning
            }
        else:
            metadata_vae = self.model.state_dict()

        compressed_data["metadata_synthesizer"].add(
            "state_dict",
            metadata_vae,
            SerializationFormat.CLOUDPICKLE,
        )

    def load_state_dict(self, compressed_dataset):
        """Load state dictionary."""
        if "metadata_synthesizer" not in compressed_dataset:
            raise ValueError("No metadata_synthesizer found in compressed dataset")

        logger = logging.getLogger(__name__)

        # Load metadata synthesizer data
        metadata_synthesizer_data = compressed_dataset["metadata_synthesizer"]

        self.node_encoder = metadata_synthesizer_data["node_encoder"]
        self.start_time_scaler = metadata_synthesizer_data["start_time_scaler"]
        self.duration_scaler = metadata_synthesizer_data["duration_scaler"]
        self.is_fitted = metadata_synthesizer_data["is_fitted"]
        vocab_size = metadata_synthesizer_data["vocab_size"]

        # Initialize model
        self.model = MetadataVAE(
            vocab_size,
            self.config.metadata_hidden_dim,
            self.config.metadata_latent_dim,
            self.config.use_flow_prior,
            self.config.prior_flow_layers,
            self.config.prior_flow_hidden_dim,
            self.config.num_beta_components,
        )
        self.model.to(self.device)

        # Load model state dict
        if "state_dict" in metadata_synthesizer_data:
            model_state = metadata_synthesizer_data["state_dict"]
            logger.info("Loading metadata synthesizer model")
            self.model.load_state_dict(model_state, strict=False)
        else:
            raise ValueError("No state_dict found in metadata_synthesizer")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
