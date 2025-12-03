from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional

from .normalizing_flow import NormalizingFlow


@dataclass
class VAEOutput:
    """Output structure for MetadataVAE forward and decode methods."""

    # Final predictions (direct outputs, no distribution parameters)
    gap_ratio: torch.Tensor  # Final gap_from_parent_ratio in [0, 1]
    duration_ratio: torch.Tensor  # Final child_duration_ratio in [0, 1]
    status_code_logits: torch.Tensor  # Logits for status code classification

    # Latent variables (for training)
    mu: torch.Tensor  # Latent mean
    logvar: torch.Tensor  # Latent log variance
    z: torch.Tensor  # Latent sample


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
        use_focal_loss: bool = False,
        focal_loss_gamma: float = 2.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.embed_dim = self.hidden_dim
        self.use_flow_prior = use_flow_prior
        self.num_beta_components = num_beta_components
        self.use_focal_loss = use_focal_loss
        self.focal_loss_gamma = focal_loss_gamma

        # Embedding for node names (+1 for NO_PARENT token at index vocab_size)
        self.node_embedding = nn.Embedding(vocab_size + 1, self.embed_dim)

        input_dim = (
            2 + 2 * self.embed_dim
        )  # 2 numerical (duration, child_idx) + 2 embeddings

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

        # Decoder: direct prediction of gap_ratio, duration_ratio + status code classification
        # Output: [gap_ratio, duration_ratio, status_code_logits_1...status_code_logits_8]
        decoder_input_dim = latent_dim + input_dim
        output_dim = 2 + 8  # gap_ratio + duration_ratio + 8 status code logits
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
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

    def decode(
        self,
        z: torch.Tensor,
        conditioning: torch.Tensor,
        mu: torch.Tensor = None,
        logvar: torch.Tensor = None,
    ) -> VAEOutput:
        """Decode latent representation to predictions.

        Args:
            z: Latent vector
            conditioning: Conditioning information
            mu: Optional latent mean (for including in output)
            logvar: Optional latent log variance (for including in output)

        Returns:
            VAEOutput with all distribution parameters and predictions
        """
        # Concatenate latent vector with conditioning information
        decoder_input = torch.cat([z, conditioning], dim=1)

        # Get decoder output: [gap_ratio, duration_ratio, 8 status_code_logits]
        output = self.decoder(decoder_input)

        # Parse outputs directly (no Beta mixture)
        gap_ratio = torch.sigmoid(output[:, 0])  # [0, 1]
        duration_ratio = torch.sigmoid(output[:, 1])  # [0, 1]
        status_code_logits = output[:, 2:10]  # (batch_size, 8)

        return VAEOutput(
            gap_ratio=gap_ratio,
            duration_ratio=duration_ratio,
            status_code_logits=status_code_logits,
            mu=mu,
            logvar=logvar,
            z=z,
        )

    def forward(
        self,
        parent_duration: torch.Tensor,
        child_idx: torch.Tensor,
        parent_node_idx: torch.Tensor,
        child_node_idx: torch.Tensor,
    ) -> VAEOutput:
        """
        Args:
            parent_duration: Tensor of shape (batch_size,)
            child_idx: Tensor of shape (batch_size,) - index of child among siblings (ordered by startTime)
            parent_node_idx: Tensor of shape (batch_size,) - node name indices
            child_node_idx: Tensor of shape (batch_size,) - node name indices

        Returns:
            VAEOutput containing all predictions and distribution parameters
        """
        # Apply log transformation to duration to handle large values
        log_duration = torch.log(
            parent_duration + 1e-6
        )  # Add small epsilon to avoid log(0)

        # Get embeddings
        parent_emb = self.node_embedding(parent_node_idx)  # (batch_size, 32)
        child_emb = self.node_embedding(child_node_idx)  # (batch_size, 32)

        # Concatenate all inputs (removed parent_start_time)
        x = torch.cat(
            [
                log_duration.unsqueeze(1),
                child_idx.unsqueeze(1),
                parent_emb,
                child_emb,
            ],
            dim=1,
        )  # (batch_size, 2 + 2 * embed)

        # VAE forward pass
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Decode with mu and logvar included
        return self.decode(z, x, mu=mu, logvar=logvar)

    def focal_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        """Focal loss for multi-class classification.

        Focal loss down-weights easy/well-classified examples and focuses on hard examples.
        FL(p_t) = -(1 - p_t)^gamma * log(p_t)

        Args:
            logits: Predicted logits of shape (batch_size, num_classes)
            targets: Target class indices of shape (batch_size,)
            gamma: Focusing parameter (typically 2.0)

        Returns:
            Sum of focal loss over the batch
        """
        ce_loss = functional.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = (1 - p_t) ** gamma
        return (focal_weight * ce_loss).sum()

    def loss_function(
        self,
        x: torch.Tensor,
        output: VAEOutput,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss given targets and model output.

        Args:
            x: Target tensor with shape (batch_size, 3) containing:
               [gap_ratio, duration_ratio, status_code_idx]
            output: VAEOutput from forward pass
            beta: Weight for KL divergence term

        Returns:
            Tuple of (total_loss, recon_loss, kl_loss)
        """
        # Split targets
        gap_target = x[:, 0]
        child_duration_target = x[:, 1]
        status_code_target = x[:, 2].long()  # Status code indices

        # MSE loss for gap and duration predictions
        gap_loss = functional.mse_loss(output.gap_ratio, gap_target, reduction="sum")
        duration_loss = functional.mse_loss(
            output.duration_ratio, child_duration_target, reduction="sum"
        )

        # Status code classification loss
        if self.use_focal_loss:
            status_code_loss = self.focal_loss(
                output.status_code_logits, status_code_target, self.focal_loss_gamma
            )
        else:
            status_code_loss = functional.cross_entropy(
                output.status_code_logits, status_code_target, reduction="sum"
            )

        # Total reconstruction loss
        recon_loss = gap_loss + duration_loss + status_code_loss

        # KL divergence loss
        if self.use_flow_prior:
            # For flow-based prior: KL[q(z|x) || p_flow(z)]
            # q(z|x) log prob
            posterior_log_prob = (
                Normal(output.mu, torch.exp(0.5 * output.logvar))
                .log_prob(output.z)
                .sum(dim=-1)
            )
            # p_flow(z) log prob
            prior_log_prob = self.flow_prior.log_prob(output.z)
            # KL divergence
            kl_loss = (posterior_log_prob - prior_log_prob).sum()
        else:
            # Standard VAE KL divergence
            kl_loss = -0.5 * torch.sum(
                1 + output.logvar - output.mu.pow(2) - output.logvar.exp()
            )

        total_loss = recon_loss + kl_loss * beta
        return total_loss, recon_loss, kl_loss

    def sample(
        self,
        parent_duration: torch.Tensor,
        child_idx: torch.Tensor,
        parent_node_idx: torch.Tensor,
        child_node_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate samples by sampling from the prior distribution (no encoder needed).

        Args:
            parent_duration: Tensor of shape (batch_size,)
            child_idx: Tensor of shape (batch_size,) - index of child among siblings
            parent_node_idx: Tensor of shape (batch_size,) - node name indices
            child_node_idx: Tensor of shape (batch_size,) - node name indices

        Returns:
            Tensor of shape (batch_size, 3) containing [gap_ratio, duration_ratio, status_code_idx]
        """
        batch_size = parent_duration.shape[0]

        # Apply log transformation to duration to handle large values (same as in forward)
        log_duration = torch.log(parent_duration + 1e-6)

        # Get embeddings and create conditioning vector
        parent_emb = self.node_embedding(parent_node_idx)  # (batch_size, 32)
        child_emb = self.node_embedding(child_node_idx)  # (batch_size, 32)

        conditioning = torch.cat(
            [
                log_duration.unsqueeze(1),
                child_idx.unsqueeze(1),
                parent_emb,
                child_emb,
            ],
            dim=1,
        )  # (batch_size, 2 + 2 * embed)

        # Sample from prior distribution
        if self.use_flow_prior:
            z = self.flow_prior.sample(batch_size, conditioning.device)
        else:
            z = torch.randn(batch_size, self.latent_dim, device=conditioning.device)

        # Decode with conditioning (no mu/logvar needed for sampling)
        output = self.decode(z, conditioning)

        # Sample status code from categorical distribution
        status_code_probs = functional.softmax(output.status_code_logits, dim=1)
        status_code_dist = Categorical(status_code_probs)
        status_code_idx = status_code_dist.sample()  # (batch_size,)

        # Return combined output: [gap_ratio, duration_ratio, status_code_idx]
        return torch.stack(
            [output.gap_ratio, output.duration_ratio, status_code_idx.float()],
            dim=1,
        )
