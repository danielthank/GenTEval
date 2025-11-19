from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Beta, Categorical, Normal
from torch.nn import functional

from .normalizing_flow import NormalizingFlow


@dataclass
class VAEOutput:
    """Output structure for MetadataVAE forward and decode methods."""

    # Final predictions
    gap_ratio: torch.Tensor  # Final gap_from_parent_ratio
    duration_ratio: torch.Tensor  # Final child_duration_ratio
    status_code_logits: torch.Tensor  # Logits for status code classification

    # Latent variables (for training)
    mu: torch.Tensor  # Latent mean
    logvar: torch.Tensor  # Latent log variance
    z: torch.Tensor  # Latent sample

    # Distribution parameters (for loss computation)
    gap_x_scale: torch.Tensor  # Scaling factor for gap Beta distribution
    gap_mixture_weights: torch.Tensor  # Mixture weights for gap Beta
    gap_alphas: torch.Tensor  # Alpha parameters for gap Beta
    gap_betas: torch.Tensor  # Beta parameters for gap Beta
    duration_x_scale: torch.Tensor  # Scaling factor for duration Beta distribution
    duration_mixture_weights: torch.Tensor  # Mixture weights for duration Beta
    duration_alphas: torch.Tensor  # Alpha parameters for duration Beta
    duration_betas: torch.Tensor  # Beta parameters for duration Beta


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

        # Decoder: mixture Beta likelihood for both gap and duration + status code classification
        # Output: [gap_x_scale, gap_π_1...gap_π_K, gap_α_1...gap_α_K, gap_β_1...gap_β_K,
        #          duration_x_scale, duration_π_1...duration_π_K, duration_α_1...duration_α_K, duration_β_1...duration_β_K,
        #          status_code_logits_1...status_code_logits_8]
        decoder_input_dim = latent_dim + input_dim
        beta_output_dim = (
            2 + 6 * num_beta_components + 8
        )  # gap_x_scale + 3*K gap params, duration_x_scale + 3*K duration params, 8 status code logits
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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

        # Beta mixture likelihood + status code
        output = self.decoder(decoder_input)
        K = self.num_beta_components

        # Parse gap Beta mixture components
        gap_x_scale = torch.sigmoid(output[:, 0])  # [0, 1] scaling factor
        gap_mixture_weights = torch.softmax(
            output[:, 1 : 1 + K], dim=1
        )  # (batch_size, K)
        gap_alphas = (
            torch.nn.functional.softplus(output[:, 1 + K : 1 + 2 * K]) + 1e-6
        )  # (batch_size, K)
        gap_betas = (
            torch.nn.functional.softplus(output[:, 1 + 2 * K : 1 + 3 * K]) + 1e-6
        )  # (batch_size, K)

        # Parse duration Beta mixture components
        duration_x_scale = torch.sigmoid(output[:, 1 + 3 * K])  # [0, 1] scaling factor
        duration_mixture_weights = torch.softmax(
            output[:, 2 + 3 * K : 2 + 4 * K], dim=1
        )  # (batch_size, K)
        duration_alphas = (
            torch.nn.functional.softplus(output[:, 2 + 4 * K : 2 + 5 * K]) + 1e-6
        )  # (batch_size, K)
        duration_betas = (
            torch.nn.functional.softplus(output[:, 2 + 5 * K : 2 + 6 * K]) + 1e-6
        )  # (batch_size, K)

        # Status code logits (last 8 outputs)
        status_code_logits = output[:, 2 + 6 * K : 2 + 6 * K + 8]  # (batch_size, 8)

        # Sample both ratios from mixture of scaled Beta distributions
        gap_from_parent_ratio = self._sample_mixture_beta(
            gap_mixture_weights, gap_alphas, gap_betas, gap_x_scale
        )
        child_duration_ratio = self._sample_mixture_beta(
            duration_mixture_weights, duration_alphas, duration_betas, duration_x_scale
        )

        return VAEOutput(
            gap_ratio=gap_from_parent_ratio,
            duration_ratio=child_duration_ratio,
            status_code_logits=status_code_logits,
            mu=mu,
            logvar=logvar,
            z=z,
            gap_x_scale=gap_x_scale,
            gap_mixture_weights=gap_mixture_weights,
            gap_alphas=gap_alphas,
            gap_betas=gap_betas,
            duration_x_scale=duration_x_scale,
            duration_mixture_weights=duration_mixture_weights,
            duration_alphas=duration_alphas,
            duration_betas=duration_betas,
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

        # Mixture Beta distribution negative log-likelihood for gap_from_parent_ratio
        gap_mixture_beta_nll = self._mixture_beta_log_prob(
            gap_target,
            output.gap_mixture_weights,
            output.gap_alphas,
            output.gap_betas,
            output.gap_x_scale,
        )

        # Mixture Beta distribution negative log-likelihood for child_duration_ratio
        duration_mixture_beta_nll = self._mixture_beta_log_prob(
            child_duration_target,
            output.duration_mixture_weights,
            output.duration_alphas,
            output.duration_betas,
            output.duration_x_scale,
        )

        # Cross-entropy loss for status code classification
        status_code_loss = functional.cross_entropy(
            output.status_code_logits, status_code_target, reduction="sum"
        )

        # Total reconstruction loss
        recon_loss = gap_mixture_beta_nll + duration_mixture_beta_nll + status_code_loss

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
