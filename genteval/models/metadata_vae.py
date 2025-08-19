import torch
from torch import nn
from torch.distributions import Beta, Categorical, Normal
from torch.nn import functional

from .normalizing_flow import NormalizingFlow


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

        # Decoder: mixture Beta likelihood
        # Output: [gap_from_parent_ratio, x_scale, π_1...π_K, α_1...α_K, β_1...β_K]
        decoder_input_dim = latent_dim + input_dim
        beta_output_dim = (
            2 + 3 * num_beta_components
        )  # gap, x_scale, K mixture weights, K alphas, K betas
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
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
        parent_duration: torch.Tensor,
        child_idx: torch.Tensor,
        parent_node_idx: torch.Tensor,
        child_node_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
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
            Tensor of shape (batch_size, 2) containing sampled outputs
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
