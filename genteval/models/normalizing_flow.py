import torch
from torch import nn
from torch.distributions import Normal


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
