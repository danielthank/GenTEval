import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from compressors import CompressedDataset, SerializationFormat


class StartTimeVAE(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64):
        super(StartTimeVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss

    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample start times from the learned distribution."""
        with torch.no_grad():
            # Get device from model parameters
            device = next(self.parameters()).device
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
            return samples


class StartTimeSynthesizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = StartTimeVAE(
            latent_dim=config.start_time_latent_dim,
            hidden_dim=config.start_time_hidden_dim,
        )

        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.model.to(self.device)  # Move model to GPU

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.scaler_mean = 0
        self.scaler_std = 1

    def fit(self, start_times: np.ndarray):
        """Train the VAE on start times."""
        self.logger.info("Training StartTime VAE")

        # Normalize start times
        self.scaler_mean = np.mean(start_times)
        self.scaler_std = np.std(start_times)
        normalized_times = (start_times - self.scaler_mean) / self.scaler_std

        # Convert to tensor and move to device
        data = torch.FloatTensor(normalized_times.reshape(-1, 1)).to(self.device)

        # Training loop
        self.model.train()
        pbar = tqdm(range(self.config.start_time_epochs), desc="Training StartTime VAE")
        for epoch in pbar:
            total_loss = 0
            num_batches = 0

            for i in range(0, len(data), self.config.batch_size):
                batch = data[i : i + self.config.batch_size]

                self.optimizer.zero_grad()
                recon, mu, logvar = self.model(batch)

                # Ensure consistent shapes for loss computation
                recon_flat = recon.view(-1)
                batch_flat = batch.view(-1)
                loss = self.model.loss_function(recon_flat, batch_flat, mu, logvar)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Update progress bar with current loss
            avg_loss = total_loss / max(num_batches, 1)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

    def save_state_dict(
        self, compressed_data: CompressedDataset, decoder_only: bool = False
    ):
        """Save state dictionary with optional decoder-only mode."""

        compressed_data.add(
            "start_time_synthesizer",
            CompressedDataset(
                data={
                    "scaler_mean": (
                        self.scaler_mean,
                        SerializationFormat.CLOUDPICKLE,
                    ),
                    "scaler_std": (
                        self.scaler_std,
                        SerializationFormat.CLOUDPICKLE,
                    ),
                }
            ),
            SerializationFormat.CLOUDPICKLE,
        )

        if decoder_only:
            start_time_vae = {
                k: v
                for k, v in self.model.state_dict().items()
                if k.startswith("decoder")
            }
        else:
            start_time_vae = self.model.state_dict()

        compressed_data["start_time_synthesizer"].add(
            "state_dict",
            start_time_vae,
            SerializationFormat.CLOUDPICKLE,
        )

    def load_state_dict(self, compressed_dataset):
        """Load state dictionary."""
        if "start_time_synthesizer" not in compressed_dataset:
            raise ValueError("No start_time_synthesizer found in compressed dataset")

        logger = logging.getLogger(__name__)

        # Load start time synthesizer data
        start_time_synthesizer_data = compressed_dataset["start_time_synthesizer"]

        self.scaler_mean = start_time_synthesizer_data["scaler_mean"]
        self.scaler_std = start_time_synthesizer_data["scaler_std"]

        # Load model state dict
        if "state_dict" in start_time_synthesizer_data:
            model_state = start_time_synthesizer_data["state_dict"]
            logger.info("Loading start time synthesizer model")
            self.model.load_state_dict(model_state, strict=False)
        else:
            raise ValueError("No state_dict found in start_time_synthesizer")

    def sample(self, num_samples: int) -> np.ndarray:
        """Generate new start times."""
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(num_samples)
            # Move to CPU for numpy conversion and denormalize
            samples = samples.cpu().numpy() * self.scaler_std + self.scaler_mean
            return samples.flatten()
