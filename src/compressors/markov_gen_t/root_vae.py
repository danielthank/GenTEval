import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from compressors import CompressedDataset, SerializationFormat


class RootVAE(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 128, latent_dim: int = 32):
        super(RootVAE, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Embedding for root node names
        self.node_embedding = nn.Embedding(vocab_size, 32)

        # Input: [start_time, node_embedding]
        input_dim = 1 + 32  # 1 numerical + embedding of size 32

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder - takes both latent and conditioning info
        decoder_input_dim = latent_dim + input_dim  # latent + conditioning variables
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output: duration
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

    def decode(self, z: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        # Concatenate latent vector with conditioning information
        decoder_input = torch.cat([z, conditioning], dim=1)
        return self.decoder(decoder_input)

    def forward(
        self,
        start_time: torch.Tensor,
        node_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            start_time: Tensor of shape (batch_size,)
            node_idx: Tensor of shape (batch_size,) - root node name indices

        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        # Get embeddings
        node_emb = self.node_embedding(node_idx)  # (batch_size, 32)

        # Concatenate inputs
        x = torch.cat(
            [
                start_time.unsqueeze(1),
                node_emb,
            ],
            dim=1,
        )  # (batch_size, 33)

        # VAE forward pass
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x)  # Pass conditioning info to decoder
        return recon, mu, logvar

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss

    def sample(
        self,
        start_time: torch.Tensor,
        node_idx: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate samples by sampling from the prior distribution (no encoder needed).

        Args:
            start_time: Tensor of shape (batch_size,)
            node_idx: Tensor of shape (batch_size,) - root node name indices
            num_samples: Number of samples to generate for each input

        Returns:
            Tensor of shape (batch_size, num_samples, 1) containing sampled durations
        """
        batch_size = start_time.shape[0]

        # Get embeddings and create conditioning vector
        node_emb = self.node_embedding(node_idx)  # (batch_size, 32)

        conditioning = torch.cat(
            [
                start_time.unsqueeze(1),
                node_emb,
            ],
            dim=1,
        )  # (batch_size, 33)

        samples = []
        for _ in range(num_samples):
            # Sample from prior distribution (standard normal)
            z = torch.randn(batch_size, self.latent_dim, device=conditioning.device)

            # Decode with conditioning
            sample = self.decode(z, conditioning)
            samples.append(sample)

        # Stack samples: (batch_size, num_samples, 1)
        return torch.stack(samples, dim=1)


class RootDurationSynthesizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize placeholder model (will be properly initialized after fitting)
        self.model = None
        self.node_encoder = LabelEncoder()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Root duration synthesizer using device: {self.device}")

        # Scalers for normalization
        self.start_time_scaler_mean = 0
        self.start_time_scaler_std = 1
        self.duration_scaler_mean = 0
        self.duration_scaler_std = 1

    def fit(self, traces):
        """Train the VAE on root span data."""
        self.logger.info("Training Root Duration VAE")

        # Collect root span data
        root_start_times = []
        root_durations = []
        root_node_names = []

        for trace in traces:
            # Find root spans (spans with no parent)
            root_spans = []
            for span_id, span_data in trace.spans.items():
                if span_data["parentSpanId"] is None:
                    root_spans.append(span_data)

            for root_span in root_spans:
                root_start_times.append(root_span["startTime"])
                root_durations.append(root_span["duration"])
                root_node_names.append(root_span["nodeName"])

        if not root_start_times:
            raise ValueError("No root spans found in traces")

        self.logger.info(f"Found {len(root_start_times)} root spans")

        # Prepare data
        root_start_times = np.array(root_start_times)
        root_durations = np.array(root_durations)
        root_node_names = np.array(root_node_names)

        # Normalize start times
        self.start_time_scaler_mean = np.mean(root_start_times)
        self.start_time_scaler_std = np.std(root_start_times)
        normalized_start_times = (
            root_start_times - self.start_time_scaler_mean
        ) / self.start_time_scaler_std

        # Normalize durations (log transform first to handle skewed distribution)
        log_durations = np.log(root_durations + 1)  # +1 to avoid log(0)
        self.duration_scaler_mean = np.mean(log_durations)
        self.duration_scaler_std = np.std(log_durations)
        normalized_durations = (
            log_durations - self.duration_scaler_mean
        ) / self.duration_scaler_std

        # Encode node names
        self.node_encoder.fit(root_node_names)
        node_indices = self.node_encoder.transform(root_node_names)

        # Initialize model now that we know vocab size
        vocab_size = len(self.node_encoder.classes_)
        self.model = RootVAE(
            vocab_size=vocab_size,
            hidden_dim=self.config.metadata_hidden_dim,  # Reuse metadata config
            latent_dim=self.config.metadata_latent_dim,
        )
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # Convert to tensors
        start_time_tensor = torch.FloatTensor(normalized_start_times).to(self.device)
        duration_tensor = torch.FloatTensor(normalized_durations).to(self.device)
        node_tensor = torch.LongTensor(node_indices).to(self.device)

        # Training loop
        self.model.train()
        dataset_size = len(start_time_tensor)

        pbar = tqdm(
            range(self.config.metadata_epochs), desc="Training Root Duration VAE"
        )  # Reuse metadata epochs
        for epoch in pbar:
            total_loss = 0
            num_batches = 0

            # Create batches
            for i in range(0, dataset_size, self.config.batch_size):
                end_idx = min(i + self.config.batch_size, dataset_size)

                batch_start_times = start_time_tensor[i:end_idx]
                batch_durations = duration_tensor[i:end_idx]
                batch_nodes = node_tensor[i:end_idx]

                self.optimizer.zero_grad()

                # Forward pass
                recon_duration, mu, logvar = self.model(batch_start_times, batch_nodes)

                # Compute loss - ensure shapes match
                recon_duration_flat = recon_duration.view(-1)
                batch_durations_flat = batch_durations.view(-1)
                loss = self.model.loss_function(
                    recon_duration_flat, batch_durations_flat, mu, logvar
                )

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Update progress bar
            avg_loss = total_loss / max(num_batches, 1)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

    def synthesize_root_duration_batch(
        self, start_times: List[float], node_names: List[str], num_samples: int = 1
    ) -> List[float]:
        """Generate root span durations for multiple start times and node names at once.

        Args:
            start_times: List of start times
            node_names: List of node names
            num_samples: Number of samples to generate per input (default: 1)

        Returns:
            List of durations. If num_samples > 1, returns num_samples results
            for each input concatenated in order.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if len(start_times) != len(node_names):
            raise ValueError("start_times and node_names must have the same length")

        if not start_times:
            return []

        self.model.eval()
        with torch.no_grad():
            batch_size = len(start_times)

            # Normalize start times
            normalized_start_times = [
                (start_time - self.start_time_scaler_mean) / self.start_time_scaler_std
                for start_time in start_times
            ]

            # Encode node names
            node_indices = []
            for node_name in node_names:
                if node_name not in self.node_encoder.classes_:
                    # Use a random existing node name if unseen
                    node_name = np.random.choice(self.node_encoder.classes_)
                node_indices.append(self.node_encoder.transform([node_name])[0])

            # Convert to tensors
            start_time_tensor = torch.FloatTensor(normalized_start_times).to(
                self.device
            )
            node_tensor = torch.LongTensor(node_indices).to(self.device)

            # Sample durations in batch (no encoder needed for generation)
            samples = self.model.sample(
                start_time_tensor, node_tensor, num_samples=num_samples
            )

            # Denormalize durations
            results = []
            for i in range(batch_size):
                for j in range(num_samples):
                    normalized_duration = samples[i, j, 0].cpu().numpy()
                    log_duration = (
                        normalized_duration * self.duration_scaler_std
                        + self.duration_scaler_mean
                    )
                    duration = np.exp(log_duration) - 1  # Reverse log transform
                    results.append(max(1, float(duration)))  # Ensure positive duration

            return results

    def save_state_dict(
        self, compressed_data: CompressedDataset, decoder_only: bool = False
    ):
        """Save state dictionary with optional decoder-only mode."""

        compressed_data.add(
            "root_synthesizer",
            CompressedDataset(
                data={
                    "node_encoder": (
                        self.node_encoder,
                        SerializationFormat.CLOUDPICKLE,
                    ),
                    "start_time_scaler_mean": (
                        self.start_time_scaler_mean,
                        SerializationFormat.CLOUDPICKLE,
                    ),
                    "start_time_scaler_std": (
                        self.start_time_scaler_std,
                        SerializationFormat.CLOUDPICKLE,
                    ),
                    "duration_scaler_mean": (
                        self.duration_scaler_mean,
                        SerializationFormat.CLOUDPICKLE,
                    ),
                    "duration_scaler_std": (
                        self.duration_scaler_std,
                        SerializationFormat.CLOUDPICKLE,
                    ),
                    "vocab_size": (
                        len(self.node_encoder.classes_)
                        if hasattr(self.node_encoder, "classes_")
                        else 0,
                        SerializationFormat.CLOUDPICKLE,
                    ),
                }
            ),
            SerializationFormat.CLOUDPICKLE,
        )

        if decoder_only:
            root_vae = {
                k: v
                for k, v in self.model.state_dict().items()
                if k.startswith("decoder")
                or k.startswith("node_embedding")  # Keep embeddings for conditioning
            }
        else:
            root_vae = self.model.state_dict()

        compressed_data["root_synthesizer"].add(
            "state_dict",
            root_vae,
            SerializationFormat.CLOUDPICKLE,
        )

    def load_state_dict(self, compressed_dataset):
        """Load state dictionary."""
        if "root_synthesizer" not in compressed_dataset:
            raise ValueError("No root_synthesizer found in compressed dataset")

        logger = logging.getLogger(__name__)

        # Load root synthesizer data
        root_synthesizer_data = compressed_dataset["root_synthesizer"]

        self.node_encoder = root_synthesizer_data["node_encoder"]
        self.start_time_scaler_mean = root_synthesizer_data["start_time_scaler_mean"]
        self.start_time_scaler_std = root_synthesizer_data["start_time_scaler_std"]
        self.duration_scaler_mean = root_synthesizer_data["duration_scaler_mean"]
        self.duration_scaler_std = root_synthesizer_data["duration_scaler_std"]
        vocab_size = root_synthesizer_data["vocab_size"]

        # Initialize model
        self.model = RootVAE(
            vocab_size,
            self.config.metadata_hidden_dim,
            self.config.metadata_latent_dim,
        )
        self.model.to(self.device)

        # Load model state dict
        if "state_dict" in root_synthesizer_data:
            model_state = root_synthesizer_data["state_dict"]
            logger.info("Loading root synthesizer model")
            self.model.load_state_dict(model_state, strict=False)
        else:
            raise ValueError("No state_dict found in root_synthesizer")
