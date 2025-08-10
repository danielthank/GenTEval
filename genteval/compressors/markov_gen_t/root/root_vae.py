import logging

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.preprocessing import LabelEncoder
from torch import nn
from tqdm import tqdm

from genteval.compressors import CompressedDataset, SerializationFormat


class RootVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        time_bucket_vocab_size: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
    ):
        super(RootVAE, self).__init__()
        self.vocab_size = vocab_size
        self.time_bucket_vocab_size = time_bucket_vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Embedding for root node names
        self.node_embedding = nn.Embedding(vocab_size, 32)

        # Embedding for time buckets (categorical)
        self.time_embedding = nn.Embedding(time_bucket_vocab_size, 16)

        # Input: [time_embedding, node_embedding]
        input_dim = 16 + 32  # time embedding + node embedding

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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        time_bucket_idx: torch.Tensor,
        node_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            time_bucket_idx: Tensor of shape (batch_size,) - time bucket indices
            node_idx: Tensor of shape (batch_size,) - root node name indices

        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        # Get embeddings
        time_emb = self.time_embedding(time_bucket_idx)  # (batch_size, 16)
        node_emb = self.node_embedding(node_idx)  # (batch_size, 32)
        time_emb = time_emb.view(time_emb.size(0), -1)  # Ensure correct shape

        # Concatenate inputs
        x = torch.cat([time_emb, node_emb], dim=1)  # (batch_size, 48)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss

    def sample(
        self,
        time_bucket_idx: torch.Tensor,
        node_idx: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate samples by sampling from the prior distribution (no encoder needed).

        Args:
            time_bucket_idx: Tensor of shape (batch_size,) - time bucket indices
            node_idx: Tensor of shape (batch_size,) - root node name indices
            num_samples: Number of samples to generate for each input

        Returns:
            Tensor of shape (batch_size, num_samples, 1) containing sampled durations
        """
        batch_size = time_bucket_idx.shape[0]

        # Get embeddings and create conditioning vector
        time_emb = self.time_embedding(time_bucket_idx)  # (batch_size, 16)
        node_emb = self.node_embedding(node_idx)  # (batch_size, 32)

        conditioning = torch.cat([time_emb, node_emb], dim=1)  # (batch_size, 48)

        samples = []
        for _ in range(num_samples):
            # Sample from prior distribution (standard normal)
            z = torch.randn(batch_size, self.latent_dim, device=conditioning.device)

            # Decode with conditioning
            sample = self.decode(z, conditioning)
            samples.append(sample)

        # Stack samples: (batch_size, num_samples, 1)
        return torch.stack(samples, dim=1)


class RootDurationVAESynthesizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Time bucketing configuration (hardcoded, same as root_mlp)
        self.bucket_size_us = 60 * 1000000  # 1 minute in microseconds

        # Initialize placeholder model (will be properly initialized after fitting)
        self.model = None
        self.node_encoder = LabelEncoder()
        self.time_bucket_encoder = LabelEncoder()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Root duration synthesizer using device: {self.device}")

        # Scalers for normalization (only for durations now)
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

        # Convert start times to time buckets (categorical, not normalized)
        time_buckets = root_start_times // self.bucket_size_us

        # Encode time buckets as categories
        self.time_bucket_encoder.fit(time_buckets)
        time_bucket_indices = self.time_bucket_encoder.transform(time_buckets)

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

        # Initialize model now that we know vocab sizes
        vocab_size = len(self.node_encoder.classes_)
        time_bucket_vocab_size = len(self.time_bucket_encoder.classes_)
        self.model = RootVAE(
            vocab_size=vocab_size,
            time_bucket_vocab_size=time_bucket_vocab_size,
            hidden_dim=self.config.root_hidden_dim,
            latent_dim=self.config.root_latent_dim,
        )
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # Convert to tensors
        time_bucket_tensor = torch.LongTensor(time_bucket_indices).to(self.device)
        duration_tensor = torch.FloatTensor(normalized_durations).to(self.device)
        node_tensor = torch.LongTensor(node_indices).to(self.device)

        # Training loop
        self.model.train()
        dataset_size = len(time_bucket_tensor)

        pbar = tqdm(range(self.config.root_epochs), desc="Training Root Duration VAE")
        for epoch in pbar:
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            num_batches = 0

            # Create batches
            for i in range(0, dataset_size, self.config.batch_size):
                end_idx = min(i + self.config.batch_size, dataset_size)

                batch_time_buckets = time_bucket_tensor[i:end_idx]
                batch_durations = duration_tensor[i:end_idx]
                batch_nodes = node_tensor[i:end_idx]

                self.optimizer.zero_grad()

                # Forward pass
                recon_duration, mu, logvar = self.model(batch_time_buckets, batch_nodes)

                # Compute loss - ensure shapes match
                recon_duration_flat = recon_duration.view(-1)
                batch_durations_flat = batch_durations.view(-1)
                total_loss, recon_loss, kl_loss = self.model.loss_function(
                    recon_duration_flat, batch_durations_flat, mu, logvar
                )

                total_loss.backward()
                self.optimizer.step()

                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                num_batches += 1

            # Update progress bar
            avg_total_loss = epoch_total_loss / max(num_batches, 1)
            avg_recon_loss = epoch_recon_loss / max(num_batches, 1)
            avg_kl_loss = epoch_kl_loss / max(num_batches, 1)
            pbar.set_postfix(
                {
                    "Total": f"{avg_total_loss:.4f}",
                    "Recon": f"{avg_recon_loss:.4f}",
                    "KL": f"{avg_kl_loss:.4f}",
                }
            )

            # Log to wandb
            if wandb.run is not None:
                wandb.log(
                    {
                        "root_vae_total_loss": avg_total_loss,
                        "root_vae_recon_loss": avg_recon_loss,
                        "root_vae_kl_loss": avg_kl_loss,
                        "root_vae_epoch": epoch,
                    }
                )

    def synthesize_root_duration_batch(
        self, start_times: list[float], node_names: list[str], num_samples: int = 1
    ) -> list[float]:
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

            # Convert start times to time buckets (categorical, consistent with training)
            time_buckets = [
                start_time // self.bucket_size_us for start_time in start_times
            ]

            # Encode time buckets as categories
            time_bucket_indices = []
            for bucket in time_buckets:
                if bucket in self.time_bucket_encoder.classes_:
                    time_bucket_indices.append(
                        self.time_bucket_encoder.transform([bucket])[0]
                    )
                else:
                    # Use a random existing time bucket if unseen
                    random_bucket = np.random.choice(self.time_bucket_encoder.classes_)
                    time_bucket_indices.append(
                        self.time_bucket_encoder.transform([random_bucket])[0]
                    )

            # Encode node names
            node_indices = []
            for node_name in node_names:
                if node_name not in self.node_encoder.classes_:
                    # Use a random existing node name if unseen
                    node_name = np.random.choice(self.node_encoder.classes_)
                node_indices.append(self.node_encoder.transform([node_name])[0])

            # Convert to tensors
            time_bucket_tensor = torch.LongTensor(time_bucket_indices).to(self.device)
            node_tensor = torch.LongTensor(node_indices).to(self.device)

            # Sample durations in batch (no encoder needed for generation)
            samples = self.model.sample(
                time_bucket_tensor, node_tensor, num_samples=num_samples
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
                    "time_bucket_encoder": (
                        self.time_bucket_encoder,
                        SerializationFormat.CLOUDPICKLE,
                    ),
                    "duration_scaler_mean": (
                        self.duration_scaler_mean,
                        SerializationFormat.MSGPACK,
                    ),
                    "duration_scaler_std": (
                        self.duration_scaler_std,
                        SerializationFormat.MSGPACK,
                    ),
                    "vocab_size": (
                        len(self.node_encoder.classes_)
                        if hasattr(self.node_encoder, "classes_")
                        else 0,
                        SerializationFormat.MSGPACK,
                    ),
                    "time_bucket_vocab_size": (
                        len(self.time_bucket_encoder.classes_)
                        if hasattr(self.time_bucket_encoder, "classes_")
                        else 0,
                        SerializationFormat.MSGPACK,
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
                or k.startswith("node_embedding")
                or k.startswith("time_embedding")
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
        self.time_bucket_encoder = root_synthesizer_data["time_bucket_encoder"]
        self.duration_scaler_mean = root_synthesizer_data["duration_scaler_mean"]
        self.duration_scaler_std = root_synthesizer_data["duration_scaler_std"]
        vocab_size = root_synthesizer_data["vocab_size"]
        time_bucket_vocab_size = root_synthesizer_data["time_bucket_vocab_size"]

        # Initialize model
        self.model = RootVAE(
            vocab_size,
            time_bucket_vocab_size,
            self.config.root_hidden_dim,
            self.config.root_latent_dim,
        )
        self.model.to(self.device)

        # Load model state dict
        if "state_dict" in root_synthesizer_data:
            model_state = root_synthesizer_data["state_dict"]
            logger.info("Loading root synthesizer model")
            self.model.load_state_dict(model_state, strict=False)
        else:
            raise ValueError("No state_dict found in root_synthesizer")
