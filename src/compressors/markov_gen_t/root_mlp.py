import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import wandb

from .. import CompressedDataset, SerializationFormat


class RootMLP(nn.Module):
    def __init__(
        self, vocab_size: int, time_bucket_vocab_size: int, hidden_dim: int = 128
    ):
        super(RootMLP, self).__init__()
        self.vocab_size = vocab_size
        self.time_bucket_vocab_size = time_bucket_vocab_size
        self.hidden_dim = hidden_dim

        # Embedding for root node names
        self.node_embedding = nn.Embedding(vocab_size, 32)

        # Embedding for time buckets (categorical)
        self.time_embedding = nn.Embedding(time_bucket_vocab_size, 16)

        # Input: [time_embedding, node_embedding]
        input_dim = 16 + 32  # time embedding + node embedding

        # Simple MLP network (similar to the decoder in VAE)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Output: E[X] and E[X^2]
        )

    def forward(
        self,
        time_bucket_idx: torch.Tensor,
        node_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            time_bucket_idx: Tensor of shape (batch_size,) - time bucket indices
            node_idx: Tensor of shape (batch_size,) - root node name indices

        Returns:
            Tensor of predicted moments (batch_size, 2) - [E[X], E[X^2]]
        """
        # Get embeddings
        time_emb = self.time_embedding(time_bucket_idx)  # (batch_size, 16)
        node_emb = self.node_embedding(node_idx)  # (batch_size, 32)

        # Concatenate inputs
        x = torch.cat([time_emb, node_emb], dim=1)  # (batch_size, 48)

        # Forward pass through MLP
        moments = self.mlp(x)  # (batch_size, 2) - [E[X], E[X^2]]
        return moments


class RootDurationMLPSynthesizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Time bucketing configuration (hardcoded)
        self.bucket_size_us = 60 * 1000000  # 1 minute in microseconds

        # Initialize placeholder model (will be properly initialized after fitting)
        self.model = None
        self.node_encoder = LabelEncoder()
        self.time_bucket_encoder = LabelEncoder()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Root duration MLP synthesizer using device: {self.device}")

        # Scalers for normalization (only for durations now)
        self.duration_scaler_mean = 0
        self.duration_scaler_std = 1

    def fit(self, traces):
        """Train the MLP on root span data."""
        self.logger.info("Training Root Duration MLP")

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

        # Compute both E[X] and E[X^2] targets
        normalized_durations_squared = normalized_durations**2

        # Encode node names
        self.node_encoder.fit(root_node_names)
        node_indices = self.node_encoder.transform(root_node_names)

        # Initialize model now that we know vocab sizes
        vocab_size = len(self.node_encoder.classes_)
        time_bucket_vocab_size = len(self.time_bucket_encoder.classes_)
        self.model = RootMLP(
            vocab_size=vocab_size,
            time_bucket_vocab_size=time_bucket_vocab_size,
            hidden_dim=self.config.root_hidden_dim,
        )
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # Convert to tensors
        time_bucket_tensor = torch.LongTensor(time_bucket_indices).to(self.device)
        # Stack E[X] and E[X^2] targets
        moments_tensor = torch.FloatTensor(
            np.stack([normalized_durations, normalized_durations_squared], axis=1)
        ).to(self.device)  # (batch_size, 2)
        node_tensor = torch.LongTensor(node_indices).to(self.device)

        # Training loop
        self.model.train()
        dataset_size = len(time_bucket_tensor)

        pbar = tqdm(range(self.config.root_epochs), desc="Training Root Duration MLP")
        for epoch in pbar:
            total_loss = 0
            num_batches = 0

            # Create batches
            for i in range(0, dataset_size, self.config.batch_size):
                end_idx = min(i + self.config.batch_size, dataset_size)

                batch_time_buckets = time_bucket_tensor[i:end_idx]
                batch_moments = moments_tensor[i:end_idx]  # (batch_size, 2)
                batch_nodes = node_tensor[i:end_idx]

                self.optimizer.zero_grad()

                # Forward pass
                predicted_moments = self.model(batch_time_buckets, batch_nodes)

                # Compute loss - Use L2 loss for both E[X] and E[X^2]
                loss = torch.mean((predicted_moments - batch_moments) ** 2)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Update progress bar
            avg_loss = total_loss / max(num_batches, 1)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

            # Log to wandb
            if wandb.run is not None:
                wandb.log({"root_mlp_loss": avg_loss, "root_mlp_epoch": epoch})

    def synthesize_root_duration_batch(
        self, start_times: List[float], node_names: List[str], num_samples: int = 1
    ) -> List[float]:
        """Generate root span durations for multiple start times and node names at once.

        Args:
            start_times: List of start times
            node_names: List of node names
            num_samples: Number of samples to generate per input (default: 1)

        Returns:
            List of durations. If num_samples > 1, samples Gaussian noise around
            the predicted mean with predicted standard deviation.
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

            # Predict moments: E[X] and E[X^2]
            predicted_moments = self.model(time_bucket_tensor, node_tensor)

            # Extract moments and compute variance/std
            results = []
            for i in range(batch_size):
                e_x = predicted_moments[i, 0].cpu().numpy()  # E[X]
                e_x2 = predicted_moments[i, 1].cpu().numpy()  # E[X^2]

                # Compute variance: Var[X] = E[X^2] - (E[X])^2
                variance = e_x2 - (e_x**2)
                # Ensure variance is non-negative (numerical stability)
                variance = max(0, variance)
                std = np.sqrt(variance)

                # Generate samples
                for _ in range(num_samples):
                    normalized_duration = np.random.normal(e_x, std)

                    # Denormalize duration
                    log_duration = (
                        normalized_duration * self.duration_scaler_std
                        + self.duration_scaler_mean
                    )
                    duration = np.exp(log_duration) - 1  # Reverse log transform
                    duration = max(1, float(duration))  # Ensure positive duration

                    results.append(duration)

            return results

    def save_state_dict(
        self, compressed_data: CompressedDataset, decoder_only: bool = False
    ):
        """Save state dictionary."""

        compressed_data.add(
            "root_mlp_synthesizer",
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

        # For MLP, decoder_only doesn't apply since it's already simple
        # But we keep the parameter for interface compatibility
        root_mlp = self.model.state_dict()

        compressed_data["root_mlp_synthesizer"].add(
            "state_dict",
            root_mlp,
            SerializationFormat.CLOUDPICKLE,
        )

    def load_state_dict(self, compressed_dataset):
        """Load state dictionary."""
        if "root_mlp_synthesizer" not in compressed_dataset:
            raise ValueError("No root_mlp_synthesizer found in compressed dataset")

        logger = logging.getLogger(__name__)

        # Load root synthesizer data
        root_synthesizer_data = compressed_dataset["root_mlp_synthesizer"]

        self.node_encoder = root_synthesizer_data["node_encoder"]
        self.time_bucket_encoder = root_synthesizer_data["time_bucket_encoder"]
        self.duration_scaler_mean = root_synthesizer_data["duration_scaler_mean"]
        self.duration_scaler_std = root_synthesizer_data["duration_scaler_std"]

        vocab_size = root_synthesizer_data["vocab_size"]
        time_bucket_vocab_size = root_synthesizer_data["time_bucket_vocab_size"]

        # Initialize model
        self.model = RootMLP(
            vocab_size,
            time_bucket_vocab_size,
            self.config.root_hidden_dim,
        )
        self.model.to(self.device)

        # Load model state dict
        if "state_dict" in root_synthesizer_data:
            model_state = root_synthesizer_data["state_dict"]
            logger.info("Loading root MLP synthesizer model")
            self.model.load_state_dict(model_state, strict=False)
        else:
            raise ValueError("No state_dict found in root_mlp_synthesizer")
