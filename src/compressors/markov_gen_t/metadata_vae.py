import logging
import pickle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class MetadataVAE(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 128, latent_dim: int = 32):
        super(MetadataVAE, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Embedding for node names
        self.node_embedding = nn.Embedding(vocab_size, 32)

        # Input: [parent_start_time, parent_duration, parent_embedding, child_embedding]
        input_dim = 2 + 32 + 32  # 2 numerical + 2 embeddings of size 32

        # Encoder (single layer)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder (single layer) - takes both latent and conditioning info
        decoder_input_dim = latent_dim + input_dim  # latent + conditioning variables
        self.decoder = nn.Sequential(
            nn.Linear(
                decoder_input_dim, 2
            ),  # Output: [gap_from_parent_ratio, child_duration_ratio]
            nn.Sigmoid(),  # Bound outputs to [0, 1]
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
        parent_start_time: torch.Tensor,
        parent_duration: torch.Tensor,
        parent_node_idx: torch.Tensor,
        child_node_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            parent_start_time: Tensor of shape (batch_size,)
            parent_duration: Tensor of shape (batch_size,)
            parent_node_idx: Tensor of shape (batch_size,) - node name indices
            child_node_idx: Tensor of shape (batch_size,) - node name indices

        Returns:
            Tuple of (reconstruction, mu, logvar)
            reconstruction contains [gap_from_parent_ratio, child_duration_ratio] in [0,1]
        """
        # Get embeddings
        parent_emb = self.node_embedding(parent_node_idx)  # (batch_size, 32)
        child_emb = self.node_embedding(child_node_idx)  # (batch_size, 32)

        # Concatenate all inputs
        x = torch.cat(
            [
                parent_start_time.unsqueeze(1),
                parent_duration.unsqueeze(1),
                parent_emb,
                child_emb,
            ],
            dim=1,
        )  # (batch_size, 66)

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
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss

    def sample(
        self,
        parent_start_time: torch.Tensor,
        parent_duration: torch.Tensor,
        parent_node_idx: torch.Tensor,
        child_node_idx: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate samples by sampling from the prior distribution (no encoder needed).

        Args:
            parent_start_time: Tensor of shape (batch_size,)
            parent_duration: Tensor of shape (batch_size,)
            parent_node_idx: Tensor of shape (batch_size,) - node name indices
            child_node_idx: Tensor of shape (batch_size,) - node name indices
            num_samples: Number of samples to generate for each input

        Returns:
            Tensor of shape (batch_size, num_samples, 2) containing sampled outputs
        """
        batch_size = parent_start_time.shape[0]

        # Get embeddings and create conditioning vector
        parent_emb = self.node_embedding(parent_node_idx)  # (batch_size, 32)
        child_emb = self.node_embedding(child_node_idx)  # (batch_size, 32)

        conditioning = torch.cat(
            [
                parent_start_time.unsqueeze(1),
                parent_duration.unsqueeze(1),
                parent_emb,
                child_emb,
            ],
            dim=1,
        )  # (batch_size, 66)

        samples = []
        for _ in range(num_samples):
            # Sample from prior distribution (standard normal)
            z = torch.randn(batch_size, self.latent_dim, device=conditioning.device)

            # Decode with conditioning
            sample = self.decode(z, conditioning)
            samples.append(sample)

        # Stack samples: (batch_size, num_samples, 2)
        return torch.stack(samples, dim=1)


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

    def _prepare_training_data(self, traces: List) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from traces."""
        training_inputs = []
        training_targets = []

        # Collect all node names for encoding
        all_node_names = set()
        for trace in traces:
            for span_id, span_data in trace.spans.items():
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
                # Build parent-child relationships
                for span_id, span_data in trace.spans.items():
                    parent_id = span_data["parentSpanId"]
                    if parent_id and parent_id in trace.spans:
                        parent_data = trace.spans[parent_id]

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

                        # Store raw data (with string node names)
                        raw_training_data.append(
                            {
                                "parent_start_time": parent_start_time,
                                "parent_duration": parent_duration,
                                "parent_node": parent_node,
                                "child_node": child_node,
                                "gap_from_parent_ratio": gap_from_parent_ratio,
                                "child_duration_ratio": child_duration_ratio,
                            }
                        )

                        # Collect for scaling (only need to scale inputs)
                        all_start_times.append(parent_start_time)
                        all_durations.append(parent_duration)

            except Exception as e:
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
        training_inputs = np.zeros((num_examples, 4))
        training_targets = np.zeros((num_examples, 2))

        # Extract all data at once using list comprehensions (vectorized)
        training_inputs[:, 0] = [
            item["parent_start_time"] for item in raw_training_data
        ]
        training_inputs[:, 1] = [item["parent_duration"] for item in raw_training_data]
        training_inputs[:, 2] = parent_node_indices
        training_inputs[:, 3] = child_node_indices

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
        # Scale inputs
        training_inputs[:, 0] = (
            training_inputs[:, 0] - self.start_time_scaler["mean"]
        ) / self.start_time_scaler["std"]
        training_inputs[:, 1] = (
            training_inputs[:, 1] - self.duration_scaler["mean"]
        ) / self.duration_scaler["std"]

        # No scaling needed for targets (ratios are already [0,1])

        return training_inputs, training_targets

    def fit(self, traces: List):
        """Train the metadata synthesis model."""
        self.logger.info("Training Metadata Neural Network")

        # Prepare data
        inputs, targets = self._prepare_training_data(traces)

        # Initialize model
        vocab_size = len(self.node_encoder.classes_)
        self.model = MetadataVAE(
            vocab_size, self.config.metadata_hidden_dim, self.config.metadata_latent_dim
        )
        self.model.to(self.device)  # Move model to GPU
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # Convert to tensors and move to device
        inputs_tensor = torch.FloatTensor(inputs).to(self.device)
        targets_tensor = torch.FloatTensor(targets).to(self.device)

        # Training loop
        self.logger.info("Training with %d examples", len(inputs_tensor))
        self.model.train()
        pbar = tqdm(range(self.config.metadata_epochs), desc="Training Metadata NN")
        for epoch in pbar:
            total_loss = 0
            num_batches = 0

            for i in range(0, len(inputs_tensor), self.config.batch_size):
                batch_inputs = inputs_tensor[i : i + self.config.batch_size]
                batch_targets = targets_tensor[i : i + self.config.batch_size]

                if len(batch_inputs) == 0:
                    continue

                # Extract features
                parent_start_time = batch_inputs[:, 0]
                parent_duration = batch_inputs[:, 1]
                parent_node_idx = batch_inputs[:, 2].long()
                child_node_idx = batch_inputs[:, 3].long()

                self.optimizer.zero_grad()
                recon, mu, logvar = self.model(
                    parent_start_time, parent_duration, parent_node_idx, child_node_idx
                )

                # Ensure consistent shapes for loss computation
                recon_flat = recon.view(-1)
                batch_targets_flat = batch_targets.view(-1)
                loss = self.model.loss_function(
                    recon_flat, batch_targets_flat, mu, logvar
                )
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Update progress bar with current loss
            avg_loss = total_loss / max(num_batches, 1)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

        self.is_fitted = True

    def synthesize_metadata_batch(
        self,
        parent_start_times: List[float],
        parent_durations: List[float],
        parent_nodes: List[str],
        child_nodes: List[str],
        num_samples: int = 1,
    ) -> List[Tuple[float, float]]:
        """
        Generate child start_times and durations for multiple parent-child pairs at once.

        Args:
            parent_start_times: List of parent start times
            parent_durations: List of parent durations
            parent_nodes: List of parent node names
            child_nodes: List of child node names
            num_samples: Number of samples to generate per input (default: 1)

        Returns:
            List of tuples (child_start_time, child_duration). If num_samples > 1,
            returns num_samples results for each input concatenated in order.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before synthesis")

        if not (
            len(parent_start_times)
            == len(parent_durations)
            == len(parent_nodes)
            == len(child_nodes)
        ):
            raise ValueError("All input lists must have the same length")

        if not parent_start_times:
            return []

        self.model.eval()
        with torch.no_grad():
            batch_size = len(parent_start_times)

            # Encode all node names
            parent_node_indices = []
            child_node_indices = []

            for parent_node, child_node in zip(parent_nodes, child_nodes):
                try:
                    parent_node_idx = self.node_encoder.transform([parent_node])[0]
                except ValueError:
                    parent_node_idx = np.random.randint(
                        0, len(self.node_encoder.classes_)
                    )

                try:
                    child_node_idx = self.node_encoder.transform([child_node])[0]
                except ValueError:
                    child_node_idx = np.random.randint(
                        0, len(self.node_encoder.classes_)
                    )

                parent_node_indices.append(parent_node_idx)
                child_node_indices.append(child_node_idx)

            # Scale all inputs
            scaled_start_times = [
                (start_time - self.start_time_scaler["mean"])
                / self.start_time_scaler["std"]
                for start_time in parent_start_times
            ]
            scaled_durations = [
                (duration - self.duration_scaler["mean"]) / self.duration_scaler["std"]
                for duration in parent_durations
            ]

            # Prepare batch tensors and move to device
            parent_start_tensor = torch.FloatTensor(scaled_start_times).to(self.device)
            parent_duration_tensor = torch.FloatTensor(scaled_durations).to(self.device)
            parent_node_tensor = torch.LongTensor(parent_node_indices).to(self.device)
            child_node_tensor = torch.LongTensor(child_node_indices).to(self.device)

            # Sample using VAE in batch (no encoder needed for generation)
            samples = self.model.sample(
                parent_start_tensor,
                parent_duration_tensor,
                parent_node_tensor,
                child_node_tensor,
                num_samples=num_samples,
            )

            # Process batch results
            results = []
            for i in range(batch_size):
                for j in range(num_samples):
                    # Get ratio outputs (already bounded [0,1] by sigmoid)
                    gap_from_parent_ratio = samples[i, j, 0].item()
                    child_duration_ratio = samples[i, j, 1].item()

                    # Convert ratios back to absolute values
                    gap_from_parent = gap_from_parent_ratio * parent_durations[i]
                    child_duration = child_duration_ratio * parent_durations[i]

                    # Compute child start time
                    child_start_time = parent_start_times[i] + max(0, gap_from_parent)
                    child_duration = max(1, child_duration)

                    results.append((child_start_time, child_duration))

            return results

    def get_state_dict(self):
        """Get state dictionary for serialization."""
        return {
            "model_state": self.model.state_dict() if self.model else None,
            "node_encoder": pickle.dumps(self.node_encoder),
            "start_time_scaler": self.start_time_scaler,
            "duration_scaler": self.duration_scaler,
            "is_fitted": self.is_fitted,
            "vocab_size": len(self.node_encoder.classes_)
            if hasattr(self.node_encoder, "classes_")
            else 0,
        }

    def load_state_dict(self, state_dict):
        """Load state dictionary from serialization."""
        self.node_encoder = pickle.loads(state_dict["node_encoder"])
        self.start_time_scaler = state_dict["start_time_scaler"]
        self.duration_scaler = state_dict["duration_scaler"]
        self.is_fitted = state_dict["is_fitted"]

        if state_dict["model_state"] and state_dict["vocab_size"] > 0:
            self.model = MetadataVAE(
                state_dict["vocab_size"],
                self.config.metadata_hidden_dim,
                self.config.metadata_latent_dim,
            )
            self.model.load_state_dict(state_dict["model_state"])
            self.model.to(self.device)  # Move loaded model to device
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.learning_rate
            )
