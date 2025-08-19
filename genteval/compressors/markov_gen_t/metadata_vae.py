import logging

import numpy as np
import torch
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from genteval.compressors import CompressedDataset, SerializationFormat
from genteval.models import MetadataVAE


class MetadataSynthesizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.node_encoder = LabelEncoder()

        # Dictionary to store models per time bucket: {time_bucket: model}
        self.models = {}
        self.optimizers = {}

        self.start_time_scaler = {"mean": 0, "std": 1}
        self.duration_scaler = {"mean": 0, "std": 1}
        self.is_fitted = False

        # Time bucketing configuration (same as RootDurationTableSynthesizer)
        self.bucket_size_us = 60 * 1000000 * 100  # 1 minute in microseconds

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def _prepare_training_data(
        self, traces: list
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Prepare training data from traces, grouped by time bucket."""

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

        # Dictionary to store raw training data by time bucket
        raw_training_data_by_bucket = {}

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

                        # Calculate time bucket
                        time_bucket = int(parent_start_time // self.bucket_size_us)

                        # Calculate ratios and ensure they're bounded [0, 1]
                        gap_from_parent_ratio = max(
                            0, min(1, gap_from_parent / parent_duration)
                        )
                        child_duration_ratio = max(
                            0, min(1, child_duration / parent_duration)
                        )

                        # Normalize child_idx by total number of children
                        normalized_child_idx = child_idx / len(children)

                        # Initialize bucket if not exists
                        if time_bucket not in raw_training_data_by_bucket:
                            raw_training_data_by_bucket[time_bucket] = []

                        # Store raw data (with string node names and normalized child_idx)
                        raw_training_data_by_bucket[time_bucket].append(
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

        if not raw_training_data_by_bucket:
            raise ValueError("No valid training data found")

        # Compute scaling parameters
        self.start_time_scaler = {
            "mean": np.mean(all_start_times),
            "std": np.std(all_start_times) + 1e-8,
        }
        self.duration_scaler = {
            "mean": np.mean(all_durations),
            "std": np.std(all_durations) + 1e-8,
        }

        # Process data for each time bucket
        training_data_by_bucket = {}

        for time_bucket, raw_training_data in raw_training_data_by_bucket.items():
            self.logger.info(
                f"Processing time bucket {time_bucket} with {len(raw_training_data)} examples"
            )

            # Batch transform all node names for this bucket
            all_parent_nodes = [item["parent_node"] for item in raw_training_data]
            all_child_nodes = [item["child_node"] for item in raw_training_data]

            parent_node_indices = self.node_encoder.transform(all_parent_nodes)
            child_node_indices = self.node_encoder.transform(all_child_nodes)

            # Create training arrays for this bucket
            num_examples = len(raw_training_data)
            training_inputs = np.zeros((num_examples, 5))
            training_targets = np.zeros((num_examples, 2))

            # Extract all data at once using list comprehensions (vectorized)
            training_inputs[:, 0] = [
                item["parent_start_time"] for item in raw_training_data
            ]
            training_inputs[:, 1] = [
                item["parent_duration"] for item in raw_training_data
            ]
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

            # Apply scaling to inputs only (targets are already bounded ratios [0,1])
            training_inputs[:, 0] = (
                training_inputs[:, 0] - self.start_time_scaler["mean"]
            ) / self.start_time_scaler["std"]
            training_inputs[:, 1] = (
                training_inputs[:, 1] - self.duration_scaler["mean"]
            ) / self.duration_scaler["std"]

            training_data_by_bucket[time_bucket] = (training_inputs, training_targets)

        return training_data_by_bucket

    def _evaluate_model(self, model, val_loader, beta=1.0):
        """Evaluate a specific model on validation set and return average loss."""
        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                # Move to device
                inputs_device = batch_inputs.to(self.device)
                targets_device = batch_targets.to(self.device)

                # Extract features
                parent_start_time = inputs_device[:, 0]
                parent_duration = inputs_device[:, 1]
                normalized_child_idx = inputs_device[:, 2]
                parent_node_idx = inputs_device[:, 3].long()
                child_node_idx = inputs_device[:, 4].long()

                model_output = model(
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
                val_loss, _, _ = model.loss_function(
                    recon,
                    targets_device,
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
        """Train separate metadata synthesis models for each time bucket sequentially with train/val split and early stopping."""
        self.logger.info(
            "Training Metadata Neural Networks per time bucket sequentially"
        )

        # Prepare data grouped by time bucket
        training_data_by_bucket = self._prepare_training_data(traces)

        # Initialize vocab size once
        vocab_size = len(self.node_encoder.classes_)

        # Sort time buckets for sequential training
        sorted_time_buckets = sorted(training_data_by_bucket.keys())
        self.logger.info(
            f"Training models sequentially for time buckets: {sorted_time_buckets}"
        )

        previous_model = None

        # Train models sequentially in time order
        for time_bucket in sorted_time_buckets:
            inputs, targets = training_data_by_bucket[time_bucket]
            self.logger.info(
                f"Training model for time bucket {time_bucket} with {len(inputs)} examples"
            )

            # Skip time buckets with too few samples
            if len(inputs) < 10:
                self.logger.warning(
                    f"Skipping time bucket {time_bucket} due to insufficient data ({len(inputs)} examples)"
                )
                continue

            # Train/validation split (80/20)
            train_inputs, val_inputs, train_targets, val_targets = train_test_split(
                inputs, targets, test_size=0.2, random_state=42
            )

            self.logger.info(
                f"Time bucket {time_bucket}: Training with {len(train_inputs)} examples, validating with {len(val_inputs)} examples"
            )

            # Initialize model for this time bucket
            model = MetadataVAE(
                vocab_size,
                self.config.metadata_hidden_dim,
                self.config.metadata_latent_dim,
                self.config.use_flow_prior,
                self.config.prior_flow_layers,
                self.config.prior_flow_hidden_dim,
                self.config.num_beta_components,
            )

            # If we have a previous model, initialize from its state
            if previous_model is not None:
                self.logger.info(
                    f"Initializing time bucket {time_bucket} model from previous time bucket model"
                )
                model.load_state_dict(previous_model.state_dict())
                # Use a lower learning rate for fine-tuning
                fine_tune_lr_factor = self.config.sequential_training_lr_factor
                learning_rate = self.config.learning_rate * fine_tune_lr_factor
                self.logger.info(
                    f"Using reduced learning rate {learning_rate} (factor: {fine_tune_lr_factor}) for fine-tuning"
                )
            else:
                learning_rate = self.config.learning_rate
                self.logger.info(
                    f"Training first model from scratch with learning rate {learning_rate}"
                )

            model.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Store model and optimizer
            self.models[time_bucket] = model
            self.optimizers[time_bucket] = optimizer

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

            # Training loop for this time bucket
            model.train()
            pbar = tqdm(
                range(self.config.metadata_epochs),
                desc=f"Training Metadata NN for bucket {time_bucket}",
            )
            for epoch in pbar:
                # Get current beta value for this epoch
                current_beta = self.config.beta

                model.train()
                epoch_total_loss = 0
                epoch_recon_loss = 0
                epoch_kl_loss = 0
                num_batches = 0

                for batch_inputs, batch_targets in train_loader:
                    # Move to device
                    inputs_device = batch_inputs.to(self.device)
                    targets_device = batch_targets.to(self.device)

                    # Extract features
                    parent_start_time = inputs_device[:, 0]
                    parent_duration = inputs_device[:, 1]
                    normalized_child_idx = inputs_device[:, 2]
                    parent_node_idx = inputs_device[:, 3].long()
                    child_node_idx = inputs_device[:, 4].long()

                    optimizer.zero_grad()
                    model_output = model(
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
                    total_loss, recon_loss, kl_loss = model.loss_function(
                        recon,
                        targets_device,
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
                    optimizer.step()

                    epoch_total_loss += total_loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_kl_loss += kl_loss.item()
                    num_batches += 1

                # Calculate training losses
                avg_total_loss = epoch_total_loss / max(num_batches, 1)
                avg_recon_loss = epoch_recon_loss / max(num_batches, 1)
                avg_kl_loss = epoch_kl_loss / max(num_batches, 1)

                # Evaluate on validation set
                val_loss = self._evaluate_model(model, val_loader, current_beta)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict().copy()
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
                            f"metadata_vae_bucket_{time_bucket}_train_loss": avg_total_loss,
                            f"metadata_vae_bucket_{time_bucket}_val_loss": val_loss,
                            f"metadata_vae_bucket_{time_bucket}_recon_loss": avg_recon_loss,
                            f"metadata_vae_bucket_{time_bucket}_kl_loss": avg_kl_loss,
                            f"metadata_vae_bucket_{time_bucket}_beta": current_beta,
                            f"metadata_vae_bucket_{time_bucket}_epoch": epoch,
                        }
                    )

                # Early stopping
                if patience_counter >= patience:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch + 1} for time bucket {time_bucket}"
                    )
                    break

            # Load best model state
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                self.logger.info(
                    f"Time bucket {time_bucket}: Restored best model with validation loss: {best_val_loss:.4f}"
                )

            # Set this model as the previous model for the next time bucket
            previous_model = model

        self.logger.info(
            f"Trained {len(self.models)} models for {len(self.models)} time buckets"
        )
        self.is_fitted = True

    def _get_model_for_time_bucket(self, time_bucket: int) -> MetadataVAE:
        """Get the appropriate model for a given time bucket, with fallback strategy."""
        if time_bucket in self.models:
            return self.models[time_bucket]

        # Fallback: find the closest time bucket
        if self.models:
            closest_bucket = min(self.models.keys(), key=lambda b: abs(b - time_bucket))
            self.logger.debug(
                f"Using model from time bucket {closest_bucket} for bucket {time_bucket}"
            )
            return self.models[closest_bucket]

        raise ValueError("No models available for inference")

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
        Uses the appropriate VAE model based on the time bucket of each parent start time.

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
            raise ValueError("Models must be fitted before synthesis")

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

        # Group samples by time bucket
        samples_by_bucket = {}
        for i, parent_start_time in enumerate(parent_start_times):
            time_bucket = int(parent_start_time // self.bucket_size_us)
            if time_bucket not in samples_by_bucket:
                samples_by_bucket[time_bucket] = []
            samples_by_bucket[time_bucket].append(i)

        # Process each time bucket separately
        results = [None] * len(parent_start_times)

        for time_bucket, indices in samples_by_bucket.items():
            # Get the appropriate model for this time bucket
            model = self._get_model_for_time_bucket(time_bucket)
            model.eval()

            with torch.no_grad():
                # Extract data for this time bucket
                bucket_start_times = [parent_start_times[i] for i in indices]
                bucket_durations = [parent_durations[i] for i in indices]
                bucket_child_indices = [child_indices[i] for i in indices]
                bucket_parent_nodes = [parent_nodes[i] for i in indices]
                bucket_child_nodes = [child_nodes[i] for i in indices]

                # Encode all node names in batch
                try:
                    parent_node_indices = self.node_encoder.transform(
                        bucket_parent_nodes
                    )
                except ValueError:
                    parent_node_indices = []
                    for parent_node in bucket_parent_nodes:
                        try:
                            parent_node_idx = self.node_encoder.transform(
                                [parent_node]
                            )[0]
                        except ValueError:
                            rng = np.random.default_rng()
                            parent_node_idx = rng.integers(
                                0, len(self.node_encoder.classes_)
                            )
                        parent_node_indices.append(parent_node_idx)
                    parent_node_indices = np.array(parent_node_indices)

                try:
                    child_node_indices = self.node_encoder.transform(bucket_child_nodes)
                except ValueError:
                    child_node_indices = []
                    for child_node in bucket_child_nodes:
                        try:
                            child_node_idx = self.node_encoder.transform([child_node])[
                                0
                            ]
                        except ValueError:
                            rng = np.random.default_rng()
                            child_node_idx = rng.integers(
                                0, len(self.node_encoder.classes_)
                            )
                        child_node_indices.append(child_node_idx)
                    child_node_indices = np.array(child_node_indices)

                # Scale all inputs
                scaled_start_times = [
                    (start_time - self.start_time_scaler["mean"])
                    / self.start_time_scaler["std"]
                    for start_time in bucket_start_times
                ]
                scaled_durations = [
                    (duration - self.duration_scaler["mean"])
                    / self.duration_scaler["std"]
                    for duration in bucket_durations
                ]

                # Prepare batch tensors and move to device
                parent_start_tensor = torch.FloatTensor(scaled_start_times).to(
                    self.device
                )
                parent_duration_tensor = torch.FloatTensor(scaled_durations).to(
                    self.device
                )
                normalized_child_idx_tensor = torch.FloatTensor(
                    bucket_child_indices
                ).to(self.device)
                parent_node_tensor = torch.LongTensor(parent_node_indices).to(
                    self.device
                )
                child_node_tensor = torch.LongTensor(child_node_indices).to(self.device)

                # Sample using VAE in batch (no encoder needed for generation)
                samples = model.sample(
                    parent_start_tensor,
                    parent_duration_tensor,
                    normalized_child_idx_tensor,
                    parent_node_tensor,
                    child_node_tensor,
                )

                # Process batch results and store in original order
                for j, original_idx in enumerate(indices):
                    # Get ratio outputs (already bounded [0,1] by sigmoid)
                    gap_from_parent_ratio = samples[j, 0].item()
                    child_duration_ratio = samples[j, 1].item()

                    # Convert ratios back to absolute values
                    gap_from_parent = gap_from_parent_ratio * bucket_durations[j]
                    child_duration = child_duration_ratio * bucket_durations[j]

                    # Compute child start time
                    child_start_time = bucket_start_times[j] + max(0, gap_from_parent)
                    child_duration = max(1, child_duration)

                    results[original_idx] = (child_start_time, child_duration)

        return results

    def save_state_dict(
        self, compressed_data: CompressedDataset, decoder_only: bool = False
    ):
        """Save state dictionary with optional decoder-only mode."""

        # Prepare models data to save
        models_data = {}
        for time_bucket, model in self.models.items():
            if decoder_only:
                model_state = {
                    k: v
                    for k, v in model.state_dict().items()
                    if k.startswith(
                        ("decoder", "node_embedding")
                    )  # Keep embeddings for conditioning
                }
            else:
                model_state = model.state_dict()
            models_data[time_bucket] = model_state

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
                    "bucket_size_us": (
                        self.bucket_size_us,
                        SerializationFormat.MSGPACK,
                    ),
                    "models_data": (models_data, SerializationFormat.CLOUDPICKLE),
                }
            ),
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

        # Load bucket size (with backward compatibility)
        if "bucket_size_us" in metadata_synthesizer_data:
            self.bucket_size_us = metadata_synthesizer_data["bucket_size_us"]
        else:
            self.bucket_size_us = 60 * 1000000 * 5  # Default fallback

        # Load models for each time bucket
        if "models_data" in metadata_synthesizer_data:
            models_data = metadata_synthesizer_data["models_data"]
            self.models = {}
            self.optimizers = {}

            for time_bucket, model_state in models_data.items():
                # Initialize model for this time bucket
                model = MetadataVAE(
                    vocab_size,
                    self.config.metadata_hidden_dim,
                    self.config.metadata_latent_dim,
                    self.config.use_flow_prior,
                    self.config.prior_flow_layers,
                    self.config.prior_flow_hidden_dim,
                    self.config.num_beta_components,
                )
                model.to(self.device)

                # Load model state dict
                model.load_state_dict(model_state, strict=False)

                # Store model and optimizer
                self.models[time_bucket] = model
                self.optimizers[time_bucket] = torch.optim.Adam(
                    model.parameters(), lr=self.config.learning_rate
                )

            logger.info(
                f"Loaded {len(self.models)} models for time buckets: {sorted(self.models.keys())}"
            )
        elif "state_dict" in metadata_synthesizer_data:
            # Backward compatibility: load single model as fallback
            model_state = metadata_synthesizer_data["state_dict"]

            # Initialize single model
            model = MetadataVAE(
                vocab_size,
                self.config.metadata_hidden_dim,
                self.config.metadata_latent_dim,
                self.config.use_flow_prior,
                self.config.prior_flow_layers,
                self.config.prior_flow_hidden_dim,
                self.config.num_beta_components,
            )
            model.to(self.device)
            model.load_state_dict(model_state, strict=False)

            # Store as bucket 0 for compatibility
            self.models = {0: model}
            self.optimizers = {
                0: torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            }

            logger.info("Loaded single model for backward compatibility")
        else:
            raise ValueError(
                "No models_data or state_dict found in metadata_synthesizer"
            )
