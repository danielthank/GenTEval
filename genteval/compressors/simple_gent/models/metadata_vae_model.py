import logging
import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from genteval.models import MetadataVAE

from .node_feature import NodeFeature


class MetadataVAEModel:
    """
    MetadataVAE model adapter for simple_gent that replaces both gap_ratio_model and duration_ratio_model.
    Uses neural networks to jointly model gap ratios and duration ratios.
    """

    def __init__(self, config, vocab_size: int):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.vocab_size = vocab_size

        # Dictionary to store models per time bucket: {time_bucket: model}
        self.models = {}
        self.optimizers = {}

        self.is_fitted = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def fit(self, traces):
        """Learn MetadataVAE models for both gap and duration ratios from traces."""
        self.logger.info("Training MetadataVAE model")

        # Prepare training data grouped by time bucket
        training_data_by_bucket = self._prepare_training_data(traces)

        # Use vocab size from constructor
        vocab_size = self.vocab_size

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
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
            )

            val_dataset = TensorDataset(
                torch.FloatTensor(val_inputs), torch.FloatTensor(val_targets)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

            # Early stopping parameters
            best_val_loss = float("inf")
            patience = self.config.early_stopping_patience
            patience_counter = 0
            best_model_state = None

            # Training loop for this time bucket
            model.train()
            epochs = self.config.metadata_epochs
            pbar = tqdm(
                range(epochs),
                desc=f"Training MetadataVAE for bucket {time_bucket}",
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

                    # Extract features (removed parent_start_time)
                    parent_duration = inputs_device[:, 0]
                    normalized_child_idx = inputs_device[:, 1]
                    parent_node_idx = inputs_device[:, 2].long()
                    child_node_idx = inputs_device[:, 3].long()

                    optimizer.zero_grad()
                    model_output = model(
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

    def _prepare_training_data(self, traces) -> dict:
        """Prepare training data from traces, grouped by time bucket."""

        # Note: traces are already preprocessed with nodeIdx instead of nodeName

        # Collect training data
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
                        parent_node_idx = parent_data["nodeIdx"]
                        child_node_idx = span_data["nodeIdx"]

                        child_start_time = span_data["startTime"]
                        child_duration = span_data["duration"]
                        gap_from_parent = child_start_time - parent_start_time

                        # Skip invalid data
                        if parent_duration <= 0 or child_duration <= 0:
                            continue

                        # Calculate time bucket
                        time_bucket = int(
                            parent_start_time // self.config.time_bucket_duration_us
                        )

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

                        # Store raw data (with node indices and normalized child_idx)
                        raw_training_data_by_bucket[time_bucket].append(
                            {
                                "parent_duration": parent_duration,
                                "normalized_child_idx": normalized_child_idx,
                                "parent_node_idx": parent_node_idx,
                                "child_node_idx": child_node_idx,
                                "gap_from_parent_ratio": gap_from_parent_ratio,
                                "child_duration_ratio": child_duration_ratio,
                            }
                        )

            except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
                self.logger.warning(f"Error processing trace: {e}")
                continue

        if not raw_training_data_by_bucket:
            raise ValueError("No valid training data found")

        # Process data for each time bucket
        training_data_by_bucket = {}

        for time_bucket, raw_training_data in raw_training_data_by_bucket.items():
            # Extract node indices directly (no transformation needed)
            parent_node_indices = [
                item["parent_node_idx"] for item in raw_training_data
            ]
            child_node_indices = [item["child_node_idx"] for item in raw_training_data]

            # Create training arrays for this bucket
            num_examples = len(raw_training_data)
            training_inputs = np.zeros((num_examples, 4))  # Reduced from 5 to 4 inputs
            training_targets = np.zeros((num_examples, 2))

            # Extract all data at once using list comprehensions (vectorized)
            training_inputs[:, 0] = [
                item["parent_duration"] for item in raw_training_data
            ]
            training_inputs[:, 1] = [
                item["normalized_child_idx"] for item in raw_training_data
            ]
            training_inputs[:, 2] = parent_node_indices
            training_inputs[:, 3] = child_node_indices

            training_targets[:, 0] = [
                item["gap_from_parent_ratio"] for item in raw_training_data
            ]
            training_targets[:, 1] = [
                item["child_duration_ratio"] for item in raw_training_data
            ]

            # No manual normalization - let the model handle it
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

                # Extract features (removed parent_start_time)
                parent_duration = inputs_device[:, 0]
                normalized_child_idx = inputs_device[:, 1]
                parent_node_idx = inputs_device[:, 2].long()
                child_node_idx = inputs_device[:, 3].long()

                model_output = model(
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

    def sample_ratios_batch(
        self, requests: list[tuple[int, NodeFeature, NodeFeature, float, float]]
    ) -> list[tuple[float, float]]:
        """
        Batch sample gap and duration ratios for multiple requests.
        Groups by time_bucket and batch processes efficiently.

        Args:
            requests: List of (time_bucket, parent_feature, child_feature,
                             parent_duration, normalized_child_idx) tuples
                     where normalized_child_idx = child_idx / child_count

        Returns:
            List of (gap_ratio, duration_ratio) tuples in same order as requests
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before sampling")

        if not requests:
            return []

        # Group requests by time bucket
        requests_by_bucket = {}
        for i, (
            time_bucket,
            parent_feature,
            child_feature,
            parent_duration,
            normalized_child_idx,
        ) in enumerate(requests):
            if time_bucket not in requests_by_bucket:
                requests_by_bucket[time_bucket] = []
            requests_by_bucket[time_bucket].append(
                (
                    i,
                    parent_feature,
                    child_feature,
                    parent_duration,
                    normalized_child_idx,
                )
            )

        # Process each time bucket separately
        results = [None] * len(requests)

        for time_bucket, bucket_requests in requests_by_bucket.items():
            # Get the appropriate model for this time bucket
            model = self._get_model_for_time_bucket(time_bucket)
            model.eval()

            with torch.no_grad():
                # Extract features for batch processing (removed parent_start_time)
                indices = [req[0] for req in bucket_requests]
                parent_features = [req[1] for req in bucket_requests]
                child_features = [req[2] for req in bucket_requests]
                parent_durations = [req[3] for req in bucket_requests]
                normalized_child_indices = [req[4] for req in bucket_requests]

                # Extract node indices from NodeFeatures (they're already encoded)
                parent_node_indices = np.array([pf.node_idx for pf in parent_features])
                child_node_indices = np.array([cf.node_idx for cf in child_features])

                # Use raw duration values - model will handle normalization
                parent_duration_tensor = torch.FloatTensor(parent_durations).to(
                    self.device
                )
                child_idx_tensor = torch.FloatTensor(normalized_child_indices).to(
                    self.device
                )
                parent_node_tensor = torch.LongTensor(parent_node_indices).to(
                    self.device
                )
                child_node_tensor = torch.LongTensor(child_node_indices).to(self.device)

                # Batch sample using VAE (single forward pass for entire batch!)
                batch_samples = model.sample(
                    parent_duration_tensor,
                    child_idx_tensor,
                    parent_node_tensor,
                    child_node_tensor,
                )

                # Process batch results and store in original order
                for j, original_idx in enumerate(indices):
                    # Extract ratios and ensure they're in valid range
                    gap_ratio = float(np.clip(batch_samples[j, 0].item(), 0.0, 1.0))
                    duration_ratio = float(
                        np.clip(batch_samples[j, 1].item(), 0.0, 1.0)
                    )

                    results[original_idx] = (gap_ratio, duration_ratio)

        return results

    def save_state_dict(self, proto_models):
        """Save model state to protobuf message."""
        from genteval.compressors.simple_gent.proto import simple_gent_pb2

        # Group data by time buckets
        time_bucket_data = {}
        for time_bucket, model in self.models.items():
            if time_bucket not in time_bucket_data:
                time_bucket_data[time_bucket] = []

            metadata_vae_model = simple_gent_pb2.MetadataVAEModel()
            metadata_vae_model.time_bucket = time_bucket
            metadata_vae_model.vocab_size = self.vocab_size
            metadata_vae_model.hidden_dim = model.hidden_dim
            metadata_vae_model.latent_dim = model.latent_dim
            metadata_vae_model.use_flow_prior = model.use_flow_prior
            metadata_vae_model.prior_flow_layers = self.config.prior_flow_layers
            metadata_vae_model.prior_flow_hidden_dim = self.config.prior_flow_hidden_dim
            metadata_vae_model.num_beta_components = model.num_beta_components

            # Serialize only decoder and node_embedding parts of the model
            model_state = {
                k: v
                for k, v in model.state_dict().items()
                if k.startswith(
                    ("decoder", "node_embedding")
                )  # Keep embeddings for conditioning
            }
            metadata_vae_model.model_state_dict = pickle.dumps(model_state)

            time_bucket_data[time_bucket].append(metadata_vae_model)

        # Add to protobuf message
        for time_bucket, metadata_vae_models in time_bucket_data.items():
            # Find or create time bucket
            bucket_models = None
            for tb in proto_models.time_buckets:
                if tb.time_bucket == time_bucket:
                    bucket_models = tb
                    break

            if bucket_models is None:
                bucket_models = proto_models.time_buckets.add()
                bucket_models.time_bucket = time_bucket

            # Add metadata VAE models to this bucket
            bucket_models.metadata_vae_models.extend(metadata_vae_models)

            # Note: node encoder is now saved at the global level in SimpleGenTModels

    def load_state_dict(self, proto_models):
        """Load model state from protobuf message."""
        self.models = {}
        self.optimizers = {}

        # Load from protobuf message
        for time_bucket_models in proto_models.time_buckets:
            time_bucket = time_bucket_models.time_bucket

            # Note: node encoder is loaded at the global level by SimpleGenTCompressor

            for metadata_vae_model in time_bucket_models.metadata_vae_models:
                # Initialize model
                model = MetadataVAE(
                    metadata_vae_model.vocab_size,
                    metadata_vae_model.hidden_dim,
                    metadata_vae_model.latent_dim,
                    metadata_vae_model.use_flow_prior,
                    metadata_vae_model.prior_flow_layers,
                    metadata_vae_model.prior_flow_hidden_dim,
                    metadata_vae_model.num_beta_components,
                )
                model.to(self.device)

                # Load model state dict (only decoder and node_embedding parts)
                model_state_dict = pickle.loads(metadata_vae_model.model_state_dict)
                model.load_state_dict(
                    model_state_dict, strict=False
                )  # strict=False for partial loading

                # Store model and optimizer
                self.models[time_bucket] = model
                self.optimizers[time_bucket] = torch.optim.Adam(
                    model.parameters(), lr=self.config.learning_rate
                )

        self.is_fitted = True
        total_models = len(self.models)
        self.logger.info(
            f"Loaded {total_models} MetadataVAE models across {len(self.models)} time buckets"
        )
