import logging
from collections import defaultdict

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from genteval.compressors.simple_gent.proto import simple_gent_pb2

from .node_feature import NodeFeature


class RootDurationModel:
    """
    Root duration model: (root_node_idx, root_child_cnt) -> mixture of Gaussian (models log(duration))

    Similar to root_table.py but organized by time buckets and using NodeFeature.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Storage: {time_bucket: {NodeFeature: (GaussianMixture, min_duration, max_duration)}}
        self.duration_models = defaultdict(dict)

        # Temporary storage during fitting
        self._temp_durations = defaultdict(
            lambda: defaultdict(list)
        )  # {time_bucket: {NodeFeature: [log_durations]}}
        self._temp_original_durations = defaultdict(
            lambda: defaultdict(list)
        )  # {time_bucket: {NodeFeature: [original_durations]}}

    def fit(self, traces):
        """Learn GMM models for root durations from traces."""
        self.logger.info(f"Training root duration model on {len(traces)} traces")

        # First pass: collect duration samples
        self.logger.info("Collecting root span duration samples")
        total_root_spans = 0
        total_duration_samples = 0

        for trace in tqdm(traces, desc="Collecting duration samples"):
            trace_start_time = trace.start_time
            time_bucket = int(trace_start_time // self.config.time_bucket_duration_us)

            # Find root spans
            trace_root_count = 0
            for span_id, span_data in trace.spans.items():
                if span_data.get("parentSpanId") is None:
                    # Count children
                    child_count = sum(
                        1
                        for s in trace.spans.values()
                        if s.get("parentSpanId") == span_id
                    )

                    node_idx = span_data["nodeIdx"]
                    root_feature = NodeFeature(
                        node_idx=node_idx, child_count=child_count
                    )

                    duration = span_data["duration"]
                    log_duration = np.log(duration + 1)  # +1 to avoid log(0)

                    self._temp_durations[time_bucket][root_feature].append(log_duration)
                    self._temp_original_durations[time_bucket][root_feature].append(
                        duration
                    )

                    trace_root_count += 1
                    total_duration_samples += 1

            total_root_spans += trace_root_count

        self.logger.info(
            f"Collected {total_duration_samples} duration samples from {total_root_spans} root spans"
        )
        self.logger.info(
            f"Found duration data for {len(self._temp_durations)} time buckets"
        )

        # Second pass: fit GMM for each (time_bucket, feature) combination
        self.logger.info("Fitting Gaussian Mixture Models for each feature")

        # Calculate total work for progress bar
        total_feature_combinations = sum(
            len(bucket_features) for bucket_features in self._temp_durations.values()
        )
        self.logger.info(
            f"Fitting GMMs for {total_feature_combinations} unique (time_bucket, feature) combinations"
        )

        successful_fits = 0
        fallback_fits = 0
        single_sample_fits = 0
        skipped_fits = 0

        # Create progress bar for all feature combinations
        with tqdm(total=total_feature_combinations, desc="Fitting GMM models") as pbar:
            for time_bucket, bucket_features in self._temp_durations.items():
                for feature, log_durations in bucket_features.items():
                    original_durations = self._temp_original_durations[time_bucket][
                        feature
                    ]

                    min_duration = min(original_durations)
                    max_duration = max(original_durations)

                    if len(log_durations) >= self.config.min_samples_for_gmm:
                        samples_array = np.array(log_durations).reshape(-1, 1)

                        # Choose number of components
                        n_components = min(
                            self.config.max_gmm_components,
                            max(1, len(log_durations) // 10),
                        )

                        try:
                            gmm = GaussianMixture(
                                n_components=n_components,
                                covariance_type="full",
                            )
                            gmm.fit(samples_array)
                            self.duration_models[time_bucket][feature] = (
                                gmm,
                                min_duration,
                                max_duration,
                            )
                            successful_fits += 1
                        except Exception as e:
                            # Fallback to single component
                            self.logger.warning(
                                f"GMM fitting failed for {time_bucket}/{feature}: {e}. Using single component."
                            )
                            gmm = GaussianMixture(n_components=1)
                            gmm.fit(samples_array)
                            self.duration_models[time_bucket][feature] = (
                                gmm,
                                min_duration,
                                max_duration,
                            )
                            fallback_fits += 1

                    elif len(log_durations) == 1:
                        # Single sample: create simple GMM
                        value = log_durations[0]
                        samples_array = np.array([value, value + 1e-6]).reshape(-1, 1)
                        gmm = GaussianMixture(n_components=1)
                        gmm.fit(samples_array)
                        self.duration_models[time_bucket][feature] = (
                            gmm,
                            min_duration,
                            max_duration,
                        )
                        single_sample_fits += 1
                    else:
                        # Not enough samples, skip
                        skipped_fits += 1
                        self.logger.debug(
                            f"Skipping GMM for {time_bucket}/{feature}: only {len(log_durations)} samples"
                        )

                    pbar.update(1)

        # Clear temporary storage
        self._temp_durations.clear()
        self._temp_original_durations.clear()

        total_models = sum(len(bucket) for bucket in self.duration_models.values())
        self.logger.info(
            f"GMM fitting complete: {total_models} models across {len(self.duration_models)} time buckets"
        )
        self.logger.info(
            f"Fit results: {successful_fits} successful, {fallback_fits} fallback, "
            f"{single_sample_fits} single-sample, {skipped_fits} skipped"
        )

    def _get_gmm_model(
        self, time_bucket: int, feature: NodeFeature
    ) -> tuple[GaussianMixture, float, float]:
        """Get GMM model for a given time bucket and feature with fallbacks."""
        # Direct lookup
        if (
            time_bucket in self.duration_models
            and feature in self.duration_models[time_bucket]
        ):
            return self.duration_models[time_bucket][feature]

        # Fallback 1: same feature in closest time bucket
        if self.duration_models:
            for bucket_time in sorted(
                self.duration_models.keys(), key=lambda t: abs(t - time_bucket)
            ):
                if feature in self.duration_models[bucket_time]:
                    return self.duration_models[bucket_time][feature]

        # Fallback 2: same node name with any child count in closest time bucket
        if self.duration_models:
            for bucket_time in sorted(
                self.duration_models.keys(), key=lambda t: abs(t - time_bucket)
            ):
                for candidate_feature in self.duration_models[bucket_time]:
                    if candidate_feature.name == feature.name:
                        return self.duration_models[bucket_time][candidate_feature]

        # Fallback 3: any model in closest time bucket
        if self.duration_models:
            closest_bucket = min(
                self.duration_models.keys(), key=lambda t: abs(t - time_bucket)
            )
            if self.duration_models[closest_bucket]:
                # Use first available model
                return next(iter(self.duration_models[closest_bucket].values()))

        # Final fallback: create default GMM
        default_samples = np.random.normal(5.0, 1.0, 100).reshape(
            -1, 1
        )  # log(duration) around 5.0
        default_gmm = GaussianMixture(n_components=1)
        default_gmm.fit(default_samples)
        return default_gmm, 1.0, float("inf")

    def sample_duration(self, time_bucket: int, feature: NodeFeature) -> float:
        """Sample a duration for a given time bucket and feature."""
        gmm, min_duration, max_duration = self._get_gmm_model(time_bucket, feature)

        if min_duration == max_duration:
            return float(min_duration)

        # Reject sampling with bounds checking
        max_attempts = 100
        for _ in range(max_attempts):
            # Sample from GMM (log space)
            log_duration = gmm.sample(1)[0][0, 0]
            duration = np.exp(log_duration) - 1  # Reverse the +1 from log transform

            # Check bounds
            if min_duration <= duration <= max_duration:
                return max(1.0, float(duration))  # Ensure minimum duration of 1

        # Fallback: return middle value
        return float((min_duration + max_duration) / 2)

    def sample_duration_batch(
        self, time_buckets: list[int], features: list[NodeFeature]
    ) -> list[float]:
        """Sample durations for multiple (time_bucket, feature) pairs."""
        if len(time_buckets) != len(features):
            raise ValueError("time_buckets and features must have the same length")

        results = []
        for time_bucket, feature in zip(time_buckets, features, strict=False):
            duration = self.sample_duration(time_bucket, feature)
            results.append(duration)

        return results

    def save_state_dict(self, proto_models):
        """Save model state to protobuf message."""

        # Group data by time buckets
        time_bucket_data = {}
        for time_bucket, bucket_models in self.duration_models.items():
            if time_bucket not in time_bucket_data:
                time_bucket_data[time_bucket] = []

            for feature, (gmm, min_dur, max_dur) in bucket_models.items():
                root_duration_model = simple_gent_pb2.RootDurationModel()
                root_duration_model.feature.node_idx = feature.node_idx
                root_duration_model.feature.child_count = feature.child_count
                root_duration_model.min_duration = min_dur
                root_duration_model.max_duration = max_dur

                # Save GMM parameters
                root_duration_model.distribution.n_components = gmm.n_components
                root_duration_model.distribution.weights.extend(gmm.weights_)
                root_duration_model.distribution.means.extend(gmm.means_.flatten())
                root_duration_model.distribution.variances.extend(
                    gmm.covariances_.flatten()
                )

                time_bucket_data[time_bucket].append(root_duration_model)

        # Add to protobuf message
        for time_bucket, root_duration_models in time_bucket_data.items():
            # Find or create time bucket
            bucket_models = None
            for tb in proto_models.time_buckets:
                if tb.time_bucket == time_bucket:
                    bucket_models = tb
                    break

            if bucket_models is None:
                bucket_models = proto_models.time_buckets.add()
                bucket_models.time_bucket = time_bucket

            # Add root duration models to this bucket
            bucket_models.root_duration_models.extend(root_duration_models)

    def load_state_dict(self, proto_models):
        """Load model state from protobuf message."""

        self.duration_models = defaultdict(dict)

        # Load from protobuf message
        for time_bucket_models in proto_models.time_buckets:
            time_bucket = time_bucket_models.time_bucket

            for root_duration_model in time_bucket_models.root_duration_models:
                feature = NodeFeature(
                    node_idx=root_duration_model.feature.node_idx,
                    child_count=root_duration_model.feature.child_count,
                )

                # Reconstruct GMM
                gmm = GaussianMixture(
                    n_components=root_duration_model.distribution.n_components,
                    covariance_type="full",
                )

                # Set GMM parameters
                gmm.weights_ = np.array(root_duration_model.distribution.weights)
                gmm.means_ = np.array(root_duration_model.distribution.means).reshape(
                    -1, 1
                )

                # Reshape covariances for full covariance type
                n_comp = root_duration_model.distribution.n_components
                covariances = np.array(root_duration_model.distribution.variances)
                gmm.covariances_ = covariances.reshape(n_comp, 1, 1)

                # Set other required attributes
                gmm.converged_ = True
                gmm.n_iter_ = 1

                min_dur = root_duration_model.min_duration
                max_dur = root_duration_model.max_duration

                self.duration_models[time_bucket][feature] = (gmm, min_dur, max_dur)

        total_models = sum(len(bucket) for bucket in self.duration_models.values())
        self.logger.info(
            f"Loaded {total_models} GMM models across {len(self.duration_models)} time buckets"
        )
