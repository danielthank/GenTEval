import logging
from collections import defaultdict

import numpy as np
from sklearn.mixture import GaussianMixture

from genteval.compressors import CompressedDataset, SerializationFormat


class RootDurationTableSynthesizer:
    """
    Hashtable-based synthesizer that models root duration using GMMs.
    for each (startTimeBucket, operationType) combination and uses reject sampling.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Time bucketing configuration (hardcoded)
        self.bucket_size_us = 60 * 1000000  # 1 minute in microseconds

        # Hashtable to store GMM models and duration bounds: {(time_bucket, operation_type): (GaussianMixture, min_duration, max_duration)}
        self.stats_table = defaultdict(lambda: None)

        # Temporary storage during fitting: {(time_bucket, operation_type): list of log durations}
        self._temp_stats = defaultdict(list)
        # Store original durations for min/max calculation: {(time_bucket, operation_type): list of original durations}
        self._temp_original_durations = defaultdict(list)

    def fit(self, traces):
        """Build the hashtable from root span data."""
        self.logger.info("Building Root Duration Hashtable")

        # Collect root span data
        root_start_times = []
        root_durations = []
        root_node_names = []

        for trace in traces:
            # Find root spans (spans with no parent)
            root_spans = [
                span_data
                for span_data in trace.spans.values()
                if span_data["parentSpanId"] is None
            ]

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

        # Convert start times to time buckets
        time_buckets = root_start_times // self.bucket_size_us

        # Use log-transformed durations directly (no normalization needed)
        log_durations = np.log(root_durations + 1)  # +1 to avoid log(0)

        # First pass: collect log duration samples and original durations for each key
        for i in range(len(time_buckets)):
            key = (int(time_buckets[i]), root_node_names[i])
            x = log_durations[i]
            original_duration = root_durations[i]
            self._temp_stats[key].append(x)
            self._temp_original_durations[key].append(original_duration)

        # Second pass: fit GMM for each key and store with duration bounds
        for key, samples in self._temp_stats.items():
            original_durations = self._temp_original_durations[key]
            min_duration = min(original_durations)
            max_duration = max(original_durations)

            if len(samples) >= 2:  # Need at least 2 samples for GMM
                samples_array = np.array(samples).reshape(-1, 1)

                # Choose number of components (max 3, min 1, based on data size)
                n_components = min(3, max(1, len(samples) // 10))

                try:
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type="full",
                    )
                    gmm.fit(samples_array)
                    self.stats_table[key] = (gmm, min_duration, max_duration)
                except Exception as e:
                    # Fallback to single component if fitting fails
                    self.logger.warning(
                        f"GMM fitting failed for {key}: {e}. Using single component."
                    )
                    gmm = GaussianMixture(n_components=1)
                    gmm.fit(samples_array)
                    self.stats_table[key] = (gmm, min_duration, max_duration)
            elif len(samples) == 1:
                # Single sample: create degenerate GMM by duplicating the sample
                value = samples[0]
                # Add tiny noise to create 2 samples for GMM fitting
                samples_array = np.array([value, value + 1e-6]).reshape(-1, 1)
                gmm = GaussianMixture(n_components=1)
                gmm.fit(samples_array)
                self.stats_table[key] = (gmm, min_duration, max_duration)

        # Clear temporary storage
        self._temp_stats.clear()
        self._temp_original_durations.clear()

        self.logger.info(
            f"Built hashtable with {len(self.stats_table)} unique (time_bucket, operation_type) combinations"
        )

    def _get_gmm_model(
        self, time_bucket: int, node_name: str
    ) -> tuple[GaussianMixture, float, float]:
        """Get GMM model, min_duration, and max_duration for a given (time_bucket, operation_type) combination."""
        key = (time_bucket, node_name)

        if key in self.stats_table and self.stats_table[key] is not None:
            return self.stats_table[key]  # Returns (gmm, min_duration, max_duration)

        # First fallback: find same node_name with closest time_bucket
        if self.stats_table:
            # Find entries with the same node_name
            same_node_entries = {}
            for (tb, nn), entry in self.stats_table.items():
                if nn == node_name and entry is not None:
                    same_node_entries[tb] = entry

            if same_node_entries:
                # Find the closest time_bucket
                closest_time_bucket = min(
                    same_node_entries.keys(), key=lambda tb: abs(tb - time_bucket)
                )
                return same_node_entries[
                    closest_time_bucket
                ]  # Returns (gmm, min_duration, max_duration)

            # Second fallback: use all available samples if no same node_name found
            all_samples = []
            all_min_durations = []
            all_max_durations = []
            for entry in self.stats_table.values():
                if entry is not None:
                    gmm, min_duration, max_duration = entry
                    # Sample from existing GMMs to create fallback data
                    samples = gmm.sample(100)[0].flatten()
                    all_samples.extend(samples)
                    all_min_durations.append(min_duration)
                    all_max_durations.append(max_duration)

            if all_samples:
                all_samples_array = np.array(all_samples).reshape(-1, 1)
                fallback_gmm = GaussianMixture(n_components=1)
                fallback_gmm.fit(all_samples_array)
                # Use the min/max of all durations as fallback bounds
                fallback_min_duration = (
                    min(all_min_durations) if all_min_durations else 1.0
                )
                fallback_max_duration = (
                    max(all_max_durations) if all_max_durations else float("inf")
                )
                return fallback_gmm, fallback_min_duration, fallback_max_duration

        # Default fallback: single component with mean=0, std=1, and default duration bounds
        default_samples = np.random.normal(0, 1, 100).reshape(-1, 1)
        default_gmm = GaussianMixture(n_components=1)
        default_gmm.fit(default_samples)
        return default_gmm, 1.0, float("inf")

    def synthesize_root_duration_batch(
        self, start_times: list[float], node_names: list[str]
    ) -> list[float]:
        """Generate root span durations for multiple start times and node names at once.

        Args:
            start_times: List of start times
            node_names: List of node names

        Returns:
            List of durations sampled from GMM.
        """
        if not self.stats_table:
            raise ValueError("Hashtable not built. Call fit() first.")

        if len(start_times) != len(node_names):
            raise ValueError("start_times and node_names must have the same length")

        if not start_times:
            return []

        results = []

        for start_time, node_name in zip(start_times, node_names, strict=False):
            # Convert start time to time bucket
            time_bucket = int(start_time // self.bucket_size_us)

            # Get GMM model and duration bounds from hashtable
            gmm, min_duration, max_duration = self._get_gmm_model(
                time_bucket, node_name
            )

            if min_duration == max_duration:
                results.append(float(min_duration))
                continue

            max_attempts = 100  # Maximum attempts for reject sampling
            attempt = 0

            while attempt < max_attempts:
                # Sample from GMM (returns shape (1, 1))
                log_duration = gmm.sample(1)[0][0, 0]

                # Convert back to actual duration
                duration = np.exp(log_duration) - 1  # Reverse log transform
                duration = max(1, float(duration))  # Ensure positive duration

                # Accept if duration is within observed bounds
                if min_duration <= duration <= max_duration:
                    results.append(duration)
                    break

                attempt += 1

            # Fallback if reject sampling fails
            if attempt >= max_attempts:
                # Use a random value between min and max as fallback
                fallback_duration = np.random.uniform(min_duration, max_duration)
                results.append(float(fallback_duration))
                self.logger.warning(
                    f"Reject sampling failed for time_bucket={time_bucket}, node_name={node_name}. Using random fallback between {min_duration:.2f} and {max_duration:.2f}."
                )

        return results

    def save_state_dict(
        self, compressed_data: CompressedDataset, decoder_only: bool = False
    ):
        """Save state dictionary."""

        # Convert defaultdict to regular dict for serialization
        # GMM models are serializable with cloudpickle
        stats_dict = dict(self.stats_table)

        compressed_data.add(
            "root_table_synthesizer",
            CompressedDataset(
                data={
                    "stats_table": (stats_dict, SerializationFormat.CLOUDPICKLE),
                }
            ),
            SerializationFormat.CLOUDPICKLE,
        )

    def load_state_dict(self, compressed_dataset):
        """Load state dictionary."""
        if "root_table_synthesizer" not in compressed_dataset:
            raise ValueError("No root_table_synthesizer found in compressed dataset")

        logger = logging.getLogger(__name__)

        # Load root synthesizer data
        root_synthesizer_data = compressed_dataset["root_table_synthesizer"]

        # Convert back to defaultdict with GMM models and duration bounds
        stats_dict = root_synthesizer_data["stats_table"]
        self.stats_table = defaultdict(lambda: None)
        self.stats_table.update(stats_dict)

        logger.info(f"Loaded root hashtable with {len(self.stats_table)} entries")
