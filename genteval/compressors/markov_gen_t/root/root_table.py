import logging
from collections import defaultdict

import numpy as np

from genteval.compressors import CompressedDataset, SerializationFormat


class RootDurationTableSynthesizer:
    """
    Hashtable-based synthesizer that memorizes E[X], Variance, min_z, and max_z
    for each (startTimeBucket, operationType) combination and uses reject sampling.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Time bucketing configuration (hardcoded)
        self.bucket_size_us = 60 * 1000000  # 1 minute in microseconds

        # Hashtable to store statistics: {(time_bucket, operation_type): (mean, variance, min_z, max_z)}
        self.stats_table = defaultdict(
            lambda: [0.0, 0.0, 0.0, 0.0]
        )  # [mean, variance, min_z, max_z]

        # Temporary storage during fitting: {(time_bucket, operation_type): (count, sum_x, sum_x2, min_value, max_value)}
        self._temp_stats = defaultdict(
            lambda: [0, 0.0, 0.0, float("inf"), float("-inf")]
        )

    def fit(self, traces):
        """Build the hashtable from root span data."""
        self.logger.info("Building Root Duration Hashtable")

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

        # Convert start times to time buckets
        time_buckets = root_start_times // self.bucket_size_us

        # Use log-transformed durations directly (no normalization needed)
        log_durations = np.log(root_durations + 1)  # +1 to avoid log(0)

        # First pass: collect temporary statistics
        for i in range(len(time_buckets)):
            key = (int(time_buckets[i]), root_node_names[i])
            x = log_durations[i]

            # Update temporary statistics: count, sum_x, sum_x2, min_value, max_value
            self._temp_stats[key][0] += 1  # count
            self._temp_stats[key][1] += x  # sum_x
            self._temp_stats[key][2] += x * x  # sum_x2
            self._temp_stats[key][3] = min(self._temp_stats[key][3], x)  # min_value
            self._temp_stats[key][4] = max(self._temp_stats[key][4], x)  # max_value

        # Second pass: compute final statistics
        for key, (
            count,
            sum_x,
            sum_x2,
            min_value,
            max_value,
        ) in self._temp_stats.items():
            if count > 0:
                mean = sum_x / count
                variance = max(0, (sum_x2 / count) - (mean**2))  # Ensure non-negative
                std = (
                    np.sqrt(variance) if variance > 0 else 1e-6
                )  # Avoid division by zero

                # Compute normalized z-scores
                min_z = (min_value - mean) / std if std > 0 else 0.0
                max_z = (max_value - mean) / std if std > 0 else 0.0

                # Store final statistics
                self.stats_table[key] = [mean, variance, min_z, max_z]

        # Clear temporary storage
        self._temp_stats.clear()

        self.logger.info(
            f"Built hashtable with {len(self.stats_table)} unique (time_bucket, operation_type) combinations"
        )

    def _get_statistics(
        self, time_bucket: int, node_name: str
    ) -> tuple[float, float, float, float]:
        """Get mean, variance, min_z, and max_z for a given (time_bucket, operation_type) combination."""
        key = (time_bucket, node_name)

        if key in self.stats_table:
            mean, variance, min_z, max_z = self.stats_table[key]
            return mean, variance, min_z, max_z
        # Fallback: use global statistics or return default values
        if self.stats_table:
            # Compute global statistics as fallback
            all_means = []
            all_variances = []
            all_min_zs = []
            all_max_zs = []

            for mean, variance, min_z, max_z in self.stats_table.values():
                all_means.append(mean)
                all_variances.append(variance)
                all_min_zs.append(min_z)
                all_max_zs.append(max_z)

            global_mean = np.mean(all_means)
            global_variance = np.mean(all_variances)
            global_min_z = np.min(all_min_zs)
            global_max_z = np.max(all_max_zs)

            return global_mean, global_variance, global_min_z, global_max_z
        # Default fallback if no data
        return 0.0, 1.0, -3.0, 3.0

    def synthesize_root_duration_batch(
        self, start_times: list[float], node_names: list[str], num_samples: int = 1
    ) -> list[float]:
        """Generate root span durations for multiple start times and node names at once.

        Args:
            start_times: List of start times
            node_names: List of node names
            num_samples: Number of samples to generate per input (default: 1)

        Returns:
            List of durations using reject sampling within the observed z-score bounds.
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

            # Get statistics from hashtable
            mean, variance, min_z, max_z = self._get_statistics(time_bucket, node_name)

            # Ensure variance is non-negative (numerical stability)
            variance = max(1e-12, variance)
            std = np.sqrt(variance)

            # Generate samples using reject sampling
            for _ in range(num_samples):
                # Check if min_z == 0 and max_z == 0 (all values were at the mean)
                if min_z == 0.0 and max_z == 0.0:
                    # Just use the mean directly
                    duration = np.exp(mean) - 1
                    duration = max(1, float(duration))
                    results.append(duration)
                    continue

                max_attempts = 1000  # Prevent infinite loops
                attempt = 0

                while attempt < max_attempts:
                    # Sample from normal distribution
                    log_duration = np.random.normal(mean, std)

                    # Compute z-score
                    z_score = (log_duration - mean) / std

                    # Accept if within observed bounds
                    if min_z <= z_score <= max_z:
                        # Convert back to actual duration
                        duration = np.exp(log_duration) - 1  # Reverse log transform
                        duration = max(1, float(duration))  # Ensure positive duration
                        results.append(duration)
                        break

                    attempt += 1

                # Fallback if reject sampling fails (shouldn't happen often)
                if attempt >= max_attempts:
                    # Just use the mean
                    duration = np.exp(mean) - 1
                    duration = max(1, float(duration))
                    results.append(duration)
                    self.logger.warning(
                        f"Reject sampling failed for time_bucket={time_bucket}, node_name={node_name}. Using mean."
                    )
                    self.logger.warning(
                        f"Mean: {mean}, Variance: {variance}, Min Z: {min_z}, Max Z: {max_z}"
                    )

        return results

    def save_state_dict(
        self, compressed_data: CompressedDataset, decoder_only: bool = False
    ):
        """Save state dictionary."""

        # Convert defaultdict to regular dict for serialization
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

        # Convert back to defaultdict
        stats_dict = root_synthesizer_data["stats_table"]
        self.stats_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.stats_table.update(stats_dict)

        logger.info(f"Loaded root hashtable with {len(self.stats_table)} entries")
