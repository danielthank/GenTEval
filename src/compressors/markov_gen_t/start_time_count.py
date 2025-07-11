import logging
from collections import Counter
from typing import Dict

import numpy as np

from .. import CompressedDataset, SerializationFormat


class StartTimeCountSynthesizer:
    """
    Simple start time synthesizer based on time bucket distribution.

    This approach:
    1. Counts occurrences in each time bucket (minute-level)
    2. Samples time buckets according to their probability distribution
    3. Uses uniform distribution within each selected time bucket
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Store time bucket distribution
        self.bucket_counts: Dict[int, int] = {}
        self.bucket_probabilities: Dict[int, float] = {}
        self.total_samples = 0

        # Time bucket size in microseconds (1 minute)
        self.bucket_size_us = 60 * 1000000

    def _get_time_bucket(self, start_time: int) -> int:
        """Convert start time to time bucket (minute-level)."""
        return start_time // self.bucket_size_us

    def _get_bucket_start_time(self, bucket: int) -> int:
        """Get the start time of a time bucket."""
        return bucket * self.bucket_size_us

    def fit(self, start_times: np.ndarray):
        """Learn the time bucket distribution from training data."""
        self.logger.info("Learning time bucket distribution")

        # Convert start times to time buckets
        time_buckets = [
            self._get_time_bucket(int(start_time)) for start_time in start_times
        ]

        # Count occurrences in each bucket
        self.bucket_counts = dict(Counter(time_buckets))
        self.total_samples = len(start_times)

        # Calculate probabilities
        self.bucket_probabilities = {
            bucket: count / self.total_samples
            for bucket, count in self.bucket_counts.items()
        }

        self.logger.info(
            f"Learned distribution over {len(self.bucket_counts)} time buckets"
        )
        self.logger.info(
            f"Time range: {min(self.bucket_counts.keys())} to {max(self.bucket_counts.keys())} (buckets)"
        )

    def save_state_dict(
        self, compressed_data: CompressedDataset, decoder_only: bool = False
    ):
        """Save the learned time bucket distribution."""

        compressed_data.add(
            "start_time_count_synthesizer",
            CompressedDataset(
                data={
                    "bucket_counts": (
                        self.bucket_counts,
                        SerializationFormat.MSGPACK,
                    ),
                    "bucket_probabilities": (
                        self.bucket_probabilities,
                        SerializationFormat.MSGPACK,
                    ),
                    "total_samples": (
                        self.total_samples,
                        SerializationFormat.MSGPACK,
                    ),
                    "bucket_size_us": (
                        self.bucket_size_us,
                        SerializationFormat.MSGPACK,
                    ),
                }
            ),
            SerializationFormat.CLOUDPICKLE,
        )

    def load_state_dict(self, compressed_dataset):
        """Load the learned time bucket distribution."""
        if "start_time_count_synthesizer" not in compressed_dataset:
            raise ValueError(
                "No start_time_count_synthesizer found in compressed dataset"
            )

        logger = logging.getLogger(__name__)
        logger.info("Loading start time count synthesizer")

        # Load synthesizer data
        synthesizer_data = compressed_dataset["start_time_count_synthesizer"]

        self.bucket_counts = synthesizer_data["bucket_counts"]
        self.bucket_probabilities = synthesizer_data["bucket_probabilities"]
        self.total_samples = synthesizer_data["total_samples"]
        self.bucket_size_us = synthesizer_data["bucket_size_us"]

        logger.info(f"Loaded distribution over {len(self.bucket_counts)} time buckets")

    def sample(self, num_samples: int) -> np.ndarray:
        """Generate new start times based on learned time bucket distribution."""
        if not self.bucket_probabilities:
            raise ValueError("Model not trained. Call fit() first.")

        # Extract buckets and their probabilities
        buckets = list(self.bucket_probabilities.keys())
        probabilities = list(self.bucket_probabilities.values())

        # Sample time buckets according to learned distribution
        sampled_buckets = np.random.choice(buckets, size=num_samples, p=probabilities)

        # Generate uniform start times within each selected bucket
        start_times = []
        for bucket in sampled_buckets:
            bucket_start = self._get_bucket_start_time(bucket)
            bucket_end = bucket_start + self.bucket_size_us

            # Uniform distribution within the bucket
            start_time = np.random.uniform(bucket_start, bucket_end)
            start_times.append(int(start_time))

        return np.array(start_times)

    def get_bucket_statistics(self) -> Dict:
        """Get statistics about the learned time bucket distribution."""
        if not self.bucket_counts:
            return {}

        return {
            "num_buckets": len(self.bucket_counts),
            "total_samples": self.total_samples,
            "min_bucket": min(self.bucket_counts.keys()),
            "max_bucket": max(self.bucket_counts.keys()),
            "min_count": min(self.bucket_counts.values()),
            "max_count": max(self.bucket_counts.values()),
            "avg_count": sum(self.bucket_counts.values()) / len(self.bucket_counts),
            "bucket_coverage": len(self.bucket_counts)
            / (max(self.bucket_counts.keys()) - min(self.bucket_counts.keys()) + 1),
        }
