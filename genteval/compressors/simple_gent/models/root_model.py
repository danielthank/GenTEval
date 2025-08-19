import logging
from collections import defaultdict

import numpy as np

from genteval.compressors.simple_gent.proto import simple_gent_pb2

from .node_feature import NodeFeature


class RootModel:
    """
    Root model: (root_node_idx, root_child_cnt) -> number

    Models the count/frequency of root nodes with specific features
    within each time bucket.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Storage: {time_bucket: {NodeFeature: count}}
        self.root_counts = defaultdict(lambda: defaultdict(int))

        # Total counts per time bucket for sampling probabilities
        self.total_counts = defaultdict(int)

    def fit(self, traces):
        """Learn root node counts from traces."""
        self.logger.info("Training root model")

        root_features_per_bucket = defaultdict(list)

        # Extract root nodes from each trace
        for trace in traces:
            trace_start_time = trace.start_time
            time_bucket = int(trace_start_time // self.config.time_bucket_duration_us)

            # Find root spans (spans with no parent)
            for span_id, span_data in trace.spans.items():
                if span_data.get("parentSpanId") is None:
                    # Count children of this root span
                    child_count = sum(
                        1
                        for s in trace.spans.values()
                        if s.get("parentSpanId") == span_id
                    )

                    node_idx = span_data["nodeIdx"]
                    root_feature = NodeFeature(
                        node_idx=node_idx, child_count=child_count
                    )

                    root_features_per_bucket[time_bucket].append(root_feature)

        # Count occurrences per time bucket
        for time_bucket, features in root_features_per_bucket.items():
            for feature in features:
                self.root_counts[time_bucket][feature] += 1
                self.total_counts[time_bucket] += 1

        self.logger.info(
            f"Learned root counts for {len(self.root_counts)} time buckets, "
            f"total roots: {sum(self.total_counts.values())}"
        )

    def sample_root_features(self, count: int) -> list[tuple[int, NodeFeature]]:
        """Sample root node features with their time buckets.

        Returns:
            List of (time_bucket, NodeFeature) tuples
        """
        if not self.total_counts:
            # No data - return default
            default_time_bucket = 0
            fallback_node_idx = 0
            return [
                (
                    default_time_bucket,
                    NodeFeature(node_idx=fallback_node_idx, child_count=0),
                )
                for _ in range(count)
            ]

        # First, sample time buckets based on total_counts distribution
        time_buckets = list(self.total_counts.keys())
        time_bucket_weights = list(self.total_counts.values())

        # Normalize to probabilities
        time_bucket_weights = np.array(time_bucket_weights, dtype=float)
        time_bucket_probs = time_bucket_weights / time_bucket_weights.sum()

        # Sample time buckets for each trace
        sampled_time_buckets = np.random.choice(
            time_buckets, size=count, p=time_bucket_probs, replace=True
        )

        results = []
        for time_bucket in sampled_time_buckets:
            # Sample a feature from this time bucket
            if self.root_counts.get(time_bucket):
                features = list(self.root_counts[time_bucket].keys())
                weights = list(self.root_counts[time_bucket].values())

                # Normalize weights to probabilities
                weights = np.array(weights, dtype=float)
                probabilities = weights / weights.sum()

                # Sample one feature
                sampled_idx = np.random.choice(len(features), p=probabilities)
                sampled_feature = features[sampled_idx]
            else:
                # Fallback - use index 0
                fallback_node_idx = 0
                sampled_feature = NodeFeature(node_idx=fallback_node_idx, child_count=0)

            results.append((time_bucket, sampled_feature))

        return results

    def save_state_dict(self, proto_models):
        """Save model state to protobuf message."""

        # Group data by time buckets
        time_bucket_data = {}
        for time_bucket, bucket_counts in self.root_counts.items():
            if time_bucket not in time_bucket_data:
                time_bucket_data[time_bucket] = []

            for feature, count in bucket_counts.items():
                root_count_model = simple_gent_pb2.RootCountModel()
                root_count_model.feature.node_idx = feature.node_idx
                root_count_model.feature.child_count = feature.child_count
                root_count_model.count = count
                time_bucket_data[time_bucket].append(root_count_model)

        # Add to protobuf message
        for time_bucket, root_models in time_bucket_data.items():
            # Find or create time bucket
            bucket_models = None
            for tb in proto_models.time_buckets:
                if tb.time_bucket == time_bucket:
                    bucket_models = tb
                    break

            if bucket_models is None:
                bucket_models = proto_models.time_buckets.add()
                bucket_models.time_bucket = time_bucket

            # Add root models to this bucket
            bucket_models.root_models.extend(root_models)

    def load_state_dict(self, proto_models):
        """Load model state from protobuf message."""
        self.root_counts = defaultdict(lambda: defaultdict(int))
        self.total_counts = defaultdict(int)

        # Load from protobuf message
        for time_bucket_models in proto_models.time_buckets:
            time_bucket = time_bucket_models.time_bucket

            for root_model in time_bucket_models.root_models:
                feature = NodeFeature(
                    node_idx=root_model.feature.node_idx,
                    child_count=root_model.feature.child_count,
                )
                self.root_counts[time_bucket][feature] = root_model.count
                self.total_counts[time_bucket] += root_model.count

        self.logger.info(f"Loaded root model with {len(self.root_counts)} time buckets")
