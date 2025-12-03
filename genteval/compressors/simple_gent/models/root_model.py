import logging
from collections import defaultdict
from datetime import datetime

import numpy as np
from tqdm import tqdm

from genteval.compressors.simple_gent.proto import simple_gent_pb2

from .node_feature import NodeFeature
from .trace_type import TraceType


class RootModel:
    """
    Root model: (root_node_idx, root_child_cnt) -> number

    Models the count/frequency of root nodes with specific features
    within each time bucket, stratified by trace type (normal vs error).
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Storage: {TraceType: {time_bucket: {NodeFeature: count}}}
        self.root_counts = {
            TraceType.NORMAL: defaultdict(lambda: defaultdict(int)),
            TraceType.ERROR: defaultdict(lambda: defaultdict(int)),
        }

        # Total counts per TraceType and time bucket for sampling probabilities
        self.total_counts = {
            TraceType.NORMAL: defaultdict(int),
            TraceType.ERROR: defaultdict(int),
        }

    def fit(self, traces):
        """Learn root node counts from traces, stratified by trace type."""
        self.logger.info(f"Training root model on {len(traces)} traces")

        # {trace_type: {time_bucket: [features]}}
        root_features_per_bucket = {
            TraceType.NORMAL: defaultdict(list),
            TraceType.ERROR: defaultdict(list),
        }
        total_root_spans = 0
        min_start_time = float("inf")
        max_start_time = float("-inf")
        normal_trace_count = 0
        error_trace_count = 0

        # Extract root nodes from each trace
        self.logger.info("Extracting root spans from traces (stratified by trace type)")
        for trace in tqdm(traces, desc="Processing traces for root features"):
            trace_start_time = trace.start_time
            time_bucket = int(trace_start_time // self.config.time_bucket_duration_us)

            # Track min/max start times
            min_start_time = min(min_start_time, trace_start_time)
            max_start_time = max(max_start_time, trace_start_time)

            trace_type = TraceType.from_trace(trace, self.config.stratified_sampling)
            if trace_type == TraceType.ERROR:
                error_trace_count += 1
            else:
                normal_trace_count += 1

            # Find root spans (spans with no parent or missing parent)
            trace_root_count = 0
            for span_id, span_data in trace.spans.items():
                parent_id = span_data.get("parentSpanId")

                # Treat as root if: no parent, or parent doesn't exist in trace
                if parent_id is None or parent_id not in trace.spans:
                    # Count children of this root span
                    child_count = sum(
                        1
                        for s in trace.spans.values()
                        if s.get("parentSpanId") == span_id
                    )

                    node_idx = span_data["nodeIdx"]
                    root_feature = NodeFeature(
                        node_idx=node_idx, child_idx=0, child_count=child_count
                    )

                    root_features_per_bucket[trace_type][time_bucket].append(
                        root_feature
                    )
                    trace_root_count += 1

            total_root_spans += trace_root_count

        # Output min/max start times
        if min_start_time != float("inf") and max_start_time != float("-inf"):
            min_dt = datetime.fromtimestamp(min_start_time / 1_000_000)
            max_dt = datetime.fromtimestamp(max_start_time / 1_000_000)
            self.logger.info(f"Min start_time: {min_start_time} ({min_dt})")
            self.logger.info(f"Max start_time: {max_start_time} ({max_dt})")
        else:
            self.logger.info("No traces processed - min/max start_time not available")

        self.logger.info(
            f"Extracted {total_root_spans} root spans from {len(traces)} traces"
        )
        self.logger.info(
            f"Trace type distribution: {normal_trace_count} normal, {error_trace_count} error"
        )

        # Count occurrences per trace type and time bucket
        self.logger.info("Computing root feature counts per trace type and time bucket")
        total_unique_features = 0

        for trace_type in [TraceType.NORMAL, TraceType.ERROR]:
            for time_bucket, features in tqdm(
                root_features_per_bucket[trace_type].items(),
                desc=f"Processing {trace_type} time buckets",
            ):
                bucket_feature_count = 0
                for feature in features:
                    self.root_counts[trace_type][time_bucket][feature] += 1
                    self.total_counts[trace_type][time_bucket] += 1
                    bucket_feature_count += 1

                unique_features_in_bucket = len(
                    self.root_counts[trace_type][time_bucket]
                )
                total_unique_features += unique_features_in_bucket
                self.logger.debug(
                    f"[{trace_type}] Time bucket {time_bucket}: {bucket_feature_count} root spans, "
                    f"{unique_features_in_bucket} unique features"
                )

        normal_buckets = len(self.root_counts[TraceType.NORMAL])
        error_buckets = len(self.root_counts[TraceType.ERROR])
        normal_total = sum(self.total_counts[TraceType.NORMAL].values())
        error_total = sum(self.total_counts[TraceType.ERROR].values())

        self.logger.info(
            f"Root model training complete: "
            f"normal={normal_buckets} buckets/{normal_total} spans, "
            f"error={error_buckets} buckets/{error_total} spans, "
            f"{total_unique_features} total unique features"
        )

    def sample_root_features(self) -> list[tuple[int, NodeFeature]]:
        """Get all root node features with their time buckets (combines all trace types).

        Returns:
            List of (time_bucket, NodeFeature) tuples
        """
        stratified_results = self.sample_root_features_stratified()
        return [
            (time_bucket, feature) for time_bucket, feature, _ in stratified_results
        ]

    def sample_root_features_stratified(self) -> list[tuple[int, NodeFeature, str]]:
        """Get all root node features with exact counts from training data.

        Returns exact counts for each (time_bucket, feature, trace_type) combination,
        guaranteeing faithful reproduction of the original distribution.

        Returns:
            List of (time_bucket, NodeFeature, trace_type) tuples
        """
        results = []

        for trace_type in [TraceType.NORMAL, TraceType.ERROR]:
            for time_bucket, feature_counts in self.root_counts[trace_type].items():
                for feature, count in feature_counts.items():
                    for _ in range(count):
                        results.append((time_bucket, feature, trace_type))

        total_normal = sum(self.total_counts[TraceType.NORMAL].values())
        total_error = sum(self.total_counts[TraceType.ERROR].values())
        self.logger.debug(
            f"Returning exact counts: {total_normal} normal, {total_error} error"
        )

        np.random.shuffle(results)
        return results

    def save_state_dict(self, proto_models):
        """Save model state to protobuf message."""

        # Group data by time buckets
        time_bucket_data = {}

        for trace_type in [TraceType.NORMAL, TraceType.ERROR]:
            proto_trace_type = trace_type.to_proto()

            for time_bucket, bucket_counts in self.root_counts[trace_type].items():
                if time_bucket not in time_bucket_data:
                    time_bucket_data[time_bucket] = []

                for feature, count in bucket_counts.items():
                    root_count_model = simple_gent_pb2.RootCountModel()
                    root_count_model.feature.node_idx = feature.node_idx
                    root_count_model.feature.child_idx = feature.child_idx
                    root_count_model.feature.child_count = feature.child_count
                    root_count_model.count = count
                    root_count_model.trace_type = proto_trace_type
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
        self.root_counts = {
            TraceType.NORMAL: defaultdict(lambda: defaultdict(int)),
            TraceType.ERROR: defaultdict(lambda: defaultdict(int)),
        }
        self.total_counts = {
            TraceType.NORMAL: defaultdict(int),
            TraceType.ERROR: defaultdict(int),
        }

        # Load from protobuf message
        for time_bucket_models in proto_models.time_buckets:
            time_bucket = time_bucket_models.time_bucket

            for root_model in time_bucket_models.root_models:
                feature = NodeFeature(
                    node_idx=root_model.feature.node_idx,
                    child_idx=root_model.feature.child_idx,
                    child_count=root_model.feature.child_count,
                )

                trace_type = TraceType.from_proto(root_model.trace_type)

                self.root_counts[trace_type][time_bucket][feature] = root_model.count
                self.total_counts[trace_type][time_bucket] += root_model.count

        normal_buckets = len(self.root_counts[TraceType.NORMAL])
        error_buckets = len(self.root_counts[TraceType.ERROR])
        self.logger.info(
            f"Loaded root model with {normal_buckets} normal buckets, "
            f"{error_buckets} error buckets"
        )
