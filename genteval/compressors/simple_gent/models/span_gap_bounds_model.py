import logging
from collections import defaultdict

from tqdm import tqdm

from genteval.compressors.simple_gent.proto import simple_gent_pb2

from .node_feature import NodeFeature


class SpanGapBoundsModel:
    """
    Span gap bounds model: (node_idx, child_count) -> (min_gap, max_gap)

    Lightweight model that tracks min/max gap_from_parent bounds for all child spans
    organized by time buckets and using NodeFeature for efficient reject sampling.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Storage: {time_bucket: {NodeFeature: {'min': min_gap, 'max': max_gap}}}
        self.gap_bounds = defaultdict(
            lambda: defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})
        )

    def fit(self, traces):
        """Learn gap_from_parent bounds for all child spans from traces."""
        self.logger.info(f"Training span gap bounds model on {len(traces)} traces")

        # First pass: collect gap samples for all child spans
        self.logger.info("Collecting all span gap samples")
        total_spans = 0
        total_gap_samples = 0

        for trace in tqdm(traces, desc="Collecting gap samples"):
            trace_start_time = trace.start_time
            time_bucket = int(trace_start_time // self.config.time_bucket_duration_us)

            # Build parent-to-children count mapping once per trace (O(n))
            child_counts = defaultdict(int)
            for span_data in trace.spans.values():
                parent_id = span_data.get("parentSpanId")
                if parent_id:
                    child_counts[parent_id] += 1

            # Process all child spans (spans with parents)
            for span_id, span_data in trace.spans.items():
                parent_id = span_data.get("parentSpanId")

                # Skip root spans (no parent = no gap)
                if parent_id is None or parent_id not in trace.spans:
                    continue

                child_count = child_counts[span_id]
                node_idx = span_data["nodeIdx"]
                span_feature = NodeFeature(
                    node_idx=node_idx, child_idx=0, child_count=child_count
                )

                # Calculate gap_from_parent
                parent_data = trace.spans[parent_id]
                parent_start_time = parent_data["startTime"]
                child_start_time = span_data["startTime"]
                gap_from_parent = child_start_time - parent_start_time

                # Skip invalid gaps (negative gaps are possible in malformed data)
                if gap_from_parent < 0:
                    continue

                # Update min/max bounds directly in final storage
                bounds = self.gap_bounds[time_bucket][span_feature]
                bounds["min"] = min(bounds["min"], gap_from_parent)
                bounds["max"] = max(bounds["max"], gap_from_parent)

                total_spans += 1
                total_gap_samples += 1

        self.logger.info(
            f"Collected {total_gap_samples} gap samples from {total_spans} child spans"
        )
        self.logger.info(f"Found gap data for {len(self.gap_bounds)} time buckets")

        total_bounds = sum(len(bucket) for bucket in self.gap_bounds.values())
        self.logger.info(
            f"Gap bounds calculation complete: {total_bounds} bounds across {len(self.gap_bounds)} time buckets"
        )

    def get_gap_bounds(
        self, time_bucket: int, feature: NodeFeature
    ) -> tuple[float, float]:
        """Get gap bounds for a given time bucket and feature with fallbacks."""
        # Direct lookup
        if time_bucket in self.gap_bounds and feature in self.gap_bounds[time_bucket]:
            bounds = self.gap_bounds[time_bucket][feature]
            return bounds["min"], bounds["max"]

        # Fallback 1: same feature in closest time bucket
        if self.gap_bounds:
            for bucket_time in sorted(
                self.gap_bounds.keys(), key=lambda t: abs(t - time_bucket)
            ):
                if feature in self.gap_bounds[bucket_time]:
                    bounds = self.gap_bounds[bucket_time][feature]
                    return bounds["min"], bounds["max"]

        # Fallback 2: same node index with any child count in closest time bucket
        if self.gap_bounds:
            for bucket_time in sorted(
                self.gap_bounds.keys(), key=lambda t: abs(t - time_bucket)
            ):
                for candidate_feature in self.gap_bounds[bucket_time]:
                    if candidate_feature.node_idx == feature.node_idx:
                        bounds = self.gap_bounds[bucket_time][candidate_feature]
                        return bounds["min"], bounds["max"]

        # Fallback 3: any bounds in closest time bucket
        if self.gap_bounds:
            closest_bucket = min(
                self.gap_bounds.keys(), key=lambda t: abs(t - time_bucket)
            )
            if self.gap_bounds[closest_bucket]:
                # Use first available bounds
                bounds = next(iter(self.gap_bounds[closest_bucket].values()))
                return bounds["min"], bounds["max"]

        # Final fallback: default wide bounds (0 to very large gap)
        return (0.0, float("inf"))

    def save_state_dict(self, proto_models):
        """Save model state to protobuf message."""

        # Group data by time buckets
        time_bucket_data = {}
        for time_bucket, bucket_bounds in self.gap_bounds.items():
            if time_bucket not in time_bucket_data:
                time_bucket_data[time_bucket] = []

            for feature, bounds_info in bucket_bounds.items():
                min_gap = bounds_info["min"]
                max_gap = bounds_info["max"]
                span_gap_bounds = simple_gent_pb2.SpanGapBounds()
                span_gap_bounds.feature.node_idx = feature.node_idx
                span_gap_bounds.feature.child_idx = feature.child_idx
                span_gap_bounds.feature.child_count = feature.child_count
                span_gap_bounds.min_gap = min_gap
                span_gap_bounds.max_gap = max_gap

                time_bucket_data[time_bucket].append(span_gap_bounds)

        # Add to protobuf message
        for time_bucket, span_gap_bounds_list in time_bucket_data.items():
            # Find or create time bucket
            bucket_models = None
            for tb in proto_models.time_buckets:
                if tb.time_bucket == time_bucket:
                    bucket_models = tb
                    break

            if bucket_models is None:
                bucket_models = proto_models.time_buckets.add()
                bucket_models.time_bucket = time_bucket

            # Add span gap bounds to this bucket
            bucket_models.span_gap_bounds.extend(span_gap_bounds_list)

    def load_state_dict(self, proto_models):
        """Load model state from protobuf message."""

        self.gap_bounds = defaultdict(
            lambda: defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})
        )

        # Load from protobuf message
        for time_bucket_models in proto_models.time_buckets:
            time_bucket = time_bucket_models.time_bucket

            for span_gap_bounds in time_bucket_models.span_gap_bounds:
                feature = NodeFeature(
                    node_idx=span_gap_bounds.feature.node_idx,
                    child_idx=span_gap_bounds.feature.child_idx,
                    child_count=span_gap_bounds.feature.child_count,
                )

                min_gap = span_gap_bounds.min_gap
                max_gap = span_gap_bounds.max_gap

                self.gap_bounds[time_bucket][feature] = {
                    "min": min_gap,
                    "max": max_gap,
                }

        total_bounds = sum(len(bucket) for bucket in self.gap_bounds.values())
        self.logger.info(
            f"Loaded {total_bounds} gap bounds across {len(self.gap_bounds)} time buckets"
        )
