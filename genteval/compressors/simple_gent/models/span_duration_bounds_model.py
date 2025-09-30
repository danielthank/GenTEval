import logging
from collections import defaultdict

from tqdm import tqdm

from genteval.compressors.simple_gent.proto import simple_gent_pb2

from .node_feature import NodeFeature


class SpanDurationBoundsModel:
    """
    Span duration bounds model: (node_idx, child_count) -> (min_duration, max_duration)

    Lightweight model that tracks min/max duration bounds for all spans (not just roots)
    organized by time buckets and using NodeFeature for efficient reject sampling.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Storage: {time_bucket: {NodeFeature: {'min': min_duration, 'max': max_duration}}}
        self.duration_bounds = defaultdict(
            lambda: defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})
        )

    def fit(self, traces):
        """Learn duration bounds for all spans from traces."""
        self.logger.info(f"Training span duration bounds model on {len(traces)} traces")

        # First pass: collect duration samples for all spans
        self.logger.info("Collecting all span duration samples")
        total_spans = 0
        total_duration_samples = 0

        for trace in tqdm(traces, desc="Collecting duration samples"):
            trace_start_time = trace.start_time
            time_bucket = int(trace_start_time // self.config.time_bucket_duration_us)

            # Build parent-to-children count mapping once per trace (O(n))
            child_counts = defaultdict(int)
            for span_data in trace.spans.values():
                parent_id = span_data.get("parentSpanId")
                if parent_id:
                    child_counts[parent_id] += 1

            # Process all spans (not just roots)
            for span_id, span_data in trace.spans.items():
                child_count = child_counts[span_id]

                node_idx = span_data["nodeIdx"]
                span_feature = NodeFeature(node_idx=node_idx, child_count=child_count)

                duration = span_data["duration"]

                # Skip invalid durations
                if duration <= 0:
                    continue

                # Update min/max bounds directly in final storage
                bounds = self.duration_bounds[time_bucket][span_feature]
                bounds["min"] = min(bounds["min"], duration)
                bounds["max"] = max(bounds["max"], duration)

                total_spans += 1
                total_duration_samples += 1

        self.logger.info(
            f"Collected {total_duration_samples} duration samples from {total_spans} spans"
        )
        self.logger.info(
            f"Found duration data for {len(self.duration_bounds)} time buckets"
        )

        total_bounds = sum(len(bucket) for bucket in self.duration_bounds.values())
        self.logger.info(
            f"Duration bounds calculation complete: {total_bounds} bounds across {len(self.duration_bounds)} time buckets"
        )

    def get_duration_bounds(
        self, time_bucket: int, feature: NodeFeature
    ) -> tuple[float, float]:
        """Get duration bounds for a given time bucket and feature with fallbacks."""
        # Direct lookup
        if (
            time_bucket in self.duration_bounds
            and feature in self.duration_bounds[time_bucket]
        ):
            bounds = self.duration_bounds[time_bucket][feature]
            return bounds["min"], bounds["max"]

        # Fallback 1: same feature in closest time bucket
        if self.duration_bounds:
            for bucket_time in sorted(
                self.duration_bounds.keys(), key=lambda t: abs(t - time_bucket)
            ):
                if feature in self.duration_bounds[bucket_time]:
                    bounds = self.duration_bounds[bucket_time][feature]
                    return bounds["min"], bounds["max"]

        # Fallback 2: same node name with any child count in closest time bucket
        if self.duration_bounds:
            for bucket_time in sorted(
                self.duration_bounds.keys(), key=lambda t: abs(t - time_bucket)
            ):
                for candidate_feature in self.duration_bounds[bucket_time]:
                    if candidate_feature.name == feature.name:
                        bounds = self.duration_bounds[bucket_time][candidate_feature]
                        return bounds["min"], bounds["max"]

        # Fallback 3: any bounds in closest time bucket
        if self.duration_bounds:
            closest_bucket = min(
                self.duration_bounds.keys(), key=lambda t: abs(t - time_bucket)
            )
            if self.duration_bounds[closest_bucket]:
                # Use first available bounds
                bounds = next(iter(self.duration_bounds[closest_bucket].values()))
                return bounds["min"], bounds["max"]

        # Final fallback: default wide bounds
        return (1.0, float("inf"))

    def get_duration_bounds_batch(
        self, time_buckets: list[int], features: list[NodeFeature]
    ) -> list[tuple[float, float]]:
        """Get duration bounds for multiple (time_bucket, feature) pairs."""
        if len(time_buckets) != len(features):
            raise ValueError("time_buckets and features must have the same length")

        results = []
        for time_bucket, feature in zip(time_buckets, features, strict=False):
            bounds = self.get_duration_bounds(time_bucket, feature)
            results.append(bounds)

        return results

    def save_state_dict(self, proto_models):
        """Save model state to protobuf message."""

        # Group data by time buckets
        time_bucket_data = {}
        for time_bucket, bucket_bounds in self.duration_bounds.items():
            if time_bucket not in time_bucket_data:
                time_bucket_data[time_bucket] = []

            for feature, bounds_info in bucket_bounds.items():
                min_dur = bounds_info["min"]
                max_dur = bounds_info["max"]
                span_duration_bounds = simple_gent_pb2.SpanDurationBounds()
                span_duration_bounds.feature.node_idx = feature.node_idx
                span_duration_bounds.feature.child_count = feature.child_count
                span_duration_bounds.min_duration = min_dur
                span_duration_bounds.max_duration = max_dur

                time_bucket_data[time_bucket].append(span_duration_bounds)

        # Add to protobuf message
        for time_bucket, span_duration_bounds_list in time_bucket_data.items():
            # Find or create time bucket
            bucket_models = None
            for tb in proto_models.time_buckets:
                if tb.time_bucket == time_bucket:
                    bucket_models = tb
                    break

            if bucket_models is None:
                bucket_models = proto_models.time_buckets.add()
                bucket_models.time_bucket = time_bucket

            # Add span duration bounds to this bucket
            bucket_models.span_duration_bounds.extend(span_duration_bounds_list)

    def load_state_dict(self, proto_models):
        """Load model state from protobuf message."""

        self.duration_bounds = defaultdict(
            lambda: defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})
        )

        # Load from protobuf message
        for time_bucket_models in proto_models.time_buckets:
            time_bucket = time_bucket_models.time_bucket

            for span_duration_bounds in time_bucket_models.span_duration_bounds:
                feature = NodeFeature(
                    node_idx=span_duration_bounds.feature.node_idx,
                    child_count=span_duration_bounds.feature.child_count,
                )

                min_dur = span_duration_bounds.min_duration
                max_dur = span_duration_bounds.max_duration

                self.duration_bounds[time_bucket][feature] = {
                    "min": min_dur,
                    "max": max_dur,
                }

        total_bounds = sum(len(bucket) for bucket in self.duration_bounds.values())
        self.logger.info(
            f"Loaded {total_bounds} duration bounds across {len(self.duration_bounds)} time buckets"
        )
