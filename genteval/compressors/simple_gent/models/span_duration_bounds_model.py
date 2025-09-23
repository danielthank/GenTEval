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

        # Storage: {time_bucket: {NodeFeature: (min_duration, max_duration)}}
        self.duration_bounds = defaultdict(dict)

        # Temporary storage during fitting
        self._temp_durations = defaultdict(
            lambda: defaultdict(list)
        )  # {time_bucket: {NodeFeature: [durations]}}

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

            # Process all spans (not just roots)
            for span_id, span_data in trace.spans.items():
                # Count children for this span
                child_count = sum(
                    1 for s in trace.spans.values() if s.get("parentSpanId") == span_id
                )

                node_idx = span_data["nodeIdx"]
                span_feature = NodeFeature(node_idx=node_idx, child_count=child_count)

                duration = span_data["duration"]

                # Skip invalid durations
                if duration <= 0:
                    continue

                self._temp_durations[time_bucket][span_feature].append(duration)

                total_spans += 1
                total_duration_samples += 1

        self.logger.info(
            f"Collected {total_duration_samples} duration samples from {total_spans} spans"
        )
        self.logger.info(
            f"Found duration data for {len(self._temp_durations)} time buckets"
        )

        # Second pass: calculate min/max bounds for each (time_bucket, feature) combination
        self.logger.info("Calculating min/max bounds for each feature")

        # Calculate total work for progress bar
        total_feature_combinations = sum(
            len(bucket_features) for bucket_features in self._temp_durations.values()
        )
        self.logger.info(
            f"Calculating bounds for {total_feature_combinations} unique (time_bucket, feature) combinations"
        )

        processed_combinations = 0

        # Create progress bar for all feature combinations
        with tqdm(
            total=total_feature_combinations, desc="Calculating duration bounds"
        ) as pbar:
            for time_bucket, bucket_features in self._temp_durations.items():
                for feature, durations in bucket_features.items():
                    min_duration = min(durations)
                    max_duration = max(durations)

                    self.duration_bounds[time_bucket][feature] = (
                        min_duration,
                        max_duration,
                    )
                    processed_combinations += 1

                    pbar.update(1)

        # Clear temporary storage
        self._temp_durations.clear()

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
            return self.duration_bounds[time_bucket][feature]

        # Fallback 1: same feature in closest time bucket
        if self.duration_bounds:
            for bucket_time in sorted(
                self.duration_bounds.keys(), key=lambda t: abs(t - time_bucket)
            ):
                if feature in self.duration_bounds[bucket_time]:
                    return self.duration_bounds[bucket_time][feature]

        # Fallback 2: same node name with any child count in closest time bucket
        if self.duration_bounds:
            for bucket_time in sorted(
                self.duration_bounds.keys(), key=lambda t: abs(t - time_bucket)
            ):
                for candidate_feature in self.duration_bounds[bucket_time]:
                    if candidate_feature.name == feature.name:
                        return self.duration_bounds[bucket_time][candidate_feature]

        # Fallback 3: any bounds in closest time bucket
        if self.duration_bounds:
            closest_bucket = min(
                self.duration_bounds.keys(), key=lambda t: abs(t - time_bucket)
            )
            if self.duration_bounds[closest_bucket]:
                # Use first available bounds
                return next(iter(self.duration_bounds[closest_bucket].values()))

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

            for feature, (min_dur, max_dur) in bucket_bounds.items():
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

        self.duration_bounds = defaultdict(dict)

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

                self.duration_bounds[time_bucket][feature] = (min_dur, max_dur)

        total_bounds = sum(len(bucket) for bucket in self.duration_bounds.values())
        self.logger.info(
            f"Loaded {total_bounds} duration bounds across {len(self.duration_bounds)} time buckets"
        )
