from collections import defaultdict

import numpy as np

from genteval.dataset import Dataset
from genteval.evaluators.evaluator import Evaluator


class DurationEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        # [group] = [..durations]
        duration = defaultdict(list)

        # [group][time_buckett] = [..durations]
        duration_by_time = defaultdict(lambda: defaultdict(list))

        # [group][time_bucket][percentile] = value
        duration_by_time_percentiles = defaultdict(lambda: defaultdict(dict))

        # Get inject_time from labels (convert from seconds to microseconds)
        inject_time_us = int(labels.get("inject_time", 0)) * 1000000

        def get_span_depth(span_id, trace, depth=0):
            """Calculate the depth of a span in the trace tree"""
            span = trace.get(span_id)
            if not span or span["parentSpanId"] is None:
                return depth
            return get_span_depth(span["parentSpanId"], trace, depth + 1)

        def get_child_count(span_id, trace):
            """Count the number of direct children for a span"""
            child_count = 0
            for other_span in trace.values():
                if other_span["parentSpanId"] == span_id:
                    child_count += 1
            return child_count

        for trace in dataset.traces.values():
            for span_id, span in trace.items():
                span_duration = span["duration"]

                duration["all"].append(span_duration)

                if span["parentSpanId"] is not None:
                    parent_span = trace.get(span["parentSpanId"])
                    if parent_span:
                        parent_duration = parent_span["duration"]
                        if parent_duration > 0:
                            # TODO: consider not cliping the duration
                            duration["pair_all"].append(
                                span_duration / parent_duration
                                if span_duration <= parent_duration
                                else 1
                            )

                span_depth = get_span_depth(span_id, trace)

                if 0 <= span_depth <= 4:
                    service = span["nodeName"].split("!@#")[0]
                    span_start_time = span["startTime"]
                    span_duration = span["duration"]
                    start_time = span_start_time // (60 * 1000000)  # minute bucket

                    duration[f"depth_{span_depth}"].append(span_duration)

                    if (
                        span_start_time + span_duration < inject_time_us
                        or span_start_time >= inject_time_us
                    ):
                        duration[f"depth_{span_depth}_service_{service}"].append(
                            span_duration
                        )

                    duration_by_time[f"depth_{span_depth}_service_{service}"][
                        start_time
                    ].append(span_duration)
                    duration_by_time[f"depth_{span_depth}"][start_time].append(
                        span_duration
                    )

        percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        for group, time_buckets in duration_by_time.items():
            for time_bucket, distribution in time_buckets.items():
                for p in percentiles:
                    duration_by_time_percentiles[group][time_bucket][f"p{p}"] = (
                        np.percentile(distribution, p)
                    )

        return {
            "duration": duration,
            "duration_by_time_percentiles": duration_by_time_percentiles,
        }
