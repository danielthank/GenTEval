from collections import defaultdict

from genteval.dataset import Dataset
from genteval.evaluators.evaluator import Evaluator


class CountOverTimeEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        # [group][time_bucket] = count
        span_count_by_time = defaultdict(lambda: defaultdict(int))

        # Get inject_time from labels (convert from seconds to microseconds)
        # Note: inject_time_us could be used for filtering spans before/after incident
        # inject_time_us = int(labels.get("inject_time", 0)) * 1000000

        def get_span_depth(span_id, trace, depth=0):
            """Calculate the depth of a span in the trace tree"""
            span = trace.get(span_id)
            if not span or span["parentSpanId"] is None:
                return depth
            return get_span_depth(span["parentSpanId"], trace, depth + 1)

        for trace in dataset.traces.values():
            for span_id, span in trace.items():
                span_depth = get_span_depth(span_id, trace)

                if 0 <= span_depth <= 4:
                    span_start_time = span["startTime"]
                    start_time = span_start_time // (60 * 1000000)  # minute bucket

                    # Count spans by depth and time bucket
                    span_count_by_time[f"depth_{span_depth}"][start_time] += 1

                    # Also count in "all" category
                    span_count_by_time["all"][start_time] += 1

        return {
            "span_count_by_time": dict(span_count_by_time),
        }
