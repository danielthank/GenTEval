from collections import defaultdict

import numpy as np

from genteval.dataset import Dataset
from genteval.evaluators.evaluator import Evaluator


class DurationEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        # duration distribution by service
        duration_distribution = defaultdict(list)
        duration_pair_distribution = defaultdict(list)
        # duration distribution at specific depths (0-4)
        duration_by_depth = {}
        duration_by_depth_by_service = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        # span durations before and after incident injection by depth (0-4)
        duration_before_incident_by_depth = {}
        duration_after_incident_by_depth = {}

        # Initialize for depths 0-4
        for depth in range(5):  # 0, 1, 2, 3, 4
            duration_by_depth[depth] = defaultdict(list)
            duration_before_incident_by_depth[depth] = defaultdict(list)
            duration_after_incident_by_depth[depth] = defaultdict(list)

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
                duration = span["duration"]

                duration_distribution["all"].append(duration)

                if span["parentSpanId"] is not None:
                    parent_span = trace.get(span["parentSpanId"])
                    if parent_span:
                        parent_duration = parent_span["duration"]
                        if parent_duration > 0:
                            # TODO: consider not cliping the duration
                            duration_pair_distribution["all"].append(
                                duration / parent_duration
                                if duration <= parent_duration
                                else 1
                            )

            # Process spans at all depths (0-4) for time buckets and before/after incident analysis
            for span_id, span in trace.items():
                span_depth = get_span_depth(span_id, trace)

                # Only process depths 0-4
                if 0 <= span_depth <= 4:
                    service = span["nodeName"].split("!@#")[0]
                    span_start_time = span["startTime"]
                    span_duration = span["duration"]
                    start_time = span_start_time // (60 * 1000000)  # minute bucket

                    # collect duration distribution for this depth
                    duration_by_depth[span_depth]["all"].append(span_duration)

                    # collect span duration by service and time bucket
                    duration_by_depth_by_service[span_depth][service][
                        start_time
                    ].append(span_duration)

                    # Determine if this span is before or after incident injection
                    if span_start_time + span_duration < inject_time_us:
                        # Span ended before incident injection
                        duration_before_incident_by_depth[span_depth]["all"].append(
                            span_duration
                        )
                    elif span_start_time >= inject_time_us:
                        # Span started after incident injection
                        duration_after_incident_by_depth[span_depth]["all"].append(
                            span_duration
                        )

        # calculate percentiles (p0, p10, p20, ..., p90, p100) for each depth, service and time bucket
        duration_by_depth_by_service_percentiles = {}
        percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        for depth, services in duration_by_depth_by_service.items():
            duration_by_depth_by_service_percentiles[depth] = {}
            for service, time_buckets in services.items():
                duration_by_depth_by_service_percentiles[depth][service] = []
                for timebucket, durations in time_buckets.items():
                    if durations:  # only calculate if there are durations
                        percentile_values = {}
                        for p in percentiles:
                            percentile_values[f"p{p}"] = np.percentile(durations, p)

                        bucket_data = {"timebucket": timebucket, **percentile_values}
                        duration_by_depth_by_service_percentiles[depth][service].append(
                            bucket_data
                        )

                # sort by timebucket for consistent ordering
                duration_by_depth_by_service_percentiles[depth][service].sort(
                    key=lambda x: x["timebucket"]
                )

        # Build return dictionary with all depth data
        result = {
            "duration": duration_distribution,
            "duration_pair": duration_pair_distribution,
            "duration_by_depth_by_service": duration_by_depth_by_service_percentiles,
            "duration_before_incident_by_depth": duration_before_incident_by_depth,
            "duration_after_incident_by_depth": duration_after_incident_by_depth,
        }

        # Add individual depth data for specific depth analysis
        for depth in range(5):  # 0, 1, 2, 3, 4
            result[f"duration_depth_{depth}"] = duration_by_depth[depth]

        return result
