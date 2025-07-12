from collections import defaultdict

import numpy as np

from ...dataset import Dataset
from ..evaluator import Evaluator


class DurationEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        # duration distribution by service
        duration_distribution = defaultdict(list)
        duration_pair_distribution = defaultdict(list)
        # depth 0 (root) span duration by service and time bucket for p50/p90 calculation
        depth_0_span_time_durations = defaultdict(lambda: defaultdict(list))

        # Depth 0 (root) span durations before and after incident injection
        duration_depth_0_before_incident = defaultdict(list)
        duration_depth_0_after_incident = defaultdict(list)
        
        # Depth 1 span durations before and after incident injection
        duration_depth_1_before_incident = defaultdict(list)
        duration_depth_1_after_incident = defaultdict(list)

        # Get inject_time from labels (convert from seconds to microseconds)
        inject_time_us = int(labels.get("inject_time", 0)) * 1000000

        def get_span_depth(span_id, trace, depth=0):
            """Calculate the depth of a span in the trace tree"""
            span = trace.get(span_id)
            if not span or span["parentSpanId"] is None:
                return depth
            return get_span_depth(span["parentSpanId"], trace, depth + 1)

        for trace in dataset.traces.values():
            for span in trace.values():
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

            # Process depth 0 (root) spans for time buckets and before/after incident analysis
            for span_id, span in trace.items():
                if get_span_depth(span_id, trace) == 0:
                    service = span["nodeName"].split("@")[0]
                    span_start_time = span["startTime"]
                    span_duration = span["duration"]
                    start_time = span_start_time // (60 * 1000000)  # minute bucket

                    # collect depth 0 span duration by service and time bucket for p50/p90
                    depth_0_span_time_durations[service][start_time].append(span_duration)
                    
                    # Determine if this depth 0 span is before or after incident injection
                    if span_start_time + span_duration < inject_time_us:
                        # Span ended before incident injection
                        duration_depth_0_before_incident["all"].append(span_duration)
                    elif span_start_time >= inject_time_us:
                        # Span started after incident injection
                        duration_depth_0_after_incident["all"].append(span_duration)

            # Process depth 1 spans for before/after incident analysis
            for span_id, span in trace.items():
                if get_span_depth(span_id, trace) == 1:
                    span_start_time = span["startTime"]
                    span_duration = span["duration"]
                    
                    # Determine if this depth 1 span is before or after incident injection
                    if span_start_time + span_duration < inject_time_us:
                        # Span ended before incident injection
                        duration_depth_1_before_incident["all"].append(span_duration)
                    elif span_start_time >= inject_time_us:
                        # Span started after incident injection
                        duration_depth_1_after_incident["all"].append(span_duration)

        # calculate p50 and p90 for each depth 0 span service and time bucket
        duration_depth_0_p50_by_service = {}
        duration_depth_0_p90_by_service = {}
        for service, time_buckets in depth_0_span_time_durations.items():
            duration_depth_0_p50_by_service[service] = []
            duration_depth_0_p90_by_service[service] = []
            for timebucket, durations in time_buckets.items():
                if durations:  # only calculate if there are durations
                    p50 = np.percentile(durations, 50)
                    p90 = np.percentile(durations, 90)
                    count = len(durations)  # Number of traces in this bucket
                    duration_depth_0_p50_by_service[service].append(
                        {"timebucket": timebucket, "p50": p50, "count": count}
                    )
                    duration_depth_0_p90_by_service[service].append(
                        {"timebucket": timebucket, "p90": p90, "count": count}
                    )
            # sort by timebucket for consistent ordering
            duration_depth_0_p50_by_service[service].sort(key=lambda x: x["timebucket"])
            duration_depth_0_p90_by_service[service].sort(key=lambda x: x["timebucket"])

        return {
            "duration": duration_distribution,
            "duration_pair": duration_pair_distribution,
            "duration_depth_0_p50_by_service": duration_depth_0_p50_by_service,
            "duration_depth_0_p90_by_service": duration_depth_0_p90_by_service,
            "duration_depth_0_before_incident": duration_depth_0_before_incident,
            "duration_depth_0_after_incident": duration_depth_0_after_incident,
            "duration_depth_1_before_incident": duration_depth_1_before_incident,
            "duration_depth_1_after_incident": duration_depth_1_after_incident,
        }
