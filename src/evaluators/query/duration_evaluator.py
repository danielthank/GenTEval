from collections import defaultdict

import numpy as np

from dataset import Dataset
from evaluators import Evaluator


class DurationEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        # duration distribution by service
        duration_distribution = defaultdict(list)
        duration_pair_distribution = defaultdict(list)
        # duration by service and time bucket for p90 calculation
        service_time_durations = defaultdict(lambda: defaultdict(list))

        # NEW: Root span durations before and after incident injection
        root_duration_before_incident = defaultdict(list)
        root_duration_after_incident = defaultdict(list)

        # Get inject_time from labels (convert from seconds to microseconds)
        inject_time_us = int(labels.get("inject_time", 0)) * 1000000

        for trace in dataset.traces.values():
            # Find root spans (spans with no parent)
            root_spans = [
                span for span in trace.values() if span["parentSpanId"] is None
            ]

            for span in trace.values():
                service = span["nodeName"].split("@")[0]
                start_time = span["startTime"] // (60 * 1000000)  # minute bucket
                duration = span["duration"]

                duration_distribution["all"].append(duration)
                # collect duration by service and time bucket
                service_time_durations[service][start_time].append(duration)

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

            # NEW: Process root spans for before/after incident analysis
            for root_span in root_spans:
                service = root_span["nodeName"].split("@")[0]
                span_start_time = root_span["startTime"]
                span_duration = root_span["duration"]

                # Determine if this root span is before or after incident injection
                if span_start_time + span_duration < inject_time_us:
                    # Span ended before incident injection
                    root_duration_before_incident["all"].append(span_duration)
                    root_duration_before_incident[service].append(span_duration)
                elif span_start_time >= inject_time_us:
                    # Span started after incident injection
                    root_duration_after_incident["all"].append(span_duration)
                    root_duration_after_incident[service].append(span_duration)
                # Note: We skip spans that overlap with the injection time for cleaner analysis

        # calculate p90 for each service and time bucket
        duration_p90_by_service = {}
        for service, time_buckets in service_time_durations.items():
            duration_p90_by_service[service] = []
            for timebucket, durations in time_buckets.items():
                if durations:  # only calculate if there are durations
                    p90 = np.percentile(durations, 90)
                    duration_p90_by_service[service].append(
                        {"timebucket": timebucket, "p90": p90}
                    )
            # sort by timebucket for consistent ordering
            duration_p90_by_service[service].sort(key=lambda x: x["timebucket"])

        return {
            "duration": duration_distribution,
            "duration_pair": duration_pair_distribution,
            "duration_p90_by_service": duration_p90_by_service,
            "root_duration_before_incident": root_duration_before_incident,
            "root_duration_after_incident": root_duration_after_incident,
        }
