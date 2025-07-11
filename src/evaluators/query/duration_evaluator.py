from collections import defaultdict

import numpy as np

from ...dataset import Dataset
from ..evaluator import Evaluator


class DurationEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        # duration distribution by service
        duration_distribution = defaultdict(list)
        duration_pair_distribution = defaultdict(list)
        # root span duration by service and time bucket for p50/p90 calculation
        root_span_time_durations = defaultdict(lambda: defaultdict(list))

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

            # NEW: Process root spans for before/after incident analysis and time buckets
            for root_span in root_spans:
                service = root_span["nodeName"].split("@")[0]
                span_start_time = root_span["startTime"]
                span_duration = root_span["duration"]
                start_time = span_start_time // (60 * 1000000)  # minute bucket

                # collect root span duration by service and time bucket for p50/p90
                root_span_time_durations[service][start_time].append(span_duration)

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

        # calculate p50 and p90 for each root span service and time bucket
        root_duration_p50_by_service = {}
        root_duration_p90_by_service = {}
        for service, time_buckets in root_span_time_durations.items():
            root_duration_p50_by_service[service] = []
            root_duration_p90_by_service[service] = []
            for timebucket, durations in time_buckets.items():
                if durations:  # only calculate if there are durations
                    p50 = np.percentile(durations, 50)
                    p90 = np.percentile(durations, 90)
                    count = len(durations)  # Number of traces in this bucket
                    root_duration_p50_by_service[service].append(
                        {"timebucket": timebucket, "p50": p50, "count": count}
                    )
                    root_duration_p90_by_service[service].append(
                        {"timebucket": timebucket, "p90": p90, "count": count}
                    )
            # sort by timebucket for consistent ordering
            root_duration_p50_by_service[service].sort(key=lambda x: x["timebucket"])
            root_duration_p90_by_service[service].sort(key=lambda x: x["timebucket"])

        # print number of root spans before and after incident
        print(
            f"Root spans before incident: {len(root_duration_before_incident['all'])}"
        )
        print(f"Root spans after incident: {len(root_duration_after_incident['all'])}")

        # print number of spans
        print(f"Total spans: {sum(len(trace) for trace in dataset.traces.values())}")

        return {
            "duration": duration_distribution,
            "duration_pair": duration_pair_distribution,
            "root_duration_p50_by_service": root_duration_p50_by_service,
            "root_duration_p90_by_service": root_duration_p90_by_service,
            "root_duration_before_incident": root_duration_before_incident,
            "root_duration_after_incident": root_duration_after_incident,
        }
