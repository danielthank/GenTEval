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

        for trace in dataset.traces.values():
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
        }
