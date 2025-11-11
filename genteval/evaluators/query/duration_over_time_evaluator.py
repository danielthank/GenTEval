from collections import defaultdict

import numpy as np

from genteval.dataset import Dataset
from genteval.evaluators.evaluator import Evaluator


class DurationOverTimeEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        # [group][time_bucket] = [..durations]
        duration_by_time = defaultdict(lambda: defaultdict(list))

        # [group][time_bucket][percentile] = value
        duration_by_time_percentiles = defaultdict(lambda: defaultdict(dict))

        for trace in dataset.traces.values():
            for span in trace.values():
                service_name = span["nodeName"].split("!@#")[0]
                http_status_code = span.get("http.status_code")
                span_start_time = span["startTime"]
                span_duration = span["duration"]

                # Time bucket (minute)
                time_bucket = span_start_time // (60 * 1000000)

                # Group by "all"
                duration_by_time["all"][time_bucket].append(span_duration)

                # Group by service name
                duration_by_time[f"service.name:{service_name}"][time_bucket].append(
                    span_duration
                )

                # Group by HTTP status code
                if http_status_code:
                    duration_by_time[f"http.status_code:{http_status_code}"][
                        time_bucket
                    ].append(span_duration)

                    # Group by combined service + status code
                    duration_by_time[
                        f"service.name:{service_name}!@#http.status_code:{http_status_code}"
                    ][time_bucket].append(span_duration)

        # Calculate percentiles (p0 to p100)
        percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        for group, time_buckets in duration_by_time.items():
            for time_bucket, distribution in time_buckets.items():
                for p in percentiles:
                    duration_by_time_percentiles[group][time_bucket][f"p{p}"] = (
                        np.percentile(distribution, p)
                    )

        # Calculate total spans by group
        total_spans_by_group = {}
        for group, time_buckets in duration_by_time.items():
            total_count = sum(len(durations) for durations in time_buckets.values())
            total_spans_by_group[group] = total_count

        return {
            "duration_percentiles_by_time": duration_by_time_percentiles,
            "total_spans_by_group": total_spans_by_group,
        }
