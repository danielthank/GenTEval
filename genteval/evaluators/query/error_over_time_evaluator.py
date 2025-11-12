from collections import defaultdict

from genteval.dataset import Dataset
from genteval.evaluators.evaluator import Evaluator


class ErrorOverTimeEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        # group -> time_bucket -> count
        span_count_by_time = defaultdict(lambda: defaultdict(int))

        for trace in dataset.traces.values():
            for span in trace.values():
                # Filter for ERROR status only (status.code = 2)
                # Extract status.code from nodeName (format: service!@#method!@#operation!@#status_code)
                node_parts = span["nodeName"].split("!@#")
                status_code = node_parts[3]
                if status_code != "2":
                    continue

                span_start_time = span["startTime"]
                time_bucket = span_start_time // (60 * 1000000)  # minute bucket

                # Extract service name and HTTP status code
                service_name = node_parts[0]
                http_status_code = span.get("http.status_code")

                # Count all error spans
                span_count_by_time["all"][time_bucket] += 1

                # Count error spans by service name
                service_group_key = f"service.name:{service_name}"
                span_count_by_time[service_group_key][time_bucket] += 1

                # Count error spans by HTTP status code
                if http_status_code:
                    status_group_key = f"http.status_code:{http_status_code}"
                    span_count_by_time[status_group_key][time_bucket] += 1

                    # Count error spans by service name + HTTP status code combination
                    combined_key = f"service.name:{service_name}!@#http.status_code:{http_status_code}"
                    span_count_by_time[combined_key][time_bucket] += 1

        # Calculate total error spans per group
        total_spans_by_group = {
            group: sum(time_buckets.values())
            for group, time_buckets in span_count_by_time.items()
        }

        return {
            "span_rate_by_time": span_count_by_time,
            "total_spans_by_group": total_spans_by_group,
        }
