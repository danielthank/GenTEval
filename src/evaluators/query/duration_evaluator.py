from collections import defaultdict

from dataset import Dataset
from evaluators import Evaluator


class DurationEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        # duration distribution by service
        duration_distribution = defaultdict(list)
        duration_pair_distribution = defaultdict(list)
        for trace in dataset.traces.values():
            for span in trace.values():
                # service = span["nodeName"].split("@")[0]
                # start_time = span["startTime"] // (60 * 1000000)
                duration = span["duration"]
                duration_distribution["all"].append(duration)
                if span["parentSpanId"] is not None:
                    parent_span = trace.get(span["parentSpanId"])
                    if parent_span:
                        parent_duration = parent_span["duration"]
                        duration_pair_distribution["all"].append(
                            duration / parent_duration
                        )
        return {
            "duration": duration_distribution,
            "duration_pair": duration_pair_distribution,
        }
