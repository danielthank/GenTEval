from collections import defaultdict

from ...dataset import Dataset
from ..evaluator import Evaluator


class SpanCountEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        # span count distribution per trace
        span_count_distribution = defaultdict(list)

        for trace in dataset.traces.values():
            # Count total number of spans in this trace
            num_spans = len(trace)
            span_count_distribution["all"].append(num_spans)

        return {
            "span_count": span_count_distribution,
        }
