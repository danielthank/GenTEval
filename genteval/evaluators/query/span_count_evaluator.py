from collections import defaultdict

from genteval.dataset import Dataset
from genteval.evaluators import Evaluator
from genteval.utils.data_structures import count_spans_per_tree


class SpanCountEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        # tree span count distribution (each trace may have multiple trees)
        tree_span_count_distribution = defaultdict(list)

        for trace in dataset.traces.values():
            # Count spans per tree using Union-Find
            tree_sizes = count_spans_per_tree(trace)
            tree_span_count_distribution["all"].extend(tree_sizes)

        return {
            "span_count": tree_span_count_distribution,
        }
