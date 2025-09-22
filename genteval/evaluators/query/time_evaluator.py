from genteval.dataset import Dataset
from genteval.evaluators import Evaluator


class TimeEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, labels):
        """Evaluate compression time from the dataset.

        Args:
            dataset: Dataset containing compression timing information
            labels: Labels (not used for time evaluation)

        Returns:
            Dictionary containing compression time metrics
        """
        # Extract compression time if available
        compression_time = getattr(dataset, "compression_time_seconds", None)

        return {"compression_time_seconds": compression_time}
