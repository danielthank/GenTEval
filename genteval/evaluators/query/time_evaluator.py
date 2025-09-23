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
        # Extract compression times if available
        compression_time_cpu = getattr(dataset, "compression_time_cpu_seconds", None)
        compression_time_gpu = getattr(dataset, "compression_time_gpu_seconds", None)

        # Calculate total compression time
        total_compression_time = None
        if compression_time_cpu is not None and compression_time_gpu is not None:
            total_compression_time = compression_time_cpu + compression_time_gpu
        elif compression_time_cpu is not None:
            total_compression_time = compression_time_cpu
        elif compression_time_gpu is not None:
            total_compression_time = compression_time_gpu

        return {
            "compression_time_cpu_seconds": compression_time_cpu,
            "compression_time_gpu_seconds": compression_time_gpu,
            "compression_time_total_seconds": total_compression_time,
        }
