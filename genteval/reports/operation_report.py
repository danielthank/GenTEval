"""Operation report generator with F1 score calculation."""

from typing import Any

from genteval.bin.utils import get_dir_with_root

from .base_report import BaseReport


class OperationReport(BaseReport):
    """Report generator for operation evaluation with F1 score calculation."""

    def calculate_metrics(self, x: set, y: set) -> tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score between two sets.

        Args:
            x: Ground truth set
            y: Predicted set

        Returns:
            Tuple of (precision, recall, f1_score)
        """
        true_positives = len(x.intersection(y))
        false_positives = len(y - x)
        false_negatives = len(x - y)

        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

    def generate(self, run_dirs) -> dict[str, Any]:
        """Generate operation report with F1 score calculations."""
        for app_name, service, fault, run in run_dirs():
            for compressor in self.compressors:
                if compressor in {"original"}:
                    self.print_skip_message(
                        f"Compressor {compressor} is not supported for operation evaluation, "
                        f"skipping for {app_name}_{service}_{fault}_{run}."
                    )
                    continue

                original_results_path = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / "head_sampling_1"
                    / "evaluated"
                    / "operation_results.json"
                )

                if not self.file_exists(original_results_path):
                    self.print_skip_message(
                        f"Original results file {original_results_path} does not exist, skipping."
                    )
                    continue

                results_path = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / compressor
                    / "evaluated"
                    / "operation_results.json"
                )

                if not self.file_exists(results_path):
                    self.print_skip_message(
                        f"Results file {results_path} does not exist, skipping."
                    )
                    continue

                original = self.load_json_file(original_results_path)
                results = self.load_json_file(results_path)

                # Process operation data
                for group in original["operation"]:
                    if group not in results["operation"]:
                        continue

                    precision, recall, f1 = self.calculate_metrics(
                        set(original["operation"][group]),
                        set(results["operation"][group]),
                    )
                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["operation_precision"]["values"].append(
                        precision
                    )
                    self.report[report_group]["operation_recall"]["values"].append(
                        recall
                    )
                    self.report[report_group]["operation_f1"]["values"].append(f1)

                # Process operation_pair data
                for group in original["operation_pair"]:
                    if group not in results["operation_pair"]:
                        continue

                    precision, recall, f1 = self.calculate_metrics(
                        set(original["operation_pair"][group]),
                        set(results["operation_pair"][group]),
                    )
                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["operation_pair_precision"][
                        "values"
                    ].append(precision)
                    self.report[report_group]["operation_pair_recall"]["values"].append(
                        recall
                    )
                    self.report[report_group]["operation_pair_f1"]["values"].append(f1)

        # Calculate averages and clean up
        for group in self.report.values():
            for metric_group in group.values():
                if isinstance(metric_group, dict) and "values" in metric_group:
                    metric_group["avg"] = (
                        sum(metric_group["values"]) / len(metric_group["values"])
                        if metric_group["values"]
                        else float("nan")
                    )
                    del metric_group["values"]

        return dict(self.report)
