"""Operation report generator with F1 score calculation."""

from typing import Any, Dict, Set

from .base_report import BaseReport


class OperationReport(BaseReport):
    """Report generator for operation evaluation with F1 score calculation."""

    def f1_score(self, x: Set, y: Set) -> float:
        """
        Calculate F1 score between two sets.

        Args:
            x: Ground truth set
            y: Predicted set

        Returns:
            F1 score
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
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def generate(self, run_dirs) -> Dict[str, Any]:
        """Generate operation report with F1 score calculations."""
        for app_name, service, fault, run in run_dirs():
            for compressor in self.compressors:
                if compressor == "original" or compressor == "head_sampling_1":
                    self.print_skip_message(
                        f"Compressor {compressor} is not supported for operation evaluation, "
                        f"skipping for {app_name}_{service}_{fault}_{run}."
                    )
                    continue

                original_results_path = self.root_dir.joinpath(
                    app_name,
                    f"{service}_{fault}",
                    str(run),
                    "head_sampling_1",
                    "evaluated",
                    "operation_results.json",
                )

                if not self.file_exists(original_results_path):
                    self.print_skip_message(
                        f"Original results file {original_results_path} does not exist, skipping."
                    )
                    continue

                results_path = self.root_dir.joinpath(
                    app_name,
                    f"{service}_{fault}",
                    str(run),
                    compressor,
                    "evaluated",
                    "operation_results.json",
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

                    f1 = self.f1_score(
                        set(original["operation"][group]),
                        set(results["operation"][group]),
                    )
                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["operation_f1"].append(f1)

                # Process operation_pair data
                for group in original["operation_pair"]:
                    if group not in results["operation_pair"]:
                        continue

                    f1 = self.f1_score(
                        set(original["operation_pair"][group]),
                        set(results["operation_pair"][group]),
                    )
                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["operation_pair_f1"].append(f1)

        # Calculate averages and clean up
        for group in self.report:
            if "operation_f1" in self.report[group]:
                self.report[group]["operation_f1_avg"] = sum(
                    self.report[group]["operation_f1"]
                ) / len(self.report[group]["operation_f1"])
                del self.report[group]["operation_f1"]

            if "operation_pair_f1" in self.report[group]:
                self.report[group]["operation_pair_f1_avg"] = sum(
                    self.report[group]["operation_pair_f1"]
                ) / len(self.report[group]["operation_pair_f1"])
                del self.report[group]["operation_pair_f1"]

        return dict(self.report)
