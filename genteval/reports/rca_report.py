"""Unified RCA report generator."""

from typing import Any

from genteval.bin.utils import get_dir_with_root

from .base_report import BaseReport


class RCAReport(BaseReport):
    """Report generator for RCA evaluation (supports multiple algorithms)."""

    def __init__(self, compressors, root_dir, results_filename):
        """
        Initialize the RCA report generator.

        Args:
            compressors: List of compressor names to evaluate
            root_dir: Root directory containing the output data
            results_filename: Name of the results JSON file (e.g., "trace_rca_results.json", "micro_rank_results.json")
        """
        super().__init__(compressors, root_dir)
        self.results_filename = results_filename

    def ac_at_k(self, answer: str, ranks: list, k: int) -> bool:
        """
        Calculate accuracy at k.

        Args:
            answer: The correct answer
            ranks: List of ranked results
            k: The k value for accuracy calculation

        Returns:
            True if answer is in top k ranks, False otherwise
        """
        return answer in ranks[:k]

    def generate(self, run_dirs) -> dict[str, Any]:
        """Generate RCA report."""
        services = set()

        for app_name, service, fault, run in run_dirs():
            for compressor in self.compressors:
                report_group = f"{app_name}_{compressor}"

                results_path = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / compressor
                    / "evaluated"
                    / self.results_filename
                )

                if not self.file_exists(results_path):
                    self.print_skip_message(
                        f"Results file {results_path} does not exist, skipping."
                    )
                    continue

                results = self.load_json_file(results_path)
                if "ranks" not in results:
                    continue
                ranks = results["ranks"]

                for rank in ranks:
                    services.add(rank)

                for k in range(1, 6):
                    self.report[report_group][f"ac{k}"]["values"].append(
                        self.ac_at_k(service, ranks, k)
                    )

        # Calculate averages and clean up
        for group in self.report.values():
            ac_avgs = []
            for metric_group in group.values():
                if isinstance(metric_group, dict) and "values" in metric_group:
                    metric_group["avg"] = (
                        sum(metric_group["values"]) / len(metric_group["values"])
                        if metric_group["values"]
                        else float("nan")
                    )
                    ac_avgs.append(metric_group["avg"])
                    del metric_group["values"]

            # Calculate avg5 from the individual ac averages
            if ac_avgs:
                group["avg5"] = {"avg": sum(ac_avgs) / len(ac_avgs)}

        return dict(self.report)
