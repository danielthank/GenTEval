"""Trace RCA report generator."""

from typing import Any, Dict

from .base_report import BaseReport


class TraceRCAReport(BaseReport):
    """Report generator for trace RCA evaluation."""

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
        return answer in ranks[:k] if k <= len(ranks) else False

    def generate(self, run_dirs) -> Dict[str, Any]:
        """Generate trace RCA report."""
        services = set()

        for app_name, service, fault, run in run_dirs():
            for compressor in self.compressors:
                report_group = f"{app_name}_{compressor}"

                results_path = self.root_dir.joinpath(
                    app_name,
                    f"{service}_{fault}",
                    str(run),
                    compressor,
                    "evaluated",
                    "trace_rca_results.json",
                )

                if not self.file_exists(results_path):
                    self.print_skip_message(
                        f"Results file {results_path} does not exist, skipping."
                    )
                    continue

                results = self.load_json_file(results_path)
                ranks = results.get("ranks", [])

                for rank in ranks:
                    services.add(rank)

                for k in range(1, 6):
                    self.report[report_group][f"ac{k}"].append(
                        self.ac_at_k(service, ranks, k)
                    )

                # Calculate compressed size
                compressed_dir = self.root_dir.joinpath(
                    app_name,
                    f"{service}_{fault}",
                    str(run),
                    compressor,
                    "compressed",
                    "data",
                )
                total_size = 0
                for file in compressed_dir.glob("**/*"):
                    if file.is_file():
                        total_size += file.stat().st_size
                self.report[report_group]["size"].append(total_size)

        # Calculate averages
        for fault in self.report:
            for k in range(1, 6):
                self.report[fault][f"ac{k}"] = sum(self.report[fault][f"ac{k}"]) / len(
                    self.report[fault][f"ac{k}"]
                )
            self.report[fault]["avg5"] = (
                sum(self.report[fault][f"ac{k}"] for k in range(1, 6)) / 5
            )
            self.report[fault]["size"] = sum(self.report[fault]["size"]) / len(
                self.report[fault]["size"]
            )

        return dict(self.report)
