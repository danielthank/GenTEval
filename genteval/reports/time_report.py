"""Time report generator for compression timing analysis."""

from typing import Any

from genteval.bin.utils import get_dir_with_root

from .base_report import BaseReport


class TimeReport(BaseReport):
    """Report generator for compression timing analysis."""

    def generate(self, run_dirs) -> dict[str, Any]:
        """Generate time report."""

        for app_name, service, fault, run in run_dirs():
            for compressor in self.compressors:
                if compressor in {"original"}:
                    self.print_skip_message(
                        f"Compressor {compressor} is not supported for time evaluation, "
                        f"skipping for {app_name}_{service}_{fault}_{run}."
                    )
                    continue

                results_path = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / compressor
                    / "evaluated"
                    / "time_results.json"
                )

                if not self.file_exists(results_path):
                    self.print_skip_message(
                        f"Results file {results_path} does not exist, skipping."
                    )
                    continue

                results = self.load_json_file(results_path)
                report_group = f"{app_name}_{compressor}"

                # Extract compression time
                compression_time = results.get("compression_time_seconds")
                if compression_time is not None:
                    self.report[report_group]["compression_time_seconds"][
                        "values"
                    ].append(compression_time)

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
