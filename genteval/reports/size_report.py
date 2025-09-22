"""Size report generator for compressed data analysis."""

from typing import Any

from genteval.bin.utils import get_dir_with_root

from .base_report import BaseReport


class SizeReport(BaseReport):
    """Report generator for compressed data size analysis."""

    def generate(self, run_dirs) -> dict[str, Any]:
        """Generate size report."""

        for app_name, service, fault, run in run_dirs():
            for compressor in self.compressors:
                report_group = f"{app_name}_{compressor}"

                # Calculate compressed size
                compressed_dir = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / compressor
                    / "compressed"
                    / "data"
                )

                if not compressed_dir.exists():
                    self.print_skip_message(
                        f"Compressed directory {compressed_dir} does not exist, skipping."
                    )
                    continue

                total_size = 0
                for file in compressed_dir.glob("**/*"):
                    if file.is_file():
                        total_size += file.stat().st_size

                self.report[report_group]["size"]["values"].append(total_size)

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
