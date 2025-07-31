"""Size report generator for compressed data analysis."""

from typing import Any

from .base_report import BaseReport


class SizeReport(BaseReport):
    """Report generator for compressed data size analysis."""

    def generate(self, run_dirs) -> dict[str, Any]:
        """Generate size report."""

        for app_name, service, fault, run in run_dirs():
            for compressor in self.compressors:
                report_group = f"{app_name}_{compressor}"

                # Calculate compressed size
                compressed_dir = self.root_dir.joinpath(
                    app_name,
                    f"{service}_{fault}",
                    str(run),
                    compressor,
                    "compressed",
                    "data",
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

                self.report[report_group]["size"].append(total_size)

        # Calculate averages
        for group in self.report:
            if "size" in self.report[group] and self.report[group]["size"]:
                self.report[group]["size"] = sum(self.report[group]["size"]) / len(
                    self.report[group]["size"]
                )

        return dict(self.report)
