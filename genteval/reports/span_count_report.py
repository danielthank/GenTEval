"""Span count report generator with Wasserstein distance visualization."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from genteval.bin.utils import get_dir_with_root

from .base_report import BaseReport


class SpanCountReport(BaseReport):
    """Report generator for span count evaluation with Wasserstein distance visualization."""

    def __init__(self, compressors, root_dir):
        """Initialize the span count report generator."""
        super().__init__(compressors, root_dir)
        # Create output directory for visualizations
        self.viz_output_dir = root_dir / "visualizations" / "span_count"
        self.viz_output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_wasserstein_distributions(
        self, original_data, compressed_data, group_name, compressor, app_name
    ):
        """Visualize the distributions used in Wasserstein distance calculation."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot CDFs for better visualization of Wasserstein distance
        original_sorted = np.sort(original_data)
        compressed_sorted = np.sort(compressed_data)
        original_cdf = np.arange(1, len(original_sorted) + 1) / len(original_sorted)
        compressed_cdf = np.arange(1, len(compressed_sorted) + 1) / len(
            compressed_sorted
        )

        ax.plot(
            original_sorted,
            original_cdf,
            label="Original CDF",
            color="blue",
            linewidth=2,
        )
        ax.plot(
            compressed_sorted,
            compressed_cdf,
            label="Compressed CDF",
            color="red",
            linewidth=2,
        )
        ax.set_xlabel("Number of Spans per Trace")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"Cumulative Distribution Functions - {group_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set reasonable bounds for span count data
        max_spans = max(max(original_data, default=0), max(compressed_data, default=0))
        ax.set_xlim(0, max_spans)  # Add some padding for better visualization

        # Calculate and display Wasserstein distance
        wdist = wasserstein_distance(original_data, compressed_data)
        fig.suptitle(
            f"{app_name} - {compressor} - {group_name}\nWasserstein Distance: {wdist:.4f}",
            fontsize=14,
        )

        plt.tight_layout()

        # Save the plot
        safe_group_name = group_name.replace("/", "_").replace(" ", "_")
        filename = f"{app_name}_{compressor}_{safe_group_name}_wasserstein_dist.png"
        filepath = self.viz_output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return wdist

    def generate(self, run_dirs) -> dict[str, Any]:
        """Generate span count report with Wasserstein distance calculations and visualizations."""
        for app_name, service, fault, run in run_dirs():
            for compressor in self.compressors:
                if compressor in {"original"}:
                    self.print_skip_message(
                        f"Compressor {compressor} is not supported for span count evaluation, "
                        f"skipping for {app_name}_{service}_{fault}_{run}."
                    )
                    continue

                original_results_path = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / "head_1_1"
                    / "evaluated"
                    / "span_count_results.json"
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
                    / "span_count_results.json"
                )

                if not self.file_exists(results_path):
                    self.print_skip_message(
                        f"Results file {results_path} does not exist, skipping."
                    )
                    continue

                original = self.load_json_file(original_results_path)
                results = self.load_json_file(results_path)

                # Process span_count data
                for group in original["span_count"]:
                    if group not in results["span_count"]:
                        continue

                    # Visualize and calculate Wasserstein distance
                    wdist = self.visualize_wasserstein_distributions(
                        original["span_count"][group],
                        results["span_count"][group],
                        f"span_count_{group}",
                        compressor,
                        f"{app_name}_{service}_{fault}_{run}",
                    )

                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["span_count_wdist"]["values"].append(
                        wdist
                    )

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
