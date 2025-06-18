"""Duration report generator with Wasserstein distance visualization."""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from .base_report import BaseReport


class DurationReport(BaseReport):
    """Report generator for duration evaluation with Wasserstein distance visualization."""

    def __init__(self, compressors, root_dir):
        """Initialize the duration report generator."""
        super().__init__(compressors, root_dir)
        # Create output directory for visualizations
        self.viz_output_dir = root_dir / "visualizations" / "duration"
        self.viz_output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_wasserstein_distributions(
        self, original_data, compressed_data, group_name, compressor, app_name
    ):
        """Visualize the distributions used in Wasserstein distance calculation."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Check if this is duration_pair data to apply bounds
        is_duration_pair = "duration_pair" in group_name

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
        ax.set_xlabel("Duration" if not is_duration_pair else "Duration Pair Ratio")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"Cumulative Distribution Functions - {group_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set bounds for duration_pair data
        if is_duration_pair:
            ax.set_xlim(0, 1)
        else:
            ax.set_xlim(0, 500000)

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

    def generate(self, run_dirs) -> Dict[str, Any]:
        """Generate duration report with Wasserstein distance calculations and visualizations."""
        for app_name, service, fault, run in run_dirs():
            for compressor in self.compressors:
                if compressor == "original" or compressor == "head_sampling_1":
                    self.print_skip_message(
                        f"Compressor {compressor} is not supported for duration evaluation, "
                        f"skipping for {app_name}_{service}_{fault}_{run}."
                    )
                    continue

                original_results_path = self.root_dir.joinpath(
                    app_name,
                    f"{service}_{fault}",
                    str(run),
                    "head_sampling_1",
                    "evaluated",
                    "duration_results.json",
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
                    "duration_results.json",
                )

                if not self.file_exists(results_path):
                    self.print_skip_message(
                        f"Results file {results_path} does not exist, skipping."
                    )
                    continue

                original = self.load_json_file(original_results_path)
                results = self.load_json_file(results_path)

                # Process duration data
                for group in original["duration"]:
                    if group not in results["duration"]:
                        continue

                    # Visualize and calculate Wasserstein distance
                    """
                    wdist = self.visualize_wasserstein_distributions(
                        original["duration"][group],
                        results["duration"][group],
                        f"duration_{group}",
                        compressor,
                        f"{app_name}_{service}_{fault}_{run}",
                    )
                    """
                    wdist = wasserstein_distance(
                        original["duration"][group],
                        results["duration"][group],
                    )

                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["duration_wdis"].append(wdist)

                # Process duration_pair data
                for group in original["duration_pair"]:
                    if group not in results["duration_pair"]:
                        continue

                    # Visualize and calculate Wasserstein distance
                    wdist = self.visualize_wasserstein_distributions(
                        original["duration_pair"][group],
                        results["duration_pair"][group],
                        f"duration_pair_{group}",
                        compressor,
                        f"{app_name}_{service}_{fault}_{run}",
                    )

                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["duration_pair_wdis"].append(wdist)

        # Calculate averages and clean up
        for group in self.report:
            if "duration_wdis" in self.report[group]:
                self.report[group]["duration_wdis_avg"] = sum(
                    self.report[group]["duration_wdis"]
                ) / len(self.report[group]["duration_wdis"])
                del self.report[group]["duration_wdis"]

            if "duration_pair_wdis" in self.report[group]:
                self.report[group]["duration_pair_wdis_avg"] = sum(
                    self.report[group]["duration_pair_wdis"]
                ) / len(self.report[group]["duration_pair_wdis"])
                del self.report[group]["duration_pair_wdis"]

        return dict(self.report)
