"""Duration report generator with Wasserstein distance visualization."""

from collections import defaultdict
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

    def visualize_duration_p90_comparison(
        self, original_p90_data, compressed_p90_data, compressor, app_name
    ):
        """Visualize duration p90 comparison by service with line charts and calculate MAPE."""
        # Get all services that exist in both datasets
        common_services = set(original_p90_data.keys()) & set(
            compressed_p90_data.keys()
        )

        if not common_services:
            print(f"No common services found for {app_name}_{compressor}")
            return {}

        # Calculate number of subplots needed
        num_services = len(common_services)
        cols = min(3, num_services)  # Max 3 columns
        rows = (num_services + cols - 1) // cols  # Ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if num_services == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        mape_results = {}

        for idx, service in enumerate(sorted(common_services)):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            # Extract and sort data by timebucket
            original_data = sorted(
                original_p90_data[service], key=lambda x: x["timebucket"]
            )
            compressed_data = sorted(
                compressed_p90_data[service], key=lambda x: x["timebucket"]
            )

            # Find all timebuckets from both datasets (union instead of intersection)
            original_buckets = {
                item["timebucket"]: item["p90"] for item in original_data
            }
            compressed_buckets = {
                item["timebucket"]: item["p90"] for item in compressed_data
            }
            all_buckets = sorted(
                set(original_buckets.keys()) | set(compressed_buckets.keys())
            )

            if not all_buckets:
                ax.text(
                    0.5,
                    0.5,
                    f"No data\nfor\n{service}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(service)
                continue

            # Extract p90 values for all timebuckets, fill missing with 0
            original_p90s = [original_buckets.get(bucket, 0) for bucket in all_buckets]
            compressed_p90s = [
                compressed_buckets.get(bucket, 0) for bucket in all_buckets
            ]

            # Plot lines
            ax.plot(
                range(len(all_buckets)),
                original_p90s,
                label="head_sampling_1",
                color="blue",
                linewidth=2,
                marker="o",
            )
            ax.plot(
                range(len(all_buckets)),
                compressed_p90s,
                label=compressor,
                color="red",
                linewidth=2,
                marker="s",
            )

            # Calculate MAPE
            original_array = np.array(original_p90s)
            compressed_array = np.array(compressed_p90s)

            # Avoid division by zero
            non_zero_mask = original_array > 0
            if np.any(non_zero_mask):
                mape = (
                    np.mean(
                        np.abs(
                            (
                                original_array[non_zero_mask]
                                - compressed_array[non_zero_mask]
                            )
                            / original_array[non_zero_mask]
                        )
                    )
                    * 100
                )
            else:
                mape = 0

            mape_results[service] = mape

            ax.set_title(f"{service}\nMAPE: {mape:.2f}%")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Duration P90")
            ax.legend()
            ax.grid(True, alpha=0.3)
            # Set y-axis to start from 0
            ax.set_ylim(bottom=0)

        # Hide unused subplots
        for idx in range(num_services, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)

        plt.tight_layout()

        # Save the plot
        filename = f"{app_name}_{compressor}_duration_p90_comparison.png"
        filepath = self.viz_output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return mape_results

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
                    """

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

                # Process duration p90 data if available
                if (
                    "duration_p90_by_service" in original
                    and "duration_p90_by_service" in results
                ):
                    # Collect p90 data for this app-compressor combination
                    p90_data_key = f"{app_name}_{compressor}"

                    # Store p90 data temporarily for aggregation
                    if not hasattr(self, "p90_data_store"):
                        self.p90_data_store = defaultdict(
                            lambda: {
                                "original": defaultdict(list),
                                "compressed": defaultdict(list),
                                "metadata": {"app_name": None, "compressor": None},
                            }
                        )

                    # Store metadata for proper extraction later
                    self.p90_data_store[p90_data_key]["metadata"]["app_name"] = app_name
                    self.p90_data_store[p90_data_key]["metadata"]["compressor"] = (
                        compressor
                    )

                    # Aggregate p90 data across runs
                    for service_name, service_data in original[
                        "duration_p90_by_service"
                    ].items():
                        self.p90_data_store[p90_data_key]["original"][
                            service_name
                        ].extend(service_data)

                    for service_name, service_data in results[
                        "duration_p90_by_service"
                    ].items():
                        self.p90_data_store[p90_data_key]["compressed"][
                            service_name
                        ].extend(service_data)

        # Generate p90 visualizations after collecting all data
        if hasattr(self, "p90_data_store"):
            for app_compressor_key, data in self.p90_data_store.items():
                # Extract original app_name and compressor from stored metadata
                app_name = data["metadata"]["app_name"]
                compressor = data["metadata"]["compressor"]

                # Merge and sort p90 data by timebucket for each service
                merged_original = {}
                merged_compressed = {}

                for service in data["original"]:
                    # Merge duplicate timebuckets by averaging p90 values
                    timebucket_p90s = defaultdict(list)
                    for item in data["original"][service]:
                        timebucket_p90s[item["timebucket"]].append(item["p90"])

                    merged_original[service] = [
                        {"timebucket": tb, "p90": np.mean(p90s)}
                        for tb, p90s in timebucket_p90s.items()
                    ]

                for service in data["compressed"]:
                    # Merge duplicate timebuckets by averaging p90 values
                    timebucket_p90s = defaultdict(list)
                    for item in data["compressed"][service]:
                        timebucket_p90s[item["timebucket"]].append(item["p90"])

                    merged_compressed[service] = [
                        {"timebucket": tb, "p90": np.mean(p90s)}
                        for tb, p90s in timebucket_p90s.items()
                    ]

                # Generate visualization and calculate MAPE
                mape_results = self.visualize_duration_p90_comparison(
                    merged_original,
                    merged_compressed,
                    compressor,  # Use original compressor name
                    app_name,  # Use original app name
                )

                # Store MAPE results in report
                self.report[app_compressor_key]["duration_p90_mape"] = mape_results

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

            if "duration_p90_mape" in self.report[group]:
                # Calculate average MAPE across all services
                all_mape_values = list(self.report[group]["duration_p90_mape"].values())
                if all_mape_values:
                    self.report[group]["duration_p90_mape_avg"] = sum(
                        all_mape_values
                    ) / len(all_mape_values)
                # Keep the individual service MAPE values as well
                # del self.report[group]["duration_p90_mape"]  # Comment out to keep individual service MAPEs

        # Clean up temporary p90 data store
        if hasattr(self, "p90_data_store"):
            delattr(self, "p90_data_store")

        return dict(self.report)
