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
        # Create output directories for visualizations
        self.viz_output_dir = root_dir / "visualizations" / "duration"
        self.duration_all_dir = self.viz_output_dir / "duration_all_wasserstein_dist"
        self.duration_pair_dir = (
            self.viz_output_dir / "duration_pair_all_wasserstein_dist"
        )
        self.duration_p90_dir = self.viz_output_dir / "duration_p90"
        self.duration_before_after_dir = (
            self.viz_output_dir / "duration_before_after_incident"
        )

        # Create all subdirectories
        self.duration_all_dir.mkdir(parents=True, exist_ok=True)
        self.duration_pair_dir.mkdir(parents=True, exist_ok=True)
        self.duration_p90_dir.mkdir(parents=True, exist_ok=True)
        self.duration_before_after_dir.mkdir(parents=True, exist_ok=True)

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

        # Save the plot in appropriate subdirectory
        safe_group_name = group_name.replace("/", "_").replace(" ", "_")
        filename = f"{app_name}_{compressor}_{safe_group_name}_wasserstein_dist.png"

        # Choose the correct subdirectory based on the group type
        if "duration_pair" in group_name:
            filepath = self.duration_pair_dir / filename
        else:
            filepath = self.duration_all_dir / filename

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

        # Save the plot in duration_p90 subdirectory
        filename = f"{app_name}_{compressor}_duration_p90_comparison.png"
        filepath = self.duration_p90_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return mape_results

    def visualize_before_after_incident(
        self,
        original_before_data,
        original_after_data,
        compressed_before_data,
        compressed_after_data,
        compressor,
        app_name,
    ):
        """Visualize root span duration CDFs before and after incident injection in a single plot."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot original data - before incident (solid lines)
        if original_before_data and len(original_before_data) > 0:
            before_sorted = np.sort(original_before_data)
            before_cdf = np.arange(1, len(before_sorted) + 1) / len(before_sorted)
            ax.plot(
                before_sorted,
                before_cdf,
                label=f"Original Before ({len(original_before_data)} spans)",
                color="blue",
                linewidth=2.5,
                linestyle="-",
                marker="o",
                markersize=4,
                markevery=len(before_sorted) // 10 if len(before_sorted) > 10 else 1,
            )

        # Plot original data - after incident (dashed lines)
        if original_after_data and len(original_after_data) > 0:
            after_sorted = np.sort(original_after_data)
            after_cdf = np.arange(1, len(after_sorted) + 1) / len(after_sorted)
            ax.plot(
                after_sorted,
                after_cdf,
                label=f"Original After ({len(original_after_data)} spans)",
                color="blue",
                linewidth=2.5,
                linestyle="--",
                marker="s",
                markersize=4,
                markevery=len(after_sorted) // 10 if len(after_sorted) > 10 else 1,
            )

        # Plot compressed data - before incident (solid lines)
        if compressed_before_data and len(compressed_before_data) > 0:
            comp_before_sorted = np.sort(compressed_before_data)
            comp_before_cdf = np.arange(1, len(comp_before_sorted) + 1) / len(
                comp_before_sorted
            )
            ax.plot(
                comp_before_sorted,
                comp_before_cdf,
                label=f"{compressor} Before ({len(compressed_before_data)} spans)",
                color="red",
                linewidth=2.5,
                linestyle="-",
                marker="^",
                markersize=4,
                markevery=len(comp_before_sorted) // 10
                if len(comp_before_sorted) > 10
                else 1,
            )

        # Plot compressed data - after incident (dashed lines)
        if compressed_after_data and len(compressed_after_data) > 0:
            comp_after_sorted = np.sort(compressed_after_data)
            comp_after_cdf = np.arange(1, len(comp_after_sorted) + 1) / len(
                comp_after_sorted
            )
            ax.plot(
                comp_after_sorted,
                comp_after_cdf,
                label=f"{compressor} After ({len(compressed_after_data)} spans)",
                color="red",
                linewidth=2.5,
                linestyle="--",
                marker="d",
                markersize=4,
                markevery=len(comp_after_sorted) // 10
                if len(comp_after_sorted) > 10
                else 1,
            )

        ax.set_xlabel("Root Span Duration (μs)", fontsize=12)
        ax.set_ylabel("Cumulative Probability", fontsize=12)
        ax.set_title(
            "Root Span Duration CDF - Before vs After Incident",
            fontsize=14,
            fontweight="bold",
        )

        # Enhanced legend with visual grouping
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(
            0, 2000000
        )  # Root span duration range: 0 to 1,000,000 μs (1 second)

        # Calculate Wasserstein distances if both before and after data exist
        wdist_before = wdist_after = float("inf")
        if (
            original_before_data
            and compressed_before_data
            and len(original_before_data) > 0
            and len(compressed_before_data) > 0
        ):
            wdist_before = wasserstein_distance(
                original_before_data, compressed_before_data
            )

        if (
            original_after_data
            and compressed_after_data
            and len(original_after_data) > 0
            and len(compressed_after_data) > 0
        ):
            wdist_after = wasserstein_distance(
                original_after_data, compressed_after_data
            )

        # Overall title with Wasserstein distances
        fig.suptitle(
            f"{app_name} - {compressor}\n"
            f"W-Distance Before: {wdist_before:.4f}, W-Distance After: {wdist_after:.4f}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()

        # Save the plot
        filename = f"{app_name}_{compressor}_root_duration_before_after_incident.png"
        filepath = self.duration_before_after_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return wdist_before, wdist_after

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
                    # Generate p90 visualization immediately for this run
                    mape_results = self.visualize_duration_p90_comparison(
                        original["duration_p90_by_service"],
                        results["duration_p90_by_service"],
                        compressor,
                        f"{app_name}_{service}_{fault}_{run}",
                    )

                    # Store MAPE results in report
                    report_group = f"{app_name}_{compressor}"
                    if "duration_p90_mape_runs" not in self.report[report_group]:
                        self.report[report_group]["duration_p90_mape_runs"] = []
                    self.report[report_group]["duration_p90_mape_runs"].append(
                        mape_results
                    )

                # NEW: Process root duration before/after incident data
                if (
                    "root_duration_before_incident" in original
                    and "root_duration_after_incident" in original
                    and "root_duration_before_incident" in results
                    and "root_duration_after_incident" in results
                ):
                    # Process "all" service data for before/after incident
                    if (
                        "all" in original["root_duration_before_incident"]
                        and "all" in original["root_duration_after_incident"]
                        and "all" in results["root_duration_before_incident"]
                        and "all" in results["root_duration_after_incident"]
                    ):
                        # Generate before/after incident visualization
                        wdist_before, wdist_after = (
                            self.visualize_before_after_incident(
                                original["root_duration_before_incident"]["all"],
                                original["root_duration_after_incident"]["all"],
                                results["root_duration_before_incident"]["all"],
                                results["root_duration_after_incident"]["all"],
                                compressor,
                                f"{app_name}_{service}_{fault}_{run}",
                            )
                        )

                        # Store Wasserstein distances in report
                        report_group = f"{app_name}_{compressor}"
                        if (
                            "root_duration_before_wdist"
                            not in self.report[report_group]
                        ):
                            self.report[report_group]["root_duration_before_wdist"] = []
                        if "root_duration_after_wdist" not in self.report[report_group]:
                            self.report[report_group]["root_duration_after_wdist"] = []

                        # Only add finite distances to avoid inf values in averages
                        if wdist_before != float("inf"):
                            self.report[report_group][
                                "root_duration_before_wdist"
                            ].append(wdist_before)
                        if wdist_after != float("inf"):
                            self.report[report_group][
                                "root_duration_after_wdist"
                            ].append(wdist_after)

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

            if "duration_p90_mape_runs" in self.report[group]:
                # Calculate average MAPE across all runs and all services
                all_run_mapes = []
                for run_mape in self.report[group]["duration_p90_mape_runs"]:
                    all_run_mapes.extend(run_mape.values())

                if all_run_mapes:
                    self.report[group]["duration_p90_mape_avg"] = sum(
                        all_run_mapes
                    ) / len(all_run_mapes)
                # Keep the individual run MAPE values
                # del self.report[group]["duration_p90_mape_runs"]  # Comment out to keep individual run MAPEs

            # NEW: Calculate averages for before/after incident Wasserstein distances
            if "root_duration_before_wdist" in self.report[group]:
                if self.report[group]["root_duration_before_wdist"]:
                    self.report[group]["root_duration_before_wdist_avg"] = sum(
                        self.report[group]["root_duration_before_wdist"]
                    ) / len(self.report[group]["root_duration_before_wdist"])
                else:
                    self.report[group]["root_duration_before_wdist_avg"] = float("inf")
                del self.report[group]["root_duration_before_wdist"]

            if "root_duration_after_wdist" in self.report[group]:
                if self.report[group]["root_duration_after_wdist"]:
                    self.report[group]["root_duration_after_wdist_avg"] = sum(
                        self.report[group]["root_duration_after_wdist"]
                    ) / len(self.report[group]["root_duration_after_wdist"])
                else:
                    self.report[group]["root_duration_after_wdist_avg"] = float("inf")
                del self.report[group]["root_duration_after_wdist"]

        return dict(self.report)
