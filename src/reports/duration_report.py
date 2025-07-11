"""Duration report generator with Wasserstein distance visualization."""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from .base_report import BaseReport


class DurationReport(BaseReport):
    """Report generator for duration evaluation with Wasserstein distance visualization."""

    def __init__(self, compressors, root_dir, plot=True):
        """Initialize the duration report generator."""
        super().__init__(compressors, root_dir)
        self.plot = plot
        # Create output directories for visualizations
        self.viz_output_dir = root_dir / "visualizations" / "duration"
        self.duration_all_dir = self.viz_output_dir / "duration_all_wasserstein_dist"
        self.duration_pair_dir = (
            self.viz_output_dir / "duration_pair_all_wasserstein_dist"
        )
        self.duration_p50_dir = self.viz_output_dir / "duration_p50"
        self.duration_p90_dir = self.viz_output_dir / "duration_p90"
        self.duration_before_after_dir = (
            self.viz_output_dir / "duration_before_after_incident"
        )

        # Create all subdirectories only if plotting is enabled
        if self.plot:
            self.duration_all_dir.mkdir(parents=True, exist_ok=True)
            self.duration_pair_dir.mkdir(parents=True, exist_ok=True)
            self.duration_p50_dir.mkdir(parents=True, exist_ok=True)
            self.duration_p90_dir.mkdir(parents=True, exist_ok=True)
            self.duration_before_after_dir.mkdir(parents=True, exist_ok=True)

    def visualize_wasserstein_distributions(
        self,
        original_data,
        compressed_data,
        group_name,
        compressor,
        app_name,
        plot=True,
    ):
        """Visualize the distributions used in Wasserstein distance calculation."""
        # Calculate and return Wasserstein distance
        wdist = wasserstein_distance(original_data, compressed_data)

        if not plot:
            return wdist

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

        # Set title with Wasserstein distance
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

    def visualize_before_after_incident(
        self,
        original_before_data,
        original_after_data,
        compressed_before_data,
        compressed_after_data,
        compressor,
        app_name,
        plot=True,
    ):
        """Visualize root span duration CDFs before and after incident injection in a single plot."""
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

        if not plot:
            return wdist_before, wdist_after

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot original data - before incident (solid lines)
        if original_before_data and len(original_before_data) > 0:
            before_sorted = np.sort(original_before_data)
            before_cdf = np.arange(1, len(before_sorted) + 1) / len(before_sorted)
            ax.plot(
                before_sorted,
                before_cdf,
                label=f"Original Before ({len(original_before_data)} traces)",
                color="blue",
                linewidth=2.5,
                linestyle="-",
            )

        # Plot original data - after incident (dashed lines)
        if original_after_data and len(original_after_data) > 0:
            after_sorted = np.sort(original_after_data)
            after_cdf = np.arange(1, len(after_sorted) + 1) / len(after_sorted)
            ax.plot(
                after_sorted,
                after_cdf,
                label=f"Original After ({len(original_after_data)} traces)",
                color="blue",
                linewidth=2.5,
                linestyle="--",
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
                label=f"{compressor} Before ({len(compressed_before_data)} traces)",
                color="red",
                linewidth=2.5,
                linestyle="-",
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
                label=f"{compressor} After ({len(compressed_after_data)} traces)",
                color="red",
                linewidth=2.5,
                linestyle="--",
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
                        plot=self.plot,
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
                        plot=self.plot,
                    )

                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["duration_pair_wdis"].append(wdist)

                # Process root duration p90 data if available
                if (
                    "root_duration_p90_by_service" in original
                    and "root_duration_p90_by_service" in results
                ):
                    # Generate p90 visualization immediately for this run
                    mape_count_results = self.visualize_duration_percentile_comparison(
                        original["root_duration_p90_by_service"],
                        results["root_duration_p90_by_service"],
                        "P90",
                        compressor,
                        f"{app_name}_{service}_{fault}_{run}",
                        plot=self.plot,
                    )

                    # Store MAPE and count results in report
                    report_group = f"{app_name}_{compressor}"
                    if "root_duration_p90_mape_runs" not in self.report[report_group]:
                        self.report[report_group]["root_duration_p90_mape_runs"] = []
                    if "root_duration_p90_count_runs" not in self.report[report_group]:
                        self.report[report_group]["root_duration_p90_count_runs"] = []
                    self.report[report_group]["root_duration_p90_mape_runs"].append(
                        mape_count_results["mape"]
                    )
                    self.report[report_group]["root_duration_p90_count_runs"].append(
                        mape_count_results["counts"]
                    )  # Process root duration p50 data if available
                if (
                    "root_duration_p50_by_service" in original
                    and "root_duration_p50_by_service" in results
                ):
                    # Generate p50 visualization immediately for this run
                    mape_count_results = self.visualize_duration_percentile_comparison(
                        original["root_duration_p50_by_service"],
                        results["root_duration_p50_by_service"],
                        "P50",
                        compressor,
                        f"{app_name}_{service}_{fault}_{run}",
                        plot=self.plot,
                    )

                    # Store MAPE and count results in report
                    report_group = f"{app_name}_{compressor}"
                    if "root_duration_p50_mape_runs" not in self.report[report_group]:
                        self.report[report_group]["root_duration_p50_mape_runs"] = []
                    if "root_duration_p50_count_runs" not in self.report[report_group]:
                        self.report[report_group]["root_duration_p50_count_runs"] = []
                    self.report[report_group]["root_duration_p50_mape_runs"].append(
                        mape_count_results["mape"]
                    )
                    self.report[report_group]["root_duration_p50_count_runs"].append(
                        mape_count_results["counts"]
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
                                plot=self.plot,
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

            if "root_duration_p90_mape_runs" in self.report[group]:
                # Calculate weighted average MAPE across all runs and all services
                all_run_mapes = []
                all_run_counts = []

                for i, run_mape in enumerate(
                    self.report[group]["root_duration_p90_mape_runs"]
                ):
                    run_counts = self.report[group]["root_duration_p90_count_runs"][i]
                    for service in run_mape:
                        if service in run_counts:
                            all_run_mapes.append(run_mape[service])
                            all_run_counts.append(run_counts[service])

                if all_run_mapes and sum(all_run_counts) > 0:
                    # Calculate weighted average
                    total_weighted_mape = sum(
                        mape * count
                        for mape, count in zip(all_run_mapes, all_run_counts)
                    )
                    total_count = sum(all_run_counts)
                    self.report[group]["root_duration_p90_mape_avg"] = (
                        total_weighted_mape / total_count
                    )
                # Keep the individual run MAPE values
                # del self.report[group]["root_duration_p90_mape_runs"]  # Comment out to keep individual run MAPEs

            if "root_duration_p50_mape_runs" in self.report[group]:
                # Calculate weighted average MAPE across all runs and all services
                all_run_mapes = []
                all_run_counts = []

                for i, run_mape in enumerate(
                    self.report[group]["root_duration_p50_mape_runs"]
                ):
                    run_counts = self.report[group]["root_duration_p50_count_runs"][i]
                    for service in run_mape:
                        if service in run_counts:
                            all_run_mapes.append(run_mape[service])
                            all_run_counts.append(run_counts[service])

                if all_run_mapes and sum(all_run_counts) > 0:
                    # Calculate weighted average
                    total_weighted_mape = sum(
                        mape * count
                        for mape, count in zip(all_run_mapes, all_run_counts)
                    )
                    total_count = sum(all_run_counts)
                    self.report[group]["root_duration_p50_mape_avg"] = (
                        total_weighted_mape / total_count
                    )
                # Keep the individual run MAPE values
                # del self.report[group]["root_duration_p50_mape_runs"]  # Comment out to keep individual run MAPEs

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

    def visualize_duration_percentile_comparison(
        self,
        original_data,
        compressed_data,
        percentile_name,
        compressor,
        app_name,
        plot=True,
    ):
        """Visualize duration percentile comparison by service with line charts and calculate MAPE."""
        # Get all services that exist in both datasets
        common_services = set(original_data.keys()) & set(compressed_data.keys())

        if not common_services:
            print(f"No common services found for {app_name}_{compressor}")
            return {}

        mape_results = {}
        count_results = {}  # Track trace counts for weighting

        # Helper function for interpolating missing values
        def interpolate_missing_values(values):
            values = values.copy()  # Don't modify original
            n = len(values)

            # Handle edge cases where all values are None
            if all(v is None for v in values):
                return [0] * n

            # Forward fill from first non-None value
            first_valid = next((i for i, v in enumerate(values) if v is not None), None)
            if first_valid is not None:
                for i in range(first_valid):
                    values[i] = values[first_valid]

            # Backward fill from last non-None value
            last_valid = next(
                (i for i, v in enumerate(reversed(values)) if v is not None), None
            )
            if last_valid is not None:
                last_valid = n - 1 - last_valid
                for i in range(last_valid + 1, n):
                    values[i] = values[last_valid]

            # Linear interpolation for missing values between valid values
            for i in range(n):
                if values[i] is None:
                    # Find previous and next valid values
                    prev_idx = next(
                        (j for j in range(i - 1, -1, -1) if values[j] is not None),
                        None,
                    )
                    next_idx = next(
                        (j for j in range(i + 1, n) if values[j] is not None), None
                    )

                    if prev_idx is not None and next_idx is not None:
                        # Linear interpolation
                        prev_val = values[prev_idx]
                        next_val = values[next_idx]
                        weight = (i - prev_idx) / (next_idx - prev_idx)
                        values[i] = prev_val + weight * (next_val - prev_val)
                    elif prev_idx is not None:
                        values[i] = values[prev_idx]
                    elif next_idx is not None:
                        values[i] = values[next_idx]
                    else:
                        values[i] = 0

            return values

        # Calculate MAPE and counts for all services (shared logic)
        service_data = {}  # Store processed data for each service
        for service in sorted(common_services):
            # Extract and sort data by timebucket
            original_service_data = sorted(
                original_data[service], key=lambda x: x["timebucket"]
            )
            compressed_service_data = sorted(
                compressed_data[service], key=lambda x: x["timebucket"]
            )

            # Find all timebuckets from both datasets (union instead of intersection)
            original_buckets = {
                item["timebucket"]: item[percentile_name.lower()]
                for item in original_service_data
            }
            compressed_buckets = {
                item["timebucket"]: item[percentile_name.lower()]
                for item in compressed_service_data
            }
            # Extract count information for weighting
            original_counts = {
                item["timebucket"]: item.get(
                    "count", 1
                )  # Default to 1 if count not available
                for item in original_service_data
            }
            compressed_counts = {
                item["timebucket"]: item.get(
                    "count", 1
                )  # Default to 1 if count not available
                for item in compressed_service_data
            }
            all_buckets = sorted(
                set(original_buckets.keys()) | set(compressed_buckets.keys())
            )

            if not all_buckets:
                continue

            # Extract percentile values and counts for all timebuckets, interpolate missing values
            original_values = []
            compressed_values = []
            trace_counts = []  # Track trace counts for each bucket

            # First pass: collect values and counts, mark missing as None
            for bucket in all_buckets:
                original_values.append(original_buckets.get(bucket, None))
                compressed_values.append(compressed_buckets.get(bucket, None))
                # Use original count or compressed count (prefer original, fallback to compressed, default to 0)
                count = original_counts.get(bucket, compressed_counts.get(bucket, 0))
                trace_counts.append(count)

            # Second pass: interpolate missing values
            original_values = interpolate_missing_values(original_values)
            compressed_values = interpolate_missing_values(compressed_values)

            # Calculate weighted MAPE (Mean Absolute Percentage Error)
            original_array = np.array(original_values)
            compressed_array = np.array(compressed_values)
            counts_array = np.array(trace_counts)

            # Only calculate MAPE for non-zero original values
            non_zero_mask = original_array > 0
            if np.any(non_zero_mask):
                # Calculate absolute percentage errors for each time bucket
                ape_values = np.abs(
                    (original_array[non_zero_mask] - compressed_array[non_zero_mask])
                    / original_array[non_zero_mask]
                )
                # Weight by trace counts
                weights = counts_array[non_zero_mask]
                if np.sum(weights) > 0:
                    mape = np.average(ape_values, weights=weights) * 100
                else:
                    mape = np.mean(ape_values) * 100  # Fallback to unweighted average
            else:
                mape = 0

            # Calculate total trace count for this service across all time buckets
            total_trace_count = np.sum(counts_array)

            mape_results[service] = mape
            count_results[service] = total_trace_count

            # Store processed data for plotting if needed
            service_data[service] = {
                "all_buckets": all_buckets,
                "original_values": original_values,
                "compressed_values": compressed_values,
                "mape": mape,
            }

        # If plot is False, return early with calculated results
        if not plot:
            return {"mape": mape_results, "counts": count_results}

        # Calculate number of subplots needed for plotting
        num_services = len(common_services)
        cols = min(3, num_services)  # Max 3 columns
        rows = (num_services + cols - 1) // cols  # Ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if num_services == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for idx, service in enumerate(sorted(common_services)):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            # Use pre-calculated data for this service
            if service not in service_data:
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

            data = service_data[service]
            all_buckets = data["all_buckets"]
            original_values = data["original_values"]
            compressed_values = data["compressed_values"]
            mape = data["mape"]

            # Plot the data
            x_indices = range(len(all_buckets))
            ax.plot(
                x_indices,
                original_values,
                label=f"Original {percentile_name}",
                marker="o",
                linewidth=2,
            )
            ax.plot(
                x_indices,
                compressed_values,
                label=f"{compressor} {percentile_name}",
                marker="s",
                linewidth=2,
            )

            ax.set_title(f"{service}\nMAPE: {mape:.2f}%")
            ax.set_xlabel("Time Index")
            ax.set_ylabel(f"Duration {percentile_name}")
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

        # Save the plot in appropriate subdirectory
        output_dir = getattr(self, f"duration_{percentile_name.lower()}_dir")
        filename = (
            f"{app_name}_{compressor}_duration_{percentile_name.lower()}_comparison.png"
        )
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return {"mape": mape_results, "counts": count_results}
