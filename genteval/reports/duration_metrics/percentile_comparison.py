"""Percentile comparison metric for duration analysis."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class PercentileComparisonMetric:
    """Handles percentile comparison visualizations and calculations."""

    def __init__(self, output_dirs):
        """Initialize with output directories."""
        self.duration_depth_0_p50_dir = output_dirs["duration_depth_0_p50_dir"]
        self.duration_depth_0_p90_dir = output_dirs["duration_depth_0_p90_dir"]
        self.duration_depth_1_p50_dir = output_dirs["duration_depth_1_p50_dir"]
        self.duration_depth_1_p90_dir = output_dirs["duration_depth_1_p90_dir"]
        self.viz_output_dir = output_dirs["viz_output_dir"]

    def _calculate_cosine_similarity(self, original_data, compressed_data):
        """Calculate cosine similarity between two duration arrays."""
        if not original_data or not compressed_data:
            return float("nan")

        # Convert to numpy arrays and ensure they have the same length
        original_array = np.array(original_data, dtype=float)
        compressed_array = np.array(compressed_data, dtype=float)

        # Find the minimum length to ensure equal dimensions
        min_len = min(len(original_array), len(compressed_array))
        if min_len == 0:
            return float("nan")

        # Truncate arrays to same length
        original_array = original_array[:min_len]
        compressed_array = compressed_array[:min_len]

        # Reshape for cosine_similarity function (needs 2D arrays)
        original_reshaped = original_array.reshape(1, -1)
        compressed_reshaped = compressed_array.reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(original_reshaped, compressed_reshaped)[0, 0]

        return similarity

    def visualize_duration_percentile_comparison(
        self,
        original_data,
        compressed_data,
        percentile_name,
        compressor,
        app_name,
        plot=True,
        depth=0,
    ):
        """Visualize duration percentile comparison by service with line charts and calculate MAPE."""
        # Get all services that exist in both datasets
        common_services = set(original_data.keys()) & set(compressed_data.keys())

        if not common_services:
            print(f"No common services found for {app_name}_{compressor}")
            return {}

        mape_results = {}
        count_results = {}  # Track trace counts for weighting
        cosine_sim_results = {}  # Track cosine similarity for each service

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
                original_values.append(original_buckets.get(bucket))
                compressed_values.append(compressed_buckets.get(bucket))
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

            # Calculate cosine similarity for this service
            cosine_sim = self._calculate_cosine_similarity(
                original_values, compressed_values
            )

            mape_results[service] = mape
            count_results[service] = total_trace_count
            cosine_sim_results[service] = cosine_sim

            # Store processed data for plotting if needed
            service_data[service] = {
                "all_buckets": all_buckets,
                "original_values": original_values,
                "compressed_values": compressed_values,
                "mape": mape,
                "cosine_sim": cosine_sim,
            }

        # If plot is False, return early with calculated results
        if not plot:
            return {
                "mape": mape_results,
                "counts": count_results,
                "cosine_sim": cosine_sim_results,
            }

        # Calculate number of subplots needed for plotting
        num_services = len(common_services)
        cols = min(3, num_services)  # Max 3 columns
        rows = (num_services + cols - 1) // cols  # Ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

        # Handle axes indexing for different subplot configurations
        if num_services == 1:
            # Single subplot case
            axes_flat = [axes]
        elif rows == 1:
            # Single row case
            axes_flat = (
                axes if isinstance(axes, np.ndarray) and axes.ndim == 1 else [axes]
            )
        else:
            # Multiple rows case
            axes_flat = axes.flatten()

        for idx, service in enumerate(sorted(common_services)):
            ax = axes_flat[idx]

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
            cosine_sim = data["cosine_sim"]

            # Plot the data (convert from μs to ms)
            x_indices = range(len(all_buckets))
            original_values_ms = np.array(original_values) / 1000  # Convert μs to ms
            compressed_values_ms = (
                np.array(compressed_values) / 1000
            )  # Convert μs to ms

            ax.plot(
                x_indices,
                original_values_ms,
                label=f"Original {percentile_name}",
                marker="o",
                linewidth=2,
            )
            ax.plot(
                x_indices,
                compressed_values_ms,
                label=f"{compressor} {percentile_name}",
                marker="s",
                linewidth=2,
            )

            ax.set_title(f"{service}\nMAPE: {mape:.2f}% | Cosine Sim: {cosine_sim:.3f}")
            ax.set_xlabel("Time Index")
            ax.set_ylabel(f"Duration {percentile_name} (ms)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            # Set y-axis to start from 0
            ax.set_ylim(bottom=0)

        # Hide unused subplots
        for idx in range(num_services, rows * cols):
            if idx < len(axes_flat):
                axes_flat[idx].set_visible(False)

        plt.tight_layout()

        # Save the plot in appropriate subdirectory based on depth and percentile
        if depth == 0:
            if percentile_name.lower() == "p50":
                output_dir = self.duration_depth_0_p50_dir
            elif percentile_name.lower() == "p90":
                output_dir = self.duration_depth_0_p90_dir
            else:
                output_dir = self.viz_output_dir  # Fallback
        elif depth == 1:
            if percentile_name.lower() == "p50":
                output_dir = self.duration_depth_1_p50_dir
            elif percentile_name.lower() == "p90":
                output_dir = self.duration_depth_1_p90_dir
            else:
                output_dir = self.viz_output_dir  # Fallback
        else:
            output_dir = self.viz_output_dir  # Fallback for other depths

        filename = f"{app_name}_{compressor}_duration_depth_{depth}_{percentile_name.lower()}_comparison.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "mape": mape_results,
            "counts": count_results,
            "cosine_sim": cosine_sim_results,
        }
