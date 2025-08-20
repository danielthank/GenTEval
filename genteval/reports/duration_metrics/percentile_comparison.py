"""Percentile comparison metric for duration analysis."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from genteval.utils import align_time_series_data


class PercentileComparisonMetric:
    """Handles percentile comparison visualizations and calculations."""

    def __init__(self):
        pass

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

        # Calculate MAPE and counts for all services (shared logic)
        service_data = {}  # Store processed data for each service
        for service in sorted(common_services):
            # Extract percentile data as simple dictionaries
            original_percentile_dict = {
                item["timebucket"]: item[percentile_name.lower()]
                for item in original_data[service]
                if percentile_name.lower() in item
            }
            compressed_percentile_dict = {
                item["timebucket"]: item[percentile_name.lower()]
                for item in compressed_data[service]
                if percentile_name.lower() in item
            }

            if not original_percentile_dict or not compressed_percentile_dict:
                continue

            # Extract count information for weighting
            original_counts = {
                item["timebucket"]: item.get("count", 1)
                for item in original_data[service]
            }
            compressed_counts = {
                item["timebucket"]: item.get("count", 1)
                for item in compressed_data[service]
            }

            # Align and interpolate time series data
            original_values, compressed_values = align_time_series_data(
                original_percentile_dict, compressed_percentile_dict
            )

            # Get all timebuckets for count extraction
            all_buckets = sorted(
                set(original_percentile_dict.keys())
                | set(compressed_percentile_dict.keys())
            )

            # Extract trace counts for each bucket
            trace_counts = []
            for bucket in all_buckets:
                # Use original count or compressed count (prefer original, fallback to compressed, default to 0)
                count = original_counts.get(bucket, compressed_counts.get(bucket, 0))
                trace_counts.append(count)

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

        # For the legacy p50/p90 method, plotting is handled in the existing logic above
        # Just return the calculated results
        return {
            "mape": mape_results,
            "counts": count_results,
            "cosine_sim": cosine_sim_results,
        }

    def process_duration_by_depth_by_service(
        self,
        original_data,
        results_data,
        compressor,
        app_name,
        service,
        fault,
        run,
        plot=True,
    ):
        """
        Process duration_by_depth_by_service data for all depths, services, and percentiles.

        Args:
            original_data: Duration data from original dataset {depth: {service: [buckets]}}
            results_data: Duration data from compressed dataset {depth: {service: [buckets]}}
            compressor: Name of the compressor
            app_name: Application name
            service: Service name
            fault: Fault type
            run: Run identifier
            plot: Whether to generate plots

        Returns:
            Dict with structure {depth: {percentile: {"mape": {}, "counts": {}, "cosine_sim": {}}}}
        """
        # Define all percentiles to process
        percentiles = [
            "p0",
            "p10",
            "p20",
            "p30",
            "p40",
            "p50",
            "p60",
            "p70",
            "p80",
            "p90",
            "p100",
        ]

        # Store results for each depth and percentile
        results = defaultdict(dict)

        # Process each depth (0-4)
        for depth in range(5):  # 0, 1, 2, 3, 4
            depth_str = f"{depth}"
            if depth_str in original_data and depth_str in results_data:
                # Calculate metrics for each percentile across all services
                for percentile in percentiles:
                    percentile_metrics = self._calculate_percentile_metrics_for_depth(
                        original_data[depth_str], results_data[depth_str], percentile
                    )

                    if percentile_metrics:
                        results[depth_str][percentile] = percentile_metrics

                # Generate plots for each service if requested
                if plot:
                    for service_name in original_data[depth_str]:
                        if service_name in results_data[depth_str]:
                            original_service_data = original_data[depth_str][
                                service_name
                            ]
                            results_service_data = results_data[depth_str][service_name]

                            # Process each percentile for plotting
                            for percentile in percentiles:
                                self._create_percentile_plot(
                                    original_service_data,
                                    results_service_data,
                                    depth,
                                    service_name,
                                    percentile,
                                    compressor,
                                    app_name,
                                    service,
                                    fault,
                                    run,
                                )

        return results

    def _calculate_percentile_metrics_for_depth(
        self, original_services_data, results_services_data, percentile
    ):
        """Calculate MAPE and cosine similarity for a specific percentile across all services at a given depth."""

        all_service_mapes = {}
        all_service_counts = {}
        all_original_values = []
        all_results_values = []

        # Process each service
        for service_name in original_services_data:
            if service_name in results_services_data:
                original_service_data = original_services_data[service_name]
                results_service_data = results_services_data[service_name]

                # Extract percentile data as simple dictionaries
                original_percentile_dict = {
                    bucket["timebucket"]: bucket[percentile]
                    for bucket in original_service_data
                    if percentile in bucket
                }
                results_percentile_dict = {
                    bucket["timebucket"]: bucket[percentile]
                    for bucket in results_service_data
                    if percentile in bucket
                }

                if original_percentile_dict and results_percentile_dict:
                    # Align and interpolate time series data
                    original_values, results_values = align_time_series_data(
                        original_percentile_dict, results_percentile_dict
                    )

                    if len(original_values) > 0 and len(results_values) > 0:
                        # Calculate MAPE for this service
                        original_array = np.array(original_values)
                        results_array = np.array(results_values)

                        # Avoid division by zero
                        non_zero_mask = original_array != 0
                        if np.any(non_zero_mask):
                            mape = (
                                np.mean(
                                    np.abs(
                                        (
                                            original_array[non_zero_mask]
                                            - results_array[non_zero_mask]
                                        )
                                        / original_array[non_zero_mask]
                                    )
                                )
                                * 100
                            )
                            all_service_mapes[service_name] = mape
                            all_service_counts[service_name] = len(original_values)

                            # Add to overall arrays for cosine similarity
                            all_original_values.extend(original_values)
                            all_results_values.extend(results_values)

        # Calculate overall cosine similarity
        cosine_sim_results = {}
        if len(all_original_values) > 0 and len(all_results_values) > 0:
            original_vector = np.array(all_original_values).reshape(1, -1)
            results_vector = np.array(all_results_values).reshape(1, -1)

            # Calculate cosine similarity
            cos_sim = cosine_similarity(original_vector, results_vector)[0, 0]

            # Store cosine similarity for each service (same value for all)
            for service_name in all_service_mapes:
                cosine_sim_results[service_name] = cos_sim

        if all_service_mapes:
            return {
                "mape": all_service_mapes,
                "counts": all_service_counts,
                "cosine_sim": cosine_sim_results,
            }

        return None

    def _create_percentile_plot(
        self,
        original_data,
        results_data,
        depth,
        service_name,
        percentile,
        compressor,
        app_name,
        service,
        fault,
        run,
    ):
        """Create a plot for a specific percentile across time buckets."""

        # Extract time buckets and percentile values for original data
        original_times = []
        original_values = []
        for bucket in original_data:
            if percentile in bucket:
                original_times.append(bucket["timebucket"])
                original_values.append(bucket[percentile])

        # Extract time buckets and percentile values for results data
        results_times = []
        results_values = []
        for bucket in results_data:
            if percentile in bucket:
                results_times.append(bucket["timebucket"])
                results_values.append(bucket[percentile])

        if not original_times or not results_times:
            return

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(original_times, original_values, "b-", label="Original", linewidth=2)
        plt.plot(
            results_times, results_values, "r--", label=f"{compressor}", linewidth=2
        )

        plt.xlabel("Time Bucket")
        plt.ylabel(f"Duration {percentile.upper()} (microseconds)")
        plt.title(
            f"Duration {percentile.upper()} - Depth {depth} - {service_name}\n{app_name}_{service}_{fault}_{run}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_dir = (
            Path("output")
            / app_name
            / f"{service}_{fault}"
            / f"{run}"
            / "visualization"
            / "duration"
            / f"depth_{depth}"
            / percentile
        )
        plot_dir.mkdir(parents=True, exist_ok=True)

        plot_path = plot_dir / f"{compressor}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Created plot: {plot_path}")
