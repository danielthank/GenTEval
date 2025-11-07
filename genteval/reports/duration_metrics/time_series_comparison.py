"""Generic time series comparison metric for both duration and count evaluations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class TimeSeriesComparisonMetric:
    """Generic time series comparison with MAPE and cosine similarity calculations."""

    def __init__(self):
        pass

    def calculate_time_series_metrics(self, original_time_series, results_time_series):
        """
        Core MAPE and cosine similarity calculation for any time series data.

        Args:
            original_time_series: List of original values over time
            results_time_series: List of compressed/results values over time

        Returns:
            Dict with {"mape": float, "cosine_sim": float}
        """
        if len(original_time_series) == 0 or len(results_time_series) == 0:
            return {"mape": float("inf"), "cosine_sim": 0.0}

        # Calculate MAPE using correct formula: abs((orig - res) / (orig + res)) * 100
        mape_values = []
        for orig, res in zip(original_time_series, results_time_series, strict=False):
            if orig > 0:
                mape = abs((orig - res) / (orig + res)) * 100
            else:
                mape = 0 if res == 0 else float("inf")
            mape_values.append(mape)

        # Calculate average MAPE (excluding inf values)
        finite_mape_values = [m for m in mape_values if not np.isinf(m)]
        avg_mape = np.mean(finite_mape_values) if finite_mape_values else float("inf")

        # Calculate cosine similarity
        if len(original_time_series) > 1:
            original_vector = np.array(original_time_series).reshape(1, -1)
            results_vector = np.array(results_time_series).reshape(1, -1)
            cosine_sim = cosine_similarity(original_vector, results_vector)[0, 0]
        else:
            cosine_sim = (
                1.0 if original_time_series[0] == results_time_series[0] else 0.0
            )

        return {"mape": avg_mape, "cosine_sim": cosine_sim}

    def process_duration_percentile_time_series(
        self,
        original_data,
        results_data,
        group_key,
        compressor,
        app_name,
        service,
        fault,
        run,
        plot=True,
    ):
        """
        Process duration percentile data - maintains backward compatibility with PercentileComparisonMetric.

        Args:
            original_data: Duration percentile data from original dataset {time_bucket: {percentile: value}}
            results_data: Duration percentile data from compressed dataset {time_bucket: {percentile: value}}
            group_key: The group key identifier (format: depth_{span_depth}_service_{service})
            compressor: Name of the compressor
            app_name: Application name
            service: Service name
            fault: Fault type
            run: Run identifier
            plot: Whether to generate plots

        Returns:
            Dict with structure {percentile: {"mape": value, "cosine_sim": value}}
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

        # Store results for each percentile
        results = {}

        # Get all common time buckets between original and results data
        common_time_buckets = set(original_data.keys()) & set(results_data.keys())

        # Calculate metrics for each percentile across time buckets
        for percentile in percentiles:
            original_time_series = []
            results_time_series = []

            # Collect values for this percentile across all time buckets
            for time_bucket in sorted(common_time_buckets):
                original_time_data = original_data[time_bucket]
                results_time_data = results_data[time_bucket]

                if percentile in original_time_data and percentile in results_time_data:
                    original_time_series.append(original_time_data[percentile])
                    results_time_series.append(results_time_data[percentile])

            if len(original_time_series) > 0:
                # Use generic time series calculation
                metrics = self.calculate_time_series_metrics(
                    original_time_series, results_time_series
                )
                results[percentile] = metrics

        # Generate plots if requested
        if plot:
            self._create_percentile_plots(
                original_data,
                results_data,
                group_key,
                compressor,
                app_name,
                service,
                fault,
                run,
            )

        return results

    def process_count_over_time_series(
        self, original_data, results_data, group_key, compressor_name=None
    ):
        """
        Process rate over time data - method for rate evaluation.

        Args:
            original_data: Original dataset with span_rate_by_time structure
            results_data: Compressed dataset with span_rate_by_time structure
            group_key: The group key (e.g., "all", "http.status_code:200")
            compressor_name: Name of the compressor to determine scaling factor

        Returns:
            Dict with {"mape": float, "cosine_sim": float}
        """
        try:
            # Extract time series for the specific group
            original_count_data = original_data.get("span_rate_by_time", {}).get(
                group_key, {}
            )
            results_count_data = results_data.get("span_rate_by_time", {}).get(
                group_key, {}
            )

            common_time_buckets = set(original_count_data.keys()) & set(
                results_count_data.keys()
            )

            if not common_time_buckets:
                return {"mape": float("inf"), "cosine_sim": 0.0}

            original_time_series = []
            results_time_series = []

            # Determine scaling factor for head sampling compressors
            scale_factor = 1
            if compressor_name and "head_sampling_" in compressor_name:
                try:
                    # Extract the sampling ratio from compressor name (e.g., "head_sampling_20" -> 20)
                    ratio_str = compressor_name.split("head_sampling_")[1]
                    scale_factor = int(ratio_str)
                except (IndexError, ValueError):
                    scale_factor = 1  # Default to 1 if parsing fails

            for time_bucket in sorted(common_time_buckets):
                original_time_series.append(original_count_data[time_bucket])
                # Scale up the compressed count to account for sampling ratio
                scaled_count = results_count_data[time_bucket] * scale_factor
                results_time_series.append(scaled_count)

            # Use same generic calculation
            return self.calculate_time_series_metrics(
                original_time_series, results_time_series
            )

        except (KeyError, ValueError, TypeError):
            return {"mape": float("inf"), "cosine_sim": 0.0}

    def _create_percentile_plots(
        self,
        original_data,
        results_data,
        group_key,
        compressor,
        app_name,
        service,
        fault,
        run,
    ):
        """Create comprehensive plots showing all percentiles with MAPE and cosine similarity metrics."""

        # Get all time buckets and sort them
        all_time_buckets = sorted(set(original_data.keys()) & set(results_data.keys()))

        if not all_time_buckets:
            return

        # Calculate metrics for all percentiles
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

        metrics_data = {}
        percentile_data = {}

        for percentile in percentiles:
            original_time_series = []
            results_time_series = []
            time_points = []

            # Collect data for this percentile across time buckets
            for time_bucket in all_time_buckets:
                if (
                    percentile in original_data[time_bucket]
                    and percentile in results_data[time_bucket]
                ):
                    time_points.append(time_bucket)
                    original_time_series.append(original_data[time_bucket][percentile])
                    results_time_series.append(results_data[time_bucket][percentile])

            if len(original_time_series) > 0:
                percentile_data[percentile] = {
                    "time_points": time_points,
                    "original": original_time_series,
                    "results": results_time_series,
                }

                # Use generic time series calculation for consistent MAPE formula
                metrics = self.calculate_time_series_metrics(
                    original_time_series, results_time_series
                )
                metrics_data[percentile] = metrics

        # Create 11 subplots (one for each percentile)
        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
        axes = axes.flatten()  # Make it easier to iterate

        for idx, percentile in enumerate(percentiles):
            ax = axes[idx]

            if percentile in percentile_data:
                data = percentile_data[percentile]

                # Plot original vs compressed time series for this percentile
                ax.plot(
                    data["time_points"],
                    data["original"],
                    "b-o",
                    label="Original",
                    linewidth=2,
                    markersize=4,
                )
                ax.plot(
                    data["time_points"],
                    data["results"],
                    "r--s",
                    label=f"{compressor}",
                    linewidth=2,
                    markersize=4,
                )

                ax.set_xlabel("Time Bucket")
                ax.set_ylabel("Duration (microseconds)")
                ax.set_title(f"{percentile.upper()} Over Time")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="x", rotation=45)

                # Add MAPE and cosine similarity as text annotations
                if percentile in metrics_data:
                    mape_val = metrics_data[percentile]["mape"]
                    cosine_val = metrics_data[percentile]["cosine_sim"]

                    # Format MAPE value (handle inf case)
                    if np.isinf(mape_val):
                        mape_text = "MAPE: ∞%"
                    else:
                        mape_text = f"MAPE: {mape_val:.1f}%"

                    cosine_text = f"Cos Sim: {cosine_val:.3f}"

                    # Add text annotations in the upper left of each subplot
                    ax.text(
                        0.02,
                        0.98,
                        mape_text,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        fontsize=10,
                        bbox={
                            "boxstyle": "round,pad=0.3",
                            "facecolor": "orange",
                            "alpha": 0.7,
                        },
                    )
                    ax.text(
                        0.02,
                        0.88,
                        cosine_text,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        fontsize=10,
                        bbox={
                            "boxstyle": "round,pad=0.3",
                            "facecolor": "lightgreen",
                            "alpha": 0.7,
                        },
                    )
            else:
                # If no data for this percentile, hide the subplot
                ax.set_visible(False)

        # Hide the last subplot if we have exactly 11 percentiles (4x3=12 subplots)
        if len(percentiles) < len(axes):
            for i in range(len(percentiles), len(axes)):
                axes[i].set_visible(False)

        plt.suptitle(
            f"Duration Percentiles Analysis: {app_name}_{service}_{fault}_{run} ({compressor})",
            fontsize=16,
            y=0.98,  # Move title up by about 2 lines height
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for title

        # Build path correctly handling None fault
        service_fault_dir = f"{service}_{fault}" if fault is not None else service

        plot_dir = (
            Path("output")
            / app_name
            / service_fault_dir
            / f"{run}"
            / "visualization"
            / "percentile"
            / group_key
        )
        plot_dir.mkdir(parents=True, exist_ok=True)

        plot_path = plot_dir / f"{compressor}_percentiles.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Created percentile subplots: {plot_path}")
