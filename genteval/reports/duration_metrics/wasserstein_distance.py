"""Wasserstein distance metric for duration analysis."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


class WassersteinDistanceMetric:
    """Handles Wasserstein distance calculations and visualizations for duration data."""

    def __init__(self, output_dirs):
        """Initialize with output directories."""
        self.duration_all_dir = output_dirs["duration_all_dir"]
        self.duration_pair_dir = output_dirs["duration_pair_dir"]
        self.duration_depth_0_dir = output_dirs["duration_depth_0_dir"]
        self.duration_depth_1_dir = output_dirs["duration_depth_1_dir"]
        self.duration_depth_0_by_service_dir = output_dirs[
            "duration_depth_0_by_service_dir"
        ]
        self.duration_depth_1_by_service_dir = output_dirs[
            "duration_depth_1_by_service_dir"
        ]

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
        # Convert duration data to milliseconds for consistency (duration_pair data is not converted)
        is_duration_pair = "duration_pair" in group_name
        if is_duration_pair:
            # Duration pair ratios - no conversion needed
            original_for_wdist = original_data
            compressed_for_wdist = compressed_data
        else:
            # Duration data - convert from μs to ms
            original_for_wdist = np.array(original_data) / 1000
            compressed_for_wdist = np.array(compressed_data) / 1000

        # Calculate and return Wasserstein distance
        wdist = wasserstein_distance(original_for_wdist, compressed_for_wdist)

        if not plot:
            return wdist

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Check if this is duration_pair data to apply bounds
        is_duration_pair = "duration_pair" in group_name

        # Plot CDFs for better visualization of Wasserstein distance
        if is_duration_pair:
            # Duration pair ratios - no conversion needed
            original_sorted = np.sort(original_data)
            compressed_sorted = np.sort(compressed_data)
        else:
            # Duration data - convert from μs to ms
            original_sorted = np.sort(np.array(original_data) / 1000)
            compressed_sorted = np.sort(np.array(compressed_data) / 1000)

        original_cdf = np.arange(1, len(original_sorted) + 1) / len(original_sorted)
        compressed_cdf = np.arange(1, len(compressed_sorted) + 1) / len(
            compressed_sorted
        )

        # Calculate statistics for the data used in plotting
        original_mean = np.mean(original_sorted)
        compressed_mean = np.mean(compressed_sorted)
        original_std = np.std(original_sorted)
        compressed_std = np.std(compressed_sorted)

        ax.plot(
            original_sorted,
            original_cdf,
            label=f"Original CDF (μ={original_mean:.3f}, σ={original_std:.3f})",
            color="blue",
            linewidth=2,
        )
        ax.plot(
            compressed_sorted,
            compressed_cdf,
            label=f"Compressed CDF (μ={compressed_mean:.3f}, σ={compressed_std:.3f})",
            color="red",
            linewidth=2,
        )
        ax.set_xlabel(
            "Duration (ms)" if not is_duration_pair else "Duration Pair Ratio"
        )
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"Cumulative Distribution Functions - {group_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set bounds for duration_pair data
        if is_duration_pair:
            ax.set_xlim(0, 1)
        else:
            # Set xlim to max of duration data
            max_duration = max(np.max(original_sorted), np.max(compressed_sorted))
            ax.set_xlim(0, max_duration)

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
        elif "duration_depth_0_by_service" in group_name:
            filepath = self.duration_depth_0_by_service_dir / filename
        elif "duration_depth_1_by_service" in group_name:
            filepath = self.duration_depth_1_by_service_dir / filename
        elif "duration_depth_0" in group_name:
            filepath = self.duration_depth_0_dir / filename
        elif "duration_depth_1" in group_name:
            filepath = self.duration_depth_1_dir / filename
        else:
            filepath = self.duration_all_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return wdist

    def visualize_wasserstein_by_service(
        self,
        original_data,
        compressed_data,
        group_name,
        compressor,
        app_name,
        plot=True,
    ):
        """Visualize service-wise distributions and calculate weighted average Wasserstein distance."""
        total_wdist = 0
        total_weight = 0
        service_wdists = {}
        service_counts = {}

        if not plot:
            # Calculate weighted average without plotting
            for service in original_data:
                if (
                    service in compressed_data
                    and original_data[service]
                    and compressed_data[service]
                ):
                    original_for_wdist = (
                        np.array(original_data[service]) / 1000
                    )  # Convert to ms
                    compressed_for_wdist = np.array(compressed_data[service]) / 1000

                    wdist = wasserstein_distance(
                        original_for_wdist, compressed_for_wdist
                    )
                    weight = len(original_data[service])

                    service_wdists[service] = wdist
                    service_counts[service] = weight
                    total_wdist += wdist * weight
                    total_weight += weight

            weighted_avg_wdist = (
                total_wdist / total_weight if total_weight > 0 else float("inf")
            )
            return weighted_avg_wdist

        # Get common services
        common_services = set(original_data.keys()) & set(compressed_data.keys())
        common_services = [
            s for s in common_services if original_data[s] and compressed_data[s]
        ]

        if not common_services:
            return float("inf")

        # Calculate subplot dimensions
        n_services = len(common_services)
        if n_services <= 4:
            cols = 2
            rows = (n_services + 1) // 2
        elif n_services <= 9:
            cols = 3
            rows = (n_services + 2) // 3
        else:
            cols = 4
            rows = (n_services + 3) // 4

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        service_idx = 0
        for service in sorted(common_services):
            if service_idx >= len(axes):
                break

            ax = axes[service_idx]

            # Convert to milliseconds
            original_for_wdist = np.array(original_data[service]) / 1000
            compressed_for_wdist = np.array(compressed_data[service]) / 1000

            # Calculate Wasserstein distance
            wdist = wasserstein_distance(original_for_wdist, compressed_for_wdist)
            weight = len(original_data[service])

            service_wdists[service] = wdist
            service_counts[service] = weight
            total_wdist += wdist * weight
            total_weight += weight

            # Plot CDF
            original_sorted = np.sort(original_for_wdist)
            compressed_sorted = np.sort(compressed_for_wdist)

            original_cdf = np.arange(1, len(original_sorted) + 1) / len(original_sorted)
            compressed_cdf = np.arange(1, len(compressed_sorted) + 1) / len(
                compressed_sorted
            )

            ax.plot(
                original_sorted,
                original_cdf,
                label="Original",
                color="blue",
                linewidth=2,
            )
            ax.plot(
                compressed_sorted,
                compressed_cdf,
                label="Compressed",
                color="red",
                linewidth=2,
            )

            ax.set_xlabel("Duration (ms)")
            ax.set_ylabel("Cumulative Probability")
            ax.set_title(f"{service}\nWD: {wdist:.4f} (n={weight})")
            ax.legend()
            ax.grid(True, alpha=0.3)

            service_idx += 1

        # Hide unused subplots
        for idx in range(service_idx, len(axes)):
            axes[idx].set_visible(False)

        # Calculate weighted average
        weighted_avg_wdist = (
            total_wdist / total_weight if total_weight > 0 else float("inf")
        )

        # Set main title
        fig.suptitle(
            f"{app_name} - {compressor} - {group_name}\n"
            f"Weighted Average Wasserstein Distance: {weighted_avg_wdist:.4f}",
            fontsize=14,
        )

        plt.tight_layout()

        # Save the plot
        safe_group_name = group_name.replace("/", "_").replace(" ", "_")
        filename = (
            f"{app_name}_{compressor}_{safe_group_name}_by_service_wasserstein_dist.png"
        )

        if "duration_depth_0_by_service" in group_name:
            filepath = self.duration_depth_0_by_service_dir / filename
        elif "duration_depth_1_by_service" in group_name:
            filepath = self.duration_depth_1_by_service_dir / filename
        else:
            filepath = self.duration_all_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return weighted_avg_wdist
