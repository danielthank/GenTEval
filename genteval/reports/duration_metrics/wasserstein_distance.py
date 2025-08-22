"""Wasserstein distance metric for duration analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


class WassersteinDistanceMetric:
    """Handles Wasserstein distance calculations and visualizations for duration data."""

    def __init__(self):
        pass

    def visualize_wasserstein_distributions(
        self,
        original_data,
        compressed_data,
        group_key,
        compressor,
        app_name,
        service=None,
        fault=None,
        run=None,
        plot=True,
    ):
        """Visualize the distributions used in Wasserstein distance calculation."""
        # Convert duration data to milliseconds for consistency (duration_pair data is not converted)
        is_duration_pair = "pair" in group_key
        if not is_duration_pair:
            original_data = np.array(original_data) / 1000
            compressed_data = np.array(compressed_data) / 1000

        # Calculate and return Wasserstein distance
        wdist = wasserstein_distance(original_data, compressed_data)

        if not plot:
            return wdist

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        original_sorted = np.sort(original_data)
        compressed_sorted = np.sort(compressed_data)

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
        ax.set_title(f"Cumulative Distribution Functions - {group_key}")
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
            f"{app_name} - {compressor} - {group_key}\nWasserstein Distance: {wdist:.4f}",
            fontsize=14,
        )

        plt.tight_layout()

        # Save the plot using new directory structure
        from pathlib import Path

        filename = f"{compressor}.png"

        # Determine subdirectory based on group type
        plot_dir = (
            Path("output")
            / app_name
            / f"{service}_{fault}"
            / f"{run}"
            / "visualization"
            / "wdist"
            / f"{group_key}"
        )

        plot_dir.mkdir(parents=True, exist_ok=True)
        filepath = plot_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Created Wasserstein plot: {filepath}")

        return wdist

    def _extract_depth_from_group_name(self, group_name):
        """Extract depth number from group name like 'duration_depth_2' or 'duration_depth_3_by_service'."""
        import re

        match = re.search(r"duration_depth_(\d+)", group_name)
        return match.group(1) if match else "0"

    def visualize_wasserstein_by_service(
        self,
        original_data,
        compressed_data,
        group_name,
        compressor,
        app_name,
        service=None,
        fault=None,
        run=None,
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

        # Save the plot using new directory structure
        filename = f"{compressor}.png"

        # Determine subdirectory based on group type
        if "by_service" in group_name:
            # Extract depth from group name (e.g., "duration_depth_2_by_service")
            depth = self._extract_depth_from_group_name(group_name)
            plot_dir = (
                Path("output")
                / app_name
                / f"{service}_{fault}"
                / f"{run}"
                / "visualization"
                / "duration"
                / f"depth_{depth}"
                / "by_service"
            )
        else:
            # Extract depth from group name (e.g., "duration_depth_2")
            depth = self._extract_depth_from_group_name(group_name)
            plot_dir = (
                Path("output")
                / app_name
                / f"{service}_{fault}"
                / f"{run}"
                / "visualization"
                / "duration"
                / f"depth_{depth}"
                / "all"
            )

        plot_dir.mkdir(parents=True, exist_ok=True)
        filepath = plot_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Created Wasserstein by service plot: {filepath}")

        return weighted_avg_wdist
