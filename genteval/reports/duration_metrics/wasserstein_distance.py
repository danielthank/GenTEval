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
            ax.set_xlim(0, 100000)  # 0 to 500 milliseconds (was 500,000 μs)

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
