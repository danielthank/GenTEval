"""Depth before/after incident metric for duration analysis."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


class DepthBeforeAfterMetric:
    """Handles before/after incident analysis for depth-specific spans."""

    def __init__(self, output_dirs):
        """Initialize with output directories."""
        self.duration_depth_0_before_after_dir = output_dirs[
            "duration_depth_0_before_after_dir"
        ]
        self.duration_depth_1_before_after_dir = output_dirs[
            "duration_depth_1_before_after_dir"
        ]
        self.viz_output_dir = output_dirs["viz_output_dir"]

    def visualize_depth_before_after_incident(
        self,
        original_before_data,
        original_after_data,
        compressed_before_data,
        compressed_after_data,
        compressor,
        app_name,
        depth,
        plot=True,
    ):
        """Visualize span duration CDFs before and after incident injection for specified depth."""
        # Calculate Wasserstein distances if both before and after data exist
        wdist_before = wdist_after = float("inf")
        if (
            original_before_data
            and compressed_before_data
            and len(original_before_data) > 0
            and len(compressed_before_data) > 0
        ):
            # Convert to milliseconds for consistent distance calculation
            original_before_ms = np.array(original_before_data) / 1000
            compressed_before_ms = np.array(compressed_before_data) / 1000
            wdist_before = wasserstein_distance(
                original_before_ms, compressed_before_ms
            )

        if (
            original_after_data
            and compressed_after_data
            and len(original_after_data) > 0
            and len(compressed_after_data) > 0
        ):
            # Convert to milliseconds for consistent distance calculation
            original_after_ms = np.array(original_after_data) / 1000
            compressed_after_ms = np.array(compressed_after_data) / 1000
            wdist_after = wasserstein_distance(original_after_ms, compressed_after_ms)

        if not plot:
            return wdist_before, wdist_after

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left subplot: Before incident
        if original_before_data and len(original_before_data) > 0:
            before_sorted = np.sort(
                np.array(original_before_data) / 1000
            )  # Convert μs to ms
            before_cdf = np.arange(1, len(before_sorted) + 1) / len(before_sorted)
            ax1.plot(
                before_sorted,
                before_cdf,
                label=f"Original ({len(original_before_data)} traces)",
                color="blue",
                linewidth=2.5,
            )

        if compressed_before_data and len(compressed_before_data) > 0:
            comp_before_sorted = np.sort(
                np.array(compressed_before_data) / 1000
            )  # Convert μs to ms
            comp_before_cdf = np.arange(1, len(comp_before_sorted) + 1) / len(
                comp_before_sorted
            )
            ax1.plot(
                comp_before_sorted,
                comp_before_cdf,
                label=f"{compressor} ({len(compressed_before_data)} traces)",
                color="red",
                linewidth=2.5,
            )

        ax1.set_xlabel(f"Depth {depth} Span Duration (ms)", fontsize=12)
        ax1.set_ylabel("Cumulative Probability", fontsize=12)
        ax1.set_title(
            f"Before Incident\nW-Distance: {wdist_before:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        ax1.legend(loc="best", fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 10000)  # Span duration range: 0 to 10000 ms

        # Right subplot: After incident
        if original_after_data and len(original_after_data) > 0:
            after_sorted = np.sort(
                np.array(original_after_data) / 1000
            )  # Convert μs to ms
            after_cdf = np.arange(1, len(after_sorted) + 1) / len(after_sorted)
            ax2.plot(
                after_sorted,
                after_cdf,
                label=f"Original ({len(original_after_data)} traces)",
                color="blue",
                linewidth=2.5,
            )

        if compressed_after_data and len(compressed_after_data) > 0:
            comp_after_sorted = np.sort(
                np.array(compressed_after_data) / 1000
            )  # Convert μs to ms
            comp_after_cdf = np.arange(1, len(comp_after_sorted) + 1) / len(
                comp_after_sorted
            )
            ax2.plot(
                comp_after_sorted,
                comp_after_cdf,
                label=f"{compressor} ({len(compressed_after_data)} traces)",
                color="red",
                linewidth=2.5,
            )

        ax2.set_xlabel(f"Depth {depth} Span Duration (ms)", fontsize=12)
        ax2.set_ylabel("Cumulative Probability", fontsize=12)
        ax2.set_title(
            f"After Incident\nW-Distance: {wdist_after:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        ax2.legend(loc="best", fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 10000)  # Span duration range: 0 to 10000 ms

        # Overall title
        fig.suptitle(
            f"{app_name} - {compressor}\nDepth {depth} Span Duration CDF - Before vs After Incident",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()

        # Save the plot in appropriate subdirectory based on depth
        filename = (
            f"{app_name}_{compressor}_depth_{depth}_duration_before_after_incident.png"
        )
        if depth == 0:
            filepath = self.duration_depth_0_before_after_dir / filename
        elif depth == 1:
            filepath = self.duration_depth_1_before_after_dir / filename
        else:
            # Fallback for other depths
            filepath = self.viz_output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return wdist_before, wdist_after
