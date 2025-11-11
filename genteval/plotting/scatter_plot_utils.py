"""Shared utilities for scatter plot visualization."""

import matplotlib.pyplot as plt
import numpy as np


# Color scheme for different durations (matching draw.py)
DURATION_COLORS = {"1min": "red", "5min": "blue", "10min": "green"}
DURATION_MARKERS = {"1min": "o", "5min": "s", "10min": "^"}


def setup_plot_style():
    """Setup plot figure and style."""
    plt.figure(figsize=(12, 8))
    plt.style.use("default")
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14


def group_experiments_by_duration(data_points, metric):
    """
    Group experiments by duration for GenT and separate head sampling.

    Args:
        data_points: List of data points with 'compressor', 'cost_per_million', and metric field
        metric: Name of the metric field to extract (e.g., 'mape_fidelity')

    Returns:
        Tuple of (duration_groups, head_sampling_points)
        - duration_groups: dict {duration: [(cost, fidelity, compressor), ...]}
        - head_sampling_points: list [(cost, fidelity, compressor), ...]
    """
    duration_groups = {}
    head_sampling_points = []

    for point in data_points:
        compressor = point["compressor"]
        cost = point["cost_per_million"]
        fidelity = point[metric]

        if "gent" in compressor:
            # Extract duration (e.g., "gent_1_1" -> "1min", "gent_5_2" -> "5min")
            parts = compressor.split("_")
            if "gent" in parts:
                gent_idx = parts.index("gent")
                if gent_idx + 1 < len(parts):
                    duration = f"{parts[gent_idx + 1]}min"
                    if duration not in duration_groups:
                        duration_groups[duration] = []
                    duration_groups[duration].append((cost, fidelity, compressor))
        elif "head_sampling" in compressor:
            head_sampling_points.append((cost, fidelity, compressor))

    return duration_groups, head_sampling_points


def plot_duration_groups(duration_groups):
    """
    Plot GenT duration groups with mean ± std error bars.

    Args:
        duration_groups: dict {duration: [(cost, fidelity, compressor), ...]}
    """
    for duration in sorted(duration_groups.keys()):
        group_data = duration_groups[duration]
        group_costs = [d[0] for d in group_data]
        group_fidelities = [d[1] for d in group_data]

        mean_cost = np.mean(group_costs)
        mean_fidelity = np.mean(group_fidelities)
        std_cost = np.std(group_costs)
        std_fidelity = np.std(group_fidelities)

        color = DURATION_COLORS.get(duration, "black")
        marker = DURATION_MARKERS.get(duration, "o")

        # Plot error bars (mean ± std)
        plt.errorbar(
            x=mean_cost,
            y=mean_fidelity,
            xerr=std_cost,
            yerr=std_fidelity,
            marker=marker,
            markersize=5,
            label=f"GenT {duration} CPU (mean ± std)",
            color=color,
            alpha=0.8,
            capsize=5,
            capthick=2,
            linewidth=2,
        )

        # Plot individual points
        plt.scatter(
            x=group_costs,
            y=group_fidelities,
            marker=marker,
            s=25,
            color=color,
            alpha=0.4,
            edgecolors="black",
            linewidth=1,
        )

        # Annotate mean
        plt.annotate(
            f"GenT {duration} CPU",
            (mean_cost, mean_fidelity),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.7,
            },
            arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.5},
        )


def plot_head_sampling_points(head_sampling_points):
    """
    Plot head sampling points with diamond markers.

    Args:
        head_sampling_points: list [(cost, fidelity, compressor), ...]
    """
    if not head_sampling_points:
        return

    hs_costs = [d[0] for d in head_sampling_points]
    hs_fidelities = [d[1] for d in head_sampling_points]
    hs_names = [d[2].split("_")[-1] for d in head_sampling_points]  # Extract ratio

    plt.scatter(
        x=hs_costs,
        y=hs_fidelities,
        marker="D",
        s=120,
        label="Head Sampling",
        alpha=0.8,
        edgecolors="black",
        linewidth=1,
    )

    # Annotate each head sampling point
    for cost, fidelity, name in zip(hs_costs, hs_fidelities, hs_names, strict=False):
        plt.annotate(
            f"1:{name}",
            (cost, fidelity),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.7,
            },
            arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.5},
        )


def format_plot_axes(title, metric, filter_level):
    """
    Format plot axes, title, labels, and grid.

    Args:
        title: Base title (e.g., "Rate Over Time", "Duration Over Time")
        metric: Metric name ('mape_fidelity' or 'cosine_fidelity')
        filter_level: Filter level (0, 1, or 2)
    """
    metric_name = "MAPE" if metric == "mape_fidelity" else "Cosine Similarity"
    filter_desc = {
        0: "0 filters",
        1: "1 filter",
        2: "2 filters",
    }

    plt.title(
        f"{title}: {metric_name} vs Cost - {filter_desc[filter_level]}",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Cost per Million Spans ($)", fontsize=14, fontweight="bold")
    plt.ylabel(f"{metric_name} Fidelity (%)", fontsize=14, fontweight="bold")
    plt.xscale("log")
    plt.ylim(0, 100)

    # Legend
    plt.legend(
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=11,
    )

    # Grid
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.5)
