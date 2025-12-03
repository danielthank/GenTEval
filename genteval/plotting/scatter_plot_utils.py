"""Shared utilities for scatter plot visualization."""

import matplotlib.pyplot as plt
import numpy as np


# Experiment name to display name mapping
EXPERIMENT_DISPLAY_NAMES = {
    "rootcount-stratified": "Root Count + Stratified",
    "rootcount-focal": "Root Count + Focal Loss",
    "rootcount": "Root Count",
    "prev": "GenT (Previous)",
}

# Color scheme for GenT experiments
EXPERIMENT_COLORS = {
    "rootcount-stratified": "red",
    "rootcount-focal": "green",
    "rootcount": "blue",
    "prev": "orange",
}
EXPERIMENT_MARKERS = {
    "rootcount-stratified": "o",
    "rootcount-focal": "^",
    "rootcount": "s",
    "prev": "p",
}

# Head sampling style (consistent color for all rates)
HEAD_SAMPLING_MARKER = "D"
HEAD_SAMPLING_COLOR = "purple"


def setup_plot_style():
    """Setup plot figure and style."""
    plt.figure(figsize=(12, 8))
    plt.style.use("default")
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14


def get_display_name(experiment_name: str) -> str:
    """Get display name for an experiment, using mapping or the raw name."""
    return EXPERIMENT_DISPLAY_NAMES.get(experiment_name, experiment_name)


def group_experiments_by_name(data_points, metric):
    """
    Group experiments by experiment name for GenT and by sampling rate for head sampling.

    Args:
        data_points: List of data points with 'compressor', 'cost_per_million', and metric field
        metric: Name of the metric field to extract (e.g., 'mape_fidelity')

    Returns:
        Tuple of (experiment_groups, head_sampling_groups)
        - experiment_groups: dict {experiment_name: [(cost, fidelity, compressor), ...]}
        - head_sampling_groups: dict {sampling_rate: [(cost, fidelity, compressor), ...]}
    """
    experiment_groups = {}
    head_sampling_groups = {}

    for point in data_points:
        compressor = point["compressor"]
        cost = point["cost_per_million"]
        fidelity = point[metric]

        parts = compressor.split("_")

        # Format: {prefix}_gent_{experiment_name}_{iteration}
        # or: {prefix}_head_{sampling_rate}_{iteration}
        if "gent" in parts:
            gent_idx = parts.index("gent")
            if gent_idx + 2 < len(parts):
                experiment_name = parts[gent_idx + 1]
                if experiment_name not in experiment_groups:
                    experiment_groups[experiment_name] = []
                experiment_groups[experiment_name].append((cost, fidelity, compressor))
        elif "head" in parts:
            head_idx = parts.index("head")
            if head_idx + 2 < len(parts):
                sampling_rate = parts[head_idx + 1]
                if sampling_rate not in head_sampling_groups:
                    head_sampling_groups[sampling_rate] = []
                head_sampling_groups[sampling_rate].append((cost, fidelity, compressor))

    return experiment_groups, head_sampling_groups


def plot_experiment_groups(experiment_groups):
    """
    Plot GenT experiment groups with mean ± std error bars.

    Args:
        experiment_groups: dict {experiment_name: [(cost, fidelity, compressor), ...]}
    """
    for experiment_name in sorted(experiment_groups.keys()):
        group_data = experiment_groups[experiment_name]
        group_costs = [d[0] for d in group_data]
        group_fidelities = [d[1] for d in group_data]

        mean_cost = np.mean(group_costs)
        mean_fidelity = np.mean(group_fidelities)
        std_cost = np.std(group_costs)
        std_fidelity = np.std(group_fidelities)

        color = EXPERIMENT_COLORS.get(experiment_name, "black")
        marker = EXPERIMENT_MARKERS.get(experiment_name, "o")
        display_name = get_display_name(experiment_name)

        # Plot error bars (mean ± std)
        plt.errorbar(
            x=mean_cost,
            y=mean_fidelity,
            xerr=std_cost,
            yerr=std_fidelity,
            marker=marker,
            markersize=5,
            label=f"{display_name} (mean ± std)",
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
            display_name,
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


def plot_head_sampling_points(head_sampling_groups):
    """
    Plot head sampling groups with mean ± std error bars.

    Args:
        head_sampling_groups: dict {sampling_rate: [(cost, fidelity, compressor), ...]}
    """
    if not head_sampling_groups:
        return

    # Sort by sampling rate (as int for proper ordering)
    sorted_rates = sorted(head_sampling_groups.keys(), key=lambda x: int(x))
    first_rate = True

    for rate in sorted_rates:
        group_data = head_sampling_groups[rate]
        group_costs = [d[0] for d in group_data]
        group_fidelities = [d[1] for d in group_data]

        mean_cost = np.mean(group_costs)
        mean_fidelity = np.mean(group_fidelities)
        std_cost = np.std(group_costs)
        std_fidelity = np.std(group_fidelities)

        display_name = f"1:{rate}"

        # Plot error bars (mean ± std)
        # Only add legend label for the first rate to avoid duplicate "Head Sampling" entries
        plt.errorbar(
            x=mean_cost,
            y=mean_fidelity,
            xerr=std_cost,
            yerr=std_fidelity,
            marker=HEAD_SAMPLING_MARKER,
            markersize=5,
            label="Head Sampling (mean ± std)" if first_rate else None,
            color=HEAD_SAMPLING_COLOR,
            alpha=0.8,
            capsize=5,
            capthick=2,
            linewidth=2,
        )

        # Plot individual points
        plt.scatter(
            x=group_costs,
            y=group_fidelities,
            marker=HEAD_SAMPLING_MARKER,
            s=25,
            color=HEAD_SAMPLING_COLOR,
            alpha=0.4,
            edgecolors="black",
            linewidth=1,
        )

        # Annotate mean
        plt.annotate(
            display_name,
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

        first_rate = False


def format_plot_axes(title, y_label):
    """
    Format plot axes, title, labels, and grid.

    Args:
        title: Full title for the plot (e.g., "Rate Over Time: MAPE vs Cost - 0 filters")
        y_label: Y-axis label (e.g., "MAPE Fidelity (%)", "TraceRCA Avg@5 Fidelity (%)")
    """
    plt.title(
        title,
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Cost per Million Spans ($)", fontsize=14, fontweight="bold")
    plt.ylabel(y_label, fontsize=14, fontweight="bold")
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


def build_filter_title(base_title, metric, filter_level):
    """
    Build title for rate/duration plots with filter levels.

    Args:
        base_title: Base title (e.g., "Rate Over Time", "Duration Over Time")
        metric: Metric name ('mape_fidelity' or 'cosine_fidelity')
        filter_level: Filter level (0, 1, or 2)

    Returns:
        Full formatted title string
    """
    metric_name = "MAPE" if metric == "mape_fidelity" else "Cosine Similarity"
    filter_desc = {
        0: "0 filters",
        1: "1 filter",
        2: "2 filters",
    }
    return f"{base_title}: {metric_name} vs Cost - {filter_desc[filter_level]}"
