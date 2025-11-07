"""Rate over time visualization module."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from genteval.plotting.data import ReportParser


def extract_rate_data_by_filter(report_data, experiments, filter_level, weighted=False):
    """
    Extract rate over time data for specific filter level.

    Args:
        report_data: Full report JSON data
        experiments: List of ExperimentData from ReportParser
        filter_level: 0 (all only), 1 (single dimension), 2 (combinations)
        weighted: If True, use weighted average by span count for levels 1 and 2

    Returns:
        List of data points: [{
            'compressor': str,
            'group': str,
            'mape_fidelity': float,
            'cosine_fidelity': float,
            'count': int,
            'cost_per_million': float
        }]
    """
    if "reports" not in report_data or "rate_over_time" not in report_data["reports"]:
        return []

    rate_data = report_data["reports"]["rate_over_time"]

    # Create lookup for cost by compressor key
    cost_lookup = {}
    for exp in experiments:
        cost_lookup[exp.compressor_key] = exp.total_cost_per_million_spans

    if filter_level == 0:
        # For filter level 0, just use "all" group as before
        data_points = []
        for compressor_key, groups in rate_data.items():
            cost = cost_lookup.get(compressor_key, 0)
            if "all" in groups:
                metrics = groups["all"]
                data_points.append(
                    {
                        "compressor": compressor_key,
                        "group": "all",
                        "mape_fidelity": metrics.get("mape_fidelity", 0),
                        "cosine_fidelity": metrics.get("cosine_fidelity", 0),
                        "count": metrics.get("count", 0),
                        "cost_per_million": cost,
                    }
                )
        return data_points

    # For filter levels 1 and 2, compute average across all groups
    data_points = []
    for compressor_key, groups in rate_data.items():
        cost = cost_lookup.get(compressor_key, 0)

        mape_values = []
        cosine_values = []
        count_values = []

        for group_key, metrics in groups.items():
            include = False

            if filter_level == 1:
                # Single dimension: has ":" but not "!@#", and not "all"
                include = (
                    ":" in group_key and "!@#" not in group_key and group_key != "all"
                )
            elif filter_level == 2:
                # Combinations: has "!@#"
                include = "!@#" in group_key

            if include:
                mape_values.append(metrics.get("mape_fidelity", 0))
                cosine_values.append(metrics.get("cosine_fidelity", 0))
                count_values.append(metrics.get("count", 0))

        # Average across all groups for this compressor
        if mape_values:
            if weighted:
                # Weighted average by span count
                weights = np.array(count_values)
                avg_mape = np.average(mape_values, weights=weights)
                avg_cosine = np.average(cosine_values, weights=weights)
            else:
                # Simple average
                avg_mape = np.mean(mape_values)
                avg_cosine = np.mean(cosine_values)

            data_points.append(
                {
                    "compressor": compressor_key,
                    "group": f"average_of_{len(mape_values)}_groups",
                    "mape_fidelity": avg_mape,
                    "cosine_fidelity": avg_cosine,
                    "count": sum(count_values),
                    "cost_per_million": cost,
                }
            )

    return data_points


def plot_rate_scatter(data_points, metric, filter_level, output_dir):
    """
    Create scatter plot of fidelity vs cost.

    Args:
        data_points: List of data points from extract_rate_data_by_filter
        metric: 'mape_fidelity' or 'cosine_fidelity'
        filter_level: 0, 1, or 2
        output_dir: Directory to save plot
    """
    if not data_points:
        print(f"No data points for filter level {filter_level}, metric {metric}")
        return

    # Setup plot
    plt.figure(figsize=(12, 8))
    plt.style.use("default")
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14

    # Group data by duration for GenT experiments
    duration_groups = {}  # {duration: [(cost, fidelity, compressor), ...]}
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

    # Color scheme matching draw.py
    colors = {"1min": "red", "5min": "blue", "10min": "green"}
    markers = {"1min": "o", "5min": "s", "10min": "^"}

    # Plot GenT duration groups with mean ± std
    for duration in sorted(duration_groups.keys()):
        group_data = duration_groups[duration]
        group_costs = [d[0] for d in group_data]
        group_fidelities = [d[1] for d in group_data]

        mean_cost = np.mean(group_costs)
        mean_fidelity = np.mean(group_fidelities)
        std_cost = np.std(group_costs)
        std_fidelity = np.std(group_fidelities)

        color = colors.get(duration, "black")
        marker = markers.get(duration, "o")

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

    # Plot head sampling points
    if head_sampling_points:
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
        for cost, fidelity, name in zip(
            hs_costs, hs_fidelities, hs_names, strict=False
        ):
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

    # Labels and title
    metric_name = "MAPE" if metric == "mape_fidelity" else "Cosine Similarity"
    filter_desc = {
        0: "0 filters",
        1: "1 filter",
        2: "2 filters",
    }

    plt.title(
        f"Rate Over Time: {metric_name} vs Cost - {filter_desc[filter_level]}",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Cost per Million Spans ($)", fontsize=14, fontweight="bold")
    plt.ylabel(f"{metric_name} Fidelity (%)", fontsize=14, fontweight="bold")
    plt.xscale("log")
    plt.ylim(0, 100)

    # Legend is created automatically from the plot labels
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

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metric_short = "mape" if metric == "mape_fidelity" else "cosine"
    filter_suffix = {0: "0_filter", 1: "1_filter", 2: "2_filters"}
    filename = f"rate_over_time_{filter_suffix[filter_level]}_{metric_short}.png"

    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path / filename}")


def generate_all_rate_plots(report_path, output_dir="./plots", weighted=False):
    """
    Generate all 6 rate over time scatter plots.

    Args:
        report_path: Path to the report JSON file
        output_dir: Directory to save plots
        weighted: If True, use weighted average by span count for filter levels 1 and 2
    """
    # Load report data
    with open(report_path) as f:
        report_data = json.load(f)

    # Use ReportParser to get experiment data with costs
    parser = ReportParser()
    experiments = parser.parse_report(report_path)

    if not experiments:
        print("No experiments found in report")
        return

    print(f"Found {len(experiments)} experiments")

    # Generate plots for each filter level and metric
    for filter_level in [0, 1, 2]:
        data_points = extract_rate_data_by_filter(
            report_data, experiments, filter_level, weighted=weighted
        )
        print(f"Filter level {filter_level}: {len(data_points)} data points")

        for metric in ["mape_fidelity", "cosine_fidelity"]:
            plot_rate_scatter(data_points, metric, filter_level, output_dir)

    print(f"All rate over time plots generated in {output_dir}")
