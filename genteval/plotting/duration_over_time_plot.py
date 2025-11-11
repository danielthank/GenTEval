"""Duration over time visualization module."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from genteval.plotting.data import ReportParser
from genteval.plotting.scatter_plot_utils import (
    build_filter_title,
    format_plot_axes,
    group_experiments_by_duration,
    plot_duration_groups,
    plot_head_sampling_points,
    setup_plot_style,
)


def extract_duration_data_by_filter(
    report_data, experiments, filter_level, weighted=False
):
    """
    Extract duration over time data for specific filter level.

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
    if (
        "reports" not in report_data
        or "duration_over_time" not in report_data["reports"]
    ):
        return []

    duration_data = report_data["reports"]["duration_over_time"]

    # Create lookup for cost by compressor key
    cost_lookup = {}
    for exp in experiments:
        cost_lookup[exp.compressor_key] = exp.total_cost_per_million_spans

    if filter_level == 0:
        # For filter level 0, just use "all" group
        data_points = []
        for compressor_key, groups in duration_data.items():
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
    for compressor_key, groups in duration_data.items():
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


def plot_duration_scatter(data_points, metric, filter_level, output_dir):
    """
    Create scatter plot of fidelity vs cost.

    Args:
        data_points: List of data points from extract_duration_data_by_filter
        metric: 'mape_fidelity' or 'cosine_fidelity'
        filter_level: 0, 1, or 2
        output_dir: Directory to save plot
    """
    if not data_points:
        print(f"No data points for filter level {filter_level}, metric {metric}")
        return

    # Setup plot
    setup_plot_style()

    # Group data by duration for GenT experiments
    duration_groups, head_sampling_points = group_experiments_by_duration(
        data_points, metric
    )

    # Plot GenT duration groups and head sampling
    plot_duration_groups(duration_groups)
    plot_head_sampling_points(head_sampling_points)

    # Format axes, title, labels, and grid
    metric_name = "MAPE" if metric == "mape_fidelity" else "Cosine Similarity"
    title = build_filter_title("Duration Over Time", metric, filter_level)
    y_label = f"{metric_name} Fidelity (%)"
    format_plot_axes(title, y_label)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metric_short = "mape" if metric == "mape_fidelity" else "cosine"
    filter_suffix = {0: "0_filter", 1: "1_filter", 2: "2_filters"}
    filename = f"duration_over_time_{filter_suffix[filter_level]}_{metric_short}.png"

    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path / filename}")


def generate_all_duration_plots(report_path, output_dir="./plots", weighted=False):
    """
    Generate all 6 duration over time scatter plots.

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
        data_points = extract_duration_data_by_filter(
            report_data, experiments, filter_level, weighted=weighted
        )
        print(f"Filter level {filter_level}: {len(data_points)} data points")

        for metric in ["mape_fidelity", "cosine_fidelity"]:
            plot_duration_scatter(data_points, metric, filter_level, output_dir)

    print(f"All duration over time plots generated in {output_dir}")
