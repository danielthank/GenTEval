"""Graph fidelity visualization module."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from genteval.plotting.data import ReportParser
from genteval.plotting.scatter_plot_utils import (
    format_plot_axes,
    group_experiments_by_name,
    plot_experiment_groups,
    plot_head_sampling_points,
    setup_plot_style,
)


def extract_graph_data(report_data, experiments):
    """
    Extract graph fidelity data from report.

    Args:
        report_data: Full report JSON data
        experiments: List of ExperimentData from ReportParser

    Returns:
        List of data points: [{
            'compressor': str,
            'fidelity': float,
            'cost_per_million': float
        }]
    """
    if "reports" not in report_data or "graph" not in report_data["reports"]:
        return []

    graph_report = report_data["reports"]["graph"]

    # Create lookup for cost by compressor key
    cost_lookup = {}
    for exp in experiments:
        cost_lookup[exp.compressor_key] = exp.total_cost_per_million_spans

    data_points = []
    for compressor_key, metrics in graph_report.items():
        cost = cost_lookup.get(compressor_key, 0)

        # Extract graph fidelity by averaging across time buckets
        fidelity_values = []
        for metric_name, metric_value in metrics.items():
            if metric_name.startswith("time_") and isinstance(metric_value, dict):
                if "fidelity" in metric_value:
                    fidelity_values.append(metric_value["fidelity"])

        # Calculate average fidelity across all time buckets
        if fidelity_values:
            avg_fidelity = sum(fidelity_values) / len(fidelity_values)
        else:
            avg_fidelity = 0.0

        data_points.append(
            {
                "compressor": compressor_key,
                "fidelity": avg_fidelity,
                "cost_per_million": cost,
            }
        )

    return data_points


def plot_graph_scatter(data_points, output_dir):
    """
    Create scatter plot of graph fidelity vs cost.

    Args:
        data_points: List of data points from extract_graph_data
        output_dir: Directory to save plot
    """
    if not data_points:
        print("No data points for graph fidelity")
        return

    # Setup plot
    setup_plot_style()

    # Group data by duration for GenT experiments
    experiment_groups, head_sampling_points = group_experiments_by_name(
        data_points, "fidelity"
    )

    # Plot GenT duration groups and head sampling
    plot_experiment_groups(experiment_groups)
    plot_head_sampling_points(head_sampling_points)

    # Format axes, title, labels, and grid
    title = "Graph Fidelity vs Cost"
    y_label = "Graph Fidelity (%)"
    format_plot_axes(title, y_label)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = "graph_fidelity.png"

    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path / filename}")


def generate_all_graph_plots(report_path, output_dir="./plots"):
    """
    Generate graph fidelity scatter plot.

    Args:
        report_path: Path to the report JSON file
        output_dir: Directory to save plots
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

    # Generate graph fidelity plot
    data_points = extract_graph_data(report_data, experiments)
    print(f"graph_fidelity: {len(data_points)} data points")

    if data_points:
        plot_graph_scatter(data_points, output_dir)

    print(f"All graph plots generated in {output_dir}")
