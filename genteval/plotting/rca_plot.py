"""RCA (Root Cause Analysis) visualization module for TraceRCA and MicroRank."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from genteval.plotting.data import ReportParser
from genteval.plotting.scatter_plot_utils import (
    format_plot_axes,
    group_experiments_by_duration,
    plot_duration_groups,
    plot_head_sampling_points,
    setup_plot_style,
)


def extract_rca_data(report_data, experiments, metric_name):
    """
    Extract RCA data for a specific metric (tracerca or microrank).

    Args:
        report_data: Full report JSON data
        experiments: List of ExperimentData from ReportParser
        metric_name: 'tracerca' or 'microrank'

    Returns:
        List of data points: [{
            'compressor': str,
            'avg5_fidelity': float,
            'cost_per_million': float
        }]
    """
    report_key = "trace_rca" if metric_name == "tracerca" else "micro_rank"

    if "reports" not in report_data or report_key not in report_data["reports"]:
        return []

    rca_report = report_data["reports"][report_key]

    # Create lookup for cost by compressor key
    cost_lookup = {}
    for exp in experiments:
        cost_lookup[exp.compressor_key] = exp.total_cost_per_million_spans

    data_points = []
    for compressor_key, metrics in rca_report.items():
        cost = cost_lookup.get(compressor_key, 0)

        # Extract avg5 metric
        avg5 = metrics.get("avg5", 0.0)

        # Convert to percentage (multiply by 100)
        avg5_fidelity = avg5 * 100

        data_points.append(
            {
                "compressor": compressor_key,
                "avg5_fidelity": avg5_fidelity,
                "cost_per_million": cost,
            }
        )

    return data_points


def plot_rca_scatter(data_points, metric_name, output_dir):
    """
    Create scatter plot of RCA fidelity vs cost.

    Args:
        data_points: List of data points from extract_rca_data
        metric_name: 'tracerca' or 'microrank'
        output_dir: Directory to save plot
    """
    if not data_points:
        print(f"No data points for {metric_name}")
        return

    # Setup plot
    setup_plot_style()

    # Group data by duration for GenT experiments
    duration_groups, head_sampling_points = group_experiments_by_duration(
        data_points, "avg5_fidelity"
    )

    # Plot GenT duration groups and head sampling
    plot_duration_groups(duration_groups)
    plot_head_sampling_points(head_sampling_points)

    # Format axes, title, labels, and grid
    metric_display = "TraceRCA" if metric_name == "tracerca" else "MicroRank"
    title = f"{metric_display} Avg@5 Fidelity vs Cost"
    y_label = f"{metric_display} Avg@5 Fidelity (%)"
    format_plot_axes(title, y_label)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"rca_{metric_name}_avg5.png"

    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path / filename}")


def generate_all_rca_plots(report_path, output_dir="./plots"):
    """
    Generate RCA scatter plots for both TraceRCA and MicroRank.

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

    # Generate plots for both TraceRCA and MicroRank
    for metric_name in ["tracerca", "microrank"]:
        data_points = extract_rca_data(report_data, experiments, metric_name)
        print(f"{metric_name}: {len(data_points)} data points")

        if data_points:
            plot_rca_scatter(data_points, metric_name, output_dir)

    print(f"All RCA plots generated in {output_dir}")
