import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from genteval.plotting.data import ReportParser
from genteval.plotting.duration_over_time_plot import generate_all_duration_plots
from genteval.plotting.rate_over_time_plot import generate_all_rate_plots
from genteval.plotting.rca_plot import generate_all_rca_plots


# Hardcoded data arrays removed - now using JSON data source


def extract_mape_data_for_heatmap(report_data, compressor_key):
    """Extract MAPE data from the JSON report for a specific compressor."""
    if "reports" not in report_data or "duration" not in report_data["reports"]:
        raise ValueError("Invalid report format: missing duration data")

    duration_data = report_data["reports"]["duration"]

    # Find the matching compressor (could be part of a compound key like "app_compressor")
    matching_keys = [key for key in duration_data.keys() if compressor_key in key]
    if not matching_keys:
        available_keys = list(duration_data.keys())
        raise ValueError(
            f"Compressor '{compressor_key}' not found. Available: {available_keys}"
        )

    if len(matching_keys) > 1:
        print(f"Warning: Multiple matches for '{compressor_key}': {matching_keys}")
        print(f"Using first match: {matching_keys[0]}")

    compressor_data = duration_data[matching_keys[0]]

    # Extract MAPE metrics - pattern: depth_{X}_p{Y}_mape
    mape_data = {}
    for metric_name, metric_value in compressor_data.items():
        if "_mape" in metric_name and "depth_" in metric_name:
            parts = metric_name.split("_")

            # Find depth and percentile
            depth_idx = None
            percentile_idx = None

            for i, part in enumerate(parts):
                if part == "depth" and i + 1 < len(parts):
                    try:
                        depth_idx = int(parts[i + 1])
                    except ValueError:
                        continue
                elif part.startswith("p") and part[1:].isdigit():
                    percentile_idx = int(part[1:])

            if depth_idx is not None and percentile_idx is not None:
                if depth_idx not in mape_data:
                    mape_data[depth_idx] = {}
                mape_data[depth_idx][percentile_idx] = metric_value["avg"]

    return mape_data, matching_keys[0]


def extract_mape_data_for_status_code_heatmap(report_data, compressor_key):
    """Extract MAPE data for status codes from the JSON report for a specific compressor."""
    if "reports" not in report_data or "duration" not in report_data["reports"]:
        raise ValueError("Invalid report format: missing duration data")

    duration_data = report_data["reports"]["duration"]

    # Find the matching compressor (could be part of a compound key like "app_compressor")
    matching_keys = [key for key in duration_data.keys() if compressor_key in key]
    if not matching_keys:
        available_keys = list(duration_data.keys())
        raise ValueError(
            f"Compressor '{compressor_key}' not found. Available: {available_keys}"
        )

    if len(matching_keys) > 1:
        print(f"Warning: Multiple matches for '{compressor_key}': {matching_keys}")
        print(f"Using first match: {matching_keys[0]}")

    compressor_data = duration_data[matching_keys[0]]

    # Extract MAPE metrics - pattern: http.status_code_{code}_p{Y}_mape
    mape_data = {}
    for metric_name, metric_value in compressor_data.items():
        if "_mape" in metric_name and "http.status_code_" in metric_name:
            # Pattern: http.status_code_{code}_p{percentile}_mape
            # When split by "_": ['http.status', 'code', '{code}', 'p{percentile}', 'mape']
            parts = metric_name.split("_")

            if len(parts) >= 5:
                # Status code is at index 2 (after 'http.status' and 'code')
                status_code = parts[2]

                # Percentile is at index 3 (starts with 'p')
                percentile_part = parts[3]
                if percentile_part.startswith("p") and percentile_part[1:].isdigit():
                    percentile_idx = int(percentile_part[1:])

                    if status_code not in mape_data:
                        mape_data[status_code] = {}
                    mape_data[status_code][percentile_idx] = metric_value["avg"]

    return mape_data, matching_keys[0]


def create_mape_heatmap(mape_data, compressor_name, output_dir):
    """Create a 2D heatmap of MAPE values across depth and percentiles."""

    # Define expected depths and percentiles
    depths = list(range(5))  # 0, 1, 2, 3, 4
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create 2D array for heatmap
    heatmap_data = np.full((len(depths), len(percentiles)), np.nan)

    # Fill the array with available data
    for depth_idx, depth in enumerate(depths):
        if depth in mape_data:
            for perc_idx, percentile in enumerate(percentiles):
                if percentile in mape_data[depth]:
                    heatmap_data[depth_idx, perc_idx] = (
                        100 - mape_data[depth][percentile]
                    )

    # Create the heatmap
    plt.figure(figsize=(12, 8))

    # Use a colormap where lower values (better MAPE) are green
    mask = np.isnan(heatmap_data)

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",  # Red-Yellow-Green reversed (lower is better)
        mask=mask,
        xticklabels=[f"p{p}" for p in percentiles],
        yticklabels=[f"Depth {d}" for d in depths],
        square=True,
        cbar=False,
        vmin=0,
        vmax=100,
    )

    plt.title(
        f"MAPE Heatmap: {compressor_name}\n(Higher values = Better fidelity)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Percentiles", fontsize=14, fontweight="bold")
    plt.ylabel("Span Depth", fontsize=14, fontweight="bold")

    # Improve layout
    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / f"{compressor_name}_mape_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        facecolor="white",
        edgecolor="none",
    )

    print(f"Saved MAPE heatmap to: {output_path.resolve()}")
    plt.close()

    return output_path


def create_status_code_mape_heatmap(mape_data, compressor_name, output_dir):
    """Create a 2D heatmap of MAPE values across HTTP status codes and percentiles."""

    # Get unique status codes and sort them
    status_codes = sorted(mape_data.keys())
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create 2D array for heatmap
    heatmap_data = np.full((len(status_codes), len(percentiles)), np.nan)

    # Fill the array with available data
    for code_idx, status_code in enumerate(status_codes):
        if status_code in mape_data:
            for perc_idx, percentile in enumerate(percentiles):
                if percentile in mape_data[status_code]:
                    heatmap_data[code_idx, perc_idx] = (
                        100 - mape_data[status_code][percentile]
                    )

    # Create the heatmap
    plt.figure(figsize=(12, 8))

    # Use a colormap where lower values (better MAPE) are green
    mask = np.isnan(heatmap_data)

    # Create labels for status codes (use "empty" for empty string)
    status_code_labels = [code if code else "empty" for code in status_codes]

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",  # Red-Yellow-Green reversed (lower is better)
        mask=mask,
        xticklabels=[f"p{p}" for p in percentiles],
        yticklabels=status_code_labels,
        square=True,
        cbar=False,
        vmin=0,
        vmax=100,
    )

    plt.title(
        f"MAPE Heatmap by HTTP Status Code: {compressor_name}\n(Higher values = Better fidelity)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Percentiles", fontsize=14, fontweight="bold")
    plt.ylabel("HTTP Status Code", fontsize=14, fontweight="bold")

    # Improve layout
    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / f"{compressor_name}_status_code_mape_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        facecolor="white",
        edgecolor="none",
    )

    print(f"Saved status code MAPE heatmap to: {output_path.resolve()}")
    plt.close()

    return output_path


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate visualizations from GenTEval reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "scatter",
            "scatter_graph",
            "heatmap",
            "heatmap_status_code",
            "rate_over_time",
            "duration_over_time",
            "rca",
        ],
        default="scatter",
        help="Visualization mode: scatter (GenT CPU 1min/5min/10min vs head sampling, default), scatter_graph (graph fidelity scatter plot), heatmap (depth heatmap), heatmap_status_code (HTTP status code heatmap), rate_over_time (rate over time fidelity vs cost), duration_over_time (duration over time fidelity vs cost), or rca (TraceRCA and MicroRank fidelity vs cost)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/report.json",
        help="Path to JSON report file (default: output/report.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/visualizations",
        help="Output directory for generated plots (default: output/visualizations)",
    )
    parser.add_argument(
        "--compressor",
        type=str,
        help="Compressor name to visualize (required for heatmap modes)",
    )

    args = parser.parse_args()

    if args.mode == "heatmap":
        # Heatmap mode - requires input JSON and compressor
        if not args.input:
            parser.error("--input is required for heatmap mode")
        if not args.compressor:
            parser.error("--compressor is required for heatmap mode")

        # Load the JSON report
        try:
            with open(args.input) as f:
                report_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find input file: {args.input}")
            return 1
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in input file: {e}")
            return 1

        # Extract MAPE data and create heatmap
        try:
            mape_data, full_compressor_name = extract_mape_data_for_heatmap(
                report_data, args.compressor
            )
            create_mape_heatmap(mape_data, full_compressor_name, args.output_dir)
            print(
                f"Successfully generated heatmap for compressor: {full_compressor_name}"
            )
            return 0
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    elif args.mode == "heatmap_status_code":
        # Status code heatmap mode - requires input JSON and compressor
        if not args.input:
            parser.error("--input is required for heatmap_status_code mode")
        if not args.compressor:
            parser.error("--compressor is required for heatmap_status_code mode")

        # Load the JSON report
        try:
            with open(args.input) as f:
                report_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find input file: {args.input}")
            return 1
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in input file: {e}")
            return 1

        # Extract MAPE data for status codes and create heatmap
        try:
            mape_data, full_compressor_name = extract_mape_data_for_status_code_heatmap(
                report_data, args.compressor
            )
            create_status_code_mape_heatmap(
                mape_data, full_compressor_name, args.output_dir
            )
            print(
                f"Successfully generated status code heatmap for compressor: {full_compressor_name}"
            )
            return 0
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    elif args.mode == "scatter_graph":
        # Graph fidelity scatter plot mode
        return _create_graph_fidelity_scatter_plot(args.input, args.output_dir)

    elif args.mode == "rate_over_time":
        # Rate over time scatter plots mode
        generate_all_rate_plots(args.input, output_dir=args.output_dir, weighted=True)
        print(f"Rate over time scatter plots generated in {args.output_dir}")
        return 0

    elif args.mode == "duration_over_time":
        # Duration over time scatter plots mode
        generate_all_duration_plots(
            args.input, output_dir=args.output_dir, weighted=True
        )
        print(f"Duration over time scatter plots generated in {args.output_dir}")
        return 0

    elif args.mode == "rca":
        # RCA scatter plots mode (TraceRCA and MicroRank)
        generate_all_rca_plots(args.input, output_dir=args.output_dir)
        print(f"RCA scatter plots generated in {args.output_dir}")
        return 0

    else:
        # Default scatter plot mode: GenT CPU vs head sampling
        return _create_gent_vs_head_sampling_plot(args.input, args.output_dir)


def _create_graph_fidelity_scatter_plot(input_file, output_dir):
    """Create scatter plot for graph fidelity scores."""
    from genteval.plotting.data import CostConfig, ReportParser

    # Parse the report
    parser = ReportParser(cost_config=CostConfig())
    experiments = parser.parse_report(input_file)

    if not experiments:
        print("Error: No experiments found in report")
        return 1

    # Filter experiments with graph fidelity data (>= 0, not just > 0)
    valid_experiments = [e for e in experiments if e.graph_fidelity >= 0]

    if not valid_experiments:
        print("Error: No experiments with graph fidelity data found")
        return 1

    # Create output directory
    import pathlib

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Separate GenT CPU and head sampling experiments by duration
    # Handle gent_{min}_{run} naming pattern (e.g., gent_1_2 = 1min, run 2)
    gent_1min = []
    gent_5min = []
    gent_10min = []

    for e in valid_experiments:
        if "gent" in e.compressor_key and "_gent_" in e.compressor_key:
            # Extract duration from app_gent_{min}_{run} pattern
            # e.g., otel-demo-transformed_gent_1_2
            parts = e.compressor_key.split("_gent_")
            if len(parts) == 2:
                gent_parts = parts[1].split("_")
                if len(gent_parts) >= 2:
                    try:
                        duration_min = int(gent_parts[0])
                        if duration_min == 1:
                            gent_1min.append(e)
                        elif duration_min == 5:
                            gent_5min.append(e)
                        elif duration_min == 10:
                            gent_10min.append(e)
                    except (ValueError, IndexError):
                        pass

    head_sampling_exps = [
        e
        for e in valid_experiments
        if "head_sampling" in e.compressor_key and not e.compressor_key.endswith("_1")
    ]

    # Prepare data for plotting
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 8))
    plt.style.use("default")
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14

    # Color scheme matching the standard scatter plot
    colors = {"1min": "red", "5min": "blue", "10min": "green"}
    markers = {"1min": "o", "5min": "s", "10min": "^"}

    # Plot GenT experiments by duration
    duration_data = [
        ("1min", gent_1min, "GenT 1min CPU"),
        ("5min", gent_5min, "GenT 5min CPU"),
        ("10min", gent_10min, "GenT 10min CPU"),
    ]

    for duration, exps, label in duration_data:
        if exps:
            x_vals = [e.total_cost_per_million_spans for e in exps]
            y_vals = [e.graph_fidelity for e in exps]

            color = colors[duration]
            marker = markers[duration]

            # Calculate mean and std
            mean_x = np.mean(x_vals)
            mean_y = np.mean(y_vals)
            std_x = np.std(x_vals)
            std_y = np.std(y_vals)

            # Plot error bars (mean ± std)
            plt.errorbar(
                x=mean_x,
                y=mean_y,
                xerr=std_x,
                yerr=std_y,
                marker=marker,
                markersize=5,
                label=f"{label} (mean ± std)",
                color=color,
                alpha=0.8,
                capsize=5,
                capthick=2,
                linewidth=2,
            )

            # Plot individual points
            plt.scatter(
                x=x_vals,
                y=y_vals,
                marker=marker,
                s=25,
                color=color,
                alpha=0.4,
                edgecolors="black",
                linewidth=1,
            )

            # Annotate mean
            plt.annotate(
                label,
                (mean_x, mean_y),
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

    # Plot head sampling experiments
    if head_sampling_exps:
        x_vals = [e.total_cost_per_million_spans for e in head_sampling_exps]
        y_vals = [e.graph_fidelity for e in head_sampling_exps]
        names = [e.name for e in head_sampling_exps]

        plt.scatter(
            x=x_vals,
            y=y_vals,
            marker="D",
            s=120,
            label="Head Sampling",
            alpha=0.8,
            edgecolors="black",
            linewidth=1,
        )

        # Annotate each head sampling point
        for x, y, name in zip(x_vals, y_vals, names, strict=False):
            plt.annotate(
                name,
                (x, y),
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

    # Formatting
    plt.title("Graph Fidelity vs Cost", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Cost per Million Spans ($)", fontsize=14, fontweight="bold")
    plt.ylabel("Graph Fidelity Score (%)", fontsize=14, fontweight="bold")
    plt.xscale("log")
    plt.ylim(0, 105)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.5)
    plt.legend(
        loc="best",
        fontsize=10,
        frameon=True,
        shadow=True,
        fancybox=True,
        framealpha=0.9,
    )
    plt.tight_layout()

    # Save plot
    output_file = output_path / "gent_vs_head_sampling_graph_fidelity.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Successfully generated graph fidelity scatter plot: {output_file}")
    return 0


def _create_gent_vs_head_sampling_plot(input_file, output_dir):
    """Create scatter plot comparing GenT CPU (all durations) vs head sampling."""
    try:
        # Parse data from JSON report for metrics
        parser = ReportParser()
        experiments = parser.parse_report(input_file)

        # Filter for GenT CPU experiments and head sampling
        gent_cpu_experiments = [
            e for e in experiments if not e.is_head_sampling and e.compute_type == "CPU"
        ]
        head_sampling_experiments = [e for e in experiments if e.is_head_sampling]

        if not gent_cpu_experiments and not head_sampling_experiments:
            print("Warning: No GenT CPU or head sampling data found in report")
            return 1

        all_experiments = gent_cpu_experiments + head_sampling_experiments

        # Convert to the format expected by existing plotting functions
        names = [exp.name for exp in all_experiments]

        operation_f1_fidelity = [exp.operation_f1_fidelity for exp in all_experiments]
        operation_pair_f1_fidelity = [
            exp.operation_pair_f1_fidelity for exp in all_experiments
        ]
        cost_per_million_spans = [
            exp.total_cost_per_million_spans for exp in all_experiments
        ]
        transmission_cost_per_million_spans = [
            exp.transmission_cost_per_million_spans for exp in all_experiments
        ]

        # Create enhanced drawing function that handles CPU/GPU separation
        def draw_and_save_enhanced(
            x_values,
            x_title,
            y_values,
            y_title: str,
            plot_title: str,
            out_fname: str,
            dpi: int = 300,
        ):
            # Separate GenT CPU data by duration
            duration_groups = {}
            for x, y, name, exp in zip(
                x_values, y_values, names, all_experiments, strict=False
            ):
                if not exp.is_head_sampling:
                    duration = exp.duration
                    if duration not in duration_groups:
                        duration_groups[duration] = []
                    duration_groups[duration].append((x, y, name, exp))

            # Head sampling data
            head_sampling_data = [
                (x, y, name)
                for x, y, name, exp in zip(
                    x_values, y_values, names, all_experiments, strict=False
                )
                if exp.is_head_sampling
            ]

            # Create the plot
            plt.figure(figsize=(12, 8))
            plt.style.use("default")
            plt.rcParams["font.size"] = 11
            plt.rcParams["axes.labelsize"] = 12
            plt.rcParams["axes.titlesize"] = 14

            # Color scheme for different durations (CPU only)
            colors = {"1min": "red", "5min": "blue", "10min": "green"}
            markers = {"1min": "o", "5min": "s", "10min": "^"}

            # Plot GenT CPU data groups by duration
            for duration, group_data in duration_groups.items():
                if group_data:
                    group_x, group_y, group_names, group_exps = zip(
                        *group_data, strict=False
                    )
                    group_x_mean, group_y_mean = np.mean(group_x), np.mean(group_y)
                    group_x_std, group_y_std = np.std(group_x), np.std(group_y)

                    color = colors.get(duration, "black")
                    marker = markers.get(duration, "o")

                    plt.errorbar(
                        x=group_x_mean,
                        y=group_y_mean,
                        xerr=group_x_std,
                        yerr=group_y_std,
                        marker=marker,
                        markersize=5,
                        label=f"GenT {duration} CPU (mean ± std)",
                        color=color,
                        alpha=0.8,
                        capsize=5,
                        capthick=2,
                        linewidth=2,
                    )
                    plt.scatter(
                        x=group_x,
                        y=group_y,
                        marker=marker,
                        s=25,
                        color=color,
                        alpha=0.4,
                        edgecolors="black",
                        linewidth=1,
                    )

                    plt.annotate(
                        f"GenT {duration} CPU",
                        (group_x_mean, group_y_mean),
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

            # Plot head sampling data
            if head_sampling_data:
                head_x, head_y, head_names = zip(*head_sampling_data, strict=False)
                plt.scatter(
                    x=head_x,
                    y=head_y,
                    marker="D",
                    s=120,
                    label="Head Sampling",
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=1,
                )

                for x, y, name in head_sampling_data:
                    plt.annotate(
                        name,
                        (x, y),
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

            # Formatting
            plt.title(plot_title, fontsize=18, fontweight="bold", pad=20)
            plt.xlabel(x_title, fontsize=14, fontweight="bold")
            plt.ylabel(y_title, fontsize=14, fontweight="bold")
            plt.xscale("log")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            plt.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.5)
            plt.legend(
                loc="best", frameon=True, fancybox=True, shadow=True, fontsize=11
            )

            # Set axis limits
            if x_values and y_values:
                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)
                plt.xlim(x_min * 0.5, x_max * 2)
                plt.ylim(y_min - 2, y_max + 2)

            plt.gca().set_facecolor("#f8f9fa")
            plt.tight_layout()

            # Save plot
            out_path = Path(output_dir) / out_fname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                out_path,
                dpi=dpi,
                bbox_inches="tight",
                format="png",
                facecolor="white",
                edgecolor="none",
            )
            print(f"Saved plot to: {out_path.resolve()}")
            plt.close()

        # Create the plots
        draw_and_save_enhanced(
            x_values=cost_per_million_spans,
            x_title="Total Cost per Million Spans (log scale)",
            y_values=operation_f1_fidelity,
            y_title="Operation F1 Fidelity (%)",
            plot_title="GenT CPU (1min/5min/10min) vs Head Sampling - Operation F1 Fidelity",
            out_fname="gent_vs_head_sampling_operation_f1.png",
        )

        draw_and_save_enhanced(
            x_values=cost_per_million_spans,
            x_title="Total Cost per Million Spans (log scale)",
            y_values=operation_pair_f1_fidelity,
            y_title="Operation Pair F1 Fidelity (%)",
            plot_title="GenT CPU (1min/5min/10min) vs Head Sampling - Operation Pair F1 Fidelity",
            out_fname="gent_vs_head_sampling_operation_pair_f1.png",
        )

    except FileNotFoundError:
        print(f"Error: Could not find input file: {input_file}")
        return 1
    except Exception as e:
        print(f"Error creating plots: {e}")
        return 1
    else:
        return 0


if __name__ == "__main__":
    main()
