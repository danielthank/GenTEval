import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from genteval.plotting.data import ReportParser


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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate visualizations from GenTEval reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["scatter", "heatmap"],
        default="scatter",
        help="Visualization mode: scatter (GenT CPU 1min/5min/10min vs head sampling, default) or heatmap",
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
        help="Compressor name to visualize (required for heatmap mode)",
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

    else:
        # Default scatter plot mode: GenT CPU vs head sampling
        return _create_gent_vs_head_sampling_plot(args.input, args.output_dir)


def _create_gent_vs_head_sampling_plot(input_file, output_dir):
    """Create scatter plot comparing GenT CPU (all durations) vs head sampling."""
    try:
        # Parse data from JSON report
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
        mape_fidelity = [exp.mape_fidelity for exp in all_experiments]
        cos_fidelity = [exp.cos_fidelity for exp in all_experiments]
        operation_f1_fidelity = [exp.operation_f1_fidelity for exp in all_experiments]
        operation_pair_f1_fidelity = [
            exp.operation_pair_f1_fidelity for exp in all_experiments
        ]
        child_parent_ratio_fidelity = [
            exp.child_parent_ratio_fidelity for exp in all_experiments
        ]
        child_parent_overall_fidelity = [
            exp.child_parent_overall_fidelity for exp in all_experiments
        ]
        child_parent_depth1_fidelity = [
            exp.child_parent_depth1_fidelity for exp in all_experiments
        ]
        child_parent_depth2_fidelity = [
            exp.child_parent_depth2_fidelity for exp in all_experiments
        ]
        child_parent_depth3_fidelity = [
            exp.child_parent_depth3_fidelity for exp in all_experiments
        ]
        child_parent_depth4_fidelity = [
            exp.child_parent_depth4_fidelity for exp in all_experiments
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
                        markersize=12,
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
                        s=60,
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
            y_values=mape_fidelity,
            y_title="MAPE Fidelity (%)",
            plot_title="GenT CPU (1min/5min/10min) vs Head Sampling - MAPE Fidelity",
            out_fname="gent_vs_head_sampling_mape.png",
        )

        draw_and_save_enhanced(
            x_values=cost_per_million_spans,
            x_title="Total Cost per Million Spans (log scale)",
            y_values=cos_fidelity,
            y_title="Cosine Similarity Fidelity (%)",
            plot_title="GenT CPU (1min/5min/10min) vs Head Sampling - Cosine Similarity",
            out_fname="gent_vs_head_sampling_cosine.png",
        )

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

        draw_and_save_enhanced(
            x_values=cost_per_million_spans,
            x_title="Total Cost per Million Spans (log scale)",
            y_values=child_parent_ratio_fidelity,
            y_title="Child/Parent Duration Ratio Fidelity (%)",
            plot_title="GenT CPU (1min/5min/10min) vs Head Sampling - Child/Parent Duration Ratio Fidelity",
            out_fname="gent_vs_head_sampling_child_parent_ratio.png",
        )

        draw_and_save_enhanced(
            x_values=cost_per_million_spans,
            x_title="Total Cost per Million Spans (log scale)",
            y_values=child_parent_overall_fidelity,
            y_title="Overall Child/Parent Duration Fidelity (%)",
            plot_title="GenT CPU (1min/5min/10min) vs Head Sampling - Overall Child/Parent Duration Fidelity",
            out_fname="gent_vs_head_sampling_child_parent_overall.png",
        )

        draw_and_save_enhanced(
            x_values=cost_per_million_spans,
            x_title="Total Cost per Million Spans (log scale)",
            y_values=child_parent_depth1_fidelity,
            y_title="Depth 1 Child/Parent Duration Fidelity (%)",
            plot_title="GenT CPU (1min/5min/10min) vs Head Sampling - Depth 1 Child/Parent Duration Fidelity",
            out_fname="gent_vs_head_sampling_child_parent_depth1.png",
        )

        draw_and_save_enhanced(
            x_values=cost_per_million_spans,
            x_title="Total Cost per Million Spans (log scale)",
            y_values=child_parent_depth2_fidelity,
            y_title="Depth 2 Child/Parent Duration Fidelity (%)",
            plot_title="GenT CPU (1min/5min/10min) vs Head Sampling - Depth 2 Child/Parent Duration Fidelity",
            out_fname="gent_vs_head_sampling_child_parent_depth2.png",
        )

        draw_and_save_enhanced(
            x_values=cost_per_million_spans,
            x_title="Total Cost per Million Spans (log scale)",
            y_values=child_parent_depth3_fidelity,
            y_title="Depth 3 Child/Parent Duration Fidelity (%)",
            plot_title="GenT CPU (1min/5min/10min) vs Head Sampling - Depth 3 Child/Parent Duration Fidelity",
            out_fname="gent_vs_head_sampling_child_parent_depth3.png",
        )

        draw_and_save_enhanced(
            x_values=cost_per_million_spans,
            x_title="Total Cost per Million Spans (log scale)",
            y_values=child_parent_depth4_fidelity,
            y_title="Depth 4 Child/Parent Duration Fidelity (%)",
            plot_title="GenT CPU (1min/5min/10min) vs Head Sampling - Depth 4 Child/Parent Duration Fidelity",
            out_fname="gent_vs_head_sampling_child_parent_depth4.png",
        )

        return 0

    except FileNotFoundError:
        print(f"Error: Could not find input file: {input_file}")
        return 1
    except Exception as e:
        print(f"Error creating plots: {e}")
        return 1


if __name__ == "__main__":
    main()
