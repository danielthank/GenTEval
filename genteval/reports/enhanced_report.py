"""Enhanced report generator with beautiful formatting and visualizations."""

import copy
import json
import pathlib
from typing import Any


try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

    # Fallback classes for when Rich is not available
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

    class Table:
        def __init__(self, *args, **kwargs):
            self.rows = []

        def add_column(self, *args, **kwargs):
            pass

        def add_row(self, *args, **kwargs):
            self.rows.append(args)

    class Text:
        def __init__(self, text, style=None):
            self.text = text

        def __str__(self):
            return str(self.text)

    class Panel:
        def __init__(self, text, title="", **kwargs):
            self.text = text
            self.title = title

    class BoxType:
        ROUNDED = None

    box = BoxType()


class EnhancedReportGenerator:
    """Enhanced report generator with beautiful formatting."""

    def __init__(self, console=None):
        """Initialize the enhanced report generator."""
        if RICH_AVAILABLE:
            self.console = console or Console()
        else:
            self.console = Console()  # Use our fallback class

    def format_metric_value(self, value: float, metric_type: str = "default") -> "Text":
        """Format metric values with appropriate colors and styling."""
        if value is None:
            return Text("N/A", style="dim")

        if metric_type == "accuracy" or metric_type == "f1":
            # Higher is better (0-1 range) - includes precision and recall
            if value >= 0.8:
                return Text(f"{value:.4f}", style="bold green")
            if value >= 0.6:
                return Text(f"{value:.4f}", style="yellow")
            return Text(f"{value:.4f}", style="red")
        if metric_type == "wasserstein":
            # Lower is better
            if value <= 0.1:
                return Text(f"{value:.4f}", style="bold green")
            if value <= 0.3:
                return Text(f"{value:.4f}", style="yellow")
            return Text(f"{value:.4f}", style="red")
        if metric_type == "mape":
            # Lower is better (percentage error)
            if value <= 5.0:
                return Text(f"{value:.2f}%", style="bold green")
            if value <= 15.0:
                return Text(f"{value:.2f}%", style="yellow")
            return Text(f"{value:.2f}%", style="red")
        if metric_type == "size":
            # Format file sizes nicely
            if value >= 1024:  # KB
                return Text(f"{value / 1024:.2f} KB", style="cyan")
            return Text(f"{value} B", style="cyan")
        if metric_type == "time":
            # Always format time in seconds
            return Text(f"{value:.4f}", style="magenta")
        return Text(f"{value:.4f}", style="white")

    def create_summary_table(self, report_data: dict[str, Any], title: str) -> "Table":
        """Create a summary table for a specific metric type."""
        table = Table(title=title, box=box.ROUNDED, title_style="bold magenta")

        # Add columns
        table.add_column("Compressor", style="cyan", no_wrap=True)

        # Determine metric columns based on data
        if not report_data:
            return table

        sample_key = next(iter(report_data.keys()))
        sample_data = report_data[sample_key]

        # Define the desired order for operation metrics
        operation_metric_order = [
            "operation_precision",
            "operation_recall",
            "operation_f1",
            "operation_pair_precision",
            "operation_pair_recall",
            "operation_pair_f1",
        ]

        # Filter metrics - all metrics now have nested "avg" key
        available_metrics = []
        for metric_key, metric_data in sample_data.items():
            if isinstance(metric_data, dict) and "avg" in metric_data:
                available_metrics.append(metric_key)

        # Sort metrics with custom order for operation metrics
        def metric_sort_key(metric):
            if metric in operation_metric_order:
                return (0, operation_metric_order.index(metric))
            return (1, metric)  # Other metrics come after, sorted alphabetically

        ordered_metrics = sorted(available_metrics, key=metric_sort_key)

        for metric in ordered_metrics:
            # Create more readable column names for duration depth metrics
            column_name = metric.replace("_", " ").title()
            if "Duration Depth" in column_name:
                # Make duration depth metrics more readable
                column_name = column_name.replace("Duration Depth 0", "Depth 0")
                column_name = column_name.replace("Duration Depth 1", "Depth 1")
                column_name = column_name.replace("Wdist", "W-Dist")
                column_name = column_name.replace("Mape", "MAPE")
                column_name = column_name.replace("Cosine Sim", "Cos-Sim")
            table.add_column(column_name, justify="right")

        # Add rows
        for compressor_group, metrics in sorted(report_data.items()):
            row = [compressor_group]
            for metric in ordered_metrics:
                # Check if metric exists for this compressor
                if metric not in metrics:
                    row.append(Text("N/A", style="dim"))
                    continue

                # Extract the avg value from the nested structure
                metric_data = metrics[metric]

                value = metric_data["avg"]

                if "f1" in metric or "precision" in metric or "recall" in metric:
                    formatted_value = self.format_metric_value(value, "accuracy")
                elif "wdist" in metric:
                    formatted_value = self.format_metric_value(value, "wasserstein")
                elif "mape" in metric:
                    formatted_value = self.format_metric_value(value, "mape")
                elif "cosine_sim" in metric:
                    formatted_value = self.format_metric_value(value, "accuracy")
                elif "size" in metric:
                    formatted_value = self.format_metric_value(value, "size")
                elif any(
                    x in metric
                    for x in [
                        "compression_time_cpu_seconds",
                        "compression_time_gpu_seconds",
                        "compression_time_total_seconds",
                        "time",
                    ]
                ):
                    formatted_value = self.format_metric_value(value, "time")
                else:
                    formatted_value = self.format_metric_value(value)

                row.append(formatted_value)
            table.add_row(*row)

        return table

    def create_rca_table(self, report_data: dict[str, Any]) -> "Table":
        """Create a specialized table for RCA metrics."""
        table = Table(
            title="RCA Performance Metrics", box=box.ROUNDED, title_style="bold magenta"
        )

        table.add_column("Compressor", style="cyan", no_wrap=True)
        table.add_column("AC@1", justify="right")
        table.add_column("AC@2", justify="right")
        table.add_column("AC@3", justify="right")
        table.add_column("AC@4", justify="right")
        table.add_column("AC@5", justify="right")
        table.add_column("Avg@5", justify="right", style="bold")

        for compressor_group, metrics in sorted(report_data.items()):
            row = [compressor_group]
            for k in range(1, 6):
                metric_data = metrics[f"ac{k}"]
                if "values" in metric_data and "avg" not in metric_data:
                    values = metric_data["values"]
                    value = sum(values) / len(values) if values else float("nan")
                else:
                    value = metric_data["avg"]
                formatted_value = self.format_metric_value(value, "accuracy")
                row.append(formatted_value)

            avg5_data = metrics["avg5"]
            if "values" in avg5_data and "avg" not in avg5_data:
                values = avg5_data["values"]
                avg5 = sum(values) / len(values) if values else float("nan")
            else:
                avg5 = avg5_data["avg"]
            row.append(self.format_metric_value(avg5, "accuracy"))
            table.add_row(*row)

        return table

    def create_count_over_time_table(self, report_data: dict[str, Any]) -> "Table":
        """Create a specialized table for Count Over Time metrics."""
        table = Table(
            title="Count Over Time Performance Metrics",
            box=box.ROUNDED,
            title_style="bold magenta",
        )
        table.add_column("Compressor", style="cyan", no_wrap=True)
        table.add_column("MAPE Fidelity (%)", justify="right", style="bold")
        table.add_column("Cosine Fidelity (%)", justify="right", style="bold")

        for compressor_group, metrics in sorted(report_data.items()):
            row = [compressor_group]

            # MAPE fidelity score
            mape_fidelity = metrics.get("count_over_time_mape_fidelity_score", 0.0)
            row.append(self.format_metric_value(mape_fidelity, "percentage"))

            # Cosine fidelity score
            cosine_fidelity = metrics.get("count_over_time_cosine_fidelity_score", 0.0)
            row.append(self.format_metric_value(cosine_fidelity, "percentage"))

            table.add_row(*row)

        return table

    def _print_duration_sections(self, report_data: dict[str, Any]):
        """Print duration metrics organized by sections."""
        # Calculate and display both fidelity scores first
        mape_fidelity_scores = self.calculate_mape_fidelity_score(report_data)
        cos_fidelity_scores = self.calculate_cos_fidelity_score(report_data)

        if mape_fidelity_scores or cos_fidelity_scores:
            fidelity_table = Table(
                title="Duration Fidelity Scores",
                box=box.ROUNDED,
                title_style="bold magenta",
            )
            fidelity_table.add_column("Compressor", style="cyan", no_wrap=True)
            fidelity_table.add_column("MAPE Fidelity", justify="right")
            fidelity_table.add_column("Cos Fidelity", justify="right")

            # Get all compressors from both scores
            all_compressors = set(mape_fidelity_scores.keys()) | set(
                cos_fidelity_scores.keys()
            )

            for compressor in sorted(all_compressors):
                mape_score = mape_fidelity_scores.get(compressor, 0.0)
                cos_score = cos_fidelity_scores.get(compressor, 0.0)

                formatted_mape = self.format_metric_value(mape_score, "accuracy")
                formatted_cos = self.format_metric_value(cos_score, "accuracy")

                fidelity_table.add_row(compressor, formatted_mape, formatted_cos)

            self.console.print(fidelity_table)
            self.console.print()

        # Create separate data structures for each section
        wdist_data = {}
        depth0_data = {}
        depth1_data = {}

        for compressor, metrics in report_data.items():
            wdist_metrics = {}
            depth0_metrics = {}
            depth1_metrics = {}

            for metric_name, metric_data in metrics.items():
                if "_wdist" in metric_name:
                    wdist_metrics[metric_name] = metric_data
                elif "depth_0" in metric_name and (
                    "_mape" in metric_name or "_cosine_sim" in metric_name
                ):
                    # Only include p50 and p90 for cleaner display
                    if "_p50_" in metric_name or "_p90_" in metric_name:
                        depth0_metrics[metric_name] = metric_data
                elif "depth_1" in metric_name and (
                    "_mape" in metric_name or "_cosine_sim" in metric_name
                ):
                    # Only include p50 and p90 for cleaner display
                    if "_p50_" in metric_name or "_p90_" in metric_name:
                        depth1_metrics[metric_name] = metric_data

            if wdist_metrics:
                wdist_data[compressor] = wdist_metrics
            if depth0_metrics:
                depth0_data[compressor] = depth0_metrics
            if depth1_metrics:
                depth1_data[compressor] = depth1_metrics

        # Print each section
        if wdist_data:
            wdist_table = self.create_summary_table(
                wdist_data, "Duration Wasserstein Distance Metrics"
            )
            self.console.print(wdist_table)
            self.console.print()

        if depth0_data:
            depth0_table = self.create_summary_table(
                depth0_data, "Duration Depth 0 Metrics (P50 & P90)"
            )
            self.console.print(depth0_table)
            self.console.print()

        if depth1_data:
            depth1_table = self.create_summary_table(
                depth1_data, "Duration Depth 1 Metrics (P50 & P90)"
            )
            self.console.print(depth1_table)
            self.console.print()

    def create_performance_overview(
        self, all_reports: dict[str, dict[str, Any]]
    ) -> "Panel":
        """Create a performance overview panel."""
        overview_text = []

        # Count total evaluations
        total_compressors = set()
        total_metrics = 0

        for report_type, report_data in all_reports.items():
            total_compressors.update(report_data.keys())
            total_metrics += len(report_data)

        overview_text.append(
            f"📊 Total Compressors Evaluated: {len(total_compressors)}"
        )
        overview_text.append(f"🔍 Total Metric Groups: {total_metrics}")
        overview_text.append(f"📈 Report Types Generated: {len(all_reports)}")

        # Find best performing compressors for each metric type
        for report_type, report_data in all_reports.items():
            if not report_data:
                continue

            if report_type in ["trace_rca", "micro_rank"]:
                # For RCA, higher avg5 is better
                def get_avg5_value(x):
                    avg5_data = x[1]["avg5"]
                    if "values" in avg5_data and "avg" not in avg5_data:
                        values = avg5_data["values"]
                        return sum(values) / len(values) if values else float("nan")
                    return avg5_data["avg"]

                best_compressor = max(report_data.items(), key=get_avg5_value)
                overview_text.append(
                    f"🏆 Best {report_type.replace('_', ' ').title()}: "
                    f"{best_compressor[0]} ({get_avg5_value(best_compressor):.4f})"
                )
            elif report_type == "count_over_time":
                # For count over time, higher fidelity scores are better
                def get_count_fidelity_value(x):
                    mape_score = x[1].get("count_over_time_mape_fidelity_score", 0.0)
                    cosine_score = x[1].get(
                        "count_over_time_cosine_fidelity_score", 0.0
                    )
                    return (
                        mape_score + cosine_score
                    ) / 2  # Average of both fidelity scores

                best_compressor = max(report_data.items(), key=get_count_fidelity_value)
                overview_text.append(
                    f"🏆 Best {report_type.replace('_', ' ').title()}: "
                    f"{best_compressor[0]} ({get_count_fidelity_value(best_compressor):.1f}% avg fidelity)"
                )
            elif report_type == "time":
                # For compression time, lower is better
                def get_time_value(x):
                    # Try total time first, then fall back to individual CPU/GPU times
                    metric = x[1].get("compression_time_total_seconds")
                    if not metric:
                        metric = x[1].get("compression_time_cpu_seconds")
                    if not metric:
                        metric = x[1].get("compression_time_gpu_seconds")
                    if not metric:
                        return float("inf")
                    if "values" in metric and "avg" not in metric:
                        values = metric["values"]
                        return sum(values) / len(values) if values else float("inf")
                    return metric.get("avg", float("inf"))

                try:
                    fastest = min(report_data.items(), key=get_time_value)
                    fastest_time = get_time_value(fastest)
                    formatted_time = str(self.format_metric_value(fastest_time, "time"))
                    overview_text.append(
                        f"⏱️ Fastest Compression: {fastest[0]} ({formatted_time})"
                    )
                except ValueError:
                    # No valid items
                    pass
            elif "f1" in str(report_data):
                # For F1 scores, higher is better
                sample_metrics = next(iter(report_data.values()))
                f1_metrics = [k for k in sample_metrics if "f1" in k]
                if f1_metrics:

                    def get_f1_value(x):
                        f1_data = x[1][f1_metrics[0]]
                        if "values" in f1_data and "avg" not in f1_data:
                            values = f1_data["values"]
                            return sum(values) / len(values) if values else float("nan")
                        return f1_data["avg"]

                    best_compressor = max(report_data.items(), key=get_f1_value)
                    overview_text.append(
                        f"🏆 Best {report_type.title()}: "
                        f"{best_compressor[0]} ({get_f1_value(best_compressor):.4f})"
                    )

        overview_panel = Panel(
            "\n".join(overview_text),
            title="📋 Evaluation Overview",
            title_align="left",
            border_style="blue",
        )

        return overview_panel

    def calculate_mape_fidelity_score(
        self, duration_data: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate MAPE fidelity score by averaging MAPE across depths 0-4 and all percentiles."""
        mape_fidelity_scores = {}

        for compressor, metrics in duration_data.items():
            mape_values = []

            # Collect MAPE values for depth 0-4 and all percentiles (p0-p100)
            for metric_name, metric_data in metrics.items():
                if "_mape" in metric_name:
                    # Check if it's a depth metric (depth_0, depth_1, depth_2, depth_3, depth_4)
                    for depth in range(5):  # 0-4
                        if f"depth_{depth}_" in metric_name:
                            # Extract MAPE value
                            if isinstance(metric_data, dict) and "avg" in metric_data:
                                mape_val = metric_data["avg"]
                                # Only include finite values
                                if not (
                                    isinstance(mape_val, float)
                                    and (
                                        mape_val == float("inf") or mape_val != mape_val
                                    )
                                ):
                                    mape_values.append(mape_val)
                            break

            # Calculate average MAPE (lower is better, so fidelity = 100 - avg_mape)
            if mape_values:
                avg_mape = sum(mape_values) / len(mape_values)
                # Convert MAPE to fidelity score (100% - MAPE%, clamped to 0-100)
                mape_fidelity_score = max(0, min(100, 100 - avg_mape))
                mape_fidelity_scores[compressor] = mape_fidelity_score
            else:
                mape_fidelity_scores[compressor] = 0.0

        return mape_fidelity_scores

    def calculate_cos_fidelity_score(
        self, duration_data: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate Cos fidelity score by averaging cosine similarity across depths 0-4 and all percentiles."""
        cos_fidelity_scores = {}

        for compressor, metrics in duration_data.items():
            cos_sim_values = []

            # Collect cosine similarity values for depth 0-4 and all percentiles (p0-p100)
            for metric_name, metric_data in metrics.items():
                if "_cosine_sim" in metric_name:
                    # Check if it's a depth metric (depth_0, depth_1, depth_2, depth_3, depth_4)
                    for depth in range(5):  # 0-4
                        if f"depth_{depth}_" in metric_name:
                            # Extract cosine similarity value
                            if isinstance(metric_data, dict) and "avg" in metric_data:
                                cos_sim_val = metric_data["avg"]
                                # Only include finite values
                                if not (
                                    isinstance(cos_sim_val, float)
                                    and (
                                        cos_sim_val == float("inf")
                                        or cos_sim_val != cos_sim_val
                                    )
                                ):
                                    cos_sim_values.append(cos_sim_val)
                            break

            # Calculate average cosine similarity and scale to 0-100
            if cos_sim_values:
                avg_cos_sim = sum(cos_sim_values) / len(cos_sim_values)
                # Scale cosine similarity (0-1) to fidelity score (0-100)
                cos_fidelity_score = avg_cos_sim * 100
                cos_fidelity_scores[compressor] = cos_fidelity_score
            else:
                cos_fidelity_scores[compressor] = 0.0

        return cos_fidelity_scores

    def calculate_mape_fidelity_score_by_status_code(
        self, duration_data: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate MAPE fidelity score by averaging MAPE across all status codes and percentiles."""
        mape_fidelity_scores = {}

        for compressor, metrics in duration_data.items():
            mape_values = []

            # Collect MAPE values for all status codes and all percentiles (p0-p100)
            for metric_name, metric_data in metrics.items():
                if "_mape" in metric_name and "http.status_code_" in metric_name:
                    # Extract MAPE value
                    if isinstance(metric_data, dict) and "avg" in metric_data:
                        mape_val = metric_data["avg"]
                        # Only include finite values
                        if not (
                            isinstance(mape_val, float)
                            and (mape_val == float("inf") or mape_val != mape_val)
                        ):
                            mape_values.append(mape_val)

            # Calculate average MAPE (lower is better, so fidelity = 100 - avg_mape)
            if mape_values:
                avg_mape = sum(mape_values) / len(mape_values)
                # Convert MAPE to fidelity score (100% - MAPE%, clamped to 0-100)
                mape_fidelity_score = max(0, min(100, 100 - avg_mape))
                mape_fidelity_scores[compressor] = mape_fidelity_score
            else:
                mape_fidelity_scores[compressor] = 0.0

        return mape_fidelity_scores

    def calculate_cos_fidelity_score_by_status_code(
        self, duration_data: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate Cos fidelity score by averaging cosine similarity across all status codes and percentiles."""
        cos_fidelity_scores = {}

        for compressor, metrics in duration_data.items():
            cos_sim_values = []

            # Collect cosine similarity values for all status codes and all percentiles (p0-p100)
            for metric_name, metric_data in metrics.items():
                if "_cosine_sim" in metric_name and "http.status_code_" in metric_name:
                    # Extract cosine similarity value
                    if isinstance(metric_data, dict) and "avg" in metric_data:
                        cos_sim_val = metric_data["avg"]
                        # Only include finite values
                        if not (
                            isinstance(cos_sim_val, float)
                            and (
                                cos_sim_val == float("inf")
                                or cos_sim_val != cos_sim_val
                            )
                        ):
                            cos_sim_values.append(cos_sim_val)

            # Calculate average cosine similarity and scale to 0-100
            if cos_sim_values:
                avg_cos_sim = sum(cos_sim_values) / len(cos_sim_values)
                # Scale cosine similarity (0-1) to fidelity score (0-100)
                cos_fidelity_score = avg_cos_sim * 100
                cos_fidelity_scores[compressor] = cos_fidelity_score
            else:
                cos_fidelity_scores[compressor] = 0.0

        return cos_fidelity_scores

    def print_enhanced_report(self, all_reports: dict[str, dict[str, Any]]):
        """Print enhanced, beautifully formatted reports."""

        if RICH_AVAILABLE:
            self._print_rich_report(all_reports)
        else:
            self._print_fallback_report(all_reports)

    def _print_rich_report(self, all_reports: dict[str, dict[str, Any]]):
        """Print enhanced report using Rich formatting."""
        # Print header
        self.console.print()
        self.console.print("=" * 74, style="bold blue")
        self.console.print(
            "    GenTEval Compression Evaluation Report", style="bold blue"
        )
        self.console.print("=" * 74, style="bold blue")
        self.console.print()

        # Print overview
        if all_reports:
            overview = self.create_performance_overview(all_reports)
            self.console.print(overview)
            self.console.print()

        # Print individual report tables
        for report_type, report_data in all_reports.items():
            if not report_data:
                continue

            self.console.print(
                f"📊 {report_type.replace('_', ' ').title()} Results",
                style="bold yellow",
            )
            self.console.print("-" * 50, style="dim")

            if report_type in ["trace_rca", "micro_rank"]:
                table = self.create_rca_table(report_data)
                self.console.print(table)
                self.console.print()
            elif report_type == "count_over_time":
                table = self.create_count_over_time_table(report_data)
                self.console.print(table)
                self.console.print()
            elif report_type == "duration":
                # Create separate sections for duration metrics
                self._print_duration_sections(report_data)
            else:
                table = self.create_summary_table(
                    report_data, f"{report_type.replace('_', ' ').title()} Metrics"
                )
                self.console.print(table)
                self.console.print()

        # Print footer
        self.console.print("✨ Report generation complete!", style="bold green")
        self.console.print()

    def _print_fallback_report(self, all_reports: dict[str, dict[str, Any]]):
        """Print report using basic formatting when Rich is not available."""
        print()
        print("=" * 74)
        print("    GenTEval Compression Evaluation Report")
        print("=" * 74)
        print()

        # Print overview
        if all_reports:
            total_compressors = set()
            for report_data in all_reports.values():
                total_compressors.update(report_data.keys())

            print(f"Total Compressors Evaluated: {len(total_compressors)}")
            print(f"Report Types Generated: {len(all_reports)}")
            print()

        # Print individual reports
        for report_type, report_data in all_reports.items():
            if not report_data:
                continue

            print(f"{report_type.replace('_', ' ').title()} Results")
            print("-" * 50)

            if report_type == "duration":
                self._print_duration_sections_fallback(report_data)
            else:
                for compressor_group, metrics in sorted(report_data.items()):
                    print(f"\n{compressor_group}:")
                    for metric, metric_data in sorted(metrics.items()):
                        # Extract the avg value from the nested structure
                        value = metric_data["avg"]

                        # Create more readable metric names
                        display_name = metric.replace("_", " ").title()
                        if "Duration Depth" in display_name:
                            display_name = display_name.replace(
                                "Duration Depth 0", "Depth 0"
                            )
                            display_name = display_name.replace(
                                "Duration Depth 1", "Depth 1"
                            )
                            display_name = display_name.replace("Wdist", "W-Dist")
                            display_name = display_name.replace("Mape", "MAPE")
                            display_name = display_name.replace("Cosine Sim", "Cos-Sim")

                        # Specialized pretty formatting for known types
                        if isinstance(value, (int, float)):
                            lower_metric = metric.lower()
                            if any(
                                x in lower_metric
                                for x in [
                                    "compression_time_cpu_seconds",
                                    "compression_time_gpu_seconds",
                                    "compression_time_total_seconds",
                                    "time",
                                ]
                            ):
                                # Always format time in seconds
                                seconds = float(value)
                                pretty = f"{seconds:.4f} s"
                                print(f"  {display_name}: {pretty}")
                            else:
                                print(f"  {display_name}: {value:.4f}")
                        else:
                            print(f"  {display_name}: {value}")
                print()

        print("Report generation complete!")
        print()

    def _print_duration_sections_fallback(self, report_data: dict[str, Any]):
        """Print duration sections for fallback mode."""
        # Calculate and display both fidelity scores first
        mape_fidelity_scores = self.calculate_mape_fidelity_score(report_data)
        cos_fidelity_scores = self.calculate_cos_fidelity_score(report_data)

        if mape_fidelity_scores or cos_fidelity_scores:
            print("\nDuration Fidelity Scores")
            print("-" * 50)
            print(f"{'Compressor':<20} {'MAPE Fidelity':<15} {'Cos Fidelity':<15}")
            print("-" * 50)

            all_compressors = set(mape_fidelity_scores.keys()) | set(
                cos_fidelity_scores.keys()
            )
            for compressor in sorted(all_compressors):
                mape_score = mape_fidelity_scores.get(compressor, 0.0)
                cos_score = cos_fidelity_scores.get(compressor, 0.0)
                print(f"{compressor:<20} {mape_score:>13.2f}% {cos_score:>13.2f}%")
            print()

        # Create separate data structures for each section
        wdist_data = {}
        depth0_data = {}
        depth1_data = {}

        for compressor, metrics in report_data.items():
            wdist_metrics = {}
            depth0_metrics = {}
            depth1_metrics = {}

            for metric_name, metric_data in metrics.items():
                if "_wdist" in metric_name:
                    wdist_metrics[metric_name] = metric_data
                elif "depth_0" in metric_name and (
                    "_mape" in metric_name or "_cosine_sim" in metric_name
                ):
                    if "_p50_" in metric_name or "_p90_" in metric_name:
                        depth0_metrics[metric_name] = metric_data
                elif "depth_1" in metric_name and (
                    "_mape" in metric_name or "_cosine_sim" in metric_name
                ):
                    if "_p50_" in metric_name or "_p90_" in metric_name:
                        depth1_metrics[metric_name] = metric_data

            if wdist_metrics:
                wdist_data[compressor] = wdist_metrics
            if depth0_metrics:
                depth0_data[compressor] = depth0_metrics
            if depth1_metrics:
                depth1_data[compressor] = depth1_metrics

        # Print sections
        sections = [
            ("Wasserstein Distance Metrics", wdist_data),
            ("Depth 0 Metrics (P50 & P90)", depth0_data),
            ("Depth 1 Metrics (P50 & P90)", depth1_data),
        ]

        for section_title, section_data in sections:
            if section_data:
                print(f"\n{section_title}")
                print("-" * 30)
                for compressor, metrics in sorted(section_data.items()):
                    print(f"\n  {compressor}:")
                    for metric, metric_data in sorted(metrics.items()):
                        value = metric_data["avg"]
                        display_name = metric.replace("_", " ").title()
                        display_name = display_name.replace("Wdist", "W-Dist")
                        display_name = display_name.replace("Mape", "MAPE")
                        display_name = display_name.replace("Cosine Sim", "Cos-Sim")

                        if isinstance(value, float):
                            print(f"    {display_name}: {value:.4f}")
                        else:
                            print(f"    {display_name}: {value}")

    def save_enhanced_json_report(
        self, all_reports: dict[str, dict[str, Any]], output_path: pathlib.Path
    ):
        enhanced_reports = copy.deepcopy(all_reports)

        fidelity_scores = {}
        if "duration" in all_reports:
            duration_data = all_reports["duration"]
            mape_fidelity_scores = self.calculate_mape_fidelity_score(duration_data)
            cosine_fidelity_scores = self.calculate_cos_fidelity_score(duration_data)
            mape_fidelity_scores_by_status_code = (
                self.calculate_mape_fidelity_score_by_status_code(duration_data)
            )
            cosine_fidelity_scores_by_status_code = (
                self.calculate_cos_fidelity_score_by_status_code(duration_data)
            )

            fidelity_scores = {
                "mape_fidelity_scores": mape_fidelity_scores,
                "cosine_similarity_fidelity_scores": cosine_fidelity_scores,
                "mape_fidelity_scores_by_status_code": mape_fidelity_scores_by_status_code,
                "cosine_similarity_fidelity_scores_by_status_code": cosine_fidelity_scores_by_status_code,
            }

        enhanced_report = {
            "metadata": {
                "generator": "GenTEval Enhanced Report Generator",
                "report_types": list(enhanced_reports.keys()),
                "total_compressors": len(
                    set().union(
                        *[report.keys() for report in enhanced_reports.values()]
                    )
                ),
                "fidelity_scores": fidelity_scores,
            },
            "reports": enhanced_reports,
        }

        with open(output_path, "w") as f:
            json.dump(enhanced_report, f, indent=2, default=str)
