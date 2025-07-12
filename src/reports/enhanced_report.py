"""Enhanced report generator with beautiful formatting and visualizations."""

import json
import pathlib
from typing import Any, Dict

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
            elif value >= 0.6:
                return Text(f"{value:.4f}", style="yellow")
            else:
                return Text(f"{value:.4f}", style="red")
        elif metric_type == "wasserstein":
            # Lower is better
            if value <= 0.1:
                return Text(f"{value:.4f}", style="bold green")
            elif value <= 0.3:
                return Text(f"{value:.4f}", style="yellow")
            else:
                return Text(f"{value:.4f}", style="red")
        elif metric_type == "mape":
            # Lower is better (percentage error)
            if value <= 5.0:
                return Text(f"{value:.2f}%", style="bold green")
            elif value <= 15.0:
                return Text(f"{value:.2f}%", style="yellow")
            else:
                return Text(f"{value:.2f}%", style="red")
        elif metric_type == "size":
            # Format file sizes nicely
            if value >= 1024 * 1024 * 1024:  # GB
                return Text(f"{value / (1024**3):.2f} GB", style="cyan")
            elif value >= 1024 * 1024:  # MB
                return Text(f"{value / (1024**2):.2f} MB", style="cyan")
            elif value >= 1024:  # KB
                return Text(f"{value / 1024:.2f} KB", style="cyan")
            else:
                return Text(f"{value} B", style="cyan")
        else:
            return Text(f"{value:.4f}", style="white")

    def create_summary_table(self, report_data: Dict[str, Any], title: str) -> "Table":
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
            "operation_precision_avg",
            "operation_recall_avg",
            "operation_f1_avg",
            "operation_pair_precision_avg",
            "operation_pair_recall_avg",
            "operation_pair_f1_avg",
        ]

        # Filter and order metrics
        available_metrics = []
        for metric in sample_data.keys():
            if (
                metric.endswith("_f1_avg")
                or metric.endswith("_precision_avg")
                or metric.endswith("_recall_avg")
                or metric.endswith("_wdist_avg")
                or metric.endswith("_mape_avg")
                or metric == "size"
            ):
                available_metrics.append(metric)

        # Sort metrics with custom order for operation metrics
        def metric_sort_key(metric):
            if metric in operation_metric_order:
                return (0, operation_metric_order.index(metric))
            else:
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
            table.add_column(column_name, justify="right")

        # Add rows
        for compressor_group, metrics in sorted(report_data.items()):
            row = [compressor_group]
            for metric in ordered_metrics:
                value = metrics[metric]
                if "f1" in metric or "precision" in metric or "recall" in metric:
                    formatted_value = self.format_metric_value(value, "accuracy")
                elif "wdist" in metric:
                    formatted_value = self.format_metric_value(value, "wasserstein")
                elif "mape" in metric:
                    formatted_value = self.format_metric_value(value, "mape")
                elif "size" in metric:
                    formatted_value = self.format_metric_value(value, "size")
                else:
                    formatted_value = self.format_metric_value(value)

                row.append(formatted_value)
            table.add_row(*row)

        return table

    def create_rca_table(self, report_data: Dict[str, Any]) -> "Table":
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
                value = metrics.get(f"ac{k}", 0)
                formatted_value = self.format_metric_value(value, "accuracy")
                row.append(formatted_value)

            avg5 = metrics.get("avg5", 0)
            row.append(self.format_metric_value(avg5, "accuracy"))
            table.add_row(*row)

        return table

    def create_performance_overview(
        self, all_reports: Dict[str, Dict[str, Any]]
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
                best_compressor = max(
                    report_data.items(), key=lambda x: x[1].get("avg5", 0)
                )
                overview_text.append(
                    f"🏆 Best {report_type.replace('_', ' ').title()}: "
                    f"{best_compressor[0]} ({best_compressor[1].get('avg5', 0):.4f})"
                )
            elif "f1" in str(report_data):
                # For F1 scores, higher is better
                sample_metrics = next(iter(report_data.values()))
                f1_metrics = [k for k in sample_metrics.keys() if k.endswith("_f1_avg")]
                if f1_metrics:
                    best_compressor = max(
                        report_data.items(), key=lambda x: x[1].get(f1_metrics[0], 0)
                    )
                    overview_text.append(
                        f"🏆 Best {report_type.title()}: "
                        f"{best_compressor[0]} ({best_compressor[1].get(f1_metrics[0], 0):.4f})"
                    )

        overview_panel = Panel(
            "\n".join(overview_text),
            title="📋 Evaluation Overview",
            title_align="left",
            border_style="blue",
        )

        return overview_panel

    def print_enhanced_report(self, all_reports: Dict[str, Dict[str, Any]]):
        """Print enhanced, beautifully formatted reports."""

        if RICH_AVAILABLE:
            self._print_rich_report(all_reports)
        else:
            self._print_fallback_report(all_reports)

    def _print_rich_report(self, all_reports: Dict[str, Dict[str, Any]]):
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
            else:
                table = self.create_summary_table(
                    report_data, f"{report_type.replace('_', ' ').title()} Metrics"
                )

            self.console.print(table)
            self.console.print()

        # Print footer
        self.console.print("✨ Report generation complete!", style="bold green")
        self.console.print()

    def _print_fallback_report(self, all_reports: Dict[str, Dict[str, Any]]):
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

            for compressor_group, metrics in sorted(report_data.items()):
                print(f"\n{compressor_group}:")
                for metric, value in sorted(metrics.items()):
                    # Create more readable metric names
                    display_name = metric.replace("_", " ").title()
                    if "Duration Depth" in display_name:
                        display_name = display_name.replace("Duration Depth 0", "Depth 0")
                        display_name = display_name.replace("Duration Depth 1", "Depth 1")
                        display_name = display_name.replace("Wdist", "W-Dist")
                        display_name = display_name.replace("Mape", "MAPE")
                    
                    if isinstance(value, float):
                        print(f"  {display_name}: {value:.4f}")
                    else:
                        print(f"  {display_name}: {value}")
            print()

        print("Report generation complete!")
        print()

    def save_enhanced_json_report(
        self, all_reports: Dict[str, Dict[str, Any]], output_path: pathlib.Path
    ):
        """Save an enhanced JSON report with metadata."""
        enhanced_report = {
            "metadata": {
                "generator": "GenTEval Enhanced Report Generator",
                "report_types": list(all_reports.keys()),
                "total_compressors": len(
                    set().union(*[report.keys() for report in all_reports.values()])
                ),
            },
            "reports": all_reports,
            "summary": {},
        }

        # Add summary statistics
        for report_type, report_data in all_reports.items():
            if not report_data:
                continue

            summary = {"compressors": list(report_data.keys())}

            if report_type in ["trace_rca", "micro_rank"]:
                # Calculate average accuracy across all compressors
                avg_accuracies = []
                for compressor_data in report_data.values():
                    avg_accuracies.append(compressor_data.get("avg5", 0))
                summary["overall_avg_accuracy"] = (
                    sum(avg_accuracies) / len(avg_accuracies) if avg_accuracies else 0
                )

            enhanced_report["summary"][report_type] = summary

        with open(output_path, "w") as f:
            json.dump(enhanced_report, f, indent=2, default=str)
