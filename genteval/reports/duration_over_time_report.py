"""Duration over time report generator matching rate over time structure."""

from typing import Any

from genteval.bin.utils import get_dir_with_root

from .base_report import BaseReport
from .duration_metrics import TimeSeriesComparisonMetric


class DurationOverTimeReport(BaseReport):
    """Report generator for duration over time evaluation with per-group fidelity scores."""

    def __init__(self, compressors, root_dir, plot=False):
        """Initialize the duration over time report generator."""
        super().__init__(compressors, root_dir)
        self.plot = plot
        self.time_series_metric = TimeSeriesComparisonMetric()

    def generate(self, run_dirs) -> dict[str, Any]:
        """Generate duration over time report with per-group fidelity scores."""
        for app_name, service, fault, run in run_dirs():
            for compressor in self.compressors:
                if compressor in {"original"}:
                    self.print_skip_message(
                        f"Compressor {compressor} is not supported for duration_over_time evaluation, "
                        f"skipping for {app_name}_{service}_{fault}_{run}."
                    )
                    continue

                original_results_path = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / "head_sampling_1"
                    / "evaluated"
                    / "duration_over_time_results.json"
                )

                if not self.file_exists(original_results_path):
                    self.print_skip_message(
                        f"Original results file {original_results_path} does not exist, skipping."
                    )
                    continue

                results_path = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / compressor
                    / "evaluated"
                    / "duration_over_time_results.json"
                )

                if not self.file_exists(results_path):
                    self.print_skip_message(
                        f"Results file {results_path} does not exist, skipping."
                    )
                    continue

                original = self.load_json_file(original_results_path)
                results = self.load_json_file(results_path)

                report_group = f"{app_name}_{compressor}"

                # Process each group in duration_percentiles_by_time
                original_percentiles = original.get("duration_percentiles_by_time", {})
                results_percentiles = results.get("duration_percentiles_by_time", {})
                original_totals = original.get("total_spans_by_group", {})

                for group_key in original_percentiles:
                    if group_key not in results_percentiles:
                        continue

                    # Calculate MAPE and cosine similarity for each percentile
                    percentile_results = (
                        self.time_series_metric.process_duration_percentile_time_series(
                            original_percentiles[group_key],
                            results_percentiles[group_key],
                            group_key,
                            compressor,
                            app_name,
                            service,
                            fault,
                            run,
                            self.plot,
                        )
                    )

                    # Calculate average MAPE and cosine across all percentiles
                    mape_values = []
                    cosine_values = []

                    for percentile_data in percentile_results.values():
                        mape = percentile_data["mape"]
                        cosine_sim = percentile_data["cosine_sim"]

                        # Filter out inf values
                        if mape != float("inf"):
                            mape_values.append(mape)
                        if cosine_sim != float("inf"):
                            cosine_values.append(cosine_sim)

                    # Calculate averages
                    avg_mape = (
                        sum(mape_values) / len(mape_values)
                        if mape_values
                        else float("inf")
                    )
                    avg_cosine = (
                        sum(cosine_values) / len(cosine_values)
                        if cosine_values
                        else 0.0
                    )

                    # Convert MAPE to fidelity (match rate_over_time naming)
                    mape_fidelity = (
                        max(0, 100 - avg_mape) if avg_mape != float("inf") else 0.0
                    )
                    cosine_fidelity = avg_cosine * 100  # Convert to percentage

                    # Get span count for this group
                    count = original_totals.get(group_key, 0)

                    # Store per-group results (matching rate_over_time structure)
                    self.report[report_group][group_key] = {
                        "mape_fidelity": mape_fidelity,
                        "cosine_fidelity": cosine_fidelity,
                        "count": count,
                    }

        return dict(self.report)
