"""Rate over time report generator with MAPE and Cosine similarity analysis."""

from typing import Any

import numpy as np

from genteval.bin.utils import get_dir_with_root

from .base_report import BaseReport
from .duration_metrics import TimeSeriesComparisonMetric


class RateOverTimeReport(BaseReport):
    """Report generator for rate over time evaluation with MAPE and Cosine similarity."""

    def __init__(self, compressors, root_dir, plot=True):
        """Initialize the rate over time report generator."""
        super().__init__(compressors, root_dir)
        self.plot = plot
        self.time_series_metric = TimeSeriesComparisonMetric()

    def _calculate_rate_over_time_fidelity(
        self, original_data, compressed_data, group_key, compressor_name
    ):
        """Calculate MAPE and Cosine similarity using the generic metric class with correct formula."""
        return self.time_series_metric.process_count_over_time_series(
            original_data, compressed_data, group_key, compressor_name
        )

    def generate(self, run_dirs) -> dict[str, Any]:
        """Generate rate over time report with MAPE and Cosine similarity calculations."""
        results = {}

        for app_name, service, fault, run in run_dirs():
            for compressor in self.compressors:
                if compressor in {"original", "head_sampling_1"}:
                    self.print_skip_message(
                        f"Compressor {compressor} is not supported for rate over time evaluation, "
                        f"skipping for {app_name}_{service}_{fault}_{run}."
                    )
                    continue

                # Get original data
                original_results_file = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / "head_sampling_1"
                    / "evaluated"
                    / "rate_over_time_results.json"
                )

                if not original_results_file.exists():
                    self.print_skip_message(
                        f"Missing original rate over time results for {app_name}_{service}_{fault}_{run}"
                    )
                    continue

                # Get compressed data
                compressed_results_file = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / compressor
                    / "evaluated"
                    / "rate_over_time_results.json"
                )

                if not compressed_results_file.exists():
                    self.print_skip_message(
                        f"Missing compressed rate over time results for {compressor} in {app_name}_{service}_{fault}_{run}"
                    )
                    continue

                # Load data
                original_data = self.load_json_file(original_results_file)
                compressed_data = self.load_json_file(compressed_results_file)

                if not original_data or not compressed_data:
                    continue

                compressor_key = f"{app_name}_{compressor}"
                if compressor_key not in results:
                    results[compressor_key] = {}

                # Get all groups from original data's span_rate_by_time
                original_span_rate = original_data.get("span_rate_by_time", {})
                original_total_spans = original_data.get("total_spans_by_group", {})

                for group_key in original_span_rate:
                    fidelity_result = self._calculate_rate_over_time_fidelity(
                        original_data, compressed_data, group_key, compressor
                    )

                    # Calculate fidelity scores for this group
                    mape_fidelity = max(0, 100 - fidelity_result["mape"]) if not np.isinf(fidelity_result["mape"]) else 0.0
                    cosine_fidelity = fidelity_result["cosine_sim"] * 100

                    # Store per-group scores with count from original data
                    results[compressor_key][group_key] = {
                        "mape_fidelity": mape_fidelity,
                        "cosine_fidelity": cosine_fidelity,
                        "count": original_total_spans.get(group_key, 0),
                    }

        return results
