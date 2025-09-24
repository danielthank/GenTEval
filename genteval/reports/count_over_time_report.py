"""Count over time report generator with MAPE and Cosine similarity analysis."""

from typing import Any

import numpy as np

from genteval.bin.utils import get_dir_with_root

from .base_report import BaseReport
from .duration_metrics import TimeSeriesComparisonMetric


class CountOverTimeReport(BaseReport):
    """Report generator for count over time evaluation with MAPE and Cosine similarity."""

    def __init__(self, compressors, root_dir, plot=True):
        """Initialize the count over time report generator."""
        super().__init__(compressors, root_dir)
        self.plot = plot
        self.time_series_metric = TimeSeriesComparisonMetric()

    def _calculate_count_over_time_fidelity(
        self, original_data, compressed_data, group_key
    ):
        """Calculate MAPE and Cosine similarity using the generic metric class with correct formula."""
        return self.time_series_metric.process_count_over_time_series(
            original_data, compressed_data, group_key
        )

    def generate(self, run_dirs) -> dict[str, Any]:
        """Generate count over time report with MAPE and Cosine similarity calculations."""
        results = {}

        for app_name, service, fault, run in run_dirs():
            for compressor in self.compressors:
                if compressor in {"original", "head_sampling_1"}:
                    self.print_skip_message(
                        f"Compressor {compressor} is not supported for count over time evaluation, "
                        f"skipping for {app_name}_{service}_{fault}_{run}."
                    )
                    continue

                # Get original data
                original_results_file = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / "head_sampling_1"
                    / "evaluated"
                    / "count_over_time_results.json"
                )

                if not original_results_file.exists():
                    self.print_skip_message(
                        f"Missing original count over time results for {app_name}_{service}_{fault}_{run}"
                    )
                    continue

                # Get compressed data
                compressed_results_file = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / compressor
                    / "evaluated"
                    / "count_over_time_results.json"
                )

                if not compressed_results_file.exists():
                    self.print_skip_message(
                        f"Missing compressed count over time results for {compressor} in {app_name}_{service}_{fault}_{run}"
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

                # Calculate fidelity for each depth level (0-4) and "all"
                depth_mape_scores = []
                depth_cosine_scores = []

                for depth in range(5):
                    group_key = f"depth_{depth}"
                    fidelity_result = self._calculate_count_over_time_fidelity(
                        original_data, compressed_data, group_key
                    )

                    if not np.isinf(fidelity_result["mape"]):
                        depth_mape_scores.append(max(0, 100 - fidelity_result["mape"]))
                    depth_cosine_scores.append(fidelity_result["cosine_sim"] * 100)

                # Also calculate for "all" spans
                all_fidelity_result = self._calculate_count_over_time_fidelity(
                    original_data, compressed_data, "all"
                )
                if not np.isinf(all_fidelity_result["mape"]):
                    depth_mape_scores.append(max(0, 100 - all_fidelity_result["mape"]))
                depth_cosine_scores.append(all_fidelity_result["cosine_sim"] * 100)

                # Calculate overall fidelity scores
                overall_mape_fidelity = (
                    np.mean(depth_mape_scores) if depth_mape_scores else 0.0
                )
                overall_cosine_fidelity = (
                    np.mean(depth_cosine_scores) if depth_cosine_scores else 0.0
                )

                results[compressor_key]["count_over_time_mape_fidelity_score"] = (
                    overall_mape_fidelity
                )
                results[compressor_key]["count_over_time_cosine_fidelity_score"] = (
                    overall_cosine_fidelity
                )

        return results
