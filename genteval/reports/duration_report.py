"""Duration report generator with Wasserstein distance visualization."""

from typing import Any

from genteval.bin.utils import get_dir_with_root

from .base_report import BaseReport
from .duration_metrics import (
    DepthBeforeAfterMetric,
    TimeSeriesComparisonMetric,
    WassersteinDistanceMetric,
)


class DurationReport(BaseReport):
    """Report generator for duration evaluation with Wasserstein distance visualization."""

    def __init__(self, compressors, root_dir, plot=True):
        """Initialize the duration report generator."""
        super().__init__(compressors, root_dir)
        self.plot = plot

        self.wasserstein_metric = WassersteinDistanceMetric()
        self.depth_before_after_metric = DepthBeforeAfterMetric()
        self.time_series_metric = TimeSeriesComparisonMetric()

    def _has_before_after_incident_data_for_depth(self, original, results, depth):
        """Check if both datasets have before/after incident data for the given depth."""
        depth_str = str(depth)
        try:
            # Try to access all required data paths
            original["duration_before_incident_by_depth"][depth_str]["all"]
            original["duration_after_incident_by_depth"][depth_str]["all"]
            results["duration_before_incident_by_depth"][depth_str]["all"]
            results["duration_after_incident_by_depth"][depth_str]["all"]
        except KeyError:
            return False
        else:
            return True

    def generate(self, run_dirs) -> dict[str, Any]:
        """Generate duration report with Wasserstein distance calculations and visualizations."""
        for app_name, service, fault, run in run_dirs():
            for compressor in self.compressors:
                if compressor in {"original", "head_sampling_1"}:
                    self.print_skip_message(
                        f"Compressor {compressor} is not supported for duration evaluation, "
                        f"skipping for {app_name}_{service}_{fault}_{run}."
                    )
                    continue

                original_results_path = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / "head_sampling_1"
                    / "evaluated"
                    / "duration_results.json"
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
                    / "duration_results.json"
                )

                if not self.file_exists(results_path):
                    self.print_skip_message(
                        f"Results file {results_path} does not exist, skipping."
                    )
                    continue

                original = self.load_json_file(original_results_path)
                results = self.load_json_file(results_path)

                # Process duration data
                for group_key in original["duration"]:
                    if group_key not in results["duration"]:
                        continue

                    # Visualize and calculate Wasserstein distance
                    wdist = self.wasserstein_metric.visualize_wasserstein_distributions(
                        original["duration"][group_key],
                        results["duration"][group_key],
                        group_key,
                        compressor,
                        app_name,
                        service,
                        fault,
                        run,
                        self.plot,
                    )

                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group][f"{group_key}_wdist"]["values"].append(
                        wdist
                    )

                for group_key in original["duration_by_time_percentiles"]:
                    if group_key not in results["duration_by_time_percentiles"]:
                        continue

                    mape_count_results = (
                        self.time_series_metric.process_duration_percentile_time_series(
                            original["duration_by_time_percentiles"][group_key],
                            results["duration_by_time_percentiles"][group_key],
                            group_key,
                            compressor,
                            app_name,
                            service,
                            fault,
                            run,
                            self.plot,
                        )
                    )

                    # Store MAPE and cosine sim results in report
                    report_group = f"{app_name}_{compressor}"

                    # Store results for each percentile (structure is now {percentile: {mape, cosine_sim}})
                    for percentile in mape_count_results:
                        key_prefix = f"{group_key}_{percentile}"
                        self.report[report_group][f"{key_prefix}_mape"][
                            "values"
                        ].append(mape_count_results[percentile]["mape"])
                        self.report[report_group][f"{key_prefix}_cosine_sim"][
                            "values"
                        ].append(mape_count_results[percentile]["cosine_sim"])

                # Process duration before/after incident data for depths 0 and 1
                """
                for depth in [0, 1]:
                    if self._has_before_after_incident_data_for_depth(
                        original, results, depth
                    ):
                        depth_str = str(depth)
                        # Generate before/after incident visualization
                        wdist_before, wdist_after = (
                            self.depth_before_after_metric.visualize_depth_before_after_incident(
                                original["duration_before_incident_by_depth"][
                                    depth_str
                                ]["all"],
                                original["duration_after_incident_by_depth"][depth_str][
                                    "all"
                                ],
                                results["duration_before_incident_by_depth"][depth_str][
                                    "all"
                                ],
                                results["duration_after_incident_by_depth"][depth_str][
                                    "all"
                                ],
                                compressor,
                                app_name=app_name,
                                service=service,
                                fault=fault,
                                run=run,
                                depth=depth,
                                plot=self.plot,
                            )
                        )

                        # Store Wasserstein distances in report
                        report_group = f"{app_name}_{compressor}"

                        # Only add finite distances to avoid inf values in averages
                        if wdist_before != float("inf"):
                            self.report[report_group][
                                f"duration_depth_{depth}_before_wdist"
                            ].append(wdist_before)
                        if wdist_after != float("inf"):
                            self.report[report_group][
                                f"duration_depth_{depth}_after_wdist"
                            ].append(wdist_after)

                """

        # Calculate averages and clean up
        for report_group in self.report.values():
            for metric_group in report_group.values():
                if isinstance(metric_group, dict) and "values" in metric_group:
                    metric_group["avg"] = (
                        sum(metric_group["values"]) / len(metric_group["values"])
                        if metric_group["values"]
                        else float("nan")
                    )
                    del metric_group["values"]

        return dict(self.report)
