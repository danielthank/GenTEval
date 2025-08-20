"""Duration report generator with Wasserstein distance visualization."""

from typing import Any

from genteval.bin.utils import get_dir_with_root

from .base_report import BaseReport
from .duration_metrics import (
    DepthBeforeAfterMetric,
    PercentileComparisonMetric,
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
        self.percentile_comparison_metric = PercentileComparisonMetric()

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
                for group in original["duration"]:
                    if group not in results["duration"]:
                        continue

                    # Visualize and calculate Wasserstein distance
                    wdist = self.wasserstein_metric.visualize_wasserstein_distributions(
                        original["duration"][group],
                        results["duration"][group],
                        f"duration_{group}",
                        compressor,
                        app_name=app_name,
                        service=service,
                        fault=fault,
                        run=run,
                        plot=self.plot,
                    )

                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["duration_wdist"].append(wdist)

                # Process duration_pair data
                for group in original["duration_pair"]:
                    if group not in results["duration_pair"]:
                        continue

                    # Visualize and calculate Wasserstein distance
                    wdist = self.wasserstein_metric.visualize_wasserstein_distributions(
                        original["duration_pair"][group],
                        results["duration_pair"][group],
                        f"duration_pair_{group}",
                        compressor,
                        app_name=app_name,
                        service=service,
                        fault=fault,
                        run=run,
                        plot=self.plot,
                    )

                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["duration_pair_wdist"].append(wdist)

                # Process duration_depth_0 data
                if "duration_depth_0" in original and "duration_depth_0" in results:
                    for group in original["duration_depth_0"]:
                        if group not in results["duration_depth_0"]:
                            continue

                        # Visualize and calculate Wasserstein distance
                        wdist = (
                            self.wasserstein_metric.visualize_wasserstein_distributions(
                                original["duration_depth_0"][group],
                                results["duration_depth_0"][group],
                                f"duration_depth_0_{group}",
                                compressor,
                                app_name=app_name,
                                service=service,
                                fault=fault,
                                run=run,
                                plot=self.plot,
                            )
                        )

                        report_group = f"{app_name}_{compressor}"
                        self.report[report_group]["duration_depth_0_wdist"].append(
                            wdist
                        )

                # Process duration_depth_1 data
                if "duration_depth_1" in original and "duration_depth_1" in results:
                    for group in original["duration_depth_1"]:
                        if group not in results["duration_depth_1"]:
                            continue

                        # Visualize and calculate Wasserstein distance
                        wdist = (
                            self.wasserstein_metric.visualize_wasserstein_distributions(
                                original["duration_depth_1"][group],
                                results["duration_depth_1"][group],
                                f"duration_depth_1_{group}",
                                compressor,
                                app_name=app_name,
                                service=service,
                                fault=fault,
                                run=run,
                                plot=self.plot,
                            )
                        )

                        report_group = f"{app_name}_{compressor}"
                        self.report[report_group]["duration_depth_1_wdist"].append(
                            wdist
                        )

                # Process duration before/after incident data for depths 0 and 1
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

                # Process duration_by_depth_by_service data for all percentiles
                if (
                    "duration_by_depth_by_service" in original
                    and "duration_by_depth_by_service" in results
                ):
                    mape_count_results = self.percentile_comparison_metric.process_duration_by_depth_by_service(
                        original["duration_by_depth_by_service"],
                        results["duration_by_depth_by_service"],
                        compressor,
                        app_name,
                        service,
                        fault,
                        run,
                        plot=self.plot,
                    )

                    # Store MAPE and count results in report
                    report_group = f"{app_name}_{compressor}"
                    for depth in mape_count_results:
                        for percentile in mape_count_results[depth]:
                            key_prefix = f"duration_depth_{depth}_{percentile}"
                            self.report[report_group][f"{key_prefix}_mape_runs"].append(
                                mape_count_results[depth][percentile]["mape"]
                            )
                            self.report[report_group][
                                f"{key_prefix}_count_runs"
                            ].append(mape_count_results[depth][percentile]["counts"])
                            self.report[report_group][
                                f"{key_prefix}_cosine_sim"
                            ].append(
                                mape_count_results[depth][percentile]["cosine_sim"]
                            )

        # Calculate averages and clean up
        for group in self.report:
            if "duration_wdist" in self.report[group]:
                self.report[group]["duration_wdist_avg"] = sum(
                    self.report[group]["duration_wdist"]
                ) / len(self.report[group]["duration_wdist"])
                del self.report[group]["duration_wdist"]

            if "duration_pair_wdist" in self.report[group]:
                self.report[group]["duration_pair_wdist_avg"] = sum(
                    self.report[group]["duration_pair_wdist"]
                ) / len(self.report[group]["duration_pair_wdist"])
                del self.report[group]["duration_pair_wdist"]

            if "duration_depth_0_wdist" in self.report[group]:
                self.report[group]["duration_depth_0_wdist_avg"] = sum(
                    self.report[group]["duration_depth_0_wdist"]
                ) / len(self.report[group]["duration_depth_0_wdist"])
                del self.report[group]["duration_depth_0_wdist"]

            if "duration_depth_1_wdist" in self.report[group]:
                self.report[group]["duration_depth_1_wdist_avg"] = sum(
                    self.report[group]["duration_depth_1_wdist"]
                ) / len(self.report[group]["duration_depth_1_wdist"])
                del self.report[group]["duration_depth_1_wdist"]

            # Calculate averages for depth 0 before/after incident Wasserstein distances
            if "duration_depth_0_before_wdist" in self.report[group]:
                if self.report[group]["duration_depth_0_before_wdist"]:
                    self.report[group]["duration_depth_0_before_wdist_avg"] = sum(
                        self.report[group]["duration_depth_0_before_wdist"]
                    ) / len(self.report[group]["duration_depth_0_before_wdist"])
                else:
                    self.report[group]["duration_depth_0_before_wdist_avg"] = float(
                        "inf"
                    )
                del self.report[group]["duration_depth_0_before_wdist"]

            if "duration_depth_0_after_wdist" in self.report[group]:
                if self.report[group]["duration_depth_0_after_wdist"]:
                    self.report[group]["duration_depth_0_after_wdist_avg"] = sum(
                        self.report[group]["duration_depth_0_after_wdist"]
                    ) / len(self.report[group]["duration_depth_0_after_wdist"])
                else:
                    self.report[group]["duration_depth_0_after_wdist_avg"] = float(
                        "inf"
                    )
                del self.report[group]["duration_depth_0_after_wdist"]

            # Calculate averages for depth 1 before/after incident Wasserstein distances
            if "duration_depth_1_before_wdist" in self.report[group]:
                if self.report[group]["duration_depth_1_before_wdist"]:
                    self.report[group]["duration_depth_1_before_wdist_avg"] = sum(
                        self.report[group]["duration_depth_1_before_wdist"]
                    ) / len(self.report[group]["duration_depth_1_before_wdist"])
                else:
                    self.report[group]["duration_depth_1_before_wdist_avg"] = float(
                        "inf"
                    )
                del self.report[group]["duration_depth_1_before_wdist"]

            if "duration_depth_1_after_wdist" in self.report[group]:
                if self.report[group]["duration_depth_1_after_wdist"]:
                    self.report[group]["duration_depth_1_after_wdist_avg"] = sum(
                        self.report[group]["duration_depth_1_after_wdist"]
                    ) / len(self.report[group]["duration_depth_1_after_wdist"])
                else:
                    self.report[group]["duration_depth_1_after_wdist_avg"] = float(
                        "inf"
                    )
                del self.report[group]["duration_depth_1_after_wdist"]

            # Calculate averages for all percentile metrics (depth 0-4, p0-p100)
            percentiles = [
                "p0",
                "p10",
                "p20",
                "p30",
                "p40",
                "p50",
                "p60",
                "p70",
                "p80",
                "p90",
                "p100",
            ]
            for depth in range(5):  # 0, 1, 2, 3, 4
                for percentile in percentiles:
                    key_prefix = f"duration_depth_{depth}_{percentile}"

                    # Calculate MAPE averages
                    mape_key = f"{key_prefix}_mape_runs"
                    if mape_key in self.report[group]:
                        all_run_mapes = []
                        all_run_counts = []

                        for i, run_mape in enumerate(self.report[group][mape_key]):
                            run_counts = self.report[group][f"{key_prefix}_count_runs"][
                                i
                            ]
                            for service in run_mape:
                                if service in run_counts:
                                    all_run_mapes.append(run_mape[service])
                                    all_run_counts.append(run_counts[service])

                        if all_run_mapes and sum(all_run_counts) > 0:
                            # Calculate weighted average
                            total_weighted_mape = sum(
                                mape * count
                                for mape, count in zip(
                                    all_run_mapes, all_run_counts, strict=False
                                )
                            )
                            total_count = sum(all_run_counts)
                            self.report[group][f"{key_prefix}_mape_avg"] = (
                                total_weighted_mape / total_count
                            )

                    # Calculate cosine similarity averages
                    cosine_key = f"{key_prefix}_cosine_sim"
                    if cosine_key in self.report[group]:
                        if self.report[group][cosine_key]:
                            # Calculate weighted average cosine similarity across all runs and all services
                            all_cosine_sims = []
                            all_cosine_counts = []

                            for i, run_cosine_sim in enumerate(
                                self.report[group][cosine_key]
                            ):
                                run_counts = self.report[group][
                                    f"{key_prefix}_count_runs"
                                ][i]
                                for service in run_cosine_sim:
                                    if service in run_counts:
                                        all_cosine_sims.append(run_cosine_sim[service])
                                        all_cosine_counts.append(run_counts[service])

                            if all_cosine_sims and sum(all_cosine_counts) > 0:
                                # Calculate weighted average
                                total_weighted_cosine_sim = sum(
                                    cosine_sim * count
                                    for cosine_sim, count in zip(
                                        all_cosine_sims, all_cosine_counts, strict=False
                                    )
                                )
                                total_count = sum(all_cosine_counts)
                                self.report[group][f"{key_prefix}_cosine_sim_avg"] = (
                                    total_weighted_cosine_sim / total_count
                                )
                            else:
                                self.report[group][f"{key_prefix}_cosine_sim_avg"] = (
                                    float("nan")
                                )
                        else:
                            self.report[group][f"{key_prefix}_cosine_sim_avg"] = float(
                                "nan"
                            )
                        del self.report[group][cosine_key]

        return dict(self.report)
