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
        # Create output directories for visualizations
        self.viz_output_dir = root_dir / "visualizations" / "duration"
        self.duration_all_dir = self.viz_output_dir / "duration_all_wasserstein_dist"
        self.duration_pair_dir = (
            self.viz_output_dir / "duration_pair_all_wasserstein_dist"
        )
        self.duration_depth_0_dir = (
            self.viz_output_dir / "duration_depth_0_wasserstein_dist"
        )
        self.duration_depth_1_dir = (
            self.viz_output_dir / "duration_depth_1_wasserstein_dist"
        )
        self.duration_depth_0_p50_dir = self.viz_output_dir / "duration_depth_0_p50"
        self.duration_depth_0_p90_dir = self.viz_output_dir / "duration_depth_0_p90"
        self.duration_depth_1_p50_dir = self.viz_output_dir / "duration_depth_1_p50"
        self.duration_depth_1_p90_dir = self.viz_output_dir / "duration_depth_1_p90"
        self.duration_depth_0_before_after_dir = (
            self.viz_output_dir / "duration_depth_0_before_after_incident"
        )
        self.duration_depth_1_before_after_dir = (
            self.viz_output_dir / "duration_depth_1_before_after_incident"
        )
        self.duration_depth_0_by_service_dir = (
            self.viz_output_dir / "duration_depth_0_by_service_wasserstein_dist"
        )
        self.duration_depth_1_by_service_dir = (
            self.viz_output_dir / "duration_depth_1_by_service_wasserstein_dist"
        )

        # Create all subdirectories only if plotting is enabled
        if self.plot:
            self.duration_all_dir.mkdir(parents=True, exist_ok=True)
            self.duration_pair_dir.mkdir(parents=True, exist_ok=True)
            self.duration_depth_0_dir.mkdir(parents=True, exist_ok=True)
            self.duration_depth_1_dir.mkdir(parents=True, exist_ok=True)
            self.duration_depth_0_p50_dir.mkdir(parents=True, exist_ok=True)
            self.duration_depth_0_p90_dir.mkdir(parents=True, exist_ok=True)
            self.duration_depth_1_p50_dir.mkdir(parents=True, exist_ok=True)
            self.duration_depth_1_p90_dir.mkdir(parents=True, exist_ok=True)
            self.duration_depth_0_before_after_dir.mkdir(parents=True, exist_ok=True)
            self.duration_depth_1_before_after_dir.mkdir(parents=True, exist_ok=True)
            self.duration_depth_0_by_service_dir.mkdir(parents=True, exist_ok=True)
            self.duration_depth_1_by_service_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metric classes
        self.output_dirs = {
            "viz_output_dir": self.viz_output_dir,
            "duration_all_dir": self.duration_all_dir,
            "duration_pair_dir": self.duration_pair_dir,
            "duration_depth_0_dir": self.duration_depth_0_dir,
            "duration_depth_1_dir": self.duration_depth_1_dir,
            "duration_depth_0_p50_dir": self.duration_depth_0_p50_dir,
            "duration_depth_0_p90_dir": self.duration_depth_0_p90_dir,
            "duration_depth_1_p50_dir": self.duration_depth_1_p50_dir,
            "duration_depth_1_p90_dir": self.duration_depth_1_p90_dir,
            "duration_depth_0_before_after_dir": self.duration_depth_0_before_after_dir,
            "duration_depth_1_before_after_dir": self.duration_depth_1_before_after_dir,
            "duration_depth_0_by_service_dir": self.duration_depth_0_by_service_dir,
            "duration_depth_1_by_service_dir": self.duration_depth_1_by_service_dir,
        }

        self.wasserstein_metric = WassersteinDistanceMetric(self.output_dirs)
        self.depth_before_after_metric = DepthBeforeAfterMetric(self.output_dirs)
        self.percentile_comparison_metric = PercentileComparisonMetric(self.output_dirs)

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
                        f"{app_name}_{service}_{fault}_{run}",
                        plot=self.plot,
                    )
                    """
                    wdist = wasserstein_distance(
                        original["duration"][group],
                        results["duration"][group],
                    )
                    """

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
                        f"{app_name}_{service}_{fault}_{run}",
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
                                f"{app_name}_{service}_{fault}_{run}",
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
                                f"{app_name}_{service}_{fault}_{run}",
                                plot=self.plot,
                            )
                        )

                        report_group = f"{app_name}_{compressor}"
                        self.report[report_group]["duration_depth_1_wdist"].append(
                            wdist
                        )

                # Process duration_depth_0_by_service data
                if (
                    "duration_depth_0_by_service" in original
                    and "duration_depth_0_by_service" in results
                ):
                    # Visualize and calculate weighted average Wasserstein distance
                    weighted_wdist = (
                        self.wasserstein_metric.visualize_wasserstein_by_service(
                            original["duration_depth_0_by_service"],
                            results["duration_depth_0_by_service"],
                            "duration_depth_0_by_service",
                            compressor,
                            f"{app_name}_{service}_{fault}_{run}",
                            plot=self.plot,
                        )
                    )

                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group][
                        "duration_depth_0_by_service_wdist"
                    ].append(weighted_wdist)

                # Process duration_depth_1_by_service data
                if (
                    "duration_depth_1_by_service" in original
                    and "duration_depth_1_by_service" in results
                ):
                    # Visualize and calculate weighted average Wasserstein distance
                    weighted_wdist = (
                        self.wasserstein_metric.visualize_wasserstein_by_service(
                            original["duration_depth_1_by_service"],
                            results["duration_depth_1_by_service"],
                            "duration_depth_1_by_service",
                            compressor,
                            f"{app_name}_{service}_{fault}_{run}",
                            plot=self.plot,
                        )
                    )

                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group][
                        "duration_depth_1_by_service_wdist"
                    ].append(weighted_wdist)

                # Process duration_depth_0_p90_by_service data if available
                if (
                    "duration_depth_0_p90_by_service" in original
                    and "duration_depth_0_p90_by_service" in results
                ):
                    # Generate p90 visualization immediately for this run
                    mape_count_results = self.percentile_comparison_metric.visualize_duration_percentile_comparison(
                        original["duration_depth_0_p90_by_service"],
                        results["duration_depth_0_p90_by_service"],
                        "P90",
                        compressor,
                        f"{app_name}_{service}_{fault}_{run}",
                        plot=self.plot,
                    )

                    # Store MAPE and count results in report
                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["duration_depth_0_p90_mape_runs"].append(
                        mape_count_results["mape"]
                    )
                    self.report[report_group]["duration_depth_0_p90_count_runs"].append(
                        mape_count_results["counts"]
                    )
                    self.report[report_group]["duration_depth_0_p90_cosine_sim"].append(
                        mape_count_results["cosine_sim"]
                    )

                # Process duration_depth_0_p50_by_service data if available
                if (
                    "duration_depth_0_p50_by_service" in original
                    and "duration_depth_0_p50_by_service" in results
                ):
                    # Generate p50 visualization immediately for this run
                    mape_count_results = self.percentile_comparison_metric.visualize_duration_percentile_comparison(
                        original["duration_depth_0_p50_by_service"],
                        results["duration_depth_0_p50_by_service"],
                        "P50",
                        compressor,
                        f"{app_name}_{service}_{fault}_{run}",
                        plot=self.plot,
                    )

                    # Store MAPE and count results in report
                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["duration_depth_0_p50_mape_runs"].append(
                        mape_count_results["mape"]
                    )
                    self.report[report_group]["duration_depth_0_p50_count_runs"].append(
                        mape_count_results["counts"]
                    )
                    self.report[report_group]["duration_depth_0_p50_cosine_sim"].append(
                        mape_count_results["cosine_sim"]
                    )

                # Process duration_depth_1_p90_by_service data if available
                if (
                    "duration_depth_1_p90_by_service" in original
                    and "duration_depth_1_p90_by_service" in results
                ):
                    # Generate p90 visualization immediately for this run
                    mape_count_results = self.percentile_comparison_metric.visualize_duration_percentile_comparison(
                        original["duration_depth_1_p90_by_service"],
                        results["duration_depth_1_p90_by_service"],
                        "P90",
                        compressor,
                        f"{app_name}_{service}_{fault}_{run}",
                        plot=self.plot,
                        depth=1,
                    )

                    # Store MAPE and count results in report
                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["duration_depth_1_p90_mape_runs"].append(
                        mape_count_results["mape"]
                    )
                    self.report[report_group]["duration_depth_1_p90_count_runs"].append(
                        mape_count_results["counts"]
                    )
                    self.report[report_group]["duration_depth_1_p90_cosine_sim"].append(
                        mape_count_results["cosine_sim"]
                    )

                # Process duration_depth_1_p50_by_service data if available
                if (
                    "duration_depth_1_p50_by_service" in original
                    and "duration_depth_1_p50_by_service" in results
                ):
                    # Generate p50 visualization immediately for this run
                    mape_count_results = self.percentile_comparison_metric.visualize_duration_percentile_comparison(
                        original["duration_depth_1_p50_by_service"],
                        results["duration_depth_1_p50_by_service"],
                        "P50",
                        compressor,
                        f"{app_name}_{service}_{fault}_{run}",
                        plot=self.plot,
                        depth=1,
                    )

                    # Store MAPE and count results in report
                    report_group = f"{app_name}_{compressor}"
                    self.report[report_group]["duration_depth_1_p50_mape_runs"].append(
                        mape_count_results["mape"]
                    )
                    self.report[report_group]["duration_depth_1_p50_count_runs"].append(
                        mape_count_results["counts"]
                    )
                    self.report[report_group]["duration_depth_1_p50_cosine_sim"].append(
                        mape_count_results["cosine_sim"]
                    )

                # Process depth 0 duration before/after incident data
                if (
                    "duration_depth_0_before_incident" in original
                    and "duration_depth_0_after_incident" in original
                    and "duration_depth_0_before_incident" in results
                    and "duration_depth_0_after_incident" in results
                ):
                    # Process "all" service data for before/after incident
                    if (
                        "all" in original["duration_depth_0_before_incident"]
                        and "all" in original["duration_depth_0_after_incident"]
                        and "all" in results["duration_depth_0_before_incident"]
                        and "all" in results["duration_depth_0_after_incident"]
                    ):
                        # Generate before/after incident visualization for depth 0
                        wdist_before, wdist_after = (
                            self.depth_before_after_metric.visualize_depth_before_after_incident(
                                original["duration_depth_0_before_incident"]["all"],
                                original["duration_depth_0_after_incident"]["all"],
                                results["duration_depth_0_before_incident"]["all"],
                                results["duration_depth_0_after_incident"]["all"],
                                compressor,
                                f"{app_name}_{service}_{fault}_{run}",
                                depth=0,
                                plot=self.plot,
                            )
                        )

                        # Store Wasserstein distances in report
                        report_group = f"{app_name}_{compressor}"

                        # Only add finite distances to avoid inf values in averages
                        if wdist_before != float("inf"):
                            self.report[report_group][
                                "duration_depth_0_before_wdist"
                            ].append(wdist_before)
                        if wdist_after != float("inf"):
                            self.report[report_group][
                                "duration_depth_0_after_wdist"
                            ].append(wdist_after)

                # Process depth 1 duration before/after incident data
                if (
                    "duration_depth_1_before_incident" in original
                    and "duration_depth_1_after_incident" in original
                    and "duration_depth_1_before_incident" in results
                    and "duration_depth_1_after_incident" in results
                ):
                    # Process "all" service data for before/after incident
                    if (
                        "all" in original["duration_depth_1_before_incident"]
                        and "all" in original["duration_depth_1_after_incident"]
                        and "all" in results["duration_depth_1_before_incident"]
                        and "all" in results["duration_depth_1_after_incident"]
                    ):
                        # Generate before/after incident visualization for depth 1
                        wdist_before, wdist_after = (
                            self.depth_before_after_metric.visualize_depth_before_after_incident(
                                original["duration_depth_1_before_incident"]["all"],
                                original["duration_depth_1_after_incident"]["all"],
                                results["duration_depth_1_before_incident"]["all"],
                                results["duration_depth_1_after_incident"]["all"],
                                compressor,
                                f"{app_name}_{service}_{fault}_{run}",
                                depth=1,
                                plot=self.plot,
                            )
                        )

                        # Store Wasserstein distances in report
                        report_group = f"{app_name}_{compressor}"

                        # Only add finite distances to avoid inf values in averages
                        if wdist_before != float("inf"):
                            self.report[report_group][
                                "duration_depth_1_before_wdist"
                            ].append(wdist_before)
                        if wdist_after != float("inf"):
                            self.report[report_group][
                                "duration_depth_1_after_wdist"
                            ].append(wdist_after)

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

            if "duration_depth_0_by_service_wdist" in self.report[group]:
                self.report[group]["duration_depth_0_by_service_wdist_avg"] = sum(
                    self.report[group]["duration_depth_0_by_service_wdist"]
                ) / len(self.report[group]["duration_depth_0_by_service_wdist"])
                del self.report[group]["duration_depth_0_by_service_wdist"]

            if "duration_depth_1_by_service_wdist" in self.report[group]:
                self.report[group]["duration_depth_1_by_service_wdist_avg"] = sum(
                    self.report[group]["duration_depth_1_by_service_wdist"]
                ) / len(self.report[group]["duration_depth_1_by_service_wdist"])
                del self.report[group]["duration_depth_1_by_service_wdist"]

            if "duration_depth_0_p90_mape_runs" in self.report[group]:
                # Calculate weighted average MAPE across all runs and all services
                all_run_mapes = []
                all_run_counts = []

                for i, run_mape in enumerate(
                    self.report[group]["duration_depth_0_p90_mape_runs"]
                ):
                    run_counts = self.report[group]["duration_depth_0_p90_count_runs"][
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
                    self.report[group]["duration_depth_0_p90_mape_avg"] = (
                        total_weighted_mape / total_count
                    )

            if "duration_depth_0_p50_mape_runs" in self.report[group]:
                # Calculate weighted average MAPE across all runs and all services
                all_run_mapes = []
                all_run_counts = []

                for i, run_mape in enumerate(
                    self.report[group]["duration_depth_0_p50_mape_runs"]
                ):
                    run_counts = self.report[group]["duration_depth_0_p50_count_runs"][
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
                    self.report[group]["duration_depth_0_p50_mape_avg"] = (
                        total_weighted_mape / total_count
                    )

            if "duration_depth_1_p90_mape_runs" in self.report[group]:
                # Calculate weighted average MAPE across all runs and all services
                all_run_mapes = []
                all_run_counts = []

                for i, run_mape in enumerate(
                    self.report[group]["duration_depth_1_p90_mape_runs"]
                ):
                    run_counts = self.report[group]["duration_depth_1_p90_count_runs"][
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
                    self.report[group]["duration_depth_1_p90_mape_avg"] = (
                        total_weighted_mape / total_count
                    )

            if "duration_depth_1_p50_mape_runs" in self.report[group]:
                # Calculate weighted average MAPE across all runs and all services
                all_run_mapes = []
                all_run_counts = []

                for i, run_mape in enumerate(
                    self.report[group]["duration_depth_1_p50_mape_runs"]
                ):
                    run_counts = self.report[group]["duration_depth_1_p50_count_runs"][
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
                    self.report[group]["duration_depth_1_p50_mape_avg"] = (
                        total_weighted_mape / total_count
                    )

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

            # Calculate averages for duration percentile cosine similarities
            if "duration_depth_0_p90_cosine_sim" in self.report[group]:
                if self.report[group]["duration_depth_0_p90_cosine_sim"]:
                    # Calculate weighted average cosine similarity across all runs and all services
                    all_cosine_sims = []
                    all_cosine_counts = []

                    for i, run_cosine_sim in enumerate(
                        self.report[group]["duration_depth_0_p90_cosine_sim"]
                    ):
                        run_counts = self.report[group][
                            "duration_depth_0_p90_count_runs"
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
                        self.report[group]["duration_depth_0_p90_cosine_sim_avg"] = (
                            total_weighted_cosine_sim / total_count
                        )
                    else:
                        self.report[group]["duration_depth_0_p90_cosine_sim_avg"] = (
                            float("nan")
                        )
                else:
                    self.report[group]["duration_depth_0_p90_cosine_sim_avg"] = float(
                        "nan"
                    )
                del self.report[group]["duration_depth_0_p90_cosine_sim"]

            if "duration_depth_0_p50_cosine_sim" in self.report[group]:
                if self.report[group]["duration_depth_0_p50_cosine_sim"]:
                    # Calculate weighted average cosine similarity across all runs and all services
                    all_cosine_sims = []
                    all_cosine_counts = []

                    for i, run_cosine_sim in enumerate(
                        self.report[group]["duration_depth_0_p50_cosine_sim"]
                    ):
                        run_counts = self.report[group][
                            "duration_depth_0_p50_count_runs"
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
                        self.report[group]["duration_depth_0_p50_cosine_sim_avg"] = (
                            total_weighted_cosine_sim / total_count
                        )
                    else:
                        self.report[group]["duration_depth_0_p50_cosine_sim_avg"] = (
                            float("nan")
                        )
                else:
                    self.report[group]["duration_depth_0_p50_cosine_sim_avg"] = float(
                        "nan"
                    )
                del self.report[group]["duration_depth_0_p50_cosine_sim"]

            if "duration_depth_1_p90_cosine_sim" in self.report[group]:
                if self.report[group]["duration_depth_1_p90_cosine_sim"]:
                    # Calculate weighted average cosine similarity across all runs and all services
                    all_cosine_sims = []
                    all_cosine_counts = []

                    for i, run_cosine_sim in enumerate(
                        self.report[group]["duration_depth_1_p90_cosine_sim"]
                    ):
                        run_counts = self.report[group][
                            "duration_depth_1_p90_count_runs"
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
                        self.report[group]["duration_depth_1_p90_cosine_sim_avg"] = (
                            total_weighted_cosine_sim / total_count
                        )
                    else:
                        self.report[group]["duration_depth_1_p90_cosine_sim_avg"] = (
                            float("nan")
                        )
                else:
                    self.report[group]["duration_depth_1_p90_cosine_sim_avg"] = float(
                        "nan"
                    )
                del self.report[group]["duration_depth_1_p90_cosine_sim"]

            if "duration_depth_1_p50_cosine_sim" in self.report[group]:
                if self.report[group]["duration_depth_1_p50_cosine_sim"]:
                    # Calculate weighted average cosine similarity across all runs and all services
                    all_cosine_sims = []
                    all_cosine_counts = []

                    for i, run_cosine_sim in enumerate(
                        self.report[group]["duration_depth_1_p50_cosine_sim"]
                    ):
                        run_counts = self.report[group][
                            "duration_depth_1_p50_count_runs"
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
                        self.report[group]["duration_depth_1_p50_cosine_sim_avg"] = (
                            total_weighted_cosine_sim / total_count
                        )
                    else:
                        self.report[group]["duration_depth_1_p50_cosine_sim_avg"] = (
                            float("nan")
                        )
                else:
                    self.report[group]["duration_depth_1_p50_cosine_sim_avg"] = float(
                        "nan"
                    )
                del self.report[group]["duration_depth_1_p50_cosine_sim"]

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

        return dict(self.report)
