import json
from dataclasses import dataclass


@dataclass
class CostConfig:
    transmission_per_gb: float = 0.1
    gpu_per_hour: float = 0.38
    cpu_per_hour: float = 0.16
    span_count: int = 1845349
    time_duration_minutes: int = 48 * 30 * 24 * 60


@dataclass
class ExperimentData:
    name: str
    compressor_key: str
    compute_type: str
    duration: str
    mape_fidelity: float
    cos_fidelity: float
    mape_fidelity_by_status_code: float
    cos_fidelity_by_status_code: float
    operation_f1_fidelity: float
    operation_pair_f1_fidelity: float
    child_parent_ratio_fidelity: float
    child_parent_overall_fidelity: float
    child_parent_depth1_fidelity: float
    child_parent_depth2_fidelity: float
    child_parent_depth3_fidelity: float
    child_parent_depth4_fidelity: float
    tracerca_avg5_fidelity: float
    microrank_avg5_fidelity: float
    count_over_time_mape_fidelity: float
    count_over_time_cosine_fidelity: float
    graph_fidelity: float
    size_kb: float
    cpu_time_seconds: float
    gpu_time_seconds: float
    span_count: int
    time_duration_minutes: int
    transmission_cost: float
    compute_cost: float
    total_cost: float
    transmission_cost_per_million_spans: float
    compute_cost_per_million_spans: float
    total_cost_per_million_spans: float
    cost_per_minute: float
    is_head_sampling: bool = False


class ReportParser:
    def __init__(self, cost_config: CostConfig | None = None):
        self.cost_config = cost_config or CostConfig()

    def parse_report(self, report_path: str) -> list[ExperimentData]:
        with open(report_path) as f:
            report_data = json.load(f)

        experiments = []

        # Extract compressor names from any report section
        compressor_names = self._get_compressor_names(report_data)

        for compressor_name in compressor_names:
            try:
                experiment = self._parse_compressor(report_data, compressor_name)
                if experiment:
                    experiments.append(experiment)
            except (KeyError, ValueError, TypeError) as e:
                print(f"Warning: Failed to parse {compressor_name}: {e}")

        return experiments

    def _get_compressor_names(self, report_data: dict) -> list[str]:
        compressor_names = set()

        # Get from reports section
        if "reports" in report_data:
            for report_content in report_data["reports"].values():
                if "compressors" in report_content:
                    compressor_names.update(report_content["compressors"])
                elif isinstance(report_content, dict):
                    compressor_names.update(report_content.keys())

        return list(compressor_names)

    def _parse_compressor(
        self, report_data: dict, compressor_name: str
    ) -> ExperimentData | None:
        # Extract data from different report sections first (needed for compute type detection)
        size_data = self._extract_size_data(report_data, compressor_name)
        time_data = self._extract_time_data(report_data, compressor_name)
        fidelity_data = self._extract_fidelity_data(report_data, compressor_name)
        fidelity_data_by_status_code = self._extract_fidelity_data_by_status_code(
            report_data, compressor_name
        )
        operation_data = self._extract_operation_data(report_data, compressor_name)
        child_parent_ratio_data = self._extract_child_parent_ratio_data(
            report_data, compressor_name
        )
        rca_data = self._extract_rca_data(report_data, compressor_name)
        count_over_time_data = self._extract_count_over_time_data(
            report_data, compressor_name
        )
        graph_data = self._extract_graph_data(report_data, compressor_name)

        # Parse compressor name to extract metadata (pass time_data for compute type detection)
        name_parts = self._parse_compressor_name(compressor_name, time_data)
        if not name_parts:
            return None

        if not all([size_data, time_data, fidelity_data]):
            print(f"Warning: Missing data for {compressor_name}")
            return None

        if not operation_data:
            operation_data = {"operation_f1": 0.0, "operation_pair_f1": 0.0}

        if not child_parent_ratio_data:
            child_parent_ratio_data = {
                "child_parent_ratio_wdist": 0.0,
                "child_parent_overall_wdist": 0.0,
                "child_parent_depth1_wdist": 0.0,
                "child_parent_depth2_wdist": 0.0,
                "child_parent_depth3_wdist": 0.0,
                "child_parent_depth4_wdist": 0.0,
            }

        if not rca_data:
            rca_data = {
                "tracerca_avg5": 0.0,
                "microrank_avg5": 0.0,
            }

        if not count_over_time_data:
            count_over_time_data = {
                "count_over_time_mape_fidelity": 0.0,
                "count_over_time_cosine_fidelity": 0.0,
            }

        if not graph_data:
            graph_data = {"graph_fidelity": 0.0}

        if not fidelity_data_by_status_code:
            fidelity_data_by_status_code = {
                "mape": 0.0,
                "cosine": 0.0,
            }

        # Calculate costs
        cost_data = self._calculate_costs(
            size_data["size_bytes"], time_data["cpu_seconds"], time_data["gpu_seconds"]
        )

        return ExperimentData(
            name=name_parts["display_name"],
            compressor_key=compressor_name,
            compute_type=name_parts["compute_type"],
            duration=name_parts["duration"],
            mape_fidelity=fidelity_data["mape"],
            cos_fidelity=fidelity_data["cosine"],
            mape_fidelity_by_status_code=fidelity_data_by_status_code["mape"],
            cos_fidelity_by_status_code=fidelity_data_by_status_code["cosine"],
            operation_f1_fidelity=operation_data["operation_f1"] * 100,
            operation_pair_f1_fidelity=operation_data["operation_pair_f1"] * 100,
            child_parent_ratio_fidelity=max(
                0, 100 - child_parent_ratio_data["child_parent_ratio_wdist"] * 1000
            ),
            child_parent_overall_fidelity=max(
                0, 100 - child_parent_ratio_data["child_parent_overall_wdist"] * 1000
            ),
            child_parent_depth1_fidelity=max(
                0, 100 - child_parent_ratio_data["child_parent_depth1_wdist"] * 1000
            ),
            child_parent_depth2_fidelity=max(
                0, 100 - child_parent_ratio_data["child_parent_depth2_wdist"] * 1000
            ),
            child_parent_depth3_fidelity=max(
                0, 100 - child_parent_ratio_data["child_parent_depth3_wdist"] * 1000
            ),
            child_parent_depth4_fidelity=max(
                0, 100 - child_parent_ratio_data["child_parent_depth4_wdist"] * 1000
            ),
            tracerca_avg5_fidelity=rca_data["tracerca_avg5"] * 100,
            microrank_avg5_fidelity=rca_data["microrank_avg5"] * 100,
            count_over_time_mape_fidelity=count_over_time_data[
                "count_over_time_mape_fidelity"
            ],
            count_over_time_cosine_fidelity=count_over_time_data[
                "count_over_time_cosine_fidelity"
            ],
            graph_fidelity=graph_data["graph_fidelity"],
            size_kb=size_data["size_bytes"] / 1024,
            cpu_time_seconds=time_data["cpu_seconds"],
            gpu_time_seconds=time_data["gpu_seconds"],
            span_count=self.cost_config.span_count,
            time_duration_minutes=self.cost_config.time_duration_minutes,
            transmission_cost=cost_data["transmission_cost"],
            compute_cost=cost_data["compute_cost"],
            total_cost=cost_data["total_cost"],
            transmission_cost_per_million_spans=cost_data[
                "transmission_cost_per_million"
            ],
            compute_cost_per_million_spans=cost_data["compute_cost_per_million"],
            total_cost_per_million_spans=cost_data["total_cost_per_million"],
            cost_per_minute=cost_data["cost_per_minute"],
            is_head_sampling=name_parts["is_head_sampling"],
        )

    def _parse_compressor_name(
        self, compressor_name: str, time_data: dict | None = None
    ) -> dict | None:
        if "head_sampling" in compressor_name:
            ratio = compressor_name.split("_")[-1]
            return {
                "display_name": f"1:{ratio}",
                "compute_type": "",
                "duration": "",
                "is_head_sampling": True,
            }

        if "gent" in compressor_name:
            parts = compressor_name.split("_")
            try:
                duration_idx = parts.index("gent") + 1
                if duration_idx < len(parts):
                    duration_num = parts[duration_idx]
                    duration = f"{duration_num}min"

                    # Determine compute type from time data if available
                    compute_type = "CPU"  # Default
                    if time_data:
                        gpu_seconds = time_data.get("gpu_seconds", 0)
                        cpu_seconds = time_data.get("cpu_seconds", 0)
                        if gpu_seconds > 0:
                            compute_type = "GPU"
                        elif cpu_seconds > 0:
                            compute_type = "CPU"

                    # Also check name suffix if time data doesn't help
                    last_part = parts[-1].lower()
                    if last_part in ["cpu", "gpu"]:
                        compute_type = last_part.upper()

                    return {
                        "display_name": f"GenT {duration}",
                        "compute_type": compute_type,
                        "duration": duration,
                        "is_head_sampling": False,
                    }
            except (ValueError, IndexError):
                pass

        return None

    def _extract_size_data(
        self, report_data: dict, compressor_name: str
    ) -> dict | None:
        try:
            size_report = report_data["reports"]["size"]
            if compressor_name in size_report:
                size_bytes = size_report[compressor_name]["size"]["avg"]
                return {"size_bytes": size_bytes}
        except KeyError:
            pass

        return None

    def _extract_time_data(
        self, report_data: dict, compressor_name: str
    ) -> dict | None:
        try:
            time_report = report_data["reports"]["time"]
            if compressor_name in time_report:
                data = time_report[compressor_name]
                cpu_seconds = data.get("compression_time_cpu_seconds", {}).get("avg", 0)
                gpu_seconds = data.get("compression_time_gpu_seconds", {}).get("avg", 0)
                return {"cpu_seconds": cpu_seconds, "gpu_seconds": gpu_seconds}
        except KeyError:
            pass

        # For head sampling, no computation time (0 for both CPU and GPU)
        if "head_sampling" in compressor_name:
            return {"cpu_seconds": 0, "gpu_seconds": 0}

        return None

    def _extract_fidelity_data_by_status_code(
        self, report_data: dict, compressor_name: str
    ) -> dict | None:
        try:
            # Extract from metadata.fidelity_scores
            if (
                "metadata" in report_data
                and "fidelity_scores" in report_data["metadata"]
            ):
                fidelity_scores = report_data["metadata"]["fidelity_scores"]

                mape_by_status_code = fidelity_scores.get(
                    "mape_fidelity_scores_by_status_code", {}
                ).get(compressor_name, 0)
                cosine_by_status_code = fidelity_scores.get(
                    "cosine_similarity_fidelity_scores_by_status_code", {}
                ).get(compressor_name, 0)

                return {"mape": mape_by_status_code, "cosine": cosine_by_status_code}
        except KeyError:
            pass

        return None

    def _extract_fidelity_data(
        self, report_data: dict, compressor_name: str
    ) -> dict | None:
        try:
            # Extract from metadata.fidelity_scores (new location after enhanced_report fix)
            if (
                "metadata" in report_data
                and "fidelity_scores" in report_data["metadata"]
            ):
                fidelity_scores = report_data["metadata"]["fidelity_scores"]

                mape = fidelity_scores.get("mape_fidelity_scores", {}).get(
                    compressor_name, 0
                )
                cosine = fidelity_scores.get(
                    "cosine_similarity_fidelity_scores", {}
                ).get(compressor_name, 0)

                return {"mape": mape, "cosine": cosine}

            # Fallback: Try legacy locations in duration report (for backward compatibility)
            duration_report = report_data["reports"]["duration"]

            # Try separate mape and cosine fields in duration report (legacy)
            mape = 0
            cosine = 0

            if "mape_fidelity_scores" in duration_report:
                mape = duration_report["mape_fidelity_scores"].get(compressor_name, 0)

            if "cosine_similarity_fidelity_scores" in duration_report:
                cosine = duration_report["cosine_similarity_fidelity_scores"].get(
                    compressor_name, 0
                )

        except KeyError:
            return None
        else:
            return {"mape": mape, "cosine": cosine}

    def _extract_operation_data(
        self, report_data: dict, compressor_name: str
    ) -> dict | None:
        try:
            operation_report = report_data["reports"]["operation"]
            if compressor_name in operation_report:
                operation_f1 = (
                    operation_report[compressor_name]
                    .get("operation_f1", {})
                    .get("avg", 0.0)
                )
                operation_pair_f1 = (
                    operation_report[compressor_name]
                    .get("operation_pair_f1", {})
                    .get("avg", 0.0)
                )
                return {
                    "operation_f1": operation_f1,
                    "operation_pair_f1": operation_pair_f1,
                }
        except KeyError:
            pass

        return None

    def _extract_child_parent_ratio_data(
        self, report_data: dict, compressor_name: str
    ) -> dict | None:
        try:
            duration_report = report_data["reports"]["duration"]
            if compressor_name in duration_report:
                result = {}

                # Calculate average child/parent ratio W-dist across all depths
                wdist_values = []
                for depth in range(5):  # depths 0-4
                    key = f"pair_depth_{depth}_wdist"
                    if key in duration_report[compressor_name]:
                        wdist_values.append(
                            duration_report[compressor_name][key]["avg"]
                        )

                if wdist_values:
                    avg_wdist = sum(wdist_values) / len(wdist_values)
                    result["child_parent_ratio_wdist"] = avg_wdist

                # Get individual depth W-dist values
                for depth in range(1, 5):  # depths 1-4
                    key = f"pair_depth_{depth}_wdist"
                    if key in duration_report[compressor_name]:
                        result[f"child_parent_depth{depth}_wdist"] = duration_report[
                            compressor_name
                        ][key]["avg"]

                # Get overall child/parent ratio W-dist from pair_all_wdist
                if "pair_all_wdist" in duration_report[compressor_name]:
                    result["child_parent_overall_wdist"] = duration_report[
                        compressor_name
                    ]["pair_all_wdist"]["avg"]

                # Return result if we have either averaged or overall data
                if result:
                    return result

                # Fallback: use pair_all_wdist for all if depth-specific data not available
                if "pair_all_wdist" in duration_report[compressor_name]:
                    pair_all_wdist = duration_report[compressor_name]["pair_all_wdist"][
                        "avg"
                    ]
                    return {
                        "child_parent_ratio_wdist": pair_all_wdist,
                        "child_parent_overall_wdist": pair_all_wdist,
                        "child_parent_depth1_wdist": pair_all_wdist,
                        "child_parent_depth2_wdist": pair_all_wdist,
                        "child_parent_depth3_wdist": pair_all_wdist,
                        "child_parent_depth4_wdist": pair_all_wdist,
                    }

        except KeyError:
            pass

        return None

    def _calculate_costs(
        self, size_bytes: float, cpu_seconds: float, gpu_seconds: float
    ) -> dict:
        # Size conversion
        size_gb = size_bytes / (1024**3)
        transmission_cost = size_gb * self.cost_config.transmission_per_gb

        # Time conversion
        cpu_cost = (cpu_seconds / 3600) * self.cost_config.cpu_per_hour
        gpu_cost = (gpu_seconds / 3600) * self.cost_config.gpu_per_hour
        compute_cost = cpu_cost + gpu_cost

        total_cost = transmission_cost + compute_cost

        # Per million spans
        spans_in_millions = self.cost_config.span_count / 1_000_000
        transmission_cost_per_million = transmission_cost / spans_in_millions
        compute_cost_per_million = compute_cost / spans_in_millions
        total_cost_per_million = total_cost / spans_in_millions

        # Per minute
        cost_per_minute = total_cost / self.cost_config.time_duration_minutes

        return {
            "transmission_cost": transmission_cost,
            "compute_cost": compute_cost,
            "total_cost": total_cost,
            "transmission_cost_per_million": transmission_cost_per_million,
            "compute_cost_per_million": compute_cost_per_million,
            "total_cost_per_million": total_cost_per_million,
            "cost_per_minute": cost_per_minute,
        }

    def _extract_rca_data(self, report_data: dict, compressor_name: str) -> dict | None:
        """Extract TraceRCA and MicroRank avg5 data from the JSON report."""
        try:
            result = {}

            # Extract TraceRCA avg5
            if "trace_rca" in report_data.get("reports", {}):
                trace_rca_report = report_data["reports"]["trace_rca"]
                if compressor_name in trace_rca_report:
                    avg5_data = trace_rca_report[compressor_name].get("avg5", {})
                    if "avg" in avg5_data:
                        result["tracerca_avg5"] = avg5_data["avg"]
                    elif "values" in avg5_data:
                        values = avg5_data["values"]
                        result["tracerca_avg5"] = (
                            sum(values) / len(values) if values else 0.0
                        )
                    else:
                        result["tracerca_avg5"] = 0.0
                else:
                    result["tracerca_avg5"] = 0.0
            else:
                result["tracerca_avg5"] = 0.0

            # Extract MicroRank avg5
            if "micro_rank" in report_data.get("reports", {}):
                micro_rank_report = report_data["reports"]["micro_rank"]
                if compressor_name in micro_rank_report:
                    avg5_data = micro_rank_report[compressor_name].get("avg5", {})
                    if "avg" in avg5_data:
                        result["microrank_avg5"] = avg5_data["avg"]
                    elif "values" in avg5_data:
                        values = avg5_data["values"]
                        result["microrank_avg5"] = (
                            sum(values) / len(values) if values else 0.0
                        )
                    else:
                        result["microrank_avg5"] = 0.0
                else:
                    result["microrank_avg5"] = 0.0
            else:
                result["microrank_avg5"] = 0.0

        except KeyError:
            return None
        else:
            return result

    def _extract_count_over_time_data(
        self, report_data: dict, compressor_name: str
    ) -> dict | None:
        """Extract count over time fidelity data from the JSON report."""
        try:
            result = {}

            # Extract count over time data
            if "count_over_time" in report_data.get("reports", {}):
                count_over_time_report = report_data["reports"]["count_over_time"]
                if compressor_name in count_over_time_report:
                    compressor_data = count_over_time_report[compressor_name]

                    # Extract MAPE fidelity score
                    result["count_over_time_mape_fidelity"] = compressor_data.get(
                        "count_over_time_mape_fidelity_score", 0.0
                    )

                    # Extract Cosine fidelity score
                    result["count_over_time_cosine_fidelity"] = compressor_data.get(
                        "count_over_time_cosine_fidelity_score", 0.0
                    )
                else:
                    result["count_over_time_mape_fidelity"] = 0.0
                    result["count_over_time_cosine_fidelity"] = 0.0
            else:
                result["count_over_time_mape_fidelity"] = 0.0
                result["count_over_time_cosine_fidelity"] = 0.0

        except KeyError:
            return None
        else:
            return result

    def _extract_graph_data(
        self, report_data: dict, compressor_name: str
    ) -> dict | None:
        """Extract graph fidelity data from the JSON report by averaging time bucket fidelities."""
        try:
            result = {}

            # Extract from graph report
            if "graph" in report_data.get("reports", {}):
                graph_report = report_data["reports"]["graph"]
                if compressor_name in graph_report:
                    compressor_data = graph_report[compressor_name]

                    # Collect fidelity scores from all time buckets
                    fidelity_scores = []
                    for key, value in compressor_data.items():
                        if key.startswith("time_") and isinstance(value, dict):
                            if "fidelity" in value:
                                fidelity_scores.append(value["fidelity"])

                    # Calculate average fidelity
                    if fidelity_scores:
                        result["graph_fidelity"] = sum(fidelity_scores) / len(
                            fidelity_scores
                        )
                    else:
                        result["graph_fidelity"] = 0.0
                else:
                    result["graph_fidelity"] = 0.0
            else:
                result["graph_fidelity"] = 0.0

        except KeyError:
            return None
        else:
            return result
