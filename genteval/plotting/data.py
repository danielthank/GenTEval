import json
from dataclasses import dataclass


@dataclass
class CostConfig:
    transmission_per_gb: float = 0.1
    gpu_per_hour: float = 0.38
    cpu_per_hour: float = 0.16
    span_count: int = 1845349
    time_duration_minutes: int = 35


@dataclass
class ExperimentData:
    name: str
    compressor_key: str
    experiment_name: str
    iteration: str
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
        size_data = self._extract_size_data(report_data, compressor_name)
        time_data = self._extract_time_data(report_data, compressor_name)

        name_parts = self._parse_compressor_name(compressor_name)
        if not name_parts:
            return None

        if not all([size_data, time_data]):
            print(f"Warning: Missing size or time data for {compressor_name}")
            return None

        # Calculate costs
        cost_data = self._calculate_costs(
            size_data["size_bytes"], time_data["cpu_seconds"], time_data["gpu_seconds"]
        )

        return ExperimentData(
            name=name_parts["display_name"],
            compressor_key=compressor_name,
            experiment_name=name_parts["experiment_name"],
            iteration=name_parts["iteration"],
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

    def _parse_compressor_name(self, compressor_name: str) -> dict | None:
        parts = compressor_name.split("_")

        # Find "head" or "gent" in parts (may have prefix like "otel-demo-transformed")
        # Format: head_{sampling_rate}_{iteration} -> display as "1:{rate}"
        if "head" in parts:
            head_idx = parts.index("head")
            if head_idx + 2 < len(parts):
                sampling_rate = parts[head_idx + 1]
                iteration = parts[head_idx + 2]
                return {
                    "display_name": f"1:{sampling_rate}",
                    "experiment_name": sampling_rate,
                    "iteration": iteration,
                    "is_head_sampling": True,
                }

        if "gent" in parts:
            gent_idx = parts.index("gent")
            # Expect: gent_{experiment_name}_{iteration}
            if gent_idx + 2 < len(parts):
                experiment_name = parts[gent_idx + 1]
                iteration = parts[gent_idx + 2]
                return {
                    "display_name": f"GenT {experiment_name}",
                    "experiment_name": experiment_name,
                    "iteration": iteration,
                    "is_head_sampling": False,
                }

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
        if "_head_" in compressor_name:
            return {"cpu_seconds": 0, "gpu_seconds": 0}

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
