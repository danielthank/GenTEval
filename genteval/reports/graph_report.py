from typing import Any

import networkx as nx

from genteval.bin.utils import get_dir_with_root

from .base_report import BaseReport


class GraphReport(BaseReport):
    """
    Report that calculates graph edit distance between ground truth and compressed service graphs.
    """

    def calculate_distance(self, G1: nx.DiGraph, G2: nx.DiGraph) -> float:
        """
        Calculate graph edit distance with custom cost functions.
        Modify the cost functions here to change distance calculation.

        Uses optimize_graph_edit_distance for approximate but much faster results.

        Args:
            G1: Ground truth graph
            G2: Compressed graph

        Returns:
            Approximate graph edit distance
        """
        # Use optimize_graph_edit_distance for speed (returns a generator)
        # Take the first (best) approximation
        distance_generator = nx.optimize_graph_edit_distance(
            G1,
            G2,
            node_subst_cost=lambda n1, n2: 0 if n1 == n2 else 1,
            node_del_cost=lambda n: 1,
            node_ins_cost=lambda n: 1,
            edge_subst_cost=lambda e1, e2: abs(
                e1.get("weight", 0) - e2.get("weight", 0)
            ),
            edge_del_cost=lambda e: e.get("weight", 1),
            edge_ins_cost=lambda e: e.get("weight", 1),
        )
        return next(distance_generator)

    def calculate_graph_fidelity(
        self, distance: float, num_nodes: int, total_edge_weight: float
    ) -> float:
        """
        Calculate fidelity score from graph edit distance.

        Args:
            distance: Graph edit distance
            num_nodes: Number of nodes in reference graph
            total_edge_weight: Sum of all edge weights in reference graph

        Returns:
            Fidelity score (0-100), where 100 is perfect match
        """
        reference_size = num_nodes + total_edge_weight
        if reference_size == 0:
            return 100.0

        # Distance as percentage of graph size
        distance_ratio = distance / reference_size

        # Invert to get fidelity (lower distance = higher fidelity)
        fidelity = max(0.0, 100.0 - distance_ratio * 100)

        return fidelity

    def json_to_networkx(self, graph_data: dict) -> nx.DiGraph:
        """
        Convert JSON graph representation to NetworkX DiGraph.

        Args:
            graph_data: Dictionary with "nodes" and "edges" keys

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()

        # Add nodes
        nodes = graph_data.get("nodes", [])
        G.add_nodes_from(nodes)

        # Add edges with weights
        edges = graph_data.get("edges", {})
        for edge_str, weight in edges.items():
            if "->" in edge_str:
                parent, child = edge_str.split("->")
                G.add_edge(parent, child, weight=weight)

        return G

    def calculate_compression_ratio(
        self, app_name: str, service: str, fault: str, run: int, compressor: str
    ) -> float:
        """
        Calculate compression ratio by parsing compressor name.

        head_sampling_N means 1 out of N spans is sampled, so compression ratio is N.

        Args:
            app_name: Application name
            service: Service name
            fault: Fault type
            run: Run number
            compressor: Compressor name (e.g., "head_sampling_50")

        Returns:
            Compression ratio (N from head_sampling_N, or 1.0 if not head_sampling)
        """
        import re

        # Parse head_sampling_N pattern
        match = re.search(r'head_sampling_(\d+(?:\.\d+)?)', compressor)
        if match:
            ratio = float(match.group(1))
            return ratio

        # Default to 1.0 (no compression)
        return 1.0

    def scale_edge_weights(self, graph_data: dict, scale_factor: float) -> dict:
        """
        Scale edge weights by a factor (for head_sampling compressors).

        Args:
            graph_data: Dictionary with "nodes" and "edges" keys
            scale_factor: Factor to multiply edge weights by

        Returns:
            New graph data with scaled edge weights
        """
        scaled_data = {
            "nodes": graph_data.get("nodes", []),
            "edges": {},
        }

        for edge_str, weight in graph_data.get("edges", {}).items():
            scaled_data["edges"][edge_str] = weight * scale_factor

        return scaled_data

    def generate(self, run_dirs) -> dict[str, Any]:
        """
        Generate graph edit distance report.

        Args:
            run_dirs: Function that returns iterator over (app_name, service, fault, run)

        Returns:
            Report dictionary with structure: {app_compressor: {time_bucket: {"avg": distance}}}
        """
        run_count = 0

        for app_name, service, fault, run in run_dirs():
            run_count += 1

            for compressor in self.compressors:
                # Skip unsupported compressors
                if compressor in {"original", "head_sampling_1"}:
                    continue

                # Load ground truth
                original_results_path = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / "head_sampling_1"
                    / "evaluated"
                    / "graph_results.json"
                )

                if not self.file_exists(original_results_path):
                    self.print_skip_message(
                        f"Original results {original_results_path} not found, skipping."
                    )
                    continue

                # Load compressed
                results_path = (
                    get_dir_with_root(self.root_dir, app_name, service, fault, run)
                    / compressor
                    / "evaluated"
                    / "graph_results.json"
                )

                if not self.file_exists(results_path):
                    self.print_skip_message(
                        f"Compressed results {results_path} not found, skipping."
                    )
                    continue

                original = self.load_json_file(original_results_path)
                results = self.load_json_file(results_path)

                # Calculate compression ratio for head_sampling
                compression_ratio = 1.0
                if compressor.startswith("head_sampling"):
                    compression_ratio = self.calculate_compression_ratio(
                        app_name, service, fault, run, compressor
                    )

                report_group = f"{app_name}_{compressor}"

                # Process each time bucket
                original_graphs = original.get("service_graph_by_time", {})
                compressed_graphs = results.get("service_graph_by_time", {})

                for idx, (time_bucket, original_graph_data) in enumerate(
                    original_graphs.items(), 1
                ):
                    if time_bucket not in compressed_graphs:
                        continue

                    compressed_graph_data = compressed_graphs[time_bucket]

                    # Scale edge weights for head_sampling
                    if compressor.startswith("head_sampling"):
                        compressed_graph_data = self.scale_edge_weights(
                            compressed_graph_data, compression_ratio
                        )

                    # Convert to NetworkX graphs
                    G_original = self.json_to_networkx(original_graph_data)
                    G_compressed = self.json_to_networkx(compressed_graph_data)

                    # Calculate distance
                    distance = self.calculate_distance(G_original, G_compressed)

                    # Calculate fidelity
                    num_nodes = G_original.number_of_nodes()
                    total_edge_weight_g1 = sum(
                        data.get('weight', 0)
                        for _, _, data in G_original.edges(data=True)
                    )
                    total_edge_weight_g2 = sum(
                        data.get('weight', 0)
                        for _, _, data in G_compressed.edges(data=True)
                    )
                    total_edge_weight = total_edge_weight_g1 + total_edge_weight_g2
                    fidelity = self.calculate_graph_fidelity(distance, num_nodes, total_edge_weight)

                    # Store in report with time_bucket key
                    metric_key = f"time_{time_bucket}"
                    self.report[report_group][metric_key]["values"].append(distance)
                    self.report[report_group][metric_key]["fidelity_values"] = self.report[report_group][metric_key].get("fidelity_values", [])
                    self.report[report_group][metric_key]["fidelity_values"].append(fidelity)

        # Calculate averages and cleanup
        for report_group in self.report.values():
            for metric_group in report_group.values():
                if isinstance(metric_group, dict) and "values" in metric_group:
                    metric_group["avg"] = (
                        sum(metric_group["values"]) / len(metric_group["values"])
                        if metric_group["values"]
                        else float("nan")
                    )
                    del metric_group["values"]

                # Calculate fidelity average
                if isinstance(metric_group, dict) and "fidelity_values" in metric_group:
                    metric_group["fidelity"] = (
                        sum(metric_group["fidelity_values"]) / len(metric_group["fidelity_values"])
                        if metric_group["fidelity_values"]
                        else float("nan")
                    )
                    del metric_group["fidelity_values"]

        return dict(self.report)
