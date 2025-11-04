#!/usr/bin/env python3
"""
CLI tool to compare two graph_results.json files, export diagrams, and calculate graph edit distance.
"""

import argparse
import json
import pathlib
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch


def load_graph_json(json_path: pathlib.Path) -> dict:
    """Load graph_results.json file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def json_to_networkx(graph_data: dict) -> nx.DiGraph:
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


def calculate_distance(G1: nx.DiGraph, G2: nx.DiGraph) -> float:
    """
    Calculate graph edit distance with custom cost functions.

    Args:
        G1: First graph
        G2: Second graph

    Returns:
        Approximate graph edit distance
    """
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


def calculate_graph_fidelity(distance: float, num_nodes: int, total_edge_weight: float) -> float:
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


def export_graph_diagram(
    G: nx.DiGraph,
    output_path: pathlib.Path,
    title: str,
    highlight_edges: dict = None
):
    """
    Export graph diagram as PNG using matplotlib and networkx.

    Args:
        G: NetworkX graph to visualize
        output_path: Path to save the PNG file
        title: Title for the diagram
        highlight_edges: Dictionary of edges to highlight with different colors
    """
    plt.figure(figsize=(12, 8))

    # Use hierarchical layout for directed graphs
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        # Fallback to shell layout if spring layout fails
        pos = nx.shell_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=3000,
        alpha=0.9,
        edgecolors='black',
        linewidths=2
    )

    # Draw edges with weights
    edges = G.edges()
    weights = [G[u][v].get('weight', 1) for u, v in edges]

    # Normalize weights for edge width
    max_weight = max(weights) if weights else 1
    edge_widths = [2 + (w / max_weight) * 3 for w in weights]

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        width=edge_widths,
        alpha=0.6,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='->'
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=9,
        font_weight='bold',
        font_family='sans-serif'
    )

    # Draw edge labels (weights)
    edge_labels = {(u, v): f"{G[u][v].get('weight', 1)}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels,
        font_size=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Exported diagram: {output_path}")


def compare_graphs(
    graph1_path: pathlib.Path,
    graph2_path: pathlib.Path,
    timestamp: str,
    output_dir: pathlib.Path
) -> Tuple[float, float]:
    """
    Compare two graphs and export diagrams.

    Args:
        graph1_path: Path to first graph_results.json
        graph2_path: Path to second graph_results.json
        timestamp: Time bucket to compare
        output_dir: Directory to export diagrams

    Returns:
        Tuple of (graph_edit_distance, fidelity_score)
    """
    # Load graphs
    graph1_data = load_graph_json(graph1_path)
    graph2_data = load_graph_json(graph2_path)

    # Get graphs for the specified timestamp
    graphs1 = graph1_data.get("service_graph_by_time", {})
    graphs2 = graph2_data.get("service_graph_by_time", {})

    if timestamp not in graphs1:
        raise ValueError(f"Timestamp {timestamp} not found in {graph1_path}")
    if timestamp not in graphs2:
        raise ValueError(f"Timestamp {timestamp} not found in {graph2_path}")

    # Convert to NetworkX
    G1 = json_to_networkx(graphs1[timestamp])
    G2 = json_to_networkx(graphs2[timestamp])

    # Calculate distance and fidelity
    distance = calculate_distance(G1, G2)

    num_nodes = G1.number_of_nodes()
    total_edge_weight = sum(
        data.get('weight', 0)
        for _, _, data in G1.edges(data=True)
    )
    fidelity = calculate_graph_fidelity(distance, num_nodes, total_edge_weight)

    # Export diagrams
    output_dir.mkdir(parents=True, exist_ok=True)

    graph1_name = graph1_path.parent.parent.name
    graph2_name = graph2_path.parent.parent.name

    export_graph_diagram(
        G1,
        output_dir / f"graph1_{graph1_name}_{timestamp}.png",
        f"Graph 1: {graph1_name} (Time: {timestamp})"
    )

    export_graph_diagram(
        G2,
        output_dir / f"graph2_{graph2_name}_{timestamp}.png",
        f"Graph 2: {graph2_name} (Time: {timestamp})"
    )

    return distance, fidelity


def print_graph_statistics(G: nx.DiGraph, name: str):
    """Print graph statistics."""
    print(f"\n{name}:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    if G.number_of_edges() > 0:
        total_weight = sum(data.get('weight', 0) for _, _, data in G.edges(data=True))
        avg_weight = total_weight / G.number_of_edges()
        print(f"  Total edge weight: {total_weight}")
        print(f"  Average edge weight: {avg_weight:.2f}")

    if G.number_of_nodes() > 0:
        print(f"  Nodes: {', '.join(sorted(G.nodes()))}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two graph_results.json files and calculate graph edit distance"
    )
    parser.add_argument(
        "graph1",
        type=str,
        help="Path to first graph_results.json file"
    )
    parser.add_argument(
        "graph2",
        type=str,
        help="Path to second graph_results.json file"
    )
    parser.add_argument(
        "timestamp",
        type=str,
        help="Timestamp (time bucket) to compare"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="graph_comparison_output",
        help="Output directory for graph diagrams (default: graph_comparison_output)"
    )
    parser.add_argument(
        "--list-timestamps",
        "-l",
        action="store_true",
        help="List available timestamps in both graphs and exit"
    )

    args = parser.parse_args()

    graph1_path = pathlib.Path(args.graph1)
    graph2_path = pathlib.Path(args.graph2)
    output_dir = pathlib.Path(args.output_dir)

    # Validate inputs
    if not graph1_path.exists():
        print(f"Error: {graph1_path} does not exist")
        return 1

    if not graph2_path.exists():
        print(f"Error: {graph2_path} does not exist")
        return 1

    # List timestamps if requested
    if args.list_timestamps:
        graph1_data = load_graph_json(graph1_path)
        graph2_data = load_graph_json(graph2_path)

        timestamps1 = set(graph1_data.get("service_graph_by_time", {}).keys())
        timestamps2 = set(graph2_data.get("service_graph_by_time", {}).keys())

        print(f"\nTimestamps in {graph1_path}:")
        for ts in sorted(timestamps1):
            print(f"  {ts}")

        print(f"\nTimestamps in {graph2_path}:")
        for ts in sorted(timestamps2):
            print(f"  {ts}")

        common_timestamps = timestamps1 & timestamps2
        print(f"\nCommon timestamps: {len(common_timestamps)}")
        if common_timestamps:
            print("  " + ", ".join(sorted(common_timestamps)))

        return 0

    # Perform comparison
    print(f"\n{'='*70}")
    print(f"Graph Comparison Tool")
    print(f"{'='*70}")
    print(f"\nGraph 1: {graph1_path}")
    print(f"Graph 2: {graph2_path}")
    print(f"Timestamp: {args.timestamp}")
    print(f"Output directory: {output_dir}")

    try:
        # Load and display graph statistics
        graph1_data = load_graph_json(graph1_path)
        graph2_data = load_graph_json(graph2_path)

        graphs1 = graph1_data.get("service_graph_by_time", {})
        graphs2 = graph2_data.get("service_graph_by_time", {})

        G1 = json_to_networkx(graphs1[args.timestamp])
        G2 = json_to_networkx(graphs2[args.timestamp])

        print_graph_statistics(G1, "Graph 1 Statistics")
        print_graph_statistics(G2, "Graph 2 Statistics")

        # Compare and export
        print(f"\n{'='*70}")
        print("Comparing graphs and exporting diagrams...")
        print(f"{'='*70}")

        distance, fidelity = compare_graphs(
            graph1_path,
            graph2_path,
            args.timestamp,
            output_dir
        )

        # Display results
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"\nGraph Edit Distance: {distance:.2f}")
        print(f"Graph Fidelity Score: {fidelity:.2f}%")
        print(f"\nDiagrams exported to: {output_dir.absolute()}")
        print(f"{'='*70}\n")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
