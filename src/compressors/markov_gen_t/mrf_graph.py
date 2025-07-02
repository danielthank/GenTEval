import logging
from collections import Counter, defaultdict
from typing import List

import networkx as nx
import numpy as np
from scipy.special import logsumexp

from compressors import CompressedDataset, SerializationFormat
from compressors.trace import Trace


class TreeNode:
    """Represents a node in the generated tree structure."""

    def __init__(self, node_name: str, depth: int, child_cnt: int = 0):
        self.node_name = node_name
        self.depth = depth
        self.child_cnt = child_cnt  # Expected number of children
        self.children = []
        self.parent = None

    def add_child(self, child):
        """Add a child node."""
        child.parent = self
        self.children.append(child)

    def get_child_count(self) -> int:
        """Get the number of children."""
        return len(self.children)

    def get_expected_child_count(self) -> int:
        """Get the expected number of children."""
        return self.child_cnt


class MarkovRandomField:
    """Markov Random Field for modeling graph structures with node attributes."""

    def __init__(
        self,
        order: int = 1,
        max_depth: int = 10,
        max_children: int = 5000,
        node_weight: float = 1.0,  # Lambda parameter for balancing edge vs node potentials
    ):
        self.order = order
        self.max_depth = max_depth
        self.max_children = max_children
        self.node_weight = node_weight  # This is your lambda
        self.logger = logging.getLogger(__name__)

        # Node features and relationships
        self.node_names = set()
        self.root_features = (
            Counter()
        )  # (node_name, child_cnt) -> count for root nodes only
        self.edge_features = Counter()  # (parent_node_name, parent_child_cnt, child_node_name, child_child_cnt, depth_diff) -> count

        # MRF parameters (learned via maximum likelihood)
        self.root_potentials = {}  # Unary potentials for root nodes
        self.edge_potentials = {}  # Pairwise potentials for edges
        self.node_potentials = {}  # Global node type potentials
        self.is_trained = False

    def _extract_graph_features(self, trace: Trace):
        """Extract node and edge features from trace and collect statistics directly."""
        # Build graph from trace
        graph = nx.DiGraph()
        root_spans = []

        for span_id, span_data in trace._spans.items():
            graph.add_node(span_id, **span_data)
            if span_data.get("parentSpanId") is None:
                root_spans.append(span_id)
            else:
                parent_id = span_data["parentSpanId"]
                if parent_id in trace._spans:
                    graph.add_edge(parent_id, span_id)

        # Extract features for each connected component
        for root in root_spans:
            self._extract_subgraph_features(graph, root, trace)

    def _extract_subgraph_features(self, graph: nx.DiGraph, root: str, trace: Trace):
        """Extract features from a subgraph rooted at given node and collect statistics."""
        # BFS to assign depths and extract node features
        queue = [(root, 0)]
        visited = {root}

        while queue:
            node_id, depth = queue.pop(0)

            try:
                node_name = trace.spans[node_id]["nodeName"]
                children = list(graph.successors(node_id))
                child_cnt = min(len(children), self.max_children)

                if depth == 0:
                    root_features = (node_name, child_cnt)
                    self.root_features[root_features] += 1
                self.node_names.add(node_name)

                for child in children:
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, depth + 1))

                        child_name = trace.spans[child]["nodeName"]
                        child_children = list(graph.successors(child))
                        child_child_cnt = min(len(child_children), self.max_children)
                        depth_diff = 1  # Always 1 for parent-child relationship

                        edge_features = (
                            node_name,
                            child_cnt,
                            child_name,
                            child_child_cnt,
                            depth_diff,
                        )

                        self.edge_features[edge_features] += 1

            except Exception as e:
                self.logger.warning(f"Error processing node {node_id}: {e}")
                continue

    def fit(self, traces: List[Trace]):
        """Learn MRF parameters from traces."""
        self.logger.info("Training Markov Random Field for graph structure modeling")

        # Extract features and collect statistics directly
        for trace in traces:
            self._extract_graph_features(trace)

        if not self.root_features:
            self.logger.warning("No valid graph features found for MRF training")
            return

        self._learn_potentials()
        self.is_trained = True

        self.logger.info(
            f"Trained MRF with {len(self.root_potentials)} root potential types and "
            f"{len(self.edge_potentials)} edge potential types, "
            f"{len(self.node_names)} unique node names"
        )

    def _learn_potentials(self):
        """Learn MRF potentials using maximum likelihood estimation."""
        # Compute root potentials (log probabilities) - only for root nodes
        total_roots = sum(self.root_features.values())
        for root_features, count in self.root_features.items():
            self.root_potentials[root_features] = np.log(count / total_roots)

        # Compute edge potentials (log conditional probabilities)
        parent_counts = defaultdict(int)
        for edge_features in self.edge_features:
            parent_key = (
                edge_features[0],
                edge_features[1],
            )  # (parent_node_name, parent_child_cnt)
            parent_counts[parent_key] += self.edge_features[edge_features]

        for edge_features, count in self.edge_features.items():
            parent_key = (
                edge_features[0],
                edge_features[1],
            )  # (parent_node_name, parent_child_cnt)
            parent_total = parent_counts[parent_key]
            if parent_total > 0:
                self.edge_potentials[edge_features] = np.log(count / parent_total)

        # Compute node potentials (global node type frequencies)
        node_counts = defaultdict(int)
        total_nodes = 0

        # Count all nodes from edge features (children)
        for edge_features, count in self.edge_features.items():
            child_name, child_cnt = edge_features[2], edge_features[3]
            node_key = (child_name, child_cnt)
            node_counts[node_key] += count
            total_nodes += count

        # Also count root nodes
        for root_features, count in self.root_features.items():
            node_name, child_cnt = root_features
            node_key = (node_name, child_cnt)
            node_counts[node_key] += count
            total_nodes += count

        # Compute node potentials (log probabilities)
        self.node_potentials = {}
        for node_key, count in node_counts.items():
            self.node_potentials[node_key] = np.log(count / total_nodes)

    def generate_tree_structure(self, max_nodes: int = 50) -> TreeNode:
        """Generate tree structure using MRF sampling."""
        if not self.is_trained:
            self.logger.warning("MRF not trained, cannot generate tree structure")
            return None

        return self._sample_tree_structure(max_nodes)

    def _sample_tree_structure(self, max_nodes: int) -> TreeNode:
        """Sample tree structure from MRF using Gibbs sampling approach."""
        # Sample root node based on node potentials
        root_features = self._sample_root_node()
        if root_features is None:
            return None

        node_name, child_cnt, depth = root_features
        root = TreeNode(node_name, depth, child_cnt)
        nodes_generated = 1

        # Build tree using recursive sampling
        self._sample_subtree(root, max_nodes, nodes_generated)

        return root

    def _sample_root_node(self):
        """Sample root node (depth 0) from learned distribution."""
        root_candidates = list(self.root_potentials.keys())

        if not root_candidates:
            return None

        # Sample based on root potentials
        potentials = [
            self.root_potentials.get(features, -np.inf) for features in root_candidates
        ]
        potentials = np.array(potentials)

        # Convert log potentials to probabilities
        potentials = potentials - logsumexp(potentials)
        probs = np.exp(potentials)

        try:
            idx = np.random.choice(len(root_candidates), p=probs)
            node_name, child_cnt = root_candidates[idx]
            return (node_name, child_cnt, 0)  # Add depth 0 for root
        except Exception:
            node_name, child_cnt = (
                root_candidates[0] if root_candidates else (None, None)
            )
            return (node_name, child_cnt, 0) if node_name is not None else None

    def _sample_subtree(
        self, parent_node: TreeNode, max_nodes: int, nodes_generated: int
    ) -> int:
        """Recursively sample subtree structure."""
        if nodes_generated >= max_nodes:
            return nodes_generated

        # Check if we've reached max_depth during generation
        if parent_node.depth >= self.max_depth:
            return nodes_generated

        parent_features = (
            parent_node.node_name,
            parent_node.child_cnt,
            parent_node.depth,
        )

        # Sample children based on edge potentials
        for _ in range(parent_node.get_expected_child_count()):
            if nodes_generated >= max_nodes:
                break

            child_features = self._sample_child_node(parent_features)
            if child_features is None:
                continue

            child_name, child_cnt, child_depth = child_features
            child_node = TreeNode(child_name, child_depth, child_cnt)
            parent_node.add_child(child_node)
            nodes_generated += 1

            # Recursively generate subtree
            nodes_generated = self._sample_subtree(
                child_node, max_nodes, nodes_generated
            )

        return nodes_generated

    def _sample_child_node(self, parent_features):
        """Sample child node given parent features."""
        parent_node_name, parent_child_cnt, parent_depth = parent_features

        # Find all possible child features with correct depth difference from edge potentials
        child_candidates = []
        for edge_features in self.edge_potentials.keys():
            p_node_name, p_child_cnt, c_node_name, c_child_cnt, depth_diff = (
                edge_features
            )
            if (
                p_node_name == parent_node_name
                and p_child_cnt == parent_child_cnt
                and depth_diff == 1
            ):  # depth_diff is always 1 for parent-child
                child_candidates.append((c_node_name, c_child_cnt, parent_depth + 1))

        if not child_candidates:
            return None

        # Sample based on BOTH edge potentials AND node potentials (lambda weighting)
        potentials = []
        for c_features in child_candidates:
            c_node_name, c_child_cnt, c_depth = c_features

            # Edge potential (conditional probability)
            edge_key = (parent_node_name, parent_child_cnt, c_node_name, c_child_cnt, 1)
            edge_potential = self.edge_potentials.get(edge_key, -np.inf)

            # Node potential (global probability)
            node_key = (c_node_name, c_child_cnt)
            node_potential = self.node_potentials.get(node_key, -np.inf)

            # Combine with lambda weighting (self.node_weight)
            combined_potential = edge_potential + self.node_weight * node_potential
            potentials.append(combined_potential)

        potentials = np.array(potentials)

        # Convert to probabilities
        if np.all(np.isinf(potentials)):
            # Uniform sampling as fallback
            probs = np.ones(len(child_candidates)) / len(child_candidates)
        else:
            potentials = potentials - logsumexp(potentials)
            probs = np.exp(potentials)

        try:
            idx = np.random.choice(len(child_candidates), p=probs)
            return child_candidates[idx]
        except Exception:
            return child_candidates[0] if child_candidates else None

    def save_state_dict(self, compressed_data: CompressedDataset):
        """Save MRF state to compressed dataset - only generation essentials."""
        compressed_data.add(
            "mrf_graph",
            CompressedDataset(
                data={
                    "root_potentials": (
                        dict(self.root_potentials),
                        SerializationFormat.MSGPACK,
                    ),
                    "edge_potentials": (
                        dict(self.edge_potentials),
                        SerializationFormat.MSGPACK,
                    ),
                    "node_potentials": (
                        dict(self.node_potentials),
                        SerializationFormat.MSGPACK,
                    ),
                    "node_names": (
                        list(self.node_names),
                        SerializationFormat.MSGPACK,
                    ),
                    "order": (self.order, SerializationFormat.MSGPACK),
                    "max_depth": (self.max_depth, SerializationFormat.MSGPACK),
                    "max_children": (self.max_children, SerializationFormat.MSGPACK),
                    "node_weight": (self.node_weight, SerializationFormat.MSGPACK),
                    "is_trained": (self.is_trained, SerializationFormat.MSGPACK),
                }
            ),
            SerializationFormat.CLOUDPICKLE,
        )

    def load_state_dict(self, compressed_dataset: CompressedDataset):
        """Load MRF state from compressed dataset."""
        if "mrf_graph" not in compressed_dataset:
            raise ValueError("No mrf_graph found in compressed dataset")
        mrf_data = compressed_dataset["mrf_graph"]

        # Load only the generation essentials
        self.root_potentials = dict(mrf_data["root_potentials"])
        self.edge_potentials = dict(mrf_data["edge_potentials"])
        self.node_potentials = dict(mrf_data["node_potentials"])
        self.node_names = set(mrf_data["node_names"])
        self.order = mrf_data["order"]
        self.max_depth = mrf_data["max_depth"]
        self.max_children = mrf_data["max_children"]
        # self.node_weight = mrf_data["node_weight"]
        self.node_weight = 0.5
        self.is_trained = mrf_data["is_trained"]

        # Initialize empty training data structures (not needed for generation)
        self.root_features = Counter()
        self.edge_features = Counter()
