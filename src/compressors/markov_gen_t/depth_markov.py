import logging
from collections import Counter, defaultdict
from typing import List, Tuple

import networkx as nx
import numpy as np

from .. import CompressedDataset, SerializationFormat
from ..trace import Trace


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


class DepthMarkovChain:
    def __init__(self, order: int = 1, max_depth: int = 10, max_children: int = 5000):
        self.order = order
        self.max_depth = max_depth
        self.max_children = max_children
        self.logger = logging.getLogger(__name__)

        # State is (node_name, depth, child_cnt)
        self.transition_matrix = defaultdict(Counter)
        self.start_states = Counter()
        self.node_names = set()

    def _extract_tree_sequences(self, trace: Trace) -> List[List[Tuple]]:
        """Extract DFS sequences with (node_name, depth, child_cnt) states from trace."""
        sequences = []

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

        # DFS traversal with (node_name, depth, child_cnt) information
        for root in root_spans:
            sequence = []
            self._dfs_extract_sequence(graph, root, 0, sequence, trace)
            if len(sequence) > self.order:
                sequences.append(sequence)

        return sequences

    def _dfs_extract_sequence(
        self,
        graph: nx.DiGraph,
        node: str,
        depth: int,
        sequence: List[Tuple],
        trace: Trace,
    ):
        """DFS traversal recording (node_name, depth, child_cnt) tuples."""
        try:
            node_name = trace.spans[node]["nodeName"]
            children = list(graph.successors(node))
            child_cnt = min(len(children), self.max_children)
            clamped_depth = min(depth, self.max_depth)

            state = (node_name, clamped_depth, child_cnt)
            sequence.append(state)

            self.node_names.add(node_name)

            # Sort children by startTime before visiting in DFS order
            children_sorted = sorted(
                children, key=lambda child_id: trace.spans[child_id]["nodeName"]
            )

            # Visit children in DFS order (sorted by startTime)
            for child in children_sorted:
                self._dfs_extract_sequence(graph, child, depth + 1, sequence, trace)

        except Exception as e:
            self.logger.warning(f"Error processing node {node}: {e}")

    def fit(self, traces: List[Trace]):
        """Learn Markov chain from traces."""
        self.logger.info(
            "Training Tree Markov Chain with (node_name, depth, child_cnt) states"
        )
        all_sequences = []

        for trace in traces:
            sequences = self._extract_tree_sequences(trace)
            all_sequences.extend(sequences)

        if not all_sequences:
            self.logger.warning("No valid sequences found for Markov chain training")
            return

        self._build_transition_matrix(all_sequences)
        self._normalize_transitions()
        self.logger.info(
            f"Trained Markov chain with {len(self.transition_matrix)} states and "
            f"{len(self.node_names)} unique node names"
        )

    def _build_transition_matrix(self, sequences: List[List[Tuple]]):
        """Build transition matrix from sequences."""
        for sequence in sequences:
            if len(sequence) < self.order + 1:
                continue

            start_state = (
                tuple(sequence[: self.order]) if self.order > 1 else sequence[0]
            )
            self.start_states[start_state] += 1

            # Build transitions
            for i in range(len(sequence) - self.order):
                if self.order == 1:
                    current_state = sequence[i]
                    next_state = sequence[i + 1]
                else:
                    current_state = tuple(sequence[i : i + self.order])
                    next_state = sequence[i + self.order]

                self.transition_matrix[current_state][next_state] += 1

    def _normalize_transitions(self):
        """Convert counts to probabilities."""
        for state in self.transition_matrix:
            total = sum(self.transition_matrix[state].values())
            if total > 0:
                for next_state in self.transition_matrix[state]:
                    self.transition_matrix[state][next_state] /= total

    def generate_tree_structure(self, max_nodes: int = 50) -> TreeNode:
        """Generate tree structure using DFS reconstruction based on child_cnt."""
        if not self.start_states:
            self.logger.warning("No start states available for generation")
            return None

        # Generate and reconstruct tree directly
        root = self._reconstruct_tree_from_dfs(max_nodes)
        return root

    def _reconstruct_tree_from_dfs(self, max_nodes: int) -> TreeNode:
        """Generate tree structure by sampling states during DFS construction."""
        # Sample start state
        start_state = self._sample_start_state()
        if start_state is None:
            return None

        # Initialize state tracking for Markov chain
        if self.order == 1:
            current_state = start_state
            root_node_name, root_depth, root_child_cnt = start_state
        else:
            current_state = start_state
            # For higher order, the start state is a tuple of states
            root_node_name, root_depth, root_child_cnt = start_state[-1]

        # Create root node with expected child count
        root = TreeNode(root_node_name, root_depth, root_child_cnt)
        nodes_generated = 1

        # Use DFS to build tree by sampling states on-demand
        nodes_generated, _ = self._build_tree_dfs(
            root, current_state, max_nodes, nodes_generated
        )

        return root

    def _build_tree_dfs(
        self,
        parent_node: TreeNode,
        current_state,
        max_nodes: int,
        nodes_generated: int,
    ) -> Tuple[int, any]:
        """Recursively build tree using DFS while sampling states."""
        if nodes_generated >= max_nodes:
            return nodes_generated, current_state

        # Generate children
        for _ in range(parent_node.get_expected_child_count()):
            if nodes_generated >= max_nodes:
                break

            # Rejection sampling: keep sampling next state until we get one with correct depth
            target_depth = parent_node.depth + 1
            max_attempts = 100

            for attempt in range(max_attempts):
                candidate_state = self._sample_next_state(current_state)
                if candidate_state is None:
                    break

                _, candidate_depth, _ = candidate_state
                if candidate_depth == target_depth:
                    next_state = candidate_state
                    break

            if candidate_state is None:
                break

            next_state = candidate_state
            node_name, depth, child_cnt_next = next_state

            # Create child node with expected child count
            child_node = TreeNode(node_name, depth, child_cnt_next)
            parent_node.add_child(child_node)
            nodes_generated += 1

            # Update current state for next iteration
            if self.order == 1:
                new_current_state = next_state
            else:
                new_current_state = current_state[1:] + (next_state,)

            # Recursively build subtree
            nodes_generated, current_state = self._build_tree_dfs(
                child_node, new_current_state, max_nodes, nodes_generated
            )

        return nodes_generated, current_state

    def _sample_start_state(self):
        """Sample initial state from start state distribution."""
        if not self.start_states:
            return None

        states = list(self.start_states.keys())
        probs = [self.start_states[state] for state in states]
        probs = np.array(probs, dtype=float)
        probs = probs / np.sum(probs)

        try:
            idx = np.random.choice(len(states), p=probs)
            return states[idx]
        except Exception:
            return states[0] if states else None

    def _sample_next_state(self, current_state):
        """Sample next state given current state."""
        if current_state not in self.transition_matrix:
            return None

        next_states = list(self.transition_matrix[current_state].keys())
        probs = [self.transition_matrix[current_state][state] for state in next_states]

        if not next_states:
            return None

        probs = np.array(probs, dtype=float)
        if np.sum(probs) == 0:
            return None

        probs = probs / np.sum(probs)

        try:
            idx = np.random.choice(len(next_states), p=probs)
            return next_states[idx]
        except Exception:
            return next_states[0] if next_states else None

    def save_state_dict(self, compressed_data: CompressedDataset):
        compressed_data.add(
            "depth_markov_chain",
            CompressedDataset(
                data={
                    "transition_matrix": (
                        dict(self.transition_matrix),
                        SerializationFormat.MSGPACK,
                    ),
                    "start_states": (
                        dict(self.start_states),
                        SerializationFormat.MSGPACK,
                    ),
                    "node_names": (
                        list(self.node_names),
                        SerializationFormat.MSGPACK,
                    ),
                    "order": (self.order, SerializationFormat.MSGPACK),
                    "max_depth": (self.max_depth, SerializationFormat.MSGPACK),
                    "max_children": (self.max_children, SerializationFormat.MSGPACK),
                }
            ),
            SerializationFormat.CLOUDPICKLE,
        )

    def load_state_dict(self, compressed_dataset: CompressedDataset):
        """Load state dictionary from compressed dataset."""
        if "depth_markov_chain" not in compressed_dataset:
            raise ValueError("No depth_markov_chain found in compressed dataset")
        depth_markov_data = compressed_dataset["depth_markov_chain"]

        # Convert back to appropriate data structures
        self.transition_matrix = defaultdict(
            Counter,
            {
                state: Counter(transitions)
                for state, transitions in depth_markov_data["transition_matrix"].items()
            },
        )

        self.start_states = Counter(depth_markov_data["start_states"])
        self.node_names = set(depth_markov_data["node_names"])
        self.order = depth_markov_data["order"]
        self.max_depth = depth_markov_data["max_depth"]
        self.max_children = depth_markov_data["max_children"]
