import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from compressors.trace import Trace


class TreeNode:
    """Represents a node in the generated tree structure."""

    def __init__(self, node_name: str, depth: int, span_id: str = None):
        self.node_name = node_name
        self.depth = depth
        self.span_id = span_id
        self.children = []
        self.parent = None

    def add_child(self, child):
        """Add a child node."""
        child.parent = self
        self.children.append(child)

    def get_child_count(self) -> int:
        """Get the number of children."""
        return len(self.children)


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

        # For child count distribution per (node_name, depth)
        self.child_count_distributions = defaultdict(Counter)

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
            self.child_count_distributions[(node_name, clamped_depth)][child_cnt] += 1

            # Visit children in DFS order
            for child in children:
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

            # Record start state weighted by sequence length
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

        # Generate DFS sequence
        sequence = self._generate_dfs_sequence(max_nodes)
        if not sequence:
            self.logger.warning("Failed to generate DFS sequence")
            return None

        # Reconstruct tree from DFS sequence
        root = self._reconstruct_tree_from_dfs(sequence)
        return root

    def _generate_dfs_sequence(self, max_nodes: int) -> List[Tuple]:
        """Generate a DFS sequence of (node_name, depth, child_cnt) states."""
        sequence = []

        # Sample start state
        start_state = self._sample_start_state()
        if start_state is None:
            return []

        if self.order == 1:
            current_state = start_state
            sequence.append(current_state)
        else:
            current_state = start_state
            sequence.extend(current_state)

        # Generate sequence
        nodes_generated = len(sequence)
        while nodes_generated < max_nodes:
            if current_state not in self.transition_matrix:
                break

            next_state = self._sample_next_state(current_state)
            if next_state is None:
                break

            sequence.append(next_state)

            if self.order == 1:
                current_state = next_state
            else:
                current_state = current_state[1:] + (next_state,)

            nodes_generated += 1

        return sequence

    def _reconstruct_tree_from_dfs(self, sequence: List[Tuple]) -> TreeNode:
        """Reconstruct tree structure from DFS sequence using child_cnt information."""
        if not sequence:
            return None

        # Stack to keep track of nodes and their expected children
        stack = []
        root = None
        node_counter = 0

        for i, (node_name, depth, child_cnt) in enumerate(sequence):
            # Create new node
            new_node = TreeNode(node_name, depth, f"span_{node_counter}")
            node_counter += 1

            if depth == 0:
                # Root node
                root = new_node
                stack = [(new_node, child_cnt)]
            else:
                # Find parent in stack (should be at depth-1)
                while stack and stack[-1][0].depth >= depth:
                    stack.pop()

                if stack:
                    parent, remaining_children = stack[-1]
                    parent.add_child(new_node)

                    # Update remaining children count for parent
                    stack[-1] = (parent, remaining_children - 1)

                    # Remove parent from stack if it has no more children expected
                    if remaining_children - 1 <= 0:
                        stack.pop()

                # Add current node to stack if it has children
                if child_cnt > 0:
                    stack.append((new_node, child_cnt))

        return root

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

    def sample_child_count(self, node_name: str, depth: int) -> int:
        """Sample child count for a given (node_name, depth) pair."""
        key = (node_name, min(depth, self.max_depth))
        if key not in self.child_count_distributions:
            return 0

        counts = list(self.child_count_distributions[key].keys())
        probs = [self.child_count_distributions[key][count] for count in counts]

        if not counts:
            return 0

        probs = np.array(probs, dtype=float)
        probs = probs / np.sum(probs)

        try:
            idx = np.random.choice(len(counts), p=probs)
            return counts[idx]
        except Exception:
            return counts[0] if counts else 0

    def tree_to_trace_format(self, root: TreeNode) -> Dict[str, Any]:
        """Convert generated tree to trace format."""
        spans = {}

        def traverse(node, parent_id=None):
            span_data = {
                "nodeName": node.node_name,
                "parentSpanId": parent_id,
                "depth": node.depth,
            }
            spans[node.span_id] = span_data

            for child in node.children:
                traverse(child, node.span_id)

        traverse(root)
        return {"spans": spans}

    def get_state_dict(self):
        """Get state dictionary for serialization."""
        return {
            "transition_matrix": dict(self.transition_matrix),
            "start_states": dict(self.start_states),
            "child_count_distributions": dict(self.child_count_distributions),
            "node_names": list(self.node_names),
            "order": self.order,
            "max_depth": self.max_depth,
            "max_children": self.max_children,
        }

    def load_state_dict(self, state_dict):
        """Load state dictionary from serialization."""
        self.transition_matrix = defaultdict(Counter, state_dict["transition_matrix"])
        self.start_states = Counter(state_dict["start_states"])
        self.child_count_distributions = defaultdict(
            Counter, state_dict["child_count_distributions"]
        )
        self.node_names = set(state_dict["node_names"])
        self.order = state_dict["order"]
        self.max_depth = state_dict["max_depth"]
        self.max_children = state_dict["max_children"]
