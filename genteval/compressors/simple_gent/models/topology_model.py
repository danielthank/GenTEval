import logging
from collections import defaultdict

import networkx as nx
import numpy as np
from tqdm import tqdm

from genteval.compressors.simple_gent.proto import simple_gent_pb2
from genteval.utils.data_structures import count_spans_per_tree

from .node_feature import NodeFeature
from .trace_type import TraceType


class TreeNode:
    """Represents a node in the generated tree structure."""

    def __init__(self, feature: NodeFeature, depth: int = 0):
        self.feature = feature
        self.depth = depth
        self.children = []
        self.parent = None

    def add_child(self, child):
        """Add a child node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    @property
    def child_count(self) -> int:
        return len(self.children)


class TopologyModel:
    """
    Topology model: (parent_node_idx, parent_child_cnt, child_node_idx, child_child_cnt) -> float (potential)

    Stratified by trace type (normal vs error) to ensure correct error pattern generation.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Edge potentials: {trace_type: {time_bucket: {((parent_feature, child_idx), child_feature): potential}}}
        self.edge_potentials = {
            TraceType.NORMAL: defaultdict(lambda: defaultdict(float)),
            TraceType.ERROR: defaultdict(lambda: defaultdict(float)),
        }

        # Cache for child candidates per (parent_feature, child_idx)
        # {trace_type: {time_bucket: {(parent_feature, child_idx): [(child_feature, prob)]}}}
        self.child_candidates_cache = {
            TraceType.NORMAL: defaultdict(dict),
            TraceType.ERROR: defaultdict(dict),
        }

        # Max nodes per time bucket (loaded from protobuf)
        self.max_nodes_per_bucket = {}  # {time_bucket: max_nodes}

    def _calculate_max_nodes_per_bucket(self, traces):
        """Calculate the maximum number of nodes per time bucket."""

        self.logger.info("Calculating maximum nodes per time bucket")
        max_nodes_per_bucket = defaultdict(int)

        for trace in tqdm(traces, desc="Calculating max nodes per bucket"):
            trace_start_time = trace.start_time
            time_bucket = int(trace_start_time // self.config.time_bucket_duration_us)

            # Count spans per tree in this trace
            tree_sizes = count_spans_per_tree(trace.spans)
            if tree_sizes:
                max_tree_size = max(tree_sizes)
                max_nodes_per_bucket[time_bucket] = max(
                    max_nodes_per_bucket[time_bucket], max_tree_size
                )

        self.logger.info(
            f"Found {len(max_nodes_per_bucket)} time buckets with max nodes"
        )
        return dict(max_nodes_per_bucket)

    def fit(self, traces):
        """Learn topology potentials from traces, stratified by trace type."""
        self.logger.info(f"Training topology model on {len(traces)} traces")

        # Calculate max_nodes_count per time bucket
        self.max_nodes_per_bucket = self._calculate_max_nodes_per_bucket(traces)

        # {trace_type: {time_bucket: {((parent_feature, child_idx), child_feature): count}}}
        edge_counts = {
            TraceType.NORMAL: defaultdict(lambda: defaultdict(int)),
            TraceType.ERROR: defaultdict(lambda: defaultdict(int)),
        }

        self.logger.info(
            "Extracting topology features from traces (stratified by trace type)"
        )
        total_root_spans = 0
        total_edges_extracted = 0
        normal_trace_count = 0
        error_trace_count = 0

        for trace in tqdm(traces, desc="Processing traces"):
            trace_start_time = trace.start_time
            time_bucket = int(trace_start_time // self.config.time_bucket_duration_us)

            trace_type = TraceType.from_trace(trace, self.config.stratified_sampling)
            if trace_type == TraceType.ERROR:
                error_trace_count += 1
            else:
                normal_trace_count += 1

            # Build graph from trace
            graph = nx.DiGraph()
            root_spans = []

            for span_id, span_data in trace.spans.items():
                graph.add_node(span_id, **span_data)
                parent_id = span_data.get("parentSpanId")

                if parent_id is None:
                    # No parent - this is a root span
                    root_spans.append(span_id)
                elif parent_id in trace.spans:
                    # Parent exists - add edge
                    graph.add_edge(parent_id, span_id)
                else:
                    # Parent doesn't exist - treat as root span
                    root_spans.append(span_id)

            total_root_spans += len(root_spans)

            # Extract features from each tree
            for root in root_spans:
                edges_before = sum(
                    len(bucket) for bucket in edge_counts[trace_type].values()
                )
                self._extract_tree_features(
                    graph,
                    root,
                    time_bucket,
                    edge_counts[trace_type],
                )
                edges_after = sum(
                    len(bucket) for bucket in edge_counts[trace_type].values()
                )
                total_edges_extracted += edges_after - edges_before

        self.logger.info(
            f"Extracted features from {total_root_spans} trees, {total_edges_extracted} edges total"
        )
        self.logger.info(
            f"Trace type distribution: {normal_trace_count} normal, {error_trace_count} error"
        )

        # Convert counts to potentials using log-likelihood
        self.logger.info("Converting edge counts to potentials")
        total_processed_buckets = 0
        total_processed_edges = 0

        for trace_type in [TraceType.NORMAL, TraceType.ERROR]:
            for time_bucket, bucket_edges in tqdm(
                edge_counts[trace_type].items(),
                desc=f"Processing {trace_type} time buckets",
            ):
                total_edges = sum(bucket_edges.values())
                if total_edges == 0:
                    continue

                total_processed_buckets += 1

                for (
                    (parent_feature, child_idx),
                    child_feature,
                ), count in bucket_edges.items():
                    potential = np.log(count) - np.log(total_edges)
                    self.edge_potentials[trace_type][time_bucket][
                        ((parent_feature, child_idx), child_feature)
                    ] = potential
                    total_processed_edges += 1

        self.logger.info(
            f"Processed {total_processed_buckets} time buckets with {total_processed_edges} edge potentials"
        )

        # Build child candidates cache
        self.logger.info("Building child candidates cache")
        self._build_child_candidates_cache()

        normal_edges = sum(
            len(bucket) for bucket in self.edge_potentials[TraceType.NORMAL].values()
        )
        error_edges = sum(
            len(bucket) for bucket in self.edge_potentials[TraceType.ERROR].values()
        )
        self.logger.info(
            f"Topology model training complete: "
            f"normal={len(self.edge_potentials[TraceType.NORMAL])} buckets/{normal_edges} edges, "
            f"error={len(self.edge_potentials[TraceType.ERROR])} buckets/{error_edges} edges"
        )

    def _extract_tree_features(
        self, graph: nx.DiGraph, root: str, time_bucket: int, edge_counts
    ):
        """Extract parent-child feature pairs from a tree."""
        # BFS to extract features
        queue = [(root, 0)]
        visited = {root}

        while queue:
            node_id, depth = queue.pop(0)
            node_data = graph.nodes[node_id]
            node_name_idx = node_data["nodeIdx"]

            # Get children and sort by startTime
            children_data = []
            for child_id in graph.successors(node_id):
                if child_id not in visited:
                    child_node_data = graph.nodes[child_id]
                    children_data.append((child_id, child_node_data["startTime"]))

            # Sort children by startTime to match MetadataVAE ordering
            children_data.sort(key=lambda x: x[1])
            parent_child_count = len(children_data)

            # Create parent feature (child_idx=-1 indicates parent reference, not child position)
            parent_feature = NodeFeature(
                node_idx=node_name_idx,
                child_idx=-1,  # N/A - this node is being used as parent
                child_count=parent_child_count,
            )

            # Process edges to children with child_idx
            for child_idx, (child_id, _) in enumerate(children_data):
                visited.add(child_id)
                child_data = graph.nodes[child_id]
                child_name_idx = child_data["nodeIdx"]

                # Count grandchildren for child feature
                grandchildren = list(graph.successors(child_id))
                child_child_count = len(grandchildren)

                # Create child feature with actual child_idx position
                child_feature = NodeFeature(
                    node_idx=child_name_idx,
                    child_idx=child_idx,
                    child_count=child_child_count,
                )

                # Record edge with tuple key: (parent_feature, child_idx)
                edge_counts[time_bucket][
                    ((parent_feature, child_idx), child_feature)
                ] += 1

                # Continue BFS
                queue.append((child_id, depth + 1))

    def _build_child_candidates_cache(self):
        """Build cache of possible children for each (parent_feature, child_idx) pair."""
        total_parent_features = 0

        for trace_type in [TraceType.NORMAL, TraceType.ERROR]:
            for time_bucket, bucket_edges in tqdm(
                self.edge_potentials[trace_type].items(),
                desc=f"Building cache for {trace_type}",
            ):
                parent_children = defaultdict(list)

                # Group by (parent_feature, child_idx) tuple
                for (
                    (parent_feature, child_idx),
                    child_feature,
                ), potential in bucket_edges.items():
                    parent_key = (parent_feature, child_idx)
                    parent_children[parent_key].append((child_feature, potential))

                # Convert to probabilities and cache
                for parent_key, children_list in parent_children.items():
                    # Convert log potentials to probabilities
                    potentials = np.array([pot for _, pot in children_list])
                    # Use softmax to convert to probabilities
                    probabilities = np.exp(potentials - np.max(potentials))
                    probabilities = probabilities / probabilities.sum()

                    children_with_probs = [
                        (child_feature, prob)
                        for (child_feature, _), prob in zip(
                            children_list, probabilities, strict=False
                        )
                    ]

                    self.child_candidates_cache[trace_type][time_bucket][parent_key] = (
                        children_with_probs
                    )
                    total_parent_features += 1

        self.logger.info(
            f"Built child candidates cache for {total_parent_features} unique (parent_feature, child_idx) pairs"
        )

    def generate_tree_structure(
        self,
        root_feature: NodeFeature,
        time_bucket: int,
        trace_type: str = TraceType.NORMAL,
    ) -> TreeNode | None:
        """Generate a tree structure starting from a root feature.

        Args:
            root_feature: Root node feature
            time_bucket: Time bucket for this tree
            trace_type: Type of trace to generate ("normal" or "error")

        Returns:
            TreeNode representing the root of the generated tree
        """
        # Get max_nodes from stored protobuf data
        max_nodes = self._get_max_nodes_for_bucket(time_bucket)
        if max_nodes <= 0:
            return None

        root_node = TreeNode(root_feature, depth=0)
        nodes_created = 1

        # BFS queue: (node, expected_children)
        queue = [(root_node, root_feature.child_count)]

        while queue and nodes_created < max_nodes:
            current_node, expected_children = queue.pop(0)

            if current_node.depth >= self.config.max_depth:
                continue

            # Generate children for this node
            for child_idx in range(min(expected_children, self.config.max_children)):
                if nodes_created >= max_nodes:
                    break

                child_feature = self._sample_child_feature(
                    current_node.feature,
                    time_bucket,
                    child_idx,
                    trace_type,
                )
                if child_feature is None:
                    continue

                child_node = TreeNode(child_feature, depth=current_node.depth + 1)
                current_node.add_child(child_node)

                # Add to queue if it should have children
                if child_feature.child_count > 0:
                    queue.append((child_node, child_feature.child_count))

                nodes_created += 1

        return root_node

    def _sample_child_feature(
        self,
        parent_feature: NodeFeature,
        time_bucket: int,
        child_idx: int,
        trace_type: TraceType,
    ) -> NodeFeature | None:
        """Sample a child feature given a parent feature and child position."""
        parent_ref = NodeFeature(
            node_idx=parent_feature.node_idx,
            child_idx=-1,
            child_count=parent_feature.child_count,
        )
        cache_key = (parent_ref, child_idx)

        candidates = self.child_candidates_cache[trace_type][time_bucket].get(cache_key)
        if not candidates:
            return None

        features, probabilities = zip(*candidates, strict=False)
        return np.random.choice(features, p=probabilities)

    def _get_max_nodes_for_bucket(self, time_bucket: int) -> int:
        """Get max nodes limit for a time bucket with fallbacks."""
        # Direct lookup
        if time_bucket in self.max_nodes_per_bucket:
            return self.max_nodes_per_bucket[time_bucket]

        # Fallback to closest time bucket
        if self.max_nodes_per_bucket:
            closest_bucket = min(
                self.max_nodes_per_bucket.keys(), key=lambda b: abs(b - time_bucket)
            )
            return self.max_nodes_per_bucket[closest_bucket]

        # Final fallback to config default
        return 1000

    def save_state_dict(self, proto_models):
        """Save model state to protobuf message."""

        # Group data by time buckets
        time_bucket_data = {}

        for trace_type in [TraceType.NORMAL, TraceType.ERROR]:
            proto_trace_type = trace_type.to_proto()

            for time_bucket, bucket_edges in self.edge_potentials[trace_type].items():
                if time_bucket not in time_bucket_data:
                    time_bucket_data[time_bucket] = []

                for (
                    (parent_feature, child_idx),
                    child_feature,
                ), potential in bucket_edges.items():
                    topology_model = simple_gent_pb2.TopologyModel()
                    topology_model.parent_feature.node_idx = parent_feature.node_idx
                    topology_model.parent_feature.child_idx = (
                        child_idx  # Save the position, not -1
                    )
                    topology_model.parent_feature.child_count = (
                        parent_feature.child_count
                    )
                    topology_model.child_feature.node_idx = child_feature.node_idx
                    topology_model.child_feature.child_idx = child_feature.child_idx
                    topology_model.child_feature.child_count = child_feature.child_count
                    topology_model.potential = potential
                    topology_model.trace_type = proto_trace_type
                    time_bucket_data[time_bucket].append(topology_model)

        # Add to protobuf message
        for time_bucket, topology_models in time_bucket_data.items():
            # Find or create time bucket
            bucket_models = None
            for tb in proto_models.time_buckets:
                if tb.time_bucket == time_bucket:
                    bucket_models = tb
                    break

            if bucket_models is None:
                bucket_models = proto_models.time_buckets.add()
                bucket_models.time_bucket = time_bucket

            # Set max_nodes_count if provided
            if time_bucket in self.max_nodes_per_bucket:
                bucket_models.max_nodes_count = self.max_nodes_per_bucket[time_bucket]

            # Add topology models to this bucket
            bucket_models.topology_models.extend(topology_models)

    def load_state_dict(self, proto_models):
        """Load model state from protobuf message."""
        self.edge_potentials = {
            TraceType.NORMAL: defaultdict(lambda: defaultdict(float)),
            TraceType.ERROR: defaultdict(lambda: defaultdict(float)),
        }

        # Load from protobuf message
        for time_bucket_models in proto_models.time_buckets:
            time_bucket = time_bucket_models.time_bucket

            # Load max_nodes_count for this time bucket
            self.max_nodes_per_bucket[time_bucket] = time_bucket_models.max_nodes_count

            for topology_model in time_bucket_models.topology_models:
                # Extract child_idx from parent_feature field (it was saved as the position)
                child_idx = topology_model.parent_feature.child_idx

                # Create parent_feature with child_idx=-1 (parent reference)
                parent_feature = NodeFeature(
                    node_idx=topology_model.parent_feature.node_idx,
                    child_idx=-1,  # N/A - parent reference
                    child_count=topology_model.parent_feature.child_count,
                )
                child_feature = NodeFeature(
                    node_idx=topology_model.child_feature.node_idx,
                    child_idx=topology_model.child_feature.child_idx,
                    child_count=topology_model.child_feature.child_count,
                )

                trace_type = TraceType.from_proto(topology_model.trace_type)

                # Reconstruct tuple key structure
                self.edge_potentials[trace_type][time_bucket][
                    ((parent_feature, child_idx), child_feature)
                ] = topology_model.potential

        # Reset cache before rebuilding
        self.child_candidates_cache = {
            TraceType.NORMAL: defaultdict(dict),
            TraceType.ERROR: defaultdict(dict),
        }

        # Rebuild cache
        self._build_child_candidates_cache()

        normal_buckets = len(self.edge_potentials[TraceType.NORMAL])
        error_buckets = len(self.edge_potentials[TraceType.ERROR])
        self.logger.info(
            f"Loaded topology model with {normal_buckets} normal buckets, "
            f"{error_buckets} error buckets"
        )
