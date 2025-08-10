"""Shared data structures for GenTEval."""


def count_spans_per_tree(spans):
    """Use Union-Find to efficiently count spans per tree.

    Args:
        spans: Dictionary of span_id -> span_data with parentSpanId field

    Returns:
        List of tree sizes (number of spans in each connected component)
    """
    if not spans:
        return []

    # Create mapping from span_id to index
    span_ids = list(spans.keys())
    id_to_index = {span_id: i for i, span_id in enumerate(span_ids)}

    # Initialize Union-Find
    uf = UnionFind(len(span_ids))

    # Union each child with its parent
    for span_id, span_data in spans.items():
        parent_id = span_data["parentSpanId"]
        if parent_id is not None and parent_id in id_to_index:
            child_idx = id_to_index[span_id]
            parent_idx = id_to_index[parent_id]
            uf.union(parent_idx, child_idx)

    # Get sizes of all connected components (trees)
    return uf.get_component_sizes()


class UnionFind:
    """Union-Find data structure with path compression and union by rank."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n  # Track size of each component
        self.rank = [0] * n

    def find(self, x):
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """Union two components by rank."""
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return

        # Union by rank for balance
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]  # Update component size

        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

    def get_component_sizes(self):
        """Return sizes of all connected components."""
        root_sizes = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in root_sizes:
                root_sizes[root] = self.size[root]
        return list(root_sizes.values())
