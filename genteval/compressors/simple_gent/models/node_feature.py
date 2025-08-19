from dataclasses import dataclass


@dataclass(frozen=True)
class NodeFeature:
    """
    Represents a node feature as (node_idx, child_count) pair.

    This class is used throughout the Simple GenT algorithm to represent
    node characteristics. Uses node indices instead of string names for
    efficiency and consistency with shared node encoder.

    Attributes:
        node_idx: The node index (from shared NodeEncoder)
        child_count: The number of children this node has
    """

    node_idx: int
    child_count: int

    def to_tuple(self) -> tuple[int, int]:
        """Convert to tuple representation for use as dictionary keys."""
        return (self.node_idx, self.child_count)

    @classmethod
    def from_tuple(cls, feature_tuple: tuple[int, int]) -> "NodeFeature":
        """Create NodeFeature from tuple representation."""
        return cls(node_idx=feature_tuple[0], child_count=feature_tuple[1])

    def __str__(self) -> str:
        return f"({self.node_idx}, {self.child_count})"

    def __repr__(self) -> str:
        return f"NodeFeature(node_idx={self.node_idx}, child_count={self.child_count})"
