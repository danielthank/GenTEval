import logging
import pickle

import numpy as np
from sklearn.preprocessing import OrdinalEncoder


class NodeEncoder:
    """
    Shared node encoder that wraps sklearn's OrdinalEncoder with additional functionality.

    This class provides a consistent interface for encoding node names to indices
    across all models in the simple_gent system. It handles unknown nodes gracefully
    during inference (mapping them to index 0) and provides serialization support for protobuf storage.
    """

    def __init__(self):
        # Use OrdinalEncoder with unknown_value=-1, then map to 0 in transform
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=int
        )
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)

    def fit(self, node_names: list[str]) -> "NodeEncoder":
        """
        Fit the encoder on a list of node names.

        Args:
            node_names: List of unique node names to learn the vocabulary from

        Returns:
            Self for method chaining
        """
        unique_names = list(set(node_names))
        # OrdinalEncoder expects 2D array
        self.encoder.fit(np.array(unique_names).reshape(-1, 1))
        self.is_fitted = True
        self.logger.info(
            f"Fitted NodeEncoder with {len(unique_names)} unique node names"
        )
        return self

    def transform(self, node_names: str | list[str]) -> int | np.ndarray:
        """
        Transform node names to indices.

        Args:
            node_names: Single node name string or list of node names

        Returns:
            Single index (int) or array of indices, handling unknown names gracefully
        """
        if not self.is_fitted:
            raise ValueError("NodeEncoder must be fitted before transform")

        is_single = isinstance(node_names, str)
        if is_single:
            node_names = [node_names]

        # OrdinalEncoder expects 2D array and handles unknown values automatically
        indices = self.encoder.transform(np.array(node_names).reshape(-1, 1))
        indices = indices.flatten().astype(int)

        # Map unknown values (-1) to 0
        indices = np.where(indices == -1, 0, indices)

        return indices[0] if is_single else indices

    def inverse_transform(
        self, indices: int | list[int] | np.ndarray
    ) -> str | list[str]:
        """
        Transform indices back to node names.

        Args:
            indices: Single index or list/array of indices

        Returns:
            Single node name string or list of node names
        """
        if not self.is_fitted:
            raise ValueError("NodeEncoder must be fitted before inverse_transform")

        is_single = isinstance(indices, (int, np.integer))
        if is_single:
            indices = [indices]

        # Ensure all indices are valid (clip to valid range)
        indices = np.array(indices)
        vocab_size = len(self.encoder.categories_[0])
        indices = np.clip(indices, 0, vocab_size - 1)

        # OrdinalEncoder expects 2D array
        names = self.encoder.inverse_transform(indices.reshape(-1, 1))
        names = names.flatten()

        return names[0] if is_single else list(names)

    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary (number of unique node names)."""
        if not self.is_fitted:
            return 0
        return len(self.encoder.categories_[0])

    def get_classes(self) -> np.ndarray:
        """Get the array of node name classes."""
        if not self.is_fitted:
            return np.array([])
        return self.encoder.categories_[0]

    def serialize(self) -> bytes:
        """Serialize the encoder for protobuf storage."""
        if not self.is_fitted:
            raise ValueError("NodeEncoder must be fitted before serialization")
        return pickle.dumps(self.encoder)

    @classmethod
    def deserialize(cls, data: bytes) -> "NodeEncoder":
        """Deserialize the encoder from protobuf storage."""
        encoder_instance = cls()
        encoder_instance.encoder = pickle.loads(data)
        encoder_instance.is_fitted = True
        return encoder_instance

    def save_state_dict(self, proto_models):
        """Save node encoder state to protobuf message."""
        if not self.is_fitted:
            raise ValueError("NodeEncoder must be fitted before saving")
        proto_models.node_encoder = self.serialize()

    def load_state_dict(self, proto_models):
        """Load node encoder state from protobuf message."""
        if proto_models.node_encoder:
            self.encoder = pickle.loads(proto_models.node_encoder)
            self.is_fitted = True
        else:
            raise ValueError("No node_encoder data found in protobuf")

    def __repr__(self) -> str:
        if self.is_fitted:
            return f"NodeEncoder(vocab_size={self.get_vocab_size()})"
        return "NodeEncoder(not_fitted)"
