import logging
import pickle

import numpy as np
from sklearn.preprocessing import LabelEncoder


class NodeEncoder:
    """
    Shared node encoder that wraps sklearn's LabelEncoder with additional functionality.

    This class provides a consistent interface for encoding node names to indices
    across all models in the simple_gent system. It handles unknown nodes gracefully
    during inference and provides serialization support for protobuf storage.
    """

    def __init__(self):
        self.encoder = LabelEncoder()
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
        self.encoder.fit(unique_names)
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

        # Split into known and unknown names for batch processing
        known_names = []
        unknown_indices = []

        for i, name in enumerate(node_names):
            if name in self.encoder.classes_:
                known_names.append(name)
            else:
                known_names.append(None)  # Placeholder
                unknown_indices.append(i)

        # Batch transform known names
        indices = np.zeros(len(node_names), dtype=int)

        # Get indices for known names (filter out None placeholders)
        valid_names = [name for name in known_names if name is not None]
        if valid_names:
            valid_indices = self.encoder.transform(valid_names)
            # Place valid indices back in correct positions
            valid_idx = 0
            for i, name in enumerate(known_names):
                if name is not None:
                    indices[i] = valid_indices[valid_idx]
                    valid_idx += 1

        # Handle unknown names with random valid indices
        if unknown_indices:
            vocab_size = len(self.encoder.classes_)
            random_indices = np.random.randint(0, vocab_size, size=len(unknown_indices))
            for i, unknown_idx in enumerate(unknown_indices):
                indices[unknown_idx] = random_indices[i]
                self.logger.debug(
                    f"Unknown node name '{node_names[unknown_idx]}' mapped to random index {random_indices[i]}"
                )

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

        # Ensure all indices are valid
        valid_indices = []
        for idx in indices:
            if 0 <= idx < len(self.encoder.classes_):
                valid_indices.append(idx)
            else:
                # Invalid index - use 0 as fallback
                valid_indices.append(0)
                self.logger.warning(f"Invalid index {idx} mapped to fallback index 0")

        names = self.encoder.inverse_transform(valid_indices)
        return names[0] if is_single else list(names)

    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary (number of unique node names)."""
        if not self.is_fitted:
            return 0
        return len(self.encoder.classes_)

    def get_classes(self) -> np.ndarray:
        """Get the array of node name classes."""
        if not self.is_fitted:
            return np.array([])
        return self.encoder.classes_

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
