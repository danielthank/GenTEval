import gzip
import json
import os
import pathlib
import shutil
from enum import Enum
from typing import Any

import cloudpickle
import msgpack


class SerializationFormat(Enum):
    """Enum for supported serialization formats."""

    MSGPACK = "msgpack"
    CLOUDPICKLE = "cloudpickle"


class CompressedDataset:
    """
    A key-value store for compressed data with support for msgpack and cloudpickle formats.
    Users must specify the format when adding data.
    """

    def __init__(
        self, compression_level: int = 9, data: dict[str, tuple] | None = None
    ):
        """
        Initialize a new CompressedDataset.

        Args:
            compression_level: Compression level (0-9, 9 being highest)
            data: Optional dictionary for batch initialization where keys are dataset keys
                  and values are tuples of (data, format) where format is a SerializationFormat enum

        Example:
            dataset = CompressedDataset(data={
                "model_weights": (model.state_dict(), SerializationFormat.CLOUDPICKLE),
                "config": (config_data, SerializationFormat.MSGPACK),
                "metadata": (meta_info, SerializationFormat.CLOUDPICKLE)
            })
        """
        self.data: dict[str, Any] = {}
        self.formats: dict[
            str, SerializationFormat
        ] = {}  # Track format used for each key
        self.compression_level = compression_level

        # Initialize with batch data if provided
        if data is not None:
            self.add_batch(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def add(self, key: str, value: Any, format: SerializationFormat) -> None:
        """
        Add an item to the dataset with specified serialization format.

        Args:
            key: The key to store the value under
            value: The value to store
            format: Serialization format (SerializationFormat enum)
        """
        if not isinstance(format, SerializationFormat):
            raise TypeError("format must be a SerializationFormat enum value")

        self.data[key] = value
        self.formats[key] = format

    def add_batch(self, data_dict: dict[str, tuple]) -> None:
        """
        Add multiple items to the dataset at once using a dictionary.

        Args:
            data_dict: Dictionary where keys are the dataset keys and values are tuples
                      of (data, format) where format is a SerializationFormat enum

        Example:
            dataset.add_batch({
                "model_weights": (model.state_dict(), SerializationFormat.CLOUDPICKLE),
                "config": (config_data, SerializationFormat.MSGPACK),
                "metadata": (meta_info, SerializationFormat.CLOUDPICKLE)
            })
        """
        if not isinstance(data_dict, dict):
            raise TypeError("data_dict must be a dictionary")

        # Validate all entries first before adding any
        for key, value in data_dict.items():
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError(
                    f"Value for key '{key}' must be a tuple of (data, format)"
                )

            data, format = value
            if not isinstance(format, SerializationFormat):
                raise TypeError(
                    f"Format for key '{key}' must be a SerializationFormat enum value"
                )

        # Add all items
        for key, (data, format) in data_dict.items():
            self.add(key, data, format)

    def remove(self, key: str) -> None:
        """Remove an item from the dataset."""
        if key in self.data:
            del self.data[key]
        if key in self.formats:
            del self.formats[key]

    def keys(self):
        """Return all keys in the dataset."""
        return self.data.keys()

    def get_size(self) -> int:
        """
        Get the approximate size of the dataset in bytes.

        Returns:
            Approximate size in bytes
        """
        total_size = 0
        for key, value in self.data.items():
            if key not in self.formats:
                raise ValueError(f"No format specified for key '{key}'")

            # Serialize and compress to get accurate size
            serialized = self._serialize(value, self.formats[key])
            compressed = self._compress(serialized)
            total_size += len(compressed) + len(key.encode("utf-8"))

        return total_size

    def extend(self, other: "CompressedDataset") -> None:
        """
        Extend the current dataset with another CompressedDataset.

        Args:
            other: Another CompressedDataset instance to merge
        """
        if not isinstance(other, CompressedDataset):
            raise TypeError("other must be a CompressedDataset instance")

        for key, value in other.data.items():
            if key in self.data:
                raise KeyError(f"Key '{key}' already exists in the dataset")
            self.add(key, value, other.formats[key])

    def _serialize(self, obj: Any, format: SerializationFormat) -> bytes:
        """Serialize an object based on the selected format."""
        if format == SerializationFormat.MSGPACK:
            return msgpack.packb(obj, use_bin_type=True)
        if format == SerializationFormat.CLOUDPICKLE:
            return cloudpickle.dumps(obj)
        raise ValueError(f"Unsupported serialization format: {format}")

    def _deserialize(self, data: bytes, format: SerializationFormat) -> Any:
        """Deserialize data based on the selected format."""
        if format == SerializationFormat.MSGPACK:
            return msgpack.unpackb(data, raw=False)
        if format == SerializationFormat.CLOUDPICKLE:
            return cloudpickle.loads(data)
        raise ValueError(f"Unsupported serialization format: {format}")

    def _compress(self, data: bytes) -> bytes:
        """Compress serialized data."""
        return gzip.compress(data, compresslevel=self.compression_level)

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        return gzip.decompress(data)

    def save(self, dir: pathlib.Path):
        """
        Clean the directory and save the dataset to the specified directory.

        Args:
            dir: Directory path where the dataset will be saved
        """
        # Delete existing directory if it exists
        if dir.exists():
            shutil.rmtree(dir)
        dir.mkdir(parents=True, exist_ok=True)

        # Verify all items have formats
        missing_formats = [key for key in self.data.keys() if key not in self.formats]
        if missing_formats:
            raise ValueError(
                f"Missing format specification for keys: {missing_formats}"
            )

        # Convert enum values to strings for JSON serialization
        formats_serializable = {k: v.value for k, v in self.formats.items()}

        # Save metadata
        metadata = {
            "compression_level": self.compression_level,
            "keys": list(self.data.keys()),
            "formats": formats_serializable,
        }

        with open(dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create data directory
        data_dir = dir / "data"
        if not data_dir.exists():
            data_dir.mkdir()

        # Save each item separately
        for key, value in self.data.items():
            format_to_use = self.formats[key]
            serialized = self._serialize(value, format_to_use)
            compressed = self._compress(serialized)

            with open(data_dir / key, "wb") as f:
                f.write(compressed)

    @classmethod
    def load(cls, dir: pathlib.Path) -> "CompressedDataset":
        """
        Load a dataset from the specified directory.

        Args:
            dir: Directory path where the dataset is stored

        Returns:
            Loaded CompressedDataset instance
        """
        # Load metadata
        with open(dir / "metadata.json") as f:
            metadata = json.load(f)

        # Create new instance
        dataset = cls(compression_level=metadata["compression_level"])

        # Convert string format values back to enum values
        formats_str = metadata["formats"]
        formats_enum = {k: SerializationFormat(v) for k, v in formats_str.items()}
        dataset.formats = formats_enum

        keys = metadata["keys"]

        # Load data
        data_dir = dir / "data"
        for filename in os.listdir(data_dir):
            if filename in keys:
                with open(data_dir / filename, "rb") as f:
                    compressed = f.read()

                decompressed = dataset._decompress(compressed)
                format_to_use = dataset.formats[filename]
                value = dataset._deserialize(decompressed, format_to_use)
                dataset.data[filename] = value

        return dataset
