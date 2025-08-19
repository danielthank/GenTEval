import gzip
import json
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
    GRPC = "grpc"


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
        self.proto_specs: dict[str, Any] = {}  # Track protobuf specs for GRPC format
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

    def add(
        self, key: str, value: Any, fmt: SerializationFormat, proto_spec: Any = None
    ) -> None:
        """
        Add an item to the dataset with specified serialization format.

        Args:
            key: The key to store the value under
            value: The value to store
            fmt: Serialization format (SerializationFormat enum)
            proto_spec: Protobuf message class for GRPC format (required for GRPC)
        """
        if not isinstance(fmt, SerializationFormat):
            raise TypeError("fmt must be a SerializationFormat enum value")

        if fmt == SerializationFormat.GRPC and proto_spec is None:
            raise ValueError("proto_spec is required for GRPC format")

        self.data[key] = value
        self.formats[key] = fmt
        if proto_spec is not None:
            self.proto_specs[key] = proto_spec

    def add_batch(self, data_dict: dict[str, tuple]) -> None:
        """
        Add multiple items to the dataset at once using a dictionary.

        Args:
            data_dict: Dictionary where keys are the dataset keys and values are tuples
                      of (data, fmt) or (data, fmt, proto_spec) where fmt is a SerializationFormat enum

        Example:
            dataset.add_batch({
                "model_weights": (model.state_dict(), SerializationFormat.CLOUDPICKLE),
                "config": (config_data, SerializationFormat.MSGPACK),
                "traces": (trace_data, SerializationFormat.GRPC, TracesData)
            })
        """
        if not isinstance(data_dict, dict):
            raise TypeError("data_dict must be a dictionary")

        # Validate all entries first before adding any
        for key, value in data_dict.items():
            if not isinstance(value, tuple) or len(value) not in (2, 3):
                raise ValueError(
                    f"Value for key '{key}' must be a tuple of (data, fmt) or (data, fmt, proto_spec)"
                )

            if len(value) == 2:
                data, fmt = value
                proto_spec = None
            else:
                data, fmt, proto_spec = value

            if not isinstance(fmt, SerializationFormat):
                raise TypeError(
                    f"Format for key '{key}' must be a SerializationFormat enum value"
                )

        # Add all items
        for key, value in data_dict.items():
            if len(value) == 2:
                data, fmt = value
                self.add(key, data, fmt)
            else:
                data, fmt, proto_spec = value
                self.add(key, data, fmt, proto_spec)

    def remove(self, key: str) -> None:
        """Remove an item from the dataset."""
        if key in self.data:
            del self.data[key]
        if key in self.formats:
            del self.formats[key]
        if key in self.proto_specs:
            del self.proto_specs[key]

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
            proto_spec = self.proto_specs.get(key)
            serialized = self._serialize(value, self.formats[key], proto_spec)
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
            proto_spec = other.proto_specs.get(key)
            self.add(key, value, other.formats[key], proto_spec)

    def _serialize(
        self, obj: Any, fmt: SerializationFormat, proto_spec: Any = None
    ) -> bytes:
        """Serialize an object based on the selected format."""
        if fmt == SerializationFormat.MSGPACK:
            return msgpack.packb(obj, use_bin_type=True)
        if fmt == SerializationFormat.CLOUDPICKLE:
            return cloudpickle.dumps(obj)
        if fmt == SerializationFormat.GRPC:
            if proto_spec is None:
                raise ValueError("proto_spec is required for GRPC serialization")
            # obj should be a protobuf message instance or convertible to one
            if hasattr(obj, "SerializeToString"):
                return obj.SerializeToString()
            # Try to create protobuf message from obj data
            message = proto_spec()
            if hasattr(message, "CopyFrom") and hasattr(obj, "__dict__"):
                # Handle simple object conversion
                for key, value in obj.__dict__.items():
                    if hasattr(message, key):
                        setattr(message, key, value)
            else:
                # Direct assignment for simple types
                message = obj
            return message.SerializeToString()
        raise ValueError(f"Unsupported serialization format: {fmt}")

    def _deserialize(
        self, data: bytes, fmt: SerializationFormat, proto_spec: Any = None
    ) -> Any:
        """Deserialize data based on the selected format."""
        if fmt == SerializationFormat.MSGPACK:
            return msgpack.unpackb(data, raw=False)
        if fmt == SerializationFormat.CLOUDPICKLE:
            return cloudpickle.loads(data)
        if fmt == SerializationFormat.GRPC:
            if proto_spec is None:
                raise ValueError("proto_spec is required for GRPC deserialization")
            message = proto_spec()
            message.ParseFromString(data)
            return message
        raise ValueError(f"Unsupported serialization format: {fmt}")

    def _compress(self, data: bytes) -> bytes:
        """Compress serialized data."""
        return gzip.compress(data, compresslevel=self.compression_level)

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        return gzip.decompress(data)

    def save(self, directory: pathlib.Path):
        """
        Clean the directory and save the dataset to the specified directory.

        Args:
            directory: Directory path where the dataset will be saved
        """
        # Delete existing directory if it exists
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Verify all items have formats
        missing_formats = [key for key in self.data if key not in self.formats]
        if missing_formats:
            raise ValueError(
                f"Missing format specification for keys: {missing_formats}"
            )

        # Convert enum values to strings for JSON serialization
        formats_serializable = {k: v.value for k, v in self.formats.items()}
        # Convert proto specs to module.class names for JSON serialization
        proto_specs_serializable = {}
        for k, v in self.proto_specs.items():
            if v is not None:
                # Get the actual imported module name from the current context
                full_module_name = None

                # Check if this is a protobuf class by looking at the qualified name
                if hasattr(v, "__module__") and "pb2" in v.__module__:
                    # Try to find the full module path by checking sys.modules
                    import sys

                    for module_name, module in sys.modules.items():
                        if (
                            hasattr(module, v.__name__)
                            and getattr(module, v.__name__) is v
                        ):
                            full_module_name = module_name
                            break

                if full_module_name:
                    proto_specs_serializable[k] = f"{full_module_name}.{v.__name__}"
                else:
                    proto_specs_serializable[k] = f"{v.__module__}.{v.__name__}"

        # Save metadata
        metadata = {
            "compression_level": self.compression_level,
            "keys": list(self.data.keys()),
            "formats": formats_serializable,
            "proto_specs": proto_specs_serializable,
        }

        with (directory / "metadata.json").open("w") as f:
            json.dump(metadata, f)

        # Create data directory
        data_dir = directory / "data"
        if not data_dir.exists():
            data_dir.mkdir()

        # Save each item separately
        for key, value in self.data.items():
            format_to_use = self.formats[key]
            proto_spec = self.proto_specs.get(key)
            serialized = self._serialize(value, format_to_use, proto_spec)
            compressed = self._compress(serialized)

            with (data_dir / key).open("wb") as f:
                f.write(compressed)

    @classmethod
    def load(cls, directory: pathlib.Path) -> "CompressedDataset":
        """
        Load a dataset from the specified directory.

        Args:
            directory: Directory path where the dataset is stored

        Returns:
            Loaded CompressedDataset instance
        """
        # Load metadata
        with (directory / "metadata.json").open() as f:
            metadata = json.load(f)

        # Create new instance
        dataset = cls(compression_level=metadata["compression_level"])

        # Convert string format values back to enum values
        formats_str = metadata["formats"]
        formats_enum = {k: SerializationFormat(v) for k, v in formats_str.items()}
        dataset.formats = formats_enum

        # Convert proto spec strings back to classes
        proto_specs_str = metadata.get("proto_specs", {})
        proto_specs = {}
        for k, v in proto_specs_str.items():
            if v:
                # Import the class from module.class string
                module_name, class_name = v.rsplit(".", 1)
                import importlib

                module = importlib.import_module(module_name)
                proto_specs[k] = getattr(module, class_name)
        dataset.proto_specs = proto_specs

        keys = metadata["keys"]

        # Load data
        data_dir = directory / "data"
        for file_path in data_dir.iterdir():
            filename = file_path.name
            if filename in keys:
                with file_path.open("rb") as f:
                    compressed = f.read()

                decompressed = dataset._decompress(compressed)
                format_to_use = dataset.formats[filename]
                proto_spec = dataset.proto_specs.get(filename)
                value = dataset._deserialize(decompressed, format_to_use, proto_spec)
                dataset.data[filename] = value

        return dataset
