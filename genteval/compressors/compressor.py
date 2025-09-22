import time
from abc import ABC, abstractmethod

from genteval.dataset import Dataset

from .compressed_dataset import CompressedDataset, SerializationFormat


class Compressor(ABC):
    """Abstract base class for all trace data compressors."""

    def compress(self, dataset: Dataset) -> CompressedDataset:
        """Compress a dataset into a CompressedDataset format with automatic timing measurement.

        Args:
            dataset: Input dataset containing trace data

        Returns:
            CompressedDataset containing compressed representation with timing info
        """
        start_time = time.time()
        result = self._compress_impl(dataset)
        compression_time = time.time() - start_time

        result.add(
            "compression_time_seconds", compression_time, SerializationFormat.MSGPACK
        )
        return result

    @abstractmethod
    def _compress_impl(self, dataset: Dataset) -> CompressedDataset:
        """Implementation-specific compression logic.

        Args:
            dataset: Input dataset containing trace data

        Returns:
            CompressedDataset containing compressed representation
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    def decompress(self, compressed_dataset: CompressedDataset) -> Dataset:
        """Decompress a CompressedDataset back into original Dataset format.

        Args:
            compressed_dataset: Input compressed dataset

        Returns:
            Dataset containing decompressed trace data with compression timing info
        """
        result = self._decompress_impl(compressed_dataset)

        # Recover compression time if available
        if "compression_time_seconds" in compressed_dataset:
            result.compression_time_seconds = compressed_dataset[
                "compression_time_seconds"
            ]
        else:
            result.compression_time_seconds = None

        return result

    @abstractmethod
    def _decompress_impl(self, compressed_dataset: CompressedDataset) -> Dataset:
        """Implementation-specific decompression logic.

        Args:
            compressed_dataset: Input compressed dataset

        Returns:
            Dataset containing decompressed trace data
        """
        raise NotImplementedError("This method should be implemented in the subclass")
