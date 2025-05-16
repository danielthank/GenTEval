from compressors.compressed_dataset import CompressedDataset
from dataset import Dataset


class Compressor:
    def compress(self, dataset: Dataset) -> CompressedDataset:
        raise NotImplementedError("This method should be implemented in the subclass")

    def decompress(self, compressed_dataset: CompressedDataset):
        raise NotImplementedError("This method should be implemented in the subclass")
