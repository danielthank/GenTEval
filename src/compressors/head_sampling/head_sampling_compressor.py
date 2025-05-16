from dataset import Dataset
from compressors import Compressor, CompressedDataset

class HeadSamplingCompressedDataset(CompressedDataset):

    def get_size(self) -> int:
        return 0

class HeadSamplingCompressor(Compressor):

    def __init__(self):
        super().__init__()

    def compress(self, dataset: Dataset) -> HeadSamplingCompressedDataset:
        super().compress(dataset)
        pass

    def decompress(self, compressed_dataset: HeadSamplingCompressedDataset) -> Dataset:
        pass
