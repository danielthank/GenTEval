import pathlib

from compressors import CompressedDataset, Compressor
from dataset import Dataset

from .config import GenTConfig
from .metadata import MetadataSynthesizer
from .start_time import StartTimeSynthesizer


class GenTCompressedDataset(CompressedDataset):
    def get_size(self) -> int:
        return 0

    def save(self, dir):
        pass


class GenTCompressor(Compressor):
    def __init__(self, config: GenTConfig):
        super().__init__()
        self.config = config
        self.start_time_synthesizer = StartTimeSynthesizer(config=config)
        self.metadata_synthesizer = MetadataSynthesizer(config=config)

    def compress(self, dataset: Dataset) -> GenTCompressedDataset:
        self.start_time_synthesizer.distill(dataset)
        self.metadata_synthesizer.distill(dataset)

    def save(self, dir: pathlib.Path):
        self.start_time_synthesizer.save(dir / "start_time")
        self.metadata_synthesizer.save(dir / "metadata")

    def decompress(self, compressed_dataset: GenTCompressedDataset) -> Dataset:
        pass
