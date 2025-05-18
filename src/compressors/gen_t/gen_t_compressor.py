from compressors import CompressedDataset, Compressor
from dataset import Dataset

from .config import GenTConfig
from .metadata import MetadataSynthesizer
from .start_time import StartTimeSynthesizer


class GenTCompressor(Compressor):
    def __init__(self, config: GenTConfig):
        super().__init__()
        self.config = config

    def compress(self, dataset: Dataset) -> CompressedDataset:
        compressed_dataset = CompressedDataset()
        compressed_dataset.add("gen_t_config", self.config.to_dict())

        start_time_synthesizer = StartTimeSynthesizer(config=self.config)
        metadata_synthesizer = MetadataSynthesizer(config=self.config)

        start_time_synthesizer.distill(dataset)
        start_time_synthesizer.save(compressed_dataset)

        metadata_synthesizer.distill(dataset)
        metadata_synthesizer.save(compressed_dataset)

        return compressed_dataset

    def decompress(self, compressed_dataset: CompressedDataset) -> Dataset:
        start_time_synthesizer = StartTimeSynthesizer.load(compressed_dataset)
        start_time = start_time_synthesizer.synthesize()
        print("start_time", start_time)
