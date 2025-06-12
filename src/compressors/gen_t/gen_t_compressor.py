import logging

from compressors import CompressedDataset, Compressor, SerializationFormat
from dataset import Dataset

from .config import GenTConfig
from .metadata import MetadataSynthesizer
from .start_time import StartTimeSynthesizer


class GenTCompressor(Compressor):
    def __init__(self, config: GenTConfig):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config

    def compress(self, dataset: Dataset) -> CompressedDataset:
        compressed_dataset = CompressedDataset()
        compressed_dataset.add(
            "gen_t_config", self.config.as_dict(), SerializationFormat.MSGPACK
        )

        # TODO: split dataset into 2 parts: one for spans <= 10 and one for spans > 10

        start_time_synthesizer = StartTimeSynthesizer(config=self.config)
        metadata_synthesizer = MetadataSynthesizer(config=self.config)

        logging.info("Distilling start_time")
        start_time_synthesizer.distill(dataset)
        start_time_synthesizer.save(compressed_dataset)

        logging.info("Distilling metadata")
        metadata_synthesizer.distill(dataset)
        metadata_synthesizer.save(compressed_dataset)

        return compressed_dataset

    def decompress(self, compressed_dataset: CompressedDataset) -> Dataset:
        logging.info("Synthesizing start_time")
        start_time_synthesizer = StartTimeSynthesizer.load(compressed_dataset)
        start_time = start_time_synthesizer.synthesize()

        logging.info("Synthesizing metadata")
        metadata_synthesizer = MetadataSynthesizer.load(compressed_dataset)
        return metadata_synthesizer.synthesize(start_time)
