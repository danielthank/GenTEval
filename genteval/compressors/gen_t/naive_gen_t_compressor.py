from genteval.compressors import CompressedDataset, SerializationFormat
from genteval.dataset import Dataset

from .config import GenTConfig
from .edge_synthesizer import EdgeSynthesizer


class NaiveGenTCompressor:
    def __init__(self, config: GenTConfig):
        self.config = config

    def compress(self, data: Dataset) -> CompressedDataset:
        compressed_dataset = CompressedDataset()
        compressed_dataset.add(
            "gen_t_config", self.config.as_dict(), SerializationFormat.MSGPACK
        )

        edge_synthesizer = EdgeSynthesizer(config=self.config)

        edge_synthesizer.distill(data)
        edge_synthesizer.save(compressed_dataset)
        return compressed_dataset

    def decompress(self, compressed_data: CompressedDataset) -> Dataset:
        edge_synthesizer = EdgeSynthesizer.load(compressed_data)
        return edge_synthesizer.synthesize()
