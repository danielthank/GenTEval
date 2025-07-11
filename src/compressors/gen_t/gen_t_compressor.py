import logging

from ...dataset import Dataset
from .. import CompressedDataset
from .config import GenTConfig
from .graph_gen_t_compressor import GraphGenTCompressor
from .naive_gen_t_compressor import NaiveGenTCompressor


class GenTCompressor:
    def __init__(self, config: GenTConfig):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.gen_t_compressor = GraphGenTCompressor(config)
        self.naive_gen_t_compressor = NaiveGenTCompressor(config)

    def compress(self, dataset: Dataset) -> CompressedDataset:
        large_dataset = Dataset()
        large_dataset.traces = {}
        small_dataset = Dataset()
        small_dataset.traces = {}
        for traceId, trace in dataset.traces.items():
            if len(trace) > self.config.span_cnt_threshold:
                large_dataset.traces[traceId] = trace
            else:
                small_dataset.traces[traceId] = trace

        compressed_dataset = self.naive_gen_t_compressor.compress(large_dataset)
        compressed_dataset.extend(self.gen_t_compressor.compress(small_dataset))
        return compressed_dataset

    def decompress(self, compressed_dataset: CompressedDataset) -> Dataset:
        dataset = self.naive_gen_t_compressor.decompress(compressed_dataset)
        dataset.extend(self.gen_t_compressor.decompress(compressed_dataset))
        return dataset
