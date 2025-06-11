import random

from compressors import CompressedDataset, Compressor, SerializationFormat
from dataset import Dataset


class HeadSamplingCompressor(Compressor):
    def __init__(self, sampling_rate: int):
        super().__init__()
        self.sampling_rate = sampling_rate

    def compress(self, dataset: Dataset) -> CompressedDataset:
        trace_len = len(dataset.traces)
        if trace_len < self.sampling_rate:
            raise ValueError(
                f"Dataset length {trace_len} is less than the sampling rate {self.sampling_rate}."
            )
        compressed_dataset = CompressedDataset()
        compressed_dataset.add(
            "sampling_rate", self.sampling_rate, SerializationFormat.MSGPACK
        )

        # Ranomly sample the trace with probability of 1/sampling_rate
        sampled_traces = {
            trace_id: trace
            for trace_id, trace in dataset.traces.items()
            if random.random() < 1 / self.sampling_rate
        }

        compressed_dataset.add(
            "sampled_traces", sampled_traces, SerializationFormat.MSGPACK
        )

        return compressed_dataset

    def decompress(self, compressed_dataset: CompressedDataset) -> Dataset:
        dataset = Dataset()
        dataset.traces = compressed_dataset["sampled_traces"]
        return dataset
