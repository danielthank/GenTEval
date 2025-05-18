import logging

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

from compressors import CompressedDataset, SerializationFormat
from compressors.trace import Trace
from dataset import Dataset

from .config import GenTConfig


class StartTimeSynthesizer:
    def __init__(self, config: GenTConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.synthesizer = None
        self.graph_count = None

    def _get_start_time_dataset(self, dataset: Dataset):
        column_names = ["graph", "startTime"]
        start_time_dataset = []
        for trace in dataset.traces.values():
            trace = Trace(trace)
            graph = str(trace.edges)
            if graph == "[]":  # TODO: support graph with no edges
                continue
            start_time = trace.start_time
            start_time_dataset.append([graph, start_time])
        return pd.DataFrame(start_time_dataset, columns=column_names)

    def distill(self, dataset):
        start_time_dataset = self._get_start_time_dataset(dataset)

        self.graph_count = start_time_dataset["graph"].value_counts().to_dict()
        metadata = Metadata()
        metadata.add_table("transactions")
        metadata.add_column("graph", sdtype="categorical")
        metadata.add_column("startTime", sdtype="numerical")
        self.logger.info("Training start time synthesizer")
        self.synthesizer = CTGANSynthesizer(
            metadata=metadata,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            generator_dim=self.config.generator_dim,
            discriminator_dim=self.config.discriminator_dim,
            verbose=True,
        )
        self.synthesizer.fit(start_time_dataset)

    def save(self, compressed_dataset: CompressedDataset):
        compressed_dataset.add(
            "start_time_synthesizer", self.synthesizer, SerializationFormat.CLOUDPICKLE
        )
        compressed_dataset.add(
            "graph_count", self.graph_count, SerializationFormat.MSGPACK
        )

    @staticmethod
    def load(compressed_dataset: CompressedDataset) -> "StartTimeSynthesizer":
        config = GenTConfig.from_dict(compressed_dataset["gen_t_config"])
        start_time_synthesizer = StartTimeSynthesizer(config)
        start_time_synthesizer.synthesizer = compressed_dataset[
            "start_time_synthesizer"
        ]
        start_time_synthesizer.graph_count = compressed_dataset["graph_count"]
        return start_time_synthesizer

    def synthesize(self):
        rows = []
        for graph, count in self.graph_count.items():
            rows.extend([{"graph": graph} for _ in range(count)])
        known_columns = pd.DataFrame(rows)
        return self.synthesizer.sample_remaining_columns(
            known_columns=known_columns, max_tries_per_batch=500
        )
