import logging
import uuid
from random import random

import pandas as pd
from rdt.transformers import LogScaler
from sdv.metadata import Metadata

from ...dataset import Dataset
from .. import CompressedDataset, SerializationFormat
from .config import GenTConfig
from .ctgan.gen_t_ctgan_synthesizer import GenTCTGANSynthesizer


def _get_random_trace_id():
    return "trace" + uuid.uuid4().hex


def _get_random_span_id():
    return "span" + uuid.uuid4().hex


def _df_to_dataset(df: pd.DataFrame) -> Dataset:
    dataset = Dataset()
    dataset.traces = {}
    for _, row in df.iterrows():
        trace_id = _get_random_trace_id()
        span_id = _get_random_span_id()
        dataset.traces[trace_id] = {
            span_id: {
                "nodeName": row["nodeName"],
                "startTime": int((row["startTime"] + random()) * (60 * 1000000)),
                "duration": row["duration"],
                "statusCode": None,
                "parentSpanId": None,
            }
        }
    return dataset


class NodeSynthesizer:
    def __init__(self, config: GenTConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.synthesizer = None
        self.edge_count = None

    def _get_edge_dataset(self, dataset: Dataset):
        column_names = ["startTime", "nodeName", "duration"]
        edge_dataset = []
        for trace in dataset.traces.values():
            for span in trace.values():
                edge_dataset.append(
                    [
                        span["startTime"] // (60 * 1000000),
                        span["nodeName"],
                        span["duration"],
                    ]
                )
        # min of gapFromParent
        edge_dataset = pd.DataFrame(edge_dataset, columns=column_names)
        edge_dataset = edge_dataset.astype(
            {
                "startTime": "int64",
                "nodeName": "string",
                "duration": "int64",
            },
            copy=False,
        )
        return edge_dataset

    def _get_customized_transformers(self):
        customized_transformer = {}
        customized_transformer["duration"] = LogScaler(constant=-0.001)
        return customized_transformer

    def distill(self, dataset):
        edge_dataset = self._get_edge_dataset(dataset)
        self.edge_count = len(edge_dataset)

        metadata = Metadata()
        metadata.add_table("transactions")
        metadata.add_column("startTime", sdtype="categorical")
        metadata.add_column("nodeName", sdtype="categorical")
        metadata.add_column("duration", sdtype="numerical")
        self.logger.info("Training edge synthesizer")
        self.synthesizer = GenTCTGANSynthesizer(
            metadata=metadata,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            generator_dim=self.config.generator_dim,
            discriminator_dim=self.config.discriminator_dim,
            verbose=True,
        )
        self.synthesizer.auto_assign_transformers(edge_dataset)
        self.synthesizer.update_transformers(self._get_customized_transformers())
        self.synthesizer.fit(edge_dataset)

    def save(self, compressed_dataset: CompressedDataset):
        self.synthesizer.trim()
        compressed_dataset.add(
            "edge_synthesizer", self.synthesizer, SerializationFormat.CLOUDPICKLE
        )
        compressed_dataset.add(
            "edge_count", self.edge_count, SerializationFormat.MSGPACK
        )

    @staticmethod
    def load(compressed_dataset: CompressedDataset) -> "NodeSynthesizer":
        config = GenTConfig.from_dict(compressed_dataset["gen_t_config"])
        edge_synthesizer = NodeSynthesizer(config)
        edge_synthesizer.synthesizer = compressed_dataset["edge_synthesizer"]
        edge_synthesizer.edge_count = compressed_dataset["edge_count"]
        return edge_synthesizer

    def synthesize(self):
        dataset = self.synthesizer.sample(
            num_rows=self.edge_count, max_tries_per_batch=500
        )
        return _df_to_dataset(dataset)
