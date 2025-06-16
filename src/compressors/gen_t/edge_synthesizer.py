import logging
import uuid
from random import random

import pandas as pd
from rdt.transformers import LogitScaler, LogScaler
from sdv.metadata import Metadata

from compressors import CompressedDataset, SerializationFormat
from dataset import Dataset

from .config import GenTConfig
from .ctgan.gen_t_ctgan_synthesizer import GenTCTGANSynthesizer


def _get_random_trace_id():
    return "trace" + uuid.uuid4().hex


def _get_random_span_id():
    return "span" + uuid.uuid4().hex


def _df_to_dataset(df: pd.DataFrame) -> Dataset:
    print(df)
    dataset = Dataset()
    dataset.traces = {}
    for _, row in df.iterrows():
        trace_id = _get_random_trace_id()
        parent_span_id = _get_random_span_id() + "_parent"
        child_span_id = _get_random_span_id() + "_child"
        parent_node_name, child_node_name = row["parentChild"].split("#")
        parent_start_time = int((row["startTime"] + random()) * (60 * 1000000))
        parent_duration = row["parentDuration"]
        gap_from_parent = row["gapFromParentRatio"] * parent_duration
        child_duration = row["childDurationRatio"] * parent_duration
        dataset.traces[trace_id] = {
            parent_span_id: {
                "nodeName": parent_node_name,
                "startTime": parent_start_time,
                "duration": parent_duration,
                "statusCode": None,
                "parentSpanId": None,
            },
            child_span_id: {
                "nodeName": child_node_name,
                "startTime": parent_start_time + gap_from_parent,
                "duration": child_duration,
                "statusCode": None,
                "parentSpanId": parent_span_id,
            },
        }
    return dataset


class EdgeSynthesizer:
    def __init__(self, config: GenTConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.synthesizer = None
        self.edge_count = None

    def _get_edge_dataset(self, dataset: Dataset):
        column_names = [
            "startTime",
            "parentChild",
            "parentDuration",
            "gapFromParentRatio",
            "childDurationRatio",
        ]
        edge_dataset = []
        for trace in dataset.traces.values():
            for span in trace.values():
                if span["parentSpanId"] is not None:
                    parent_span = trace.get(span["parentSpanId"])
                    if parent_span:
                        parent_duration = parent_span["duration"]
                        gap_from_parent = span["startTime"] - parent_span["startTime"]
                        gap_from_parent = max(0, min(gap_from_parent, parent_duration))
                        child_duration = span["duration"]
                        child_duration = max(0, min(child_duration, parent_duration))
                        edge_dataset.append(
                            [
                                span["startTime"] // (60 * 1000000),
                                "#".join([parent_span["nodeName"], span["nodeName"]]),
                                parent_duration,
                                gap_from_parent / parent_duration,
                                child_duration / parent_duration,
                            ]
                        )
        edge_dataset = pd.DataFrame(edge_dataset, columns=column_names)
        edge_dataset = edge_dataset.astype(
            {
                "startTime": "int64",
                "parentChild": "string",
                "parentDuration": "int64",
                "gapFromParentRatio": "float64",
                "childDurationRatio": "float64",
            },
            copy=False,
        )
        return edge_dataset

    def _get_metadata(self):
        metadata = Metadata()
        metadata.add_table("transactions")
        metadata.add_column("startTime", sdtype="categorical")
        metadata.add_column("parentChild", sdtype="categorical")
        metadata.add_column("parentDuration", sdtype="numerical")
        metadata.add_column("gapFromParentRatio", sdtype="numerical")
        metadata.add_column("childDurationRatio", sdtype="numerical")
        return metadata

    def _get_customized_transformers(self):
        customized_transformer = {}
        customized_transformer["parentDuration"] = LogScaler(constant=-0.001)
        customized_transformer["gapFromParentRatio"] = LogitScaler()
        customized_transformer["childDurationRatio"] = LogitScaler()

        return customized_transformer

    def distill(self, dataset):
        edge_dataset = self._get_edge_dataset(dataset)
        self.edge_count = len(edge_dataset)

        self.logger.info("Training edge synthesizer")
        self.synthesizer = GenTCTGANSynthesizer(
            metadata=self._get_metadata(),
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
    def load(compressed_dataset: CompressedDataset) -> "EdgeSynthesizer":
        config = GenTConfig.from_dict(compressed_dataset["gen_t_config"])
        edge_synthesizer = EdgeSynthesizer(config)
        edge_synthesizer.synthesizer = compressed_dataset["edge_synthesizer"]
        edge_synthesizer.edge_count = compressed_dataset["edge_count"]
        return edge_synthesizer

    def synthesize(self):
        # Each edge is going to be synthesized as a trace with two spans
        dataset = self.synthesizer.sample(
            num_rows=self.edge_count // 2, max_tries_per_batch=500
        )
        return _df_to_dataset(dataset)
