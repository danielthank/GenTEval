import logging
import uuid
from collections import defaultdict
from random import choice, random

import pandas as pd
from rdt.transformers import LogitScaler, LogScaler
from sdv.metadata import Metadata

from genteval.compressors import CompressedDataset, SerializationFormat
from genteval.dataset import Dataset

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
    candidate = defaultdict(list)
    processed_idx = set()
    # Process root first
    for idx, row in df.iterrows():
        if not row["parentChild"].startswith("root#"):
            continue
        parent_node_name, child_node_name = row["parentChild"].split("#")
        trace_id = _get_random_trace_id() + "_naive"
        span_id = _get_random_span_id() + "_root"
        dataset.traces[trace_id] = {
            span_id: {
                "nodeName": child_node_name,
                "startTime": int(row["startTime"] + random() * (60 * 1000000)),
                "duration": row["parentDuration"],
                "statusCode": None,
                "parentSpanId": None,
            }
        }
        candidate[child_node_name].append((trace_id, span_id))
        processed_idx.add(idx)

    # Try to expand the tree
    while True:
        current_pass_processed_idx = []
        unprocessed_df = df[~df.index.isin(processed_idx)]
        if unprocessed_df.empty:
            break
        print(f"Unprocessed edges: {len(unprocessed_df)}")
        for idx, row in unprocessed_df.iterrows():
            parent_node_name, child_node_name = row["parentChild"].split("#")
            if parent_node_name not in candidate or not candidate[parent_node_name]:
                continue
            trace_id, parent_span_id = choice(candidate[parent_node_name])
            child_span_id = _get_random_span_id() + "_non_root"
            parent_start_time = dataset.traces[trace_id][parent_span_id]["startTime"]
            parent_duration = dataset.traces[trace_id][parent_span_id]["duration"]
            gap_from_parent = row["gapFromParentRatio"] * parent_duration
            child_duration = row["childDurationRatio"] * parent_duration
            dataset.traces[trace_id][child_span_id] = {
                "nodeName": child_node_name,
                "startTime": parent_start_time + gap_from_parent,
                "duration": child_duration,
                "statusCode": None,
                "parentSpanId": parent_span_id,
            }
            candidate[child_node_name].append((trace_id, child_span_id))
            current_pass_processed_idx.append(idx)
        if not current_pass_processed_idx:
            break
        processed_idx.update(current_pass_processed_idx)

    # Process unprocessed edges as orphan edges
    unprocessed_df = df[~df.index.isin(processed_idx)]
    for _, row in unprocessed_df.iterrows():
        parent_node_name, child_node_name = row["parentChild"].split("#")
        trace_id = _get_random_trace_id() + "_naive_orphan"
        parent_span_id = _get_random_span_id() + "_parent"
        child_span_id = _get_random_span_id() + "_child"
        parent_start_time = int(row["startTime"] + random() * (60 * 1000000))
        dataset.traces[trace_id] = {
            child_span_id: {
                "nodeName": child_node_name,
                "startTime": int(
                    parent_start_time
                    + row["gapFromParentRatio"] * row["parentDuration"]
                ),
                "duration": row["childDurationRatio"] * row["parentDuration"],
                "statusCode": None,
                "parentSpanId": None,
            },
        }
        """
        parent_span_id: {
            "nodeName": parent_node_name,
            "startTime": parent_start_time,
            "duration": row["parentDuration"],
            "statusCode": None,
            "parentSpanId": None,
        },
        """
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
                if span["parentSpanId"] is not None and span["parentSpanId"] in trace:
                    parent_span = trace[span["parentSpanId"]]
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
                else:
                    edge_dataset.append(
                        [
                            span["startTime"] // (60 * 1000000),
                            "#".join(["root", span["nodeName"]]),
                            span["duration"],
                            0.0,
                            1.0,
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
            num_rows=self.edge_count, max_tries_per_batch=500
        )
        return _df_to_dataset(dataset)
