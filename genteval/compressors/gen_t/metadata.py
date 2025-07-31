import copy
import logging
import uuid
from collections import Counter, defaultdict

import pandas as pd
from rdt.transformers import LogScaler
from sdv.metadata import Metadata
from tqdm import tqdm

from genteval.compressors import CompressedDataset, SerializationFormat
from genteval.compressors.trace import Trace
from genteval.dataset import Dataset

from .config import GenTConfig
from .ctgan.gen_t_ctgan_synthesizer import GenTCTGANSynthesizer


def _get_random_trace_id():
    return "trace" + uuid.uuid4().hex


def _get_random_span_id():
    return "span" + uuid.uuid4().hex


def _df_to_dataset(df: pd.DataFrame) -> Dataset:
    dataset = Dataset()
    dataset.traces = {}
    for trace_id, chains in df.groupby("traceId"):
        spans_dict = {}
        for chain in chains.itertuples(index=False):
            if chain.isRoot:
                node_start = 0
            else:
                node_start = 1

            nodes = chain.chain.split("#")
            for i in range(node_start, len(nodes)):
                node = nodes[i]
                if i > 0:
                    # has parent
                    parent_node = nodes[i - 1]
                    spans_dict[node] = {
                        "spanId": _get_random_span_id(),
                        "nodeName": node,
                        "startTime": spans_dict[parent_node]["startTime"]
                        + getattr(chain, f"gapFromParent_{i}"),
                        "duration": getattr(chain, f"duration_{i}"),
                        "statusCode": None,
                        "parentSpanId": spans_dict[parent_node]["spanId"],
                    }
                else:
                    # root node
                    spans_dict[node] = {
                        "spanId": _get_random_span_id(),
                        "nodeName": node,
                        "startTime": chain.startTime,
                        "duration": getattr(chain, f"duration_{i}"),
                        "statusCode": None,
                        "parentSpanId": None,
                    }
        # change the key of spans_dict to spanId
        # remove the spanId from the value
        spans_dict = {
            span["spanId"]: {k: v for k, v in span.items() if k != "spanId"}
            for span in spans_dict.values()
        }
        dataset.traces[trace_id] = spans_dict
    return dataset


class MetadataSynthesizer:
    def __init__(self, config: GenTConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.root_synthesizer = None
        self.chain_synthesizer = None
        self.graph_to_chain_count = None

    def _get_sdv_metadata(self):
        metadata = Metadata()
        metadata.add_table("metadata")
        metadata.add_column("graph", sdtype="categorical")
        metadata.add_column("chain", sdtype="categorical")
        for i in range(self.config.chain_length):
            metadata.add_column(f"gapFromParent_{i}", sdtype="numerical")
            metadata.add_column(f"duration_{i}", sdtype="numerical")
        return metadata

    def _get_metadata_dataset(self, dataset: Dataset):
        root_dataset = []
        chain_dataset = []
        columns = ["graph", "chain"]
        dtypes = {"graph": "string", "chain": "string"}
        for i in range(self.config.chain_length):
            columns.append(f"gapFromParent_{i}")
            columns.append(f"duration_{i}")
            dtypes[f"gapFromParent_{i}"] = "int64"
            dtypes[f"duration_{i}"] = "int64"

        for trace_id, trace in dataset.traces.items():
            trace = Trace(trace)
            # Skip traces that have no edges or too many edges (Out of Memory)
            if len(trace) <= 1 or len(trace) > 10:
                continue

            graph = trace.graph
            if not self.graph_to_chain_count:
                self.graph_to_chain_count = {}
            if graph not in self.graph_to_chain_count:
                self.graph_to_chain_count[graph] = defaultdict(int)

            root_chains = set()
            chain_chains = set()
            for chain in trace.chains(self.config.chain_length):
                chain_str = "#".join(
                    [trace.unique_name(span_id) for span_id in chain["chain"]]
                )
                row = [
                    trace.graph,
                    chain_str,
                ]
                for idx in range(self.config.chain_length):
                    if idx < len(chain["chain"]):
                        span_id = chain["chain"][idx]
                        row.extend(
                            [
                                trace.gap_from_parent(span_id),
                                trace.duration(span_id),
                            ]
                        )
                    else:
                        row.extend([0, 0])  # TODO: NaN or 0?
                if chain["is_root"]:
                    root_dataset.append(row)
                    root_chains.add(chain_str)
                else:
                    chain_dataset.append(row)
                    chain_chains.add(chain_str)
            root_chains = sorted(root_chains)
            chain_chains = sorted(chain_chains)
            chains = {"root": root_chains, "chain": chain_chains}
            self.graph_to_chain_count[graph][str(chains)] += 1

        root = pd.DataFrame(root_dataset, columns=columns)
        chain = pd.DataFrame(chain_dataset, columns=columns)
        root.fillna(dict.fromkeys(columns[2:], 0), inplace=True)
        root = root.astype(dtypes, copy=False)
        chain.fillna(dict.fromkeys(columns[2:], 0), inplace=True)
        chain = chain.astype(dtypes, copy=False)

        return {
            "root": root,
            "chain": chain,
        }

    def _get_customized_transformers(self):
        customized_transformer = {}
        for i in range(self.config.chain_length):
            customized_transformer[f"gapFromParent_{i}"] = LogScaler(constant=-0.001)
            customized_transformer[f"duration_{i}"] = LogScaler(constant=-0.001)
        return customized_transformer

    def distill(self, dataset):
        metadata_dataset = self._get_metadata_dataset(dataset)
        sdv_metadata = self._get_sdv_metadata()

        self.logger.info("Training root synthesizer")
        self.root_synthesizer = GenTCTGANSynthesizer(
            metadata=sdv_metadata,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            generator_dim=self.config.generator_dim,
            discriminator_dim=self.config.discriminator_dim,
            enforce_rounding=False,
            verbose=True,
        )
        self.root_synthesizer.auto_assign_transformers(metadata_dataset["root"])
        self.root_synthesizer.update_transformers(self._get_customized_transformers())
        self.root_synthesizer.fit(metadata_dataset["root"])

        self.logger.info("Training chain synthesizer")
        self.chain_synthesizer = GenTCTGANSynthesizer(
            metadata=sdv_metadata,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            generator_dim=self.config.generator_dim,
            discriminator_dim=self.config.discriminator_dim,
            enforce_rounding=False,
            verbose=True,
        )
        self.chain_synthesizer.auto_assign_transformers(metadata_dataset["chain"])
        self.chain_synthesizer.update_transformers(self._get_customized_transformers())
        self.chain_synthesizer.fit(metadata_dataset["chain"])

    def save(self, compressed_dataset: CompressedDataset):
        compressed_dataset.add(
            "graph_to_chain_count",
            self.graph_to_chain_count,
            SerializationFormat.MSGPACK,
        )
        self.root_synthesizer.trim()
        compressed_dataset.add(
            "root_synthesizer",
            self.root_synthesizer,
            SerializationFormat.CLOUDPICKLE,
        )
        self.chain_synthesizer._data_processor._transformers_by_sdtype = None
        self.chain_synthesizer.trim()
        compressed_dataset.add(
            "chain_synthesizer",
            self.chain_synthesizer,
            SerializationFormat.CLOUDPICKLE,
        )

    @staticmethod
    def load(compressed_dataset: CompressedDataset) -> "MetadataSynthesizer":
        config = GenTConfig.from_dict(compressed_dataset["gen_t_config"])
        metadata_synthesizer = MetadataSynthesizer(config)
        metadata_synthesizer.graph_to_chain_count = compressed_dataset[
            "graph_to_chain_count"
        ]
        for graph in metadata_synthesizer.graph_to_chain_count:
            metadata_synthesizer.graph_to_chain_count[graph] = Counter(
                metadata_synthesizer.graph_to_chain_count[graph]
            )
        metadata_synthesizer.root_synthesizer = compressed_dataset["root_synthesizer"]
        metadata_synthesizer.chain_synthesizer = compressed_dataset["chain_synthesizer"]
        return metadata_synthesizer

    def synthesize(self, start_time: pd.DataFrame) -> Dataset:
        graph_to_chain_left = copy.deepcopy(self.graph_to_chain_count)
        root_to_synthesize = {}
        chain_to_synthesize = defaultdict(list)
        total_chains = 0
        for row in start_time.itertuples(index=False):
            graph = row.graph
            start_time = row.startTime
            chains_chosen = graph_to_chain_left[graph].most_common(1)[0]
            graph_to_chain_left[graph].subtract(chains_chosen)
            chains_chosen = eval(chains_chosen[0])
            trace_id = _get_random_trace_id()
            root_to_synthesize[trace_id, start_time, graph] = chains_chosen["root"]

            for chain in chains_chosen["chain"]:
                trigger = chain.split("#")[0]
                chain_to_synthesize[trace_id, start_time, graph, trigger].append(chain)
                total_chains += 1

        root_known = []
        for (trace_id, start_time, graph), chains in root_to_synthesize.items():
            for chain in chains:
                row = [trace_id, start_time, graph, chain]
                root_known.append(row)
        root_known = pd.DataFrame(
            root_known,
            columns=["traceId", "startTime", "graph", "chain"],
        )
        chain_sampled = self.root_synthesizer.sample_remaining_columns(
            root_known,
            max_tries_per_batch=500,
            condition_columns=["graph", "chain"],
        )
        chain_sampled["isRoot"] = True
        all_chains = chain_sampled.copy()

        with tqdm(total=total_chains, desc="Synthesizing chains") as progress_bar:
            while chain_to_synthesize:
                chain_known = []
                for trigger_chain in chain_sampled.itertuples(index=False):
                    trace_id = trigger_chain.traceId
                    graph = trigger_chain.graph
                    start_time = trigger_chain.startTime
                    for trigger_idx, trigger in enumerate(
                        trigger_chain.chain.split("#")
                    ):
                        # for each trigger check if there are chains to synthesize
                        if (
                            trace_id,
                            start_time,
                            graph,
                            trigger,
                        ) in chain_to_synthesize:
                            for chain in chain_to_synthesize[
                                trace_id, start_time, graph, trigger
                            ]:
                                row = [
                                    trace_id,
                                    start_time,
                                    graph,
                                    chain,
                                ]
                                trigger_gap_from_parent = getattr(
                                    trigger_chain, f"gapFromParent_{trigger_idx}"
                                )

                                trigger_duration = getattr(
                                    trigger_chain, f"duration_{trigger_idx}"
                                )
                                row.extend([trigger_gap_from_parent, trigger_duration])
                                chain_known.append(row)
                            del chain_to_synthesize[
                                trace_id, start_time, graph, trigger
                            ]
                chain_known = pd.DataFrame(
                    chain_known,
                    columns=[
                        "traceId",
                        "startTime",
                        "graph",
                        "chain",
                        "gapFromParent_0",
                        "duration_0",
                    ],
                )
                chain_sampled = self.chain_synthesizer.sample_remaining_columns(
                    chain_known,
                    max_tries_per_batch=500,
                    condition_columns=[
                        "graph",
                        "chain",
                    ],  # TODO: condition on startTime, gapFromParent_0 and duration_0, continuous
                    progress_bar=progress_bar,
                )
                chain_sampled["isRoot"] = False
                if chain_sampled.empty:
                    # no new chains sampled
                    break

                all_chains = pd.concat(
                    [all_chains, chain_sampled.copy()], ignore_index=True, copy=False
                )

        return _df_to_dataset(all_chains)
