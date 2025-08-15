import logging
import multiprocessing as mp
import uuid
from collections import defaultdict

import cloudpickle
import numpy as np
from tqdm import tqdm

from genteval.compressors import CompressedDataset, SerializationFormat
from genteval.compressors.trace import Trace
from genteval.dataset import Dataset
from genteval.utils.data_structures import count_spans_per_tree

from .config import MarkovGenTConfig, MetadataModel, RootModel
from .metadata_vae import MetadataSynthesizer as MetadataVAESynthesizer
from .mrf_graph import MarkovRandomField
from .root import (
    RootDurationMLPSynthesizer,
    RootDurationTableSynthesizer,
    RootDurationVAESynthesizer,
)
from .start_time_count import StartTimeCountSynthesizer


def _get_random_trace_id():
    return "trace" + uuid.uuid4().hex


def _get_random_span_id():
    return "span" + uuid.uuid4().hex


def _print_tree_dfs(node, depth=0, prefix=""):
    """Print tree structure using DFS with indentation to show depth."""
    if node is None:
        return

    # Print current node with indentation, depth, and child count
    indent = "  " * depth
    child_count = len(node.children)
    print(f"{indent}{prefix}{node.node_name} [depth={depth}, children={child_count}]")

    # Print children
    for i, child in enumerate(node.children):
        child_prefix = "├─ " if i < len(node.children) - 1 else "└─ "
        _print_tree_dfs(child, depth + 1, child_prefix)


def _generate_traces_worker(args):
    """Worker function for parallel trace generation."""
    (
        num_traces_per_worker,
        start_times_chunk,
        serialized_compressed_dataset,
    ) = args

    # Deserialize the compressed dataset in this process
    compressed_dataset = cloudpickle.loads(serialized_compressed_dataset)

    # Load the compressor with all models using the existing load method
    compressor = MarkovGenTCompressor.load(compressed_dataset, num_processes=1)

    # Generate traces for this worker
    traces = {}
    batch_size = min(64, max(1, num_traces_per_worker // 5))

    for i in range(0, num_traces_per_worker, batch_size):
        batch_end = min(i + batch_size, num_traces_per_worker)
        current_batch_size = batch_end - i

        # Collect batch data for cross-trace processing
        batch_traces = []

        # Generate tree structures for all traces in batch
        for j in range(current_batch_size):
            trace_idx = i + j
            if trace_idx < len(start_times_chunk):
                start_time = float(start_times_chunk[trace_idx])

                # Generate tree structure using Markov Random Field
                root_node = compressor.depth_markov_chain.generate_tree_structure(
                    max_nodes=10000
                )

                """
                # Print the generated tree structure
                if root_node:
                    print(
                        f"\n=== Generated Tree Structure for Trace {trace_idx + 1} ==="
                    )
                    _print_tree_dfs(root_node)
                    print("=" * 50)
                """

                if root_node:
                    trace_id = _get_random_trace_id()
                    batch_traces.append(
                        {
                            "trace_id": trace_id,
                            "start_time": start_time,
                            "root_node": root_node,
                            "spans": {},
                            "node_to_span_id": {},
                            "spans_by_level": defaultdict(list),
                        }
                    )

        # Traverse all trees and organize spans by level across all traces
        for trace_data in batch_traces:

            def traverse_tree_by_level(
                data, node, depth=0, parent_span_id=None, child_idx=0, parent_node=None
            ):
                span_id = _get_random_span_id()
                data["node_to_span_id"][node] = span_id

                # Calculate normalized child index
                if parent_node and len(parent_node.children) > 0:
                    normalized_child_idx = child_idx / len(parent_node.children)
                else:
                    normalized_child_idx = 0  # For root node or edge cases

                data["spans_by_level"][depth].append(
                    (span_id, node, parent_span_id, normalized_child_idx)
                )

                # Traverse children at next depth level
                for i, child in enumerate(node.children):
                    traverse_tree_by_level(data, child, depth + 1, span_id, i, node)

            # Start traversal from root
            traverse_tree_by_level(
                trace_data, trace_data["root_node"], parent_node=None
            )

        # Find maximum depth across all traces
        max_depth = 0
        for trace_data in batch_traces:
            if trace_data["spans_by_level"]:
                max_depth = max(max_depth, *trace_data["spans_by_level"].keys())

        # Process spans level by level across all traces for mega-batching
        for level in range(max_depth + 1):
            # Collect all spans at this level across all traces
            if level == 0:
                # Collect root spans from all traces
                all_root_start_times = []
                all_root_node_names = []
                all_root_child_counts = []
                root_span_mapping = []  # (trace_idx, span_id, node)

                for trace_idx, trace_data in enumerate(batch_traces):
                    if level in trace_data["spans_by_level"]:
                        for span_id, node, _parent_span_id, _child_idx in trace_data[
                            "spans_by_level"
                        ][level]:
                            all_root_start_times.append(trace_data["start_time"])
                            all_root_node_names.append(node.node_name)
                            all_root_child_counts.append(len(node.children))
                            root_span_mapping.append((trace_idx, span_id, node))

                # Batch process all root spans across all traces
                if all_root_start_times:
                    root_durations = compressor.root_duration_synthesizer.synthesize_root_duration_batch(
                        all_root_start_times, all_root_node_names, all_root_child_counts
                    )

                    # Create root spans
                    for (trace_idx, span_id, node), duration in zip(
                        root_span_mapping, root_durations, strict=False
                    ):
                        trace_data = batch_traces[trace_idx]
                        trace_data["spans"][span_id] = {
                            "nodeName": node.node_name,
                            "startTime": int(trace_data["start_time"]),
                            "duration": max(1, int(duration)),
                            "statusCode": None,
                            "parentSpanId": None,
                        }
            else:
                # Collect child spans from all traces at this level
                all_parent_start_times = []
                all_parent_durations = []
                all_normalized_child_indices = []
                all_parent_node_names = []
                all_child_node_names = []
                child_span_mapping = []  # (trace_idx, span_id, node, parent_span_id)

                for trace_idx, trace_data in enumerate(batch_traces):
                    if level in trace_data["spans_by_level"]:
                        for (
                            span_id,
                            node,
                            parent_span_id,
                            normalized_child_idx,
                        ) in trace_data["spans_by_level"][level]:
                            if parent_span_id in trace_data["spans"]:
                                parent_span = trace_data["spans"][parent_span_id]
                                all_parent_start_times.append(parent_span["startTime"])
                                all_parent_durations.append(parent_span["duration"])
                                all_normalized_child_indices.append(
                                    normalized_child_idx
                                )
                                all_parent_node_names.append(parent_span["nodeName"])
                                all_child_node_names.append(node.node_name)
                                child_span_mapping.append(
                                    (trace_idx, span_id, node, parent_span_id)
                                )

                # Batch process all child spans across all traces at this level
                if all_parent_start_times:
                    child_metadata = (
                        compressor.metadata_synthesizer.synthesize_metadata_batch(
                            all_parent_start_times,
                            all_parent_durations,
                            all_normalized_child_indices,
                            all_parent_node_names,
                            all_child_node_names,
                        )
                    )

                    # Create child spans
                    for (trace_idx, span_id, node, parent_span_id), (
                        child_start_time,
                        child_duration,
                    ) in zip(child_span_mapping, child_metadata, strict=False):
                        trace_data = batch_traces[trace_idx]
                        trace_data["spans"][span_id] = {
                            "nodeName": node.node_name,
                            "startTime": int(child_start_time),
                            "duration": max(1, int(child_duration)),
                            "statusCode": None,
                            "parentSpanId": parent_span_id,
                        }

        # Add completed traces to result
        for trace_data in batch_traces:
            if trace_data["spans"]:
                traces[trace_data["trace_id"]] = trace_data["spans"]

    return traces


class MarkovGenTCompressor:
    def __init__(self, config: MarkovGenTConfig = None, num_processes: int = 8):
        self.config = config or MarkovGenTConfig()
        self.num_processes = num_processes
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.start_time_synthesizer = StartTimeCountSynthesizer(self.config)
        self.depth_markov_chain = MarkovRandomField(
            order=self.config.markov_order,
            max_depth=self.config.max_depth,
            max_children=self.config.max_children,
            edge_approach=self.config.mrf_edge_approach,
        )

        # Choose root duration synthesis model
        if self.config.root_model == RootModel.MLP:
            self.root_duration_synthesizer = RootDurationMLPSynthesizer(self.config)
        elif self.config.root_model == RootModel.VAE:
            self.root_duration_synthesizer = RootDurationVAESynthesizer(self.config)
        elif self.config.root_model == RootModel.TABLE:
            self.root_duration_synthesizer = RootDurationTableSynthesizer(self.config)
        else:
            raise ValueError(f"Unknown root model: {self.config.root_model}")

        # Initialize metadata synthesizer based on config
        if self.config.metadata_model == MetadataModel.VAE:
            self.metadata_synthesizer = MetadataVAESynthesizer(self.config)
        else:
            raise ValueError(f"Unknown metadata model: {self.config.metadata_model}")

    def compress(self, dataset: Dataset) -> CompressedDataset:
        """Learn models from the dataset."""
        self.logger.info("Starting MarkovGenT compression")

        # Prepare data
        traces = []
        start_times = []

        self.logger.info("Processing traces...")
        for trace_id, trace_data in tqdm(
            dataset.traces.items(), desc="Processing traces"
        ):
            trace = Trace(trace_data)
            traces.append(trace)
            start_times.append(trace.start_time)
            """
            if len(trace_data) >= 1000:
                # write trace_data to file
                f = open(f"trace_{trace_id}.json", "w")
                import json

                f.write(json.dumps(trace_data, indent=4))
                print(f"Trace {trace_id} written to file")
                break
            """

        if not traces:
            raise ValueError("No valid traces found in dataset")

        self.logger.info(f"Processing {len(traces)} valid traces")

        # Train start time count synthesizer
        start_times_array = np.array(start_times)
        self.start_time_synthesizer.fit(start_times_array)

        # Train depth-aware Markov chain
        self.depth_markov_chain.fit(traces)

        # Train root duration synthesizer
        self.root_duration_synthesizer.fit(traces)

        # Train metadata neural network
        self.metadata_synthesizer.fit(traces)

        # Calculate total number of trees (root nodes) across all traces
        total_trees = 0
        for trace in traces:
            tree_sizes = count_spans_per_tree(trace.spans)
            total_trees += len(tree_sizes)

        # Create compressed dataset
        compressed_data = CompressedDataset()

        # Save configuration
        compressed_data.add(
            "markov_gen_t_config", self.config.to_dict(), SerializationFormat.MSGPACK
        )

        self.depth_markov_chain.save_state_dict(compressed_data)

        # Conditionally save model states based on save_decoders_only setting
        decoder_only = self.config.save_decoders_only
        self.logger.info(
            f"Saving {'decoder-only' if decoder_only else 'full'} model states"
        )

        # Save states using each synthesizer's save_state_dict method
        self.start_time_synthesizer.save_state_dict(
            compressed_data, decoder_only=decoder_only
        )
        self.root_duration_synthesizer.save_state_dict(
            compressed_data, decoder_only=decoder_only
        )
        self.metadata_synthesizer.save_state_dict(
            compressed_data, decoder_only=decoder_only
        )

        # Save number of trees (root nodes) for reconstruction
        compressed_data.add(
            "num_traces",
            total_trees,
            SerializationFormat.MSGPACK,
        )

        self.logger.info("MarkovGenT compression completed")
        return compressed_data

    def decompress(self, compressed_dataset: CompressedDataset) -> Dataset:
        """Generate new dataset using trained models."""
        self.logger.info("Starting MarkovGenT decompression")

        # Load configuration
        config_dict = compressed_dataset["markov_gen_t_config"]
        self.config = MarkovGenTConfig.from_dict(config_dict)

        # Load components
        self._load_models(compressed_dataset)

        # Generate new traces using the original dataset size
        num_traces = compressed_dataset["num_traces"]
        self.logger.info(
            f"Generating {num_traces} traces with optimized parallel processing"
        )
        generated_dataset = self._generate_traces(num_traces, compressed_dataset)

        self.logger.info(f"Generated {len(generated_dataset.traces)} traces")
        return generated_dataset

    def _load_models(self, compressed_dataset: CompressedDataset):
        """Load all models from compressed dataset."""
        # Load synthesizers (they will auto-detect loading mode)
        self.start_time_synthesizer = StartTimeCountSynthesizer(self.config)
        self.start_time_synthesizer.load_state_dict(compressed_dataset)

        # Choose root duration synthesis model
        if self.config.root_model == RootModel.MLP:
            self.root_duration_synthesizer = RootDurationMLPSynthesizer(self.config)
        elif self.config.root_model == RootModel.VAE:
            self.root_duration_synthesizer = RootDurationVAESynthesizer(self.config)
        elif self.config.root_model == RootModel.TABLE:
            self.root_duration_synthesizer = RootDurationTableSynthesizer(self.config)
        else:
            raise ValueError(f"Unknown root model: {self.config.root_model}")
        self.root_duration_synthesizer.load_state_dict(compressed_dataset)

        # Initialize metadata synthesizer based on config
        if self.config.metadata_model == MetadataModel.VAE:
            self.metadata_synthesizer = MetadataVAESynthesizer(self.config)
        else:
            raise ValueError(f"Unknown metadata model: {self.config.metadata_model}")
        self.metadata_synthesizer.load_state_dict(compressed_dataset)

        # Load depth Markov chain
        self.depth_markov_chain = MarkovRandomField(
            order=self.config.markov_order,
            max_depth=self.config.max_depth,
            max_children=self.config.max_children,
            edge_approach=self.config.mrf_edge_approach,
        )
        self.depth_markov_chain.load_state_dict(compressed_dataset)

    def _generate_traces(
        self, num_traces: int, compressed_dataset: CompressedDataset
    ) -> Dataset:
        """Generate traces using the trained models with multiprocessing."""
        self.logger.info(
            f"Generating {num_traces} traces using {self.num_processes} processes"
        )

        # Generate all start times at once
        all_start_times = self.start_time_synthesizer.sample(num_traces)

        # Serialize the compressed dataset for worker processes
        serialized_compressed_dataset = cloudpickle.dumps(compressed_dataset)

        # Split work among processes
        num_processes = self.num_processes
        traces_per_process = num_traces // num_processes
        remainder = num_traces % num_processes

        # Prepare arguments for each worker
        worker_args = []
        start_idx = 0

        for worker_id in range(num_processes):
            # Distribute remainder traces among first few workers
            current_traces = traces_per_process + (1 if worker_id < remainder else 0)
            end_idx = start_idx + current_traces

            start_times_chunk = all_start_times[start_idx:end_idx]

            worker_args.append(
                (
                    current_traces,
                    start_times_chunk,
                    serialized_compressed_dataset,
                )
            )

            start_idx = end_idx

        # Execute workers in parallel with CUDA-compatible multiprocessing
        dataset = Dataset()
        dataset.traces = {}

        # Use spawn method for CUDA compatibility
        mp_context = mp.get_context("spawn")

        with mp_context.Pool(processes=num_processes) as pool:
            # Submit all worker tasks with progress tracking
            with tqdm(
                total=num_traces,
                desc=f"Generating traces ({num_processes} workers)",
                unit="traces",
            ) as pbar:
                results = []
                # Use imap to get results as they complete for progress updates
                for result in pool.imap(_generate_traces_worker, worker_args):
                    results.append(result)
                    pbar.update(len(result))

            # Aggregate results from all workers
            total_generated = 0
            for worker_traces in results:
                dataset.traces.update(worker_traces)
                total_generated += len(worker_traces)

        self.logger.info(
            f"Generated {total_generated} traces across {num_processes} processes"
        )
        return dataset

    @staticmethod
    def load(
        compressed_dataset: CompressedDataset, num_processes: int = 8
    ) -> "MarkovGenTCompressor":
        """Load compressor from compressed dataset."""
        config_dict = compressed_dataset["markov_gen_t_config"]
        config = MarkovGenTConfig.from_dict(config_dict)

        compressor = MarkovGenTCompressor(config, num_processes=num_processes)
        compressor._load_models(compressed_dataset)

        return compressor
