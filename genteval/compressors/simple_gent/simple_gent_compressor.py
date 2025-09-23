import logging
import uuid

import numpy as np
from tqdm import tqdm

from genteval.compressors import CompressedDataset, Compressor, SerializationFormat
from genteval.compressors.simple_gent.proto import simple_gent_pb2
from genteval.compressors.trace import Trace
from genteval.dataset import Dataset
from genteval.utils.data_structures import count_spans_per_tree

from .config import SimpleGenTConfig
from .models import (
    MetadataVAEModel,
    NodeEncoder,
    RootDurationModel,
    RootModel,
    TopologyModel,
)
from .timing_utils import SplitTimer


def _get_random_trace_id():
    return "trace" + uuid.uuid4().hex


def _get_random_span_id():
    return "span" + uuid.uuid4().hex


class SimpleGenTCompressor(Compressor):
    """
    Simple GenT compressor implementing the 4-model algorithm:
    1. Root model: (root_name, root_child_cnt) -> count
    2. Topology model: (parent_feature, child_feature) -> potential
    3. Root duration model: (root_name, root_child_cnt) -> GMM
    4. MetadataVAE model: neural network for joint gap and duration ratio modeling
    """

    def __init__(self, config: SimpleGenTConfig = None):
        super().__init__()
        self.config = config or SimpleGenTConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize models (MetadataVAEModel will be created after node encoder is fitted)
        self.node_encoder = NodeEncoder()
        self.root_model = RootModel(self.config)
        self.topology_model = TopologyModel(self.config)
        self.root_duration_model = RootDurationModel(self.config)
        self.metadata_vae_model = None  # Will be created with vocab_size during fitting

        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)

    def _transform_dataset(self, traces):
        """Transform node names to indices in all traces"""
        self.logger.info(
            f"Transforming {len(traces)} traces: converting node names to indices"
        )

        all_node_names = []
        trace_span_positions = []  # [(trace_idx, span_id, position_in_batch)]

        # Collect all node names from all traces
        for trace_id, trace_data in tqdm(traces.items(), desc="Collecting node names"):
            for span_id, span_data in trace_data.items():
                node_name = span_data["nodeName"]
                position_in_batch = len(all_node_names)
                all_node_names.append(node_name)
                trace_span_positions.append((trace_id, span_id, position_in_batch))

        self.logger.info(f"Collected {len(all_node_names)} node names from spans")

        # Transform all node names to indices in batch
        all_node_indices = self.node_encoder.transform(all_node_names)
        self.logger.info("Batch transformed all node names to indices")

        # Apply transformed indices back to traces
        for trace_id, span_id, batch_pos in tqdm(
            trace_span_positions, desc="Applying indices"
        ):
            traces[trace_id][span_id]["nodeIdx"] = all_node_indices[batch_pos]
            del traces[trace_id][span_id]["nodeName"]

        self.logger.info("Successfully transformed all node names to indices")

    def _inverse_transform_dataset(self, traces):
        """Transform node indices back to names in all traces"""
        self.logger.info(
            f"Inverse transforming {len(traces)} traces: converting node indices to names"
        )

        all_node_idx = []
        trace_span_positions = []  # [(trace_idx, span_id, position_in_batch)]

        # Collect all node indices from all traces
        for trace_id, trace_data in tqdm(
            traces.items(), desc="Collecting node indices"
        ):
            for span_id, span_data in trace_data.items():
                all_node_idx.append(span_data["nodeIdx"])
                trace_span_positions.append((trace_id, span_id, len(all_node_idx) - 1))

        self.logger.info(f"Collected {len(all_node_idx)} node indices from spans")

        # Inverse transform all node indices to names in batch
        all_node_names = self.node_encoder.inverse_transform(all_node_idx)
        self.logger.info("Batch inverse transformed all node indices to names")

        # Apply transformed names back to traces
        for trace_id, span_id, batch_pos in tqdm(
            trace_span_positions, desc="Applying names"
        ):
            traces[trace_id][span_id]["nodeName"] = all_node_names[batch_pos]
            del traces[trace_id][span_id]["nodeIdx"]

        self.logger.info("Successfully inverse transformed all node indices to names")

    def _preprocess_traces_with_indices(self, traces):
        preprocessed_traces = []
        for trace_data in traces.values():
            trace = Trace(trace_data)
            preprocessed_traces.append(trace)

        return preprocessed_traces

    def _compress_impl(self, dataset: Dataset) -> CompressedDataset:
        """Learn models from the dataset (training phase)."""
        self.logger.info("Starting Simple GenT compression")

        # Initialize timing
        timer = SplitTimer()

        # Step 1: Fit shared node encoder first (CPU operation)
        with timer.cpu_context():
            self.logger.info("Fitting shared node encoder")
            all_node_names = set()
            for trace_data in dataset.traces.values():
                for span_data in trace_data.values():
                    all_node_names.add(span_data["nodeName"])

            self.node_encoder.fit(list(all_node_names))
            self.logger.info(
                f"Node encoder fitted with {self.node_encoder.get_vocab_size()} unique node names"
            )

        # Step 2: Preprocess traces - replace nodeNames with indices (CPU operation)
        with timer.cpu_context():
            self._transform_dataset(dataset.traces)
            preprocessed_traces = self._preprocess_traces_with_indices(dataset.traces)
            self.logger.info("Preprocessed traces with node indices")

            # Calculate total number of trees (root nodes) and spans across all traces
            total_trees = 0
            total_spans = 0
            for trace in preprocessed_traces:
                tree_sizes = count_spans_per_tree(trace.spans)
                total_trees += len(tree_sizes)
                total_spans += len(trace.spans)

            # Output totals
            self.logger.info(f"Total number of trees: {total_trees}")
            self.logger.info(f"Total number of spans: {total_spans}")

        # Step 3: Initialize models with vocab_size and train on preprocessed traces
        vocab_size = self.node_encoder.get_vocab_size()

        # Train CPU-based models
        with timer.cpu_context():
            self.root_model.fit(preprocessed_traces)
            self.topology_model.fit(preprocessed_traces)
            self.root_duration_model.fit(preprocessed_traces)

        # Train GPU-based model (MetadataVAE)
        self.metadata_vae_model = MetadataVAEModel(self.config, vocab_size)
        with timer.gpu_context(device=str(self.metadata_vae_model.device)):
            self.metadata_vae_model.fit(preprocessed_traces)

        # Create compressed dataset and save (CPU operation)
        with timer.cpu_context():
            compressed_data = CompressedDataset()

            # Save configuration
            compressed_data.add(
                "simple_gent_config", self.config.to_dict(), SerializationFormat.MSGPACK
            )

            # Create protobuf message and save model states
            proto_models = simple_gent_pb2.SimpleGenTModels()
            proto_models.time_bucket_duration = self.config.time_bucket_duration_us

            # Save all models using consistent interface
            self.node_encoder.save_state_dict(proto_models)

            self.root_model.save_state_dict(proto_models)
            self.topology_model.save_state_dict(proto_models)
            self.root_duration_model.save_state_dict(proto_models)
            self.metadata_vae_model.save_state_dict(proto_models)

            # Serialize protobuf message
            compressed_data.add(
                "simple_gent_models_proto",
                proto_models,
                SerializationFormat.GRPC,
                simple_gent_pb2.SimpleGenTModels,
            )

            # Save number of trees (root nodes) for reconstruction
            compressed_data.add("num_traces", total_trees, SerializationFormat.MSGPACK)

        # Save timing information
        cpu_time, gpu_time = timer.get_times()
        compressed_data.add(
            "compression_time_cpu_seconds", cpu_time, SerializationFormat.MSGPACK
        )
        compressed_data.add(
            "compression_time_gpu_seconds", gpu_time, SerializationFormat.MSGPACK
        )

        self.logger.info(
            f"Simple GenT compression completed - CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s"
        )
        return compressed_data

    def _decompress_impl(self, compressed_dataset: CompressedDataset) -> Dataset:
        """Generate new dataset using trained models (sampling phase)."""
        self.logger.info("Starting Simple GenT decompression")

        # Load configuration
        config_dict = compressed_dataset["simple_gent_config"]
        self.config = SimpleGenTConfig.from_dict(config_dict)

        # Load models
        self._load_models(compressed_dataset)

        # Generate traces
        num_traces = compressed_dataset["num_traces"]
        self.logger.info(f"Generating {num_traces} traces")

        generated_dataset = self._generate_traces(num_traces)

        self.logger.info(f"Generated {len(generated_dataset.traces)} traces")
        return generated_dataset

    def _load_models(self, compressed_dataset: CompressedDataset):
        """Load all models from compressed dataset."""
        # Load protobuf models
        proto_models = compressed_dataset["simple_gent_models_proto"]

        # Load shared node encoder first
        self.node_encoder.load_state_dict(proto_models)
        self.logger.info(
            f"Loaded node encoder with {self.node_encoder.get_vocab_size()} node names"
        )

        # Load models
        vocab_size = self.node_encoder.get_vocab_size()

        self.root_model = RootModel(self.config)
        self.root_model.load_state_dict(proto_models)

        self.topology_model = TopologyModel(self.config)
        self.topology_model.load_state_dict(proto_models)

        self.root_duration_model = RootDurationModel(self.config)
        self.root_duration_model.load_state_dict(proto_models)

        self.metadata_vae_model = MetadataVAEModel(self.config, vocab_size)
        self.metadata_vae_model.load_state_dict(proto_models)

    def _generate_traces(self, num_traces: int) -> Dataset:
        """Generate traces using the trained models."""
        dataset = Dataset()
        dataset.traces = {}

        # Step 1: Sample root features with their time buckets directly
        root_features_with_buckets = self.root_model.sample_root_features(num_traces)

        # Step 2-3: Collect all trace info for batch processing
        trace_infos = []
        for time_bucket, root_feature in tqdm(
            root_features_with_buckets, desc="Preparing traces for batch generation"
        ):
            trace_id = _get_random_trace_id()

            # Generate topology using topology model
            root_tree = self.topology_model.generate_tree_structure(
                root_feature, time_bucket
            )

            if root_tree is None:
                continue

            # Generate root duration using root duration model
            root_duration = self.root_duration_model.sample_duration(
                time_bucket, root_feature
            )

            # Calculate trace start time based on time bucket
            # Use the time bucket to determine the base time, then add some variation
            base_time = time_bucket * self.config.time_bucket_duration_us
            # Add random variation within the time bucket
            time_variation = np.random.randint(0, self.config.time_bucket_duration_us)
            trace_start_time = base_time + time_variation

            # Collect trace info for batch processing
            trace_infos.append(
                (trace_id, time_bucket, root_tree, trace_start_time, root_duration)
            )

        # Step 4: Batch generate timing for ALL traces simultaneously
        self.logger.info(f"Batch generating timing for {len(trace_infos)} traces")
        all_traces_spans = self._generate_traces_batch(trace_infos)

        # Add all generated traces to dataset
        dataset.traces.update(all_traces_spans)

        return dataset

    def _generate_trace_timing(
        self, root_tree, trace_start_time: int, root_duration: float, time_bucket: int
    ) -> dict[str, dict]:
        """Generate timing information for the tree using depth-based batch sampling."""
        spans = {}

        # Create root span
        root_span_id = _get_random_span_id()
        spans[root_span_id] = {
            "nodeIdx": root_tree.feature.node_idx,
            "startTime": int(trace_start_time),
            "duration": int(root_duration),
            "statusCode": None,
            "parentSpanId": None,
        }

        # Process tree level by level (depth-based batching)
        # Each level: [(tree_node, span_id, start_time, duration)]
        current_level = [(root_tree, root_span_id, trace_start_time, root_duration)]

        while current_level:
            next_level = []
            batch_requests = []
            child_infos = []

            # Phase 1: Collect all children at this depth level
            for (
                parent_tree_node,
                parent_span_id,
                parent_start_time,
                parent_duration,
            ) in current_level:
                children = parent_tree_node.children
                child_count = len(children)

                # Process each child of this parent
                for child_idx, child_tree_node in enumerate(children):
                    child_span_id = _get_random_span_id()

                    # Calculate normalized child index: child_idx / child_count
                    normalized_child_idx = (
                        child_idx / child_count if child_count > 0 else 0.0
                    )

                    # Create batch request with actual conditioning values (removed parent_start_time)
                    batch_requests.append(
                        (
                            time_bucket,
                            parent_tree_node.feature,  # parent_feature
                            child_tree_node.feature,  # child_feature
                            parent_duration,  # actual parent duration
                            normalized_child_idx,  # child_idx / child_count
                        )
                    )

                    # Store child info for span creation
                    child_infos.append(
                        (
                            child_tree_node,
                            child_span_id,
                            parent_span_id,
                            parent_start_time,
                            parent_duration,
                        )
                    )

            # Phase 2: Batch sample all children at this depth level
            if batch_requests:
                batch_results = self.metadata_vae_model.sample_ratios_batch(
                    batch_requests
                )

                # Phase 3: Create child spans and prepare next level
                for i, ((gap_ratio, duration_ratio), child_info) in enumerate(
                    zip(batch_results, child_infos, strict=False)
                ):
                    (
                        child_tree_node,
                        child_span_id,
                        parent_span_id,
                        parent_start_time,
                        parent_duration,
                    ) = child_info

                    # Calculate actual child timing
                    gap_from_parent = gap_ratio * parent_duration
                    child_start_time = parent_start_time + gap_from_parent
                    child_duration = duration_ratio * parent_duration

                    # Ensure reasonable bounds
                    child_start_time = max(parent_start_time, child_start_time)
                    child_duration = max(1.0, child_duration)

                    # Create child span
                    spans[child_span_id] = {
                        "nodeIdx": child_tree_node.feature.node_idx,
                        "startTime": int(child_start_time),
                        "duration": int(child_duration),
                        "statusCode": None,
                        "parentSpanId": parent_span_id,
                    }

                    # Add to next level for further processing
                    next_level.append(
                        (
                            child_tree_node,
                            child_span_id,
                            child_start_time,
                            child_duration,
                        )
                    )

            # Move to next depth level
            current_level = next_level

        return spans

    def _generate_traces_batch(self, trace_infos: list) -> dict[str, dict[str, dict]]:
        """
        Generate multiple traces simultaneously using cross-trace batching for better GPU utilization.

        Args:
            trace_infos: List of (trace_id, time_bucket, root_tree, trace_start_time, root_duration) tuples

        Returns:
            Dict of {trace_id: spans_dict}
        """
        if not trace_infos:
            return {}

        all_traces_spans = {}

        # Initialize: Create root spans for all traces and set up first level
        current_level_all_traces = []  # [(trace_id, tree_node, span_id, start_time, duration)]

        for (
            trace_id,
            time_bucket,
            root_tree,
            trace_start_time,
            root_duration,
        ) in trace_infos:
            # Create root span
            root_span_id = _get_random_span_id()
            all_traces_spans[trace_id] = {
                root_span_id: {
                    "nodeIdx": root_tree.feature.node_idx,
                    "startTime": int(trace_start_time),
                    "duration": int(root_duration),
                    "statusCode": None,
                    "parentSpanId": None,
                }
            }

            # Add root to current level for processing
            current_level_all_traces.append(
                (
                    trace_id,
                    time_bucket,
                    root_tree,
                    root_span_id,
                    trace_start_time,
                    root_duration,
                )
            )

        # Process all traces level by level with cross-trace batching
        while current_level_all_traces:
            next_level_all_traces = []
            mega_batch_requests = []
            request_to_trace_info = []  # Track which trace each request belongs to

            # Phase 1: Collect ALL children from ALL traces at current depth level
            for (
                trace_id,
                time_bucket,
                parent_tree_node,
                parent_span_id,
                parent_start_time,
                parent_duration,
            ) in current_level_all_traces:
                children = parent_tree_node.children
                child_count = len(children)

                # Process each child of this parent
                for child_idx, child_tree_node in enumerate(children):
                    child_span_id = _get_random_span_id()

                    # Calculate normalized child index: child_idx / child_count
                    normalized_child_idx = (
                        child_idx / child_count if child_count > 0 else 0.0
                    )

                    # Add to mega batch request (across all traces!) - removed parent_start_time
                    mega_batch_requests.append(
                        (
                            time_bucket,
                            parent_tree_node.feature,  # parent_feature
                            child_tree_node.feature,  # child_feature
                            parent_duration,  # actual parent duration
                            normalized_child_idx,  # child_idx / child_count
                        )
                    )

                    # Track which trace and span this request belongs to
                    request_to_trace_info.append(
                        (
                            trace_id,
                            child_tree_node,
                            child_span_id,
                            parent_span_id,
                            parent_start_time,
                            parent_duration,
                            time_bucket,
                        )
                    )

            # Phase 2: Single massive batch call across ALL traces at this depth!
            if mega_batch_requests:
                mega_batch_results = self.metadata_vae_model.sample_ratios_batch(
                    mega_batch_requests
                )

                # Phase 3: Apply batch results back to respective traces
                for i, ((gap_ratio, duration_ratio), trace_info) in enumerate(
                    zip(mega_batch_results, request_to_trace_info, strict=False)
                ):
                    (
                        trace_id,
                        child_tree_node,
                        child_span_id,
                        parent_span_id,
                        parent_start_time,
                        parent_duration,
                        time_bucket,
                    ) = trace_info

                    # Calculate actual child timing
                    gap_from_parent = gap_ratio * parent_duration
                    child_start_time = parent_start_time + gap_from_parent
                    child_duration = duration_ratio * parent_duration

                    # Ensure reasonable bounds
                    child_start_time = max(parent_start_time, child_start_time)
                    child_duration = max(1.0, child_duration)

                    # Create child span in the appropriate trace
                    all_traces_spans[trace_id][child_span_id] = {
                        "nodeIdx": child_tree_node.feature.node_idx,
                        "startTime": int(child_start_time),
                        "duration": int(child_duration),
                        "statusCode": None,
                        "parentSpanId": parent_span_id,
                    }

                    # Add to next level for further processing
                    next_level_all_traces.append(
                        (
                            trace_id,
                            time_bucket,
                            child_tree_node,
                            child_span_id,
                            child_start_time,
                            child_duration,
                        )
                    )

            # Move to next depth level across all traces
            current_level_all_traces = next_level_all_traces

        self._inverse_transform_dataset(all_traces_spans)

        return all_traces_spans

    @staticmethod
    def load(compressed_dataset: CompressedDataset) -> "SimpleGenTCompressor":
        """Load compressor from compressed dataset."""
        config_dict = compressed_dataset["simple_gent_config"]
        config = SimpleGenTConfig.from_dict(config_dict)

        compressor = SimpleGenTCompressor(config)
        compressor._load_models(compressed_dataset)

        return compressor
