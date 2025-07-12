import logging
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from ... import CompressedDataset, SerializationFormat


class RootDurationTableSynthesizer:
    """
    Simple hashtable-based synthesizer that memorizes E[X] and E[X^2] 
    for each (startTimeBucket, operationType) combination.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Time bucketing configuration (hardcoded)
        self.bucket_size_us = 60 * 1000000  # 1 minute in microseconds
        
        # Hashtable to store statistics: {(time_bucket, operation_type): (count, sum_x, sum_x2)}
        self.stats_table = defaultdict(lambda: [0, 0.0, 0.0])  # [count, sum_x, sum_x2]
        
    def fit(self, traces):
        """Build the hashtable from root span data."""
        self.logger.info("Building Root Duration Hashtable")
        
        # Collect root span data
        root_start_times = []
        root_durations = []
        root_node_names = []
        
        for trace in traces:
            # Find root spans (spans with no parent)
            root_spans = []
            for span_id, span_data in trace.spans.items():
                if span_data["parentSpanId"] is None:
                    root_spans.append(span_data)
                    
            for root_span in root_spans:
                root_start_times.append(root_span["startTime"])
                root_durations.append(root_span["duration"])
                root_node_names.append(root_span["nodeName"])
                
        if not root_start_times:
            raise ValueError("No root spans found in traces")
            
        self.logger.info(f"Found {len(root_start_times)} root spans")
        
        # Prepare data
        root_start_times = np.array(root_start_times)
        root_durations = np.array(root_durations)
        root_node_names = np.array(root_node_names)
        
        # Convert start times to time buckets
        time_buckets = root_start_times // self.bucket_size_us
        
        # Use log-transformed durations directly (no normalization needed)
        log_durations = np.log(root_durations + 1)  # +1 to avoid log(0)
        
        # Build hashtable
        for i in range(len(time_buckets)):
            key = (int(time_buckets[i]), root_node_names[i])
            x = log_durations[i]
            
            # Update statistics: count, sum_x, sum_x2
            self.stats_table[key][0] += 1  # count
            self.stats_table[key][1] += x  # sum_x
            self.stats_table[key][2] += x * x  # sum_x2
            
        self.logger.info(f"Built hashtable with {len(self.stats_table)} unique (time_bucket, operation_type) combinations")
        
    def _get_moments(self, time_bucket: int, node_name: str) -> Tuple[float, float]:
        """Get E[X] and E[X^2] for a given (time_bucket, operation_type) combination."""
        key = (time_bucket, node_name)
        
        if key in self.stats_table:
            count, sum_x, sum_x2 = self.stats_table[key]
            e_x = sum_x / count  # E[X]
            e_x2 = sum_x2 / count  # E[X^2]
            return e_x, e_x2
        else:
            # Fallback: use global statistics or return default values
            if self.stats_table:
                # Compute global mean and variance as fallback
                all_counts = []
                all_sum_x = []
                all_sum_x2 = []
                
                for count, sum_x, sum_x2 in self.stats_table.values():
                    all_counts.append(count)
                    all_sum_x.append(sum_x)
                    all_sum_x2.append(sum_x2)
                
                total_count = sum(all_counts)
                total_sum_x = sum(all_sum_x)
                total_sum_x2 = sum(all_sum_x2)
                
                global_e_x = total_sum_x / total_count
                global_e_x2 = total_sum_x2 / total_count
                
                return global_e_x, global_e_x2
            else:
                # Default fallback if no data
                return 0.0, 1.0
                
    def synthesize_root_duration_batch(
        self, start_times: List[float], node_names: List[str], num_samples: int = 1
    ) -> List[float]:
        """Generate root span durations for multiple start times and node names at once.
        
        Args:
            start_times: List of start times
            node_names: List of node names  
            num_samples: Number of samples to generate per input (default: 1)
            
        Returns:
            List of durations. If num_samples > 1, samples Gaussian noise around
            the predicted mean with predicted standard deviation.
        """
        if not self.stats_table:
            raise ValueError("Hashtable not built. Call fit() first.")
            
        if len(start_times) != len(node_names):
            raise ValueError("start_times and node_names must have the same length")
            
        if not start_times:
            return []
            
        results = []
        
        for start_time, node_name in zip(start_times, node_names):
            # Convert start time to time bucket
            time_bucket = int(start_time // self.bucket_size_us)
            
            # Get moments from hashtable
            e_x, e_x2 = self._get_moments(time_bucket, node_name)
            
            # Compute variance: Var[X] = E[X^2] - (E[X])^2
            variance = e_x2 - (e_x ** 2)
            # Ensure variance is non-negative (numerical stability)
            variance = max(0, variance)
            std = np.sqrt(variance)
            
            # Generate samples
            for _ in range(num_samples):
                log_duration = np.random.normal(e_x, std)
                
                # Convert back to actual duration
                duration = np.exp(log_duration) - 1  # Reverse log transform
                duration = max(1, float(duration))  # Ensure positive duration
                
                results.append(duration)
                
        return results
        
    def save_state_dict(self, compressed_data: CompressedDataset, decoder_only: bool = False):
        """Save state dictionary."""
        
        # Convert defaultdict to regular dict for serialization
        stats_dict = dict(self.stats_table)
        
        compressed_data.add(
            "root_table_synthesizer",
            CompressedDataset(
                data={
                    "stats_table": (stats_dict, SerializationFormat.CLOUDPICKLE),
                }
            ),
            SerializationFormat.CLOUDPICKLE,
        )
        
    def load_state_dict(self, compressed_dataset):
        """Load state dictionary."""
        if "root_table_synthesizer" not in compressed_dataset:
            raise ValueError("No root_table_synthesizer found in compressed dataset")
            
        logger = logging.getLogger(__name__)
        
        # Load root synthesizer data
        root_synthesizer_data = compressed_dataset["root_table_synthesizer"]
        
        # Convert back to defaultdict
        stats_dict = root_synthesizer_data["stats_table"]
        self.stats_table = defaultdict(lambda: [0, 0.0, 0.0])
        self.stats_table.update(stats_dict)
        
        logger.info(f"Loaded root hashtable with {len(self.stats_table)} entries")
