from collections import defaultdict

from genteval.dataset import Dataset
from genteval.evaluators import Evaluator


class GraphEvaluator(Evaluator):
    """
    Evaluator that creates a service graph for every minute.

    Nodes: Service names
    Edges: Count of spans from parent service to child service
    """

    def evaluate(self, dataset: Dataset, labels):
        """
        Create service graphs per minute bucket.

        Args:
            dataset: Dataset containing traces with span data
            labels: Labels dictionary (not used in this evaluator)

        Returns:
            Dictionary with service_graph_by_time containing graphs per minute
        """
        # Service graph per minute: time_bucket -> {nodes, edges}
        service_graph_by_time = defaultdict(
            lambda: {"nodes": set(), "edges": defaultdict(int)}
        )

        # Process all traces
        for trace in dataset.traces.values():
            for span in trace.values():
                # Extract service name from nodeName field
                service = span["nodeName"].split("!@#")[0]

                # Get time bucket (per minute)
                span_start_time = span["startTime"]
                time_bucket = span_start_time // (
                    60 * 1000000
                )  # Convert microseconds to minutes

                # Add service as a node
                service_graph_by_time[time_bucket]["nodes"].add(service)

                # Create edge from parent service to child service
                if span["parentSpanId"] is not None:
                    parent_span = trace.get(span["parentSpanId"])
                    if parent_span:
                        parent_service = parent_span["nodeName"].split("!@#")[0]
                        edge = (parent_service, service)

                        # Count edge occurrences
                        service_graph_by_time[time_bucket]["edges"][edge] += 1

        # Convert sets and tuples to JSON-serializable format
        result = {"service_graph_by_time": {}}

        for time_bucket, graph_data in sorted(service_graph_by_time.items()):
            result["service_graph_by_time"][time_bucket] = {
                "nodes": sorted(graph_data["nodes"]),
                "edges": {
                    f"{parent}->{child}": count
                    for (parent, child), count in sorted(graph_data["edges"].items())
                },
            }

        return result
