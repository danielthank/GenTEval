import random

from genteval.compressors import CompressedDataset, Compressor, SerializationFormat
from genteval.dataset import Dataset
from genteval.proto.resource_pb2 import Resource
from genteval.proto.trace_pb2 import TracesData


class HeadSamplingCompressor(Compressor):
    def __init__(self, sampling_rate: int):
        super().__init__()
        self.sampling_rate = sampling_rate

    def _compress_impl(self, dataset: Dataset) -> CompressedDataset:
        trace_len = len(dataset.traces)
        if trace_len < self.sampling_rate:
            raise ValueError(
                f"Dataset length {trace_len} is less than the sampling rate {self.sampling_rate}."
            )
        compressed_dataset = CompressedDataset()

        # Ranomly sample the trace with probability of 1/sampling_rate
        sampled_traces = {
            trace_id: trace
            for trace_id, trace in dataset.traces.items()
            if random.random() < 1 / self.sampling_rate
        }

        # Convert sampled traces to OpenTelemetry TracesData format
        traces_data = self._convert_to_opentelemetry_format(sampled_traces)
        compressed_dataset.add(
            "sampled_traces", traces_data, SerializationFormat.GRPC, TracesData
        )

        return compressed_dataset

    def _decompress_impl(self, compressed_dataset: CompressedDataset) -> Dataset:
        dataset = Dataset()
        # Extract traces from OpenTelemetry format
        traces_data = compressed_dataset["sampled_traces"]
        dataset.traces = self._convert_from_opentelemetry_format(traces_data)
        return dataset

    def _convert_to_opentelemetry_format(self, traces_dict: dict) -> TracesData:
        """Convert internal trace format to OpenTelemetry TracesData."""
        traces_data = TracesData()

        for trace_id, trace in traces_dict.items():
            # Create a ResourceSpans for each trace
            resource_spans = traces_data.resource_spans.add()

            # Create a basic resource
            resource_spans.resource.CopyFrom(Resource())

            # Create a ScopeSpans
            scope_spans = resource_spans.scope_spans.add()

            # Convert trace data to spans
            if isinstance(trace, dict):
                for span_id, span_data in trace.items():
                    span = scope_spans.spans.add()
                    span.trace_id = trace_id.encode("utf-8")[:16].ljust(16, b"\x00")
                    span.span_id = span_id.encode("utf-8")[:8].ljust(8, b"\x00")
                    span.name = str(span_data.get("nodeName", "unknown"))

                    # Store additional span data in attributes for reconstruction
                    if "startTime" in span_data:
                        attr = span.attributes.add()
                        attr.key = "startTime"
                        attr.value.int_value = int(span_data["startTime"])

                    if "duration" in span_data:
                        attr = span.attributes.add()
                        attr.key = "duration"
                        attr.value.int_value = int(span_data["duration"])

                    if (
                        "parentSpanId" in span_data
                        and span_data["parentSpanId"] is not None
                    ):
                        span.parent_span_id = (
                            str(span_data["parentSpanId"])
                            .encode("utf-8")[:8]
                            .ljust(8, b"\x00")
                        )

                    if (
                        "statusCode" in span_data
                        and span_data["statusCode"] is not None
                    ):
                        attr = span.attributes.add()
                        attr.key = "statusCode"
                        attr.value.int_value = int(span_data["statusCode"])
            else:
                # Handle simple trace format
                span = scope_spans.spans.add()
                span.trace_id = trace_id.encode("utf-8")[:16].ljust(16, b"\x00")
                span.span_id = b"\x00" * 8
                span.name = str(trace)

        return traces_data

    def _convert_from_opentelemetry_format(self, traces_data: TracesData) -> dict:
        """Convert OpenTelemetry TracesData back to internal trace format."""
        traces_dict = {}

        for resource_spans in traces_data.resource_spans:
            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    trace_id = span.trace_id.decode("utf-8").rstrip("\x00")
                    span_id = span.span_id.decode("utf-8").rstrip("\x00")

                    if trace_id not in traces_dict:
                        traces_dict[trace_id] = {}

                    # Reconstruct span data from OpenTelemetry format
                    span_data = {
                        "nodeName": span.name,
                        "startTime": 0,  # Default values to ensure all fields exist
                        "duration": 0,
                        "statusCode": 0,
                        "parentSpanId": None,
                    }

                    # Extract additional data from attributes
                    for attr in span.attributes:
                        if attr.key == "startTime":
                            span_data["startTime"] = attr.value.int_value
                        elif attr.key == "duration":
                            span_data["duration"] = attr.value.int_value
                        elif attr.key == "statusCode":
                            span_data["statusCode"] = attr.value.int_value

                    # Handle parent span ID
                    if span.parent_span_id and span.parent_span_id != b"\x00" * 8:
                        span_data["parentSpanId"] = span.parent_span_id.decode(
                            "utf-8"
                        ).rstrip("\x00")

                    traces_dict[trace_id][span_id or "default"] = span_data

        return traces_dict
