"""Trace type enum for stratified sampling."""

from enum import Enum

from genteval.compressors.simple_gent.proto import simple_gent_pb2


class TraceType(Enum):
    """Trace type for stratified normal/error sampling."""

    NORMAL = "normal"
    ERROR = "error"

    def to_proto(self) -> int:
        """Convert to protobuf enum value."""
        if self == TraceType.ERROR:
            return simple_gent_pb2.TraceType.ERROR
        return simple_gent_pb2.TraceType.NORMAL

    @classmethod
    def from_proto(cls, proto_trace_type: int) -> "TraceType":
        """Convert protobuf enum value to TraceType."""
        if proto_trace_type == simple_gent_pb2.TraceType.ERROR:
            return cls.ERROR
        return cls.NORMAL

    @classmethod
    def from_trace(cls, trace, stratified: bool = True) -> "TraceType":
        """Classify a trace as NORMAL or ERROR based on its spans.

        Args:
            trace: The trace to classify
            stratified: If False, always return NORMAL (disable stratification)
        """
        if not stratified:
            return cls.NORMAL
        return cls.ERROR if trace.is_error else cls.NORMAL
