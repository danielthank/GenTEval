"""
Data classes for HTTP request information.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class RequestInfo:
    """Information about an HTTP request"""

    url: str
    method: str
    timestamp: float  # Unix timestamp (wallTime from CDP)
    initiator: str | None = None
    initiator_type: str | None = None
    initiator_stack: dict[str, Any] | None = None
    initiator_line_number: int | None = None
    initiator_column_number: int | None = None
    parent_request_id: str | None = None
    request_id: str | None = None
    status_code: int | None = None
    response_size: int | None = None
    duration_ms: float | None = None
    headers: dict[str, str] | None = None
    children: list[str] = None
    cdp_request_id: str | None = None
    monotonic_time: float | None = (
        None  # MonotonicTime from CDP for duration calculations
    )

    def __post_init__(self):
        if self.children is None:
            self.children = []
