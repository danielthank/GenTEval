"""
HTTP Request Recording functionality.
"""

import csv
import hashlib
from dataclasses import asdict
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from .request_info import RequestInfo


class RequestRecorder:
    """Records HTTP requests and builds parent-child relationships"""

    def __init__(self):
        self.requests: dict[str, RequestInfo] = {}
        self.cdp_request_data: dict[str, dict[str, Any]] = {}  # Store CDP request data
        self.cdp_session = None
        self.trace_id = self.generate_trace_id()

    async def setup_cdp_listeners(self, cdp_session):
        """Set up CDP session listeners for detailed request information"""
        self.cdp_session = cdp_session

        # Listen for Network.requestWillBeSent events for detailed initiator info
        cdp_session.on("Network.requestWillBeSent", self.on_cdp_request_will_be_sent)

        # Listen for Network.responseReceived events for response details
        cdp_session.on("Network.responseReceived", self.on_cdp_response_received)

        # Listen for Network.loadingFinished events for final timing
        cdp_session.on("Network.loadingFinished", self.on_cdp_loading_finished)

    def should_skip_url(self, url: str) -> bool:
        """Determine if a URL should be skipped from recording"""
        # Skip data URLs (inline data like data:image/png;base64,...)
        # Also skip blob URLs, browser extension URLs, etc.
        skip_prefixes = ("data:", "blob:", "chrome-extension:", "moz-extension:")
        return url.startswith(skip_prefixes)

    async def on_cdp_request_will_be_sent(self, event):
        """Handle CDP Network.requestWillBeSent event"""
        request_id = event.get("requestId")
        request_data = event.get("request", {})
        initiator = event.get("initiator", {})
        timestamp = event.get(
            "timestamp", 0
        )  # MonotonicTime (seconds since arbitrary point)
        wall_time = event.get(
            "wallTime", 0
        )  # TimeSinceEpoch (Unix timestamp in seconds)
        url = request_data.get("url", "")
        method = request_data.get("method", "GET")

        # Skip data URLs (inline data like data:image/png;base64,...)
        # Also skip blob URLs, chrome-extension URLs, etc.
        if self.should_skip_url(url):
            return

        # Store CDP request data
        self.cdp_request_data[url] = {
            "requestId": request_id,
            "request": request_data,
            "initiator": initiator,
            "timestamp": timestamp,  # MonotonicTime for duration calculations
            "wallTime": wall_time,  # Unix timestamp for absolute time
        }

        # Get detailed initiator information from CDP data
        initiator_url, initiator_type, initiator_stack, line_number, column_number = (
            self.get_initiator_info_from_cdp(initiator)
        )

        # Create request info with enhanced initiator details
        # Note: parent_request_id will be set later in build_parent_child_relationships()
        request_info = RequestInfo(
            url=url,
            method=method,
            timestamp=wall_time,  # Use wallTime (Unix timestamp) instead of MonotonicTime
            monotonic_time=timestamp,  # Store MonotonicTime for duration calculations
            initiator=initiator_url,
            initiator_type=initiator_type,
            initiator_stack=initiator_stack,
            initiator_line_number=line_number,
            initiator_column_number=column_number,
            parent_request_id=None,  # Will be set later
            request_id=request_id,
            headers=request_data.get("headers", {}),
            cdp_request_id=request_id,
        )

        self.requests[request_id] = request_info

    async def on_cdp_response_received(self, event):
        """Handle CDP Network.responseReceived event"""
        request_id = event.get("requestId")
        response = event.get("response", {})

        # Skip if this is a data URL request
        if request_id in self.requests:
            request_info = self.requests[request_id]
            if self.should_skip_url(request_info.url):
                return

        # Update CDP data with response information
        for url_key, data in self.cdp_request_data.items():
            if data.get("requestId") == request_id:
                data["response"] = response
                break

        # Update request info with response data
        if request_id in self.requests:
            request_info = self.requests[request_id]
            request_info.status_code = response.get("status", 0)

            # Get response size if available
            headers = response.get("headers", {})
            if "content-length" in headers:
                try:
                    request_info.response_size = int(headers["content-length"])
                except (ValueError, TypeError):
                    pass

    async def on_cdp_loading_finished(self, event):
        """Handle CDP Network.loadingFinished event"""
        request_id = event.get("requestId")
        finish_timestamp = event.get("timestamp", 0)

        # Skip if this is a data URL request
        if request_id in self.requests:
            request_info = self.requests[request_id]
            if self.should_skip_url(request_info.url):
                return

        # Update CDP data with loading finished timestamp
        for url_key, data in self.cdp_request_data.items():
            if data.get("requestId") == request_id:
                data["loadingFinished"] = finish_timestamp
                break

        # Calculate duration and update request info
        if request_id in self.requests:
            request_info = self.requests[request_id]

            # Use MonotonicTime for duration calculation since both start and end are MonotonicTime
            start_monotonic_time = request_info.monotonic_time
            if start_monotonic_time is not None:
                # Calculate duration in milliseconds using MonotonicTime
                duration_seconds = finish_timestamp - start_monotonic_time
                request_info.duration_ms = duration_seconds * 1000

    def generate_trace_id(self) -> str:
        """Generate a unique trace ID for this session"""
        import time

        # Create a trace ID based on timestamp and a random component
        trace_data = f"{time.time()}{id(self)}"
        return hashlib.md5(trace_data.encode()).hexdigest()

    def generate_span_id(self, request_id: str) -> str:
        """Generate a unique span ID for a request"""
        # Create span ID based on request ID
        return hashlib.md5(request_id.encode()).hexdigest()[:16]

    def get_initiator_info_from_cdp(
        self, initiator: dict[str, Any]
    ) -> tuple[
        str | None,
        str | None,
        dict[str, Any] | None,
        int | None,
        int | None,
    ]:
        """Get detailed initiator information from CDP data"""
        initiator_url = None
        initiator_type = initiator.get("type", "unknown")
        initiator_stack = None
        line_number = None
        column_number = None

        if initiator_type == "script":
            # Script-initiated request
            stack = initiator.get("stack")
            if stack and "callFrames" in stack and stack["callFrames"]:
                frame = stack["callFrames"][0]
                initiator_url = frame.get("url")
                line_number = frame.get("lineNumber")
                column_number = frame.get("columnNumber")
                initiator_stack = stack

        elif initiator_type == "parser":
            # HTML parser initiated request
            initiator_url = initiator.get("url")
            line_number = initiator.get("lineNumber")
            column_number = initiator.get("columnNumber")

        elif initiator_type == "other":
            # Other types (redirect, etc.)
            initiator_url = initiator.get("url")

        return (
            initiator_url,
            initiator_type,
            initiator_stack,
            line_number,
            column_number,
        )

    def find_parent_request(self, initiator_url: str) -> str | None:
        """Find parent request ID based on initiator URL"""
        # Look for a request with matching URL in CDP data
        cdp_data = self.cdp_request_data.get(initiator_url)
        return cdp_data.get("requestId") if cdp_data else None

    def build_parent_child_relationships(self, root_url: str | None = None):
        """Build parent-child relationships after all requests have been collected

        Args:
            root_url: If provided, only keep requests that are part of this root URL's tree
        """
        print("Building parent-child relationships...")

        # Clear any existing relationships
        for request_info in self.requests.values():
            request_info.parent_request_id = None
            request_info.children = []

        # Iterate through all requests and establish relationships
        for request_id, request_info in self.requests.items():
            if request_info.initiator:
                # Find parent request based on initiator URL
                parent_request_id = self.find_parent_request(request_info.initiator)

                if parent_request_id and parent_request_id in self.requests:
                    # Set parent-child relationship
                    request_info.parent_request_id = parent_request_id
                    self.requests[parent_request_id].children.append(request_id)

        # If root_url is specified, filter requests to only keep those in that tree
        if root_url is not None:
            self._filter_requests_by_root_url(root_url)

        # Print summary
        root_requests = [
            req for req in self.requests.values() if not req.parent_request_id
        ]
        child_requests = [
            req for req in self.requests.values() if req.parent_request_id
        ]
        print(
            f"  Found {len(root_requests)} root requests and {len(child_requests)} child requests"
        )

    def _filter_requests_by_root_url(self, root_url: str):
        """Filter requests to only keep those that belong to the specified root URL's tree"""
        print(f"Filtering requests to keep only those from root URL: {root_url}")

        # Find the root request that matches the specified URL
        target_root_request_id = None
        for request_id, request_info in self.requests.items():
            if request_info.url == root_url and not request_info.parent_request_id:
                target_root_request_id = request_id
                break

        if target_root_request_id is None:
            print(f"Warning: No root request found for URL: {root_url}")
            return

        # Collect all request IDs that belong to the target root's tree
        requests_to_keep = set()

        def collect_tree_requests(request_id: str):
            """Recursively collect all requests in a tree"""
            if request_id in self.requests:
                requests_to_keep.add(request_id)
                # Add all children recursively
                for child_id in self.requests[request_id].children:
                    collect_tree_requests(child_id)

        # Start from the target root and collect its entire tree
        collect_tree_requests(target_root_request_id)

        # Remove requests that are not in the target tree
        requests_to_remove = []
        for request_id in self.requests:
            if request_id not in requests_to_keep:
                requests_to_remove.append(request_id)

        # Remove the unwanted requests
        for request_id in requests_to_remove:
            del self.requests[request_id]

        # Also clean up CDP data for removed requests
        cdp_urls_to_remove = []
        for url, cdp_data in self.cdp_request_data.items():
            if cdp_data.get("requestId") in requests_to_remove:
                cdp_urls_to_remove.append(url)

        for url in cdp_urls_to_remove:
            del self.cdp_request_data[url]

        print(
            f"Removed {len(requests_to_remove)} requests, kept {len(requests_to_keep)} requests"
        )

    def get_service_name(self, url: str) -> str:
        """Extract service name from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc or "unknown"
        except Exception:
            return "unknown"

    def format_time_for_csv(self, timestamp: float) -> str:
        """Format timestamp for CSV time column (HH:MM format)"""
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%H:%M")

    def get_request_tree(self) -> list[dict[str, Any]]:
        """Build the request tree structure"""
        # Find root requests (those without parents)
        root_requests = [
            req for req in self.requests.values() if not req.parent_request_id
        ]

        def build_tree(request_info: RequestInfo) -> dict[str, Any]:
            """Recursively build tree structure"""
            node = asdict(request_info)

            # Build children
            children = []
            for child_id in request_info.children:
                if child_id in self.requests:
                    children.append(build_tree(self.requests[child_id]))

            node["children"] = children
            return node

        return [build_tree(root) for root in root_requests]

    def print_tree(self, tree: list[dict[str, Any]], indent: int = 0):
        """Print the request tree in a readable format with detailed initiator info"""
        for node in tree:
            prefix = "  " * indent
            status = f" ({node['status_code']})" if node["status_code"] else ""
            duration = f" {node['duration_ms']:.0f}ms" if node["duration_ms"] else ""
            size = f" {node['response_size']}B" if node["response_size"] else ""

            print(f"{prefix}{node['method']} {node['url']}{status}{duration}{size}")

            if node["children"]:
                self.print_tree(node["children"], indent + 1)

    def print_detailed_initiator_info(self):
        """Print detailed initiator information for all requests"""
        print("\nDetailed Initiator Information:")
        print("=" * 50)

        for request_info in self.requests.values():
            print(f"\nURL: {request_info.url}")
            print(f"Method: {request_info.method}")
            print(f"Initiator Type: {request_info.initiator_type}")

            if request_info.initiator:
                print(f"Initiator URL: {request_info.initiator}")

            if request_info.initiator_line_number is not None:
                print(f"Line: {request_info.initiator_line_number}")

            if request_info.initiator_column_number is not None:
                print(f"Column: {request_info.initiator_column_number}")

            if request_info.initiator_stack:
                print("Call Stack:")
                call_frames = request_info.initiator_stack.get("callFrames", [])
                for i, frame in enumerate(call_frames[:3]):  # Show first 3 frames
                    print(
                        f"  {i + 1}. {frame.get('functionName', '<anonymous>')} at {frame.get('url', '')}:{frame.get('lineNumber', '')}:{frame.get('columnNumber', '')}"
                    )

            print("-" * 30)

    def save_to_csv(self, file_path: str):
        """Save the request data to a CSV file in OpenTelemetry format"""
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            for row in self.iter_otel_csv_rows():
                writer.writerow(row)

        print(f"Request data saved to OpenTelemetry CSV: {file_path}")

    def iter_otel_csv_rows(self):
        """Generator that yields formatted CSV rows for OpenTelemetry format"""
        fieldnames = [
            "time",
            "traceID",
            "spanID",
            "serviceName",
            "methodName",
            "operationName",
            "startTime",
            "duration",
            "statusCode",
            "parentSpanID",
        ]

        # Yield header row first
        yield fieldnames

        # Yield data rows
        for request_info in self.requests.values():
            if request_info.duration_ms is None:
                continue  # Skip requests without duration
            formatted_row = self.format_request_for_otel_csv(request_info)
            yield [formatted_row[field] for field in fieldnames]

    def format_request_for_otel_csv(self, request_info: RequestInfo) -> dict[str, str]:
        """Format request information for OpenTelemetry CSV output"""
        start_time_micros = int(request_info.timestamp * 1000000)

        # Get parent span ID if there's a parent request
        parent_span_id = ""
        if (
            request_info.parent_request_id
            and request_info.parent_request_id in self.requests
        ):
            parent_span_id = self.generate_span_id(request_info.parent_request_id)

        parsed_url = urlparse(request_info.url)
        operation_name = parsed_url.path or "/"
        if parsed_url.query:
            operation_name += "?" + parsed_url.query

        return {
            "time": self.format_time_for_csv(request_info.timestamp),
            "traceID": self.trace_id,
            "spanID": self.generate_span_id(request_info.request_id or ""),
            "serviceName": self.get_service_name(request_info.url),
            "methodName": request_info.method,
            "operationName": operation_name,
            "startTime": str(start_time_micros),
            "duration": int((request_info.duration_ms or 1) * 1000),
            "statusCode": str(request_info.status_code or 0),
            "parentSpanID": parent_span_id,
        }
