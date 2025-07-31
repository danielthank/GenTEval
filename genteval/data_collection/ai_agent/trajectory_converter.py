"""Convert AI agent trajectory data to distributed tracing format."""

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TrajectoryConverter:
    """Converts trajectory JSON data to distributed tracing CSV format."""

    def __init__(self):
        self.id_to_entry: dict[int, dict[str, Any]] = {}
        self.traces: list[dict[str, Any]] = []
        self.conversation_trace_id: str = ""
        self.artificial_parent_map: dict[
            int, int
        ] = {}  # Maps entry_id to artificial parent_id

    def generate_id(self, data: str, length: int = 16) -> str:
        """Generate a hex ID from input data."""
        return hashlib.md5(data.encode(), usedforsecurity=False).hexdigest()[:length]

    def parse_timestamp(self, timestamp_str: str) -> int:
        """Parse ISO timestamp and return microseconds since epoch."""
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1_000_000)  # Convert to microseconds

    def extract_service_info(self, entry: dict[str, Any]) -> tuple[str, str]:
        """Extract service name and operation from trajectory entry."""
        action = entry.get("action", "")
        cause = entry.get("cause")

        # Handle child/response entries (those with cause but no action)
        if not action and cause is not None and cause in self.id_to_entry:
            parent_entry = self.id_to_entry[cause]
            parent_action = parent_entry.get("action", "")

            # Map child services based on parent action
            if parent_action == "run_ipython":
                service_name = "execution-service"
                operation_name = "code_execution_result"
            elif parent_action == "edit":
                service_name = "editor-service"
                operation_name = "file_edit_result"
            elif parent_action == "browse_interactive":
                service_name = "browser-service"
                operation_name = "browse_result"
            elif parent_action == "message":
                service_name = "chat-service"
                operation_name = "message_response"
            else:
                service_name = "system-service"
                operation_name = "system_response"

            return service_name, operation_name

        # Handle primary action entries
        if action == "message":
            service_name = "chat-service"
            operation_name = "send_message"
        elif action == "run_ipython":
            service_name = "execution-service"
            operation_name = "execute_code"
        elif action == "edit":
            service_name = "editor-service"
            operation_name = "edit_file"
        elif action == "browse_interactive":
            service_name = "browser-service"
            operation_name = "browse_page"
        elif action == "read":
            service_name = "file-service"
            operation_name = "read_file"
        elif action == "finish":
            service_name = "system-service"
            operation_name = "finish_task"
        elif action.startswith("run_"):
            service_name = "execution-service"
            operation_name = action.replace("run_", "execute_")
        else:
            service_name = "system-service"
            operation_name = action or "system_operation"

        return service_name, operation_name

    def calculate_duration(
        self,
        current_entry: dict[str, Any],
        trajectory_data: list[dict[str, Any]],
        current_index: int,
    ) -> int:
        """Calculate duration for a span in microseconds."""
        current_timestamp_str = current_entry.get("timestamp", "")
        if not current_timestamp_str:
            return 1000  # Default 1ms in microseconds

        current_time = self.parse_timestamp(current_timestamp_str)

        # For entries with children (based on cause field), calculate until child completion
        current_id = current_entry.get("id")
        if current_id is not None:
            # Look for child entries that reference this entry as cause
            child_entries = [
                (i, entry)
                for i, entry in enumerate(trajectory_data)
                if entry.get("cause") == current_id and i > current_index
            ]

            if child_entries:
                # Use the timestamp of the last child
                last_child_index, last_child = max(child_entries, key=lambda x: x[0])
                child_timestamp_str = last_child.get("timestamp", "")
                if child_timestamp_str:
                    child_time = self.parse_timestamp(child_timestamp_str)
                    return max(child_time - current_time, 1000)

        # For other entries, use next entry timestamp
        if current_index < len(trajectory_data) - 1:
            next_entry = trajectory_data[current_index + 1]
            next_timestamp_str = next_entry.get("timestamp", "")
            if next_timestamp_str:
                next_time = self.parse_timestamp(next_timestamp_str)
                return max(next_time - current_time, 1000)

        # Default duration for last entry or missing timestamps
        return 1_000_000  # 1 second in microseconds

    def get_parent_span_id(self, entry: dict[str, Any]) -> str:
        """Get parent span ID from cause field or artificial parent relationship."""
        entry_id = entry.get("id")

        # First check for natural parent via cause field
        cause = entry.get("cause")
        if cause is not None and cause in self.id_to_entry:
            parent_entry = self.id_to_entry[cause]
            parent_service, _ = self.extract_service_info(parent_entry)
            parent_timestamp = parent_entry.get("timestamp", "")
            # Use same generation logic as main span ID
            return self.generate_id(
                f"span_{cause}_{parent_service}_{parent_timestamp}", 16
            )

        # Check for artificial parent relationship
        if entry_id in self.artificial_parent_map:
            artificial_parent_id = self.artificial_parent_map[entry_id]
            if artificial_parent_id in self.id_to_entry:
                parent_entry = self.id_to_entry[artificial_parent_id]
                parent_service, _ = self.extract_service_info(parent_entry)
                parent_timestamp = parent_entry.get("timestamp", "")
                # Use same generation logic as main span ID
                return self.generate_id(
                    f"span_{artificial_parent_id}_{parent_service}_{parent_timestamp}",
                    16,
                )

        return ""

    def _build_artificial_parent_map(
        self, trajectory_data: list[dict[str, Any]]
    ) -> None:
        """Build artificial parent relationships where user chat messages become root spans."""
        current_user_chat_id = None

        for i, entry in enumerate(trajectory_data):
            entry_id = entry.get("id", i)
            source = entry.get("source", "")
            action = entry.get("action", "")

            # Check if this is a user chat message
            if source == "user" and action == "message":
                current_user_chat_id = entry_id
                # User chat messages don't have artificial parents
                continue

            # If we have a current user chat message, all subsequent entries become its children
            # (unless they already have a natural parent via cause field)
            if current_user_chat_id is not None:
                cause = entry.get("cause")
                if cause is None:  # No natural parent
                    self.artificial_parent_map[entry_id] = current_user_chat_id

    def convert(self, input_file: str, output_file: str) -> None:
        """Convert trajectory JSON to traces CSV format."""
        # Load trajectory data
        input_path = Path(input_file)
        with input_path.open() as f:
            trajectory_data = json.load(f)

        # Store trajectory data for summary generation
        self.trajectory_data = trajectory_data

        # Generate single traceID for entire conversation
        if trajectory_data:
            first_entry = trajectory_data[0]
            conversation_seed = f"conversation_{first_entry.get('timestamp', '')}_{len(trajectory_data)}"
            self.conversation_trace_id = self.generate_id(conversation_seed, 32)

        # Create ID mapping for parent-child relationships
        self.id_to_entry = {
            entry.get("id", i): entry for i, entry in enumerate(trajectory_data)
        }

        # Build artificial parent relationships for user chat messages
        self._build_artificial_parent_map(trajectory_data)

        # Process each trajectory entry
        for i, entry in enumerate(trajectory_data):
            entry_id = entry.get("id", i)
            timestamp_str = entry.get("timestamp", "")

            if not timestamp_str:
                continue

            start_time_micros = self.parse_timestamp(timestamp_str)
            service_name, operation_name = self.extract_service_info(entry)

            # Use single trace ID for entire conversation, unique span ID for each entry
            trace_id = self.conversation_trace_id
            span_id = self.generate_id(
                f"span_{entry_id}_{service_name}_{timestamp_str}", 16
            )

            # Calculate duration
            duration = self.calculate_duration(entry, trajectory_data, i)

            # Get parent span ID
            parent_span_id = self.get_parent_span_id(entry)

            # Format time for display (HH:MM)
            dt = datetime.fromtimestamp(start_time_micros / 1_000_000, tz=UTC)
            time_display = dt.strftime("%H:%M")

            # Method name (simplified from operation)
            method_name = operation_name.replace("_", "").replace("-", "")

            # Create trace entry
            trace_entry = {
                "time": time_display,
                "traceID": trace_id,
                "spanID": span_id,
                "serviceName": service_name,
                "methodName": method_name,
                "operationName": operation_name,
                "startTime": start_time_micros,
                "duration": duration,
                "statusCode": "",  # Always empty as per target format
                "parentSpanID": parent_span_id,
            }

            self.traces.append(trace_entry)

        # Write to CSV
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

        output_path = Path(output_file)
        with output_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.traces)

    def get_conversion_stats(self) -> dict[str, Any]:
        """Get statistics about the conversion."""
        if not self.traces:
            return {}

        service_counts = {}
        parent_child_pairs = 0

        for trace in self.traces:
            service = trace["serviceName"]
            service_counts[service] = service_counts.get(service, 0) + 1
            if trace["parentSpanID"]:
                parent_child_pairs += 1

        return {
            "total_spans": len(self.traces),
            "service_counts": service_counts,
            "parent_child_relationships": parent_child_pairs,
            "root_spans": len(self.traces) - parent_child_pairs,
        }

    def get_span_summaries(self) -> list[dict[str, Any]]:
        """Get human-readable summaries of each span including messages."""
        if not hasattr(self, "trajectory_data"):
            return []

        summaries = []

        for i, entry in enumerate(self.trajectory_data):
            entry_id = entry.get("id", i)
            timestamp_str = entry.get("timestamp", "")
            action = entry.get("action", "")
            source = entry.get("source", "unknown")
            cause = entry.get("cause")

            # Get the corresponding trace entry
            trace_entry = None
            for trace in self.traces:
                if trace["spanID"] == self.generate_id(
                    f"span_{entry_id}_{self.extract_service_info(entry)[0]}_{timestamp_str}",
                    16,
                ):
                    trace_entry = trace
                    break

            if not trace_entry:
                continue

            # Format timestamp for display
            if timestamp_str:
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%H:%M:%S")
            else:
                formatted_time = "Unknown"

            # Extract message content
            message_content = ""
            if "message" in entry:
                if isinstance(entry["message"], str):
                    message_content = entry["message"]
                elif isinstance(entry["message"], dict):
                    message_content = entry["message"].get(
                        "content", str(entry["message"])
                    )
            elif "observation" in entry:
                message_content = str(entry["observation"])
            elif "content" in entry:
                message_content = str(entry["content"])

            # Truncate long messages
            if len(message_content) > 2000:
                message_content = message_content[:1997] + "..."

            # Create summary
            summary = {
                "time": formatted_time,
                "id": entry_id,
                "service": trace_entry["serviceName"],
                "operation": trace_entry["operationName"],
                "duration_ms": trace_entry["duration"]
                / 1000,  # Convert to milliseconds
                "is_child": bool(trace_entry["parentSpanID"]),
                "message": message_content,
                "source": source,
                "action": action,
                "cause": cause,
            }

            summaries.append(summary)

        return summaries
