"""
Duration Adjustment Classes

Contains SpanData and DurationAdjuster classes for adjusting span durations
to match the latest end time of their children.
"""

import csv
from pathlib import Path


class SpanData:
    """Represents a span with timing information"""

    def __init__(self, row: dict[str, str]):
        self.trace_id = row["traceID"]
        self.span_id = row["spanID"]
        self.parent_span_id = row["parentSpanID"] if row["parentSpanID"] else None
        self.start_time = int(row["startTime"])
        self.original_duration = int(row["duration"])
        self.adjusted_duration = int(row["duration"])  # Will be modified

        # Store original values for change tracking
        self.original_start_time = int(row["startTime"])

        # Store all other fields as-is
        self.other_fields = {
            k: v for k, v in row.items() if k not in ["startTime", "duration"]
        }

        self.children: list[str] = []  # List of child span IDs

    @property
    def end_time(self) -> int:
        """Calculate end time in milliseconds using adjusted duration"""
        return self.start_time + self.adjusted_duration

    def to_csv_row(self) -> dict[str, str]:
        """Convert back to CSV row format with adjusted duration"""
        row = self.other_fields.copy()
        row["startTime"] = str(self.start_time)
        row["duration"] = str(self.adjusted_duration)
        return row


class DurationAdjuster:
    """Adjusts span durations based on children's end times"""

    def __init__(self, verbose: bool = True):
        self.spans: dict[str, SpanData] = {}
        self.fieldnames: list[str] = []
        self.verbose = verbose

    def load_csv(self, input_file: str) -> None:
        """Load spans from CSV file"""
        with Path(input_file).open(newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            self.fieldnames = reader.fieldnames or []

            for row in reader:
                span = SpanData(row)
                self.spans[span.span_id] = span

        if self.verbose:
            print(f"Loaded {len(self.spans)} spans from {input_file}")

    def build_relationships(self) -> None:
        """Build parent-child relationships between spans"""
        for span in self.spans.values():
            if span.parent_span_id and span.parent_span_id in self.spans:
                parent = self.spans[span.parent_span_id]
                parent.children.append(span.span_id)

        # Count spans with children
        parents_count = sum(1 for span in self.spans.values() if span.children)
        if self.verbose:
            print(f"Built relationships: {parents_count} parent spans found")

    def adjust_durations(self) -> int:
        """
        Adjust durations so parent spans end at or after their children's latest end time.
        Returns the number of spans that were adjusted.
        """
        adjusted_count = 0

        # Process spans in bottom-up order (children before parents)
        # We'll use a simple approach: keep adjusting until no more changes needed
        max_iterations = len(self.spans)  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            changes_made = False

            for span in self.spans.values():
                if not span.children:
                    continue  # Skip leaf spans

                # Find the latest end time and earliest start time among all children
                latest_child_end = 0
                earliest_child_start = float("inf")
                for child_id in span.children:
                    if child_id in self.spans:
                        child = self.spans[child_id]
                        child_end = child.end_time
                        child_start = child.start_time
                        latest_child_end = max(latest_child_end, child_end)
                        earliest_child_start = min(earliest_child_start, child_start)

                # Store original values to detect changes
                old_duration = span.adjusted_duration
                old_start_time = span.start_time
                span_changed = False

                # If parent ends before latest child, adjust parent's duration
                if latest_child_end > span.end_time:
                    span.adjusted_duration = latest_child_end - span.start_time
                    span_changed = True

                # If parent starts after earliest child, adjust parent's start time
                # (keep the end time, so increase the duration)
                if earliest_child_start < span.start_time:
                    # Calculate the current end time before we change start time
                    current_end_time = span.end_time
                    # Adjust start time to earliest child
                    span.start_time = earliest_child_start
                    # Recalculate duration to maintain the same end time
                    span.adjusted_duration = current_end_time - span.start_time
                    span_changed = True

                # Track changes for iteration control and counting
                if span_changed:
                    changes_made = True
                    # Count as adjusted if this is the first time we change either duration or start time
                    if (
                        old_duration == span.original_duration
                        and old_start_time == span.original_start_time
                    ):
                        adjusted_count += 1

            # If no changes were made in this iteration, we're done
            if not changes_made:
                break

        if self.verbose:
            print(f"Adjusted {adjusted_count} spans over {iteration} iterations")
        return adjusted_count

    def write_csv(self, output_file: str) -> None:
        """Write adjusted spans to CSV file"""
        with Path(output_file).open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

            for span in self.spans.values():
                writer.writerow(span.to_csv_row())

        if self.verbose:
            print(f"Saved adjusted spans to {output_file}")

    def print_adjustment_summary(self) -> None:
        """Print summary of adjustments made"""
        adjusted_spans = [
            span
            for span in self.spans.values()
            if (
                span.adjusted_duration != span.original_duration
                or span.start_time != span.original_start_time
            )
        ]

        if not adjusted_spans:
            print("No spans required adjustment.")
            return

        print("\nAdjustment Summary:")
        print(
            f"{'Span ID':<16} {'Orig Start':<12} {'Adj Start':<12} {'Orig Dur':<10} {'Adj Dur':<10} {'Changes':<20}"
        )
        print("-" * 90)

        for span in adjusted_spans:
            duration_change = span.adjusted_duration - span.original_duration
            start_change = span.start_time - span.original_start_time

            changes = []
            if start_change != 0:
                changes.append(f"start{start_change:+d}ms")
            if duration_change != 0:
                changes.append(f"dur{duration_change:+d}ms")
            changes_str = ", ".join(changes)

            print(
                f"{span.span_id:<16} {span.original_start_time:<12} "
                f"{span.start_time:<12} {span.original_duration:<10} "
                f"{span.adjusted_duration:<10} {changes_str:<20}"
            )
