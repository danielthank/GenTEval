#!/usr/bin/env python3
"""
Pickle Size Analyzer CLI Tool

A command-line tool to analyze the size breakdown of pickled objects,
with special focus on optimizing machine learning models and large objects.

Usage:
    python pickle_analyzer.py input.pkl --detailed --output report.txt
    python pickle_analyzer.py model.pkl --optimize --remove-docs --compress
"""

import argparse
import gzip
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cloudpickle


class PickleSizeAnalyzer:
    """Enhanced pickle size analyzer with CLI support."""

    def __init__(self, obj: Any, name: str = "object"):
        self.obj = obj
        self.name = name
        self.size_breakdown = {}

    def get_object_size(self, obj: Any, seen=None) -> int:
        """Calculate deep size of an object including all referenced objects."""
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0

        seen.add(obj_id)
        size = sys.getsizeof(obj)

        if isinstance(obj, dict):
            size += sum(
                [
                    self.get_object_size(v, seen) + self.get_object_size(k, seen)
                    for k, v in obj.items()
                ]
            )
        elif hasattr(obj, "__dict__"):
            size += self.get_object_size(obj.__dict__, seen)

        return size

    def analyze_attributes(self, max_depth: int = 2) -> dict[str, dict]:
        """Analyze size of each attribute in the object with configurable depth."""
        results = {}

        def analyze_recursive(obj, prefix="", depth=0):
            print(f"Analyzing {prefix} at depth {depth}")
            if depth > max_depth:
                return

            if hasattr(obj, "__dict__"):
                for attr_name, attr_value in obj.__dict__.items():
                    full_attr_name = f"{prefix}.{attr_name}" if prefix else attr_name

                    # Get basic info about the attribute
                    attr_type = type(attr_value).__name__
                    attr_size = self.get_object_size(attr_value)

                    # Get pickle size of attribute
                    try:
                        pickled_size = len(cloudpickle.dumps(attr_value))
                    except Exception:
                        pickled_size = 0

                    # Check if it's a docstring
                    is_docstring = attr_name == "__doc__" and isinstance(
                        attr_value, str
                    )

                    results[full_attr_name] = {
                        "type": attr_type,
                        "memory_size": attr_size,
                        "pickle_size": pickled_size,
                        "is_docstring": is_docstring,
                        "depth": depth,
                    }

                    # Recurse for nested objects
                    if depth < max_depth and hasattr(attr_value, "__dict__"):
                        analyze_recursive(attr_value, full_attr_name, depth + 1)

        analyze_recursive(self.obj)
        return results

    def get_total_pickle_size(self) -> int:
        """Get total pickle size of the object."""
        try:
            return len(cloudpickle.dumps(self.obj))
        except Exception as e:
            print(f"Error calculating pickle size: {e}")
            return 0

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics about the object."""
        total_pickle_size = self.get_total_pickle_size()
        total_memory_size = self.get_object_size(self.obj)
        attr_data = self.analyze_attributes()

        docstring_size = sum(
            data["pickle_size"] for data in attr_data.values() if data["is_docstring"]
        )

        largest_attrs = sorted(
            attr_data.items(), key=lambda x: x[1]["pickle_size"], reverse=True
        )[:10]

        return {
            "object_name": self.name,
            "object_type": type(self.obj).__name__,
            "total_pickle_size": total_pickle_size,
            "total_memory_size": total_memory_size,
            "num_attributes": len(attr_data),
            "docstring_size": docstring_size,
            "docstring_percentage": (docstring_size / total_pickle_size * 100)
            if total_pickle_size > 0
            else 0,
            "largest_attributes": [
                (name, data["pickle_size"]) for name, data in largest_attrs
            ],
            "compression_ratio": total_memory_size / total_pickle_size
            if total_pickle_size > 0
            else 0,
        }

    def print_detailed_report(self, output_file=None):
        """Print or save detailed analysis report."""
        total_pickle_size = self.get_total_pickle_size()
        attr_data = self.analyze_attributes()
        summary = self.get_summary_stats()

        # Prepare output
        lines = []
        lines.append("=" * 80)
        lines.append("PICKLE SIZE ANALYSIS REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        lines.append("\nOBJECT SUMMARY:")
        lines.append(f"  Name: {summary['object_name']}")
        lines.append(f"  Type: {summary['object_type']}")
        lines.append(
            f"  Total pickle size: {summary['total_pickle_size']:,} bytes ({summary['total_pickle_size'] / 1024 / 1024:.2f} MB)"
        )
        lines.append(
            f"  Total memory size: {summary['total_memory_size']:,} bytes ({summary['total_memory_size'] / 1024 / 1024:.2f} MB)"
        )
        lines.append(f"  Number of attributes: {summary['num_attributes']}")
        lines.append(
            f"  Docstring overhead: {summary['docstring_size']:,} bytes ({summary['docstring_percentage']:.1f}%)"
        )

        lines.append("\nTOP 80 LARGEST ATTRIBUTES:")
        lines.append("-" * 80)
        lines.append(
            f"{'Attribute':<40} {'Type':<18} {'Memory':<12} {'Pickled':<12} {'% Total':<8}"
        )
        lines.append("-" * 80)

        # Sort by pickle size
        sorted_attrs = sorted(
            attr_data.items(), key=lambda x: x[1]["pickle_size"], reverse=True
        )

        for i, (attr_name, data) in enumerate(sorted_attrs[:80]):
            pct = (
                (data["pickle_size"] / total_pickle_size) * 100
                if total_pickle_size > 0
                else 0
            )
            memory_str = f"{data['memory_size']:,}B"
            pickle_str = f"{data['pickle_size']:,}B"

            # Truncate long attribute names
            display_name = attr_name if len(attr_name) <= 39 else attr_name[:36] + "..."

            lines.append(
                f"{display_name:<40} {data['type']:<18} {memory_str:>12} {pickle_str:>12} {pct:>7.1f}%"
            )

        # Docstring analysis
        docstring_attrs = [
            (name, data) for name, data in attr_data.items() if data["is_docstring"]
        ]
        if docstring_attrs:
            lines.append("\nDOCSTRING ANALYSIS:")
            lines.append(f"  Found {len(docstring_attrs)} docstring attributes")
            lines.append(
                f"  Total docstring size: {sum(data['pickle_size'] for _, data in docstring_attrs):,} bytes"
            )

            for name, data in sorted(
                docstring_attrs, key=lambda x: x[1]["pickle_size"], reverse=True
            )[:10]:
                lines.append(f"    {name}: {data['pickle_size']:,} bytes")

        # Type breakdown
        type_breakdown = {}
        for data in attr_data.values():
            attr_type = data["type"]
            if attr_type not in type_breakdown:
                type_breakdown[attr_type] = {"count": 0, "total_size": 0}
            type_breakdown[attr_type]["count"] += 1
            type_breakdown[attr_type]["total_size"] += data["pickle_size"]

        lines.append("\nTYPE BREAKDOWN:")
        lines.append("-" * 50)
        lines.append(f"{'Type':<20} {'Count':<8} {'Total Size':<15} {'Avg Size':<12}")
        lines.append("-" * 50)

        for attr_type, info in sorted(
            type_breakdown.items(), key=lambda x: x[1]["total_size"], reverse=True
        )[:15]:
            avg_size = info["total_size"] / info["count"] if info["count"] > 0 else 0
            lines.append(
                f"{attr_type:<20} {info['count']:<8} {info['total_size']:>10,}B {avg_size:>8,.0f}B"
            )

        # Output results
        report_text = "\n".join(lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report_text)
            print(f"Detailed report saved to: {output_file}")
        else:
            print(report_text)

    def print_summary(self):
        """Print a concise summary."""
        summary = self.get_summary_stats()

        print("\n📊 PICKLE SIZE SUMMARY")
        print(f"{'─' * 50}")
        print(f"Object: {summary['object_name']} ({summary['object_type']})")
        print(
            f"Pickle Size: {summary['total_pickle_size']:,} bytes ({summary['total_pickle_size'] / 1024 / 1024:.2f} MB)"
        )
        print(
            f"Memory Size: {summary['total_memory_size']:,} bytes ({summary['total_memory_size'] / 1024 / 1024:.2f} MB)"
        )
        print(f"Attributes: {summary['num_attributes']}")

        if summary["docstring_size"] > 0:
            print(
                f"📝 Docstrings: {summary['docstring_size']:,} bytes ({summary['docstring_percentage']:.1f}% of total)"
            )

        print("\n🔝 TOP SPACE CONSUMERS:")
        for i, (name, size) in enumerate(summary["largest_attributes"][:5], 1):
            pct = (
                (size / summary["total_pickle_size"]) * 100
                if summary["total_pickle_size"] > 0
                else 0
            )
            print(f"  {i}. {name}: {size:,} bytes ({pct:.1f}%)")


def load_pickle_file(filepath: str) -> Any:
    """Load a pickle file, handling both regular and compressed files."""
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Detect if file is gzip compressed
    is_compressed = False
    try:
        with gzip.open(filepath, "rb") as f:
            f.read(1)  # Try to read one byte
        is_compressed = True
    except:
        is_compressed = False

    # Load the file
    if is_compressed:
        print(f"Loading compressed pickle file: {filepath}")
        with gzip.open(filepath, "rb") as f:
            return cloudpickle.load(f)
    else:
        print(f"Loading pickle file: {filepath}")
        with open(filepath, "rb") as f:
            return cloudpickle.load(f)


def save_json_report(analyzer: PickleSizeAnalyzer, output_path: str):
    """Save analysis results as JSON."""
    summary = analyzer.get_summary_stats()
    attr_data = analyzer.analyze_attributes()

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "analyzer_version": "1.0",
        },
        "summary": summary,
        "attributes": {name: data for name, data in attr_data.items()},
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"JSON report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pickle file size breakdown and identify optimization opportunities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.pkl                          # Basic analysis
  %(prog)s model.pkl --detailed               # Detailed report
  %(prog)s model.pkl --summary --json-output report.json
  %(prog)s model.pkl --detailed --output detailed_report.txt
  %(prog)s model.pkl --max-depth 3            # Deeper attribute analysis
        """,
    )

    parser.add_argument("pickle_file", help="Path to the pickle file to analyze")

    parser.add_argument(
        "--detailed", "-d", action="store_true", help="Show detailed analysis report"
    )

    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Show summary only (default if no other options)",
    )

    parser.add_argument("--output", "-o", help="Save detailed report to file")

    parser.add_argument("--json-output", "-j", help="Save analysis results as JSON")

    parser.add_argument(
        "--max-depth",
        "-m",
        type=int,
        default=2,
        help="Maximum depth for attribute analysis (default: 2)",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress loading messages"
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.pickle_file):
        print(f"Error: File '{args.pickle_file}' not found")
        sys.exit(1)

    try:
        # Load the pickle file
        if not args.quiet:
            print(f"Loading pickle file: {args.pickle_file}")

        obj = load_pickle_file(args.pickle_file)

        if not args.quiet:
            print(f"Successfully loaded object of type: {type(obj).__name__}")

        # Create analyzer
        filename = Path(args.pickle_file).name
        analyzer = PickleSizeAnalyzer(obj, filename)

        # Perform analysis based on arguments
        if args.detailed or args.output:
            analyzer.print_detailed_report(args.output)
        elif args.summary or not any([args.detailed, args.json_output]):
            analyzer.print_summary()

        # Save JSON report if requested
        if args.json_output:
            save_json_report(analyzer, args.json_output)

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise e
        sys.exit(1)
