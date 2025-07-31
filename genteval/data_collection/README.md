# Data Collection Module

This module provides HTTP request tree recording functionality integrated from the request-tree project. It captures web requests with parent-child relationships using Playwright and exports data in OpenTelemetry-compatible format.

## Structure

```
data_collection/
├── __init__.py
├── README.md                      # This file
├── main.py                        # Original main.py for reference
├── request_info.py                # Request data structures
├── recorder.py                    # Core recording functionality
├── request_recorder_api.py        # Async API wrapper
├── schedule/                      # Schedule configuration files
├── utils/
│   ├── __init__.py
│   ├── adjust_duration.py         # Duration adjustment utility
│   └── remove_query_param.py      # Query parameter cleaning utility
└── visualization/
    └── index.html                 # Web-based trace visualizer
```

## Core Components

### RequestInfo (`request_info.py`)
Data class containing HTTP request information including:
- URL, method, timestamp
- Initiator details and parent/child relationships  
- Status codes, timing data, headers
- Request hierarchy metadata

### RequestRecorder (`recorder.py`)
Core class for recording HTTP requests and building parent-child relationships:
- `setup_cdp_listeners()`: Chrome DevTools Protocol integration
- `build_parent_child_relationships()`: Creates request hierarchies
- `get_request_tree()`: Generates tree structures
- `print_tree()`: Console tree visualization
- `save_to_csv()`: OpenTelemetry-compatible CSV export

### RequestRecorderAPI (`request_recorder_api.py`)
Async API wrapper for programmatic usage:
- `record_requests()`: Single URL recording
- `get_spans_for_url()`: Retrieve span data
- `record_and_print_tree()`: Combined record and display
- `get_spans_for_scheduled_visits()`: Multi-visit scheduling
- Rate limiting and concurrent request handling
- Network speed simulation (3G, 4G, WiFi)

## CLI Usage

The data collection functionality is available through the `collect_request_data` command:

```bash
# Record and print tree for a single URL
collect_request_data --url https://example.com

# Record with custom wait time and save to CSV
collect_request_data --url https://example.com --wait 10 --output results.csv

# Multiple visits mode with rate and network speed control
collect_request_data --url https://example.com --multiple-visits --rate 10 20 30 --speed 3g 4g wifi --interval 60

# Using schedule file for complex timing patterns
collect_request_data --url https://example.com --multiple-visits --schedule-file schedule/4g_wifi_60min.txt
```

### Command Line Options

- `--url, -u`: **Required.** URL to record
- `--wait, -w`: Wait time in seconds for additional requests (default: 5)
- `--output, -o`: Output CSV file path
- `--detailed-initiators`: Show detailed initiator information
- `--multiple-visits`: Enable multiple visits mode
- `--interval`: Interval duration in seconds for each slot (default: 60)
- `--rate, -r`: Rate of visits per minute for each slot
- `--speed`: Network speed for each slot (3g, 4g, wifi)
- `--schedule-file`: Path to schedule file with rate/speed configuration

## Programmatic Usage

```python
from genteval.data_collection.request_recorder_api import RequestRecorderAPI
import asyncio

async def example():
    api = RequestRecorderAPI()
    
    # Record spans for a single URL
    spans = await api.get_spans_for_url("https://example.com", wait_time=5)
    
    # Record and print tree
    await api.record_and_print_tree("https://example.com", wait_time=5)
    
    # Multiple URLs with rate limiting
    urls = ["https://example.com", "https://httpbin.org"]
    results = await api.get_spans_for_multiple_urls(
        urls, 
        rate_per_second=1.0,
        wait_time=5,
        headless=True
    )
    
    return results

# Run the example
results = asyncio.run(example())
```

## Output Format

The tool generates OpenTelemetry-compatible CSV files with columns:
- `time`: Request time (HH:MM format)
- `traceID`: Unique trace identifier
- `spanID`: Unique span identifier  
- `serviceName`: Service name (extracted from URL)
- `methodName`: HTTP method
- `operationName`: HTTP method + path
- `startTimeMillis`: Start time in milliseconds
- `startTime`: Start time in microseconds
- `duration`: Request duration in milliseconds
- `statusCode`: HTTP status code
- `parentSpanID`: Parent span ID (for request relationships)

## Utilities

### Duration Adjustment (`utils/adjust_duration.py`)
Post-processing tool for adjusting span durations to properly encompass child spans:

```bash
adjust_duration input.csv output.csv
```

### Query Parameter Removal (`utils/remove_query_param.py`)
Utility for cleaning query parameters from operationName columns:

```bash
remove_query_param input.csv output.csv
```

## Visualization

The `visualization/index.html` file provides a web-based trace visualizer with:
- Network graph view using vis.js showing request relationships
- Timeline view displaying request hierarchies and timing
- Interactive features with hover details and collapsible groups

## Integration with GenTEval

This data collection module is designed to work seamlessly with GenTEval's existing functionality:

1. **Trace Generation**: Collect real-world HTTP request traces for training data
2. **Evaluation Data**: Generate baseline datasets for compression algorithm evaluation  
3. **Benchmark Creation**: Create standardized request patterns for performance testing
4. **Data Preprocessing**: Clean and format trace data for machine learning pipelines

The module integrates with GenTEval's existing compressors and evaluators to provide a complete trace data lifecycle from collection to evaluation.

## Dependencies

- `playwright>=1.53.0`: Browser automation and Chrome DevTools Protocol
- Python >=3.13
- Standard library: `asyncio`, `csv`, `hashlib`, `urllib`

## Schedule File Format

Schedule files allow complex timing and network speed patterns:

```
rate: [60] * 60
speed: ["4g", "4g", "wifi", "wifi"] * 15
```

This creates 60 intervals with specified visit rates and alternating network speeds.