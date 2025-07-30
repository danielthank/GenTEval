#!/usr/bin/env python3
"""
Request Recorder API

An async API for recording HTTP requests and generating spans from websites.
"""

import asyncio
import time
from typing import Any

from playwright.async_api import async_playwright

from .recorder import RequestRecorder


class RequestRecorderAPI:
    """Async API for recording HTTP requests and generating spans"""

    def __init__(self):
        self.recorders: dict[str, RequestRecorder] = {}

    async def record_requests(
        self, url: str, wait_time: int = 5, headless: bool = False, speed: str = None
    ) -> RequestRecorder:
        """Record HTTP requests for a given URL using CDP session for detailed initiator info"""
        recorder = RequestRecorder()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            context = await browser.new_context()
            page = await context.new_page()
            cdp_session = await context.new_cdp_session(page)

            # Set network speed if specified
            if speed:
                # Playwright network conditions for different speeds
                network_conditions = {
                    "3g": {
                        "offline": False,
                        "downloadThroughput": 1.5 * 1024 * 1024 // 8,
                        "uploadThroughput": 750 * 1024 // 8,
                        "latency": 40,
                    },
                    "4g": {
                        "offline": False,
                        "downloadThroughput": 4 * 1024 * 1024 // 8,
                        "uploadThroughput": 3 * 1024 * 1024 // 8,
                        "latency": 20,
                    },
                    "wifi": {
                        "offline": False,
                        "downloadThroughput": 30 * 1024 * 1024 // 8,
                        "uploadThroughput": 15 * 1024 * 1024 // 8,
                        "latency": 2,
                    },
                }

                if speed.lower() in network_conditions:
                    conditions = network_conditions[speed.lower()]
                    await cdp_session.send(
                        "Network.emulateNetworkConditions", conditions
                    )

            # Enable Network domain to capture detailed request information
            await cdp_session.send("Network.enable")

            # Set up CDP event listeners for detailed initiator information
            await recorder.setup_cdp_listeners(cdp_session)

            print(f"Navigating to: {url}")
            await page.goto(url, wait_until="domcontentloaded")
            real_url = page.url

            # Wait for additional requests
            print(f"Waiting {wait_time} seconds for additional requests...")
            await asyncio.sleep(wait_time)

            await browser.close()

        # Build parent-child relationships after all requests are collected
        recorder.build_parent_child_relationships(root_url=real_url)

        return recorder

    async def get_spans_for_url(
        self,
        url: str,
        wait_time: int = 5,
        headless: bool = False,
        visit_id: str = None,
        speed: str = None,
    ) -> list[dict[str, Any]]:
        """
        Record HTTP requests for a given URL and return a list of spans.

        Args:
            url: The URL to record requests from
            wait_time: Time to wait for additional requests in seconds
            headless: Whether to run browser in headless mode
            visit_id: Optional unique identifier for this visit
            speed: Optional network speed to simulate (e.g., '3g', '4g')

        Returns:
            List of spans in OpenTelemetry format
        """
        try:
            # Record requests for the URL
            recorder = await self.record_requests(url, wait_time, headless, speed)

            # Create a unique key for this recorder
            if visit_id:
                recorder_key = f"{url}:{visit_id}"
            else:
                recorder_key = (
                    f"{url}:{int(time.time() * 1000000)}"  # microsecond precision
                )

            # Store the recorder with the unique key
            self.recorders[recorder_key] = recorder

            # Convert requests to spans format
            spans = []
            for request_info in recorder.requests.values():
                span = recorder.format_request_for_otel_csv(request_info)
                # Convert CSV format to a more structured format
                span_dict = {
                    "time": span["time"],
                    "traceID": span["traceID"],
                    "spanID": span["spanID"],
                    "serviceName": span["serviceName"],
                    "methodName": span["methodName"],
                    "operationName": span["operationName"],
                    "startTime": int(span["startTime"]),
                    "duration": int(span["duration"]),
                    "statusCode": int(span["statusCode"]),
                    "parentSpanID": span["parentSpanID"],
                    "url": request_info.url,
                    "method": request_info.method,
                    "timestamp": request_info.timestamp,
                    "response_size": request_info.response_size,
                    "headers": request_info.headers,
                }
                spans.append(span_dict)

            return spans

        except Exception as e:
            print(f"Error recording requests for {url}: {e}")
            return []

    def get_recorder_for_url(self, url: str) -> RequestRecorder:
        """Get the most recent recorder instance for a specific URL"""
        # Find the most recent recorder for this URL
        matching_recorders = {
            k: v for k, v in self.recorders.items() if k.startswith(url + ":")
        }
        if not matching_recorders:
            return None

        # Return the most recent one (highest timestamp)
        latest_key = max(matching_recorders.keys())
        return matching_recorders[latest_key]

    def get_all_recorders(self) -> dict[str, RequestRecorder]:
        """Get all recorder instances"""
        return self.recorders.copy()

    def print_tree_for_url(self, url: str) -> bool:
        """
        Print the request tree for a given website URL.

        Args:
            url: The URL of the website to print the tree for

        Returns:
            True if the tree was printed successfully, False if no recorder found for URL
        """
        recorder = self.get_recorder_for_url(url)
        if not recorder:
            print(f"No recorder found for URL: {url}")
            print("Available URLs:")
            # Extract unique URLs from recorder keys
            unique_urls = set()
            for recorder_key in self.recorders.keys():
                if ":" in recorder_key:
                    unique_urls.add(recorder_key.split(":", 1)[0])
            for recorded_url in unique_urls:
                print(f"  - {recorded_url}")
            return False

        print(f"\nRequest Tree for: {url}")
        print("=" * (len(url) + 18))

        # Build and print the tree
        tree = recorder.get_request_tree()
        if tree:
            recorder.print_tree(tree)
        else:
            print("No requests recorded for this URL.")

        print()  # Add spacing after the tree
        return True

    async def record_and_print_tree(
        self, url: str, wait_time: int = 5, headless: bool = False
    ) -> bool:
        """
        Record HTTP requests for a given URL and immediately print the tree.

        Args:
            url: The URL to record requests from and print tree for
            wait_time: Time to wait for additional requests in seconds
            headless: Whether to run browser in headless mode

        Returns:
            True if recording and printing was successful, False otherwise
        """
        try:
            # Record requests for the URL
            print(f"Recording requests for: {url}")
            recorder = await self.record_requests(url, wait_time, headless)

            # Store the recorder with unique key
            recorder_key = f"{url}:{int(time.time() * 1000000)}"
            self.recorders[recorder_key] = recorder

            # Print the tree
            return self.print_tree_for_url(url)

        except Exception as e:
            print(f"Error recording and printing tree for {url}: {e}")
            return False

    def print_all_trees(self):
        """Print request trees for all recorded URLs"""
        if not self.recorders:
            print("No URLs have been recorded yet.")
            return

        # Extract unique URLs from recorder keys
        unique_urls = set()
        for recorder_key in self.recorders.keys():
            if ":" in recorder_key:
                unique_urls.add(recorder_key.split(":", 1)[0])

        for url in unique_urls:
            self.print_tree_for_url(url)

    def print_detailed_initiator_info_for_url(self, url: str) -> bool:
        """
        Print detailed initiator information for a given website URL.

        Args:
            url: The URL of the website to print detailed initiator info for

        Returns:
            True if the info was printed successfully, False if no recorder found for URL
        """
        recorder = self.get_recorder_for_url(url)
        if not recorder:
            print(f"No recorder found for URL: {url}")
            print("Available URLs:")
            # Extract unique URLs from recorder keys
            unique_urls = set()
            for recorder_key in self.recorders.keys():
                if ":" in recorder_key:
                    unique_urls.add(recorder_key.split(":", 1)[0])
            for recorded_url in unique_urls:
                print(f"  - {recorded_url}")
            return False

        recorder.print_detailed_initiator_info()
        return True

    async def get_spans_for_scheduled_visits(
        self,
        url: str,
        schedule: list[dict[str, Any]],
        wait_time: int = 5,
        headless: bool = False,
    ) -> dict[int, list[dict[str, Any]]]:
        """
        Record HTTP requests for a single URL multiple times based on a schedule.

        Args:
            url: The URL to record requests from
            schedule: List of dicts with 'start_time', 'duration', 'rate_per_minute', 'speed'
            wait_time: Time to wait for additional requests in seconds per visit
            headless: Whether to run browser in headless mode

        Returns:
            Dictionary mapping start times (in seconds since epoch) to their respective spans
        """
        import time

        results = {}
        start_time = time.time()

        print(f"Starting scheduled visits to {url}")

        # Create a semaphore to control concurrent requests
        semaphore = asyncio.Semaphore(100)  # Allow up to 10 concurrent requests

        tasks = []

        # Process each slot in the schedule
        for slot_index, slot in enumerate(schedule):
            slot_start = start_time + slot["start_time"]
            slot_duration = slot["duration"]
            rate_per_minute = slot["rate_per_minute"]
            speed = slot["speed"]

            # Convert rate per minute to rate per second
            rate_per_second = rate_per_minute / 60.0

            # Calculate the interval between request starts for this slot
            if rate_per_second > 0:
                interval = 1.0 / rate_per_second
            else:
                continue  # Skip slots with 0 rate

            # Schedule all requests for this slot
            slot_end = slot_start + slot_duration
            next_request_time = slot_start

            print(
                f"Slot {slot_index + 1}: {slot_duration}s, {rate_per_minute} visits/min ({rate_per_second:.2f}/sec), {speed} speed"
            )

            while next_request_time < slot_end:
                # Create a task for this specific visit time
                task = asyncio.create_task(
                    self._visit_at_scheduled_time(
                        url,
                        int(next_request_time),
                        wait_time,
                        headless,
                        semaphore,
                        speed,
                        slot_index,
                    )
                )
                tasks.append((int(next_request_time), task))

                next_request_time += interval

        # Wait for all tasks to complete and collect results
        for visit_start_time, task in tasks:
            try:
                spans = await task
                results[visit_start_time] = spans
            except Exception as e:
                print(f"Error during visit at {visit_start_time}: {e}")
                results[visit_start_time] = []

        print(f"Completed scheduled visits. Total visits: {len(results)}")
        return results

    async def _visit_at_scheduled_time(
        self,
        url: str,
        scheduled_time: int,
        wait_time: int,
        headless: bool,
        semaphore: asyncio.Semaphore,
        speed: str,
        slot_index: int,
    ) -> list[dict[str, Any]]:
        """
        Helper method to visit a URL at a specific scheduled time with network speed.

        Args:
            url: The URL to visit
            scheduled_time: The scheduled start time for this visit
            wait_time: Time to wait for additional requests in seconds
            headless: Whether to run browser in headless mode
            semaphore: Semaphore to control concurrent requests
            speed: Network speed to simulate (e.g., '3g', '4g')
            slot_index: Index of the slot in the schedule

        Returns:
            List of spans for this visit
        """
        # Wait until it's time for this request
        current_time = time.time()
        if current_time < scheduled_time:
            await asyncio.sleep(scheduled_time - current_time)

        async with semaphore:
            actual_start_time = int(time.time())
            print(
                f"Visit starting at: {actual_start_time} (scheduled: {scheduled_time}, slot: {slot_index + 1}, speed: {speed})"
            )

            try:
                # Get spans for this visit with unique visit ID including slot and speed info
                visit_id = (
                    f"{scheduled_time}_{actual_start_time}_slot{slot_index}_{speed}"
                )
                spans = await self.get_spans_for_url(
                    url, wait_time, headless, visit_id, speed
                )
                return spans
            except Exception as e:
                print(f"Error during visit at {actual_start_time}: {e}")
                return []
