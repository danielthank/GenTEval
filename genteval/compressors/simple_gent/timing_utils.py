import time
from contextlib import contextmanager

import torch


class TimingAccumulator:
    """Accumulates timing measurements for CPU and GPU operations."""

    def __init__(self):
        self.cpu_time = 0.0
        self.gpu_time = 0.0

    def add_cpu_time(self, duration: float):
        """Add CPU time in seconds."""
        self.cpu_time += duration

    def add_gpu_time(self, duration: float):
        """Add GPU time in seconds."""
        self.gpu_time += duration

    def get_cpu_time(self) -> float:
        """Get total CPU time in seconds."""
        return self.cpu_time

    def get_gpu_time(self) -> float:
        """Get total GPU time in seconds."""
        return self.gpu_time

    def reset(self):
        """Reset both timers to zero."""
        self.cpu_time = 0.0
        self.gpu_time = 0.0


@contextmanager
def cpu_timer(accumulator: TimingAccumulator | None = None):
    """
    Context manager for measuring CPU time.

    Args:
        accumulator: Optional TimingAccumulator to add the measured time to

    Yields:
        float: The measured CPU time in seconds when context exits
    """
    start_time = time.time()
    measured_time = 0.0

    try:
        yield lambda: measured_time
    finally:
        measured_time = time.time() - start_time
        if accumulator is not None:
            accumulator.add_cpu_time(measured_time)


@contextmanager
def gpu_timer(accumulator: TimingAccumulator | None = None, device: str | None = None):
    """
    Context manager for measuring GPU time with proper CUDA synchronization.

    Args:
        accumulator: Optional TimingAccumulator to add the measured time to
        device: Optional device string (e.g., 'cuda', 'cpu'). If None, auto-detects.

    Yields:
        float: The measured GPU time in seconds when context exits
    """
    measured_time = 0.0

    # Determine if we're actually using GPU
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    if use_cuda:
        # Use CUDA events for accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        try:
            yield lambda: measured_time
        finally:
            end_event.record()
            torch.cuda.synchronize()
            measured_time = (
                start_event.elapsed_time(end_event) / 1000.0
            )  # Convert ms to seconds
            if accumulator is not None:
                accumulator.add_gpu_time(measured_time)
    else:
        # Using CPU timing - count as CPU time
        start_time = time.time()
        try:
            yield lambda: measured_time
        finally:
            measured_time = time.time() - start_time
            if accumulator is not None:
                accumulator.add_cpu_time(measured_time)


class SplitTimer:
    """
    Timer that automatically splits timing between CPU and GPU operations based on context.
    """

    def __init__(self):
        self.accumulator = TimingAccumulator()

    def cpu_context(self):
        """Get a context manager for CPU timing."""
        return cpu_timer(self.accumulator)

    def gpu_context(self, device: str | None = None):
        """Get a context manager for GPU timing."""
        return gpu_timer(self.accumulator, device)

    def get_times(self) -> tuple[float, float]:
        """Get (cpu_time, gpu_time) in seconds."""
        return self.accumulator.get_cpu_time(), self.accumulator.get_gpu_time()

    def reset(self):
        """Reset both timers."""
        self.accumulator.reset()
