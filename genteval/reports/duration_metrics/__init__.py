"""Duration metrics module for modular report generation."""

from .depth_before_after import DepthBeforeAfterMetric
from .time_series_comparison import TimeSeriesComparisonMetric
from .wasserstein_distance import WassersteinDistanceMetric


__all__ = [
    "DepthBeforeAfterMetric",
    "TimeSeriesComparisonMetric",
    "WassersteinDistanceMetric",
]
