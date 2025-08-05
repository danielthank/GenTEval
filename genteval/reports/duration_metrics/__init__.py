"""Duration metrics module for modular report generation."""

from .depth_before_after import DepthBeforeAfterMetric
from .percentile_comparison import PercentileComparisonMetric
from .wasserstein_distance import WassersteinDistanceMetric


__all__ = [
    "DepthBeforeAfterMetric",
    "PercentileComparisonMetric",
    "WassersteinDistanceMetric",
]
