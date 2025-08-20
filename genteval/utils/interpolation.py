"""Time series interpolation utilities for GenTEval."""


def interpolate_missing_values(values: list[float | int | None]) -> list[float]:
    """
    Interpolate missing values in a time series using forward fill, backward fill, and linear interpolation.

    Args:
        values: List of values where None represents missing data

    Returns:
        List of interpolated values with no None entries

    Examples:
        >>> interpolate_missing_values([1.0, None, 3.0, None, 5.0])
        [1.0, 2.0, 3.0, 4.0, 5.0]

        >>> interpolate_missing_values([None, 2.0, None])
        [2.0, 2.0, 2.0]
    """
    if not values:
        return []

    values = values.copy()  # Don't modify original
    n = len(values)

    # Handle edge case where all values are None
    if all(v is None for v in values):
        return [0.0] * n

    # Forward fill from first non-None value
    first_valid = next((i for i, v in enumerate(values) if v is not None), None)
    if first_valid is not None:
        for i in range(first_valid):
            values[i] = values[first_valid]

    # Backward fill from last non-None value
    last_valid = next(
        (i for i, v in enumerate(reversed(values)) if v is not None), None
    )
    if last_valid is not None:
        last_valid = n - 1 - last_valid
        for i in range(last_valid + 1, n):
            values[i] = values[last_valid]

    # Linear interpolation for missing values between valid values
    for i in range(n):
        if values[i] is None:
            # Find previous and next valid values
            prev_idx = next(
                (j for j in range(i - 1, -1, -1) if values[j] is not None),
                None,
            )
            next_idx = next((j for j in range(i + 1, n) if values[j] is not None), None)

            if prev_idx is not None and next_idx is not None:
                # Linear interpolation
                prev_val = values[prev_idx]
                next_val = values[next_idx]
                weight = (i - prev_idx) / (next_idx - prev_idx)
                values[i] = prev_val + weight * (next_val - prev_val)
            elif prev_idx is not None:
                values[i] = values[prev_idx]
            elif next_idx is not None:
                values[i] = values[next_idx]
            else:
                values[i] = 0.0

    return [float(v) for v in values]


def align_time_series_data(
    original_dict: dict[int | float, float | int],
    results_dict: dict[int | float, float | int],
) -> tuple[list[float], list[float]]:
    """
    Align two time series dictionaries by timebucket and interpolate missing values.

    Args:
        original_dict: Dictionary mapping {timebucket: value}
        results_dict: Dictionary mapping {timebucket: value}

    Returns:
        Tuple of (aligned_original_values, aligned_results_values) with interpolated missing values

    Examples:
        >>> original = {1: 10.0, 3: 30.0}
        >>> results = {1: 12.0, 2: 20.0, 3: 32.0}
        >>> align_time_series_data(original, results)
        ([10.0, 20.0, 30.0], [12.0, 20.0, 32.0])
    """
    if not original_dict and not results_dict:
        return [], []

    # Find union of all timebuckets from both datasets
    all_buckets = sorted(set(original_dict.keys()) | set(results_dict.keys()))

    if not all_buckets:
        return [], []

    # Extract values for all timebuckets, marking missing as None
    original_values = [original_dict.get(bucket) for bucket in all_buckets]
    results_values = [results_dict.get(bucket) for bucket in all_buckets]

    # Interpolate missing values
    interpolated_original = interpolate_missing_values(original_values)
    interpolated_results = interpolate_missing_values(results_values)

    return interpolated_original, interpolated_results
