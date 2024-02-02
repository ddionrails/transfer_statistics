from math import sqrt


from numpy import (
    add,
    arange,
    argsort,
    array,
    average,
    cumsum,
    divide,
    empty,
    float64,
    mean,
    multiply,
    quantile,
    interp,
    searchsorted,
    subtract,
)

from numpy import sum as numpy_sum

from numpy.random import choice
from numpy.typing import NDArray


def bootstrap_median(
    values: NDArray[float64], weights: NDArray[float64], runs: int = 200
) -> dict[str, float64]:
    indices_full = arange(0, values.size)

    median = weighted_median(values, weights)

    median_distribution = empty(runs)

    for index, _ in enumerate(median_distribution):
        sample_indices = choice(indices_full, size=values.size)
        sample: NDArray[float64] = values[sample_indices]
        sample_weights = weights[sample_indices]
        median_distribution[index] = weighted_median(sample, sample_weights)

    quantiles = quantile(median_distribution, q=[0.025, 0.975])
    lower_confidence = subtract(multiply(2, median), quantiles[1])
    upper_confidence = subtract(multiply(2, median), quantiles[0])
    return {
        "median": median,
        "median_lower_confidence": lower_confidence,
        "median_upper_confidence": upper_confidence,
    }


def weighted_median(values: NDArray[float64], weights: NDArray[float64]):
    sorted_indices = argsort(values)
    values_sorted = values[sorted_indices]
    weights_sorted = weights[sorted_indices]

    cum_weights = cumsum(weights_sorted)

    median_index = searchsorted(cum_weights, cum_weights[-1] / 2.0)

    weighted_median_value = values_sorted[median_index]

    return weighted_median_value


def weighted_median_slow(values: NDArray[float64], weights: NDArray[float64]) -> float64:
    """Calculate weighted median on already sorted values."""

    weights_total = numpy_sum(weights)
    weights_total_halfed = divide(numpy_sum(weights), 2)

    lower_weights_cumulated: float64 = float64(0.0)
    upper_weights_cumulated: float64 = weights_total - weights[0]
    median = float64(0.0)

    for index in range(1, weights.size - 1):
        upper_weights_cumulated = subtract(upper_weights_cumulated, weights[index + 1])

        if (
            lower_weights_cumulated < weights_total_halfed
            and upper_weights_cumulated < weights_total_halfed
        ):
            median = values[index]
            break
        if (
            lower_weights_cumulated == weights_total_halfed
            and upper_weights_cumulated < weights_total_halfed
            or lower_weights_cumulated < weights_total_halfed
            and upper_weights_cumulated == weights_total_halfed
        ):
            median = mean([values[index], values[index + 1]])
            break
        lower_weights_cumulated = add(lower_weights_cumulated, weights[index])

    return median


def weighted_boxplot_sections(
    values: NDArray[float64], weights: NDArray[float64]
) -> dict[str, float64]:
    values = array(values)
    quantiles = array([0.25, 0.5, 0.75])

    weights = array(weights)

    cumulated_quantiles = cumsum(weights) - 0.5 * weights
    cumulated_quantiles = cumulated_quantiles / numpy_sum(weights)
    _weighted_quartiles = interp(quantiles, cumulated_quantiles, values)

    inter_quartile_range = _weighted_quartiles[2] - _weighted_quartiles[0]
    low = _weighted_quartiles[0] - (1.5 * inter_quartile_range)
    high = _weighted_quartiles[2] + (1.5 * inter_quartile_range)
    if low < values[0]:
        low = values[0]
    if high > values[-1]:
        high = values[-1]

    return {
        "lower_quartile": _weighted_quartiles[0],
        "boxplot_median": _weighted_quartiles[1],
        "upper_quartile": _weighted_quartiles[2],
        "lower_whisker": low,
        "upper_whisker": high,
    }


def weighted_mean_and_confidence_interval(
    values: NDArray[float64], weights: NDArray[float64]
) -> dict[str, float64]:
    _mean = average(values, weights=weights)
    variance = average((values - _mean) ** 2, weights=weights)
    standard_deviation = sqrt(variance)
    confidence_interval = 0.95 * (standard_deviation / sqrt(values.size))
    return {
        "mean": _mean,
        "mean_lower_confidence": _mean - confidence_interval,
        "mean_upper_confidence": _mean + confidence_interval,
    }
