from itertools import repeat
from math import sqrt

from numpy import (
    arange,
    argsort,
    array,
    asarray,
    average,
    cumsum,
    empty,
    float64,
    interp,
    multiply,
    quantile,
    searchsorted,
    subtract,
)
from numpy import sum as numpy_sum
from numpy.random import choice
from numpy.typing import NDArray

from pandas import Series

from transfer_statistics.helpers import multiprocessing_wrapper

Z_ALPHA = 1.96


def bootstrap_median(
    values: NDArray[float64], weights: NDArray[float64], runs: int = 200, pool=None
) -> dict[str, float64]:
    indices_full = arange(0, values.size)

    median = weighted_median(values, weights)

    median_distribution = empty(runs)

    if pool:
        median_distribution = asarray(
            pool.map(
                multiprocessing_wrapper,
                repeat([_parallel_bootstrap_median, values, weights], runs),
            ),
            float64,
        )
    else:
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


def weighted_mean_and_confidence_interval(
    values: NDArray[float64], weights: NDArray[float64], runs: int = 200
) -> dict[str, float64]:
    all_indices = arange(0, values.size)
    _mean = average(values, weights=weights)

    mean_distribution = empty(runs)

    for index, _ in enumerate(mean_distribution):
        sample_indices = choice(all_indices, size=values.size)
        sample: NDArray[float64] = values[sample_indices]
        sample_weights: NDArray[float64] = weights[sample_indices]
        mean_distribution[index] = average(sample, weights=sample_weights)

    quantiles = quantile(mean_distribution, q=[0.025, 0.975])
    lower_confidence = subtract(multiply(2, _mean), quantiles[1])
    upper_confidence = subtract(multiply(2, _mean), quantiles[0])
    return {
        "mean": _mean,
        "mean_lower_confidence": lower_confidence,
        "mean_upper_confidence": upper_confidence,
    }


def _parallel_bootstrap_median(arguments) -> float64:
    values = arguments[0]
    weights = arguments[1]
    sample_indices = choice(values.size, size=values.size)
    sample: NDArray[float64] = values[sample_indices]
    sample_weights = weights[sample_indices]
    median = weighted_median(sample, sample_weights)
    del arguments, values, weights, sample_indices, sample, sample_weights
    return median


def weighted_median(values: NDArray[float64], weights: NDArray[float64]) -> float64:
    sorted_indices = argsort(values)
    values_sorted = values[sorted_indices]
    weights_sorted = weights[sorted_indices]

    cum_weights = cumsum(weights_sorted)

    median_index = searchsorted(cum_weights, cum_weights[-1] / 2.0)

    if median_index >= values_sorted.size:
        median_index = values_sorted.size - 1
    weighted_median_value = values_sorted[median_index]

    return weighted_median_value


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


def calculate_population_confidence_interval(row, proportion_column, n_column):

    p = row[proportion_column]
    q = 1 - p
    n = row[n_column]
    stderr = Z_ALPHA * sqrt(p * q / n)
    return Series(
        {
            "proportion_lower_confidence": p - stderr,
            "proportion_upper_confidence": p + stderr,
        }
    )
