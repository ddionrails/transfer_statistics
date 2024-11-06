from itertools import repeat
from math import sqrt
import scipy.stats as stats

from numpy import (
    add,
    arange,
    argsort,
    array,
    array2string,
    asarray,
    average,
    cumsum,
    divide,
    empty,
    float64,
    interp,
    mean,
    multiply,
    quantile,
    searchsorted,
    subtract,
)
from numpy import sum as numpy_sum
from numpy import unique
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

    lower_confidence, upper_confidence = quantile(median_distribution, q=[0.025, 0.975])
    return {
        "median": median,
        "median_lower_confidence": lower_confidence,
        "median_upper_confidence": upper_confidence,
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
    try:
        standard_deviation = sqrt(variance)
    except ValueError as error:
        raise ValueError(
            (
                f"Math error with VARIANCE: {variance}, MEAN: {_mean}, "
                f"WEIGHTS: {array2string(unique(weights), separator=', ')}"
            )
        ) from error
    t_score = stats.t.ppf((1 + 0.95) / 2, values.size - 1)
    confidence_interval = t_score * (standard_deviation / sqrt(values.size))
    return {
        "mean": _mean,
        "mean_lower_confidence": _mean - confidence_interval,
        "mean_upper_confidence": _mean + confidence_interval,
    }


def calculate_population_confidence_interval(row, proportion_column, n_column):

    p = row[proportion_column]
    q = 1 - p
    n = row[n_column]
    try:
        stderr = Z_ALPHA * sqrt(p * q / (n - 1.0))
    except ValueError as error:
        print(f"p: {p} | q: {q} | n: {n}")
        raise ValueError from error

    return Series(
        {
            "proportion_lower_confidence": p - stderr,
            "proportion_upper_confidence": p + stderr,
        }
    )
