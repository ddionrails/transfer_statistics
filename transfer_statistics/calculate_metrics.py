from typing import Any

from math import sqrt


from numpy import (
    add,
    arange,
    average,
    divide,
    empty,
    float128,
    floating,
    full,
    mean,
    multiply,
    quantile,
    sort,
    subtract,
    sum,
)
from numpy.random import choice, rand, randint
from numpy.typing import NDArray


def bootstrap_median(
    values: NDArray[float128], weights: NDArray[float128], runs: int = 200
):
    indices_full = arange(0, values.size)
    order = values.argsort()
    values = values[order]
    weights = weights[order]

    median = weighted_median(values, weights)

    median_distribution = empty(runs)

    for index, _ in enumerate(median_distribution):
        sample_indices = sort(choice(indices_full, size=values.size))
        sample: NDArray[float128] = values[sample_indices]
        sample_weights = weights[sample_indices]
        median_distribution[index] = weighted_median(sample, sample_weights)

    quantiles = quantile(median_distribution, q=[0.025, 0.975])
    lower_confidence = subtract(multiply(2, median), quantiles[1])
    upper_confidence = subtract(multiply(2, median), quantiles[0])
    return (median, lower_confidence, upper_confidence)


def weighted_median(values: NDArray[float128], weights: NDArray[float128]) -> float128:
    """Calculate weighted median on already sorted values."""

    weights_total = sum(weights)
    weights_total_halfed = divide(sum(weights), 2)

    lower_weights_cumulated: float128 = float128(0)
    upper_weights_cumulated: float128 = weights_total - weights[0]
    median = float128(0.0)

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


def weighted_mean_and_confidence_interval(
    values: NDArray[float128], weights: NDArray[float128]
) -> tuple[float128, float128, float128]:
    _average = average(values, weights=weights)
    variance = average((values - _average) ** 2, weights=weights)
    standard_deviation = sqrt(variance)
    confidence_interval = 0.95 * (standard_deviation / sqrt(values.size))
    return (_average, _average - confidence_interval, _average + confidence_interval)
