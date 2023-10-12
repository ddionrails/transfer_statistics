from numpy.typing import NDArray
from numpy import arange, float128, sort, sum, add, dtype, divide, mean, floating, full, subtract, array, empty, quantile, multiply
from numpy.random import choice, rand, randint
from typing import Any


def bootstrap_median(values: NDArray[float128], weights: NDArray[float128], runs: int=200):
    indices_full = arange(0, values.size)
    order = values.argsort()
    values = values[order]
    weights = weights[order]

    median = weighted_median(values, weights)

    median_distribution = empty(runs)

    for index, _ in enumerate(median_distribution):
        sample_indices = sort(choice(indices_full, size=values.size))
        sample: NDArray[float128] = values[sample_indices]
        # print(f"c{*sample,}")
        # print(f"c{*weights,}")
        sample_weights = weights[sample_indices]
        median_distribution[index] = weighted_median(sample, sample_weights)

    quantiles = quantile(median_distribution, q=[0.025, 0.975])
    lower_confidence = subtract(multiply(2, median), quantiles[1])
    upper_confidence = subtract(multiply(2, median), quantiles[0])
    return (median, lower_confidence, upper_confidence)

def weighted_median(values, weights) -> floating[Any]:
    """Calculate weighted median on already sorted values."""
    
    weights_total = sum(weights)
    weights_total_halfed = divide(sum(weights), 2)

    lower_weights_cumulated: float128 = float128(0)
    upper_weights_cumulated: float128 = weights_total - weights[0]
    lower_median = None
    upper_median = None


    for index in range(1, weights.size -1):
        upper_weights_cumulated = subtract(upper_weights_cumulated, weights[index+1])

        if (lower_weights_cumulated < weights_total_halfed and
            upper_weights_cumulated < weights_total_halfed):
            lower_median = values[index]
            break
        if (lower_weights_cumulated == weights_total_halfed and
            upper_weights_cumulated < weights_total_halfed or
            lower_weights_cumulated < weights_total_halfed and
            upper_weights_cumulated == weights_total_halfed):
            return mean([values[index], values[index+1]])
        lower_weights_cumulated = add(lower_weights_cumulated, weights[index])

    return lower_median

if __name__ == "__main__":
    values = randint(1, 1000, 10000)
    weights = rand(10000)
    order = values.argsort()
    values = values[order]
    weights = weights[order]
    values = arange(1, 11)
    print(values)
    weights = full(10, 1)
    print(weights)
    print(bootstrap_median(values, weights, 1000))
