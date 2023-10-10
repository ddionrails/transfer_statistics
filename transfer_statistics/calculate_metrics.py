from numpy.typing import NDArray
from numpy import arange, float128, sort, sum, add, dtype, divide, mean, floating, full
from numpy.random import choice
from typing import Any


def bootstrap_median(values: NDArray[float128], weights: NDArray[float128], runs: int=200):
    indices_full = arange(0, values.size)
    order = values.argsort()
    values = values[order]
    weights = weights[order]

def single_run(values, weights, indices_full) -> floating[Any]:
    sample_indices = sort(choice(indices_full, size=values.size))
    sample = values[sample_indices]
    print(sample)
    print(weights)
    sample_weights = weights[sample_indices]
    sample_weights_reversed = sample_weights[::-1]
    weights_total_halfed = divide(sum(sample_weights), 2)

    lower_weights_cumulated: dtype[float128] = sample_weights[0]
    upper_weights_cumulated: dtype[float128] = sample_weights[::-1][0]
    lower_median = None
    upper_median = None

    # TODO: Figure out typing
    for index, value in enumerate(sample_weights):
        if lower_weights_cumulated >= weights_total_halfed and not lower_median:
            try:
                lower_median = sample[index+1]
            except:
                lower_median = sample[index]
            print(f"Lower: {lower_median}")
        if upper_weights_cumulated >= weights_total_halfed and not upper_median:
            try:
                upper_median = sample[::-1][index+1]
            except:
                upper_median = sample[::-1][index]
            print(f"Upper: {upper_median}")

        lower_weights_cumulated = add(lower_weights_cumulated, value)
        upper_weights_cumulated = add(upper_weights_cumulated, sample_weights_reversed[index])
    if not upper_median:
        upper_median = values[0]
    if not lower_median:
        lower_median = values[::-1][0]

    return mean([lower_median, upper_median])
