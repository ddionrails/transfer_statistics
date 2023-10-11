from numpy.typing import NDArray
from numpy import arange, float128, sort, sum, add, dtype, divide, mean, floating, full, subtract
from numpy.random import choice, rand, randint
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
    
    weights_total = sum(sample_weights)
    weights_total_halfed = divide(sum(sample_weights), 2)

    lower_weights_cumulated: dtype[float128] = sample_weights[0]
    upper_weights_cumulated: dtype[float128] = weights_total - sample_weights[0]
    lower_median = None
    upper_median = None

    print(weights_total_halfed)

    # TODO: Figure out typing
    for index in range(1, sample_weights.size -2):
        upper_weights_cumulated = subtract(upper_weights_cumulated, sample_weights[index+1])

        if lower_weights_cumulated < weights_total_halfed and upper_weights_cumulated < weights_total_halfed:
            print(f"{lower_weights_cumulated}, {upper_weights_cumulated}")
            return sample[index]

        lower_weights_cumulated = add(lower_weights_cumulated, sample_weights[index])
    if not upper_median:
        upper_median = values[0]
    if not lower_median:
        lower_median = values[::-1][0]

    return mean([lower_median, upper_median])

if __name__ == "__main__":
    values = arange(1, 11)
    values = randint(1, 1000, 100)
    weights = rand(100)
    order = values.argsort()
    values = values[order]
    weights = weights[order]
    indices = arange(0, 100)
    print(single_run(values, weights, indices))