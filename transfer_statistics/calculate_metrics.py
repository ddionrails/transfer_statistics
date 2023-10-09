from numpy.typing import ArrayLike, NDArray
from numpy import arange, float128, sort, sum, cumsum, add, dtype
from numpy.random import choice


def bootstrap_median(values: NDArray[float128], weights: NDArray[float128], runs: int=200):
    indices_full = arange(0, values.size)
    order = values.argsort()
    values = values[order]
    weights = weights[order]

    def single_run() -> dtype[float128]:
        sample_indices = sort(choice(indices_full, size=values.size))
        sample = values[sample_indices]
        sample_weights = weights[sample_indices]
        weights_total_halfed = sum(sample_weights) / 2

        weights_cumulated: dtype[float128] = float128(0)

        # TODO: Figure out typing
        for value, index in enumerate(sample_weights):
            if weights_cumulated >= weights_total_halfed:
                return sample[index]
            weights_cumulated = add(weights_cumulated, value)
        return sample[index]
