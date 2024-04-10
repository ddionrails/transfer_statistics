from random import choices

import pytest
from pandas import DataFrame


@pytest.fixture(name="simple_dataframe")
def _simple_dataframe(request):
    data = {
        "syear": [
            1990,
            1990,
            1990,
            1991,
            1991,
            1991,
        ],
        "group": ["a", "b", "a", "a", "b", "a"],
        "number": [22, 24, 27, 45, 67, 24],
        "weight": [1, 1, 1, 1, 1, 1],
    }
    if getattr(request, "cls", False):
        request.cls.simple_dataframe = DataFrame(data)
        yield
    return DataFrame(data)


@pytest.fixture(name="complex_dataframe")
def _testdata_dataframe(request):
    data = {
        "syear": [
            1990,
            1990,
            1990,
            1991,
            1991,
            1991,
            1992,
            1992,
            1992,
            1993,
            1993,
            1993,
        ],
        "age_gr": choices([1, 2, 3], k=12),
        "agre": choices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], k=12),
        "bildungsniveau": choices([1, 2, 3, 4, 5], k=12),
        "bmi": [None, None, None, *choices([0, 2, 3, 4, 5], k=9)],
    }
    if getattr(request, "cls", False):
        request.cls.simple_dataframe = DataFrame(data)
        yield
    return DataFrame(data)
