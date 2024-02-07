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
