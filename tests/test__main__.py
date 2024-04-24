from unittest import TestCase

import pytest
from pandas import DataFrame

from transfer_statistics.calculate_metrics import weighted_boxplot_sections


@pytest.mark.usefixtures("simple_dataframe")
class TestMain(TestCase):
    simple_dataframe: DataFrame

    def test_calculate_numerical_statistics(self):
        print(self.simple_dataframe)

        def _test(frame, variable_name):
            frame = frame.sort_values(by=[variable_name])
            return weighted_boxplot_sections(frame[variable_name], frame["weight"])

        rest = self.simple_dataframe.groupby(by=["syear", "group"])
        rest.apply(_test, variable_name="number")
        print(rest)
