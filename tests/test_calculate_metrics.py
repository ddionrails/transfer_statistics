from unittest import TestCase

from numpy import array, float64

from transfer_statistics.calculate_metrics import (
    weighted_median,
    weighted_mean_and_confidence_interval,
    weighted_boxplot_sections,
)


class test_calculate_metrics(TestCase):
    def test_weighted_median(self):
        values = array([1, 2, 3, 4, 5])
        weights = array([1, 1, 1, 1, 1])
        expected_median = float64(3)
        self.assertEqual(
            expected_median, weighted_median(values=values, weights=weights)
        )
        weights = array([2, 1, 1, 1, 1])
        expected_median = float64(2.5)
        self.assertEqual(
            expected_median, weighted_median(values=values, weights=weights)
        )
        weights = array([3, 1, 1, 1, 1])
        expected_median = float64(2)
        self.assertEqual(
            expected_median, weighted_median(values=values, weights=weights)
        )

    def test_weighted_mean_and_confidence_interval(self):
        values = array([1, 2, 3, 4, 5])
        weights = array([1, 1, 1, 1, 1])
        expected_mean = float(3)
        result = weighted_mean_and_confidence_interval(values=values, weights=weights)
        self.assertEqual(expected_mean, result["mean"])
        self.assertLess(float(expected_mean - result["mean_lower_confidence"]), 1)
        self.assertGreater(expected_mean - result["mean_upper_confidence"], -1)

    def test_weighted_boxplot_sections(self):
        values = array([1, 2, 3, 4, 5])
        weights = array([1, 1, 1, 1, 1])
        expected_median = 3
        result = weighted_boxplot_sections(values=values, weights=weights)
        self.assertEqual(expected_median, result["boxplot_median"])
        weights = array([2, 1, 1, 1, 1])
        expected_median = 2.5
        result = weighted_boxplot_sections(values=values, weights=weights)
        self.assertEqual(expected_median, result["boxplot_median"])
        weights = array([3, 1, 1, 1, 1])
        expected_median = 2
        result = weighted_boxplot_sections(values=values, weights=weights)
        self.assertEqual(expected_median, result["boxplot_median"])
