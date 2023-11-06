from unittest import TestCase

from numpy import array

from transfer_statistics.calculate_metrics import weighted_median


class test_calculate_metrics(TestCase):
    def test_weighted_median(self):
        values = array([1, 2, 3, 4, 5])
        weights = array([1, 1, 1, 1, 1])
        expected_median = 3
        self.assertEqual(expected_median, weighted_median(values=values, weights=weights))
        weights = array([2, 1, 1, 1, 1])
        expected_median = 2.5
        self.assertEqual(expected_median, weighted_median(values=values, weights=weights))
        weights = array([3, 1, 1, 1, 1])
        expected_median = 2
        self.assertEqual(expected_median, weighted_median(values=values, weights=weights))
