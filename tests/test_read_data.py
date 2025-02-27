import unittest

from knapsack import read_data


class TestReadData(unittest.TestCase):

    def test_read_data(self):
        values = [35, 85, 30, 50, 70, 80, 55]
        weights = [12, 27, 11, 17, 20, 10, 15]
        values_f, weights_f, w = read_data('data/small.txt', None)
        self.assertEqual(values, values_f)
        self.assertEqual(weights, weights_f)

if __name__ == '__main__':
    unittest.main()
