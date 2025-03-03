import unittest

from knapsack import read_data, build_knapsack_bqm


class TestBuildKnapsackBqm(unittest.TestCase):

    def test_build_knapsack_bqm(self):
        """Test the BQM construction using data from the smallest file."""
        # Function tested on test_read_data.py
        values_f, weights_f, capacity = read_data('data/very_small.txt', 10)
        
        # Create the BQM 
        bqm = build_knapsack_bqm(values_f, weights_f, capacity)

        # This is equal to the penalty_weight inside the function.
        # it is int(10 * max(weights_f)
        p = 50

        # Linear terms: -v_i + P * w_i^2 - 2 * P * W * w_i
        expected_linear_terms = {
            0: -4 + p * 5**2 - 2 * p * capacity * 5, 
            1: -3 + p * 4**2 - 2 * p * capacity * 4,
            2: -3 + p * 3**2 - 2 * p * capacity * 3,
            3: -2 + p * 2**2 - 2 * p * capacity * 2
        }

        # Quadratic terms: 2 * P * w_i * w_j
        expected_quadratic_terms = {
            (1, 0): 2 * p * 4 * 5,
            (2, 0): 2 * p * 3 * 5,
            (2, 1): 2 * p * 3 * 4,
            (3, 0): 2 * p * 2 * 5,
            (3, 1): 2 * p * 2 * 4,
            (3, 2): 2 * p * 2 * 3
        }
        
        self.assertEqual(bqm.linear, expected_linear_terms)
        self.assertEqual(bqm.quadratic, expected_quadratic_terms)

if __name__ == '__main__':
    unittest.main()