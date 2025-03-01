import unittest
import tempfile
import os

from knapsack import read_data


class TestReadData(unittest.TestCase):
    def setUp(self):
        """Create a temporary file for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(delete = False, mode = 'w')
        self.filename = self.temp_file.name

    def tearDown(self):
        """Clean up the temporary file after testing."""
        os.remove(self.filename)
    
    def test_valid_file(self):
        """Test small.txt file that has valid data."""
        values = [35, 85, 30, 50, 70, 80, 55]
        weights = [12, 27, 11, 17, 20, 10, 15]
        values_from_f, weights_from_f, w = read_data('data/small.txt')
        
        self.assertEqual(values, values_from_f)
        self.assertEqual(weights, weights_from_f)
        self.assertEqual(w, int(0.75 * sum(weights)))

    def test_invalid_file(self):
        """Test reading an invalid file with mismatched values and weights."""
        with open(self.filename, 'w') as f:
            f.write("2 2\n")
            f.write("34\n")  # Missing weight
            f.write("5 1\n")

        # Check if ValueError is raised
        with self.assertRaises(ValueError) as context:
            read_data(self.filename)

        # Verify the error message
        self.assertEqual(
            str(context.exception), 
            f"Each line in the file must contain exactly two integers: value and weight."
        )

    def test_custom_capacity(self):
        """Test reading very_small.txt with a custom capacity."""
        custom_capacity = 10
        values = [4, 3, 3, 2]
        weights = [5, 4, 3, 2]
        values_from_f, weights_from_f, capacity = read_data('data/very_small.txt', custom_capacity)

        self.assertEqual(values_from_f, values)
        self.assertEqual(weights_from_f, weights)
        self.assertEqual(capacity, custom_capacity)

    def test_file_not_found(self):
        """Test reading a non-existent file."""
        non_existent_file = "non_existent_file.txt"

        # Check if FileNotFoundError is raised
        with self.assertRaises(FileNotFoundError) as context:
            read_data(non_existent_file)

        # Verify the error message
        self.assertEqual(str(context.exception), f"The file '{non_existent_file}' does not exist.")


if __name__ == '__main__':
    unittest.main()
