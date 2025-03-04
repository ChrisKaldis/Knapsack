import unittest
import dimod
import neal
from knapsack import read_data, build_knapsack_bqm


class TestSolution(unittest.TestCase):
    def test_same_solution(self):
        """Test that Simulated Annealing produce the same solution with 
        ExactSolver for small problems."""

        values, weights, capacity = read_data("data/very_small.txt", 10)
        
        # Create BQM
        bqm = build_knapsack_bqm(values, weights, capacity)

        # Solve the problem using the Exact Solver
        exact_sampler = dimod.ExactSolver()
        exact_sampleset = exact_sampler.sample(bqm)
        exact_solution = exact_sampleset.first.sample

        # Solve the problem using Simulated Annealing
        si_an_sampler = neal.SimulatedAnnealingSampler()
        si_an_sampleset = si_an_sampler.sample(bqm, num_reads=25)
        si_an_solution = si_an_sampleset.first.sample

        # Compare the solutions
        self.assertEqual(
            exact_solution, si_an_solution,
            "Exact Solver and Simulated Annealing produced different solutions."
        )


if __name__ == '__main__':
    unittest.main()
