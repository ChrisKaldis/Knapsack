import dimod
import neal.sampler

import argparse
import os


def read_data(
        filename: str,
        capacity: int | None = None
    ) -> tuple[list[int], list[int], int]:
    """Reads data from a given file and calculates the capacity of knapsack.

    The file contains the value and the weight of each item in a different
    line and they are splitted with space. 

    Args:
        filename: the path of the file that contains the values and weights.
        capacity: knapsack's capacity.

    Returns:
        tuple with a list of values, a list of weights and the capacity.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If an item doesn't have both `value` and `weight`.
    """
    # Check if the file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")

    values = list[int]()
    weights = list[int]()

    # Fill the values and weights lists 
    with open(filename, "r") as file:
        for line in file:
            parts = line.split()
            # Check if every item is well defined.
            if len(parts) != 2:
                raise ValueError(f"Each line in the file must contain exactly two integers: value and weight.")

            value, weight = map(int, parts)
            values.append(value)
            weights.append(weight)

    # In case capacity is not given, we define weight capacity to be equal 
    # to 75% of the total weight.
    if not capacity:
        capacity = int(0.75 * sum(weights))

    return values, weights, capacity


def build_knapsack_bqm(
        values: list[int], weights: list[int], capacity: int
    ) -> dimod.BinaryQuadraticModel:
    """Construct a bqm for the knapsack problem.
    
    Creates a Binary Quadratic Model based on a QUBO formulation of a 0/1
    knapsack problem. It is described in the README.md file.

    Args:
        values: List with the values of items.
        weights: List with the weights of items.
        capacity: Maximum weight capacity of the knapsack.

    Returns:
        BinaryQuadraticModel representing the 0/1 knapsack problem.
    """

    Q = {}
    n = len(weights)

    penalty_weight = int(10 * max(weights))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal terms: - v_i + P * w_i^2 - 2 * P * W * w_i
                Q[(i, j)] = (
                    - values[i]
                    + penalty_weight * (weights[i]**2)
                    - 2 * penalty_weight * capacity * weights[i]
                )
            else:
                # Off-diagonal terms: 2 * P * w_i * w_j
                Q[(i, j)] = 2 * penalty_weight * weights[i] * weights[j]

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    return bqm


def show_solution(sampleset: dimod.SampleSet, costs, weights):
    """It prints the selected items and infos about the answer.

    Args:
        sampleset:
        costs:
        weights:
    """

    #print(sampleset)
    # gather repeated samples
    samples = sampleset.aggregate()
    #print("after aggregate", samples)
    # keep first solution
    best_solution = samples.first
    #print("best solution",best_solution)
    # make a list with the answer's selected items 
    selected_items = [i for i, x in best_solution.sample.items() if x == 1]
    #print("slected items",selected_items)

    selected_costs = []
    selected_weights = []

    for i in selected_items:
        selected_costs.append(costs[i])
        selected_weights.append(weights[i])

    # print("items",selected_items)
    # print("costs",selected_costs)
    # print("weights",selected_weights)


def main():

    parser = argparse.ArgumentParser(
        description = "Arguments for building Knapsack problem."
    )
    parser.add_argument(
        "--f",
        type = str,
        default = "data/small.txt",
        help = "path of the file with the items values and weights."
    )
    parser.add_argument(
        "--c",
        type = int,
        required = False,
        help = "the maximum weight that you can carry with the knapsack."
    )
    args = parser.parse_args()

    costs, weights, capacity = read_data(filename = args.f, capacity = args.c)
    bqm = build_knapsack_bqm(costs, weights, capacity)
    
    #sampler = dimod.SimulatedAnnealingSampler()
    #sampler = neal.sampler.SimulatedAnnealingSampler()
    sampler = dimod.ExactSolver()

    sampleset = sampler.sample(bqm)
    show_solution(sampleset, costs, weights)


if __name__ == '__main__':
    main()