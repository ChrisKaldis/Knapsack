#!/usr/bin/env python3

# Copyright 2025 Christos Kaldis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script solves the 0/1 Knapsack problem.

The Knapsack problem is a classic optimization problem where the goal is to 
select a subset of items with maximum value, given a weight constraint. It
is described as a QUBO (quadratic unconstrained binary optimization) problem
using `dimod` library and then `neal` library that implements Simulated
Annealing in order to solve it. It is tested for small problems using Exact
Solver. 

Usage:
    python knapsack.py -f data/very_small.txt -c 5

Input File Format:
    The input file should contain the value and weight of each item, separated 
    by a space, with one item per line. For example:
        4 5
        3 4
        3 3
        2 2

License:
    Apache Linsence 2.0
"""

import dimod
import neal.sampler

import os
import argparse
import logging


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the Knapsack problem.

    Each problem has objects stored in file and a certain amount of capacity.

    Returns: 
        An argument parser Namespace.
    """
    # Set up argument parser.
    parser = argparse.ArgumentParser(
        description = "Arguments for building Knapsack problem."
    )
    # Add the path of the file with the data.
    parser.add_argument(
        "--file",
        "-f",
        type = str,
        default = "data/small.txt",
        help = (
            "Path to the file containing item values and weights "
            "(default: data/small.txt)."
        )
    )
    # Define the capacity of knapsack.
    parser.add_argument(
        "--capacity",
        "-c",
        type = int,
        required = False,
        help = (
        "the maximum weight that you can carry with the knapsack."
        "If not provided, it will be calculated as 75%% of the total weight."
        )
    )

    return parser.parse_args()


def read_data(
        filename: str, capacity: int | None = None
    ) -> tuple[list[int], list[int], int]:
    """Reads data from a given file and calculates the capacity of knapsack.

    The file contains the value and the weight of each item in a different
    line and they are splitted with space. Files have utf-8 encoding.

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
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            # Check if every item is well defined.
            if len(parts) != 2:
                raise ValueError(
                    ("Each line in the file must contain exactly "
                     "two integers: value and weight.")
                )

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

    q_matrix = dict()
    # TODO: comment the two lines below.
    n = len(weights)

    penalty_weight = int(sum(values)*capacity/sum(weights))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal terms: - v_i + P * w_i^2 - 2 * P * W * w_i
                q_matrix[(i, j)] = (
                    - values[i]
                    + penalty_weight * (weights[i]**2)
                    - 2 * penalty_weight * capacity * weights[i]
                )
            else:
                # Off-diagonal terms: 2 * P * w_i * w_j
                q_matrix[(i, j)] = 2 * penalty_weight * weights[i] * weights[j]

    bqm = dimod.BinaryQuadraticModel.from_qubo(q_matrix)

    return bqm


def show_solution(
    sampleset: dimod.SampleSet, values: list[int], weights: list[int]
    ) -> None:
    """Prints the selected items and infos about the answer.

    Args:
        sampleset: Samples returned by a dimod sampler.
        values: List with the values of items.
        weights: List with the weights of items.
    """

    # gather repeated samples together and create a new SampleSet.
    samples = sampleset.aggregate()
    # keep the lowest energy sample as solution.
    solution = samples.first
    # make a list with the answer's selected items
    selected_items = [i for i, x in solution.sample.items() if x == 1]

    selected_values = list[int]()
    selected_weights = list[int]()

    for i in selected_items:
        selected_values.append(values[i])
        selected_weights.append(weights[i])

    # TODO: check if the solution is valid.

    total_value = sum(selected_values)
    total_weight = sum(selected_weights)

    print(f"items of solution: {selected_items}")
    print(f"with total value:{total_value}")
    print(f"with total weight:{total_weight}")


def main():
    # collect the command line input arguments.
    args = parse_arguments()

    # Set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")
    logger = logging.getLogger(__name__)

    try:
        # Read data from the file
        logger.info("Reading data from file: %s", args.file)
        values, weights, capacity = read_data(
            filename=args.file, capacity=args.capacity
        )
        logger.info("Number of items: %d", len(values))
        logger.info("Total possible weight: %d", sum(weights))
        logger.info("Using capacity: %s", capacity)

        # Build the Binary Quadratic Model (BQM)
        logger.info("Building the Binary Quadratic Model (BQM)...")
        bqm = build_knapsack_bqm(values, weights, capacity)

        # Solve the problem using Simulated Annealing
        logger.info("Solving the problem using Simulated Annealing...")
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads = 25)

        # Present the answer.
        logger.info("Solution:")
        show_solution(sampleset, values, weights)

    except ValueError as e:
        logger.error("Error: %s", e)


if __name__ == "__main__":
    main()
