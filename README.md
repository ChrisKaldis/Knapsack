[![Open in GitHub Codespaces](
  https://img.shields.io/badge/Open%20in%20GitHub%20Codespaces-333?logo=github)](
  https://codespaces.new/dwave-examples/knapsack?quickstart=1)

# Knapsack

This repository solves the 0/1 Knapsack problem. It is a common problem in
combinatorial optimization where we have a set of $n$ items, each one of 
these has a weight $w_i$ and a value $v_i$. We also have a knapsack with a 
maximum capacity of $W$. The goal is to select a subset of those items that 
are below or equal of the knapsack's capacity and at the same time maximize 
the sum of values. 

The problem can be also formed as a QUBO (quadratic unconstrained binary optimization) $\min x^TQx$ as it seems below.

Maximize the total value of selected items $\sum u_i x_i$.

The total weight of selected items must not exceed the capacity 
$\sum w_i x_i \leq W$.

We can combine them as: $\min -\sum u_ix_i + P(\sum w_ix_i - W)^2$

It is equal to $\min -\sum u_ix_i + P(\sum w_i^2x_i^2 + 2 \sum w_iw_jx_ix_j - 2W\sum w_ix_i + W^2)$

The constant term $PW^2$ can be ignored since it doesn't affect the optimization.


## Usage

### Inputs

The `knapsack.py` accepts the two command line arguments.

1. `-f`: Path to the input file containing item values and weights (default: data/small.txt).
2. `-c`: Maximum weight capacity of the knapsack. If not provided, it will be calculated as 75% of the total weight.

```
$ python knapsack.py -f data/very_small.txt -c 10
```

The input file should contain the value and weight of each item, separated by a space, with one item per line. See the `very_small.txt` below:

```
4 5
3 4
3 3
2 2
```

This file represents 4 items with the following values and weights:

- Item 0: Value = 4, Weight = 5

- Item 1: Value = 3, Weight = 4

- Item 2: Value = 3, Weight = 3

- Item 3: Value = 2, Weight = 2

### Outputs

The output format seems below:

```
Reading data from file: data/very_small.txt
Number of items: 4
Total possible weight: 14
Using capacity: 10
Building the Binary Quadratic Model (BQM)...
Solving the problem using Simulated Annealing...
Solution:
items of solution: [0, 2, 3]
with total value:9
with total weight:10
```

## Code Overview

The `knapsack.py` has 5 functions.

1. `parse_arguments` parsing command-line arguments.
2. `read_data` reads the data from a file.
3. `build_knapsack_bqm` creates the problem's binary quadratic model.
4. `show_solution` translates the answer into useful information.
5. `main` presents how to use the functions above.

There are also some `unittest` in the folder tests you can find and run in order to follow the developement process using the command:

```
$ python -m unittest discover -s tests
```

## License

Released under the Apache License 2.0. See [LICENSE](LISENCE) file. 