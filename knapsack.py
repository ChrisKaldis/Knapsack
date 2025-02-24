import dimod
import neal.sampler


def read_data(filename: str, capacity: int) -> tuple[list, list, int]:
    """Reads data from a given file and calculates the capacity of knapsack.

    Args:
        filename:
        capacity:

    Returns:
        tuple
    """
    costs = []
    weights = []

    with open(filename, 'r') as file:
        for line in file:
            cost, weight = map(int, line.split())
            costs.append(cost)
            weights.append(weight)

    # In case capacity is not given, we define weight capacity to be equal 
    # to 75% of the total weight.
    if not capacity:
        capacity = int(0.75 * sum(weights))

    return costs, weights, capacity


def build_knapsack_bqm(costs: list, weights: list, capacity: int) -> dimod.BinaryQuadraticModel:
    """Construct a bqm for the knapsack problem.
    
    Args:
        costs:
        weights:
        capacity:

    Returns:
        bqm
    """

    n = len(weights)
    Q = {}

    penalty_weight = int(1.2*max(weights))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal terms: -v_i + P * w_i^2 - 2 * P * W * w_i
                Q[(i, j)] = -costs[i] + penalty_weight * (weights[i]**2) - 2 * penalty_weight * capacity * weights[i]
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
    
    # keep repeated samples only once
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

    print("items",selected_items)
    print("costs",selected_costs)
    print("weights",selected_weights)


def main():

    costs, weights, capacity = read_data(filename = 'data/small.txt', capacity = None)
    bqm = build_knapsack_bqm(costs, weights, capacity)
    
    #sampler = dimod.SimulatedAnnealingSampler()
    sampler = neal.sampler.SimulatedAnnealingSampler()
    
    sampleset = sampler.sample(bqm, num_reads=10)
    show_solution(sampleset, costs, weights)


if __name__ == '__main__':
    main()