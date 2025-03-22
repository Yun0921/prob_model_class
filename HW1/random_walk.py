import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import multiprocessing as mp


# settings
dimensions = [1]  # D dimensions
steps_list = [10**6]  # number of steps
num_walks = 1000  # number of walks


# random walk function 
def random_walk_worker(args):
    D, n = args
    position = np.zeros(D)
    origin_returns = []
    quadrant_counts = defaultdict(int)
    n_minus = 0
    n_plus = 0
    n_zero = 0
    
    for i in range(1, n + 1):
        rnd = np.random.uniform()
        step = 1 if rnd >= 0.5 else -1
        position += step

        if np.all(position == 0):
            n_zero += 1
        elif np.all(position > 0):
            n_plus += 1
        elif np.all(position < 0):
            n_minus += 1

        if np.all(position == 0):
            origin_returns.append(i)

        if not np.any(position == 0):
            key = tuple(np.sign(position))
            quadrant_counts[key] += 1

    m = 0.5 * n_zero + max(n_plus, n_minus)

    return position, origin_returns, quadrant_counts, m


# parallel random walk function
def run_parallel_random_walks(D, n, num_walks):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results_list = pool.map(random_walk_worker, [(D, n)] * num_walks)

    final_positions = []
    origin_return_steps = []
    mnRate_values = []
    quadrant_data = defaultdict(int)

    for final_pos, origin_steps, quadrants, m in results_list:
        final_positions.append(final_pos)
        origin_return_steps.extend(origin_steps)
        mnRate_values.append(m / n)

        for key, count in quadrants.items():
            quadrant_data[key] += count

    return {
        "final_positions": np.array(final_positions),
        "origin_return_steps": np.array(origin_return_steps),
        "quadrant_counts": quadrant_data,
        "mnRate_values": np.array(mnRate_values)
    }


# plot functions
def plot_distance_distributions(results, D, n):
    positions = results[D][n]["final_positions"]
    L1_distances = np.sum(np.abs(positions), axis=1)
    L2_distances = np.sqrt(np.sum(positions**2, axis=1))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(L1_distances, bins=50, density=True, alpha=0.7)
    plt.xlabel("L1 Distance")
    plt.ylabel("Probability Density")
    plt.title(f"L1 Distance Distribution (D={D}, n={n})")

    plt.subplot(1, 2, 2)
    plt.hist(L2_distances, bins=50, density=True, alpha=0.7)
    plt.xlabel("L2 Distance")
    plt.ylabel("Probability Density")
    plt.title(f"L2 Distance Distribution (D={D}, n={n})")
    plt.show()


def plot_m_n_distribution(results, n):
    mnRate_values = results[1][n]["mnRate_values"]

    plt.figure(figsize=(6, 5))
    plt.hist(mnRate_values, bins=50, density=True, alpha=0.7)
    plt.xlabel("m/n")
    plt.ylabel("Probability Density")
    plt.title(f"m/n Distribution (n={n})")
    plt.show()


def plot_quadrant_distribution(results, D, n):
    quadrant_counts = results[D][n]["quadrant_counts"]
    labels = [str(k) for k in quadrant_counts.keys()]
    values = list(quadrant_counts.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, alpha=0.7)
    plt.xlabel("Quadrant")
    plt.ylabel("Total Steps in Quadrant")
    plt.title(f"Quadrant Stay Distribution (D={D}, n={n})")
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.show()


def plot_origin_return_steps_distribution(results, D, n):
    origin_return_steps = results[D][n]["origin_return_steps"]

    plt.figure(figsize=(8, 5))
    plt.hist(origin_return_steps, bins=50, density=True, alpha=0.7)
    plt.xlabel("Steps to Return to Origin")
    plt.ylabel("Probability Density")
    plt.title(f"Distribution of Steps to Return to Origin (D={D}, n={n})")
    plt.show()


# main function
if __name__ == "__main__":
    results = {}
    for D in dimensions:
        results[D] = {}
        for n in steps_list:
            results[D][n] = run_parallel_random_walks(D, n, num_walks)

    for D in dimensions:
        for n in steps_list:
            plot_distance_distributions(results, D, n)
            plot_quadrant_distribution(results, D, n)
            plot_origin_return_steps_distribution(results, D, n)

        if D == 1:
            plot_m_n_distribution(results, n)
