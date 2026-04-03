from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import unitary_group
from thewalrus import perm

import perm_rs

random_unitary = unitary_group.rvs


def perm_rs_time(matrix: npt.NDArray[np.complex128]) -> float:
    t0 = perf_counter()
    perm_rs.permanent(matrix)
    return perf_counter() - t0


def thewalrus_time(matrix: npt.NDArray[np.complex128]) -> float:
    t0 = perf_counter()
    perm(matrix)
    return perf_counter() - t0


if __name__ == "__main__":
    # Warm up
    m = random_unitary(10)
    perm_rs_time(m)
    thewalrus_time(m)

    # Settings
    n_range = range(2, 27)  # Unitary sizes
    n_reps = 20  # Number of repeats
    best_n = 5  # Best n times to take

    # Generate times
    perm_rs_times = []
    walrus_times = []
    for n in n_range:
        print(f"n = {n}")  # noqa: T201
        mat = random_unitary(n)

        # perm rs
        sub_times = [perm_rs_time(mat) for _ in range(n_reps)]
        perm_rs_times.append(np.mean(sorted(sub_times)[:best_n]))

        # thewalurs
        sub_times = [thewalrus_time(mat) for _ in range(n_reps)]
        walrus_times.append(np.mean(sorted(sub_times)[:best_n]))

    path = Path(__file__).parent

    plt.plot(n_range, perm_rs_times, label="perm_rs")
    plt.plot(n_range, walrus_times, label="thewalrus")
    plt.xlabel("n")
    plt.ylabel("Time (seconds)")
    plt.yscale("log")
    plt.legend()
    plt.title("Permanent calculation time for different values of n")
    plt.savefig(path / "perm_calc_times.png")
    plt.clf()

    plt.plot(n_range, np.array(perm_rs_times) / np.array(walrus_times) * 100)
    plt.xlabel("n")
    plt.ylabel("Relative calculation time (%)")
    plt.axhline(100, color="black", linestyle="--")
    plt.title("Relative calculation time of perm_rs to thewalrus.")
    plt.savefig(path / "perm_relative_times.png")
