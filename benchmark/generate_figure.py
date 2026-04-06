from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import thewalrus
from scipy.stats import unitary_group

import perm_rs

random_unitary = unitary_group.rvs


def time_permanent(
    func: Callable[[npt.NDArray[np.complex128]], complex],
    matrix: npt.NDArray[np.complex128],
) -> float:
    t0 = perf_counter()
    func(matrix)
    return perf_counter() - t0


def n_best_runs(
    func: Callable[[npt.NDArray[np.complex128]], complex],
    size: int,
    n_reps: int,
    n_best: int,
) -> list[float]:
    sub_times = [
        time_permanent(func, random_unitary(size, random_state=i))
        for i in range(n_reps)
    ]
    return np.mean(sorted(sub_times)[:n_best])


if __name__ == "__main__":
    # Warm up
    m = random_unitary(10)
    for func in [
        perm_rs.permanent,
        perm_rs.permanent_single,
        perm_rs.permanent_multi,
        thewalrus.perm,
    ]:
        time_permanent(func, m)

    # Settings
    n_range = range(2, 31)  # Unitary sizes
    n_reps = 50  # Number of repeats
    n_best = 5  # Best n times to take

    bench_func = partial(n_best_runs, n_reps=n_reps, n_best=n_best)

    # Generate times
    perm_rs_t = []
    perm_rs_single_t = []
    perm_rs_multi_t = []
    walrus_t = []
    for n in n_range:
        print(f"n = {n}")  # noqa: T201

        # perm rs optimal
        perm_rs_t.append(bench_func(perm_rs.permanent, n))

        # perm rs single-threaded
        perm_rs_single_t.append(bench_func(perm_rs.permanent_single, n))

        # perm rs multi-threaded
        perm_rs_multi_t.append(bench_func(perm_rs.permanent_multi, n))

        # thewalurs
        walrus_t.append(bench_func(thewalrus.perm, n))

    path = Path(__file__).parent

    plt.plot(n_range, perm_rs_t, label="perm_rs optimal")
    plt.plot(
        n_range, perm_rs_single_t, label="perm_rs single", linestyle="dotted"
    )
    plt.plot(
        n_range, perm_rs_multi_t, label="perm_rs multi", linestyle="dotted"
    )
    plt.plot(n_range, walrus_t, label="thewalrus")
    plt.xlabel("n")
    plt.ylabel("Time (seconds)")
    plt.yscale("log")
    plt.legend()
    plt.title("Permanent calculation time for different values of n")
    plt.savefig(path / "perm_calc_times.png")
    plt.clf()

    plt.plot(
        n_range,
        np.array(perm_rs_t) / np.array(walrus_t) * 100,
        label="optimal",
    )
    plt.plot(
        n_range,
        np.array(perm_rs_single_t) / np.array(walrus_t) * 100,
        label="single-threaded",
        linestyle="dotted",
    )
    plt.plot(
        n_range,
        np.array(perm_rs_multi_t) / np.array(walrus_t) * 100,
        label="multi-threaded",
        linestyle="dotted",
    )
    plt.xlabel("n")
    plt.ylabel("Relative calculation time (%)")
    plt.axhline(100, color="black", linestyle="--", label="thewalrus")
    plt.title("Relative calculation time of perm_rs to thewalrus.")
    plt.ylim(0, 300)
    plt.legend()
    plt.savefig(path / "perm_relative_times.png")
