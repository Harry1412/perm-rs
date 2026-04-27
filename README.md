# perm-rs

This library contains a Rust re-implementation of the permanent calculation function from the [thewalrus](https://github.com/XanaduAI/thewalrus) module, using maturin and PyO3 to then make this callable from Python. This is then extended to support calculation across multiple processes, which offers signifcant performance improvements at large matrix sizes. 

The library is mainly intended for testing how rust-enhanced modules can be developed for Python and then distributed via pypi to different platforms.

## Install

Rust must first be installed if not already, the instructions for which can be found [here](https://rust-lang.org/tools/install/).

Maturin is then utilised for building the rust code and generating Python bindings, this requires a virtual or conda environment. All requirements, including maturin, may be installed using the below command. This requires pip 25.1 - use `pip install pip --upgrade` to update to this.

```bash
(venv) pip install --group dev
```

Once installed, the library is built using:

```bash
(venv) maturin develop
```

For optimal performance, it should be built with the release profile:

```bash
(venv) maturin develop --release
```

### UV

[uv](https://docs.astral.sh/uv/) can alternatively be utilised to simplify the above process. The command `uv sync` will install all requirements into a virtual environment and build the library in release mode.

## Benchmarking

The performance of the library can be benchmarked against thewalrus implementation using the following command:

```bash
pytest benchmark
```

Alternatively, a more comprehensive set of matrix sizes can be compared using the script in ``benchmark/generate_figure.py``, also comparing the single and multi-threaded approaches. The following is the output comparison between the two libraries when run using a AMD Ryzen 9 9950x3D CPU:

![perm_rs & thewalrus comparison](/benchmark/perm_calc_times.png)

There are two current trends to observe from this: 
- The single-threaded case is faster than thewalrus for small matrix sizes, but trends towards similar runtimes for large n.
- The multi-threaded case is much slower at large n, due to the overhead from this which sets the minimum time to be ~1ms, but after passing a threshold at n ~ 17 becomes much faster than both thewalrus and single-threaded implementations.
The optimised case shown here, is using the ``permanent`` function, which automatically selects between single & multi-threaded, based on the size of the provided unitary. This will likely not be optimal for all CPUs, but should be close enough that performance degredation is not too significant. If required, the value for which the function switches between single and multi-threading can be modified using ``perm_rs.settings.multi_threading_threshold``.

Plotting the relative performance of the library against thewarlus makes the above performance trends even clearer:

![perm_rs & thewalrus relative comparison](/benchmark/perm_relative_times.png)
