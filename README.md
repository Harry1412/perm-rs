# perm-rs

This library contains a Rust re-implementation of the permanent calculation function from the [thewalrus](https://github.com/XanaduAI/thewalrus) module, using maturin and PyO3 to then make this callable from Python.

It is mainly intended for testing how rust-enhanced modules can be developed for Python and then distributed via pypi to different platforms.

## Install

Rust must first be installed if not already, the instructions for which can be found [here](https://rust-lang.org/tools/install/).

Maturin is then utilised for building the rust code and generating Python bindings, this requires a virtual or conda environment. All requirements, including maturin, may be installed using the following:

```bash
(venv) pip install -r requirements.txt
```

Once installed, the library is built using:

```bash
(venv) maturin develop
```

For optimal performance, it should be built with the release profile:

```bash
(venv) maturin develop --release
```

## Benchmarking

The performance of the library can be benchmarked against thewalrus implementation using the following command:

```bash
pytest benchmark
```

Alternatively, a more comprehensive set of matrix sizes can be compared using the script in ``benchmark/generate_figure.py``. The following is the output comparison between the two libraries when run using a AMD Ryzen 9 9950x3D CPU:

![perm_rs & thewalrus comparison](/benchmark/perm_relative_times.png)

From this, it can be seen that currently the library is more performant for lower values of n, but trends towards thewalrus before becoming slower at ~n=24.