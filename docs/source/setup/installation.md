# Installation

SteinerPy is a pure-Python package published on [PyPI](https://pypi.org/project/steinerpy/).
Install it with pip:

```shell
pip install steinerpy
```

or with [uv](https://docs.astral.sh/uv/):

```shell
uv add steinerpy
```

## Requirements

- Standard CPython 3.8–3.15, with Python 3.8 retained as the minimum
- [NetworkX](https://networkx.org) — graph representation
- [highspy](https://pypi.org/project/highspy/) — the HiGHS solver backend (installed automatically)
- [SciPy](https://scipy.org) — installed automatically

Prerelease and free-threaded Python builds are outside the current support matrix.

### Python 3.15

Python 3.15 support requires a working `highspy` installation. As of November
2026, `highspy` 1.15.1 supplies Python 3.13 and 3.14 wheels for Windows and Linux
x86-64, but no Python 3.15 wheels. Installing on Python 3.15 therefore builds
HiGHS from source and requires a C++ toolchain. Upstream tracks wheel availability
in [HiGHS issue #3444](https://github.com/ERGO-Code/HiGHS/issues/3444).
Use Python 3.14 if you need installation from prebuilt wheels.

## Optional: the Gurobi backend

To use Gurobi as the solver you additionally need:

- [gurobipy](https://pypi.org/project/gurobipy/): `pip install gurobipy`
- A valid Gurobi license (free academic licenses are available from [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/))

Gurobi's Python compatibility is separate from the default HiGHS backend.
[`gurobipy` 13.0.3](https://pypi.org/project/gurobipy/13.0.3/#files) supplies
wheels through Python 3.14, but none for Python 3.15.

HiGHS is installed with SteinerPy. Both backends solve the supported models;
their returned selections can differ when several solutions tie.
See {doc}`../guide/solvers` for a comparison of the two backends.
