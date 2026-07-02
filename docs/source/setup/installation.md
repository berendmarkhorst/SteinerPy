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

- Python 3.8+
- [NetworkX](https://networkx.org) — graph representation
- [highspy](https://pypi.org/project/highspy/) — the HiGHS solver backend (installed automatically)
- [SciPy](https://scipy.org) — installed automatically

## Optional: the Gurobi backend

To use Gurobi as the solver you additionally need:

- [gurobipy](https://pypi.org/project/gurobipy/): `pip install gurobipy`
- A valid Gurobi license (free academic licenses are available from [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/))

Gurobi is never required: the HiGHS backend is always available and produces the same optimal solutions.
See {doc}`../guide/solvers` for a comparison of the two backends.
