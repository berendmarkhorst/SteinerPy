# SteinerPy

SteinerPy solves Steiner tree and forest problems on NetworkX graphs. You specify
which nodes must connect, add costs and constraints, and solve with HiGHS or Gurobi.
The result contains the selected edges, objective value, and optimality gap.

<span id="installation"></span>

```shell
pip install steinerpy
```

Start with {doc}`examples/quickstart`, or use {doc}`guide/variants` to choose the
inputs for your model. See {doc}`setup/installation` for requirements and the
optional Gurobi backend.

<span id="getting-help"></span>

For questions and bug reports, open an issue on
[GitHub](https://github.com/berendmarkhorst/SteinerPy/issues).

<span id="citing-steinerpy"></span>

If you use the package in research, see {doc}`setup/citing`.

```{toctree}
:maxdepth: 1
:caption: Getting started
:hidden:

setup/installation
setup/getting_started
examples/quickstart
```

```{toctree}
:maxdepth: 1
:caption: User guide
:hidden:

guide/variants
guide/solvers
guide/performance
guide/benchmarks
```

```{toctree}
:maxdepth: 1
:caption: Examples
:hidden:

examples/connectivity
examples/costs_and_budgets
examples/constraints
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

api/steinerpy
reference/compatibility
reference/algorithms
setup/citing
extras/references
```
