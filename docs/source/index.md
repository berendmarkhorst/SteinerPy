# SteinerPy

SteinerPy is a Python package for solving Steiner tree and Steiner forest problems — and many advanced variants — **to proven optimality**, directly on [NetworkX](https://networkx.org) graphs.
It uses the open-source [HiGHS](https://highs.dev) solver by default, with [Gurobi](https://www.gurobi.com) supported as an optional backend via lazy-cut callbacks.

The package aims for *breadth and accessibility*: a unified, pip-installable, MIT-licensed API that solves many Steiner variants exactly (with a certified optimality gap) and drops straight into a Python data pipeline.
Its sweet spot is small-to-medium instances, rapid prototyping, and research that needs to move across problem variants quickly.

Highlights:

- **One API, many variants** — Steiner tree/forest, prize-collecting, node-weighted, maximum-weight connected subgraph, directed (arborescence), hop-constrained, group, rectilinear, terminal-leaf, and budgeted variants.
- **Exact, with a certificate** — every solve reports a proven optimality gap; `gap == 0.0` means provably optimal.
- **Fast when you want it** — an opt-in dual-ascent accelerator, graph reduction tests, and a heuristic-only mode that stays in NetworkX's speed class while still certifying its gap.
- **Two solver backends** — HiGHS (always available) and Gurobi (optional, requires a license).

## Installation

SteinerPy is available on [PyPI](https://pypi.org/project/steinerpy/):

```shell
pip install steinerpy
```

See {doc}`setup/installation` for requirements and the optional Gurobi backend.

## Getting help

Feel free to open an issue on [GitHub](https://github.com/berendmarkhorst/SteinerPy/issues) if you have questions, run into a problem, or want to propose a feature.

## Citing SteinerPy

If you use SteinerPy in your research, please cite the paper that introduced it — see {doc}`setup/citing` for the BibTeX entry.

```{toctree}
:maxdepth: 1
:caption: Getting started
:hidden:

setup/installation
setup/getting_started
setup/citing
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

examples/quickstart
```

```{toctree}
:maxdepth: 1
:caption: API reference
:hidden:

api/steinerpy
```

```{toctree}
:maxdepth: 1
:caption: Extras
:hidden:

extras/references
```
