# SteinerPy

[![PyPI version](https://badge.fury.io/py/steinerpy.svg)](https://pypi.org/project/steinerpy/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/steinerpy)](https://pypi.org/project/steinerpy/)
[![Python 3.8+](https://img.shields.io/pypi/pyversions/steinerpy.svg)](https://pypi.org/project/steinerpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/berendmarkhorst/SteinerPy/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/berendmarkhorst/SteinerPy/actions)
[![codecov](https://codecov.io/gh/berendmarkhorst/SteinerPy/branch/main/graph/badge.svg)](https://codecov.io/gh/berendmarkhorst/SteinerPy)
[![Documentation](https://readthedocs.org/projects/steinerpy/badge/?version=latest)](https://steinerpy.readthedocs.io)

SteinerPy solves Steiner tree and Steiner forest problems — and many advanced variants — **to proven optimality**, directly on [NetworkX](https://networkx.org) graphs. It uses the open-source [HiGHS](https://highs.dev) solver by default, with [Gurobi](https://www.gurobi.com) supported as an optional backend.

- **One API, many variants** — Steiner tree/forest, prize-collecting, node-weighted, maximum-weight connected subgraph, directed (arborescence), hop-constrained, group, rectilinear, terminal-leaf, and budgeted variants.
- **Exact, with a certificate** — every solve reports a proven optimality gap; `gap == 0.0` means provably optimal.
- **Fast by default** — provably optimum-preserving graph reductions from the Steiner-tree literature run automatically, plus an opt-in dual-ascent accelerator and a heuristic-only mode that stays in NetworkX's speed class while still certifying its gap.

📖 **Documentation: [steinerpy.readthedocs.io](https://steinerpy.readthedocs.io)**

## Unified problem interface

Describe connectivity and costs with one class:

```python
from steinerpy import Problem

problem = Problem(graph, terminals=["A", "D"])
solution = problem.get_solution(solver="highs", time_limit=60)

problem = Problem(graph, terminals=[["A", "B"], ["C", "D"]])  # Forest
problem = Problem(digraph, root="A", terminals=["D"])
problem = Problem(graph, groups=[["A", "B"], ["C", "D"]], node_costs={"B": 2})
```

Flat `terminals` must all connect; nested lists must each connect internally,
with sharing allowed. `groups` chooses at least one member per group in one
connected tree. Group-plus-node-cost problems minimize edge costs plus selected
node costs; this combination currently requires non-negative costs and exact
solving without preprocessing.

See the [variant guide](docs/source/guide/variants.md) for supported combinations.
Explicit `variant=` calls, `terminal_groups=`, and existing classes remain supported.

## Installation

```bash
pip install steinerpy
```

Requires Python 3.8+. The HiGHS backend is installed automatically; to use Gurobi instead, install [gurobipy](https://pypi.org/project/gurobipy/) and provide a valid license.

## Quick start

```python
import networkx as nx
from steinerpy import SteinerProblem

G = nx.Graph()
G.add_edge("A", "B", weight=1)
G.add_edge("B", "C", weight=2)
G.add_edge("C", "D", weight=1)

# One terminal group = Steiner tree; multiple groups = Steiner forest
# Edge/arc costs must be non-negative.
solution = SteinerProblem(G, [["A", "D"]]).get_solution()

print(f"Optimal cost: {solution.objective}")
print(f"Selected edges: {solution.selected_edges}")
print(f"Proven optimality gap: {solution.gap}")  # 0.0 == provably optimal
```

See the [documentation](https://steinerpy.readthedocs.io) for the full catalogue of problem variants, solver selection, performance features (dual ascent, graph reductions, heuristic-only mode), benchmarks against NetworkX and `pcst_fast`, and the API reference. The [example notebook](example.ipynb) walks through the main features.

For measurements on the three PACE 2018 tracks and historical participant
results, see the [PACE benchmark report](benchmarks/reports/pace2018/README.md).

## Citing

If you use SteinerPy in your research, please cite:

```bibtex
@article{markhorst2025future,
  title={Future-proof ship pipe routing: Navigating the energy transition},
  author={Markhorst, Berend and Berkhout, Joost and Zocca, Alessandro and Pruyn, Jeroen and van der Mei, Rob},
  journal={Ocean Engineering},
  volume={319},
  pages={120113},
  year={2025},
  publisher={Elsevier}
}
```

## Contributing

Contributions are very welcome! Please read the
[contributing guidelines](CONTRIBUTING.md) and open an issue to discuss your
idea before starting on a pull request.

## License

SteinerPy is available under the [MIT license](LICENSE).

## Star History

<a href="https://www.star-history.com/?repos=berendmarkhorst%2Fsteinerpy&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=berendmarkhorst/steinerpy&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=berendmarkhorst/steinerpy&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=berendmarkhorst/steinerpy&type=date&legend=top-left" />
 </picture>
</a>
