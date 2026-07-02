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
- **Fast when you want it** — an opt-in dual-ascent accelerator, graph reduction tests, and a heuristic-only mode that stays in NetworkX's speed class while still certifying its gap.

📖 **Documentation: [steinerpy.readthedocs.io](https://steinerpy.readthedocs.io)**

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
solution = SteinerProblem(G, [["A", "D"]]).get_solution()

print(f"Optimal cost: {solution.objective}")
print(f"Selected edges: {solution.selected_edges}")
print(f"Proven optimality gap: {solution.gap}")  # 0.0 == provably optimal
```

See the [documentation](https://steinerpy.readthedocs.io) for the full catalogue of problem variants, solver selection, performance features (dual ascent, graph reductions, heuristic-only mode), benchmarks against NetworkX and `pcst_fast`, and the API reference. The [example notebook](example.ipynb) walks through the main features.

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
