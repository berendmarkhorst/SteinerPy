# SteinerPy

[![PyPI version](https://badge.fury.io/py/steinerpy.svg)](https://pypi.org/project/steinerpy/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/steinerpy)](https://pypi.org/project/steinerpy/)
[![Python 3.8+](https://img.shields.io/pypi/pyversions/steinerpy.svg)](https://pypi.org/project/steinerpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/berendmarkhorst/SteinerPy/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/berendmarkhorst/SteinerPy/actions)
[![codecov](https://codecov.io/gh/berendmarkhorst/SteinerPy/branch/main/graph/badge.svg)](https://codecov.io/gh/berendmarkhorst/SteinerPy)

A Python package for solving Steiner Tree and Steiner Forest Problems using the HiGHS solver and NetworkX graphs.

## Installation

Install SteinerPy from PyPI:

```bash
pip install steinerpy
```

Or using uv:

```bash
uv add steinerpy
```

### Requirements

- Python 3.8+
- NetworkX
- HiGHS Solver (via highspy)

## Quick Start

```python
import networkx as nx
from steinerpy import SteinerProblem

# Create a graph
G = nx.Graph()
G.add_edge('A', 'B', weight=1)
G.add_edge('B', 'C', weight=2)
G.add_edge('C', 'D', weight=1)

# Define terminal groups
terminal_groups = [['A', 'D']]

# Solve the Steiner problem
problem = SteinerProblem(G, terminal_groups)
solution = problem.get_solution()

print(f"Optimal cost: {solution.objective}")
print(f"Selected edges: {solution.selected_edges}")
```

## Usage Examples

See the `example.ipynb` notebook for detailed usage examples.

## Dependencies

- `networkx`: For graph representation and manipulation
- `highspy`: For optimization solving

If you use this package in your research, please cite:

```
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