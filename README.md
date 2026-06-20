# SteinerPy

[![PyPI version](https://badge.fury.io/py/steinerpy.svg)](https://pypi.org/project/steinerpy/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/steinerpy)](https://pypi.org/project/steinerpy/)
[![Python 3.8+](https://img.shields.io/pypi/pyversions/steinerpy.svg)](https://pypi.org/project/steinerpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/berendmarkhorst/SteinerPy/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/berendmarkhorst/SteinerPy/actions)
[![codecov](https://codecov.io/gh/berendmarkhorst/SteinerPy/branch/main/graph/badge.svg)](https://codecov.io/gh/berendmarkhorst/SteinerPy)

A Python package for solving Steiner Tree and Steiner Forest Problems — and several advanced variants — using the HiGHS solver and NetworkX graphs.  Gurobi is also supported as an alternative solver via lazy-cut callbacks.

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

To use Gurobi as the solver, you additionally need:

- [gurobipy](https://pypi.org/project/gurobipy/) (install with `pip install gurobipy`)
- A valid Gurobi license

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

# Solve with HiGHS (default)
problem = SteinerProblem(G, terminal_groups)
solution = problem.get_solution()

print(f"Optimal cost: {solution.objective}")
print(f"Selected edges: {solution.selected_edges}")

# Solve with Gurobi (requires gurobipy + license)
solution_gurobi = problem.get_solution(solver="gurobi")
print(f"Gurobi optimal cost: {solution_gurobi.objective}")
```

## Supported Problem Variants

Every variant below can be solved as a **Steiner Tree** (a single group of terminals that must all be connected) or as a **Steiner Forest** (multiple independent groups, each connected within itself but not necessarily to other groups).  Simply pass a list of terminal lists to `terminal_groups` — one list for a tree, multiple lists for a forest.

| Variant | Class | Description |
|---------|-------|-------------|
| **Steiner Tree / Forest** | `SteinerProblem` | Classic minimum-cost subgraph that connects one or more groups of terminals. |
| **Prize-Collecting Steiner Tree / Forest** | `PrizeCollectingProblem` | Each terminal carries a prize; the solver balances the prize collected against the edge and penalty costs, so not all terminals need to be connected. |
| **Node-Weighted Steiner Tree / Forest** | `NodeWeightedSteinerProblem` | Nodes carry costs instead of (or in addition to) edges.  Internally uses a node-splitting transformation to convert the problem to a standard edge-weighted formulation. |
| **Maximum-Weight Connected Subgraph** | `MaxWeightConnectedSubgraph` | Finds a connected subgraph maximising the total node weight.  Nodes with negative weights are included only when they are needed as connectors.  Convenience subclass of `PrizeCollectingProblem`. |
| **Directed Steiner Tree (Arborescence)** | `DirectedSteinerProblem` | The graph is directed (`nx.DiGraph`) and a designated root node must reach every terminal via directed paths. |

### Optional Constraint Modifiers

The two side-constraints below are **optional keyword arguments** that can be combined freely with any problem class above:

| Modifier kwarg | Effect |
|----------------|--------|
| `max_degree=k` | Every node in the solution has at most *k* incident edges. |
| `budget=B` | Total edge cost may not exceed *B*; the solver maximises the number of connected terminals and returns a `BudgetSolution`. |

```python
# Degree-constrained Steiner tree
solution = SteinerProblem(graph, terminal_groups, max_degree=2).get_solution()

# Budget-constrained Steiner tree
solution = SteinerProblem(graph, terminal_groups, budget=10.0).get_solution()

# Both constraints together (previously impossible with dedicated classes)
solution = SteinerProblem(graph, terminal_groups, max_degree=2, budget=10.0).get_solution()

# Degree-constrained prize-collecting problem
solution = PrizeCollectingProblem(graph, terminal_groups, node_prizes, max_degree=3).get_solution()
```

> **Note:** `DegreeConstrainedSteinerProblem` and `BudgetConstrainedSteinerProblem` are still available for backward compatibility but are deprecated — pass the corresponding kwargs to the base class instead.

## Solver Selection

Every problem class exposes a `solver` parameter on `get_solution()`.  Two backends are supported:

| `solver` value | Backend | Notes |
|----------------|---------|-------|
| `"highs"` (default) | [HiGHS](https://highs.dev/) via *highspy* | Always available; cut-based formulation solved iteratively (re-solve loop). |
| `"gurobi"` | [Gurobi](https://www.gurobi.com/) via *gurobipy* | Optional; requires *gurobipy* and a valid Gurobi license.  Connectivity cuts are injected as **lazy constraints** inside a branch-and-cut callback, which lets Gurobi exploit its full branch-and-bound tree. |

```python
# Use HiGHS (default — no extra installation required)
solution = SteinerProblem(graph, terminal_groups).get_solution()

# Use Gurobi (requires gurobipy + license)
solution = SteinerProblem(graph, terminal_groups).get_solution(solver="gurobi")
```

Both solvers implement the same cut-based (DO-D) formulation from Markhorst et al. (2025) and produce identical optimal solutions.  Gurobi may be faster on larger instances because callbacks avoid repeated re-solves from scratch.

## Dual-Ascent Accelerator (opt-in)

An optional [Wong (1984)](https://doi.org/10.1007/BF02612335) dual-ascent procedure can speed up the exact solve.  It computes, cheaply, a lower bound, a feasible primal solution, and *reduced costs*; following [Leitner et al. (2018)](https://doi.org/10.1007/s10589-017-9966-x) the reduced costs are used to **fix variables to zero** (eliminate arcs/edges that cannot appear in any optimal solution) before HiGHS/Gurobi runs.  When the bound matches the heuristic, the instance is solved **without building the ILP at all**.

```python
# Enable per problem (constructor) ...
solution = SteinerProblem(graph, terminal_groups, dual_ascent=True).get_solution()

# ... or per call (overrides the constructor flag)
solution = SteinerProblem(graph, terminal_groups).get_solution(dual_ascent=True)
```

It is **off by default** and returns the same optimum as the baseline.  Supported for Steiner **tree**, **forest** (multi-root), and **directed** (`DirectedSteinerProblem`) problems; it is skipped automatically when a `budget` or `max_degree` modifier is set, and for the prize-collecting / node-weighted variants.  See [`benchmarks/`](benchmarks/) for a SteinLib comparison harness.

When enabled it additionally **warm-starts** the cut loop with the Steiner cuts found during dual ascent and runs a **multi-start primal** from several roots, so many instances are solved entirely by dual ascent with no ILP.

### Bound-based graph reduction (opt-in)

```python
# Delete edges proven non-optimal by the dual-ascent reduced costs, then cascade
# the degree reductions — shrinking the model before the solve.
solution = SteinerProblem(graph, terminal_groups, da_reduce=True).get_solution()
```

`da_reduce` is a **reduction test**: it removes edges that the dual-ascent reduced costs prove cannot appear in any optimal solution and cascades the degree-1/degree-2 reductions to a fixpoint.  It requires `preprocess=True` (the default), applies to undirected problems only, is skipped under a `budget`/`max_degree` modifier, and **preserves the optimum** (solutions still map back to the original graph).  It composes with `dual_ascent=True`.

### Heavy graph reductions (opt-in)

```python
# Special Distance + long-edge edge-deletion tests, interleaved with the degree
# reductions to a fixpoint — heavier preprocessing for harder instances.
solution = SteinerProblem(graph, terminal_groups, heavy=True).get_solution()

# Fine-grained control (heavy=True turns on whichever applies):
SteinerProblem(graph, terminal_groups, special_distance=True, long_edge=False)
```

`heavy=True` enables two classic **alternative-based reduction tests** from the
Steiner-tree literature, both of which *only delete edges that are provably in
no optimal solution* and then cascade the degree-1/degree-2 reductions:

- **Special Distance (bottleneck Steiner distance) test** — deletes an edge
  `e = {v, w}` when the bottleneck distance between `v` and `w` through the
  *terminal distance network* is below `c(e)` (Rehfeldt & Koch, *Math. Prog. B*
  197, 2023, Thm 1; surveyed in Ljubić, *Networks* 77, 2021, §4). Catches long
  edges that have a cheaper terminal-hopping detour, which the degree reductions
  miss. **Steiner tree only** (automatically skipped for the multi-group forest,
  where terminal-hopping would be unsound).
- **Long-edge / alternative-path test** — deletes an edge when a strictly
  cheaper detour exists in `G \ e`. Valid for both Steiner **tree** and
  **forest**.

Both tests are implemented with the fast constructions used by state-of-the-art
SPG solvers: the Special Distance test builds the terminal distance network from
a single multi-source Dijkstra (terminal Voronoi diagram) and Mehlhorn's (1988)
boundary MST — `O(m + n log n)` rather than one shortest-path tree per terminal —
and the long-edge test runs one bounded Dijkstra per *vertex* rather than per
*edge* (Rehfeldt & Koch 2023, §2.3). In practice this is several times faster on
large, terminal-rich instances (≈5–10× for Special Distance, ≈3–5× for long-edge).

Like `da_reduce`, `heavy` requires `preprocess=True` (the default), applies to
undirected problems only, is skipped under a `budget`/`max_degree` modifier (those
variants do not minimise edge cost), and **preserves the optimum** — solutions
still map back to the original graph. It composes with `da_reduce=True` and
`dual_ascent=True`; a good "throw everything at it" configuration is
`SteinerProblem(graph, terminals, heavy=True, da_reduce=True, dual_ascent=True)`.

## Heuristic-only mode (opt-in)

An exact solver can't match a polynomial-time heuristic such as `networkx.steiner_tree` in general.  When you want that speed and can accept an approximate answer, pass `exact=False`:

```python
# Return the dual-ascent primal directly — no ILP is built or solved.
solution = SteinerProblem(graph, terminal_groups).get_solution(exact=False)

print(solution.objective)  # heuristic tree weight (an upper bound on the optimum)
print(solution.gap)        # PROVEN optimality gap: 0.0 == provably optimal
```

Unlike a pure heuristic, the returned `Solution.gap` is a **valid optimality certificate**: `gap == 0.0` means the heuristic tree is provably optimal, and a positive gap bounds how far it could be from the optimum — something `networkx.steiner_tree` (which gives no lower bound) cannot provide.  It is **networkx-speed-class** (no ILP), supported for plain Steiner **tree**/**forest** and **directed** problems, and raises `NotImplementedError` for the budget/degree-constrained variants.  The default is `exact=True` (solve to optimality).

## Usage Examples

See the `example.ipynb` notebook for detailed usage examples.

## Dependencies

- `networkx`: For graph representation and manipulation
- `highspy`: For optimization solving (HiGHS backend, required)
- `gurobipy`: For optimization solving (Gurobi backend, optional — requires a Gurobi license)

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

## Star History

<a href="https://www.star-history.com/?repos=berendmarkhorst%2Fsteinerpy&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=berendmarkhorst/steinerpy&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=berendmarkhorst/steinerpy&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=berendmarkhorst/steinerpy&type=date&legend=top-left" />
 </picture>
</a>
