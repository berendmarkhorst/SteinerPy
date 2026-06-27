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
| **Partial / Full Terminal Steiner Tree** | `PartialTerminalSteinerProblem`, `FullTerminalSteinerProblem` | Designated terminals (or *all* terminals, for the full variant) must be **leaves** of the tree.  Transformed to a plain Steiner tree problem (Rehfeldt thesis §5.1). |
| **Group Steiner Tree** | `GroupSteinerProblem` | Connect at least one vertex from each of several vertex *groups*.  Voss (1988) super-terminal transformation to a plain Steiner tree problem (thesis §5.7). |
| **Hop-Constrained Directed Steiner Tree** | `HopConstrainedSteinerProblem` | Directed arborescence with a bound on the number of arcs (hops); terminals have no outgoing arcs (thesis §5.8). |
| **Rectilinear Steiner Minimum Tree** | `RectilinearSteinerProblem` | Minimum-length tree of horizontal/vertical segments through a set of points (L1 metric), solved exactly on the Hanan grid (thesis §5.4). |
| **Max-Weight Connected Subgraph with Budget** | `BudgetedMaxWeightConnectedSubgraph` | Maximise total node weight subject to a **vertex-cost budget** on the chosen subgraph (thesis §5.6).  Supports both `"highs"` and `"gurobi"`. |

These five variants implement the "further related problems" of Chapter 5 of D. Rehfeldt's PhD thesis (*Faster algorithms for Steiner tree and related problems*, TU Berlin 2021), reusing the same directed-cut kernel by transformation.

```python
from steinerpy import (
    PartialTerminalSteinerProblem, FullTerminalSteinerProblem, GroupSteinerProblem,
    HopConstrainedSteinerProblem, RectilinearSteinerProblem,
    BudgetedMaxWeightConnectedSubgraph,
)

# Partial-terminal (terminals B and D must be leaves):
PartialTerminalSteinerProblem(G, [['A', 'B', 'D']], partial_terminals=['B', 'D']).get_solution()

# Group Steiner tree (one vertex from each group):
GroupSteinerProblem(G, [['A', 'B'], ['C', 'D']]).get_solution()

# Hop-constrained directed Steiner tree (<= 3 arcs):
HopConstrainedSteinerProblem(DG, root='r', terminals=['x', 'y'], hop_limit=3).get_solution()

# Rectilinear Steiner minimum tree through points in the plane:
RectilinearSteinerProblem([(0, 0), (2, 0), (0, 2)]).get_solution()

# Max-weight connected subgraph with a vertex-cost budget:
BudgetedMaxWeightConnectedSubgraph(G, node_weights, node_costs, node_budget=5).get_solution()
```

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

## Prize-collecting / MWCSP acceleration (opt-in)

`PrizeCollectingProblem` and `MaxWeightConnectedSubgraph` default to a penalty/Big-M flow ILP.  For the **classic forgo-prize PCSTP** (and the MWCSP) you can opt into a much faster path that — following Rehfeldt & Koch (MWCSP 2019; PCSTP 2020) — transforms the problem to a rooted Steiner arborescence and reuses the dual-ascent + directed-cut machinery:

```python
# Exact, accelerated (often proves optimality without any ILP):
sol = PrizeCollectingProblem(graph, [[root]], node_prizes, penalty_cost=0).get_solution(pc_transform=True)

# Heuristic-only: dual-ascent primal, no ILP, with a PROVEN optimality gap:
sol = PrizeCollectingProblem(graph, [[root]], node_prizes, penalty_cost=0).get_solution(exact=False)

# Prize-safe edge reduction (prize-constrained distance) before solving:
sol = PrizeCollectingProblem(graph, [[root]], node_prizes, penalty_cost=0, pc_reduce=True).get_solution(pc_transform=True)

# MWCSP uses the exact MWCSP -> PCSTP -> SAP mapping:
sol = MaxWeightConnectedSubgraph(graph, node_weights).get_solution(pc_transform=True)
```

All three flags are **off by default** (the penalty ILP is unchanged) and gated to the classic forgo-prize objective: a `penalty_budget`, multiple terminal groups, or a non-zero `penalty_cost` raises `NotImplementedError` (use the default penalty ILP for those).  `pc_reduce` deletes only edges provably in no optimal solution and removes no nodes, so every prize is preserved.

## Scope and scale

SteinerPy is a **Python / NetworkX library**, and it is honest about where that
puts it. Its goal is *breadth and accessibility* — a unified, pip-installable,
MIT-licensed API that solves many Steiner variants exactly (with a certified
optimality gap), and drops straight into a Python data pipeline. Its sweet spot
is **small-to-medium instances** (up to roughly a few thousand nodes/edges),
rapid prototyping, and research that needs to move across variants quickly.

It is **not** a performance competitor to dedicated C/C++ exact solvers such as
[SCIP-Jack](https://scipjack.zib.de) (the DIMACS 2014 / PACE 2018 winner). On the
hard SteinLib / DIMACS families — instances with tens of thousands of
nodes/edges, or the notoriously difficult PUC/hypercube sets — SCIP-Jack is
faster by orders of magnitude, thanks to a decade of C engineering, a far deeper
reduction-test arsenal, and SCIP's full branch-and-cut. The benchmarks below use
the **mid-size** SteinLib `C` tier (500 nodes, up to 12 500 edges) — still well
short of a large-scale exact race, but heavy enough that the dual-ascent
acceleration and reductions earn their keep; the point is *solution quality and a
certified gap versus fast heuristics*, not raw scale. If you need to solve
million-edge instances to proven optimality, reach for SCIP-Jack; if you want
exactness, breadth, and Python ergonomics on tractable instances, SteinerPy is
built for you.

## Benchmarks

Two scripts at the repository root benchmark SteinerPy against the standard
baselines on **literature instances** (not random graphs), scoring every method
on one optimality-gap axis against published / proven optima:

- [`benchmark_compare.py`](benchmark_compare.py) — Steiner tree, vs `networkx`
  `approximation.steiner_tree` (Kou and Mehlhorn), on the bundled SteinLib
  C-series (the `B` and `D` tiers are bundled too).
- [`benchmark_pcstp.py`](benchmark_pcstp.py) — Prize-Collecting Steiner (PCSPG),
  vs [`pcst_fast`](https://github.com/fraenkel-lab/pcst_fast) (Goemans–Williamson
  2-approx), on the DIMACS 2014 JMP set.

The numbers below were produced on an **Apple M4, Python 3.13.4, networkx 3.6.1,
Gurobi 13.0.2**; absolute times are machine-dependent, the gaps are not. Each
table is generated verbatim by the script's `--markdown` flag — see *Reproducing*
below.

### Steiner Tree &mdash; SteinerPy vs NetworkX

_Instances: SteinLib `C` (20 files, 500 nodes, up to 12 500 edges / 250 terminals) &middot; solver: gurobi &middot; time limit: 60s/solve._

| Method | Avg time (s) | Avg gap % | # optimal |
|--------|-------------:|----------:|----------:|
| NetworkX-kou | 0.139 | 6.04 | 1/20 |
| NetworkX-mehlhorn | 0.010 | 6.06 | 1/20 |
| SteinerPy-exact | 9.585 | 0.00 | 19/20 |
| SteinerPy-exact+DA | 9.049 | 0.00 | 20/20 |
| SteinerPy-heuristic | 0.245 | 4.45 | 2/20 |

Average exact-vs-DA speedup (SteinerPy-exact / SteinerPy-exact+DA): **2.62&times;**.

- **SteinerPy-exact+DA** solves all 20 to proven optimality. Crucially, on `c18`
  (500 nodes, 12 500 edges, 83 terminals) the **plain** exact solve finds no
  feasible solution within 60 s (gap `∞` below), while the dual-ascent
  accelerator (`dual_ascent=True`) proves optimality — it doesn't just speed up
  easy instances, it makes a hard one solvable (and is ~2.6× faster on average).
- **NetworkX-kou / -mehlhorn** are the fastest but ≈6% off on average and solve
  only 1/20 to optimality, with no bound to tell you so.
- **SteinerPy-heuristic** (`exact=False`, no ILP) stays in NetworkX's speed class
  (~0.25 s), beats both NetworkX heuristics on quality (4.45% vs ~6%), *and*
  reports a **certified** gap — `heur cgap%` in the detail below bounds how far it
  could be from the optimum (0 = proven optimal), something `networkx.steiner_tree`
  cannot provide.

<details>
<summary>Per-instance detail</summary>

| Instance | n | m | k | opt | NetworkX-kou t | NetworkX-kou gap% | NetworkX-mehlhorn t | NetworkX-mehlhorn gap% | SteinerPy-exact t | SteinerPy-exact gap% | SteinerPy-exact+DA t | SteinerPy-exact+DA gap% | SteinerPy-heuristic t | SteinerPy-heuristic gap% | heur cgap% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c01 | 500 | 625 | 5 | 85 | 0.003 | 3.53 | 0.001 | 3.53 | 0.148 | 0.00 | 0.066 | 0.00 | 0.014 | 3.53 | 5.68 |
| c02 | 500 | 625 | 10 | 144 | 0.004 | 0.00 | 0.001 | 0.00 | 0.175 | 0.00 | 0.127 | 0.00 | 0.021 | 0.00 | 11.11 |
| c03 | 500 | 625 | 83 | 754 | 0.038 | 3.32 | 0.003 | 3.32 | 0.461 | 0.00 | 0.461 | 0.00 | 0.094 | 1.99 | 24.84 |
| c04 | 500 | 625 | 125 | 1079 | 0.061 | 3.06 | 0.004 | 3.15 | 0.687 | 0.00 | 0.705 | 0.00 | 0.123 | 2.59 | 27.01 |
| c05 | 500 | 625 | 250 | 1579 | 0.154 | 1.58 | 0.006 | 1.65 | 1.005 | 0.00 | 0.828 | 0.00 | 0.281 | 0.38 | 20.69 |
| c06 | 500 | 1000 | 5 | 55 | 0.003 | 9.09 | 0.002 | 9.09 | 0.390 | 0.00 | 0.014 | 0.00 | 0.020 | 0.00 | 0.00 |
| c07 | 500 | 1000 | 10 | 102 | 0.006 | 12.75 | 0.002 | 12.75 | 0.534 | 0.00 | 0.232 | 0.00 | 0.035 | 11.76 | 11.40 |
| c08 | 500 | 1000 | 83 | 509 | 0.051 | 4.13 | 0.004 | 4.52 | 2.589 | 0.00 | 1.576 | 0.00 | 0.115 | 2.95 | 27.86 |
| c09 | 500 | 1000 | 125 | 707 | 0.079 | 3.11 | 0.005 | 2.83 | 17.663 | 0.00 | 9.833 | 0.00 | 0.168 | 1.56 | 26.18 |
| c10 | 500 | 1000 | 250 | 1093 | 0.171 | 2.56 | 0.007 | 2.20 | 2.850 | 0.00 | 3.611 | 0.00 | 0.313 | 0.82 | 25.77 |
| c11 | 500 | 2500 | 5 | 32 | 0.005 | 15.62 | 0.004 | 15.62 | 3.616 | 0.00 | 1.364 | 0.00 | 0.044 | 15.62 | 13.51 |
| c12 | 500 | 2500 | 10 | 46 | 0.011 | 4.35 | 0.004 | 6.52 | 2.504 | 0.00 | 1.829 | 0.00 | 0.071 | 4.35 | 8.33 |
| c13 | 500 | 2500 | 83 | 258 | 0.090 | 6.59 | 0.007 | 6.98 | 11.188 | 0.00 | 15.777 | 0.00 | 0.194 | 4.65 | 32.59 |
| c14 | 500 | 2500 | 125 | 323 | 0.135 | 5.57 | 0.008 | 5.57 | 5.674 | 0.00 | 6.400 | 0.00 | 0.234 | 2.79 | 36.75 |
| c15 | 500 | 2500 | 250 | 556 | 0.287 | 2.70 | 0.011 | 3.24 | 5.515 | 0.00 | 5.617 | 0.00 | 0.426 | 0.54 | 38.64 |
| c16 | 500 | 12500 | 5 | 11 | 0.017 | 9.09 | 0.015 | 9.09 | 4.993 | 0.00 | 12.430 | 0.00 | 0.159 | 9.09 | 8.33 |
| c17 | 500 | 12500 | 10 | 18 | 0.034 | 11.11 | 0.015 | 5.56 | 14.741 | 0.00 | 20.621 | 0.00 | 0.218 | 11.11 | 35.00 |
| c18 | 500 | 12500 | 83 | 113 | 0.288 | 13.27 | 0.027 | 13.27 | 63.319 | &infin; | 62.913 | 0.00 | 0.513 | 7.96 | 53.28 |
| c19 | 500 | 12500 | 125 | 146 | 0.436 | 8.90 | 0.031 | 11.64 | 44.254 | 0.00 | 19.367 | 0.00 | 0.671 | 6.85 | 46.15 |
| c20 | 500 | 12500 | 250 | 267 | 0.915 | 0.37 | 0.043 | 0.75 | 9.390 | 0.00 | 17.200 | 0.00 | 1.194 | 0.37 | 48.13 |

</details>

> The `∞` on `c18` marks a method that hit the time limit without a feasible
> incumbent; such non-finite gaps are excluded from the summary averages above
> (the `# optimal` column still reflects the miss: SteinerPy-exact 19/20).

### Prize-Collecting Steiner (PCSPG) &mdash; SteinerPy vs pcst_fast

_Instances: DIMACS 2014 JMP `K100/P100` (6 files) &middot; solver: gurobi &middot; time limit: 30s._

| Method | Avg time (s) | Avg gap % | # optimal |
|--------|-------------:|----------:|----------:|
| SteinerPy-exact | 0.165 | 0.00 | 6/6 |
| SteinerPy-heuristic | 0.008 | 2.94 | 2/6 |
| pcst_fast | 0.000 | 2.30 | 3/6 |

`SteinerPy-exact` (`pc_transform=True`) proves optimality on all six in well under
a second; `pcst_fast` is essentially instantaneous but ≈2–3% off with no
certificate, and `SteinerPy-heuristic` matches its speed class while reporting a
certified gap. (The bundled `K400/P400` instances are larger; run the full set
yourself to see the time-limit behaviour.)

### Reproducing

```bash
# Steiner tree vs networkx (exact + heuristic) on the C tier — the table above
python benchmark_compare.py --markdown --instances benchmarks/data/C --time-limit 60

# Heuristics only (no Gurobi license needed; scales to large instances)
python benchmark_compare.py --markdown --heuristics-only --solver highs

# PCSPG vs pcst_fast on the bundled JMP set
python benchmark_pcstp.py --markdown

# Any directory of SteinLib/PCSPG .stp files (default is the B tier)
python benchmark_compare.py --markdown --instances path/to/instances
```

Drop the `--markdown` flag for the annotated fixed-width console report. Both
scripts default to Gurobi; pass `--solver highs` for the always-available HiGHS
backend.

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

## References

SteinerPy builds on a large body of prior work. The algorithms below are
credited inline in the relevant source modules; this section collects them in
one place.

**Formulation & exact solving**

- R. T. Wong (1984), *A dual ascent approach for Steiner tree problems on a
  directed graph*, Mathematical Programming 28, 271–287,
  [doi:10.1007/BF02612335](https://doi.org/10.1007/BF02612335)
  — dual ascent for the directed-cut (arborescence) dual
  (`steinerpy/dual_ascent.py`).
- M. Leitner, I. Ljubić, M. Luipersbeck, M. Sinnl (2018), *A dual ascent-based
  branch-and-bound framework for the prize-collecting Steiner tree and related
  problems*, INFORMS Journal on Computing 30(2), 402–420,
  [doi:10.1287/ijoc.2017.0788](https://doi.org/10.1287/ijoc.2017.0788)
  — reduced-cost variable fixing (`steinerpy/dual_ascent.py`).
- D. Schmidt, B. Zey, F. Margot (2021), *Stronger MIP formulations for the
  Steiner forest problem*, Mathematical Programming 186, 373–407,
  [doi:10.1007/s10107-019-01460-6](https://doi.org/10.1007/s10107-019-01460-6)
  — branch-and-cut acceleration via creep flows and back cuts, used in the
  directed-cut separation (`steinerpy/mathematical_model.py`).

**Graph reductions**

- D. Rehfeldt, T. Koch (2023), *Implications, conflicts, and reductions for
  Steiner trees*, Mathematical Programming 197(2), 903–966,
  [doi:10.1007/s10107-021-01757-5](https://doi.org/10.1007/s10107-021-01757-5)
  — Special Distance / bottleneck Steiner distance edge-deletion tests
  (`steinerpy/graph_reducer.py`).
- D. Rehfeldt, T. Koch (2020), *On the exact solution of prize-collecting
  Steiner tree problems*, ZIB-Report 20-11
  ([PDF](https://optimization-online.org/wp-content/uploads/2020/04/7749.pdf);
  published in INFORMS Journal on Computing,
  [doi:10.1287/ijoc.2021.1087](https://doi.org/10.1287/ijoc.2021.1087))
  — PCSTP/MWCSP transformations and the prize-constrained distance (PCD)
  reductions (`steinerpy/pc_transform.py`, `steinerpy/pc_reductions.py`).
- C. W. Duin (1993), *Steiner's problem in graphs*, PhD thesis, University of
  Amsterdam, and T. Polzin & S. Vahdati Daneshmand (2001), *Improved algorithms
  for the Steiner problem in networks*, Discrete Applied Mathematics 112(1–3),
  263–300,
  [doi:10.1016/S0166-218X(00)00319-X](https://doi.org/10.1016/S0166-218X(00)00319-X)
  — alternative-based reduction tests (`steinerpy/graph_reducer.py`).
- I. Ljubić (2021), *Solving Steiner trees: Recent advances, challenges, and
  perspectives*, Networks 77(2), 177–204,
  [doi:10.1002/net.22005](https://doi.org/10.1002/net.22005)
  — survey informing the reduction and dual-ascent implementations.

**Heuristics & classic constructions**

- L. Kou, G. Markowsky, L. Berman (1981), *A fast algorithm for Steiner trees*,
  Acta Informatica 15(2), 141–145,
  [doi:10.1007/BF00288961](https://doi.org/10.1007/BF00288961)
  — shortest-path Steiner tree heuristic and tree cleanup
  (`steinerpy/objects.py`, `steinerpy/dual_ascent.py`).
- K. Mehlhorn (1988), *A faster approximation algorithm for the Steiner problem
  in graphs*, Information Processing Letters 27(3), 125–128,
  [doi:10.1016/0020-0190(88)90066-X](https://doi.org/10.1016/0020-0190(88)90066-X)
  — terminal Voronoi / boundary-MST construction
  (`steinerpy/graph_reducer.py`, `steinerpy/objects.py`).

**Problem transformations**

- M. Hanan (1966), *On Steiner's problem with rectilinear distance*, SIAM
  Journal on Applied Mathematics 14(2), 255–265,
  [doi:10.1137/0114025](https://doi.org/10.1137/0114025)
  — the Hanan grid reduction for rectilinear Steiner minimum trees
  (`steinerpy/rectilinear.py`).
- S. Voß (1999), *The Steiner tree problem with hop constraints*, Annals of
  Operations Research 86, 321–345,
  [doi:10.1023/A:1018967121276](https://doi.org/10.1023/A:1018967121276)
  — super-terminal transformation for the group Steiner tree problem
  (`steinerpy/objects.py`).

## Star History

<a href="https://www.star-history.com/?repos=berendmarkhorst%2Fsteinerpy&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=berendmarkhorst/steinerpy&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=berendmarkhorst/steinerpy&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=berendmarkhorst/steinerpy&type=date&legend=top-left" />
 </picture>
</a>
