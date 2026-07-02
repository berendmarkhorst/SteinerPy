# Problem variants

Every variant below can be solved as a **Steiner tree** (a single group of terminals that must all be connected) or as a **Steiner forest** (multiple independent groups, each connected within itself but not necessarily to other groups).
Simply pass a list of terminal lists as `terminal_groups` — one list for a tree, multiple lists for a forest.

| Variant | Class | Description |
|---------|-------|-------------|
| Steiner tree / forest | {class}`~steinerpy.SteinerProblem` | Classic minimum-cost subgraph that connects one or more groups of terminals. |
| Prize-collecting Steiner tree / forest | {class}`~steinerpy.PrizeCollectingProblem` | Each terminal carries a prize; the solver balances the prize collected against the edge and penalty costs, so not all terminals need to be connected. |
| Node-weighted Steiner tree / forest | {class}`~steinerpy.NodeWeightedSteinerProblem` | Nodes carry costs instead of (or in addition to) edges. Internally uses a node-splitting transformation to convert the problem to a standard edge-weighted formulation. |
| Maximum-weight connected subgraph | {class}`~steinerpy.MaxWeightConnectedSubgraph` | Finds a connected subgraph maximising the total node weight. Nodes with negative weights are included only when they are needed as connectors. Convenience subclass of `PrizeCollectingProblem`. |
| Directed Steiner tree (arborescence) | {class}`~steinerpy.DirectedSteinerProblem` | The graph is directed (`nx.DiGraph`) and a designated root node must reach every terminal via directed paths. |
| Partial / full terminal Steiner tree | {class}`~steinerpy.PartialTerminalSteinerProblem`, {class}`~steinerpy.FullTerminalSteinerProblem` | Designated terminals (or *all* terminals, for the full variant) must be **leaves** of the tree. Transformed to a plain Steiner tree problem (Rehfeldt thesis §5.1). |
| Group Steiner tree | {class}`~steinerpy.GroupSteinerProblem` | Connect at least one vertex from each of several vertex *groups*. Voss (1988) super-terminal transformation to a plain Steiner tree problem (thesis §5.7). |
| Hop-constrained directed Steiner tree | {class}`~steinerpy.HopConstrainedSteinerProblem` | Directed arborescence with a bound on the number of arcs (hops); terminals have no outgoing arcs (thesis §5.8). |
| Rectilinear Steiner minimum tree | {class}`~steinerpy.RectilinearSteinerProblem` | Minimum-length tree of horizontal/vertical segments through a set of points (L1 metric), solved exactly on the Hanan grid (thesis §5.4). |
| Max-weight connected subgraph with budget | {class}`~steinerpy.BudgetedMaxWeightConnectedSubgraph` | Maximise total node weight subject to a **vertex-cost budget** on the chosen subgraph (thesis §5.6). |

Several of these variants implement the "further related problems" of Chapter 5 of D. Rehfeldt's PhD thesis (*Faster algorithms for Steiner tree and related problems*, TU Berlin 2021), reusing the same directed-cut kernel by transformation.

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

## Optional constraint modifiers

Two side-constraints are available as **optional keyword arguments** that can be combined freely with any problem class above:

| Modifier kwarg | Effect |
|----------------|--------|
| `max_degree=k` | Every node in the solution has at most *k* incident edges. |
| `budget=B` | Total edge cost may not exceed *B*; the solver maximises the number of connected terminals and returns a {class}`~steinerpy.BudgetSolution`. |

```python
# Degree-constrained Steiner tree
solution = SteinerProblem(graph, terminal_groups, max_degree=2).get_solution()

# Budget-constrained Steiner tree
solution = SteinerProblem(graph, terminal_groups, budget=10.0).get_solution()

# Both constraints together
solution = SteinerProblem(graph, terminal_groups, max_degree=2, budget=10.0).get_solution()

# Degree-constrained prize-collecting problem
solution = PrizeCollectingProblem(graph, terminal_groups, node_prizes, max_degree=3).get_solution()
```

:::{note}
`DegreeConstrainedSteinerProblem` and `BudgetConstrainedSteinerProblem` are still available for backward compatibility but are deprecated — pass the corresponding kwargs to the base class instead.
:::
