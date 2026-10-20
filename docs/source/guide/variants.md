# Problem variants

## One configurable problem class

Use `Problem` to describe connectivity and costs. For the supported feature
combinations below, the implementation follows from those inputs:

```python
from steinerpy import Problem

problem = Problem(graph, terminals=["A", "D"], max_degree=3)
solution = problem.get_solution(solver="highs", time_limit=60)

problem = Problem(digraph, root="A", terminals=["D"])
problem = Problem(graph, groups=[["A", "B"], ["C", "D"]], node_costs={"B": 2})
problem = Problem(variant="rectilinear", points=[(0, 0), (1, 1), (2, 0)])
```

`terminals=["A", "B"]` requires every listed node in one connected tree.
`terminals=[["A", "B"], ["C", "D"]]` requires A–B and C–D connectivity,
without requiring a connection between the sets. These forest requirements
may share edges. Flat and one-list nested forms are equivalent.

`groups=[["A", "B"], ["C", "D"]]` instead requires **at least one** member
of each group in **one connected tree**. With groups, flat terminals can also
require particular nodes; a single nested terminal list is equivalent. Multiple
forest terminal lists combined with groups are not yet supported.

Use lists for nested terminal syntax; tuple-valued node labels remain atomic.
Mixed flat/nested inputs are rejected. `terminal_groups=` remains an alias for
the old nested syntax, including the positional second argument; supplying both
`terminals` and `terminal_groups` is an error.

| Features (no `variant` required) | Supported behavior |
|---|---|
| Undirected graph + flat or nested `terminals` | Steiner tree or forest; existing degree/edge-budget options apply |
| Directed graph + `root` + one set of `terminals` | Rooted directed Steiner problem |
| Directed graph + `root` + `terminals` + `hop_limit` | Existing hop-constrained problem |
| Undirected graph + `groups` | Existing group Steiner problem |
| Directed graph + `root` + `groups` | Existing directed group Steiner problem |
| Directed graph + `root` + `groups` + `hop_limit` | Directed group problem with a limit on selected original arcs |
| Directed graph + `root` + `groups` + `max_degree` | Directed group problem with original-node total degree limits; may also use `hop_limit` |
| Directed graph + `root` + `groups` + `budget` | Maximize reached groups within an original-edge cost budget; optional degree/hop limits apply |
| Undirected graph + `groups` + `node_costs` and/or mandatory `terminals` | New exact composition path, described below |
| Same composition + `hop_limit` | Limit total selected original edges after node splitting |

Other named variants remain available through explicit `variant=` calls.
Existing explicit calls and legacy classes retain their solution types and
solver restrictions. Directed groups support budgets, degree and hop limits.
The group/node-cost composition supports original-edge hop limits. Other
undirected-group and legacy node-weighted modifier combinations remain restricted.

`Problem(...)` validates combinations before constructing a solver strategy or
transforming the graph. Invalid combinations raise `TypeError` or `ValueError`
with the unsupported feature named. In particular:

- Group/node-cost composition requires non-negative finite costs and no
  preprocessing or budgets. Original-edge hop and original-node degree limits are supported.
- Terminal-leaf variants cannot use edge budgets: their transformation changes
  edge costs, so the existing budget would not describe the original network.
- The legacy node-weighted variant requires an undirected graph; its node
  splitting creates both directions for every edge.
- `pc_transform=True` requires a compatible classic prize-collecting or
  maximum-weight objective without degree limits or budgets. Directed prizes
  use `variant="directed_prize_collecting"` and its directed transformation.
- The node-budgeted maximum-weight variant uses `node_budget`; additional
  edge/penalty budgets and prize-transform/reduction options are unsupported.

Node costs and prizes must be finite and non-negative. Maximum-weight variants
retain signed node weights; the node-budgeted variant's edge attributes remain
topology-only. Existing automatic preprocessing disabling for directed and
prize-collecting implementations is retained. Solver-time options and backend
availability are checked when solving.

## Directed groups with hop and degree limits

```python
problem = Problem(
    digraph, root="r", groups=[["A", "B"], ["C"]], hop_limit=3, max_degree=2
)
solution = problem.get_solution(solver="highs")  # Gurobi is also supported
```

`hop_limit` is a non-negative integer limiting the **total number of selected
original arcs**, not the length of each root-to-member path. Original zero-cost
arcs count; artificial group connectors do not. Groups may overlap and contain
the root. When all groups are required, a zero limit is feasible only if the
root belongs to every group.
Group members may be internal vertices on paths to other groups.

`max_degree` is a non-negative integer limiting incoming plus outgoing selected
original arcs at **each original node**, including the root and group members.
Artificial connectors do not count. This is total degree, not just out-degree:
an internal node with one incoming and two outgoing original arcs has degree
three. Zero-cost original arcs still count. When all groups are required, a
zero degree limit is feasible only if the root belongs to every group.

Both options are available on `DirectedGroupSteinerProblem`, separately or
together. Decoded solutions are independently checked against the original-arc
and original-node limits. Infeasibility raises `RuntimeError`. Additional
mandatory terminals and node costs remain unsupported with this combination.

### Directed groups with an edge budget

```python
problem = Problem(
    digraph, root="r", groups=[["A", "B"], ["C"]], budget=5,
    hop_limit=3, max_degree=2,
)
solution = problem.get_solution(solver="highs")  # Gurobi is also supported
```

As in the existing budget API, `budget` changes the goal: **maximize the number
of reached groups** within the original-edge cost budget. It does not require
every group. Each original arc is charged once; missing costs default to one,
and `weight=` selects the cost attribute. Artificial connectors cost zero.
The budget must be finite and non-negative.

A group counts once if any member is reachable from the root. Overlapping or
identical groups count separately, and groups containing the root count even
with no selected arcs. The root itself adds no extra count. Zero reached
groups is a valid optimum. Optional degree and hop limits retain their
original-node/arc meanings, including on zero-cost original arcs.

Results are `BudgetSolution` objects. `objective` and `connected_terminals`
count reached groups; `total_terminals` is the number of groups. `penalties`
contains `{"group_i": 1}` for each missed zero-based group index. Returned
arcs belong to the original graph and are reachable from the root; decoding
checks original costs and coverage. Incomplete solves with a valid incumbent
retain `gap=inf`; solves without a valid incumbent raise `RuntimeError`.
The same options work on `DirectedGroupSteinerProblem`.

## Combining group choices and node costs

This path minimizes

$$\sum_{e \in E} c_e x_e + \sum_{v \in V} c_v y_v$$

over a connected tree meeting all group and mandatory-terminal requirements.
Every selected node is charged exactly once, including terminal nodes. Missing
node costs are zero; missing edge costs are one. `weight=` selects the edge-cost
attribute. All costs must be finite and non-negative.

For example, adding a cost to A changes which member of the first group wins:

```python
import networkx as nx
from steinerpy import Problem

graph = nx.Graph()
graph.add_weighted_edges_from([("A", "C", 1), ("B", "C", 3)])
groups = [["A", "B"], ["C"]]

edge_only = Problem(graph, groups=groups).get_solution()
with_nodes = Problem(graph, groups=groups, node_costs={"A": 10}).get_solution()

assert edge_only.objective == 1       # A–C
assert with_nodes.objective == 3      # B–C
assert set(with_nodes.selected_nodes) == {"B", "C"}
```

Add `hop_limit` to limit the **total number of selected original edges**:

```python
problem = Problem(graph, groups=groups, node_costs={"A": 10}, hop_limit=1)
solution = problem.get_solution(solver="highs")  # Gurobi is also supported
```

`max_degree` can also be supplied, alone or with `hop_limit`. It counts original
edges incident to each original node, combining both split ports and excluding
paid/source/group arcs. Decoding independently checks original-node degrees.

The limit is a non-negative integer. Original zero-cost edges count. Paid-node
traversals, the artificial source and group connectors consume no hops; node
costs are still charged normally. All groups and mandatory terminals remain
required. A zero limit allows a single original node only when it meets every
requirement, and that node's cost is still paid. Decoding independently checks
the selected original-edge count.

Hop limits also work with mandatory `terminals` on this composition path. Use
`node_costs={}` to select it when no node charges are needed. The legacy
`node_weighted` variant and plain undirected group adapter retain their existing
modifier restrictions.

Results are `NodeWeightedSolution` objects with original edges and
`selected_nodes`, including singleton solutions with no edges. Inputs are
snapshotted for this path; artificial nodes and arcs never appear in results.
Infeasibility or termination without a valid incumbent raises `RuntimeError`.
An interrupted solve with a valid incumbent returns `gap=inf`, without claiming
optimality.

HiGHS and Gurobi use their existing directed-cut solve loops. Preprocessing is
automatically disabled; explicitly enabling it is rejected. This first
composition requires a simple undirected graph without self-loops and does not
yet support signed rewards, budgets, heuristic-only solving,
optimum enumeration, or accelerator/reduction options.

The node-splitting stage is shared with the legacy node-weighted solver. It
retains explicit mappings to original nodes and edges; group sinks attach after
each member's paid node traversal. Distinct original labels such as `1` and
`"1"` retain separate identities. Hop limits count only cross-arcs representing
original edges, using the same provenance-aware hop constraints as directed
groups. Paid and artificial arcs are excluded.

Group-terminal attachment is also shared with the legacy undirected and
directed group solvers. The stage records which original group member each
artificial connector represents. Directed group terminals are sinks, and
composition connects them to output nodes after paid node traversal.

## Explicit variants and compatibility

The following explicit calls remain available without migrating existing code:

| `variant` | Data arguments besides `graph` | Existing implementation |
|---|---|---|
| `steiner` | `terminal_groups` | `SteinerProblem` |
| `directed` | `root`, `terminals` | `DirectedSteinerProblem` |
| `group` | `groups` | `GroupSteinerProblem` |
| `directed_group` | `groups`, `root` | `DirectedGroupSteinerProblem` |
| `partial_terminal` | `terminal_groups`, `partial_terminals` | `PartialTerminalSteinerProblem` |
| `full_terminal` | `terminal_groups` | `FullTerminalSteinerProblem` |
| `hop_constrained` | `root`, `terminals`, `hop_limit` | `HopConstrainedSteinerProblem` |
| `prize_collecting` | `terminal_groups`, `node_prizes`; optional `penalty_cost`, `penalty_budget` | `PrizeCollectingProblem` |
| `directed_prize_collecting` | `node_prizes`; optional `root` | `DirectedPrizeCollectingProblem` |
| `node_weighted` | `terminal_groups`, `node_weights` | `NodeWeightedSteinerProblem` |
| `max_weight_connected` | `node_weights`; optional `root` | `MaxWeightConnectedSubgraph` |
| `budgeted_max_weight_connected` | `node_weights`, `node_costs`, `node_budget`; optional `root` | `BudgetedMaxWeightConnectedSubgraph` |
| `rectilinear` | `points` (omit `graph`) | `RectilinearSteinerProblem` |

Pass `budget=` and `max_degree=` as modifiers where the selected implementation
supports them. `budget` is an edge-cost budget; `node_budget` is a vertex-cost
budget. Solver arguments belong to `get_solution()`. `Problem.variants` lists
the supported names.

`get_optimal_solutions()` retains each legacy variant's enumeration restrictions.
For advanced access, `problem.implementation` is the selected legacy object or
the new private composition strategy. Low-level model builders accept the legacy
objects; the composition strategy owns its compilation and exposes `cut_stats`.
`problem.variant` is the resolved legacy name, or `None` for a composed problem;
`Problem.variants` lists the explicit legacy names. No legacy interface is
deprecated by this change.

## Objective conventions

Edge costs, node costs, and rewards can share a linear minimization expression:

$$L(x,y) = \sum_e c_e x_e + \sum_v c_v y_v.$$

Selected nodes have positive coefficients for costs and negative coefficients
for rewards. A variant can report a different value through a constant and a
sign change: `reported = sign * L + offset`.

| Solve path | Edge coefficients | Node coefficients | Reported value |
|---|---|---|---|
| Standard, directed, group, terminal-leaf | Original edge costs | Zero | `L` |
| Node-weighted, or group plus node costs | Original edge costs | Original node costs | `L` |
| Classic prize collecting (`pc_transform=True`), including directed prizes | Original edge costs | Negative prizes | `L + sum(all prizes)` |
| Maximum-weight transform (`pc_transform=True`, also heuristic mode) | Zero | Negative signed node weights | `-L` |
| Node-budgeted maximum weight | Zero | Negative signed node weights | `-L`; node costs enter the budget constraint |

For classic prize collecting, the offset restores the prizes of omitted nodes:
`edge costs - collected prizes + all prizes`. For example, collecting all prizes
worth 10 with edge cost 2 gives `L = -8` and a reported cost of 2. Maximizing
signed node weight instead minimizes its negation and reports the opposite sign.

Both node-budgeted maximum-weight solvers use the same signed node-weight
coefficients and reporting conversion as the normalized specification. Negative
node weights remain costs for undesirable connectors; missing node weights
default to zero. Edge attributes supply topology only, and the separate
`node_costs` mapping determines budget usage.

Some existing paths require additional variables:

- The legacy prize-collecting default minimizes edge costs minus collected
  prizes plus `penalty_cost` for each unconnected terminal requirement. Its
  report has no prize offset. The default penalty is 1000.
  Its solver and the normalized specification share the same coefficient
  conversion. Native solving/reporting use an edge cost of one when the
  selected weight attribute is missing.
- The legacy maximum-weight default uses that penalty formulation with rewards
  `max(node_weight, 0)` and penalty `max(positive rewards) + 1`. Its reported
  value follows the minimized penalty objective. The transform path uses the
  signed-weight objective in the table.
- An edge-budgeted tree/forest minimizes the number of unconnected terminal
  requirements and reports `total requirements - unconnected requirements`.
  Repeated vertices in separate forest requirements count separately.

These penalty variables and the existing connectivity restrictions remain
explicit. Coefficient conversion alone does not make the legacy penalty model's
feasible solutions identical to classic prize collecting. Solve-time
`pc_transform` overrides select the corresponding convention.

Transformation bookkeeping is also separate: SAP cost shifts, MWCS-to-PCSTP
constants, terminal-leaf Big-M corrections, mandatory node-cost corrections,
and preprocessing fixed costs map solver values back to the original graph.
Rectilinear inputs need their Hanan grid before graph coefficients exist.
Optimality gaps retain the existing solver-specific handling.

This common representation describes existing objectives. General signed
`node_costs` and additional feature combinations still require supported
formulations and transformations. Missing edge costs retain their existing
defaults: one for standard/composed costs, zero in legacy node splitting.

## Variant definitions

The variants accepting `terminal_groups` can be solved as a **Steiner tree** (a single group of terminals that must all be connected) or as a **Steiner forest** (multiple independent groups, each connected within itself but not necessarily to other groups).
Simply pass a list of terminal lists as `terminal_groups` — one list for a tree, multiple lists for a forest.

| Variant | Class | Description |
|---------|-------|-------------|
| Steiner tree / forest | {class}`~steinerpy.SteinerProblem` | Classic minimum-cost subgraph that connects one or more groups of terminals. |
| Prize-collecting Steiner tree / forest | {class}`~steinerpy.PrizeCollectingProblem` | Each terminal carries a prize; the solver balances the prize collected against the edge and penalty costs, so not all terminals need to be connected. |
| Node-weighted Steiner tree / forest | {class}`~steinerpy.NodeWeightedSteinerProblem` | Nodes carry costs instead of (or in addition to) edges. Internally uses a node-splitting transformation to convert the problem to a standard edge-weighted formulation. |
| Maximum-weight connected subgraph | {class}`~steinerpy.MaxWeightConnectedSubgraph` | Finds a connected subgraph maximising the total node weight. Nodes with negative weights are included only when they are needed as connectors. Convenience subclass of `PrizeCollectingProblem`. |
| Directed Steiner tree (arborescence) | {class}`~steinerpy.DirectedSteinerProblem` | The graph is directed (`nx.DiGraph`) and a designated root node must reach every terminal via directed paths. |
| Directed prize-collecting Steiner tree (arborescence) | {class}`~steinerpy.DirectedPrizeCollectingProblem` | Prize-collecting on a directed graph (`nx.DiGraph`): minimises arc cost plus forgone prizes over a directed arborescence, rooted anywhere (`root=None`) or at a mandatory root. Solved exactly via a directed PCSTP → SAP transformation (Rehfeldt & Koch 2020, adapted to digraphs). |
| Partial / full terminal Steiner tree | {class}`~steinerpy.PartialTerminalSteinerProblem`, {class}`~steinerpy.FullTerminalSteinerProblem` | Designated terminals (or *all* terminals, for the full variant) must be **leaves** of the tree. Transformed to a plain Steiner tree problem (Rehfeldt thesis §5.1). |
| Group Steiner tree | {class}`~steinerpy.GroupSteinerProblem` | Connect at least one vertex from each of several vertex *groups*. Voss (1988) super-terminal transformation to a plain Steiner tree problem (thesis §5.7). |
| Directed group Steiner tree (arborescence) | {class}`~steinerpy.DirectedGroupSteinerProblem` | Directed variant of `GroupSteinerProblem`: a rooted arborescence that reaches at least one vertex from each group. |
| Hop-constrained directed Steiner tree | {class}`~steinerpy.HopConstrainedSteinerProblem` | Directed arborescence with a bound on the number of arcs (hops); terminals have no outgoing arcs (thesis §5.8). |
| Rectilinear Steiner minimum tree | {class}`~steinerpy.RectilinearSteinerProblem` | Minimum-length tree of horizontal/vertical segments through a set of points (L1 metric), solved exactly on the Hanan grid (thesis §5.4). |
| Max-weight connected subgraph with budget | {class}`~steinerpy.BudgetedMaxWeightConnectedSubgraph` | Maximise total node weight subject to a **vertex-cost budget** on the chosen subgraph (thesis §5.6). |

Several of these variants implement the "further related problems" of Chapter 5 of D. Rehfeldt's PhD thesis (*Faster algorithms for Steiner tree and related problems*, TU Berlin 2021), reusing the same directed-cut kernel by transformation.

All edge and arc costs used by these exact Steiner formulations must be
non-negative. This restriction does not apply to node *weights* in
`MaxWeightConnectedSubgraph` or
`BudgetedMaxWeightConnectedSubgraph`, where negative values mean that a node
is an undesirable connector and are part of the problem semantics. MWCSPB also
treats edge attributes as topology-only; its objective and budget use node
weights and node costs.

```python
from steinerpy import (
    PartialTerminalSteinerProblem, FullTerminalSteinerProblem, GroupSteinerProblem,
    DirectedGroupSteinerProblem, HopConstrainedSteinerProblem, RectilinearSteinerProblem,
    BudgetedMaxWeightConnectedSubgraph, DirectedPrizeCollectingProblem,
)

# Partial-terminal (terminals B and D must be leaves):
PartialTerminalSteinerProblem(G, [['A', 'B', 'D']], partial_terminals=['B', 'D']).get_solution()

# Group Steiner tree (one vertex from each group):
GroupSteinerProblem(G, [['A', 'B'], ['C', 'D']]).get_solution()

# Directed Group Steiner tree (arborescence from 'r' reaching one vertex from each group):
DirectedGroupSteinerProblem(DG, [['A', 'B'], ['C', 'D']], root='r').get_solution()

# Hop-constrained directed Steiner tree (<= 3 arcs):
HopConstrainedSteinerProblem(DG, root='r', terminals=['x', 'y'], hop_limit=3).get_solution()

# Directed prize-collecting Steiner tree (rooted arborescence from 'r'):
DirectedPrizeCollectingProblem(DG, node_prizes={'x': 5, 'y': 3}, root='r').get_solution()

# Rectilinear Steiner minimum tree through points in the plane:
RectilinearSteinerProblem([(0, 0), (2, 0), (0, 2)]).get_solution()

# Max-weight connected subgraph with a vertex-cost budget:
BudgetedMaxWeightConnectedSubgraph(G, node_weights, node_costs, node_budget=5).get_solution()
```

## Optional constraint modifiers

Two side-constraints are available as **optional keyword arguments** on compatible
problem variants. Transformation restrictions are listed above:

| Modifier kwarg | Effect |
|----------------|--------|
| `max_degree=k` | Every node in the solution has at most *k* incident edges. |
| `budget=B` | Total edge cost may not exceed *B*; the solver maximises the number of connected terminals and returns a {class}`~steinerpy.BudgetSolution`. |

For budgeted forests, terminal groups may overlap and share selected edges;
each edge is charged once. Connected terminals are counted per group, so a
vertex required by two groups counts twice when both requirements are met.
Each group's first terminal is its root and counts as connected even when
isolated. Both HiGHS and Gurobi support edge-budget solving, including shared
forest edges, degree limits, and directed arc orientation. Select Gurobi with
`problem.get_solution(solver="gurobi")`.

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
