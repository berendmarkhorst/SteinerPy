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
forest lists are also supported: group choices share one connected component,
while the other forest requirements may remain in separate components.

Use lists for nested terminal syntax; tuple-valued node labels remain atomic.
Mixed flat/nested inputs are rejected. `terminal_groups=` remains an alias for
the old nested syntax, including the positional second argument; supplying both
`terminals` and `terminal_groups` is an error.

| Features (no `variant` required) | Supported behavior |
|---|---|
| Undirected graph + flat or nested `terminals` | Steiner tree or forest |
| Directed graph + `root` + `terminals` and/or `groups` | Reach required terminals and group choices from the root |
| Directed graph + several nested terminal lists | Directed forest; each list's first terminal is its root unless `root=` is supplied |
| `groups`, optionally with mandatory `terminals` | Select at least one member per group; forest requirements may coexist |
| Any of these + `node_costs` | Minimize original edge costs plus selected-node costs |
| Any of these + `max_degree` / `hop_limit` | Limit original-node total degree / total selected original edges or arcs |
| Any of these + `budget` (without rewards or penalties) | Maximize covered requirements within an edge-cost budget |
| `node_prizes` | Minimize edge/node costs plus prizes of omitted nodes |
| Signed `node_weights` | Maximize selected node weights; edge/node costs serve as budget resources |
| `node_costs` + `node_budget` | Limit the sum of selected-node costs, independently of the edge budget |
| `partial_terminals` | Listed terminals or group members have degree at most one when selected |

These features can be combined, including forests, groups, rewards, budgets,
degree/hop limits and terminal leaves. Choose either `node_prizes` or
`node_weights`, not both. Terminal and group requirements stay mandatory with a
reward objective unless `penalty_cost` makes them optional; with groups, explicit
terminals remain mandatory. `penalty_budget` limits missed requirements.
Rectilinear inputs and compatibility formulations use explicit `variant=` calls.

`Problem(...)` validates combinations before constructing a solver strategy or
transforming the graph. Invalid combinations raise `TypeError` or `ValueError`
with the unsupported feature named. In particular:

- Composed problems require a simple graph without self-loops and finite,
  non-negative edge/node costs. Signed rewards belong in `node_weights`.
- The old terminal-leaf classes cannot use edge budgets; use
  `Problem(..., partial_terminals=..., budget=...)` for that combination.
- The legacy node-weighted variant requires an undirected graph; its node
  splitting creates both directions for every edge.
- `pc_transform=True` requires an explicit compatible classic prize-collecting
  or maximum-weight variant without degree limits or budgets. The transformed
  directed-prize formulation uses `variant="directed_prize_collecting"`;
  feature-based directed prizes also work without that transformation.
- The explicit `budgeted_max_weight_connected` variant uses `node_budget`; additional
  edge/penalty budgets and prize-transform/reduction options are unsupported.

Node costs and prizes must be finite and non-negative. Maximum-weight objectives
accept signed node weights. For composed problems, `preprocess=True` uses
resource bounds that preserve original selections; compatible single trees can
use classic reductions. The default and available accelerators depend on the
selected formulation; see {doc}`performance`. Solver-time options and backend
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
and original-node limits. Infeasibility raises `RuntimeError`. Add mandatory
`terminals` or `node_costs` through `Problem(...)` to combine them with these
constraints; the direct legacy class retains its narrower constructor.

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
`Problem(graph, terminals=..., hop_limit=...)` or
`Problem(graph, groups=..., hop_limit=...)` when no node charges are needed.
The legacy `node_weighted` variant and direct `GroupSteinerProblem` constructor
retain their modifier restrictions.

Results are `NodeWeightedSolution` objects with original edges and
`selected_nodes`, including singleton solutions with no edges. Inputs are
snapshotted for this path; artificial nodes and arcs never appear in results.
Infeasibility or termination without a valid incumbent raises `RuntimeError`.
An interrupted solve with a valid incumbent returns `gap=inf`, without claiming
optimality.

Both HiGHS and Gurobi support these combinations. Plain group/node-cost problems
can use the shared cut formulation; additional features can select the general
model. `preprocess=True` uses safe resource filtering for these combinations.
`get_optimal_solutions()` enumerates distinct original selections, preserving
ties even with composed preprocessing. Signed rewards use `node_weights`.
Heuristic-only solving and classic accelerator options are not supported by
the composition strategies.

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

### Edge budgets with node costs

Add `budget=B` to this composition to maximize reached groups within a finite
non-negative **original-edge** budget. Among maximum-coverage solutions, the
solver minimizes original edge plus node costs in a second solve. Paid-node and
artificial arcs do not consume the edge budget. Node costs therefore influence
which equally good coverage solution is chosen, without using an edge-only
allowance. Mandatory `terminals` remain required. Covered groups must share one
connected component; separate forest requirements may remain disconnected from
it. Overlapping groups count separately. Degree and hop limits may be combined
with the budget.

The result is a `BudgetSolution`: `objective`/`connected_terminals` count reached
original groups, `total_terminals` counts all groups, `penalties` contains missed
`group_i` indices, and `selected_nodes` lists original nodes. `total_cost` is
original edge plus node cost. `gap=0` requires completion of both coverage and
cost optimization; otherwise a valid incumbent has `gap=inf`. If the second
solve has no incumbent, the first valid solution is retained. Both phases share
the requested time allowance. Invalid or missing incumbents are rejected.

```python
problem = Problem(graph, groups=groups, node_costs={"A": 10}, budget=3,
                  max_degree=2, hop_limit=2)
solution = problem.get_solution(solver="highs")
```

## Transformation compatibility

| Entry point | Hop limits | Degree limits | Edge budgets | Preprocessing |
|---|---|---|---|---|
| `Problem(graph, groups=..., node_costs=...)` | Original edges | Original node incidence | Optional groups, then original-cost tie-break | Safe resource filtering when enabled |
| `Problem(graph, groups=...)` | Original edges | Original node incidence | Optional groups | Safe resource filtering when enabled |
| `Problem(digraph, root=..., groups=...)` | Original arcs | Incoming + outgoing original arcs | Optional groups | Safe resource filtering when enabled |
| Direct `DirectedGroupSteinerProblem` | Original arcs | Incoming + outgoing original arcs | Optional groups | Disabled |
| Direct `GroupSteinerProblem` | Rejected | Rejected | Rejected | Safe resource filtering when enabled |
| `variant="node_weighted"` / `NodeWeightedSteinerProblem` | Rejected | Rejected | Rejected | Disabled; explicit true rejected |

No dummy `node_costs={}` argument is needed to combine groups with modifiers.
Group composition uses a simple graph without self-loops; parallel edges are
rejected by all three legacy transformation adapters before edge identities
can be lost. Edge-budget rows above describe coverage mode without rewards or
penalties; with those objectives, the budget is a resource constraint.

The old undirected group adapter now solves through sink-only shared
composition and still returns `Solution`. This prevents an artificial group
node from joining distinct members for free and yielding disconnected original
edges. Its augmented-graph helper remains available, but unsupported
solver/accelerator options are rejected. Constructors reject
unmapped constraints before transformation; the legacy node-weighted adapter
now accepts explicit `preprocess=False` consistently.

(problem-compatibility)=

## Explicit variants and compatibility

`Problem` is the recommended entry point for new code. Existing public classes,
their imports and `terminal_groups=` calls remain supported; this migration adds
no new deprecation or removal schedule. The already deprecated
`DegreeConstrainedSteinerProblem` and `BudgetConstrainedSteinerProblem` wrappers
remain exceptions: pass `max_degree=` or `budget=` to `Problem` instead.

| Existing call | `Problem` call |
|---|---|
| `SteinerProblem(graph, terminal_groups)` | `Problem(graph, terminals=terminal_groups)` |
| `DirectedSteinerProblem(digraph, root=r, terminals=ts)` | `Problem(digraph, root=r, terminals=ts)` |
| `GroupSteinerProblem(graph, groups)` | `Problem(graph, groups=groups)` |
| `NodeWeightedSteinerProblem(graph, terminal_groups, weights)` | `Problem(graph, variant="node_weighted", terminal_groups=terminal_groups, node_weights=weights)` preserves legacy cost defaults |
| `PrizeCollectingProblem(graph, terminal_groups, prizes)` | `Problem(graph, variant="prize_collecting", terminal_groups=terminal_groups, node_prizes=prizes)` preserves the legacy penalty objective |
| `MaxWeightConnectedSubgraph(graph, weights)` | `Problem(graph, variant="max_weight_connected", node_weights=weights)` preserves the legacy formulation |

For a feature-based node-cost model, use `node_costs=weights` instead of
`variant="node_weighted", node_weights=weights`. Check edge costs first: missing
edge weights cost **zero** in the legacy node-weighted formulation and **one**
with `node_costs`. Set them explicitly when migrating. Preserve nested terminal
lists for forest requirements; `groups` means choosing members, not connecting
every node in each list. Feature-based forests need only connect each list
internally; check connectivity and the reported objective when replacing a
legacy transformed formulation.

Likewise, `node_prizes` without an explicit compatibility variant minimizes
costs plus omitted prizes, and `node_weights` maximizes signed selected weights.
These differ from the legacy penalty objectives described below. Migration is
not a constructor rename for these formulations. Return types and extra result
fields follow the selected path; use `objective`, `edges`, `runtime`
and `gap` as the common interface.

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

`Problem.get_optimal_solutions()` supports composed features, group choices and
edge budgets through original-selection enumeration. Explicit legacy prize,
node-weighted and other transformed variants retain their enumeration
restrictions; direct legacy classes are unchanged. See {doc}`solvers` for scope.
For advanced access, `problem.implementation` is the selected legacy object or
the new private composition strategy. Low-level model builders accept the legacy
objects; the composition strategy owns its compilation and exposes `cut_stats`.
`problem.variant` is the resolved legacy name, or `None` for a composed problem;
`Problem.variants` lists the explicit legacy names.

(objective-conventions)=

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
| Composed `node_prizes` | Original edge costs | Node costs minus prizes | `L + sum(all prizes)` |
| Composed `node_weights` | Zero | Negative signed node weights | `-L`; costs enter resource budgets |
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

Use flat `terminals` for a **Steiner tree** and nested terminal lists for a
**Steiner forest**, as described above. The table links the corresponding legacy
classes for formulation details; new code normally uses `Problem`.

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
from steinerpy import Problem

# Partial-terminal (terminals B and D must be leaves):
Problem(G, terminals=['A', 'B', 'D'], partial_terminals=['B', 'D']).get_solution()

# Group Steiner tree (one vertex from each group):
Problem(G, groups=[['A', 'B'], ['C', 'D']]).get_solution()

# Directed Group Steiner tree (arborescence from 'r' reaching one vertex from each group):
Problem(DG, groups=[['A', 'B'], ['C', 'D']], root='r').get_solution()

# Hop-constrained directed Steiner tree (<= 3 arcs):
Problem(DG, root='r', terminals=['x', 'y'], hop_limit=3).get_solution()

# Directed prize-collecting Steiner tree (rooted arborescence from 'r'):
Problem(DG, node_prizes={'x': 5, 'y': 3}, root='r').get_solution()

# Rectilinear Steiner minimum tree through points in the plane:
Problem(variant='rectilinear', points=[(0, 0), (2, 0), (0, 2)]).get_solution()

# Max-weight connected subgraph with a vertex-cost budget:
Problem(G, node_weights=node_weights, node_costs=node_costs,
        node_budget=5).get_solution()
```

## Optional constraint modifiers

Two commonly used constraints are optional keyword arguments on `Problem`.
Transformation restrictions for legacy calls are listed above:

| Modifier kwarg | Effect |
|----------------|--------|
| `max_degree=k` | Every node in the solution has at most *k* incident edges. |
| `budget=B` | Total edge cost may not exceed *B*; without rewards or penalties, maximize covered terminal/group requirements and return a {class}`~steinerpy.BudgetSolution`. |

For budgeted forests, terminal groups may overlap and share selected edges;
each edge is charged once. Connected terminals are counted per group, so a
vertex required by two groups counts twice when both requirements are met.
Each group's first terminal is its root and counts as connected even when
isolated. Both HiGHS and Gurobi support edge-budget solving, including shared
forest edges, degree limits, and directed arc orientation. Select Gurobi with
`problem.get_solution(solver="gurobi")`.

```python
# Degree-constrained Steiner tree
solution = Problem(graph, terminals=terminal_groups, max_degree=2).get_solution()

# Budget-constrained Steiner tree
solution = Problem(graph, terminals=terminal_groups, budget=10.0).get_solution()

# Both constraints together
solution = Problem(graph, terminals=terminal_groups,
                   max_degree=2, budget=10.0).get_solution()

# Degree-constrained prize-collecting problem
solution = Problem(graph, terminals=terminal_groups,
                   node_prizes=node_prizes, max_degree=3).get_solution()
```

:::{note}
`DegreeConstrainedSteinerProblem` and `BudgetConstrainedSteinerProblem` are
already deprecated. Use the `Problem` keyword arguments shown above; see
{ref}`problem-compatibility` for the compatibility policy.
:::
