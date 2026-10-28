# Problem variants

Use {class}`~steinerpy.Problem` to describe a model with a NetworkX graph,
connectivity requirements, costs, and constraints. Most combinations do not need
a variant name. Start with {doc}`../examples/quickstart` for a complete first solve.

## Choose the connectivity

The difference between `terminals` and `groups` determines what must connect:

| Requirement | Input | Meaning |
|---|---|---|
| Tree | `terminals=["A", "B", "C"]` | Connect every listed node. |
| Forest | `terminals=[["A", "B"], ["C", "D"]]` | Connect each list internally. Connections may share edges; separate components are allowed. |
| Group choices | `groups=[["A", "B"], ["C", "D"]]` | Connect at least one member of each group in one component. |
| Directed tree | A `DiGraph`, `root="A"`, `terminals=["C", "D"]` | Follow directed paths from the root to every terminal. |

Flat terminals and a single nested list are equivalent. Use lists for nested
forest requirements; tuple-valued node labels remain individual nodes.
With directed forest requirements, the first terminal in each list is its root
unless you supply `root=`.

You can combine groups with mandatory terminals. For example, this requires D
and a choice of A or B:

```python
import networkx as nx
from steinerpy import Problem

graph = nx.Graph()
graph.add_weighted_edges_from([("A", "D", 1), ("B", "D", 3)])
problem = Problem(graph, groups=[["A", "B"]], terminals=["D"])
solution = problem.get_solution()
print(solution.objective)  # 1.0
```

Groups can overlap. When groups coexist with forest requirements, the group
choices share one connected component; the other requirements need only connect
internally. See {doc}`../examples/connectivity` for plotted examples.

## Choose costs and rewards

Edge costs come from the `weight` attribute and default to one. Use
`weight="cost"` for another attribute. Set costs explicitly when their absence
would change your model.

| Option | Objective or effect |
|---|---|
| No reward options | Minimize selected edge costs. |
| `node_costs={...}` | Also pay for every selected node, including terminals. Missing node costs are zero. |
| `node_prizes={...}` | Minimize edge/node costs plus prizes of omitted nodes. |
| `node_weights={...}` | Maximize the sum of selected signed weights. Edge/node costs are budget resources. |
| `budget=B` | Limit edge cost. Without rewards or penalties, maximize covered requirements, then minimize cost. |
| `node_budget=B` | Limit selected-node costs independently of the edge budget. |

Choose either prizes or signed weights. Edge costs, node costs, and prizes must
be finite and non-negative; signed node weights can be negative.
With rewards, requirements stay mandatory unless `penalty_cost` makes them
optional. Explicit terminals alongside groups remain mandatory.
`penalty_budget` limits missed requirements.

A budget therefore needs careful interpretation: adding one to a plain
connection problem can make some requirements optional, whereas adding one to a
reward model limits resources. {doc}`../examples/costs_and_budgets` compares these
objectives on small graphs.

## Add constraints

| Option | Meaning on the original input graph |
|---|---|
| `max_degree=k` | At most k incident selected edges per node; directed graphs count incoming plus outgoing arcs. |
| `hop_limit=k` | At most k selected edges or arcs in total, not k per root-to-terminal path. |
| `partial_terminals=[...]` | Listed terminals or group members have degree at most one when selected. |

Degree and hop limits are non-negative integers. Zero-cost edges still count.
Artificial connectors used internally do not. Options can be combined where
supported; invalid combinations raise `TypeError` or `ValueError`.
See {doc}`../examples/constraints` for examples where each limit changes the tree.

## Other formulations

For rectilinear geometry, supply points instead of a graph:

```python
problem = Problem(variant="rectilinear", points=[(0, 0), (2, 0), (0, 2)])
solution = problem.get_solution()
```

Named classes and explicit `variant=` calls remain available for existing code.
Some use different cost defaults and objective conventions. Consult
{ref}`problem-compatibility` before migrating them. Solver arguments such as
`time_limit` belong to `get_solution()`; see {doc}`solvers` for result handling.

## Detailed reference

<span id="one-configurable-problem-class"></span>

- [One configurable problem class](../reference/compatibility.md#input-restrictions)

<span id="directed-groups-with-hop-and-degree-limits"></span>

- [Directed groups with hop and degree limits](../reference/compatibility.md#directed-groups-with-hop-and-degree-limits)

<span id="directed-groups-with-an-edge-budget"></span>

- [Directed groups with an edge budget](../reference/compatibility.md#directed-groups-with-an-edge-budget)

<span id="combining-group-choices-and-node-costs"></span>

- [Combining group choices and node costs](../reference/compatibility.md#combining-group-choices-and-node-costs)

<span id="edge-budgets-with-node-costs"></span>

- [Edge budgets with node costs](../reference/compatibility.md#edge-budgets-with-node-costs)

<span id="transformation-compatibility"></span>

- [Transformation compatibility](../reference/compatibility.md#transformation-compatibility)

<span id="explicit-variants-and-compatibility"></span>

- [Explicit variants and compatibility](../reference/compatibility.md#explicit-variants-and-compatibility)

<span id="objective-conventions"></span>

- [Objective conventions](../reference/compatibility.md#objective-conventions)

<span id="variant-definitions"></span>

- [Variant definitions](../reference/compatibility.md#variant-definitions)

<span id="optional-constraint-modifiers"></span>

- [Optional constraint modifiers](../reference/compatibility.md#optional-constraint-modifiers)

<span id="problem-compatibility"></span>

- {ref}`problem-compatibility`
