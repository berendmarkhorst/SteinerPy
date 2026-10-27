# Getting started

A Steiner tree problem asks for a minimum-cost subgraph that connects a given set of *terminal* nodes, optionally passing through other nodes.
In SteinerPy you build a NetworkX graph, list the terminals you want connected, and call {class}`~steinerpy.Problem`:

```python
import networkx as nx
from steinerpy import Problem

# Create a graph
G = nx.Graph()
G.add_edge("A", "B", weight=1)
G.add_edge("B", "C", weight=2)
G.add_edge("C", "D", weight=1)

# Solve with HiGHS (default)
problem = Problem(G, terminals=["A", "D"])
solution = problem.get_solution()

print(f"Optimal cost: {solution.objective}")
print(f"Selected edges: {solution.edges}")
```

The returned {class}`~steinerpy.Solution` reports the objective value, the selected edges, the runtime, and — importantly — a **proven optimality gap**: `solution.gap == 0.0` means the solution is provably optimal.

Use `solution.edges` for edges in the original input graph, including edges
restored after preprocessing.

## Trees and forests

Flat `terminals` gives a Steiner **tree**. Nested lists give a Steiner
**forest**: each list must connect internally, while different lists need not
connect to each other. Their paths may share edges if that reduces cost:

```python
# Steiner forest: connect A–B and, independently, C–D
solution = Problem(G, terminals=[["A", "B"], ["C", "D"]]).get_solution()
```

For choices rather than mandatory terminals, use
`groups=[["A", "B"], ["C", "D"]]`: one connected tree must contain at least
one member of each group. Tuple-valued node labels remain atomic inside
`terminals`; use lists for the nested forest syntax.

Existing classes still accept their original `terminal_groups` arguments.
See {doc}`../guide/variants` for supported feature combinations and
{ref}`problem-compatibility` for migration guidance.

## Next steps

- {doc}`../guide/variants` — the full catalogue of supported problem variants.
- {doc}`../guide/solvers` — choosing between HiGHS and Gurobi.
- {doc}`../guide/performance` — opt-in accelerators for harder instances.
- {doc}`../examples/quickstart` — a worked notebook covering the main features.
