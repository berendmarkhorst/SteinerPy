# Getting started

A Steiner tree problem asks for a minimum-cost subgraph that connects a given set of *terminal* nodes, optionally passing through other nodes.
In SteinerPy you build a NetworkX graph, list the terminals you want connected, and call {class}`~steinerpy.SteinerProblem`:

```python
import networkx as nx
from steinerpy import SteinerProblem

# Create a graph
G = nx.Graph()
G.add_edge("A", "B", weight=1)
G.add_edge("B", "C", weight=2)
G.add_edge("C", "D", weight=1)

# Define terminal groups
terminal_groups = [["A", "D"]]

# Solve with HiGHS (default)
problem = SteinerProblem(G, terminal_groups)
solution = problem.get_solution()

print(f"Optimal cost: {solution.objective}")
print(f"Selected edges: {solution.selected_edges}")
```

The returned {class}`~steinerpy.Solution` reports the objective value, the selected edges, the runtime, and — importantly — a **proven optimality gap**: `solution.gap == 0.0` means the solution is provably optimal.

## Trees and forests

`terminal_groups` is a list of terminal lists.
One list gives a Steiner **tree** (all terminals in the group are connected); multiple lists give a Steiner **forest** (each group is connected within itself, but different groups need not be connected to each other):

```python
# Steiner forest: connect A–B and, independently, C–D
solution = SteinerProblem(G, [["A", "B"], ["C", "D"]]).get_solution()
```

This convention carries over to every problem variant in the package — see {doc}`../guide/variants`.

## Next steps

- {doc}`../guide/variants` — the full catalogue of supported problem variants.
- {doc}`../guide/solvers` — choosing between HiGHS and Gurobi.
- {doc}`../guide/performance` — opt-in accelerators for harder instances.
- {doc}`../examples/quickstart` — a worked notebook covering the main features.
