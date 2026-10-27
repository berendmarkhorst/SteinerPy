# Solver backends

Select a backend with `Problem(...).get_solution(solver=...)`.
Two backends are supported:

| `solver` value | Backend | Notes |
|----------------|---------|-------|
| `"highs"` (default) | [HiGHS](https://highs.dev/) via *highspy* | Always available; cut-based formulation solved iteratively (re-solve loop). |
| `"gurobi"` | [Gurobi](https://www.gurobi.com/) via *gurobipy* | Optional; requires *gurobipy* and a valid Gurobi license. Connectivity cuts are injected as **lazy constraints** inside a branch-and-cut callback, which lets Gurobi exploit its full branch-and-bound tree. |

```python
from steinerpy import Problem

# Use HiGHS (default — no extra installation required)
solution = Problem(graph, terminals=terminals).get_solution()

# Use Gurobi (requires gurobipy + license)
solution = Problem(graph, terminals=terminals).get_solution(solver="gurobi")
```

Use a flat list of `terminals` for a tree, or nested lists for a forest.
For ordinary tree/forest problems, both solvers implement the cut-based (DO-D)
formulation from Markhorst et al. (2025) and prove the same optimal objective;
they may return different trees when several solutions tie.
Gurobi may be faster on larger instances because callbacks avoid repeated re-solves from scratch.

Edge-budget problems (`budget=...`) use the same optional-flow formulation
and result validation with either solver. They maximize connected terminal
requirements under the edge-cost budget, including overlapping forest groups
and supported degree/hop limits. For this path, connectivity flows are added
before solving instead of generating connectivity cuts in a callback.

```python
from steinerpy import Problem

solution = Problem(
    graph, terminals=[["A", "B"], ["A", "C"]], budget=10, preprocess=False
).get_solution(solver="gurobi")
```

## Input and certificate guarantees

All edge-cost Steiner variants require non-negative values in the selected
edge/arc weight attribute. This is required by the shortest-path reductions
and exact formulations.
Negative node weights remain valid for maximum-weight connected-subgraph
problems. Inputs with a negative edge or arc cost raise `ValueError` at
construction time instead of risking a disconnected negative-cost cycle in the
reported solution.

A time limit is not an optimality certificate. If a solve stops early with an
independently validated feasible incumbent, `get_solution()` may return it
with a nonzero or unknown gap (`math.inf`). If no valid incumbent exists—or
the last incumbent predates newly added connectivity cuts—the values are
discarded and the public solve raises `RuntimeError`. In particular,
`solution.gap == 0` is reported only after the solver or an independent exact
bound proves optimality.

## Enumerating multiple optimal solutions

`get_solution()` returns one solution, with its proof status reported by `gap`.
When multiple trees tie for the optimal cost, `get_optimal_solutions()` enumerates them
instead of silently returning an arbitrary one:

```python
problem = Problem(graph, terminals=terminals, preprocess=False)
pool = problem.get_optimal_solutions(
    limit=10, time_limit=300, log_file="", solver="highs", threads=None,
)

for solution in pool:
    print(solution.objective, solution.edges)

print(len(pool), pool.exhausted)  # exhausted: every optimal solution was found,
                                   # not just the first `limit` of them
```

Here, a Steiner tree is **inclusion-minimal**: removing any selected edge
would break a required terminal-group connection. Consequently, adding a
redundant zero-cost branch or cycle does not create another solution. Distinct
minimum-cost trees that use zero-cost edges are still enumerated when each of
their edges is necessary for that tree's terminal connectivity.

For ordinary tree/forest problems, `get_optimal_solutions()` **requires
`preprocess=False`**: the default graph reduction pipeline can arbitrarily
collapse tied-cost alternatives (e.g.
terminal contraction) before any ILP runs, silently erasing them from
enumeration — calling it on a `preprocess=True` instance raises `ValueError`,
unless the problem was constructed with `enumeration_safe=True` (see below).
It also bypasses every speedup dispatch used by `get_solution()`
(trivial-instance early exit, `exact=False` heuristic mode,
biconnected-component decomposition, the Dreyfus-Wagner DP, and the
dual-ascent accelerator), since each of those either returns a single
arbitrary optimum or, in the case of dual ascent's reduced-cost variable
fixing, could soundly discard an edge that appears only in a different
tied-cost solution.

Construct the problem with **`enumeration_safe=True`** instead of
`preprocess=False` to keep reduction (and its speedup) while still enumerating
every tied optimum:

```python
problem = Problem(
    graph, terminals=terminals, preprocess=True, enumeration_safe=True
)
pool = problem.get_optimal_solutions()
```

`enumeration_safe` restricts preprocessing to reductions proven to preserve
the *complete set* of optimal solutions, not merely the optimal value:
special distance, long-edge and bound-based deletions (already exact — a
strict inequality proves the deleted edge/node lies in *no* optimum) and
non-terminal degree-1 removal are unaffected; degree-2 contraction skips a
node instead of contracting it when the contracted path *ties* an existing
parallel edge; node replacement (pseudo-elimination) is disabled outright;
and of the terminal-contraction tests, only the forced degree-1 case still
fires (the adjacent-terminal, Nearest-Vertex and Short-Links tests are
skipped). See `steinerpy.graph_reducer.preprocess_graph`'s docstring for the
full soundness argument.

`enumeration_safe=True` requires strictly positive edge weights (raises
`ValueError` otherwise), and `get_optimal_solutions()` does not support
`preprocess=True` together with `max_degree` or `hop_limit` (raises
`NotImplementedError`): the structural degree reductions always run
regardless of those modifiers and are not degree- or hop-aware, so a
contraction could otherwise map back to a solution that violates the
constraint. Use `preprocess=False` in that case.

It returns an `OptimalSolutionPool`:

- `pool.solutions` — the distinct optimal `Solution` objects found, all
  sharing the same (minimum) objective value.
- `pool.exhausted` — `True` iff a probe solve *proved* no further tied-cost
  alternative exists. `False` if `limit` was reached while ties still
  remained, or if a probe hit `time_limit` (or otherwise terminated) before
  it could be proven optimal or infeasible — in the latter case enumeration
  stops early rather than returning an unproven solution.

### Example: the diamond-graph tie

```python
import networkx as nx
from steinerpy import Problem

g = nx.Graph()
for u, v in [("A", "B"), ("B", "C"), ("C", "D"), ("A", "E"), ("E", "F"), ("F", "D")]:
    g.add_edge(u, v, weight=1)

problem = Problem(g, terminals=["A", "D"], preprocess=False)
pool = problem.get_optimal_solutions()

print(len(pool), pool.exhausted)  # 2 True
for solution in pool:
    print(sorted(solution.edges))
# [('A', 'B'), ('B', 'C'), ('C', 'D')]
# [('A', 'E'), ('E', 'F'), ('F', 'D')]
```

### Implementation and scope

For ordinary trees, both backends use the same external no-good-cut enumeration loop: the model
is rebuilt from scratch each iteration, with a cut excluding every
previously-found edge set added before re-solving. On `solver="gurobi"` this
means the native Gurobi solution pool (`PoolSearchMode`) is **not** used —
whether it respects this model's lazily-added connectivity cuts is an open,
version-dependent question that cannot be verified in this project's test
suite, so the provably correct external loop is used for both backends
(at the cost of Gurobi's single-solve pooling speed advantage).

The modular API also enumerates supported combinations of groups, forests,
node costs, prizes, signed node weights, terminal leaves, direction and resource
constraints. It compares original node/edge selections and returns distinct
inclusion-minimal optima. For example:

```python
import networkx as nx
from steinerpy import Problem

graph = nx.cycle_graph(4)
problem = Problem(graph, groups=[[0, 1], [2, 3]], budget=1)
pool = problem.get_optimal_solutions()
print(len(pool), pool.exhausted)  # 2 True
```

For these composed problems, preprocessing retains the original selections
needed for enumeration; the classic preprocessing restrictions above do not
apply to every composition. See the [variant guide](variants.md) for supported
combinations and preprocessing behavior. `time_limit` applies to each solve
in the enumeration loop, rather than to the entire pool.

Explicit legacy adapters retain their enumeration restrictions. Calls using
`variant="prize_collecting"`, `"directed_prize_collecting"`,
`"max_weight_connected"`, `"budgeted_max_weight_connected"`,
`"node_weighted"`, `"rectilinear"`, or plain legacy terminal-leaf adapters
raise `NotImplementedError`. Feature-based calls with `node_prizes=`,
`node_weights=`, `node_costs=` or `partial_terminals=` use the supported
composition path instead; their objective conventions are documented in the
{ref}`problem-compatibility` guide.
