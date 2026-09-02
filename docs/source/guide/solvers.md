# Solver backends

Every problem class exposes a `solver` parameter on `get_solution()`.
Two backends are supported:

| `solver` value | Backend | Notes |
|----------------|---------|-------|
| `"highs"` (default) | [HiGHS](https://highs.dev/) via *highspy* | Always available; cut-based formulation solved iteratively (re-solve loop). |
| `"gurobi"` | [Gurobi](https://www.gurobi.com/) via *gurobipy* | Optional; requires *gurobipy* and a valid Gurobi license. Connectivity cuts are injected as **lazy constraints** inside a branch-and-cut callback, which lets Gurobi exploit its full branch-and-bound tree. |

```python
# Use HiGHS (default — no extra installation required)
solution = SteinerProblem(graph, terminal_groups).get_solution()

# Use Gurobi (requires gurobipy + license)
solution = SteinerProblem(graph, terminal_groups).get_solution(solver="gurobi")
```

Both solvers implement the same cut-based (DO-D) formulation from Markhorst et al. (2025) and produce identical optimal solutions.
Gurobi may be faster on larger instances because callbacks avoid repeated re-solves from scratch.

## Enumerating multiple optimal solutions

`get_solution()` returns exactly one optimal Steiner tree. When multiple
trees tie for the optimal cost, `get_optimal_solutions()` enumerates them
instead of silently returning an arbitrary one:

```python
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

`get_optimal_solutions()` **requires `preprocess=False`**: the default graph
reduction pipeline can arbitrarily collapse tied-cost alternatives (e.g.
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
problem = SteinerProblem(g, [["A", "D"]], preprocess=True, enumeration_safe=True)
pool = problem.get_optimal_solutions()  # no ValueError, and still 2 tied trees
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
from steinerpy import SteinerProblem

g = nx.Graph()
for u, v in [("A", "B"), ("B", "C"), ("C", "D"), ("A", "E"), ("E", "F"), ("F", "D")]:
    g.add_edge(u, v, weight=1)

problem = SteinerProblem(g, [["A", "D"]], preprocess=False)
pool = problem.get_optimal_solutions()

print(len(pool), pool.exhausted)  # 2 True
for solution in pool:
    print(sorted(solution.edges))
# [('A', 'B'), ('B', 'C'), ('C', 'D')]
# [('A', 'E'), ('E', 'F'), ('F', 'D')]
```

### Implementation and scope

Both backends use the same external no-good-cut enumeration loop: the model
is rebuilt from scratch each iteration, with a cut excluding every
previously-found edge set added before re-solving. On `solver="gurobi"` this
means the native Gurobi solution pool (`PoolSearchMode`) is **not** used —
whether it respects this model's lazily-added connectivity cuts is an open,
version-dependent question that cannot be verified in this project's test
suite, so the provably correct external loop is used for both backends
(at the cost of Gurobi's single-solve pooling speed advantage).

`get_optimal_solutions()` is not yet supported for problem classes whose
`get_solution()` transforms the model in a way this method's plain
edge-indicator enumeration can't replicate: `PrizeCollectingProblem` (and
its subclasses), `PartialTerminalSteinerProblem` (and
`FullTerminalSteinerProblem`), `GroupSteinerProblem` (and
`DirectedGroupSteinerProblem`), `RectilinearSteinerProblem`, and
`NodeWeightedSteinerProblem`. Each raises `NotImplementedError`. Budget-
constrained instances (`budget=...`) are also unsupported, for the same
reason.
