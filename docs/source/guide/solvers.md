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
