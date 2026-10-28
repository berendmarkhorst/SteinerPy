(solver-backends)=
# Solving and interpreting results

Call `get_solution()` on a configured {class}`~steinerpy.Problem`.
HiGHS is installed with SteinerPy. Gurobi needs its Python package and a license;
see {doc}`../setup/installation`.

| `solver` | Backend | Availability |
|---|---|---|
| `"highs"` | HiGHS | Default; no additional installation. |
| `"gurobi"` | Gurobi | Optional package and license. |

```python
import networkx as nx
from steinerpy import Problem

graph = nx.Graph()
graph.add_weighted_edges_from([("A", "B", 1), ("B", "C", 2), ("A", "C", 5)])
problem = Problem(graph, terminals=["A", "C"])
solution = problem.get_solution(solver="highs", time_limit=60)
```

Both backends solve the supported model; they can return different selections
when solutions tie. `threads` controls solver threads and `log_file` selects a
solver log. A time limit can stop the solve before optimality is proved.

## Inspect the solution

```python
print(solution.edges)      # selected edges in the original graph
print(solution.objective)  # 3.0 in this example
print(solution.gap)        # 0.0: optimality proved
print(solution.runtime)    # reported solver runtime in seconds
```

`objective`, `edges`, `gap`, and `runtime` are the common result fields.
The objective depends on the model: cost for a plain tree, costs plus omitted
prizes for prize collection, signed weight for maximum-weight selection, or
coverage for a plain budget problem. Additional fields such as `selected_nodes`
and `connected_terminals` depend on the result type; see the
{ref}`solution-reference`.

Use `solution.edges` for the original graph, including edges restored after
preprocessing. Measure wall-clock time around construction and solving if you
need the total time for your application.

(input-and-certificate-guarantees)=
## Gaps and unsuccessful solves

`gap == 0.0` means optimality has been proved. A finite positive gap describes
the remaining certified bound difference; `math.inf` means no finite gap is
reported. The exact convention depends on the formulation, so do not interpret
every gap as a percentage error against a known optimum.

An interrupted solve may return a validated feasible solution with a positive
or unknown gap. If no valid incumbent exists, including an infeasible problem,
`get_solution()` raises `RuntimeError`. A time limit alone is never proof of
optimality. Invalid costs or unsupported model combinations can fail earlier,
when constructing the problem.

## Enumerating multiple optimal solutions

Use `get_optimal_solutions()` when you need distinct selections with the same
optimal objective. This example has two equally short paths:

```python
graph = nx.Graph()
graph.add_weighted_edges_from([
    ("A", "B", 1), ("B", "D", 1), ("A", "C", 1), ("C", "D", 1),
])
problem = Problem(graph, terminals=["A", "D"], preprocess=False)
pool = problem.get_optimal_solutions(limit=10, time_limit=30)
print(len(pool), pool.exhausted)  # 2 True
```

`pool.exhausted` is true only when no further optimum exists. A limit or an
interrupted proof can leave it false. `time_limit` applies to each solve in the
enumeration loop, not the entire pool.

Classic preprocessing can remove tied alternatives, which is why this example
disables it. Composed features and legacy transformed variants have different
enumeration restrictions. See the
[compatibility reference](../reference/compatibility.md#enumerating-multiple-optimal-solutions)
for scope and enumeration-safe preprocessing, and {doc}`performance` for tuning
ordinary solves.

## Detailed reference

<span id="example-the-diamond-graph-tie"></span>

- [Example: the diamond-graph tie](../reference/compatibility.md#example-the-diamond-graph-tie)

<span id="implementation-and-scope"></span>

- [Implementation and scope](../reference/compatibility.md#implementation-and-scope)
