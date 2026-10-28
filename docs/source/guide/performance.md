# Performance features

Start with the default solve and a time limit. If it is too slow, compare one
setting at a time on representative inputs, checking both runtime and
`solution.gap`. The best settings depend on the model and graph.

```python
from steinerpy import Problem

problem = Problem(graph, terminals=terminals)
solution = problem.get_solution(time_limit=60)
```

## Defaults and applicability

Plain undirected tree/forest problems enable preprocessing by default. This
removes or contracts parts of the graph while preserving the optimum value;
returned edges refer to the original graph. A different tree may be returned
when several trees have the same cost.

These defaults do not describe every composition. Group choices, node costs,
rewards, direction, and resource constraints can select other formulations.
Composed preprocessing, when enabled, uses resource filtering or compatible
classic reductions. Classic accelerator flags are not generally supported by
those strategies. See the [compatibility reference](../reference/compatibility.md)
for restrictions.

| Setting | Default on the classic path | Use |
|---|---|---|
| `preprocess` | `True` for undirected tree/forest | Reduce the graph before solving. |
| `heavy` | `True` | Enable additional reductions when applicable. Requires preprocessing. |
| `dual_ascent` | `False` | Compute a lower bound and candidate solution before the exact solve. |
| `da_reduce` | `False` | Use dual-ascent bounds to reduce an undirected graph. Requires preprocessing. |
| `exact` (solve argument) | `True` | Search for and prove an optimum, subject to the time limit. |

Small plain trees also use a few-terminal dynamic program automatically when
applicable. Connectivity-cut acceleration is automatic on classic cut-based
solver paths. Neither requires a setting in ordinary use.

## Try dual ascent

For a plain tree or forest, compare the default solve with:

```python
problem = Problem(graph, terminals=terminals, dual_ascent=True)
solution = problem.get_solution(time_limit=60)
```

Dual ascent may reduce the remaining work, or prove optimality before a mixed
integer model is needed. Its overhead may also exceed the time saved.
On compatible undirected problems, add `da_reduce=True` to use its bounds for
graph reduction. Budget and degree constraints disable these classic
accelerators; other formulations have their own restrictions.

## Accept a heuristic solution

For plain tree/forest and directed problems, `exact=False` returns a heuristic
solution with a bound:

```python
solution = Problem(graph, terminals=terminals).get_solution(exact=False)
print(solution.objective, solution.gap)
```

A zero gap still proves optimality. A positive gap reports the remaining bound
difference; it is not a promise that the solution is optimal. This mode is not
available for general feature composition or budget/degree-constrained models.
Keep the exact solve when those features are required. See {doc}`solvers` for
early termination and unknown gaps.

## Threads and repeated measurements

`get_solution(threads=...)` controls solver threads. Connectivity separation has
a separate `STEINERPY_SEP_THREADS` setting, defaulting to one; more threads can
add overhead on small tasks. Record the backend, thread count, time limit, and
package version when comparing runs. Use several representative instances and
keep timeouts and missing solutions in the results.

## Enumeration and advanced settings

For classic optimal-solution enumeration, use `preprocess=False` or the supported
`enumeration_safe=True` setting so reductions do not discard tied optima.
The latter requires positive edge costs; classic degree/hop-constrained
enumeration requires preprocessing off. See
[enumeration details](../reference/compatibility.md#enumerating-multiple-optimal-solutions).

Prize transformations, experimental primal searches, cut purging, and individual
reduction switches are described in {doc}`../reference/algorithms`. They are
specialised settings, not a recommended bundle. Published measurements and their
conditions are listed in {doc}`benchmarks`.

## Detailed reference

<span id="dual-ascent-accelerator"></span>

- [Dual-ascent accelerator](../reference/algorithms.md#dual-ascent-accelerator)

<span id="connectivity-cut-separation"></span>

- [Connectivity cut separation](../reference/algorithms.md#connectivity-cut-separation)

<span id="bound-based-graph-reduction"></span>

- [Bound-based graph reduction](../reference/algorithms.md#bound-based-graph-reduction)

<span id="heavy-graph-reductions"></span>

- [Heavy graph reductions](../reference/algorithms.md#heavy-graph-reductions)

<span id="terminal-contraction"></span>

- [Terminal contraction](../reference/algorithms.md#terminal-contraction)

<span id="bound-based-deletions-bnd"></span>

- [Bound-based deletions (BND)](../reference/algorithms.md#bound-based-deletions-bnd)

<span id="enumeration-safe-preprocessing"></span>

- [Enumeration-safe preprocessing](../reference/algorithms.md#enumeration-safe-preprocessing)

<span id="few-terminal-dynamic-program"></span>

- [Few-terminal dynamic program](../reference/algorithms.md#few-terminal-dynamic-program)

<span id="cut-separation-accelerators"></span>

- [Cut-separation accelerators](../reference/algorithms.md#cut-separation-accelerators)

<span id="heuristic-only-mode"></span>

- [Heuristic-only mode](../reference/algorithms.md#heuristic-only-mode)

<span id="experimental-primal-portfolio"></span>

- [Experimental primal portfolio](../reference/algorithms.md#experimental-primal-portfolio)

<span id="prize-collecting-mwcsp-acceleration"></span>

- [Prize-collecting / MWCSP acceleration](../reference/algorithms.md#prize-collecting--mwcsp-acceleration)
