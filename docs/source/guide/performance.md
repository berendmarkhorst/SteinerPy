# Performance features

All features on this page are **opt-in** and **preserve the optimum**: with the default settings you get the plain exact solver, and enabling any of these flags returns the same optimal objective (or, for the heuristic-only mode, a certified bound on it).

## Dual-ascent accelerator

An optional [Wong (1984)](https://doi.org/10.1007/BF02612335) dual-ascent procedure can speed up the exact solve.
It computes, cheaply, a lower bound, a feasible primal solution, and *reduced costs*; following [Leitner et al. (2018)](https://doi.org/10.1007/s10589-017-9966-x) the reduced costs are used to **fix variables to zero** (eliminate arcs/edges that cannot appear in any optimal solution) before HiGHS/Gurobi runs.
When the bound matches the heuristic, the instance is solved **without building the ILP at all**.

```python
# Enable per problem (constructor) ...
solution = SteinerProblem(graph, terminal_groups, dual_ascent=True).get_solution()

# ... or per call (overrides the constructor flag)
solution = SteinerProblem(graph, terminal_groups).get_solution(dual_ascent=True)
```

It is off by default and returns the same optimum as the baseline.
Supported for Steiner **tree**, **forest** (multi-root), and **directed** ({class}`~steinerpy.DirectedSteinerProblem`) problems; it is skipped automatically when a `budget` or `max_degree` modifier is set, and for the prize-collecting / node-weighted variants.

When enabled it additionally **warm-starts** the cut loop with the Steiner cuts found during dual ascent and runs a **multi-start primal** from several roots, so many instances are solved entirely by dual ascent with no ILP.

## Bound-based graph reduction

```python
# Delete edges proven non-optimal by the dual-ascent reduced costs, then cascade
# the degree reductions — shrinking the model before the solve.
solution = SteinerProblem(graph, terminal_groups, da_reduce=True).get_solution()
```

`da_reduce` is a **reduction test**: it removes edges that the dual-ascent reduced costs prove cannot appear in any optimal solution and cascades the degree-1/degree-2 reductions to a fixpoint.
It requires `preprocess=True` (the default), applies to undirected problems only, is skipped under a `budget`/`max_degree` modifier, and preserves the optimum (solutions still map back to the original graph).
It composes with `dual_ascent=True`.

## Heavy graph reductions

```python
# Special Distance + long-edge edge-deletion tests, interleaved with the degree
# reductions to a fixpoint — heavier preprocessing for harder instances.
solution = SteinerProblem(graph, terminal_groups, heavy=True).get_solution()

# Fine-grained control (heavy=True turns on whichever applies):
SteinerProblem(graph, terminal_groups, special_distance=True, long_edge=False)
```

`heavy=True` enables two classic **alternative-based reduction tests** from the Steiner-tree literature, both of which only delete edges that are provably in no optimal solution and then cascade the degree-1/degree-2 reductions:

- **Special Distance (bottleneck Steiner distance) test** — deletes an edge `e = {v, w}` when the bottleneck distance between `v` and `w` through the *terminal distance network* is below `c(e)` (Rehfeldt & Koch, *Math. Prog. B* 197, 2023, Thm 1; surveyed in Ljubić, *Networks* 77, 2021, §4). Catches long edges that have a cheaper terminal-hopping detour, which the degree reductions miss. **Steiner tree only** (automatically skipped for the multi-group forest, where terminal-hopping would be unsound).
- **Long-edge / alternative-path test** — deletes an edge when a strictly cheaper detour exists in `G \ e`. Valid for both Steiner **tree** and **forest**.

Both tests are implemented with the fast constructions used by state-of-the-art SPG solvers: the Special Distance test builds the terminal distance network from a single multi-source Dijkstra (terminal Voronoi diagram) and Mehlhorn's (1988) boundary MST — `O(m + n log n)` rather than one shortest-path tree per terminal — and the long-edge test runs one bounded Dijkstra per *vertex* rather than per *edge* (Rehfeldt & Koch 2023, §2.3).
In practice this is several times faster on large, terminal-rich instances (≈5–10× for Special Distance, ≈3–5× for long-edge).

Like `da_reduce`, `heavy` requires `preprocess=True` (the default), applies to undirected problems only, is skipped under a `budget`/`max_degree` modifier (those variants do not minimise edge cost), and preserves the optimum — solutions still map back to the original graph.
It composes with `da_reduce=True` and `dual_ascent=True`; a good "throw everything at it" configuration is:

```python
SteinerProblem(graph, terminals, heavy=True, da_reduce=True, dual_ascent=True)
```

## Heuristic-only mode

An exact solver can't match a polynomial-time heuristic such as `networkx.steiner_tree` in general.
When you want that speed and can accept an approximate answer, pass `exact=False`:

```python
# Return the dual-ascent primal directly — no ILP is built or solved.
solution = SteinerProblem(graph, terminal_groups).get_solution(exact=False)

print(solution.objective)  # heuristic tree weight (an upper bound on the optimum)
print(solution.gap)        # PROVEN optimality gap: 0.0 == provably optimal
```

Unlike a pure heuristic, the returned `Solution.gap` is a **valid optimality certificate**: `gap == 0.0` means the heuristic tree is provably optimal, and a positive gap bounds how far it could be from the optimum — something `networkx.steiner_tree` (which gives no lower bound) cannot provide.
It is supported for plain Steiner **tree**/**forest** and **directed** problems, and raises `NotImplementedError` for the budget/degree-constrained variants.
The default is `exact=True` (solve to optimality).

## Prize-collecting / MWCSP acceleration

{class}`~steinerpy.PrizeCollectingProblem` and {class}`~steinerpy.MaxWeightConnectedSubgraph` default to a penalty/Big-M flow ILP.
For the **classic forgo-prize PCSTP** (and the MWCSP) you can opt into a much faster path that — following Rehfeldt & Koch (MWCSP 2019; PCSTP 2020) — transforms the problem to a rooted Steiner arborescence and reuses the dual-ascent + directed-cut machinery:

```python
# Exact, accelerated (often proves optimality without any ILP):
sol = PrizeCollectingProblem(graph, [[root]], node_prizes, penalty_cost=0).get_solution(pc_transform=True)

# Heuristic-only: dual-ascent primal, no ILP, with a PROVEN optimality gap:
sol = PrizeCollectingProblem(graph, [[root]], node_prizes, penalty_cost=0).get_solution(exact=False)

# Prize-safe edge reduction (prize-constrained distance) before solving:
sol = PrizeCollectingProblem(graph, [[root]], node_prizes, penalty_cost=0, pc_reduce=True).get_solution(pc_transform=True)

# MWCSP uses the exact MWCSP -> PCSTP -> SAP mapping:
sol = MaxWeightConnectedSubgraph(graph, node_weights).get_solution(pc_transform=True)
```

All three flags are off by default (the penalty ILP is unchanged) and gated to the classic forgo-prize objective: a `penalty_budget`, multiple terminal groups, or a non-zero `penalty_cost` raises `NotImplementedError` (use the default penalty ILP for those).
`pc_reduce` deletes only edges provably in no optimal solution and removes no nodes, so every prize is preserved.
