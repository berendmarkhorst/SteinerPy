# Performance features

All features on this page **preserve the optimum**: enabling (or disabling) any of these flags returns the same optimal objective (or, for the heuristic-only mode, a certified bound on it). The heavy graph reductions are **on by default**; the dual-ascent features remain opt-in.

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

`da_reduce` is a **reduction test**: it removes edges *and nodes* that the dual-ascent reduced costs prove cannot appear in any optimal solution (bound-based edge and node elimination) and cascades the degree-1/degree-2 reductions to a fixpoint.
It requires `preprocess=True` (the default), applies to undirected problems only, is skipped under a `budget`/`max_degree` modifier, and preserves the optimum (solutions still map back to the original graph).
It composes with `dual_ascent=True`.

## Heavy graph reductions

```python
# ON by default: Special Distance + long-edge deletion and node replacement,
# interleaved with the degree reductions to a fixpoint.
solution = SteinerProblem(graph, terminal_groups).get_solution()

# Disable them all ...
SteinerProblem(graph, terminal_groups, heavy=False)

# ... or take fine-grained control (each flag defaults to the `heavy` value):
SteinerProblem(graph, terminal_groups, special_distance=True, long_edge=False,
               replace_nodes=False)
```

`heavy` (default **True**) enables three classic **alternative-based reduction tests** from the Steiner-tree literature, each of which removes only what is provably compatible with an optimal solution and then cascades the degree-1/degree-2 reductions:

- **Special Distance (bottleneck Steiner distance) test** — deletes an edge `e = {v, w}` when the bottleneck distance between `v` and `w` through the *terminal distance network* is below `c(e)` (Rehfeldt & Koch, *Math. Prog. B* 197, 2023, Thm 1; surveyed in Ljubić, *Networks* 77, 2021, §4). The bound routes through the **two** nearest terminals of each endpoint, a strict strengthening of the classic nearest-terminal bound. **Steiner tree only** (automatically skipped for the multi-group forest, where terminal-hopping would be unsound).
- **Long-edge / alternative-path test** — deletes an edge when a strictly cheaper detour exists in `G \ e`. Valid for both Steiner **tree** and **forest**.
- **Node replacement (pseudo-elimination)** — eliminates a non-terminal of degree ≤ 4 that provably has degree ≤ 2 in at least one minimum Steiner tree (Rehfeldt & Koch 2023, Prop. 4: the criterion compares the largest terminal-MST weights against the cheapest incident edges), bridging each neighbour pair with the two-edge path cost. Replacement edges are pre-filtered by the Special Distance bound and merged into cheaper parallels, so the graph never grows. **Steiner tree only.**

The tests are implemented with the fast constructions used by state-of-the-art SPG solvers: a single two-label multi-source Dijkstra (terminal Voronoi diagram) plus Mehlhorn's (1988) boundary MST — `O(m + n log n)` rather than one shortest-path tree per terminal — shared by the Special Distance and replacement tests, and one bounded Dijkstra per *vertex* rather than per *edge* for the long-edge test (Rehfeldt & Koch 2023, §2.3). The degree-1/degree-2 cascades run in place off a change-driven worklist.

`heavy` requires `preprocess=True` (the default), applies to undirected problems only, is skipped under a `budget`/`max_degree`/`hop_limit` modifier (those variants do not minimise plain edge cost), and preserves the optimum **value** — solutions still map back to the original graph, though among several equal-cost optima a different one may be returned than with `heavy=False`.
It composes with `da_reduce=True` and `dual_ascent=True`; a good "throw everything at it" configuration is:

```python
SteinerProblem(graph, terminals, da_reduce=True, dual_ascent=True)
```

## Cut-separation accelerators

The exact solve enforces connectivity lazily with directed Steiner cuts found by minimum-cut separation.
Three classic accelerators are applied automatically (no flags needed):

- **Creep flows** and **back cuts** ([Schmidt, Zey & Margot 2021](https://doi.org/10.1007/s10107-019-01460-6), §4.1) — bias each minimum cut towards few arcs, and add the terminal-side cut alongside the root-side one.
- **Nested cuts** ([Koch & Martin 1998](<https://doi.org/10.1002/(SICI)1097-0037(199810)32:3%3C207::AID-NET5%3E3.0.CO;2-O>)) — after a violated cut is found, its arcs are saturated and the max-flow re-run, yielding a second, structurally different violated cut per round. Extra max-flows are spent only on violated terminals. Tune or disable with the `STEINERPY_NESTED_CUTS` environment variable (default `1`, `0` disables; higher values add more cuts per round but grow the model faster than they save re-solve rounds on the HiGHS path).

All three change only how fast the cut loop converges, never the optimum.

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
