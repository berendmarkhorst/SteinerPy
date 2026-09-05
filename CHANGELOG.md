# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Experimental terminal-regions PCSTP reductions** (off by default):
  `pc_reduce="pcd+trd"` adds Rehfeldt & Koch's (2020) terminal-regions
  decomposition and Proposition 12 lower bound for bound-based edge deletion;
  `"pcd+trd+nodes"` also deletes certified zero-prize nodes. Edge bounds use
  the equivalent zero-prize subdivision-vertex construction. Prize-bearing
  nodes and user terminals are retained, all deletions require strict
  `LB > UB`, and `problem.pc_reduction_stats` exposes phase timings and removal
  counts. `True`/`"pcd"` remain backward-compatible PCD-only aliases. Small
  exact PCSTP oracle tests cover all four stacks.
- **Experimental HiGHS cut aging/purging** (off by default): set
  `STEINERPY_CUT_PURGE_AGE` to a positive number to delete generated directed
  cuts after that many consecutively slack re-solves. Active signatures prevent
  duplicates; purged inequalities may be rediscovered; batch deletion updates
  every surviving HiGHS row index. Structural and dual-ascent seed rows remain
  permanent. `problem.cut_stats` reports active/peak rows, purged/reintroduced
  cuts, separation rounds, and separate LP/MIP re-solve times. The policy is
  integrated into both forest and SAP HiGHS loops and composes with LP-first and
  nested cuts.
- **Fresh-process Phase-1 benchmark harness**:
  `benchmarks/benchmark_phase1.py` compares arbitrary source checkouts with
  fixed seeds, repeats, single-thread controls, dependency/commit metadata,
  objective-and-gap validation, phase counters, and isolated peak RSS for the
  PC reductions, primal portfolio, and cut-purge age sweep. It also accepts
  standard SteinLib and prize-bearing PCSPG `.stp` instances, recording known
  B-series optima and rejecting certified disagreements.
- **Experimental primal-heuristic portfolio** (off by default):
  `primal_local_search=True` applies the vertex-elimination and key-path-exchange
  neighborhoods of Uchoa & Werneck (2012), while `implied_profit=True` adds the
  implied-profit shortest-path heuristic of Rehfeldt & Koch (2023, Section
  5.1.1, equations 28–29). The options are available at construction or per
  `get_solution()` call for undirected single-group trees. Candidates from dual
  ascent, Kou, Mehlhorn, and implied profit are feasibility-checked, MST-pruned,
  and compared; local moves are accepted only on a strict objective decrease.
  The best upper bound feeds exact-solver cutoff, reduced-cost fixing,
  `LB == UB` early exit, and MIP warm start without changing the dual lower
  bound or certified-gap semantics. Separate runtime and move counters are
  exposed through `problem.heuristic_stats`.
- **`enumeration_safe=True` preprocessing mode for `get_optimal_solutions()`**
  (steinerpy#43): `get_optimal_solutions()` previously required
  `preprocess=False`, since the default reduction pipeline can silently
  collapse tied-cost alternatives before the ILP ever runs — degree-2
  contraction against a same-cost parallel edge, node replacement, and the
  adjacent-terminal/Nearest-Vertex/Short-Links terminal-contraction tests
  each pick one witness among possibly several tied ones and structurally
  erase the rest. Constructing with `enumeration_safe=True` now restricts
  preprocessing to reductions proven to preserve the **complete set** of
  optima: special distance, long-edge and bound-based deletions, and
  non-terminal degree-1 removal, are already exact (strict inequalities) and
  unaffected; degree-2 contraction skips a node instead of contracting it
  when the contracted path ties an existing parallel edge; node replacement
  is disabled outright; and of the terminal-contraction tests, only the
  forced degree-1 case still fires. `get_optimal_solutions()` now accepts
  `preprocess=True` when `enumeration_safe=True` and back-maps solutions to
  the original graph accordingly. Requires strictly positive edge weights
  (`ValueError` otherwise, since the preservation argument assumes positive
  costs) and is not supported together with `max_degree`/`hop_limit`
  (`NotImplementedError`), since the structural degree-1/degree-2 fixpoint
  runs regardless of those modifiers and is not degree- or hop-aware. The
  dual-ascent bound-based reduction (`da_reduce=True`) now also forwards
  `enumeration_safe` into its structural cascade, so it composes safely with
  the new mode.
- **`DirectedGroupSteinerProblem`**: the directed (rooted-arborescence)
  variant of `GroupSteinerProblem` — minimum-cost directed tree from a
  `root` reaching at least one vertex of each group. Uses the same
  super-terminal transformation as the undirected case, but with directed
  zero-cost arcs running from each group's real vertices into its
  super-terminal, then solves it with the directed-cut model shared with
  `DirectedSteinerProblem`.
- **Nearest-Vertex (NV), Short-Links (SL) and bound-based (BND) reductions**
  (Polzin & Vahdati Daneshmand 1998, Obs. 3.2/3.3/3.5/3.6; Steiner **tree**
  only): NV contracts a terminal's cheapest incident edge when
  `c(e') + d(v', tj) <= c(e'')` for another terminal `tj` (certified via the
  two-label Voronoi diagram); SL contracts a Voronoi region's cheapest
  boundary link when every other link costs at least its full link length —
  promoting the merged endpoint to a **new terminal** when needed
  (`ReductionTracker.added_terminals`); BND deletes nodes/edges whose
  Voronoi-radius lower bound `d1 + d2 + sum of smallest (s-2) radii` exceeds a
  Mehlhorn-SPH upper bound (`bound_based=` flag, grouped with `heavy`).
  NV/SL join the `contract_terminals` cascade; contraction sub-rounds run on a
  fresh Voronoi diagram and the diagram is rebuilt before the deletion tests
  (whose bounds must remain lower bounds). Validated by randomized sweeps
  against unreduced solves, including zero-weight-edge (group-Steiner-style)
  instances.
- **Terminal contraction (fixed-edge) reductions** (on by default, opt out with
  `contract_terminals=False`; Steiner **tree** only): two inclusion tests that
  *fix* an edge into the solution and merge its endpoints — **degree-1
  terminals** (the sole incident edge is in every feasible solution) and
  **adjacent terminals** whose connecting edge is cheapest at one endpoint (in
  at least one optimal solution, by the classic cut-exchange argument). The
  new fixed-edge channel on `ReductionTracker` (`fixed_cost`, `fixed_edges`,
  `terminal_merges`) moves the fixed cost out of the reduced model (every
  reporting site adds it back), re-homes the merged node's edges with full
  back-mapping support, and remaps the terminal groups to the surviving
  representatives. Contraction shrinks the terminal set, which strengthens the
  Special-Distance/replacement tests and can solve instances outright during
  preprocessing (the solver then returns immediately). Validated against
  `preprocess=False` solves and a brute-force oracle on hundreds of random
  instances. The full NSV/SL terminal-to-non-terminal contractions remain
  future work; the infrastructure now supports them.
- **Few-terminal exact dynamic program** (Dreyfus & Wagner 1971, in the
  Erickson–Monma–Veinott formulation): plain undirected single-group Steiner
  tree instances with at most `STEINERPY_DW_MAX_TERMINALS` terminals (default
  10, `0` disables) are now solved by an `O(3^k·n + 2^k·(m + n log n))` dynamic
  program instead of the ILP — the reductions + DP recipe of the PACE 2018
  winning solvers. Vectorised merge steps (numpy) and virtual-source scipy
  Dijkstra grow steps; auto-selected after preprocessing, exactness-preserving
  (validated against a brute-force oracle and the ILP), 4–30x faster than the
  accelerated ILP on benchmarked instances. Applies transparently to the
  transformed group-Steiner, terminal-leaf, and rectilinear variants.
- **LP-first cut loop (HiGHS)**: before the integer cut loop starts, the
  directed Steiner cuts are separated on the **LP relaxation** — each round is
  a cheap LP re-solve instead of a full branch-and-bound run, and the
  accumulated root cuts strengthen every subsequent MIP solve (the classic
  root-separation scheme of branch-and-cut Steiner codes, Koch & Martin 1998).
  Applies to the iterative HiGHS path (`run_model`, `solve_sap_highs`); the
  Gurobi path already separates fractional points in its callback.  Configure
  with `STEINERPY_LP_CUT_ROUNDS` (default 50, `0` disables).  Speedups of
  5–10x on seeded tree instances and ~1.6x on a forest instance, with
  identical optima.  A dual-ascent MIP warm start set before `run_model` is
  re-applied after the LP phase via the new `reapply_start` callback.
- **Nested cuts in the directed-cut separation** (Koch & Martin 1998): when a
  terminal's minimum cut is violated, the cut's arcs are saturated (capacity
  raised to 1) and the max-flow re-run, emitting up to `STEINERPY_NESTED_CUTS`
  (default 1, `0` disables) further violated cuts per separation round. Extra
  max-flows are spent only on violated terminals, and capacities are only ever
  raised, so every nested cut is guaranteed violated at the current solution.
  Joins the existing creep-flow and back-cut accelerators; scipy path only.

### Changed
- Near-integral connectivity candidates use capacity-checked reachability
  cuts before max-flow. Identical certificates within a separation round are
  emitted once; fractional and uncertified demands retain min-cut separation.
  LP separation retains minimum cuts even at integral relaxations.
  Denser graphs use SciPy bounded Dijkstra for long-edge reduction when the
  settled-node work cap covers the entire graph; other graphs keep the Python
  search and its work cap.
- Heavy reductions use a lazy tree-path maximum index instead of storing all
  terminal-pair bottleneck distances: construction and storage are now
  `O(|T| log |T|)`, with logarithmic queries. The final unchanged DA reduction
  pass can hand its result to the solve once, after validating graph and
  formulation inputs, avoiding duplicate dual-ascent work.
- Connectivity cut separation now certifies sufficient-capacity paths before
  running per-terminal max-flow, skips residual traversal for satisfied cuts,
  and extracts cut arcs with NumPy arrays. Separation defaults to one worker;
  `STEINERPY_SEP_THREADS` still overrides it independently of solver threads.
- **Faster Dreyfus-Wagner grow steps**: the few-terminal dynamic program now
  builds the fixed graph-plus-virtual-source CSR structure once and updates
  only the virtual-source weights for each subset, avoiding repeated sparse
  matrix construction and its temporary coordinate arrays.
- **Optimal-solution enumeration now has explicit inclusion-minimal
  semantics** (issue #45): redundant zero-cost branches and cycles are removed
  without treating them as distinct Steiner trees, consistently for branches
  attached to roots and internal nodes. Distinct tied-optimal trees whose
  zero-cost edges are necessary for connectivity remain enumerable.
- **Flow variables are now continuous**: the flow-based models (prize-collecting
  penalty ILP, budget-constrained, MWCSPB) declared every per-terminal arc-flow
  variable as a binary integer — O(|T|·|A|) integer columns. Flow integrality
  follows from the integral arc/connection variables (each block is a unit s–t
  flow with integral capacities, and flow never enters the objective), so the
  variables are now continuous in `[0, 1]`. Same optimum, far smaller MIP.
- **Model construction is now O(|A|) instead of O(|V|·|A|) per group**: the
  HiGHS and Gurobi builders scanned the full arc list per node (indegree,
  flow-conservation, degree, and root-linking constraints, plus the per-call
  terminal-group lookup in `demand_and_supply_directed`). Incoming/outgoing arc
  adjacency and the terminal→group map are now precomputed once per build.

### Fixed
- NetworkX cut separation completes the maximum flow before extracting a
  source-side cut; an intermediate preflow can leave excess at internal nodes
  and produce a source partition that is not a minimum cut.
- **Prize-constrained-distance deletion could change the PCSTP optimum** in two
  cases: it discounted the destination endpoint's prize even though equation
  (8) excludes both endpoints, and it batch-deleted tied alternatives under
  Corollary 7 even though that corollary only guarantees an optimum avoiding
  each edge individually. The implementation now excludes the destination
  prize and uses Theorem 6's strict inequality for collect-then-apply batches.
  A minimal tied-alternative regression and randomized brute-force PCSTP oracle
  sweeps cover the failure.
- **Zero-edge Steiner forests with singleton terminal groups**: the HiGHS and
  Gurobi fast paths determined feasibility from the graph's total node count,
  incorrectly rejecting an empty forest such as groups `[['A'], ['B']]` on two
  isolated nodes. They now test each terminal group independently for at most
  one distinct terminal, while a group such as `[['A', 'B']]` remains
  infeasible.
- **`get_optimal_solutions()` correctness fixes** (PR #42 review): the no-good
  cuts identified edges with `frozenset(e)`, discarding arc orientation, so an
  antiparallel arc pair `(u, v)`/`(v, u)` in a directed graph collapsed to one
  key and a cut could fail to exclude the previous solution — directed arcs
  now keep their ordered `tuple` identity, only undirected edges are
  `frozenset`-normalised. The directed-cut model's `x` <-> `y1` link
  (Constraint 3) is now an equality instead of `y1 <= x`, so a zero-cost edge
  unused by the tree can no longer be toggled on in `x` for free and counted
  as a spurious extra "distinct" solution. The enumeration loop now inspects
  the solver status (`run_model`/`run_model_gurobi` return an added `status`
  of `"optimal"`/`"infeasible"`/`"incomplete"`) instead of only checking
  whether the objective is finite: a probe that times out before proving
  optimality or infeasibility (e.g. `time_limit=0`) used to either be
  silently treated as proof enumeration is exhausted, or, when the model was
  never actually solved, fall through to a bogus zero-cost empty "solution"
  repeated across iterations and trip an internal duplicate-solution
  assertion. Such a probe now stops enumeration with `exhausted=False`
  instead of raising or misreporting. `limit < 0` now raises `ValueError`
  instead of silently returning an empty, non-exhausted pool.
- **`get_optimal_solutions()` correctness fixes, round 2** (PR #42 review):
  the zero-cost-edge fix to Constraint 3 (`x` <-> `y1` equality) was only
  applied to the HiGHS builder; `build_model_gurobi()` still used `<=`, so an
  unused zero-cost edge could still be toggled on in `x` and double-counted
  as a distinct solution when `solver="gurobi"` — the Gurobi builder now
  mirrors the equality. Also, `run_model()`/`run_model_gurobi()` are public
  (re-exported from `steinerpy`), and adding the `status` return value as an
  unconditional 5th tuple element broke existing 4-value callers; both now
  take a `return_status: bool = False` parameter and default back to their
  original 4-value `(gap, runtime, objective, selected_edges)` signature,
  with `status` only appended when `return_status=True`.
- **Zero-edge reductions crashed instead of reporting infeasibility, on both
  the plain and budget-constrained solve paths**: when graph reduction left an
  empty `self.graph` while a group still held >= 2 distinct terminals, the
  empty edge set made `sum(x[e] * ... for e in self.edges)` collapse to the
  plain Python value `0` — `0 <= budget` then evaluates eagerly to a Python
  `bool`/`int` rather than building a HiGHS expression, and highspy's
  `setObjective`/`addConstr` unconditionally read `expr.bounds`, raising a
  bare `AttributeError` deep inside the solver instead of a clean error.
  `get_solution()` now raises a `RuntimeError` up front on the plain path; the
  budget-constrained path (`build_budget_model`) instead skips the now-vacuous
  budget constraint, since an empty edge set is not actually infeasible there
  (all non-root terminals are simply penalised). The trivial-instance
  fast-path also now de-duplicates each group (`len(set(g))` instead of
  `len(g)`) so a group with a repeated terminal (e.g. `['A', 'A']`) is
  correctly recognised as already-solved instead of falsely tripping the new
  infeasibility guard.
- **Directed-cut model inflated the objective when a real 2-cycle existed
  between two nodes on the optimal path** (HiGHS and Gurobi builders):
  Constraint 3 (`y1` -> `x`) bundled arc `(u, v)` with its reverse `(v, u)`
  into a single edge-cost variable `x[(u, v)]` whenever `(v, u)` appeared as
  an arc — the intended behaviour for the reverse arcs `objects.py` mirrors
  onto every *undirected* edge, which share one edge-cost variable. On a
  genuine `nx.DiGraph` with an explicit two-way arc pair, each direction is
  its own edge with its own `x` variable, so bundling forced the unused
  direction's `x` to 1 whenever the used direction was selected, inflating
  the reported objective while `gap` was still (incorrectly) certified as
  `0.0`. Fixed by only bundling when the reverse arc has no `x` variable of
  its own (i.e. it is the synthesized undirected companion, not a real
  independent edge). Affects `DirectedSteinerProblem` and anything built on
  the same directed-cut kernel (e.g. `HopConstrainedSteinerProblem`).
- **Mixed node-type crash in the reduction Dijkstras**: the long-edge test and
  the terminal-Voronoi construction pushed `(distance, node)` pairs onto their
  heaps, so a distance tie compared node labels — a `TypeError` when labels mix
  types (e.g. the group-Steiner transform's string super-terminals next to int
  nodes). Heap entries now carry a sequence tiebreaker.
- **HiGHS variable typing in the penalty/budget/MWCSPB models**:
  `addVariable(0, 1, hp.HighsVarType.kInteger)` passed the integrality enum as
  the *objective coefficient* (the third positional argument is `obj`, not
  `type`), so the prize-collecting node/penalty variables, the budget
  penalty/connection variables, and the MWCSPB node variables were silently
  created as continuous columns. They are implied-integral at any optimum with
  integral arc variables (which is why results were still correct), but they
  are now declared integer explicitly via the `type=` keyword.
- **Degree-k node replacement / pseudo-elimination** (`replace_nodes=`, part of
  `heavy`): the Rehfeldt & Koch (2023, Prop. 4) test eliminates a non-terminal
  that provably has degree ≤ 2 in at least one minimum Steiner tree (checked
  against the sorted Mehlhorn terminal-MST weights, one comparison per degree
  class), bridging each neighbour pair with the two-edge path cost. Replacement
  edges are pre-filtered by the Special Distance bound, merged into cheaper
  parallels, capped at degree 4 with a growth guard, and recorded exactly like
  degree-2 contractions so the existing solution back-mapping applies
  (back-mapped solutions are additionally de-duplicated). Steiner **tree** only
  (skipped for multiple groups).
- **Two-label Special Distance test**: the SD bound now routes through the two
  nearest terminals of *each* endpoint (two-label multi-source Dijkstra,
  `_voronoi2`), a strict strengthening of the classic nearest-terminal bound at
  about twice the (linearithmic) preprocessing cost.
- **Bound-based node elimination in dual ascent**: reduced-cost fixing now also
  fixes non-terminal *nodes* with `lb + d̃(root→v) + d̃(v→T) > ub` (Ljubić 2021
  §4; Polzin 2003) — root-agnostic, valid for tree and forest — expands them
  into incident arc/edge variable fixes for the ILP, and `da_reduce=True`
  deletes them from the graph before cascading the structural reductions.

- **Prize-collecting / MWCSP acceleration via SAP transformation** (opt-in
  `pc_transform=True`, `exact=False`, `pc_reduce=True` on
  `PrizeCollectingProblem` / `MaxWeightConnectedSubgraph`). Adapts the
  *change-the-problem-class* approach of Rehfeldt & Koch (MWCSP 2019; PCSTP
  ZIB 20-11, 2020): the classic forgo-prize PCSTP (and the MWCSP, via the
  `c(e):=-w0`, `p(v):=w(v)-w0` reduction) is transformed to a rooted Steiner
  arborescence (Transformation 2, with cost-shifting on non-proper potential
  terminals) and solved with the existing dual-ascent lower bound, reduced-cost
  variable fixing, Steiner-cut seeding, primal warm-start, and a proven-optimal
  early-exit, on a dedicated arc-based directed-cut model.
  - `pc_transform=True` — **exact** solve through the transformation (often
    proves optimality without an ILP).
  - `exact=False` — **heuristic-only** mode returning the dual-ascent primal with
    a *valid* optimality gap (`0.0` ⇒ provably optimal).
  - `pc_reduce=True` — the **prize-constrained distance** (PCD) edge-deletion test
    (PCSTP report Thm 6 / Algorithm 1): a sound, **prize-safe** (edge-only)
    reduction that shrinks the graph for both the new path and the penalty ILP.
  All three are **off by default** (the penalty/Big-M flow ILP remains the
  default solver) and gated to the classic forgo-prize PCSTP / MWCSP — a
  `penalty_budget`, multiple terminal groups, or a non-zero `penalty_cost` raises
  a clear `NotImplementedError`. Validated against an independent brute-force
  PCSTP/MWCSP oracle over hundreds of random instances.
- **Heavy graph reductions** (opt-in `heavy=True`, or granular
  `special_distance=` / `long_edge=`): two classic alternative-based
  edge-deletion reduction tests that shrink the graph before the solve.
  *Special Distance* (bottleneck Steiner distance; Rehfeldt & Koch 2023, Thm 1)
  deletes edges with a cheaper terminal-hopping detour through the terminal
  distance network (Steiner **tree** only). *Long-edge / alternative-path* deletes
  any edge with a strictly cheaper detour in `G \ e` (Steiner **tree** and
  **forest**). Both only delete edges provably in no optimal solution, cascade
  the degree-1/degree-2 reductions to a fixpoint, **preserve the optimum** (with
  a connectivity guard), and compose with `da_reduce=True` / `dual_ascent=True`.
  Require `preprocess=True`, undirected graphs, and no `budget`/`max_degree`
  modifier. Validated with a brute-force exact solver over thousands of random
  tree and forest instances. Off by default.
- **Heuristic-only mode** (`get_solution(exact=False)`): returns the dual-ascent
  primal directly with **no ILP** — much faster (networkx-`steiner_tree` speed
  class) and, unlike a pure heuristic, the returned `Solution.gap` is a *valid*
  optimality gap (`0.0` ⇒ provably optimal; a positive gap bounds how far the
  tree could be from the optimum). Supported for plain Steiner tree/forest and
  directed problems; raises `NotImplementedError` for budget/degree-constrained
  variants. Default stays `exact=True` (solve to optimality).

### Changed
- **Heavy reductions are now ON by default** (`heavy=True`): Special Distance,
  long-edge, and node replacement run automatically for undirected problems
  without `budget`/`max_degree`/`hop_limit` modifiers (prize-collecting and
  directed problems are unaffected — they never preprocess). All tests are
  provably optimum-value-preserving; among several equal-cost optima a
  different one may now be returned. Disable with `heavy=False` or the granular
  `special_distance=` / `long_edge=` / `replace_nodes=` flags.
- **Worklist-driven preprocessing**: `preprocess_graph` now runs the
  degree-1/degree-2 reductions in place off a change-driven worklist (no more
  per-pass graph copies and full node rescans), builds one Voronoi diagram /
  terminal MST per heavy round shared by the SD and replacement tests, applies
  deletions with an undo log instead of full graph snapshots, and stops the
  heavy sub-passes once a round removes less than 1% of the edges.
- `benchmarks/run_benchmarks.py` gained `--reduce {none,heavy,heavy+da}` and
  reports preprocessing time and node/edge reduction percentages per instance.
- **Faster heavy reductions** (no API or result change beyond nearest-terminal
  tie-breaking): the Special Distance test now builds the terminal distance
  network with a single multi-source Dijkstra (terminal Voronoi diagram) plus
  Mehlhorn's (1988) Voronoi-boundary MST instead of one Dijkstra *per terminal*
  and an O(|T|^2) complete-graph MST — `O(m + n log n)` overall. The long-edge
  test now runs one bounded Dijkstra *per vertex* (Rehfeldt & Koch 2023, Sec. 2.3)
  instead of one *per edge*. Measured speedups grow with size/terminal count:
  ~5-10x on the Special Distance test and ~3-5x on the long-edge test for
  graphs with hundreds of nodes and tens-to-hundreds of terminals. Reduction
  power is unchanged (verified: identical long-edge deletions; combined
  edge-deletion within tie-breaking noise) and the optimum is still preserved
  (brute-force verified over thousands of random instances).
- **Leaner dual-ascent accelerator** (no API or result change): (1) the
  multi-root pass now stops as soon as a root closes the bound (`LB==UB`),
  instead of always running all 8 candidate roots — output-identical, but it
  removes the ~6-9× wasted work on instances that early-exit without an ILP;
  (2) the Wong ascent inner loop was rewritten to maintain the saturation graph
  incrementally (saturated-arc adjacency + hand-rolled BFS) rather than
  rebuilding a `networkx` graph and recomputing `ancestors`/`descendants` every
  iteration (Duin/Pajor-style efficient implementation), roughly 2-4× faster per
  ascent. Both changes are exact — same lower bound, reduced costs, cuts, and
  optimum.
- **Dual-ascent accelerator** (opt-in `dual_ascent=True`): a Wong (1984)
  dual-ascent procedure computes a lower bound, a primal heuristic, and reduced
  costs, then applies reduced-cost variable fixing (Leitner et al. 2018) to
  shrink the ILP before solving — and solves directly (no ILP) when the bound is
  tight. Supported for Steiner tree, forest (multi-root) and directed problems;
  off by default and returns the same optimum as the baseline. New module
  `steinerpy.dual_ascent`.
- **Cut initialization** for the dual-ascent accelerator: the Steiner cuts found
  during dual ascent are now reused to warm-start the ILP cut loop (seeded as
  initial constraints) instead of being rediscovered one re-solve at a time, and
  the primal value is supplied as an objective cutoff (HiGHS and Gurobi). This
  collapses cut-loop rounds even on instances where the bound is too loose for
  reduced-cost fixing to help, and never changes the optimum (the seeded cuts are
  valid Steiner cuts). Active automatically whenever `dual_ascent=True`.
- **Multi-start primal** for the dual-ascent accelerator: the primal heuristic
  (and dual ascent) now run from several candidate roots and keep the cheapest
  feasible upper bound (and the tightest lower bound). Because the lower bound is
  usually already optimal, the tighter upper bound lets many more instances be
  solved entirely by dual ascent with no ILP, and strengthens reduced-cost
  fixing on the rest. The multi-root pass is applied per group for forests too;
  it never changes the optimum. Active automatically whenever `dual_ascent=True`.
- **Dual-ascent graph reduction** (opt-in `da_reduce=True`): a bound-based
  reduction test that deletes edges proven (by the dual-ascent reduced costs) to
  be in no optimal solution and then cascades the existing degree-1/degree-2
  reductions to a fixpoint, shrinking the graph before the solve. Undirected
  problems only, requires `preprocess=True`, and is skipped under a
  `budget`/`max_degree` modifier; the optimum is preserved (guarded by a
  connectivity check) and solutions still map back to the original graph.
- `benchmarks/` harness: SteinLib `.stp` parser, known-optima validation, and an
  HPC-friendly (resumable, parallel) runner comparing baseline vs accelerator.

## [1.0.1] - 2026-06-16

### Fixed
- Backmapping of solutions from preprocessed graphs now expands chains of
  degree-2 contractions recursively, so `SteinerProblem` no longer returns edges
  that don't exist in the original graph ([#20]).
- `PrizeCollectingProblem` (and its subclass `MaxWeightConnectedSubgraph`) no
  longer run graph preprocessing: degree-1/degree-2 reductions discarded
  non-terminal node prizes and corrupted the objective. Preprocessing is forced
  off, and explicitly passing `preprocess=True` now raises a warning ([#19]).
- `steinerpy.__version__` is now read from the installed package metadata, so it
  always matches the released version instead of drifting out of sync.

[#19]: https://github.com/berendmarkhorst/SteinerPy/issues/19
[#20]: https://github.com/berendmarkhorst/SteinerPy/issues/20

## [0.1.3] - 2025-12-18

### Fixed
- Python 3.8 compatibility by replacing union type syntax (`|`) with `typing.Union`
- Updated PyPI badge links to point to the official PyPI project page

## [0.1.2] - 2025-12-18

### Fixed
- Python 3.8 compatibility by replacing modern type annotations (`list[type]`) with `typing.List[Type]`
- Type annotations throughout the codebase now compatible with Python 3.8+
- Pytest configuration updated to use correct package name for coverage

## [0.1.1] - 2025-12-18

### Fixed
- Package structure corrected for proper PyPI installation
- Import statements updated to use `steinerpy` package name

## [0.1.0] - 2025-12-18

### Added
- Initial release of SteinerPy
- `SteinerProblem` class for defining Steiner Tree and Forest problems
- `Solution` class for handling optimization results
- Support for NetworkX graphs with custom edge weights
- HiGHS solver integration for optimization
- Basic test coverage
- Documentation and examples

### Features
- Solve Steiner Tree problems (single terminal group)
- Solve Steiner Forest problems (multiple terminal groups)
- Configurable time limits and logging
- MIT license for open source usage
