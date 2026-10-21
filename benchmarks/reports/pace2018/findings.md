# Findings and next performance work

Read the [generated report](README.md) for the current completion status and
the [per-instance CSV](results.csv) for failures as well as successes. Full-sweep
conclusions should wait until all 1,797 requested runs have finished.

The completed nine-instance comparison already identifies useful next steps:

1. **Improve the shared composition formulation.** In the 90-second sample,
   specialized HiGHS proves six optima, while generic composition proves one
   and returns three additional feasible solutions. The original wata&sigma
   program proves all nine. Start with profiling model size, construction and
   connectivity enforcement. Evaluate feature-preserving reductions and lazy
   connectivity cuts without losing support for node costs, rewards, direction,
   degree limits, terminal leaves, hops or resource budgets.
2. **Keep and strengthen specialized exact paths.** The participant's pruned
   dynamic program and Track 2 decomposition-aware solver give concrete
   comparison targets. SteinerPy currently ignores supplied tree decompositions.
   Profile before choosing between better pruning, different formulations and
   native implementations. Nine instances do not establish a general speedup.
3. **Clean and validate returned incumbents.** The baseline Gurobi result for
   Track 1 instance 098 contains a terminal-spanning tree of cost 1,500,435 plus
   a disconnected four-edge cycle of cost 16. Its reported cost is 1,500,451;
   the published optimum is 1,500,422. The checker rejects this result instead
   of crediting it as a valid tree. A follow-up should remove unnecessary
   components/cycles, recompute original cost and preserve a valid bound/gap.
   Also distinguish a solver's small nonzero stopping gap from proven optimality.
4. **Return checked incumbents before the wall deadline.** The supervisor can
   stop a process while it is still constructing a model, preprocessing or
   solving. Such a run currently contributes no solution to the score, even
   when a solver has found one internally. Cooperative deadlines and a checked
   incumbent fallback would make the package more useful on large instances.

A separate heuristic correctness defect was fixed before the measured baseline:
post-reduction cleanup used reduced terminals against the original graph and
could drop required original terminals. The benchmark revision includes the fix
and its regression tests. All configurations in the full sweep use that same
frozen revision; the rejected Gurobi output above has not been silently repaired
in the measured results.

Published PACE rankings, the full Snellius sweep and the nine-instance sample
have different scopes. Keep them separate when interpreting solution quality,
proof counts and elapsed time.
