# Findings and next performance work

All 1,797 requested runs finished on 21 October 2026. The final reporting job
completed at 20:20 Oslo time (18:20 UTC). The [generated report](README.md) and
[per-instance CSV](results.csv) include every success and failure from the same
frozen revision, `7d4d8b8`. Each configuration attempted all 599 available cases
with one thread, a 1,800-second wall deadline and an 8 GiB address-space limit.
The Gurobi guard allowed at most 24 shared WLS reservations, including cooldown,
below the license's 30-session limit.

| Configuration | Valid trees / 599 | Proven or reference optima | Wall timeouts | Memory failures | No valid incumbent returned | Validation rejections |
|---|---:|---:|---:|---:|---:|---:|
| Gurobi | 410 | 296 | 133 | 5 | 46 | 5 |
| HiGHS | 273 | 269 | 190 | 42 | 94 | 0 |
| Heuristic | 530 | 20 | 2 | 67 | 0 | 0 |

Optima are a subset of valid trees. The heuristic returns the most valid trees,
but usually does not match a known optimum. Large Track 3 cases remain the main
weakness: Gurobi, HiGHS and heuristic mode return 44, 12 and 132 valid trees out
of 199 respectively. Of the heuristic's 67 memory failures, 66 are in Track 3.
All 325 hard wall timeouts occur during the public solve call. A failure to
return a valid incumbent at the solver deadline is not itself a correctness bug.

The exact-track private-set optimum counts are below the historical leaders:
Gurobi reaches 72/100 in Track 1 and 57/100 in Track 2, versus 95/100 and 92/100
for the respective 2018 winners. Hardware and resource limits differ, so these
are context rather than a competition ranking. Track 3 scores also use a
different available instance set and potentially updated reference bounds.

## Validation failures need two different investigations

The baseline Gurobi result for Track 1 instance 098 contains a terminal-spanning
tree of cost 1,500,435 plus a disconnected four-edge cycle of cost 16. Its reported
cost is 1,500,451; the published optimum is 1,500,422. This is a confirmed
structural defect in the returned solution.

The other four rejections appear numerical: reported objectives are just below
integer reference values. They are not evidence of beating a proven optimum.

| Track / instance | Reported objective | Published optimum | Rejection |
|---|---:|---:|---|
| 2 / 088 | 164149.99954 | 164150 | Objective differs from original edge costs |
| 2 / 138 | 231028455.99675694 | 231028456 | Objective below reference lower bound |
| 3 / 013 | 5615.999924575609 | 5616 | Objective differs from original edge costs |
| 3 / 072 | 42767.9999915956 | 42768 | Objective below reference lower bound |

The measured runner did not retain rejected edge certificates. The separate
diagnostic for instance 098 establishes its structural defect, but the four
apparently numerical cases need targeted reproduction with certificates before
they can be accepted. All five remain failures in the published baseline.

## Recommended order of follow-up work

1. **Make returned solutions consistent and checkable.** Reproduce the five
   rejections, retain their edge certificates and reconcile reported objectives
   with costs recomputed on original edges. Remove unnecessary components and
   cycles where the problem's constraints permit it. Preserve valid lower bounds
   and recompute gaps; a small nonzero stopping gap is not a proof of optimality.
   Keep structural checks while investigating numerical tolerances.
2. **Return useful solutions on large cases.** Profile the 114 memory failures
   and model construction before choosing changes. Add cooperative deadlines
   and a checked incumbent fallback so a time limit can still yield a solution.
   A hard timeout currently scores zero even if the solver found an incumbent
   internally. The recorded solve phase includes model construction and does not
   by itself identify which internal algorithm exhausted memory.
3. **Continue efficient, compatible dispatch in [issue #92](https://github.com/berendmarkhorst/SteinerPy/issues/92).**
   Plain `Problem(...)` already selects a specialized path. In the separate
   nine-instance, 90-second sample, specialized HiGHS finds six optima, while
   forced generic composition finds one and three additional feasible solutions;
   the original wata&sigma program finds all nine. This compares formulations,
   not performance before and after modularization. Keep all requested features
   when selecting faster methods, and profile composition model size and
   connectivity enforcement. Track 2's supplied tree decompositions are currently
   unused; decomposition-aware solving and stronger few-terminal methods are
   concrete candidates to evaluate. Nine cases do not establish a general speedup.

Rerun the affected cases after each fix, then use the full sweep to measure the
improvement without changing the frozen baseline. Solver changes are follow-up
work; collecting these final results did not change the measured solver.

A separate heuristic correctness defect was fixed before the measured baseline:
post-reduction cleanup used reduced terminals against the original graph and
could drop required original terminals. The benchmark revision includes the fix
and its regression tests. All configurations in the full sweep use that same
frozen revision; the rejected Gurobi outputs above have not been silently repaired
in the measured results.

Published PACE rankings, the full Snellius sweep and the nine-instance sample
have different scopes. Keep them separate when interpreting solution quality,
proof counts and elapsed time.
