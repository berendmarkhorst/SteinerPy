# Algorithmic R&D: Phase 1 report

This page records the evidence for the first staged algorithmic-research branch.
All new strategies are opt-in or experimental. None is enabled by default.

## Scope and sources

Phase 1 implements:

- PCSTP terminal regions and mandatory-vertex lower bounds from Rehfeldt &
  Koch (2020), Definition/equation (18), Proposition 12, and the Voronoi
  construction following Proposition 15; PCD follows Algorithm 1 and Theorem 6;
- vertex elimination and key-path exchange from Uchoa & Werneck (2012);
- the implied-profit shortest-path construction from Rehfeldt & Koch (2023),
  Section 5.1.1, equations (28)--(29), using the paper's cheaper alternative-
  incident-edge bound;
- inactive-cut purging from Schmidt, Zey & Margot (2021), Section 5.1.1.

The wider prize-aware reduction framework of Rehfeldt, Koch & Maher (2019)
informs the next-stage roadmap, but its pseudo-elimination, root detection,
Nearest-Vertex, Short-Links, and degree-k reductions are not claimed here.

## Correctness evidence

The reduction tests use an independent connected-subset/MST brute-force PCSTP
oracle. The checked-in suite covers 100 random instances for each terminal-
region stack, 20 instances for each public PCSTP reduction level, 15 instances
for each public MWCSP reduction level, deterministic theorem witnesses, prize
preservation, terminal protection, and back-mapped objectives. A separate
500-seed × four-stack stress sweep also matched the oracle in all 2,000 cases.

The oracle sweep found a pre-existing PCD correctness defect on current main:
destination prizes were discounted contrary to equation (8), and several tied
alternatives were batch-deleted using Corollary 7. Corollary 7 only promises an
optimum avoiding one equality edge; deleting all such alternatives can remove
every optimum. The batched reducer now excludes both endpoint prizes and uses
Theorem 6's strict inequality. A minimal seed-69 regression is checked in.

Every local-search test checks both terminal connectivity and non-increasing
cost. Deterministic instances exercise each neighborhood, while 30 random
graphs exercise the public portfolio. The cut-pool unit test deletes multiple
row positions, checks compacted HiGHS row identifiers, and reintroduces a
previously purged signature. End-to-end SAP and multi-group forest tests compare
purging against the exact unpurged optimum.

## Benchmark method

`benchmarks/benchmark_phase1.py` was run against detached current main
`fae50e1f7506` and the candidate revisions recorded below. Each configuration
used Python 3.13.4, HiGHS/highspy 1.12.0, NetworkX 3.6.1, one solver thread,
five fixed synthetic seeds, and three fresh-process repetitions per seed.
Objectives and zero-gap certificates were compared before a row was accepted.
Times are medians across 15 runs; brackets are the interquartile range. Peak RSS
is process-isolated but remains an OS-level measurement. These are smoke
microbenchmarks, not a representative SteinLib/PCSPG study.

### PCSTP reductions

Candidate `27124f885750`:

| Stack | Total time, s | Preprocess, s | Edges removed | Nodes removed | Peak rows | Peak RSS, MiB |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.0462 [0.0428, 0.0467] | 0 | 0 | 0 | 105 | 77.4 |
| PCD | 0.0338 [0.0330, 0.0392] | 0.0073 | 20 | 0 | 87 | 76.9 |
| PCD + terminal regions | 0.0418 [0.0392, 0.0433] | 0.0143 | 22 | 0 | 82 | 76.8 |
| PCD + terminal regions + nodes | 0.0422 [0.0388, 0.0429] | 0.0146 | 22 | 0 [0, 1] | 82 | 76.9 |

All 60 candidate runs returned the same proven optimum. Current main's PCD
result disagreed with no-reduction on two of five seeds (objectives 61 vs 58
and 82 vs 66), independently confirming the correctness bug above; its PCD
timing is therefore invalid as a performance result. Terminal regions removed
two additional median edges, but their overhead made them slower than PCD alone
on this small suite. Node deletion had no median effect. Recommendation: retain
the new levels as experimental and off by default.

### Primal portfolio

Candidate `50961193b015`:

| Configuration | Total time, s | Heuristic time, s | Median primal gain | Median fixed variables | LB = UB exits |
|---|---:|---:|---:|---:|---:|
| dual-ascent baseline | 0.0775 [0.0748, 0.0827] | 0 | 0 | 74 | 3/15 |
| vertex/key-path local search | 0.0977 [0.0876, 0.1130] | 0.0197 | 2 | 124 | 3/15 |
| implied profit | 0.0858 [0.0824, 0.1025] | 0.0090 | 2 | 102 | 3/15 |
| combined portfolio | 0.1493 [0.1138, 0.1616] | 0.0553 | 2 | 124 | 3/15 |

All 75 main/candidate runs returned identical proven objectives. Both new
heuristics improved the primal on four of five seeds and enabled more
reduced-cost fixing, but neither reduced end-to-end time here; the combined
portfolio duplicated local-search quality at substantially higher overhead.
Recommendation: retain both switches as opt-in, do not add elite-pool
recombination yet, and seek a size/quality activation threshold on SteinLib.

### Cut purging

Candidate `27124f885750`:

| Age | Total time, s | Cuts purged | Peak rows | Peak RSS, MiB |
|---|---:|---:|---:|---:|
| off | 0.1615 [0.1145, 0.2000] | 0 | 521 | 85.1 |
| 3 | 0.1832 [0.1574, 0.2082] | 7 | 513 | 89.5 |
| 5 | 0.1795 [0.1251, 0.1807] | 3 | 520 | 85.9 |
| 10 | 0.1329 [0.1156, 0.1623] | 0 [0, 1] | 521 | 81.6 |

All 75 main/candidate runs returned identical proven objectives. Ages 3 and 5
removed rows but were slower; age 10 usually removed nothing, so its apparent
timing advantage cannot be attributed to purging. A separate age-1 stress case
churned cuts and exhausted a 60-second limit that the unpurged solve met.
No purged cut was reintroduced in this small sweep, although the unit test
exercises that path. Recommendation: keep `STEINERPY_CUT_PURGE_AGE=0` as the
default and expose the policy only for continued experiments.

## Limitations and deferred work

- Prize-bearing nodes are never physically removed. Proposition 12 candidates
  are reported as protected because the legacy graph representation lacks a
  constant objective-offset/backmapping channel for a deleted prize.
- The terminal-region construction is the deterministic Voronoi baseline; the
  paper's region-improvement local search is deferred.
- Edge bounds use the exact but comparatively expensive equivalent subdivision
  construction. A specialized formula is a worthwhile follow-up only after
  larger benchmarks justify it.
- Vertex insertion, key-vertex elimination, elite pools, and recombination are
  deferred because the first two neighborhoods already add net overhead on the
  smoke suite.
- Cut purging is implemented only for the iterative HiGHS loops. Gurobi manages
  callback cuts through its own branch-and-cut machinery.
- Phase 2 and Phase 3 items remain separate future review units: implied
  bottleneck distance, wider PCSTP reductions, direct Group Steiner, the
  rectilinear full-component feasibility study, low-treewidth DP, and new
  approximate/dynamic modes.

The raw CSVs are intentionally generated artifacts and are not committed;
rerun the documented commands to reproduce them. No change in this phase is a
candidate for default activation on the present evidence.

## Validation record

- Python 3.13.4: 1,080 passed, 26 skipped, 20 expected-failure tests passed
  unexpectedly; 91% line coverage.
- Python 3.12.12 in an isolated environment: 1,069 passed, 37 skipped, 20
  expected-failure tests passed unexpectedly; 86% line coverage. The skip
  difference comes from optional solver paths.
- Gurobi 13.0.2 smoke comparison: the combined primal portfolio and the full
  PC reduction stack matched HiGHS objectives and both reported zero gap.
- Sphinx 9.0.4 strict (`-W`) build: passed under the project's documented
  Python 3.12 docs environment.
- Black and flake8: all new standalone modules, tests, and benchmark files pass
  Black; scoped flake8 passes with `E501`, `E203`, and `W503` ignored to match
  Black's 88-column/operator formatting. The repository-wide checks remain red
  on pre-existing formatting debt: current main is not Black-clean and has no
  flake8 configuration aligning its 79-column default with Black.
- Mypy: the two new algorithm modules pass with missing third-party stubs
  ignored. The repository-wide run remains red on missing NetworkX/SciPy/HiGHS
  stubs and legacy annotations (65 errors with missing imports ignored on the
  candidate versus 69 on current main; 102 errors in the unfiltered run).
