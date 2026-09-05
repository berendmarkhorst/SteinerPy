# PR validation against current main

The performance branch has been integrated with main at `7b4142e` (version 1.0.18). Main added terminal contractions, extra reductions, a few-terminal dynamic program, nested cuts, LP-first HiGHS separation, cut aging, and stricter solve-status handling after this branch's original baseline. The earlier reports remain historical measurements; their timings should not be presented as comparisons against current main.

## Integration

- Retain main's nested minimum cuts for fractional and uncertified demands.
- Retain minimum-cut separation in both Gurobi's LP callbacks and HiGHS's LP phase, even at integral relaxations. Use the new connectivity certificates in incumbent/integer separation.
- Preserve deterministic deletion order in the native long-edge reduction, matching main's ordered reduction worklists.
- Forward enumeration-safe preprocessing alongside the validated dual-ascent handoff. Adapt reuse tests to disable terminal contractions and the few-terminal DP so they still exercise the intended ascent path.
- Preserve main's feasibility checks, status handling, and new optional research features.

## Validation

Full integrated suite: **1,140 passed, 26 skipped, 20 xpassed** (92% coverage). The xpassed cases are existing non-strict expected-failure tests. No tests failed. The updated tests also compare native and Python deletion order, in addition to comparing deletion sets against independent shortest paths.

Sphinx HTML documentation builds successfully. The local installed package metadata returns a missing version, so the build passed explicit `-D version=1.0.18 -D release=1.0.18` overrides, matching `pyproject.toml`.

## Current-main comparison

Nine instances × two configurations × two revisions × three repeats: **108 solves**. Every solve matched the known optimum within 1e-6 with zero gap. All constructed Gurobi models finished with optimal status; empty Gurobi fields in the CSV indicate a solve completed without building a Gurobi model. Reduced node and edge counts matched between revisions for every configuration.

| Instance | Main default | PR default | Speedup | Main + DA | PR + DA | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| b18 | 0.017 s | 0.016 s | 1.07× | 0.008 s | 0.009 s | 0.98× |
| c01 | 0.011 s | 0.012 s | 0.95× | 0.014 s | 0.014 s | 0.95× |
| c05 | 0.052 s | 0.022 s | 2.39× | 0.039 s | 0.021 s | 1.85× |
| c13 | 4.279 s | 2.500 s | 1.71× | 4.863 s | 2.492 s | 1.95× |
| d01 | 0.034 s | 0.035 s | 0.95× | 0.045 s | 0.046 s | 0.96× |
| d05 | 0.249 s | 0.169 s | 1.47× | 0.192 s | 0.151 s | 1.27× |
| d08 | 5.360 s | 2.547 s | 2.10× | 9.079 s | 2.700 s | 3.36× |
| d13 | 9.641 s | 3.957 s | 2.44× | 11.064 s | 2.936 s | 3.77× |
| d15 | 1.254 s | 0.405 s | 3.09× | 1.199 s | 0.481 s | 2.49× |

Times are medians of three runs. Small instances show sub-millisecond to a few milliseconds of variation and are not reliably improved. The larger c13/d08/d13/d15 cases improve in both configurations. These changes retain the same formulation, but different valid cuts can change search behavior; no universal improvement is implied.

## Reproduce

Use the same interpreter/dependencies and instance data in a clean checkout of `7b4142e` and this branch. Copy `benchmarks/profiling/profile_solver.py` into the main checkout; the harness imports code from its own checkout. Run three times per revision, alternating revision order:

```sh
python benchmarks/profiling/profile_solver.py --tag pr-1 \
  --instances b18 c01 c05 c13 d01 d05 d08 d13 d15 --limit 20
python -m pytest tests/ -q --cov=steinerpy --cov-report=term-missing
```

The default settings use four Gurobi threads. Each revision keeps its own default separation thread count (main: up to eight; candidate: one). No profiling is enabled during these timings. Runs execute sequentially, with no test or profiling workloads running concurrently. As in earlier reports, timings exclude imports/parsing and include graph copying, construction/preprocessing, solving, and reconstruction. `accelerated` enables dual ascent and its preprocessing reductions; all other solver/reduction defaults are retained, including the few-terminal DP.

Environment: macOS ARM64, Python 3.13.4, Gurobi 13.0.2, SciPy 1.18.0, NetworkX 3.6.1. Raw timing and certificate observations are in [pr-comparison.csv](pr-comparison.csv). This is a local nine-instance comparison, not a general speed guarantee or a new SCIPJack comparison.
