# Local assessment of current main

The bounded pilot in PR #99 was run locally on 23 October 2026. It compares
baseline `9c980eca523c21101a5dd624901426a98d3cde56` with main after PRs #98,
#101 and #103, `a11d90eec231349640691f721a7fdca87b1578cb`. The shared harness
revision is `f8ab541053e4746a26b8a23e90527c8388b665a3`. All three source
checkouts were clean when the run started. No solver code was changed for this
experiment.

Current main produces better results on several general-model cases. It reaches
18 known or solver-proven optima in 24 runs, compared with 10 for the baseline.
Both versions still time out six times. This is a provisional assessment of
twelve fixed cases, each repeated twice, rather than a PACE ranking.

## Conditions

- Windows 11, Intel Family 6 Model 181, Python 3.12.15, HiGHS **1.15.1**,
  NetworkX 3.6.1, NumPy 2.3.5 and SciPy 1.18.0.
- One fresh worker at a time, one solver/BLAS thread, fifteen-second whole-worker
  deadline including imports, construction, solving and certificate checks.
  The inner solver allowance reserves time for startup and validation.
- Twelve cases, two revisions, two repetitions: **48 runs**. Order is baseline /
  candidate for the first repetition and candidate / baseline for the second.
  Total measured worker wall time was 381.3 seconds (about 6.4 minutes).
- HiGHS 1.15.1 matches the previous Snellius run, using an isolated interpreter;
  the project's HiGHS 1.12 installation was not upgraded. The pre-1.15 LP
  workaround from PR #103 is inactive here. No Gurobi/WLS sessions were opened.
- Windows enforces the deadline, but not the Linux address-space cap. The
  manifest therefore records `memory_limit_enforced=false`. PACE worker RSS is
  unmeasured on Windows (the raw zero is not a zero-memory result); completed
  combined-feature workers record their process peak RSS.

The six PACE inputs are instances 001 and 002 from each track. Their general
formulation is explicitly selected, independently of the public API's faster
classic dispatch. Two classic controls use Track 1/001 and Track 2/001. Four
combined-feature cases exercise the public API. Graph generation, validation,
limits and method selection are identical for both revisions.

## Outcomes

| Outcome | Baseline (24 runs) | Current main (24 runs) |
|---|---:|---:|
| Solver proves optimality | 10 | 14 |
| Matches published optimum, without solver proof | 0 | 4 |
| Valid, unproven solution above known optimum | 2 | 0 |
| Other valid, unproven combined-feature solution | 4 | 0 |
| Whole-worker timeout, no scored result | 6 | 6 |
| Rejected PACE tree certificate | 2 | 0 |

The table below gives median full worker wall seconds across the two repetitions.
An unfinished run's duration is not a time to optimum. Costs and outcomes take
priority over timing ratios.

| Case | Baseline seconds / result | Current main seconds / result |
|---|---|---|
| General Track 1/001 | 0.922 / optimum 503 | 0.883 / optimum 503 |
| General Track 1/002 | 15.094 / both time out | 15.008 / both time out |
| General Track 2/001 | 13.680 / one rejected, one valid cost 1144 | 13.570 / known optimum 1086, both runs |
| General Track 2/002 | 14.227 / one rejected, one valid cost 637 | 14.297 / known optimum 626, both runs |
| General Track 3/001 | 15.078 / both time out | 15.071 / both time out |
| General Track 3/002 | 15.071 / both time out | 15.078 / both time out |
| Resource tree, seed 42 | 13.469 / cost 97, unproven | 2.078 / optimum 96 proved |
| Resource tree, seed 43 | 13.476 / cost 94, unproven | 3.930 / optimum 94 proved |
| Groups and forest, seed 44 | 1.172 / optimum 22 | 1.180 / optimum 22 |
| Signed rewards, seed 44 | 1.570 / optimum 19 | 1.516 / optimum 19 |
| Classic Track 1/001 | 0.812 / optimum 503 | 0.820 / optimum 503 |
| Classic Track 2/001 | 1.320 / optimum 1086 | 1.328 / optimum 1086 |

Current main's four general Track 2 results match published exact reference
values, but their solver gaps remain unknown. They are not counted as solver
proofs. The baseline's rejected Track 2 outputs are connected but contain cycles
(cycle ranks four and three); the strict PACE tree checker rejects them. Their
reported costs, 1426 and 712, and original edge certificates remain in the raw
data. They are not included among valid solutions. The pilot exits nonzero to
flag these retained baseline validation failures, after writing all 48 results
and checking cross-version proof consistency. Current main has no validation
failures in this sample.

The strongest measured benefit is earlier proof completion on the two resource
cases, plus better known-optimum coverage on the general Track 2 cases. The
classic controls remain approximately unchanged. Three general PACE cases time
out in every run; the experiment does not establish a broad speedup.

## Reproduce and inspect

Use the [pilot instructions](../../../connectivity_pilot.md) with the frozen
baseline and candidate above and a fresh output directory:

```bash
python -m benchmarks.connectivity_pilot --dataset DATASET --baseline BASELINE --candidate CANDIDATE --output LOCAL_RESULTS --jobs 1 --time-limit 15
```

Use the same pinned environment for both revisions. The local interpreter reused
the project's dependencies with an isolated HiGHS 1.15.1 installation. The
native version and package versions are retained in [manifest.json](manifest.json).
The original per-worker results, including original-edge certificates, statuses,
phase timings and native solver records, are in
[observations.jsonl](observations.jsonl). Only the local machine name and absolute
paths were removed; each result still identifies its source revision and input.

The official dataset revision is `4df73cea9c311faea7d03e6d6bffa8733c34a1aa`.
Windows Git checked out CRLF line endings, so raw byte hashes differ from the
earlier Linux sweep. `lf_instance_sha256` records the LF-normalized hash; all six
match the earlier comparison's input hashes exactly. Both local revisions read
the same bytes, and combined-feature input hashes also match across all runs.

This pilot does not rerun a PACE winner. There is no local Rust toolchain or WSL
installation for the existing wata&sigma comparison runner. The earlier
[same-hardware comparison on Snellius](../../pace2018/README.md#same-hardware-comparison-sample)
remains separate and is not presented as a measurement of current main. The
local improvements do not yet establish that SteinerPy is close to the PACE
winners. No new HPC job was submitted during this assessment.
