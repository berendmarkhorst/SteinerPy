# SteinerPy v1.1.0 on Snellius

All **9,843 requested measurements finished**, with no pending cases. Array
`28669777` (64 tasks) was submitted on 28 October 2026; summary job `28669778`
finished on **4 November 2026 at 17:28:16 Europe/Oslo**. Results were retrieved
and verified on 8 November. Slurm completion includes solver timeouts and errors;
it does not mean every instance was solved.

The measured release is tag `v1.1.0`, commit
`ef7265ee646ebd9f6b47635d68a2a4b7ee590200`. Each measurement uses one thread,
a 1,800-second whole-process deadline and an 8 GiB address-space limit.
The array allowed four concurrent allocations, each with three workers,
16 CPUs and 28 GiB. The array and summary consumed 10,427.84 allocated
CPU-hours according to Slurm (not measured solver CPU time or an invoice).

| Experiment | Measurements | Valid returned solutions | Timeouts | Other failures |
|---|---:|---:|---:|---:|
| [PACE 2018](pace/summary.md) | 1,797 | 1,222 | 349 | 226 |
| [API parity](api_parity/summary.md) | 7,644 | 4,482 | 2,886 | 276 |
| [SteinLib B/C](steinlib/summary.md) | 266 | 264 | 2 | 0 |
| [Prize collecting, JMP](pcstp/summary.md) | 136 | 136 | 0 | 0 |

## Findings

PACE covers all 599 available instances once per configuration. Optima below
include either a solver proof or a match to a published optimum; they are a
subset of valid solutions. A timeout without a returned checked solution is
not counted as feasible.

| Configuration | Valid / 599 | Proven/reference optima | Timeouts | Other failures |
|---|---:|---:|---:|---:|
| Gurobi exact | 414 | 300 | 152 | 33 |
| HiGHS exact | 277 | 273 | 195 | 127 |
| SteinerPy heuristic | 531 | 20 | 2 | 66 |

The [earlier frozen PACE run](../pace2018/README.md) had 296 Gurobi and
269 HiGHS optima. All five of its rejected Gurobi cases return valid trees in
this campaign: four match the reference optimum, while Track 1/098 returns
cost 1,500,435 versus optimum 1,500,422. No PACE result was rejected during
validation in this campaign. Improvements in returned-solution counts are
modest, and timeout counts rose. Dependency versions and concurrency differ,
so this comparison cannot isolate a release improvement or establish a speedup.

Large Track 3 instances remain the main weakness: Gurobi, HiGHS and heuristic
mode return valid trees for **46, 12 and 133 of 199** cases, respectively.
Across PACE, memory failures account for 5 Gurobi, 41 exact HiGHS and
66 heuristic results; another 28 Gurobi and 86 HiGHS runs return no
connectivity-valid incumbent.

The API experiment compares legacy `SteinerProblem(...)` with public
`Problem(...)` on 637 PACE/B/C instances, two solvers and three repetitions:
**3,822 pairs**. There are **zero objective/feasibility mismatch flags**,
2,241 matching feasible pairs (1,929 optimal), and 1,581 unavailable pairs.
Unavailable pairs do not establish solution agreement. Within each solver and
environment group, median constructor-plus-solve overhead on optimal pairs is
**0.129–0.207% (1.36–3.35 ms)**, below the campaign's predefined signal of
both 5% and 0.01 seconds. These plain problems dispatch to the specialized
legacy solver; this measures API entry-point overhead in the same release,
not forced generic composition or performance before and after modularization.

For SteinLib B/C, Gurobi exact and exact plus dual ascent each prove **38/38**
instances; the corresponding HiGHS methods each reach **37/38 known optima**,
with `c18.stp` timing out. HiGHS exact plus dual ascent retains tiny positive
solver gaps on two reference-optimal cases, so only 35 carry solver-proof flags.
All heuristic methods return valid trees. SteinerPy's heuristic
matches 18 optima, versus six each for NetworkX Kou and Mehlhorn.

All four prize-collecting methods return valid solutions on all 34 JMP cases.
Exact HiGHS proves **30/34**, Gurobi **24/34**, and their union covers 31 cases.
Among 31 instance pairs sharing an environment, `pcst_fast` has a lower
objective than SteinerPy's heuristic on 21, ties on seven and is worse on
three. The generated summary's proof column does not count later matches to
another method's PCSTP optimum: zero `pcst_fast` proofs does not mean zero
optimal solutions. Its objective matches 14 of the 29 available
same-environment exact references.

## Reproducibility and retained evidence

Python 3.12.3; NetworkX 3.6.1; NumPy 1.26.4; SciPy 1.16.3; HiGHS 1.12.0;
Gurobi 13.0.3; `pcst_fast` 1.0.10. Both recorded environments use
AMD EPYC 7H12 CPUs; they differ in Linux kernel patch version. The detailed
summaries retain separate environment groups for timing and quality statistics.
The overview above totals outcome counts only. The earlier PACE report used
HiGHS 1.15.1 and different NumPy, SciPy and NetworkX versions.

- Each experiment's `results.csv.gz` preserves every exported column,
  including original-edge certificates, objective/proof distinctions, failures,
  timings, input hashes, job IDs and environment metadata. Gzip compression is
  lossless; the four uncompressed CSVs total about 77 MB.
- [manifest.json.gz](manifest.json.gz) lists all expected cases and source,
  harness and input hashes. [source-manifest.json](source-manifest.json),
  [environment.json](environment.json), [environments/](environments/),
  [requirements.freeze.txt](requirements.freeze.txt),
  [dataset-sources.json](dataset-sources.json) and
  [submission.json](submission.json) retain the campaign provenance.
- [harness.tar.gz](harness.tar.gz) retains the six original harness files and
  their associated tests. Extract it over a separate checkout of the measured
  commit to inspect or reproduce the campaign; the harness is not installed
  into the current package. Dataset files must be supplied separately.
- [accounting.txt](accounting.txt) contains all 64 array tasks and the summary
  job, including Slurm steps. All allocation records completed with exit `0:0`.
- [verification.json](verification.json) records the retrieval checks.
  [download-sha256.txt](download-sha256.txt) records remote hashes of the
  original uncompressed exports and metadata; [SHA256SUMS](SHA256SUMS) hashes
  the files retained here.

Verification matched all 9,843 identities, input-hash declarations, settings and
environment records against the manifest, checked the source manifest against
the local Git archive of the measured commit, and verified all six harness
hashes. All four CSVs regenerated byte-for-byte from downloaded raw JSON;
all summary bodies matched apart from their generation timestamp. The copied
summaries only add download links and refer to the compressed CSV filename.

The full downloaded JSON/checkpoint records remain locally in the ignored
`benchmarks/results/snellius-v1.1.0-20261028/` directory. Original worker logs,
datasets and the complete campaign remain on Snellius under
`~/steinerpy-v1.1.0-campaign/source/benchmarks/releases/v1.1.0/`.
Retrieval did not rerun solvers, replay certificates against original input
graphs, or independently prove optimality. Validity/proof flags are the
campaign's recorded checks. The prior benchmark reports and package code
remain unchanged.
