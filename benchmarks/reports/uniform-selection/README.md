# Automatic uniform-cost formulation selection

This follow-up to [the initial uniform-cost comparison](../uniform/README.md)
separates mathematical eligibility from the choice of a fast formulation.
The initial implementation selected the node model on every eligible instance;
the grid regressions show why uniform costs alone are insufficient.

The structural rule below is now the default. Across 712 validation/recheck
measurements, it retains all paired baseline proofs and feasible incumbents,
with no material runtime or returned-objective regression under the stated
criteria. This supports the conservative automatic choice on the measured
domain; it does not imply that every eligible graph benefits.

## The rule

After the existing exact shortcuts and mathematical eligibility checks, use
the node model if either:

1. The subgraph induced by the distinct terminals is connected; or
2. All terminals are in one component and, after ignoring other components and
   repeatedly removing nonterminal leaves, every edge joins a terminal to a
   nonterminal and there are at least five edges per remaining nonterminal on
   average.

The component and leaf operations only compute the selection features. They
do not modify the solver's working graph or reduction mappings. Uniformity is
checked after preprocessing, using the configured weight attribute and the
existing missing-weight default of one. Constraints, forests, transformed
variants, explicit dual ascent/primal portfolio, and enumeration retain their
existing paths.

The first branch has a direct justification: a spanning tree on the terminals
attains the lower bound of `|T| - 1` edges. The second branch identifies an
empirically favorable domain for the smaller node formulation. Five is a
calibrated threshold, not a mathematical performance guarantee or the paper's
threshold. Here, the bipartition must agree with terminals versus nonterminals;
ordinary graph bipartiteness is insufficient.

Leave `STEINERPY_UNIFORM_NODE_MODEL` unset, or set it to `auto`, to use this
rule. Set it to `0` for the existing formulation or `1` to force the node model
within its mathematical domain. Neither override bypasses earlier shortcuts.

The [frozen rule](frozen-rule.json) records its source, source hashes, calibration
evidence hashes, and validation manifests before validation solves began.

## Calibration

The initial four graphs were followed by 52 generated discovery cases spanning
13 families and 12 deliberately difficult challenge cases. These vary terminal
placement as well as graph structure. The challenge cases include periodic and
three-dimensional grids, dense attachments, terminal cliques connected through
a grid, linked incidence graphs, and irrelevant leaf branches.

Two attractive broader rules failed this screen:

- Density alone selects a grid with a dangling clique. HiGHS's node model
  reached the five-second limit without a valid incumbent; the existing model
  proved optimality in about 0.31 seconds.
- Few connected terminal components alone selects two terminal cliques linked
  through a grid. The node model again reached the limit without an incumbent;
  the existing model proved optimality in about 0.19 seconds.

The final rule deliberately keeps these structures on the existing formulation.
It also misses some node-model wins, including king grids. This is a conservative
dispatch rule rather than an attempt to identify the fastest model on every
instance. A suspected Gurobi slowdown in the discovery screen disappeared in
three repeated pairs; both the original result and repetitions are retained.

Calibration data: [discovery HiGHS](discovery-highs.json),
[discovery Gurobi](discovery-gurobi.json), [challenge HiGHS](challenge-highs.json),
[challenge Gurobi](challenge-gurobi.json),
[Gurobi recheck](discovery-gurobi-repeat.json).

## Validation results

The initial validation screen uses one paired measurement per configuration,
backend, and instance. The table excludes the six calibration repeats from the
generated set. "Closed" includes the numerical residuals explained below;
valid-tree counts also include unproved incumbents.

| Corpus, preprocessing off | Backend | Closed: auto / directed | Valid tree: auto / directed | Node selections |
|---|---|---:|---:|---:|
| 46 new generated cases, 5 s | HiGHS | 38 / 30 | 39 / 31 | 17 |
| 46 new generated cases, 5 s | Gurobi | 44 / 44 | 46 / 46 | 17 |
| 12 public inputs, 10 s | HiGHS | 3 / 1 | 4 / 1 | 6 |
| 12 public inputs, 10 s | Gurobi | 5 / 2 | 9 / 7 | 6 |

For the new generated HiGHS cases, the strict-zero counts are 35 versus 26;
another three versus four have numerical-zero residuals. The public HiGHS
directed completion has a numerical-zero residual. The other table entries
are strict-zero proofs. Including the six repeats, generated-case closure
counts become 42/34 for HiGHS and 50/48 for Gurobi; Gurobi's two additional
proofs in that larger table are repeated dimension-six cubes, not new cases.

These four screens contain no lost proof, missing feasible tree, worse returned
objective, or material runtime regression for automatic selection relative to
the paired directed run. In the initial screen on the new 24-terminal,
degree-six incidence case, HiGHS improves from 4.03 s to 0.043 s and Gurobi from
0.39 s to 0.026 s, with the
same optimal objective. On public `bipe2u`, automatic selection proves the
optimum in 2.29 s (HiGHS) and 0.48 s (Gurobi); neither paired directed run finds
a valid incumbent. These censored pairs are proof gains, not speedup ratios.

The conservative rule leaves real gains unused: forced node solves public
`cc6-2n`, `s2`, and `s3` on HiGHS while the automatic directed path remains
incomplete. Gurobi also benefits from forcing node on some excluded inputs.
The rule was not revised to capture these validation-set wins.

Raw screen results: [generated HiGHS](heldout-highs.json),
[generated Gurobi](heldout-gurobi.json), [public HiGHS](external-highs.json),
[public Gurobi](external-gurobi.json). Peak process RSS across these respective
screens is 174.5/212.1, 112.3/121.5, 172.9/188.0, and 301.4/300.1 MiB for
auto/directed. These maxima include fallback solves and do not establish a
per-instance memory improvement. The existing Gurobi directed path exposes no cut counters;
an empty counter record means unavailable, not zero cuts.

The selected public hypercubes were also compared at **30 seconds**, with one
fresh auto/directed pair per instance and backend. No later proof, incumbent,
or objective regression appeared. HiGHS automatic returns an objective-39 tree
for `hc6u` while directed finds none; both remain incomplete on `hc7u` and
`hc8u`. Gurobi automatic proves `hc6u` in 0.18 s while directed remains
unproved at 30 s. Both return objective 77 without proof on `hc7u`; automatic
returns objective 148 without proof on `hc8u`, where directed finds no tree.
The latter matches a published upper bound, not a proven optimum in the checked
reference. Raw results: [30-second HiGHS](external-long-highs.json) and
[30-second Gurobi](external-long-gurobi.json).

Default preprocessing was then checked on the same generated instances. For
the 46 new cases:

| Backend | Closed: auto / directed | Valid tree: auto / directed | Node selections |
|---|---:|---:|---:|
| HiGHS | 43 / 42 | 43 / 42 | 4 |
| Gurobi | 44 / 44 | 46 / 46 | 4 |

The HiGHS closure counts contain six numerical-zero residuals on each side.
On both backends, 16 new cases finish through trivial/preprocessing shortcuts
and ten use Dreyfus–Wagner. Only four still select the node model; the other 16
use directed cuts. There is no selected-model loss. Including the six
calibration repeats, closure counts are 47/46 on HiGHS and 50/48 on Gurobi.
Raw results: [preprocessing HiGHS](heldout-preprocess-highs.json) and
[preprocessing Gurobi](heldout-preprocess-gurobi.json).

Five-second HiGHS cube-six runs vary in whether they find a valid incumbent,
including between forced-node and automatic runs of the same formulation.
Their paired directed runs find none. This variation is retained; a single
successful incumbent at the deadline is not evidence of a stable success rate.

Two completed selected-model pairs were near their solve limits and received
three additional alternating paired repetitions. On the 24-terminal incidence
case, HiGHS automatic proves all three in 0.044–0.046 s; all three directed
repeats time out without an incumbent. On `csd10`, Gurobi automatic proves all
three in 0.026–0.029 s; directed proves two in 9.18–9.60 s and returns an
unproved optimum-valued incumbent in the third. The first-screen timing ratios
are therefore not claimed as stable multiplicative speedups. Recheck data:
[HiGHS incidence](heldout-near-limit-highs.json),
[Gurobi csd10](external-near-limit-gurobi.json).

Finally, the original four graphs were rerun with the final automatic rule,
both preprocessing settings, and three alternating pairs per setting/backend.
Median construction-plus-solve seconds are:

| Graph | Preprocess | HiGHS auto | HiGHS directed | Gurobi auto | Gurobi directed |
|---|---|---:|---:|---:|---:|
| Cycle 60 | Off | 0.143 | 0.143 | 0.055 | 0.055 |
| Cycle 60 | On | 0.007 | 0.007 | 0.023 | 0.022 |
| Grid 8×8 | Off | 0.410 | 0.405 | 0.140 | 0.135 |
| Grid 8×8 | On | 0.305 | 0.305 | 0.208 | 0.207 |
| Cube, dimension 5 | Off | 0.847 | 5.029† | 0.049 | 0.402 |
| Cube, dimension 5 | On | 0.868 | 5.043† | 0.058 | 0.414 |
| Complete bipartite K16,32 | Off | 0.010 | 5.310† | 0.025 | 0.072 |
| Complete bipartite K16,32 | On | 0.023 | 5.392† | 0.037 | 0.084 |

Every automatic run and every Gurobi run proves optimality. † marks zero proofs
in three repetitions: all disabled HiGHS cube runs find no incumbent; each
dense setting has two missing incumbents and one unproved objective-25 tree
(optimum 16). These times are censored and must not be divided into speedup
ratios. The cycle uses directed cuts without preprocessing and a shortcut with
it; the grid retains directed cuts in both settings. Cube and dense cases use
the node model. Raw data: [original HiGHS](original-highs.json),
[original Gurobi](original-gurobi.json).

The validation and recheck reports above retain **712 measurements** in total.
All returned trees pass independent edge, connectivity, acyclicity, terminal,
and objective checks. All paired proven objectives agree, and every public
result is consistent with the independent reference bounds. The
[aggregate audit](validation-audit.json) records file hashes, counts, and
regression checks. The frozen rule
and executable code were unchanged throughout timings. A class-docstring
clarification made afterward is recorded separately in `frozen-rule.json`;
its executable AST was verified unchanged.

## Evaluation protocol

The generated discovery and validation manifests are fixed before timings are
collected. They vary graph family, size, seed, and terminal placement. An
[overlap audit](manifest-overlap.json) found that six of the 52 reserved cases
repeat deterministic calibration instances: two checkerboard grids, two sparse
grids, and two parity cubes. Their seeds do not affect those generators. The
report distinguishes the other 46 generated cases from these repeat checks;
different case IDs alone do not make a case unseen. Both
formulations run in fresh processes with one solver/BLAS thread, equal solve
limits, and independently checked original-space results. The initial screen
uses five seconds per solve, separately for HiGHS and Gurobi. Model selection
must depend only on the current working graph, terminals, and backend.

Discovery data may guide a small, explicit rule. The rule is frozen before
running the held-out suite. Both forced formulations remain available in the
benchmark so automatic selection can be compared with each on identical input.
Default preprocessing is checked separately because it can change eligibility
and terminal structure. Publicly sourced uniform instances provide an additional
stress check where available; downloaded data stays outside version control.

The acceptance checks are:

- Every returned tree must pass the independent certificate checks; proven
  objective values must agree across formulations.
- On held-out cases selected for the node model, preserve the directed model's
  successful proofs and availability of a valid incumbent within the same
  limit, and retain meaningful wins.
- For pairs where both prove optimality, flag a material slowdown when node
  time exceeds both twice the directed time and the directed time plus 0.1 s.
  Recheck flagged pairs and completed selected-model pairs taking at least 80%
  of the solve limit with three repetitions; retain all runs.
- Report proof counts, missing incumbents, and worst regressions separately
  from timing summaries. A timeout ratio is not a measured speedup.
- Keep uncertain structures on the existing directed formulation. Do not fit
  a collection of family-specific exceptions or reuse validation results as
  discovery evidence while still claiming the same validation set is unseen.

This protocol tests a conservative dispatch policy on a bounded corpus; it
cannot establish a universal performance guarantee for an NP-hard problem.

Raw status uses the public contract: `optimal` means `gap == 0`. Some existing
directed-model runs finish with gaps around `1e-16`; the summary also reports
these numerical-zero completions separately. They are not timeout incumbents.
"Closed" requires a valid tree certificate and a finite `gap <= 1e-12`.
The unit-cost inputs have integral objectives, and these residuals are far below
one edge. Interrupted node-model incumbents retain an unknown gap (`null` in
JSON). A valid tree and a completed optimality proof are separate measurements.

The runs use Windows 11, Python 3.12.15, NetworkX 3.6.1, HiGHS 1.12.0, and Gurobi
13.0.3 on one Intel processor (family 6, model 181), with one solver and BLAS
thread per fresh process. Raw files include the complete environment metadata.
The bounded single-thread experiment does not establish behavior at longer
limits, on other hardware, or on all uniform-cost graphs.

External objective checks use [reference bounds](external-bounds.json) from
the primary DIMACS papers. Nine inputs have a published optimum in the checked
sources; `hc8u` has only a historical lower/upper interval there. The two `csd`
optima are derived independently from their complete-graph subdivision
structure. Matching a published upper bound alone is not treated as a proof.

## Package verification

- Isolated PR checkout on Python 3.12.15: **3,998 passed, 29 skipped**, with nine existing
  constrained-class deprecation warnings. Overall and node-module coverage are
  both 94%.
- The earlier broader working-tree run passed 4,025 tests; its count remains in
  the audit. The isolated checkout excludes unrelated pending benchmark/release work.
- Sphinx HTML build passed with `-W --keep-going`.
- All four documentation notebooks executed successfully.
- All seven new Python files match Black and parse with Python 3.8 syntax.
  Python 3.8 runtime compatibility was not exercised locally.
- The download example verified all twelve existing input checksums.

No dependency or package-version change is required.

## Reproduction

The generated [discovery](discovery-manifest.json),
[challenge](challenge-manifest.json), and [held-out](heldout-manifest.json)
manifests are retained. The `heldout` file contains the 46 new generated cases
and six repeats described above. The [external manifest](external-manifest.json)
identifies 12 uniform
instances from the primary [SteinLib](https://steinlib.zib.de/download.php) and
[DIMACS](https://dimacs11.zib.de/downloads.html) collections, including archive
URLs, member names, and input/archive SHA-256 hashes. Downloaded instances stay
in the ignored `benchmarks/data/uniform` directory.

Run from the repository root, with the package and chosen backend installed:

```sh
python benchmarks/uniform_selection.py --split heldout --solver highs \
  --configs node directed auto --time-limit 5 --preprocess off \
  --output benchmarks/reports/uniform-selection/heldout-highs.json
python benchmarks/uniform_selection.py --split external --solver highs \
  --manifest benchmarks/reports/uniform-selection/external-manifest.json \
  --configs node directed auto --time-limit 10 --preprocess off \
  --output benchmarks/reports/uniform-selection/external-highs.json
python benchmarks/uniform_selection.py --split heldout --solver highs \
  --configs auto directed --time-limit 5 --preprocess on \
  --output benchmarks/reports/uniform-selection/heldout-preprocess-highs.json
python benchmarks/uniform_selection.py --split external --solver highs \
  --manifest benchmarks/reports/uniform-selection/external-manifest.json \
  --cases external-hc6u external-hc7u external-hc8u \
  --configs auto directed --time-limit 30 --worker-timeout 120 \
  --output benchmarks/reports/uniform-selection/external-long-highs.json
python benchmarks/benchmark_uniform.py --solver highs --repeats 3 \
  --time-limit 5 \
  --output benchmarks/reports/uniform-selection/original-highs.json
```

Repeat with `--solver gurobi` and distinct output names. Use `--cases` to select
instances for rechecks and `--repeats 3` for three alternating repetitions.
The harness records the actual formulation or earlier solve path, construction
and solve times, proof status, objective, cuts, peak process RSS, and independent
tree-certificate checks. Timeouts and missing incumbents remain in the data.
Structural descriptor computation is excluded from timings; automatic selection
overhead is included. Solver limits exclude graph/model construction, and the
worker deadline also covers preprocessing and process startup.

To fetch only the external files named by the manifest, run this Python snippet
from the repository root. It verifies both the downloaded archive and each
instance, and reads individual archive members without extracting directories.
Existing files are checked before reuse.

```python
from pathlib import Path
import hashlib
import io
import json
import tarfile
import urllib.request
import zipfile

manifest = json.loads(Path(
    "benchmarks/reports/uniform-selection/external-manifest.json"
).read_text())
archives = {}
for spec in manifest:
    destination = Path(spec["path"])
    if destination.exists():
        data = destination.read_bytes()
    else:
        url = spec["source_url"]
        if url not in archives:
            with urllib.request.urlopen(url, timeout=60) as response:
                archives[url] = response.read()
        archive = archives[url]
        assert hashlib.sha256(archive).hexdigest() == spec["archive_sha256"]
        if url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive)) as source:
                data = source.read(spec["archive_member"])
        else:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
                data = source.extractfile(spec["archive_member"]).read()
    assert hashlib.sha256(data).hexdigest() == spec["sha256"]
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
```
