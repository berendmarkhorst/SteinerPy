# Small Snellius connectivity comparison

This follows PR #98 and keeps the completed PACE 2018 report unchanged.
Compare baseline `9c980eca523c21101a5dd624901426a98d3cde56` with merged
candidate `a11d90eec231349640691f721a7fdca87b1578cb` (main after PRs #98, #101
and #103) using the same harness,
Python environment, compute node, limits and input data.

The fixed sample contains twelve cases:

- PACE instances 001 and 002 from each track, forcing the general formulation.
- Public `Problem(...)` calls for resource-tree seeds 42/43, groups-forest seed
  44 and signed-rewards seed 44.
- Classic-path controls on Track 1/001 and Track 2/001.

Two repetitions of both versions give 48 fresh workers. Each uses one HiGHS
thread, an 8 GiB address-space cap, and a 120-second whole-process deadline,
including imports, construction and original-solution checks. Each case runs
baseline/candidate, then candidate/baseline; up to three case sequences run
concurrently on one node. No Gurobi environments or WLS sessions are used.

The workers total at most 1.6 core-hours. Rome's [smallest allocation](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions)
is 16 cores and 28 GiB. Three workers fit with supervisor headroom; the supplied
40-minute job therefore reserves at most **10.67 SBU**, before any site-specific
adjustments. The charge follows allocation time, including idle reserved cores,
and should be lower if the job finishes early. Check the live partition policy
before submission. The former 64-core, 24-hour sweep script is unnecessary here.

## Prepare and submit on Snellius

Use a new pilot directory and clean, detached checkouts. Fetch the harness branch
first, then replace `HARNESS_SHA` below with its reviewed commit:

```bash
BENCH_ROOT="$HOME/steinerpy-benchmarks"
export PILOT_ROOT="$BENCH_ROOT/connectivity-pilot-20261023"
git -C "$BENCH_ROOT/source" fetch origin
mkdir "$PILOT_ROOT"
git -C "$BENCH_ROOT/source" worktree add --detach "$PILOT_ROOT/baseline" \
  9c980eca523c21101a5dd624901426a98d3cde56
git -C "$BENCH_ROOT/source" worktree add --detach "$PILOT_ROOT/candidate" \
  a11d90eec231349640691f721a7fdca87b1578cb
git -C "$BENCH_ROOT/source" worktree add --detach "$PILOT_ROOT/harness" HARNESS_SHA
cd "$PILOT_ROOT"
sbatch "$PILOT_ROOT/harness/benchmarks/slurm_connectivity_pilot.sh"
```

The existing virtual environment and official PACE dataset are reused. The
manifest records exact revisions, dependencies, CPU, node, job ID and instance
hashes. Source checkouts must be clean, and output directories must be new to
prevent mixing observations. Each worker saves its result, log, selected edges,
proof status, wall time and memory. Completed combined-feature workers also save
their selected nodes and model/solver phase measurements. The supervisor retains
timeouts and failures, and checks for conflicting proof claims and changed
generated inputs. A killed job can still leave useful individual result files.

After the job, collect `results-JOB_ID` and the Slurm log. Compare proved optima,
feasible objectives and missing results before calculating paired timing ratios.
A timeout is not a measured solve time, and an unproved incumbent is not a proved
optimum. Two repeats on twelve cases are a pilot, not a PACE ranking or a general
speedup guarantee; even unchanged controls may vary under shared-node load.

## Short local assessment

The same twelve cases and two alternating repetitions can run sequentially with
`--jobs 1 --time-limit 15`. This caps the 48 workers at twelve minutes in total,
plus supervisor overhead. The default HPC limit remains 120 seconds. Use a new
output directory so the local observations cannot be mixed with HPC results:

```bash
python -m benchmarks.connectivity_pilot --dataset DATASET --baseline BASELINE --candidate CANDIDATE --output LOCAL_RESULTS --jobs 1 --time-limit 15
```

A sparse copy of the official dataset is sufficient: retain the three `trackN.csv`
manifests and `instance001.gr`/`instance002.gr` from each `TrackN` directory. The
runner checks and hashes every selected file. Keep the pinned dataset revision
`4df73cea9c311faea7d03e6d6bffa8733c34a1aa` for comparison with the previous sweep.

Use HiGHS 1.15.1 for this comparison, matching the previous Snellius run. The
manifest records both package and native solver versions. Windows enforces the
worker deadline but does not enforce the Linux 8 GiB address-space cap; this is
recorded as `memory_limit_enforced=false`. Run one worker at a time locally.

These short local measurements are a provisional assessment. The pilot compares
SteinerPy revisions, not PACE competitors. New Snellius performance claims require
completed HPC data, and a current winner comparison requires rerunning that solver
under the same conditions.
