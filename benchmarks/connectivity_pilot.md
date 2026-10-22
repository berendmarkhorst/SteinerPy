# Small Snellius connectivity comparison

This follows PR #98 and keeps the completed PACE 2018 report unchanged.
Compare baseline `9c980eca523c21101a5dd624901426a98d3cde56` with merged
candidate `c9f4e67dbdfecabe3e243276f58542e21ca84c0e` using the same harness,
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
export PILOT_ROOT="$BENCH_ROOT/connectivity-pilot-20261022"
git -C "$BENCH_ROOT/source" fetch origin
mkdir "$PILOT_ROOT"
git -C "$BENCH_ROOT/source" worktree add --detach "$PILOT_ROOT/baseline" \
  9c980eca523c21101a5dd624901426a98d3cde56
git -C "$BENCH_ROOT/source" worktree add --detach "$PILOT_ROOT/candidate" \
  c9f4e67dbdfecabe3e243276f58542e21ca84c0e
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

The previous local measurements establish promising improvements on some hard
combinations. New Snellius performance claims require the completed pilot data.
