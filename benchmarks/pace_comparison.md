# Bounded policy and PACE comparison

Issue #104 follows the [completed connectivity pilot](reports/connectivity/snellius-pilot-20261023/README.md).
Keep its two questions separate: when should the general model be strengthened,
and how does the normal public API compare with a PACE participant?

The fixed experiment has two parts, each with two alternating repetitions:

1. Twelve combined-feature cases compare the one-second policy on main after
   PR #99 (`d982be2f56905db602fd86818b4ac5f23edf3dd5`) with the candidate policy.
   Signed-rewards seeds 40–45, resource-tree seeds 42–45 and groups-forest seeds
   42/44 include additional inputs beyond the original regression. Each fresh
   worker has a fifteen-second whole-process deadline; there are 48 workers.
2. Twelve PACE instances compare the candidate's normal
   `Problem(graph, terminals=..., preprocess=True)` dispatch using HiGHS with the
   original [wata&sigma submission](https://github.com/wata-orz/steiner_tree),
   revision `2700bdc3033d0d01c5d3648832b4535fceb70991`. Instances 001, 050, 101
   and 150 from each track provide two public and two private cases spread
   through each archive, chosen independently of solver outcomes. The shared
   whole-process deadline is 120 seconds, including certificate validation.
   There are another 48 workers. Use `track2` for Track 2 and `track1` otherwise,
   matching the existing reference runner.

Both methods have one solver thread and an 8 GiB address-space cap. Up to three
case sequences run concurrently on one Rome node. Solver versions, source
revisions, generated-input hashes and competitor binary hashes are recorded.
The default API path includes classic reductions and the few-terminal dynamic
program where applicable; general-model forcing is not used for the PACE part.
Decomposition and dual ascent remain off, as in the preceding controlled runs.
No Gurobi/WLS sessions are opened.

The 16-core, 28-GiB allocation is capped at 45 minutes: at most twelve allocated
core-hours before site charging adjustments. The workers sum to at most 1.8
core-hours; the allocation also pays for idle reserved cores. The worst-case
worker schedule is 36 minutes, leaving startup and cleanup headroom.

Prepare a new directory with clean detached `baseline`, `candidate` and `harness`
checkouts, as in the [pilot instructions](connectivity_pilot.md). Pin the official
dataset to `4df73cea9c311faea7d03e6d6bffa8733c34a1aa`, native HiGHS to 1.15.1,
and the existing built competitor to the revision above. Submit from that new
directory:

```bash
sbatch "$PWD/harness/benchmarks/slurm_pace_comparison.sh"
```

The supplied script uses the existing Snellius environment and competitor build.
The runner refuses dirty source trees, equal baseline/candidate revisions,
incomplete selected inputs, existing output directories and non-Linux systems.
It writes `manifest.json`, `combined.json`, `pace.json`, and individual worker
requests/logs/results. Original-solution checks, proof consistency and input
checks run for both parts. Failures remain in the data and cause a nonzero exit;
timeouts are retained as missing results, never times to optimum.

Report paired outcomes and objectives before timing ratios. Only compare times
to the same proved optimum where both finish. Two repeats of twelve PACE inputs
are a diagnostic comparison, not a competition ranking or a full reproduction of
the 2018 thirty-minute exact-track budget. Track 3 is an exact-solver diagnostic
here, not a rerun of its original heuristic leaderboard.

SCIP-Jack is not installed in the existing environment. Its
[archived repository](https://github.com/dRehfeldt/SCIPJack-Steiner-tree-solver)
requires separately supplied SCIP/SoPlex sources and recommends a newer solver.
This experiment reuses the pinned original wata&sigma build; a SCIP-Jack build
and matched comparison remain a separate extension, not an inferred result.
