# Uniform-cost node formulation

> Historical comparison: `auto` below used the initial rule that selected the
> node model for every mathematically eligible instance. That rule has been
> replaced by the structural selector documented in the
> [selection report](../uniform-selection/README.md). The measurements and raw
> records below are retained unchanged; they do not measure the new selector.

This bounded comparison supports [issue #110](https://github.com/berendmarkhorst/SteinerPy/issues/110).
The node formulation substantially improves the hypercube and dense examples,
but regresses on the grid. At measurement time, automatic selection tested
mathematical eligibility; it did not predict which formulation would be fastest. Set
`STEINERPY_UNIFORM_NODE_MODEL=0` to retain the directed-cut formulation.

## Method

The same working tree was measured with automatic selection enabled (`auto`)
and disabled (`directed`). Four deterministic graphs have unit edge costs:

| Case | Nodes / edges | Mandatory terminals |
|---|---:|---|
| Cycle | 60 / 60 | Every fifth node: 12 terminals |
| Grid | 64 / 112 | Even coordinates of the 8 × 8 grid: 16 terminals |
| Hypercube | 32 / 80 | Even-parity vertices of the five-dimensional cube: 16 terminals |
| Dense | 48 / 512 | The 16-node partition of the complete bipartite graph K16,32 |

Each backend ran three paired repetitions with preprocessing off and on: 48
measurements per backend, 96 in total. Configurations alternate within each
pair and reverse order on alternate repetitions. Every measurement uses a
fresh process, one solver/BLAS thread, a five-second solve limit, and a
30-second worker deadline. The few-terminal DP threshold remains ten and
decomposition is disabled. Preprocessing solves the cycle completely; every
other automatic run uses the node model. The reduced grid has 49 nodes and 84
edges; the other two graphs retain their original sizes.

Times below are median construction plus solve times, including preprocessing,
model building, and the solver's own result handling. Imports and graph
generation precede timing. The benchmark independently checks original-edge
membership, duplicate edges, connectivity, acyclicity, and objective after the
timed solve. Peak RSS in the raw records covers the entire worker process.
Rows retain missing incumbents, unproven incumbents, and errors explicitly;
an unknown gap is JSON `null`.

Environment: Windows 11 build 26100, Intel64 Family 6 Model 181 Stepping 0,
Python 3.12.15, NetworkX 3.6.1, HiGHS 1.12.0, SciPy 1.18.0, and Gurobi 13.0.3.
The working tree was based on `e5457c3524f8c4e63cf96ade833aad9f718e7df7`
with the issue #110 changes uncommitted. Raw records include that base commit,
the dirty flag, dependency versions, actual formulation, reduced sizes, and cut
statistics. These synthetic examples and local timings do not establish a
general speedup across uniform-cost instances.

## HiGHS

[Full paired results](highs.json). Proof counts are out of three repetitions
per configuration. Times for incomplete runs reflect the time budget, so their
ratios are not speedups for equivalent outcomes.

| Case | Preprocess | Auto seconds | Directed seconds | Auto / directed proofs |
|---|---|---:|---:|---:|
| Cycle | Off | 0.494 | 0.190 | 3 / 3 |
| Cycle | On | 0.011 | 0.011 | 3 / 3 |
| Grid | Off | 5.008 | 0.456 | 0 / 3 |
| Grid | On | 5.023 | 0.430 | 0 / 3 |
| Hypercube | Off | 0.959 | 5.066 | 3 / 0 |
| Hypercube | On | 0.898 | 5.057 | 3 / 0 |
| Dense | Off | 0.009 | 5.284 | 3 / 0 |
| Dense | On | 0.030 | 5.265 | 3 / 0 |

Automatic selection proves 18 of 24 solves; the six grid runs stop without a
connectivity-valid incumbent. The directed formulation proves 12 of 24 solves,
returns five checked unproven incumbents, and stops without an incumbent seven
times. All directed hypercube and dense runs reach their solve limit; returned
incumbents have costs 20 or 22 for the hypercube and 24 for the dense graph.
The node model proves the corresponding optima 20 and 16.

## Gurobi

[Full paired results](gurobi.json).

| Case | Preprocess | Auto seconds | Directed seconds | Auto / directed proofs |
|---|---|---:|---:|---:|
| Cycle | Off | 0.047 | 0.059 | 3 / 3 |
| Cycle | On | 0.025 | 0.026 | 3 / 3 |
| Grid | Off | 5.030 | 0.159 | 1 / 3 |
| Grid | On | 3.951 | 0.222 | 3 / 3 |
| Hypercube | Off | 0.070 | 0.499 | 3 / 3 |
| Hypercube | On | 0.064 | 0.466 | 3 / 3 |
| Dense | Off | 0.029 | 0.078 | 3 / 3 |
| Dense | On | 0.041 | 0.097 | 3 / 3 |

Automatic selection proves 22 of 24 solves; the remaining two unpreprocessed
grid runs return checked cost-30 trees without proof. The directed formulation
proves all 24. The node model is faster on the hypercube and dense examples,
while the grid strongly favors the directed formulation even with preprocessing.

All returned certificates pass independent checks, and every pair of proven
objectives agrees. The proven cycle, grid, hypercube, and dense objectives are
55, 30, 20, and 16 respectively. There are no worker errors or hard worker
timeouts. Earlier single-repetition, two-second smoke results are retained in
[HiGHS smoke](highs-smoke.json) and [Gurobi smoke](gurobi-smoke.json).

## Reproduce

From the issue #110 implementation checkout, with the same dependencies:

```bash
python benchmarks/benchmark_uniform.py --solver highs --output highs.json
python benchmarks/benchmark_uniform.py --solver gurobi --output gurobi.json
```

Running these commands with the current selector can choose different
formulations for `auto` than the historical runs recorded here.

Gurobi requires an installed, licensed backend. Unavailable backends are
recorded explicitly. Use `--repeats 1 --time-limit 2` for the smoke comparison,
`--cases grid` for a subset, or `--preprocess off` for unreduced inputs only.
