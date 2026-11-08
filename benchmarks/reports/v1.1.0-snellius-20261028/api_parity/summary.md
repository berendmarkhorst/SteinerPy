# v1.1.0 — api_parity

Mode: **full**. Commit: `ef7265ee646ebd9f6b47635d68a2a4b7ee590200`. Generated: 2026-11-04T16:28:15.552625+00:00.

Expected: 7644. Finished: 7644. Pending: 0.

Each environment ID identifies its recorded software, hardware and limits; environments are never pooled. See `../environment.json` and `../environments/`.

| Environment | Dataset/track | Method | Solver | Expected | Feasible | Proven/reference optimal | Timeout | Failed/unavailable | Pending | Median wall s | Median reference gap % |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 41e9ea36e64651742cafd58f | B | legacy | gurobi | 49 | 49 | 49 | 0 | 0 | 0 | 0.60243 | 0 |
| 41e9ea36e64651742cafd58f | B | legacy | highs | 50 | 50 | 50 | 0 | 0 | 0 | 0.60241 | 0 |
| 41e9ea36e64651742cafd58f | B | modular | gurobi | 49 | 49 | 49 | 0 | 0 | 0 | 0.60237 | 0 |
| 41e9ea36e64651742cafd58f | B | modular | highs | 50 | 50 | 50 | 0 | 0 | 0 | 0.60246 | 0 |
| 41e9ea36e64651742cafd58f | C | legacy | gurobi | 55 | 55 | 55 | 0 | 0 | 0 | 1.4033 | 0 |
| 41e9ea36e64651742cafd58f | C | legacy | highs | 53 | 50 | 50 | 3 | 0 | 0 | 1.554 | 0 |
| 41e9ea36e64651742cafd58f | C | modular | gurobi | 55 | 55 | 55 | 0 | 0 | 0 | 1.4032 | 0 |
| 41e9ea36e64651742cafd58f | C | modular | highs | 53 | 50 | 50 | 3 | 0 | 0 | 1.5541 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 1 | legacy | gurobi | 540 | 465 | 395 | 75 | 0 | 0 | 2.5064 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 1 | legacy | highs | 534 | 350 | 350 | 184 | 0 | 0 | 2.3043 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 1 | modular | gurobi | 540 | 465 | 395 | 75 | 0 | 0 | 2.4062 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 1 | modular | highs | 534 | 350 | 350 | 184 | 0 | 0 | 2.3045 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 2 | legacy | gurobi | 536 | 504 | 340 | 29 | 3 | 0 | 13.329 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 2 | legacy | highs | 531 | 352 | 343 | 173 | 6 | 0 | 13.473 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 2 | modular | gurobi | 536 | 504 | 340 | 29 | 3 | 0 | 13.432 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 2 | modular | highs | 531 | 352 | 343 | 173 | 6 | 0 | 13.523 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 3 | legacy | gurobi | 534 | 91 | 55 | 431 | 12 | 0 | 414.44 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 3 | legacy | highs | 527 | 30 | 28 | 395 | 102 | 0 | 296.96 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 3 | modular | gurobi | 534 | 91 | 55 | 431 | 12 | 0 | 414.05 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 3 | modular | highs | 527 | 30 | 28 | 395 | 102 | 0 | 297.18 | 0 |
| 9b16f2a2dc76c609a933acb5 | B | legacy | gurobi | 5 | 5 | 5 | 0 | 0 | 0 | 0.60248 | 0 |
| 9b16f2a2dc76c609a933acb5 | B | legacy | highs | 4 | 4 | 4 | 0 | 0 | 0 | 0.65246 | 0 |
| 9b16f2a2dc76c609a933acb5 | B | modular | gurobi | 5 | 5 | 5 | 0 | 0 | 0 | 0.60245 | 0 |
| 9b16f2a2dc76c609a933acb5 | B | modular | highs | 4 | 4 | 4 | 0 | 0 | 0 | 0.65229 | 0 |
| 9b16f2a2dc76c609a933acb5 | C | legacy | gurobi | 5 | 5 | 5 | 0 | 0 | 0 | 0.80299 | 0 |
| 9b16f2a2dc76c609a933acb5 | C | legacy | highs | 7 | 7 | 7 | 0 | 0 | 0 | 0.80314 | 0 |
| 9b16f2a2dc76c609a933acb5 | C | modular | gurobi | 5 | 5 | 5 | 0 | 0 | 0 | 0.80215 | 0 |
| 9b16f2a2dc76c609a933acb5 | C | modular | highs | 7 | 7 | 7 | 0 | 0 | 0 | 0.80219 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 1 | legacy | gurobi | 60 | 60 | 52 | 0 | 0 | 0 | 3.4576 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 1 | legacy | highs | 66 | 49 | 49 | 17 | 0 | 0 | 1.2028 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 1 | modular | gurobi | 60 | 60 | 52 | 0 | 0 | 0 | 3.4569 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 1 | modular | highs | 66 | 49 | 49 | 17 | 0 | 0 | 1.2031 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 2 | legacy | gurobi | 64 | 57 | 44 | 7 | 0 | 0 | 10.415 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 2 | legacy | highs | 69 | 41 | 41 | 28 | 0 | 0 | 14.846 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 2 | modular | gurobi | 64 | 57 | 44 | 7 | 0 | 0 | 10.314 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 2 | modular | highs | 69 | 41 | 41 | 28 | 0 | 0 | 14.822 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 3 | legacy | gurobi | 63 | 11 | 2 | 52 | 0 | 0 | 371.19 | 0.0057158 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 3 | legacy | highs | 70 | 6 | 5 | 49 | 15 | 0 | 1481.5 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 3 | modular | gurobi | 63 | 11 | 2 | 52 | 0 | 0 | 371.09 | 0.0057158 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 3 | modular | highs | 70 | 6 | 5 | 49 | 15 | 0 | 1488.1 | 0 |

Medians describe checked feasible results only; failures and pending work remain in the counts. Reference gaps use published optima or checked exact PCSTP proofs within the same environment. Missing certificates or timings remain blank in CSV.

## API parity

Only pairs with the same input checksum, solver, seed, repetition and environment are compared. Runtime overhead uses matching, independently feasible, proven/reference-optimal pairs. `total_seconds` measures constructor plus solution call; wall time also includes process startup, imports and parsing.

| Environment | Solver | Pair status | Count | Median total delta s | Median total overhead % | Median wall delta s | Median wall overhead % |
|---|---|---|---:|---:|---:|---:|---:|
| 41e9ea36e64651742cafd58f | gurobi | match | 270 | — | — | — | — |
| 41e9ea36e64651742cafd58f | gurobi | match_optimal | 894 | 0.0025464 | 0.20743 | 1.5825e-05 | 0.00073343 |
| 41e9ea36e64651742cafd58f | gurobi | unavailable | 550 | — | — | — | — |
| 41e9ea36e64651742cafd58f | highs | match | 11 | — | — | — | — |
| 41e9ea36e64651742cafd58f | highs | match_optimal | 821 | 0.0014153 | 0.12858 | 5.451e-06 | 0.00043036 |
| 41e9ea36e64651742cafd58f | highs | unavailable | 863 | — | — | — | — |
| 9b16f2a2dc76c609a933acb5 | gurobi | match | 30 | — | — | — | — |
| 9b16f2a2dc76c609a933acb5 | gurobi | match_optimal | 108 | 0.003352 | 0.19349 | 8.4373e-05 | 0.0028476 |
| 9b16f2a2dc76c609a933acb5 | gurobi | unavailable | 59 | — | — | — | — |
| 9b16f2a2dc76c609a933acb5 | highs | match | 1 | — | — | — | — |
| 9b16f2a2dc76c609a933acb5 | highs | match_optimal | 106 | 0.0013603 | 0.13543 | 2.1839e-05 | 0.00095121 |
| 9b16f2a2dc76c609a933acb5 | highs | unavailable | 109 | — | — | — | — |

Objective/feasibility mismatches: **0**. All flags are retained per measurement in `results.csv.gz`.

Interpretation: a median total overhead above both 5% and 0.01 s is a descriptive signal to investigate. This is not a significance test; repeated measurements and a representative completed dataset are required before concluding meaningful API overhead. Incomplete or cross-environment pairs do not support timing conclusions.

[Download all per-case results and certificates](results.csv.gz) (gzip-compressed CSV).
