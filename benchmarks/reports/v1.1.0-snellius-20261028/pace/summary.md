# v1.1.0 — pace

Mode: **full**. Commit: `ef7265ee646ebd9f6b47635d68a2a4b7ee590200`. Generated: 2026-11-04T16:28:12.504974+00:00.

Expected: 1797. Finished: 1797. Pending: 0.

Each environment ID identifies its recorded software, hardware and limits; environments are never pooled. See `../environment.json` and `../environments/`.

| Environment | Dataset/track | Method | Solver | Expected | Feasible | Proven/reference optimal | Timeout | Failed/unavailable | Pending | Median wall s | Median reference gap % |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 41e9ea36e64651742cafd58f | PACE2018 / 1 | exact | gurobi | 179 | 160 | 134 | 13 | 6 | 0 | 2.6555 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 1 | exact | highs | 179 | 119 | 119 | 40 | 20 | 0 | 2.1065 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 1 | heuristic | highs | 179 | 179 | 16 | 0 | 0 | 0 | 1.4038 | 4.2553 |
| 41e9ea36e64651742cafd58f | PACE2018 / 2 | exact | gurobi | 179 | 170 | 117 | 7 | 2 | 0 | 14.475 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 2 | exact | highs | 179 | 119 | 116 | 14 | 46 | 0 | 13.531 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 2 | heuristic | highs | 179 | 177 | 3 | 1 | 1 | 0 | 2.1042 | 2.6916 |
| 41e9ea36e64651742cafd58f | PACE2018 / 3 | exact | gurobi | 176 | 42 | 20 | 114 | 20 | 0 | 677.33 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 3 | exact | highs | 175 | 11 | 10 | 118 | 46 | 0 | 296.84 | 0 |
| 41e9ea36e64651742cafd58f | PACE2018 / 3 | heuristic | highs | 176 | 117 | 0 | 1 | 58 | 0 | 17.842 | 1.4443 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 1 | exact | gurobi | 21 | 20 | 18 | 1 | 0 | 0 | 6.512 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 1 | exact | highs | 21 | 14 | 14 | 3 | 4 | 0 | 10.223 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 1 | heuristic | highs | 21 | 21 | 0 | 0 | 0 | 0 | 1.7042 | 3.367 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 2 | exact | gurobi | 21 | 18 | 11 | 2 | 1 | 0 | 7.718 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 2 | exact | highs | 21 | 13 | 13 | 4 | 4 | 0 | 14.847 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 2 | heuristic | highs | 21 | 21 | 1 | 0 | 0 | 0 | 2.8051 | 1.3971 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 3 | exact | gurobi | 23 | 4 | 0 | 15 | 4 | 0 | 1799.1 | 0.00026559 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 3 | exact | highs | 24 | 1 | 1 | 16 | 7 | 0 | 1338.4 | 0 |
| 9b16f2a2dc76c609a933acb5 | PACE2018 / 3 | heuristic | highs | 23 | 16 | 0 | 0 | 7 | 0 | 13.923 | 6.411 |

Medians describe checked feasible results only; failures and pending work remain in the counts. Reference gaps use published optima or checked exact PCSTP proofs within the same environment. Missing certificates or timings remain blank in CSV.

Track 2 tree-decomposition files are not consumed by SteinerPy. Published PACE bounds are quality references, not same-hardware timing measurements.

[Download all per-case results and certificates](results.csv.gz) (gzip-compressed CSV).
