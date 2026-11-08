# v1.1.0 — steinlib

Mode: **full**. Commit: `ef7265ee646ebd9f6b47635d68a2a4b7ee590200`. Generated: 2026-11-04T16:28:15.671830+00:00.

Expected: 266. Finished: 266. Pending: 0.

Each environment ID identifies its recorded software, hardware and limits; environments are never pooled. See `../environment.json` and `../environments/`.

| Environment | Dataset/track | Method | Solver | Expected | Feasible | Proven/reference optimal | Timeout | Failed/unavailable | Pending | Median wall s | Median reference gap % |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 41e9ea36e64651742cafd58f | B | dual_ascent | gurobi | 16 | 16 | 16 | 0 | 0 | 0 | 0.60206 | 0 |
| 41e9ea36e64651742cafd58f | B | dual_ascent | highs | 16 | 16 | 16 | 0 | 0 | 0 | 0.60225 | 0 |
| 41e9ea36e64651742cafd58f | B | exact | gurobi | 16 | 16 | 16 | 0 | 0 | 0 | 0.60241 | 0 |
| 41e9ea36e64651742cafd58f | B | exact | highs | 16 | 16 | 16 | 0 | 0 | 0 | 0.60228 | 0 |
| 41e9ea36e64651742cafd58f | B | heuristic | highs | 16 | 16 | 12 | 0 | 0 | 0 | 0.60194 | 0 |
| 41e9ea36e64651742cafd58f | B | kou | none | 16 | 16 | 5 | 0 | 0 | 0 | 0.60212 | 1.5495 |
| 41e9ea36e64651742cafd58f | B | mehlhorn | none | 16 | 16 | 5 | 0 | 0 | 0 | 0.50206 | 1.8697 |
| 41e9ea36e64651742cafd58f | C | dual_ascent | gurobi | 17 | 17 | 17 | 0 | 0 | 0 | 1.403 | 0 |
| 41e9ea36e64651742cafd58f | C | dual_ascent | highs | 17 | 16 | 16 | 1 | 0 | 0 | 1.5032 | 0 |
| 41e9ea36e64651742cafd58f | C | exact | gurobi | 17 | 17 | 17 | 0 | 0 | 0 | 1.403 | 0 |
| 41e9ea36e64651742cafd58f | C | exact | highs | 18 | 17 | 17 | 1 | 0 | 0 | 1.6037 | 0 |
| 41e9ea36e64651742cafd58f | C | heuristic | highs | 17 | 17 | 2 | 0 | 0 | 0 | 1.3042 | 1.5559 |
| 41e9ea36e64651742cafd58f | C | kou | none | 18 | 18 | 1 | 0 | 0 | 0 | 0.80258 | 4.8492 |
| 41e9ea36e64651742cafd58f | C | mehlhorn | none | 18 | 18 | 0 | 0 | 0 | 0 | 0.6024 | 4.2301 |
| 9b16f2a2dc76c609a933acb5 | B | dual_ascent | gurobi | 2 | 2 | 2 | 0 | 0 | 0 | 0.60223 | 0 |
| 9b16f2a2dc76c609a933acb5 | B | dual_ascent | highs | 2 | 2 | 2 | 0 | 0 | 0 | 0.60237 | 0 |
| 9b16f2a2dc76c609a933acb5 | B | exact | gurobi | 2 | 2 | 2 | 0 | 0 | 0 | 0.60218 | 0 |
| 9b16f2a2dc76c609a933acb5 | B | exact | highs | 2 | 2 | 2 | 0 | 0 | 0 | 0.60228 | 0 |
| 9b16f2a2dc76c609a933acb5 | B | heuristic | highs | 2 | 2 | 2 | 0 | 0 | 0 | 0.60193 | 0 |
| 9b16f2a2dc76c609a933acb5 | B | kou | none | 2 | 2 | 0 | 0 | 0 | 0 | 0.55214 | 5.9214 |
| 9b16f2a2dc76c609a933acb5 | B | mehlhorn | none | 2 | 2 | 0 | 0 | 0 | 0 | 0.55205 | 7.0578 |
| 9b16f2a2dc76c609a933acb5 | C | dual_ascent | gurobi | 3 | 3 | 3 | 0 | 0 | 0 | 0.80307 | 0 |
| 9b16f2a2dc76c609a933acb5 | C | dual_ascent | highs | 3 | 3 | 3 | 0 | 0 | 0 | 0.80294 | 0 |
| 9b16f2a2dc76c609a933acb5 | C | exact | gurobi | 3 | 3 | 3 | 0 | 0 | 0 | 0.90235 | 0 |
| 9b16f2a2dc76c609a933acb5 | C | exact | highs | 2 | 2 | 2 | 0 | 0 | 0 | 0.75256 | 0 |
| 9b16f2a2dc76c609a933acb5 | C | heuristic | highs | 3 | 3 | 2 | 0 | 0 | 0 | 0.90271 | 0 |
| 9b16f2a2dc76c609a933acb5 | C | kou | none | 2 | 2 | 0 | 0 | 0 | 0 | 0.65233 | 3.8317 |
| 9b16f2a2dc76c609a933acb5 | C | mehlhorn | none | 2 | 2 | 1 | 0 | 0 | 0 | 0.60197 | 3.2609 |

Medians describe checked feasible results only; failures and pending work remain in the counts. Reference gaps use published optima or checked exact PCSTP proofs within the same environment. Missing certificates or timings remain blank in CSV.

[Download all per-case results and certificates](results.csv.gz) (gzip-compressed CSV).
