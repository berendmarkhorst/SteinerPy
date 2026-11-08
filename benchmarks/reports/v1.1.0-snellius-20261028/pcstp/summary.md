# v1.1.0 — pcstp

Mode: **full**. Commit: `ef7265ee646ebd9f6b47635d68a2a4b7ee590200`. Generated: 2026-11-04T16:28:15.708808+00:00.

Expected: 136. Finished: 136. Pending: 0.

Each environment ID identifies its recorded software, hardware and limits; environments are never pooled. See `../environment.json` and `../environments/`.

| Environment | Dataset/track | Method | Solver | Expected | Feasible | Proven/reference optimal | Timeout | Failed/unavailable | Pending | Median wall s | Median reference gap % |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 41e9ea36e64651742cafd58f | JMP | exact | gurobi | 29 | 29 | 22 | 0 | 0 | 0 | 1.7039 | 0 |
| 41e9ea36e64651742cafd58f | JMP | exact | highs | 29 | 29 | 27 | 0 | 0 | 0 | 2.3048 | 0 |
| 41e9ea36e64651742cafd58f | JMP | heuristic | highs | 29 | 29 | 1 | 0 | 0 | 0 | 0.70248 | 3.3609 |
| 41e9ea36e64651742cafd58f | JMP | pcst_fast | none | 32 | 32 | 0 | 0 | 0 | 0 | 0.60189 | 0.31526 |
| 9b16f2a2dc76c609a933acb5 | JMP | exact | gurobi | 5 | 5 | 2 | 0 | 0 | 0 | 6.0152 | 0 |
| 9b16f2a2dc76c609a933acb5 | JMP | exact | highs | 5 | 5 | 3 | 0 | 0 | 0 | 11.819 | 0 |
| 9b16f2a2dc76c609a933acb5 | JMP | heuristic | highs | 5 | 5 | 0 | 0 | 0 | 0 | 0.8023 | 3.4012 |
| 9b16f2a2dc76c609a933acb5 | JMP | pcst_fast | none | 2 | 2 | 0 | 0 | 0 | 0 | 0.60278 | 0 |

Medians describe checked feasible results only; failures and pending work remain in the counts. Reference gaps use published optima or checked exact PCSTP proofs within the same environment. Missing certificates or timings remain blank in CSV.

[Download all per-case results and certificates](results.csv.gz) (gzip-compressed CSV).
