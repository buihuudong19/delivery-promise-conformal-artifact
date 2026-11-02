# Independent cross-checks (outputs_extra)

Written by the stage `crosscheck` (2026-11-01 19:04). Each number is recomputed from the raw Olist CSV files with plain pandas (src/p2du/reporting/crosscheck.py imports nothing from the pipeline code) and compared with the pipeline tables.

Result: 364 numbers compared, 0 mismatches, largest absolute difference 3.55e-15.

| Check | What | Numbers | Mismatches | Max abs. diff. |
|---|---|---|---|---|
| C-01 | all orders in the orders table | 1 | 0 | 0.00e+00 |
| C-02 | orders after rules R1-R2 | 1 | 0 | 0.00e+00 |
| C-03 | delivered orders | 1 | 0 | 0.00e+00 |
| C-04 | Olist's mean promise | 1 | 0 | 0.00e+00 |
| C-05 | Olist's late rate | 1 | 0 | 2.78e-17 |
| C-06 | late rate, mean promise, low-rating rate by region | 15 | 0 | 5.55e-17 |
| C-07 | RQ2 deviation bins: size and crude low-rating rate | 12 | 0 | 5.55e-17 |
| C-08 | late orders rated before arrival | 2 | 0 | 0.00e+00 |
| C-09 | RQ1: Olist's coverage and mean promise per window | 60 | 0 | 3.55e-15 |
| C-10 | RQ1: stored per-order promises of the main model and the quantile average re-scored on the raw delivery dates | 240 | 0 | 3.55e-15 |
| C-11 | RQ3: Olist's realised low-rating rate per window | 30 | 0 | 9.71e-17 |

Files: `tables/crosscheck_results.csv` (every comparison), `tables/crosscheck_summary.csv`, `results_extra.tex` (macros).
