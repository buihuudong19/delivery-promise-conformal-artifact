# Pilot run of 31 October 2026 (before the reported run)

Outputs of `tools/pilot_tuning_ensemble.py`, which is not part of the pipeline. The pilot asked whether a wider tuning
grid, or averaging the stored quantiles of several models, shortens the conformal promise at the same coverage.

| File | Content |
|---|---|
| `pilot_A_ensemble.csv`, `pilot_A_matched.csv` | Part A: quantile averages of the stored predictions of the earlier run, at all 30 origins (no refit) |
| `pilot_B_grid.csv`, `pilot_B.log` | Part B: wider grids for XGBoost QR, the quantile forest and the AFT at six origins (k = 5, 10, 15, 20, 25, 30) |

The pilot used origins of both 2017 and 2018 (k = 20, 25 and 30 are 2018 origins). After it, the quantile average with
three nested candidate sets and the wider grids were added to `config.yaml`, and the member set was chosen on the 2017
origins by the rule stated there (check C12). The 2018 comparisons of the article are therefore not a strict hold-out,
and the article reports the results of the quantile average as exploratory.
