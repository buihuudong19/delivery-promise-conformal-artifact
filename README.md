# Promising Delivery Dates under Uncertainty: code and outputs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23874459.svg)](https://doi.org/10.5281/zenodo.23874459)

Replication package for the article

> Dong Bui-Huu, Vinh Pham-An and Khuong Nguyen-An. *Promising Delivery Dates under Uncertainty: Conformalised
> Machine-Learning Models and Rating-Aware Promise Policies in E-Commerce.* Manuscript submitted to *Decision Support
> Systems*.

The repository contains the code that rebuilds the study sample from the public Olist data, runs every experiment and
check, and regenerates every table, figure and number reported in the article, together with the outputs of the
reported run. Release v1.0.1 is the version used in the article; it adds the robustness analyses listed below to
v1.0.0 and leaves every earlier output unchanged. The DOI
[10.5281/zenodo.23874459](https://doi.org/10.5281/zenodo.23874459) refers to all versions archived on Zenodo.

## Contents

```
config.yaml                 every parameter of the study
run_pipeline.py             entry point (run_pipeline.bat on Windows, run_pipeline.sh on Linux/macOS)
src/p2du/                   the pipeline (data, models, conformal calibration, experiments, reporting)
tests/test_p2du.py          unit tests: leakage guards and properties of the formal results
tools/                      manifest check, LaTeX macro generator, claim checker, pilot script
outputs/                    tables, figures, LaTeX tables and macros, and manifest of the reported run
outputs_extra/              independent recomputation of reported numbers from the raw files
data/README.md              how to obtain the Olist data (not redistributed)
docs/                       review of Decision Support Systems 2020-2026; evidence on the data and on the pilot run
Dockerfile                  container with the library versions of the reported run
```

## Study design

The article treats the delivery date shown at purchase as a decision under uncertainty, in three stages.

| Stage | Research question | Main outputs |
|---|---|---|
| 1. Promise bounds | RQ1: how censoring and drift affect conformal upper bounds on the delivery time | `rq1_summary.csv`, `rq1_window_metrics.csv`, `contrasts_rq1.csv` |
| 2. Rating response | RQ2: how the probability of a one- or two-star rating responds to the deviation from the promise | `rq2_aipw_bins.csv`, `rq2_response_curve.csv`, `rq2_rdd.csv`, `rq2_s_versus_d.csv` |
| 3. Promise policy | RQ3: which per-order promises minimise the expected low-rating rate at a given mean promise | `rq3_frontier.csv`, `rq3_matched.csv`, `contrasts_rq3.csv` |

- **Notation.** Y is the delivery time from midnight of the purchase day, D = floor(Y) the delivery day, P the promise
  in days and S = D - P the deviation. A promise is kept if D <= P, so an upper bound U on Y gives the promise
  floor(U).
- **Decision time.** Every feature is computed from information available at the purchase time (as-of histories);
  `tests/test_p2du.py` checks that no outcome field enters the features. Olist's own estimate is not an input in the
  main specification.
- **Rolling origins.** 30 origins every 14 days from 2017-06-01 to 2018-07-12. At origin o, the training set holds
  orders bought before o - 42 days, the calibration set the orders of the last 42 days with outcomes as known at o
  (heavily censored), and the evaluation set the orders of the next 14 days. The 16 origins of 2017 form the selection
  period and the 14 origins of 2018 the test period.
- **Models.** Olist's promise and a route quantile table as benchmarks; linear quantile regression, XGBoost quantile
  regression and a quantile regression forest, each trained on delivered orders only and with inverse probability of
  censoring weights (IPCW); an XGBoost accelerated failure time model trained on delivered orders or with censored
  labels; and a quantile average of five models.
- **Calibration.** One-sided split conformal calibration with a multiplicative score, Mondrian calibration by customer
  region, adaptive conformal inference with delayed feedback (level updated daily, clipped below at alpha/2) and
  Mondrian-ACI. Calibration is naive (delivered orders only) or uses IPCW masses with a tail term for the share of slow
  orders beyond the calibration window. An infeasible oracle calibration with final outcomes separates the effect of
  censoring from that of drift.
- **Rating response.** Crude curve, comparison of models in S and in D, augmented inverse probability weighting over
  deviation bins with logistic and XGBoost nuisance models (5-fold cross-fitting), a monotone g-formula curve r(s),
  a regression discontinuity at S = 0 with honest intervals for a discrete running variable, Lee bounds, E-values,
  and the timing of the satisfaction survey.
- **Promise policy.** A conformal predictive distribution of D for each order gives the expected low-rating rate
  rho_i(P) of every promise; a Lagrangian sweep minimises the mean of rho_i(P_i) subject to a constraint on the mean
  promise, with a duality certificate of near-optimality. Policies are compared at Olist's mean promise on the
  realised delivery days.
- **Inference.** Paired comparisons over windows with a moving-block bootstrap (blocks of 3 windows), Holm adjustment
  within each family of contrasts and Newey-West p-values as a check.
- **Robustness analyses (v1.0.1).** Contrasts over the 2017 windows (RQ1-K, RQ3-Y); IPCW calibration without the tail
  term (RQ1-L); response curves fitted on two random halves of the rated orders, a step curve of the AIPW risks and a
  curve from ratings answered after arrival to score or set the policy (RQ3-X); promise budgets held by region
  (RQ3-R); AIPW with three-star ratings counted as not low and with standard errors clustered by seller; block
  lengths of 2, 5 and 6 windows; tuning grids extended at their edges (check C13).

## Requirements

- Python 3.12 or later. The reported run used Python 3.14.4 on Windows 11.
- The libraries in `requirements-lock.txt` (exact versions of the reported run) or `requirements.txt` (minimum
  versions):

`run_pipeline.bat` and `run_pipeline.sh` create a virtual environment `.venv` and install `requirements-lock.txt`
(falling back to `requirements.txt`) on their first run. To set up the environment by hand:

```
python -m venv .venv
.venv\Scripts\activate          (Windows)      source .venv/bin/activate      (Linux/macOS)
pip install -r requirements-lock.txt
```

The `Dockerfile` builds a container with the same library versions (see the comments at its top).

## Data

The pipeline uses the *Brazilian E-Commerce Public Dataset by Olist* (Kaggle, CC BY-NC-SA 4.0,
DOI 10.34740/kaggle/dsv/195341). The data are not redistributed here; `data/README.md` explains how to place the nine
CSV files in `data/raw/`. The pipeline checks the SHA-256 of every table against `src/p2du/data/raw_sha256.json` and
stops if a file differs from the release used in the article.

## Running the pipeline

Windows:

```
run_pipeline.bat                       full run (the previous outputs are archived first)
run_pipeline.bat --resume              continue an interrupted run; fitted origins are kept
run_pipeline.bat --quick               smoke test (writes to outputs_quick, never to outputs)
run_pipeline.bat --stage figures       one stage, or a comma-separated list of stages
run_pipeline.bat --workers 4           number of parallel fitting processes
```

Linux/macOS: `./run_pipeline.sh` with the same arguments. On Windows the default number of fitting processes is set
from the cores and free memory of the machine; elsewhere it is `workers` in `config.yaml`.

The stages, in order, are `data, eda, rq1_fit, rq1, rq2, rq3, sens_fit, sensitivity, theory, inference, figures,
macros, tables, catalog, crosscheck, manifest`. The stage `rq1_fit` fits every model at the 30 origins and is the
longest; it writes one file per origin, so an interrupted run resumes where it stopped. The full run took 72
minutes with four parallel fitting processes, of which 59 minutes were model fitting; the run that added the
robustness analyses of v1.0.1 with the fitted models reused (`--resume`) took 31 minutes.

## Checking a copy against the reported run

```
python tools/verify_manifest.py        code and shipped outputs against outputs/manifest.json
python tests/test_p2du.py              22 unit tests (or: python -m pytest tests -q)
```

- `outputs/manifest.json` records the SHA-256 of every source file and output, the library versions, the platform
  and the seed of the reported run. `tools/verify_manifest.py` hashes the code (line endings normalised) and the
  shipped outputs and compares them with it. A continuous-integration job runs it on every push.
- `outputs/tables/theory_checks.csv` holds 13 checks of the formal results and of the premises of the research
  questions on the data.
- `outputs_extra/` recomputes 364 reported numbers from the raw CSV files with code that imports nothing from the
  pipeline, and compares them with the pipeline tables.
- `tools/check_claims.py` tests the qualitative statements of the article (for example "every contrast is
  significant") against the outputs of a run.

## Outputs and the article

| Article | File in `outputs/` |
|---|---|
| Table 4 (sample by region) | `manuscript/table1_sample.csv` |
| Table 6 (models after calibration) | `manuscript/table2_rq1_models.csv` |
| Table 7 (low-rating risk by deviation) | `manuscript/table5_rq2.csv` |
| Table 8 (promise policies) | `manuscript/table6_rq3.csv` |
| Fig. 1 to Fig. 3 | `figures/fig01_framework`, `fig02_data_context`, `fig03_protocol_censoring` |
| Fig. 4, Fig. 5 | `figures/fig05_rq1_censoring`, `fig06_rq1_time` |
| Fig. 6 to Fig. 8 | `figures/fig12_rq2_effects`, `fig09_rq2_response`, `fig11_rq2_survey` |
| Fig. 9, Fig. 10 | `figures/fig13_rq3_frontier`, `fig14_rq3_anatomy` |
| Table S1 (calibration methods by period) | `manuscript/table3_rq1_methods.csv` |
| Table S2 (uncalibrated quantiles) | `tables/contrasts_rq1.csv`, family RQ1-J |
| Table S3 (route table shrinkage) | `tables/sensitivity_rq1.csv` |
| Table S4 (sensitivity analyses of RQ1) | `manuscript/table7_sensitivity.csv` |
| Table S5 (IPCW with and without the tail term) | `tables/rq1_summary.csv` (method `split_ipcw_notail`), `tables/contrasts_rq1.csv`, family RQ1-L |
| Table S6 (sensitivity analyses of RQ3) | `manuscript/table7b_sensitivity_rq3.csv` |
| Table S7 (further checks of RQ2 and RQ3) | `tables/rq2_aipw_bins.csv`, `rq2_aipw_bins_neutral0.csv`, `contrasts_rq3.csv` (RQ3-Y, RQ3-R), `contrasts_rq3_sensitivity.csv` (RQ3-X) |
| Table S8 (checks of the formal results) | `manuscript/table8_theory.csv` |
| Table S9 (paired contrasts of RQ1) | `manuscript/table4_rq1_contrasts.csv` |
| Block-length sensitivity; extended tuning grids | `tables/contrasts_block_sensitivity.csv`; `tables/rq1_tuning_extended.csv`, `rq1_grid_extended.csv` |
| Fig. S1 to Fig. S6 | `figures/fig04_rq1_efficiency`, `fig07_rq1_regions`, `fig08_rq1_quantile_calib`, `figS1_aci_sensitivity`, `fig10_rq2_s_vs_d`, `figS2_rq3_checks` |

Figures are given as vector PDF and as PNG at 600 dpi. Every number quoted in the text of the article is a LaTeX macro
in `outputs/manuscript/results_macros.tex`; `tools/gen_results_tex.py` writes these macros and the table bodies used
by the LaTeX sources of the article (not included). `outputs/manuscript/RESULTS_DIGEST.md` summarises the main results
of each research question with their source tables, and `outputs/manuscript/TABLE_INDEX.md` describes every table.

## Notes on reproducibility

- XGBoost and the quantile forest use random subsampling whose streams differ between platforms, so a run on another
  platform changes some numbers slightly. A run of the same code on Linux kept every qualitative statement of the
  article that `tools/check_claims.py` tests.
- All random components use the seed in `config.yaml` (42).
- The models were fitted in the full run of 1 November 2026 (code hash `5c666ef3f936`). The robustness analyses of
  v1.0.1 were added on 2 November 2026 by a run of every stage with `--resume` (code hash `2d402718b4a7`), which reused
  the stored fits, followed by a run of the stages `sens_fit` to `manifest` after the check of the extended tuning
  grids gained a flag for diverged fits (code hash `39539d3a3843`, the released code, recorded in
  `outputs/manifest.json`). Every output of the first run kept its value; the code that fits the main models is the
  same in all runs.
- The quantile average and the wider tuning grids were introduced after a pilot run that used origins of both 2017
  and 2018 (`tools/pilot_tuning_ensemble.py`; outputs in `docs/evidence/pilot_tuning_2026-10-31`). The members of the
  quantile average were then chosen on the 2017 origins by the rule in `config.yaml` (check C12). The 2018 comparisons
  are therefore not a strict hold-out, and the article reports the quantile average as exploratory.
- Parts of the infrastructure (configuration loading, data input, order attributes, block bootstrap, Holm and HAC
  inference, rolling windows and the figure style) are reused from the code of an earlier article by the first
  author, released under the MIT licence; each reused file or function says so in its header.

## Citation

Please cite the article above. `CITATION.cff` gives the metadata of this software and of the article.

## Licence

The code is released under the MIT licence (`LICENSE`). The Olist data are distributed by Olist on Kaggle under
CC BY-NC-SA 4.0 and are not included.
