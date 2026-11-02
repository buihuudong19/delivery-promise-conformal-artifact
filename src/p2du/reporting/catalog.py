"""Index of every output and a digest of the main results, written for the person who writes the manuscript.

  outputs/manuscript/TABLE_INDEX.md     every table in outputs/tables: what it holds, its columns and size, the RQ
                                        and the figure / manuscript table it feeds
  outputs/manuscript/RESULTS_DIGEST.md  the main numbers of each RQ in plain sentences, each with the table it comes
                                        from and the LaTeX macro to cite (outputs/manuscript/results_macros.tex)
"""
import numpy as np
import pandas as pd

DESC = {
    "sample_flow.csv": ("Data", "Sample-flow counts (rules R1-R3)", "Table 4"),
    "eda_promise_checks.csv": ("Phase 1", "Checks of the estimated delivery date (resolution, within-cell correlations with post-purchase delays), late rate, slack", "Text 3.1"),
    "eda_censoring_by_origin.csv": ("Phase 1", "Share of calibration orders undelivered at each origin", "Text 3.3"),
    "eda_censoring_by_age.csv": ("Phase 1", "Share undelivered by purchase age at the origin", "Fig. 3b"),
    "eda_survey_timing.csv": ("Phase 1", "Survey creation and answer relative to delivery and promise, late vs on time", "Text 4.2, Fig. 11"),
    "eda_region_profile.csv": ("Phase 1", "Delivery time, promise, late and low-rating rates by region", "Table 4"),
    "eda_monthly.csv": ("Phase 1", "Monthly delivery time, promise, late and low-rating rates", "Fig. 2a"),
    "eda_S_by_region.csv": ("Phase 1", "Distribution of the deviation S by region", "Fig. 2b"),
    "rq1_window_metrics.csv": ("RQ1", "Coverage, promise length, slack by origin x model variant x method x alpha x region (Olist included)", "Figs. 4-7, Tables 5-6"),
    "rq1_summary.csv": ("RQ1", "Window averages of rq1_window_metrics; worst-region coverage and region gap", "Fig. 4, Tables 5-6"),
    "rq1_pinball.csv": ("RQ1", "Pinball loss and raw coverage of the uncorrected quantiles by origin x variant x tau", "Fig. 8"),
    "rq1_pinball_summary.csv": ("RQ1", "Window averages of rq1_pinball", "Fig. 8"),
    "rq1_censoring_by_origin.csv": ("RQ1", "Training censoring, calibration observed share, IPCW weights, tail share by origin", "Text 3.3"),
    "rq1_aci_alpha_trajectory.csv": ("RQ1", "Daily ACI level alpha_t by variant and alpha", "Fig. 6b"),
    "rq1_window_metrics_promise.csv": ("RQ1 sens.", "RQ1 metrics of the models with Olist's estimate as an input", "Table 9"),
    "rq1_pinball_promise.csv": ("RQ1 sens.", "Pinball loss of the models with Olist's estimate as an input", "Table 9"),
    "rq1_raw_matched.csv": ("RQ1", "Per window and member of the quantile average: mean uncalibrated quantile promise at Olist's coverage (levels of the main fit and HIGH_TAUS)", "Table S1"),
    "rq1_route_shrinkage.csv": ("RQ1", "Route table (IPCW) with shrinkage m = 20, 50 (main), 100: coverage, length, worst region by target; length at Olist's coverage", "Table S2"),
    "rq1_matched_olist.csv": ("RQ1", "Per window: mean promise each model needs to reach Olist's coverage (and coverage at Olist's length)", "Fig. 4a, Table 5"),
    "rq1_tuning_choices.csv": ("RQ1", "Hyper-parameters chosen at every origin and whether each lies on the edge of its grid", "Table 11 (C13), Section 4"),
    "rq1_ensemble_selection.csv": ("RQ1", "Pre-specified choice of the quantile average: promise at Olist's coverage of each candidate member set in 2017 (selection) and 2018 (test)", "Table 5 note, Table 11 (C12)"),
    "rq1_summary_by_period.csv": ("RQ1", "Window means for 2017 (ACI step chosen) vs 2018 (test), and stress vs normal windows", "Table 6, Section 6.1"),
    "rq1_window_types.csv": ("RQ1", "Stress windows (Olist late rate >= 2 x median) and period of every origin", "Fig. 6, Table 6"),
    "eda_drift_weekly.csv": ("Phase 1", "Weekly deliveries, purchases, late rate, promise and delivery time (Carnival 2018 backlog)", "Text 5.1"),
    "rq2_aipw_overlap_sensitivity.csv": ("RQ2 sens.", "AIPW under propensity clips 0.005-0.05 and trimming", "Table 7 note"),
    "rq3_duality_gap.csv": ("RQ3", "Lagrangian dual bound vs feasible repair for L = 14..32 (Proposition 5, Algorithm 3)", "Fig. A.1b, Table 11"),
    "rq3_curve_calibration.csv": ("RQ3", "Out-of-time check: r(s) fitted on 2017 vs observed low-rating rate per window", "Fig. A.1a, Table 11"),
    "contrasts_rq1.csv": ("RQ1", "Paired window-level contrasts with block-bootstrap CI, Holm and HAC p-values", "Table A.1"),
    "rq2_crude_curve.csv": ("RQ2", "Crude low-rating rate by integer S with Wilson CI", "Fig. 9"),
    "rq2_crude_curve_by_answer_timing.csv": ("RQ2", "Crude curve split by answer before/after arrival", "Fig. 11"),
    "rq2_crude_curve_by_region.csv": ("RQ2", "Crude curve by region", "not reported (exploratory)"),
    "rq2_survey_mechanism.csv": ("RQ2", "Low-rating rate and shares by late x answered-before-arrival", "Text 4.2"),
    "rq2_s_versus_d.csv": ("RQ2", "Cross-validated log loss / AIC of f(D), f(S), with and without g(P)", "Fig. 10b, Table 7 note"),
    "rq2_surface_P_D.csv": ("RQ2", "Low-rating rate by promise bin x delivery-time bin", "Fig. 10a"),
    "rq2_aipw_bins.csv": ("RQ2", "AIPW risks, RD, RR, E-values, overlap by deviation bin (LR and XGBoost nuisances)", "Fig. 12a, Table 7"),
    "rq2_aipw_bins_by_region.csv": ("RQ2", "AIPW by region (LR nuisances)", "Fig. 12b"),
    "rq2_response_curve.csv": ("RQ2", "Monotone g-formula curve r(s), overall and by region (input of RQ3)", "Fig. 9, RQ3"),
    "rq2_response_curve_fit2017.csv": ("RQ2 sens.", "r(s) fitted on 2017 orders only", "Table 10"),
    "rq2_response_curve_neutral0.csv": ("RQ2 sens.", "r(s) with three-star ratings counted as not low", "Table 10"),
    "rq2_rdd.csv": ("RQ2", "RDD at S = 0 (and placebo cut-offs): estimate, SE, honest CI by bandwidth and M", "Fig. 12c"),
    "rq3_policy_windows.csv": ("RQ3", "Every policy x parameter x window: mean promise, realised/expected late and low-rating rates", "Fig. 13"),
    "rq3_frontier.csv": ("RQ3", "Window averages of rq3_policy_windows (pooled frontier)", "Fig. 13"),
    "rq3_matched.csv": ("RQ3", "Each policy at Olist's mean promise of the window; promise length at Olist's low-rating rate", "Table 8, Fig. 14c"),
    "rq3_region_promises.csv": ("RQ3", "Mean promise and outcomes by region under Olist, coverage and optimal policies", "Fig. 14b"),
    "rq3_convexity.csv": ("RQ3", "Share of orders with non-increasing / convex rho_i(P) (Lemma 4, Remark 7)", "Table 11"),
    "rq3_rho_examples.csv": ("RQ3", "rho_i(P) of six example orders", "Fig. 14a"),
    "rq3_policy_windows_fit2017.csv": ("RQ3 sens.", "Policies chosen with r(s) fitted on 2017 orders", "Table 10"),
    "rq3_matched_fit2017.csv": ("RQ3 sens.", "Matched comparisons with r(s) fitted on 2017 orders", "Table 10"),
    "rq3_policy_windows_neutral0.csv": ("RQ3 sens.", "Policies with three-star ratings as not low", "Table 10"),
    "rq3_matched_neutral0.csv": ("RQ3 sens.", "Matched comparisons with three-star ratings as not low", "Table 10"),
    "contrasts_rq3.csv": ("RQ3", "Paired contrasts of the policies against Olist", "Table 8"),
    "sensitivity_rq1.csv": ("Sens.", "Sensitivity analyses of RQ1", "Table 9"),
    "sensitivity_rq3.csv": ("Sens.", "Sensitivity analyses of RQ3", "Table 10"),
    "rq3_policy_windows_cpd_ens.csv": ("RQ3 sens.", "Policies with the quantile average of RQ1 as the base model of the CPD", "Table 10"),
    "rq3_matched_cpd_ens.csv": ("RQ3 sens.", "Matched comparisons with the quantile-average CPD", "Table 10"),
    **{f"rq3_policy_windows_tail{d}.csv": ("RQ3 sens.", f"Policies with the unplaced predictive mass at {d} days (main 49)", "Table 9B")
       for d in (42, 56, 70, 120)},
    **{f"rq3_matched_tail{d}.csv": ("RQ3 sens.", f"Matched comparisons with the unplaced predictive mass at {d} days", "Table 9B")
       for d in (42, 56, 70, 120)},
    "rq1_calibration_weights.csv": ("RQ1", "Calibration set size, delivered orders, sum and Kish effective size of the IPCW weights by origin", "Remark 3"),
    "rq2_lee_bounds.csv": ("RQ2 sens.", "Lee (2009) bounds on the jump in the low-rating rate at the promised day among always after-arrival responders", "Text 6.3"),
    "contrasts_rq3_sensitivity.csv": ("RQ3 sens.", "Paired contrasts of the rating-optimal policy under each RQ3 setting (all windows, 2018 test) and of the quantile-average against the forest CPD", "Table 10"),
    "theory_checks.csv": ("Checks", "Checks of the formal results (Propositions 1-5, Lemmas 1-4, Corollary 1) and of the premises on the data", "Table 11"),
    "theory_check_ipcw_cdf.csv": ("Checks", "IPCW vs naive vs oracle Pr(Y <= t) on each calibration set", "Table 11"),
}


def write_index(out):
    t = out / "tables"; m = out / "manuscript"; m.mkdir(parents=True, exist_ok=True)
    lines = ["# Index of the output tables", "",
             "Generated by the pipeline (stage `catalog`). Every number of the article comes from one of these files "
             "through `results_macros.tex` or the tables in this folder.", "",
             "| File | Part | Content | Used in | Rows | Columns |", "|---|---|---|---|---|---|"]
    for p in sorted(t.glob("*.csv")):
        part, content, used = DESC.get(p.name, ("", "(no description)", ""))
        try:
            df = pd.read_csv(p, nrows=200000)
            n, cols = len(df), ", ".join(df.columns[:14]) + (" ..." if df.shape[1] > 14 else "")
        except Exception:
            n, cols = "", ""
        lines.append(f"| `{p.name}` | {part} | {content} | {used} | {n} | {cols} |")
    lines += ["", "Other outputs:", "",
              "- `outputs/figures/fig*.pdf|png`: the 14 figures (see docs/DE_XUAT_14_FIGURES.md).",
              "- `outputs/manuscript/table*.tex|csv`: ready-to-use tables of the article.",
              "- `outputs/manuscript/results_macros.tex`: LaTeX macros with every number quoted in the text.",
              "- `outputs/data/`: feature file, per-origin model fits (`parts/`), per-order promises.",
              "- `outputs_extra/`: independent cross-checks from the raw CSV files (tools/crosscheck.py).",
              "- `outputs/manifest.json`: SHA-256 of every output, code hashes, library versions."]
    (m / "TABLE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _f(x, nd=1, pct=False):
    return f"{100 * x:.{nd}f}%" if pct else f"{x:.{nd}f}"


def write_digest(cfg, out):
    t = out / "tables"; m = out / "manuscript"
    L = ["# Digest of the main results", "",
         "Generated by the pipeline (stage `catalog`); regenerate after every run. Source table in brackets.", ""]
    P = pd.read_csv(t / "eda_promise_checks.csv").set_index("check")["value"]
    fl = pd.read_csv(t / "sample_flow.csv")
    L += ["## Data and Phase-1 checks", "",
          f"- Sample: {int(fl.iloc[2]['n_orders']):,} orders after the rules, {int(fl.iloc[3]['n_orders']):,} delivered [sample_flow].",
          f"- Olist's promise: mean {_f(P['promise P in days: mean'])} days; late rate {_f(P['late rate (S > 0) among delivered orders'], 1, True)}; "
          f"median slack of on-time orders {_f(P['median slack P - D among on-time orders (days)'], 0)} days [eda_promise_checks].",
          f"- Within seller x route x month cells the promise correlates {P['within seller-route-month correlation of P with handover_d']:.3f} "
          f"with the hand-over delay and {P['within seller-route-month correlation of P with approval_h']:.3f} with the approval delay "
          "(a promise fixed at purchase cannot track later delays; a small residual correlation remains) [eda_promise_checks].", ""]
    S = pd.read_csv(t / "rq1_summary.csv"); S = S[S.region == "all"]
    o = S[S.variant == "olist"].iloc[0]
    v = cfg["rq3"]["model"]
    def g(var, meth, a, col):
        x = S[(S.variant == var) & (S.method == meth) & np.isclose(S.alpha, a)]
        return float(x[col].iloc[0]) if len(x) else np.nan
    L += ["## RQ1", "",
          f"- Olist: coverage {_f(o['coverage'], 1, True)}, mean promise {_f(o['mean_promise'])} d, worst region {_f(o['worst_region'], 1, True)} [rq1_summary].",
          f"- {v}, target 90%: split naive {_f(g(v, 'split_naive', 0.1, 'coverage'), 1, True)}, split IPCW {_f(g(v, 'split_ipcw', 0.1, 'coverage'), 1, True)}, "
          f"oracle {_f(g(v, 'split_oracle', 0.1, 'coverage'), 1, True)}, Mondrian IPCW {_f(g(v, 'mondrian_ipcw', 0.1, 'coverage'), 1, True)}, "
          f"ACI {_f(g(v, 'aci_ipcw', 0.1, 'coverage'), 1, True)}; mean promise split IPCW {_f(g(v, 'split_ipcw', 0.1, 'mean_promise'))} d, "
          f"ACI {_f(g(v, 'aci_ipcw', 0.1, 'mean_promise'))} d [rq1_summary].",
          f"- {v}, target 95%: split IPCW coverage {_f(g(v, 'split_ipcw', 0.05, 'coverage'), 1, True)} at {_f(g(v, 'split_ipcw', 0.05, 'mean_promise'))} d "
          f"(Olist {_f(o['coverage'], 1, True)} at {_f(o['mean_promise'])} d) [rq1_summary]."]
    C = pd.read_csv(t / "contrasts_rq1.csv")
    for fam, lab, sc in (("RQ1-A", "IPCW minus naive calibration, coverage (pp)", 100),
                         ("RQ1-E", "worst-region coverage gain over split (pp)", 100)):
        x = C[C.family.str.startswith(fam) & C.contrast.str.contains(v)]
        for _, r in x.iterrows():
            L.append(f"- {lab}, {r['contrast']}: {r['mean'] * sc:.2f} [{r['ci_lo'] * sc:.2f}, {r['ci_hi'] * sc:.2f}], Holm p {r['p_holm']:.4f} [contrasts_rq1].")
    for fam, lab in (("RQ1-F", "mean promise at Olist's coverage minus Olist (days)"),
                     ("RQ1-I", "mean promise at Olist's coverage minus Olist (days)"),
                     ("RQ1-H", "quantile average minus member, promise at Olist's coverage (days)")):
        x = C[C.family.str.startswith(fam)]
        for _, r in x.iterrows():
            per = ", 2018 test" if "2018" in r["family"] else ""
            L.append(f"- {lab}{per}, {r['contrast']}: {r['mean']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}], Holm p {r['p_holm']:.4f} [contrasts_rq1].")
    if (t / "rq1_ensemble_selection.csv").is_file():
        Es = pd.read_csv(t / "rq1_ensemble_selection.csv")
        L.append("- Quantile average, selection on 2017 (promise at Olist's coverage minus Olist, days): "
                 + "; ".join(f"{r['candidate']} ({r['members']}): 2017 {r['length_gap_2017']:+.2f}, 2018 {r['length_gap_2018']:+.2f}"
                             + (" [selected]" if r["selected"] else "") for _, r in Es.iterrows()) + " [rq1_ensemble_selection].")
    if (t / "rq1_summary_by_period.csv").is_file():
        Bp = pd.read_csv(t / "rq1_summary_by_period.csv")
        for grp in ("2017 (selection)", "2018 (test)", "normal", "stress"):
            x = Bp[(Bp.variant == v) & np.isclose(Bp.alpha, 0.10) & (Bp.group == grp)].set_index("method")
            L.append(f"- {v}, {grp}: " + "; ".join(f"{mm} cov {_f(x.loc[mm, 'coverage'], 1, True)} (min {_f(x.loc[mm, 'coverage_min'], 1, True)}), "
                     f"len {_f(x.loc[mm, 'mean_promise'])} d" for mm in ("split_naive", "split_ipcw", "mondrian_ipcw", "aci_ipcw")
                     if mm in x.index) + " [rq1_summary_by_period].")
        Wt = pd.read_csv(t / "rq1_window_types.csv")
        L.append(f"- Stress windows (Olist late rate >= {100 * Wt['threshold'].iloc[0]:.1f}%): origins "
                 + ", ".join(str(pd.Timestamp(x).date()) for x in Wt.loc[Wt.stress, "origin"]) + " [rq1_window_types].")
    L.append("")
    A = pd.read_csv(t / "rq2_aipw_bins.csv"); A = A[A.learner == "xgb"]
    Mm = pd.read_csv(t / "rq2_survey_mechanism.csv")
    late = Mm[Mm.late == 1].set_index("answered_before_delivery")
    RD = pd.read_csv(t / "rq2_rdd.csv"); RD = RD[np.isclose(RD.M, RD.M_rule_of_thumb)]
    L += ["## RQ2", ""]
    for _, r in A.iterrows():
        L.append(f"- {r['label']}: AIPW risk {_f(r['risk'], 1, True)} (crude {_f(r['crude'], 1, True)}), RR {r['rr']:.2f} "
                 f"[{r['rr_lo']:.2f}, {r['rr_hi']:.2f}], E-value {r['e_value']:.2f} [rq2_aipw_bins].")
    L.append(f"- Late orders rated before arrival: {_f(late.loc[1, 'share_within_late_group'], 1, True)}; their low-rating rate "
             f"{_f(late.loc[1, 'mean'], 1, True)} vs {_f(late.loc[0, 'mean'], 1, True)} after arrival [rq2_survey_mechanism].")
    for sub in ("all", "answered_after_arrival"):
        x = RD[(RD.subset == sub) & (RD.cutoff == 0)]
        L.append(f"- RDD at S = 0 ({sub}): " + "; ".join(f"h={int(r['h'])}: {100 * r['estimate']:.1f} pp "
                 f"[{100 * r['ci_lo']:.1f}, {100 * r['ci_hi']:.1f}]" for _, r in x.iterrows()) + " [rq2_rdd].")
    V = pd.read_csv(t / "rq2_s_versus_d.csv").set_index("model")
    L += ["- CV log loss: " + "; ".join(f"{k} {V.loc[k, 'cv_logloss']:.4f}" for k in V.index) + " [rq2_s_versus_d].", ""]
    Q = pd.read_csv(t / "rq3_matched.csv"); C3 = pd.read_csv(t / "contrasts_rq3.csv")
    L += ["## RQ3", "",
          f"- Olist: mean promise {Q['L_olist'].mean():.2f} d, realised low-rating rate {_f(Q['low_olist'].mean(), 2, True)}, "
          f"late {_f(Q['late_olist'].mean(), 2, True)} [rq3_matched]."]
    for _, r in C3.iterrows():
        sc = 1 if "length" in r["family"] else 100
        L.append(f"- {r['family']}, {r['contrast']}: {r['mean'] * sc:.2f} [{r['ci_lo'] * sc:.2f}, {r['ci_hi'] * sc:.2f}], "
                 f"Holm p {r['p_holm']:.4f} [contrasts_rq3].")
    if (t / "contrasts_rq3_sensitivity.csv").is_file():
        L.append("- RQ3 sensitivity (contrasts_rq3_sensitivity):")
        for _, r in pd.read_csv(t / "contrasts_rq3_sensitivity.csv").iterrows():
            sc = 1 if "length" in r["family"] else 100
            L.append(f"  - {r['family']}, {r['contrast']}: {r['mean'] * sc:.2f} [{r['ci_lo'] * sc:.2f}, {r['ci_hi'] * sc:.2f}], "
                     f"Holm p {r['p_holm']:.4f}.")
    Cv = pd.read_csv(t / "rq3_convexity.csv")
    L += [f"- Lemma 4 / Remark 7 premises: non-increasing rho_i in {_f(Cv['share_nonincreasing'].mean(), 1, True)} of orders, convex in "
          f"{_f(Cv['share_convex'].mean(), 1, True)} [rq3_convexity].", ""]
    Th = pd.read_csv(t / "theory_checks.csv")
    L += ["## Checks against the theory", ""] + [f"- {r['check']} {r['status']}: {r['claim']} -> {r['value']}" for _, r in Th.iterrows()]
    (m / "RESULTS_DIGEST.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_catalog(cfg, out):
    write_index(out)
    write_digest(cfg, out)
