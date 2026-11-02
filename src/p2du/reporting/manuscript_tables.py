"""Ready-to-use tables of the article (outputs/manuscript/table*.tex with booktabs, and the same content as .csv).

Every cell is computed from outputs/tables; nothing is typed by hand. Intervals are 95% moving-block bootstrap
intervals over the evaluation windows; p-values are Holm-adjusted within each family (reporting/contrasts.py).

  table1_sample        sample flow and delivery profile by customer region
  table2_rq1_models    coverage, mean promise and worst-region coverage of every model (split conformal, IPCW)
  table3_rq1_methods   conformal methods compared for two models at alpha = 0.10
  table4_rq1_contrasts selected paired contrasts of RQ1
  table5_rq2           AIPW risks and risk ratios by deviation bin with E-values; S-versus-D model comparison
  table6_rq3           promise policies at Olist's mean promise length, with paired contrasts
  table7_sensitivity   sensitivity analyses
  table8_theory        checks of the propositions and premises on the data
"""
import numpy as np
import pandas as pd

KL = {"route": "Route table", "lqr": "Linear QR", "xgbq": "XGBoost QR", "qrf": "Quantile forest", "aft": "XGBoost AFT",
      "ens": "Quantile average",
      "olist": "Olist promise"}
ML = {"raw": "Uncorrected quantile", "split_naive": "Split, naive", "split_ipcw": "Split, IPCW",
      "split_oracle": "Split, oracle (infeasible)", "mondrian_naive": "Mondrian, naive", "mondrian_ipcw": "Mondrian, IPCW",
      "aci_ipcw": "ACI, IPCW", "mondrian_aci_ipcw": "Mondrian ACI, IPCW", "actual": "--"}
PL = {"olist": "Olist promise", "uniform": "Uniform promise", "route": "Route quantile",
      "cpd_quantile": "Coverage policy (CPD quantile)", "late_optimal": "Late-rate optimal",
      "optimal": "Rating-aware optimal", "olist_shift": "Olist promise shifted"}
REG_EN = {"Sudeste": "Southeast", "Sul": "South", "Centro-Oeste": "Centre-West", "Nordeste": "Northeast", "Norte": "North"}


def _vlabel(v):
    k, _, m = v.partition("_")
    if v == "aft":
        return "XGBoost AFT (censored labels)"
    if v == "aft_naive":
        return "XGBoost AFT (delivered only)"
    if k == "ens":
        return "Quantile average (5 models)"
    return KL[k] + ("" if not m else f" ({m.upper() if m == 'ipcw' else m})")


def _tex(df, path, caption, label, notes="", colfmt=None):
    cols = list(df.columns)
    colfmt = colfmt or ("l" + "r" * (len(cols) - 1))
    esc = lambda x: str(x).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{colfmt}}}", r"\toprule", " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(esc(x) for x in r.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines += [r"\par\smallskip", rf"\parbox{{\linewidth}}{{\scriptsize {esc(notes)}}}"]
    lines.append(r"\end{table}")
    path.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def _pct(x, nd=1):
    return "--" if pd.isna(x) else f"{100 * x:.{nd}f}"


def _ci(r, sc=1.0, nd=2):
    return f"{r['mean'] * sc:.{nd}f} [{r['ci_lo'] * sc:.{nd}f}, {r['ci_hi'] * sc:.{nd}f}]"


def _p(p):
    return "--" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def write_tables(cfg, out):
    t = out / "tables"; m = out / "manuscript"; m.mkdir(parents=True, exist_ok=True)
    # ---- Table 1
    fl = pd.read_csv(t / "sample_flow.csv")
    rp = pd.read_csv(t / "eda_region_profile.csv").set_index("customer_region").reindex(list(REG_EN))
    T1 = pd.DataFrame([dict(Region=REG_EN[g], Orders=f"{int(r['n']):,}", **{"Median Y (d)": f"{r['Y_median']:.1f}",
                       "90th pct. Y (d)": f"{r['Y_p90']:.1f}", "Mean promise (d)": f"{r['P_mean']:.1f}",
                       "Late (%)": _pct(r["late"]), "Low rating (%)": _pct(r["low"])}) for g, r in rp.iterrows()])
    notes = "Sample flow: " + "; ".join(f"{s.strip()}: {int(n):,}" for s, n in zip(fl["step"], fl["n_orders"])) + "."
    _tex(T1, m / "table1_sample", "Delivered orders and delivery profile by customer region.", "tab:sample", notes)
    # ---- Table 2
    S = pd.read_csv(t / "rq1_summary.csv"); S = S[S.region == "all"]
    rows = []
    T2V = ["route_plain", "route_ipcw", "lqr_plain", "lqr_ipcw", "xgbq_plain", "xgbq_ipcw", "qrf_plain", "qrf_ipcw",
           "aft_naive", "aft"]
    ens = (cfg.get("ensemble") or {}).get("name")
    if ens and (S.variant == ens).any():
        T2V.append(ens)
    for v in T2V:
        r = dict(Model=_vlabel(v))
        for a in cfg["alphas"]:
            x = S[(S.variant == v) & (S.method == "split_ipcw") & np.isclose(S.alpha, a)].iloc[0]
            r[f"Cov. {int(100 * (1 - a))}%"] = _pct(x["coverage"]); r[f"Len. {int(100 * (1 - a))}%"] = f"{x['mean_promise']:.1f}"
            r[f"Worst reg. {int(100 * (1 - a))}%"] = _pct(x["worst_region"])
        rows.append(r)
    Cf = pd.read_csv(t / "contrasts_rq1.csv")
    Ff = Cf[Cf.family.str.startswith("RQ1-F")].set_index("contrast")
    Fi = Cf[Cf.family.str.startswith("RQ1-I")].set_index("contrast")
    for r, v in zip(rows, T2V):
        key = f"{v} - Olist"
        r["Len. at Olist's cov. - Olist (d) [95% CI]"] = _ci(Ff.loc[key]) if key in Ff.index else "--"
        r["Same, 2018 test (d) [95% CI]"] = _ci(Fi.loc[key]) if key in Fi.index else "--"
    o = S[S.variant == "olist"].iloc[0]
    rows.append({"Model": "Olist promise", **{k: "" for k in rows[0] if k != "Model"}})
    rows[-1].update({"Cov. 95%": _pct(o["coverage"]), "Len. 95%": f"{o['mean_promise']:.1f}", "Worst reg. 95%": _pct(o["worst_region"])})
    _tex(pd.DataFrame(rows), m / "table2_rq1_models",
         "Coverage (promises kept, %), mean promise (days) and mean worst-region coverage (%) after one-sided split "
         "conformal correction with IPCW calibration, by model and target coverage (30 rolling origins).", "tab:rq1models",
         "Olist's promise has no target level; its values are shown in the 95% columns for comparison. The last column "
         "is the mean promise each model needs to reach Olist's realised coverage of the window, minus Olist's mean "
         "promise (matched ex post; block-bootstrap interval over the 30 windows); the next column restricts it to the "
         "14 test windows of 2018, after the 2017 selection of the quantile-average members. Quantile average: the mean of the "
         "log-quantiles of the route table (IPCW), linear QR (IPCW), XGBoost QR (IPCW), quantile forest (IPCW) and "
         "XGBoost AFT (censored labels), corrected by the same conformal step; the member set was chosen on the 2017 "
         "origins by a rule fixed in advance (rq1_ensemble_selection.csv).")
    # ---- Table 3
    Bp = pd.read_csv(t / "rq1_summary_by_period.csv")
    v = cfg["rq3"]["model"]
    rows = []
    for meth in ["raw", "split_naive", "split_ipcw", "split_oracle", "mondrian_naive", "mondrian_ipcw", "aci_ipcw",
                 "mondrian_aci_ipcw"]:
        r = dict(Method=ML[meth])
        for grp, lab in (("2017 (selection)", "2017"), ("2018 (test)", "2018"), ("stress", "stress windows")):
            x = Bp[(Bp.variant == v) & (Bp.method == meth) & np.isclose(Bp.alpha, 0.10) & (Bp.group == grp)]
            if len(x):
                x = x.iloc[0]
                r[f"{lab}: cov."] = _pct(x["coverage"]); r[f"{lab}: min."] = _pct(x["coverage_min"])
                r[f"{lab}: worst reg."] = _pct(x["worst_region"]); r[f"{lab}: len."] = f"{x['mean_promise']:.1f}"
        rows.append(r)
    _tex(pd.DataFrame(rows), m / "table3_rq1_methods",
         f"Conformal methods for the {KL[v.split('_')[0]]} at target coverage 90%: mean coverage, lowest window "
         "coverage, mean worst-region coverage (%) and mean promise (days), for the 2017 origins (where the ACI step "
         "was chosen), the 2018 origins (test period) and the stress windows.", "tab:rq1methods",
         "Stress windows: Olist's late rate at least twice its median over the windows (Black Friday 2017 and the "
         "weeks after Carnival 2018).")
    # ---- Table 4
    C = pd.read_csv(t / "contrasts_rq1.csv")
    keep = C[C.family.str.startswith(("RQ1-A", "RQ1-B", "RQ1-E")) |
             (C.family.str.startswith("RQ1-D") & C.family.str.contains("alpha=0.05|alpha=0.1$", regex=True))]
    keep = keep[~keep.contrast.str.contains("_plain")]
    T4 = pd.DataFrame([dict(Family=r["family"], Contrast=r["contrast"],
                            **{"Mean [95% CI]": _ci(r, 100 if "coverage" in r["family"] else 1),
                               "Holm p": _p(r["p_holm"]), "HAC p": _p(r["p_hac"])}) for _, r in keep.iterrows()])
    _tex(T4, m / "table4_rq1_contrasts", "Paired window-level contrasts of RQ1 (coverage in percentage points, "
         "length in days).", "tab:rq1contrasts", colfmt="llrrr")
    # ---- Table 5
    A = pd.read_csv(t / "rq2_aipw_bins.csv")
    rows = []
    for b, g in A.groupby("bin"):
        lr, xg = g[g.learner == "lr"].iloc[0], g[g.learner == "xgb"].iloc[0]
        rows.append({"Deviation S (days)": xg["label"], "Orders": f"{int(xg['n']):,}", "Crude (%)": _pct(xg["crude"]),
                     "AIPW-LR (%)": _pct(lr["risk"]), "AIPW-XGB (%)": _pct(xg["risk"]),
                     "RR, XGB [95% CI]": "ref." if pd.isna(xg["e_value"]) else f"{xg['rr']:.2f} [{xg['rr_lo']:.2f}, {xg['rr_hi']:.2f}]",
                     "E-value (CI)": "--" if pd.isna(xg["e_value"]) else f"{xg['e_value']:.2f} ({xg['e_value_ci']:.2f})"})
    V = pd.read_csv(t / "rq2_s_versus_d.csv").set_index("model")
    notes = "Cross-validated log loss: " + "; ".join(f"{k} {V.loc[k, 'cv_logloss']:.4f}" for k in V.index) + "."
    if (t / "rq2_aipw_overlap_sensitivity.csv").is_file():
        Ov = pd.read_csv(t / "rq2_aipw_overlap_sensitivity.csv"); Ov = Ov[(Ov.learner == "xgb") & Ov.e_value.notna()]
        rng = Ov.groupby("label")["rr"].agg(["min", "max"])
        notes += " Risk ratios (XGBoost) across propensity clips 0.005-0.05 and trimming: " + "; ".join(
            f"{k} {r['min']:.2f}-{r['max']:.2f}" for k, r in rng.iterrows()) + "."
    _tex(pd.DataFrame(rows), m / "table5_rq2", "Low-rating probability by deviation from the promise: crude rate, "
         "doubly robust (AIPW) risks with logistic and XGBoost nuisance models, risk ratios against S in -5..0, and "
         "E-values.", "tab:rq2", notes)
    # ---- Table 6
    Q = pd.read_csv(t / "rq3_matched.csv"); C3 = pd.read_csv(t / "contrasts_rq3.csv")
    rows = [dict(Policy="Olist promise", **{"Low rating, r(s)-implied (%)": _pct(Q["low_olist"].mean(), 2),
                                            "Late (%)": _pct(Q["late_olist"].mean(), 2),
                                            "Mean promise at Olist's low-rating rate (d)": f"{Q['L_olist'].mean():.2f}",
                                            "Low rating vs Olist (pp) [95% CI]": "--", "Holm p": "--"})]
    for fam in ("uniform", "route", "cpd_quantile", "late_optimal", "optimal"):
        c = C3[(C3.contrast == f"{fam} - Olist") & C3.family.str.contains("low-rating rate at")].iloc[0]
        rows.append({"Policy": PL[fam], "Low rating, r(s)-implied (%)": _pct(Q[f"{fam}_low_realised_at_L"].mean(), 2),
                     "Late (%)": _pct(Q[f"{fam}_late_realised_at_L"].mean(), 2),
                     "Mean promise at Olist's low-rating rate (d)": f"{Q[f'{fam}_L_at_low'].mean():.2f}",
                     "Low rating vs Olist (pp) [95% CI]": _ci(c, 100), "Holm p": _p(c["p_holm"])})
    _tex(pd.DataFrame(rows), m / "table6_rq3", f"Promise policies at Olist's mean promise length "
         f"({Q['L_olist'].mean():.1f} days on average over the windows): low-rating rate implied by the response curve "
         "of RQ2 at the actual delivery times, late rate at the actual delivery times, and the mean promise each policy "
         "needs to reach Olist's low-rating rate.", "tab:rq3",
         "The low-rating rates are model-implied: they combine the observed delivery times with the estimated response "
         "curve r(s). The late rates are observed.")
    # ---- Table 7
    if (t / "sensitivity_rq1.csv").is_file():
        Sn = pd.read_csv(t / "sensitivity_rq1.csv")
        Sn = Sn[np.isclose(Sn.alpha, 0.10)]
        T7 = pd.DataFrame([{"Analysis": r["analysis"], "Setting": r["setting"], "Windows": r.get("period", "all windows"),
                            "Model": KL[r["variant"].split("_")[0]],
                            "Coverage (%)": _pct(r["coverage"]), "Lowest window (%)": _pct(r["coverage_min_window"]),
                            "Worst region (%)": _pct(r["worst_region"]), "Mean promise (d)": f"{r['mean_promise']:.1f}",
                            "At cap (%)": "--" if pd.isna(r.get("share_capped", np.nan)) else _pct(r["share_capped"])}
                           for _, r in Sn.iterrows()])
        _tex(T7, m / "table7_sensitivity", "Sensitivity analyses of RQ1 at target coverage 90%.", "tab:sens",
             "At cap: share of evaluation orders whose conformal bound is not finite, so that the promise is set at the "
             f"cap of {cfg['conformal']['promise_cap_days']} days. A calibration window of K days follows each order for "
             "at most K days, so the upper quantile of the delivery time is identified only when K exceeds it (the 90% "
             "quantile of the delivery time is about 24 days); shorter windows therefore return capped promises, which "
             "motivates the main window of 42 days.", colfmt="llllrrrrr")
    # ---- Table 7b: RQ3 sensitivity (predictive distribution and response curve)
    if (t / "contrasts_rq3_sensitivity.csv").is_file():
        Cs = pd.read_csv(t / "contrasts_rq3_sensitivity.csv")
        rows = []
        for lab in ("forest CPD (main)", "quantile-average CPD", "policy from the 2017 curve", "three-star ratings as not low",
                    "tail mass at 42 days", "tail mass at 56 days", "tail mass at 70 days", "tail mass at 120 days"):
            g = lambda fam: Cs[(Cs.contrast == f"optimal - Olist: {lab}") & Cs.family.str.contains(fam, regex=False)]
            if g("(all windows)").empty:
                continue
            lo, lt, la, ln = (g("low-rating rate at Olist's mean promise (all windows)").iloc[0],
                              g("low-rating rate at Olist's mean promise (2018 test)").iloc[0],
                              g("late rate at Olist's mean promise").iloc[0], g("promise length").iloc[0])
            rows.append({"Setting": lab, "Low rating vs Olist, all windows (pp)": _ci(lo, 100),
                         "Low rating vs Olist, 2018 (pp)": _ci(lt, 100), "Late vs Olist (pp)": _ci(la, 100),
                         "Promise at Olist's low rate vs Olist (d)": _ci(ln, 1), "Holm p (all windows)": _p(lo["p_holm"])})
        notes = ("Rating-optimal policy at Olist's mean promise length in each window; paired differences over windows with "
                 "moving-block bootstrap intervals; Holm adjustment within each column family. Quantile-average CPD: the "
                 "predictive distribution is built from the quantile average of RQ1 instead of the quantile forest. Policy "
                 "from the 2017 curve: promises chosen with the response curve fitted on 2017 orders, evaluated with the "
                 "full-sample curve. Tail mass at d days: the predictive mass that the distribution cannot place (the tail "
                 "share and the mass beyond 120 days) is put at d days instead of 49.")
        E = Cs[Cs.family.str.contains("versus forest CPD")]
        if len(E):
            notes += " Quantile-average minus forest CPD: " + "; ".join(
                f"{r['contrast'].split(':')[0]}, {'late' if 'late rate' in r['family'] else 'low-rating'} rate "
                f"{_ci(r, 100)} pp (Holm p {_p(r['p_holm'])})" for _, r in E.iterrows()) + "."
        _tex(pd.DataFrame(rows), m / "table7b_sensitivity_rq3", "Sensitivity analyses of RQ3: the rating-optimal "
             "policy against Olist's promise under alternative predictive distributions and response curves.",
             "tab:sens-rq3", notes, colfmt="lrrrrr")
    # ---- Table 8
    Th = pd.read_csv(t / "theory_checks.csv")
    _tex(Th[["check", "claim", "value", "status"]].rename(columns=str.capitalize), m / "table8_theory",
         "Checks of the propositions and of the premises of the research questions on the data.", "tab:theory",
         colfmt="lp{5.5cm}p{6cm}l")
