"""Families of paired window-level contrasts of the article (block bootstrap, Holm within each family, HAC check).

  RQ1-A  calibration weighting: split_ipcw - split_naive (coverage), per model, alpha = 0.10
  RQ1-B  training weights: *_ipcw - *_plain and aft - aft_naive (coverage and mean promise, split_ipcw), alpha = 0.10
  RQ1-C  promise length of each model against the route benchmark (route_ipcw), split_ipcw, by alpha
  RQ1-D  promise length and coverage of each model (Mondrian IPCW) against Olist's promise, by alpha
  RQ1-E  worst-region coverage: Mondrian, ACI and Mondrian-ACI against split, per model, alpha = 0.10
  RQ1-F  mean promise each model needs (split, IPCW) to reach Olist's coverage of the window, minus Olist's mean
         promise (rq1_matched_olist.csv)
  RQ1-G  test period only (origins in 2018, after the ACI step was chosen on 2017): coverage, lowest-window
         behaviour (per-window coverage), worst-region coverage and length of ACI, Mondrian and Mondrian-ACI against
         split, alpha = 0.10
  RQ1-H  quantile average (config `ensemble`) against each of its members: mean promise at Olist's coverage
         (rq1_matched_olist.csv), all windows and the 2018 test windows
  RQ1-I  RQ1-F restricted to the 2018 test windows
  RQ1-J  uncalibrated quantile promise minus split-IPCW promise at Olist's coverage (rq1_raw_matched.csv and
         rq1_matched_olist.csv), per member of the quantile average and the average; windows where the uncalibrated
         levels do not reach Olist's coverage are left out (n_windows)
  RQ1-K  RQ1-F restricted to the 16 windows of 2017 (selection period)
  RQ1-L  tail term: split IPCW minus split IPCW with the tail share set to 0 (coverage and mean promise), alpha = 0.10
  RQ3    realised low-rating rate of each policy at Olist's mean promise minus Olist's rate; promise length of each
         policy at Olist's low-rating rate minus Olist's length
  RQ3-Y  the RQ3 families split by period (2017 selection, 2018 test)
  RQ3-R  region budgets: the rating-aware policy solved within each customer region at Olist's regional mean promise
  RQ3-X  checks of the response curve (contrasts_rq3_sensitivity.csv): curve cross-fitted on two halves of the rated
         orders, policy scored with the AIPW step curve, curve fitted on ratings answered after arrival
  contrasts_block_sensitivity.csv  key contrasts recomputed with moving blocks of 2, 3, 5 and 6 windows
"""
import copy
import numpy as np
import pandas as pd

from .inference import contrast_row, adjust

ML = ("lqr", "xgbq", "qrf")


def _piv(M, method, alpha, col="coverage", region="all"):
    X = M[(M.method == method) & (np.isclose(M.alpha, alpha)) & (M.region == region)]
    return X.pivot_table(index="k", columns="variant", values=col)


def _worst(M, method, alpha):
    X = M[(M.method == method) & (np.isclose(M.alpha, alpha)) & (M.region != "all")]
    return X.groupby(["k", "variant"])["coverage"].min().unstack()


def run_contrasts(cfg, out):
    t = out / "tables"
    M = pd.read_csv(t / "rq1_window_metrics.csv")
    rows = []
    a = 0.10
    ci, cn = _piv(M, "split_ipcw", a), _piv(M, "split_naive", a)
    for v in ci.columns:
        rows.append(contrast_row("RQ1-A calibration weighting (coverage)", f"{v}: IPCW - naive", (ci[v] - cn[v]).to_numpy(), cfg))
    mp = _piv(M, "split_ipcw", a, "mean_promise")
    for k in ("route",) + ML:
        rows.append(contrast_row("RQ1-B training weights (coverage)", f"{k}: ipcw - plain",
                                 (ci[f"{k}_ipcw"] - ci[f"{k}_plain"]).to_numpy(), cfg))
        rows.append(contrast_row("RQ1-B training weights (mean promise)", f"{k}: ipcw - plain",
                                 (mp[f"{k}_ipcw"] - mp[f"{k}_plain"]).to_numpy(), cfg))
    rows.append(contrast_row("RQ1-B training weights (coverage)", "aft: censored - naive", (ci["aft"] - ci["aft_naive"]).to_numpy(), cfg))
    rows.append(contrast_row("RQ1-B training weights (mean promise)", "aft: censored - naive", (mp["aft"] - mp["aft_naive"]).to_numpy(), cfg))
    ol = M[(M.variant == "olist") & (M.region == "all")].set_index("k")
    for al in cfg["alphas"]:
        p = _piv(M, "split_ipcw", al, "mean_promise")
        for v in [c for c in p.columns if c not in ("route_ipcw", "route_plain")]:
            rows.append(contrast_row(f"RQ1-C promise length vs route benchmark, alpha={al}", f"{v} - route_ipcw",
                                     (p[v] - p["route_ipcw"]).to_numpy(), cfg))
        pm = _piv(M, "mondrian_ipcw", al, "mean_promise"); cm = _piv(M, "mondrian_ipcw", al)
        for v in pm.columns:
            rows.append(contrast_row(f"RQ1-D promise length vs Olist, alpha={al}", f"{v} - Olist",
                                     (pm[v] - ol["mean_promise"].reindex(pm.index)).to_numpy(), cfg))
            rows.append(contrast_row(f"RQ1-D coverage vs Olist, alpha={al}", f"{v} - Olist",
                                     (cm[v] - ol["coverage"].reindex(cm.index)).to_numpy(), cfg))
    ws = _worst(M, "split_ipcw", a)
    for meth in ("mondrian_ipcw", "aci_ipcw", "mondrian_aci_ipcw"):
        wm = _worst(M, meth, a)
        for v in wm.columns:
            rows.append(contrast_row("RQ1-E worst-region coverage", f"{v}: {meth} - split_ipcw", (wm[v] - ws[v]).to_numpy(), cfg))
    if (t / "rq1_matched_olist.csv").is_file():
        Mo = pd.read_csv(t / "rq1_matched_olist.csv")
        for v, g in Mo.groupby("variant"):
            rows.append(contrast_row("RQ1-F promise length at Olist's coverage (days)", f"{v} - Olist",
                                     g.sort_values("k")["length_gap"].to_numpy(), cfg))
    ens = (cfg.get("ensemble") or {}).get("name")
    if ens and (t / "rq1_matched_olist.csv").is_file():
        Pm = Mo.pivot_table(index="k", columns="variant", values="length_gap")
        Wt0 = pd.read_csv(t / "rq1_window_types.csv") if (t / "rq1_window_types.csv").is_file() else None
        k18 = set(Wt0.loc[Wt0["period"].str.startswith("2018"), "k"]) if Wt0 is not None else set()
        if ens in Pm.columns:
            for m in cfg["ensemble"]["members"]:
                d = (Pm[ens] - Pm[m])
                rows.append(contrast_row("RQ1-H quantile average vs members, promise length at Olist's coverage (days)",
                                         f"{ens} - {m}", d.to_numpy(), cfg))
                if k18:
                    rows.append(contrast_row("RQ1-H quantile average vs members, 2018 test (days)",
                                             f"{ens} - {m}", d[d.index.isin(k18)].to_numpy(), cfg))
        if k18:      # RQ1-I: RQ1-F on the 2018 test windows only (the ensemble members were selected on 2017)
            for v in Pm.columns:
                rows.append(contrast_row("RQ1-I promise length at Olist's coverage, 2018 test (days)", f"{v} - Olist",
                                         Pm[v][Pm.index.isin(k18)].to_numpy(), cfg))
            for v in Pm.columns:     # RQ1-K: RQ1-F on the 2017 selection windows only
                rows.append(contrast_row("RQ1-K promise length at Olist's coverage, 2017 selection (days)", f"{v} - Olist",
                                         Pm[v][~Pm.index.isin(k18)].to_numpy(), cfg))
    if (t / "rq1_raw_matched.csv").is_file() and (t / "rq1_matched_olist.csv").is_file():
        Rw = pd.read_csv(t / "rq1_raw_matched.csv").pivot_table(index="k", columns="variant", values="raw_length_at_olist_coverage")
        Lm = pd.read_csv(t / "rq1_matched_olist.csv").pivot_table(index="k", columns="variant", values="length_at_olist_coverage")
        for v in Rw.columns:
            if v in Lm.columns:
                rows.append(contrast_row("RQ1-J uncalibrated minus split-IPCW promise at Olist's coverage (days)",
                                         f"{v}: raw - split_ipcw", (Rw[v] - Lm[v].reindex(Rw.index)).to_numpy(), cfg))
    if (t / "rq1_window_types.csv").is_file():
        Wt = pd.read_csv(t / "rq1_window_types.csv")
        test_k = set(Wt.loc[Wt["period"].str.startswith("2018"), "k"])
        Mt = M[M.k.isin(test_k)]
        cs, ls, ws = _piv(Mt, "split_ipcw", a), _piv(Mt, "split_ipcw", a, "mean_promise"), _worst(Mt, "split_ipcw", a)
        for meth in ("mondrian_ipcw", "aci_ipcw", "mondrian_aci_ipcw"):
            cm, lm, wm = _piv(Mt, meth, a), _piv(Mt, meth, a, "mean_promise"), _worst(Mt, meth, a)
            for v in cm.columns:
                rows.append(contrast_row("RQ1-G test period 2018 (coverage)", f"{v}: {meth} - split_ipcw",
                                         (cm[v] - cs[v]).to_numpy(), cfg))
                rows.append(contrast_row("RQ1-G test period 2018 (worst-region coverage)", f"{v}: {meth} - split_ipcw",
                                         (wm[v] - ws[v]).to_numpy(), cfg))
                rows.append(contrast_row("RQ1-G test period 2018 (mean promise)", f"{v}: {meth} - split_ipcw",
                                         (lm[v] - ls[v]).to_numpy(), cfg))
    if (M.method == "split_ipcw_notail").any():          # RQ1-L: what the tail term adds to IPCW calibration
        c0, l0 = _piv(M, "split_ipcw_notail", a), _piv(M, "split_ipcw_notail", a, "mean_promise")
        for v in ci.columns:
            rows.append(contrast_row("RQ1-L tail term (coverage), IPCW with tail minus IPCW without", f"{v}: tail - no tail",
                                     (ci[v] - c0[v]).to_numpy(), cfg))
        for v in mp.columns:
            rows.append(contrast_row("RQ1-L tail term (mean promise), IPCW with tail minus IPCW without", f"{v}: tail - no tail",
                                     (mp[v] - l0[v]).to_numpy(), cfg))
    R = adjust(pd.DataFrame(rows))
    R.to_csv(t / "contrasts_rq1.csv", index=False)
    if (t / "rq3_matched.csv").is_file():
        Q = pd.read_csv(t / "rq3_matched.csv")
        rows = []
        for fam in ("uniform", "route", "cpd_quantile", "late_optimal", "optimal", "olist_shift"):
            rows.append(contrast_row("RQ3 low-rating rate at Olist's mean promise", f"{fam} - Olist",
                                     (Q[f"{fam}_low_realised_at_L"] - Q["low_olist"]).to_numpy(), cfg))
            rows.append(contrast_row("RQ3 late rate at Olist's mean promise", f"{fam} - Olist",
                                     (Q[f"{fam}_late_realised_at_L"] - Q["late_olist"]).to_numpy(), cfg))
            rows.append(contrast_row("RQ3 promise length at Olist's low-rating rate", f"{fam} - Olist",
                                     (Q[f"{fam}_L_at_low"] - Q["L_olist"]).to_numpy(), cfg))
        rows.append(contrast_row("RQ3 rating-aware versus coverage policies at Olist's mean promise", "optimal - cpd_quantile",
                                 (Q["optimal_low_realised_at_L"] - Q["cpd_quantile_low_realised_at_L"]).to_numpy(), cfg))
        rows.append(contrast_row("RQ3 rating-aware versus coverage policies at Olist's mean promise", "optimal - late_optimal",
                                 (Q["optimal_low_realised_at_L"] - Q["late_optimal_low_realised_at_L"]).to_numpy(), cfg))
        test = (pd.to_datetime(Q["origin"]) >= pd.Timestamp("2018-01-01")).to_numpy()
        for per, sel in (("2017 selection", ~test), ("2018 test", test)):
            for fam in ("uniform", "route", "cpd_quantile", "late_optimal", "optimal", "olist_shift"):
                rows.append(contrast_row(f"RQ3-Y low-rating rate at Olist's mean promise ({per})", f"{fam} - Olist",
                                         (Q[f"{fam}_low_realised_at_L"] - Q["low_olist"]).to_numpy()[sel], cfg))
                rows.append(contrast_row(f"RQ3-Y late rate at Olist's mean promise ({per})", f"{fam} - Olist",
                                         (Q[f"{fam}_late_realised_at_L"] - Q["late_olist"]).to_numpy()[sel], cfg))
                rows.append(contrast_row(f"RQ3-Y promise length at Olist's low-rating rate ({per})", f"{fam} - Olist",
                                         (Q[f"{fam}_L_at_low"] - Q["L_olist"]).to_numpy()[sel], cfg))
        if (t / "rq3_matched_regionbudget.csv").is_file():
            G = pd.read_csv(t / "rq3_matched_regionbudget.csv").set_index("k").loc[Q["k"]]
            fam = "RQ3-R region budgets at Olist's regional mean promise"
            rows.append(contrast_row(fam, "low-rating rate: region budget - Olist",
                                     (G["optimal_low_realised_at_L"] - G["low_olist"]).to_numpy(), cfg))
            rows.append(contrast_row(fam, "late rate: region budget - Olist",
                                     (G["optimal_late_realised_at_L"] - G["late_olist"]).to_numpy(), cfg))
            rows.append(contrast_row(fam, "promise length at Olist's low-rating rate: region budget - Olist",
                                     (G["optimal_L_at_low"] - G["L_olist"]).to_numpy(), cfg))
            rows.append(contrast_row(fam, "low-rating rate: region budget - single budget",
                                     (G["optimal_low_realised_at_L"].to_numpy() - Q["optimal_low_realised_at_L"].to_numpy()), cfg))
        adjust(pd.DataFrame(rows)).to_csv(t / "contrasts_rq3.csv", index=False)
        rq3_sensitivity_contrasts(t, cfg, Q)
    block_sensitivity(cfg, out)


RQ3_CURVE_CHECKS = (("xfit", "rq3_matched_xfit.csv", "cross-fitted curve"),
                    ("evalaipw", "rq3_matched_evalaipw.csv", "scored with the AIPW step curve"),
                    ("after", "rq3_matched_after.csv", "curve from ratings after arrival"))

RQ3_SENS = (("main", "rq3_matched.csv", "forest CPD (main)"),
            ("cpd_ens", "rq3_matched_cpd_ens.csv", "quantile-average CPD"),
            ("fit2017", "rq3_matched_fit2017.csv", "policy from the 2017 curve"),
            ("neutral0", "rq3_matched_neutral0.csv", "three-star ratings as not low"),
            ("tail42", "rq3_matched_tail42.csv", "tail mass at 42 days"),
            ("tail56", "rq3_matched_tail56.csv", "tail mass at 56 days"),
            ("tail70", "rq3_matched_tail70.csv", "tail mass at 70 days"),
            ("tail120", "rq3_matched_tail120.csv", "tail mass at 120 days"))


def rq3_sensitivity_contrasts(t, cfg, Q):
    """RQ3 sensitivity families (contrasts_rq3_sensitivity.csv): the rating-optimal policy against Olist under each
    setting, over all windows and over the 2018 test windows, and the quantile-average CPD against the forest CPD.
    Holm adjustment within each family; the main families of contrasts_rq3.csv are not affected."""
    rows = []
    M = {}
    for key, f, lab in RQ3_SENS:
        if not (t / f).is_file():
            continue
        X = pd.read_csv(t / f); M[key] = X
        test = (pd.to_datetime(X["origin"]) >= pd.Timestamp("2018-01-01")).to_numpy()
        low = (X["optimal_low_realised_at_L"] - X["low_olist"]).to_numpy()
        late = (X["optimal_late_realised_at_L"] - X["late_olist"]).to_numpy()
        ln = (X["optimal_L_at_low"] - X["L_olist"]).to_numpy()
        rows.append(contrast_row("RQ3-S low-rating rate at Olist's mean promise (all windows)", f"optimal - Olist: {lab}", low, cfg))
        rows.append(contrast_row("RQ3-S low-rating rate at Olist's mean promise (2018 test)", f"optimal - Olist: {lab}", low[test], cfg))
        rows.append(contrast_row("RQ3-S late rate at Olist's mean promise (all windows)", f"optimal - Olist: {lab}", late, cfg))
        rows.append(contrast_row("RQ3-S promise length at Olist's low-rating rate (all windows)", f"optimal - Olist: {lab}", ln, cfg))
    if "main" in M and "cpd_ens" in M:
        A = M["main"].set_index("k"); B = M["cpd_ens"].set_index("k").loc[A.index]
        for col, fam in (("optimal_low_realised_at_L", "low-rating rate"), ("optimal_late_realised_at_L", "late rate")):
            rows.append(contrast_row(f"RQ3-S quantile-average versus forest CPD ({fam} at Olist's mean promise)",
                                     "optimal: quantile-average CPD - forest CPD", (B[col] - A[col]).to_numpy(), cfg))
        rows.append(contrast_row("RQ3-S quantile-average versus forest CPD (low-rating rate at Olist's mean promise)",
                                 "coverage policy: quantile-average CPD - forest CPD",
                                 (B["cpd_quantile_low_realised_at_L"] - A["cpd_quantile_low_realised_at_L"]).to_numpy(), cfg))
    for key, f, lab in RQ3_CURVE_CHECKS:            # separate families, so the RQ3-S families above are unchanged
        if not (t / f).is_file():
            continue
        X = pd.read_csv(t / f)
        test = (pd.to_datetime(X["origin"]) >= pd.Timestamp("2018-01-01")).to_numpy()
        low = (X["optimal_low_realised_at_L"] - X["low_olist"]).to_numpy()
        rows.append(contrast_row("RQ3-X low-rating rate at Olist's mean promise, curve checks (all windows)", f"optimal - Olist: {lab}", low, cfg))
        rows.append(contrast_row("RQ3-X low-rating rate at Olist's mean promise, curve checks (2018 test)", f"optimal - Olist: {lab}", low[test], cfg))
        rows.append(contrast_row("RQ3-X late rate at Olist's mean promise, curve checks (all windows)", f"optimal - Olist: {lab}",
                                 (X["optimal_late_realised_at_L"] - X["late_olist"]).to_numpy(), cfg))
        rows.append(contrast_row("RQ3-X promise length at Olist's low-rating rate, curve checks (all windows)", f"optimal - Olist: {lab}",
                                 (X["optimal_L_at_low"] - X["L_olist"]).to_numpy(), cfg))
    if rows:
        adjust(pd.DataFrame(rows)).to_csv(t / "contrasts_rq3_sensitivity.csv", index=False)


BLOCKS = (2, 3, 5, 6)


def block_sensitivity(cfg, out):
    """Key contrasts with moving blocks of 2, 3 (main), 5 and 6 windows (contrasts_block_sensitivity.csv). For each
    block length the Holm adjustment is redone within the original family, over the same contrasts as the main file."""
    t = out / "tables"
    C1 = pd.read_csv(t / "contrasts_rq1.csv")
    M = pd.read_csv(t / "rq1_window_metrics.csv")
    a = 0.10
    ci, cn = _piv(M, "split_ipcw", a), _piv(M, "split_naive", a)
    series = []                                       # (family, contrast, differences)
    for v in ci.columns:
        series.append(("RQ1-A calibration weighting (coverage)", f"{v}: IPCW - naive", (ci[v] - cn[v]).to_numpy()))
    if (t / "rq1_matched_olist.csv").is_file() and (t / "rq1_window_types.csv").is_file():
        Pm = pd.read_csv(t / "rq1_matched_olist.csv").pivot_table(index="k", columns="variant", values="length_gap")
        Wt = pd.read_csv(t / "rq1_window_types.csv")
        k18 = set(Wt.loc[Wt["period"].str.startswith("2018"), "k"])
        for v in Pm.columns:
            series.append(("RQ1-F promise length at Olist's coverage (days)", f"{v} - Olist", Pm[v].to_numpy()))
            series.append(("RQ1-I promise length at Olist's coverage, 2018 test (days)", f"{v} - Olist",
                           Pm[v][Pm.index.isin(k18)].to_numpy()))
    if (t / "rq3_matched.csv").is_file():
        Q = pd.read_csv(t / "rq3_matched.csv")
        test = (pd.to_datetime(Q["origin"]) >= pd.Timestamp("2018-01-01")).to_numpy()
        for fam in ("uniform", "route", "cpd_quantile", "late_optimal", "optimal", "olist_shift"):
            for nm, d in (("RQ3 low-rating rate at Olist's mean promise", Q[f"{fam}_low_realised_at_L"] - Q["low_olist"]),
                          ("RQ3 late rate at Olist's mean promise", Q[f"{fam}_late_realised_at_L"] - Q["late_olist"]),
                          ("RQ3 promise length at Olist's low-rating rate", Q[f"{fam}_L_at_low"] - Q["L_olist"])):
                series.append((nm, f"{fam} - Olist", d.to_numpy()))
                series.append((nm + " (2018 test)", f"{fam} - Olist", d.to_numpy()[test]))
    out_rows = []
    for b in BLOCKS:
        c = copy.deepcopy(cfg); c["bootstrap"]["block_len"] = b
        R = adjust(pd.DataFrame([contrast_row(f, lab, d, c) for f, lab, d in series]))
        out_rows.append(R.assign(block_len=b))
    B = pd.concat(out_rows, ignore_index=True)
    main = B[B.block_len == cfg["bootstrap"]["block_len"]].set_index(["family", "contrast"])
    B["sig_main"] = [bool(main.loc[(f, c), "p_holm"] < 0.05) for f, c in zip(B["family"], B["contrast"])]
    B["sig_here"] = B["p_holm"] < 0.05
    B["same_conclusion"] = (B["sig_main"] == B["sig_here"]) & (np.sign(B["mean"]) == np.sign(
        [main.loc[(f, c), "mean"] for f, c in zip(B["family"], B["contrast"])]))
    B.to_csv(t / "contrasts_block_sensitivity.csv", index=False)
