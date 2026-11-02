"""Checks that the computations match the formal results of the article (numbering of Manuscript-Draft.md and of the
LaTeX source) and the premises of the RQs, on the real
data. Each row of tables/theory_checks.csv states the claim, the measured value, the criterion and PASS / FAIL / INFO
(INFO = a premise that the data may or may not satisfy; it is reported, not enforced).

  C1  Proposition 1: Y <= U implies D <= floor(U), so a valid upper bound gives a kept promise
  C2  Proposition 2: the IPCW estimate of Pr(Y <= t) on each censored calibration set is closer to the
      final-snapshot (oracle) value than the naive estimate that drops undelivered orders; the origins where IPCW is
      still off are listed (its assumption, purchase time independent of delivery time, fails at demand peaks)
  C3  Lemma 2 (split-conformal validity under exchangeability): inside each evaluation window, random halves serve as
      calibration and test sets (exchangeable by construction); mean coverage >= 1 - alpha
  C4  Lemma 2 within groups (Mondrian validity under exchangeability): the same, per customer region; INFO row: how
      often each region has its own correction, and which region is the worst
  C5  Effect of drift (oracle reference, Section 4.3): coverage of the oracle temporal calibration (final outcomes of the calibration window) minus
      the nominal level (INFO: the size of the drift that no reweighting of the calibration set can remove)
  C6  Eq. (8), Algorithm 2, Remark 6: ACI long-run guarantee (Gibbs and Candes, 2021, Prop. 4.1): |mean miscoverage - alpha| <=
      (max(alpha, 1 - alpha) + gamma) / (gamma T), T = number of per-order updates; with delayed feedback, daily
      batches and the lower clip of the level the bound is not implied, so it is checked
  C7  Lemma 4: rho_i(P) non-increasing in P (PASS expected); Remark 7: convex (INFO)
  C8  Proposition 5, part 2: the mean promise of the Lagrangian sweep is non-increasing in lambda (a property of the exact per-order minimisers
      of rho_i(P) + lambda P, so a violation would reveal a computing error)
  C10 Proposition 5, part 3, and Algorithm 3: the Lagrangian sweep is near-optimal for every mean length L, measured by the gap
      between a feasible repair of the vertex below L and the Lagrangian dual bound (rq3_duality_gap.csv)
  C11 RQ3 premise: the response curve fitted on 2017 orders predicts the observed low-rating rate of the 2018
      windows under Olist's promise (out-of-time calibration, rq3_curve_calibration.csv)
  C12 selection rule of Section 4.3: the quantile average reported in RQ1 is the candidate selected by the pre-specified rule (shortest mean
      promise at Olist's coverage over the 2017 origins; rq1_ensemble_selection.csv)
  C9  RQ2 premises: the adjusted response curve (Eq. 10) is non-decreasing in S; placebo RDD intervals (Lemma 3) cover zero;
      propensity overlap (share of propensities at the clipping bound, INFO)
"""
import numpy as np
import pandas as pd

from ..conformal.one_sided import make_ecdf, level, promise_from_bound
from ..data.censoring import status_at, ipcw_weights
from .rq1_fit import load_fits


def _row(cid, claim, value, criterion, ok=None):
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    return dict(check=cid, claim=claim, value=value, criterion=criterion, status=status)


def run_theory_checks(D, cfg, out, log=print):
    t = out / "tables"
    fits = load_fits(out / "data" / "parts", "main")
    v = cfg["rq3"]["model"]
    taus = cfg["quantiles"]
    rows = []
    rng = np.random.default_rng(cfg["seed"])

    # C1
    viol, n = 0, 0
    for r in fits:
        E = D.iloc[r["eval_idx"]]; ok = E["delivered"].to_numpy()
        U = np.exp(r["pred"][v]["eval"][:, taus.index(0.9)].astype(float) + rng.normal(0, 0.3, len(E)))
        P = promise_from_bound(np.log(U), 0.0, 1e9)
        Y = E["Y"].to_numpy()[ok]; Dd = E["D"].to_numpy()[ok]; Uo = U[ok]; Po = P[ok]
        m = Y <= Uo
        viol += int((Dd[m] > Po[m]).sum()); n += int(m.sum())
    rows.append(_row("C1", "Y <= U implies a kept promise floor(U) (Proposition 1)", f"{viol} violations in {n} orders",
                     "0 violations", viol == 0))

    # C2
    errs = []
    for r in fits:
        C = D.iloc[r["cal_idx"]]
        obs, c, y = status_at(C, r["origin"])
        w, _ = ipcw_weights(obs, c, y, cfg["censoring"]["weight_cap"])
        dl = C["delivered"].to_numpy(); Yf = C["Y"].to_numpy()
        for tt in (5, 10, 15, 20, 25, 30):
            oracle = float((Yf[dl] <= tt).mean())
            ipcw = float((w[obs] * (y[obs] <= tt)).sum() / w[obs].sum())
            naive = float((y[obs] <= tt).mean())
            errs.append(dict(k=r["k"], t=tt, ipcw=abs(ipcw - oracle), naive=abs(naive - oracle)))
    E2 = pd.DataFrame(errs)
    E2.to_csv(t / "theory_check_ipcw_cdf.csv", index=False)
    rows.append(_row("C2", "IPCW reduces the error of Pr(Y <= t) on the censored calibration sets against the "
                     "naive estimate (Proposition 2)",
                     f"mean |error| IPCW {E2['ipcw'].mean():.4f} vs naive {E2['naive'].mean():.4f}; "
                     f"IPCW closer in {100 * (E2['ipcw'] < E2['naive']).mean():.0f}% of origin x t cases",
                     "IPCW closer than naive in >= 90% of cases", bool((E2["ipcw"] < E2["naive"]).mean() >= 0.9)))
    byk = E2.groupby("k")["ipcw"].mean()
    org = {r["k"]: r["origin"].date() for r in fits}
    bad = byk[byk > 0.03]
    rows.append(_row("C2", "IPCW unbiasedness requires the purchase time to be independent of the delivery time within "
                     "the calibration window; origins with mean |error| > 0.03",
                     ", ".join(f"{org[k]} ({v:.3f})" for k, v in bad.items()) or "none",
                     "reported (expected at demand peaks, where the assumption fails)"))
    # C3 / C4 (random halves of each evaluation window; delivered orders)
    cov, cov_g = {a: [] for a in cfg["alphas"]}, {}
    for r in fits:
        E = D.iloc[r["eval_idx"]]; ok = E["delivered"].to_numpy()
        ly = np.log(E["Y"].to_numpy()[ok]); g = E["customer_region"].fillna("NA").to_numpy()[ok]
        for a in cfg["alphas"]:
            q = r["pred"][v]["eval"][ok, taus.index(round(1 - a, 4))].astype(float)
            for rep in range(20):
                half = rng.random(len(ly)) < 0.5
                sc = ly[half] - q[half]
                e, nn = make_ecdf(sc, np.ones(half.sum(), bool), np.ones(half.sum()), "naive")
                cov[a].append(float((ly[~half] <= q[~half] + float(e.quantile(level(a, nn)))).mean()))
                for gg in np.unique(g):
                    mc, mt = half & (g == gg), (~half) & (g == gg)
                    if mc.sum() < 30 or mt.sum() < 10:
                        continue
                    e, nn = make_ecdf(ly[mc] - q[mc], np.ones(mc.sum(), bool), np.ones(mc.sum()), "naive")
                    cov_g.setdefault((a, gg), []).append(float((ly[mt] <= q[mt] + float(e.quantile(level(a, nn)))).mean()))
    for a in cfg["alphas"]:
        mcov = np.mean(cov[a])
        rows.append(_row("C3", f"split conformal under exchangeability (Lemma 2), alpha = {a}", f"mean coverage {mcov:.4f}",
                         f">= {1 - a - 0.005:.3f} (nominal minus Monte Carlo tolerance 0.005)", bool(mcov >= 1 - a - 0.005)))
        gm = {gg: np.mean(x) for (aa, gg), x in cov_g.items() if aa == a}
        worst = min(gm.values())
        rows.append(_row("C4", f"Mondrian (region) under exchangeability (Lemma 2 within regions), alpha = {a}",
                         "; ".join(f"{k} {vv:.3f}" for k, vv in sorted(gm.items())),
                         f"every region >= {1 - a - 0.01:.3f}", bool(worst >= 1 - a - 0.01)))

    # C4 (INFO): how often each region has its own Mondrian correction (calibration orders >= mondrian_min_n), and
    # which region is the worst under split IPCW; a region that falls back to the pooled correction is not calibrated
    # separately, so the Mondrian contrast says little about it
    mn = cfg["conformal"]["mondrian_min_n"]
    cnt = pd.DataFrame([D.iloc[r["cal_idx"]]["customer_region"].fillna("NA").value_counts().rename(r["k"]) for r in fits]).fillna(0)
    own = (cnt >= mn).mean()
    M = pd.read_csv(t / "rq1_window_metrics.csv")
    Wr = M[(M.variant == v) & (M.method == "split_ipcw") & np.isclose(M.alpha, 0.10) & (M.region != "all")]
    worst = Wr.loc[Wr.groupby("k")["coverage"].idxmin(), "region"].value_counts(normalize=True)
    rows.append(_row("C4", f"Mondrian: share of origins with >= {mn} calibration orders per region (own correction); "
                     f"region with the lowest split-IPCW coverage, alpha = 0.1",
                     "own: " + "; ".join(f"{k} {vv:.2f}" for k, vv in own.sort_index().items()) + " | worst: " +
                     "; ".join(f"{k} {vv:.2f}" for k, vv in worst.items()), "reported"))

    # C5 / C6 from the RQ1 tables
    for a in cfg["alphas"]:
        x = M[(M.variant == v) & (M.method == "split_oracle") & np.isclose(M.alpha, a) & (M.region == "all")]
        if len(x):
            rows.append(_row("C5", f"temporal oracle calibration, alpha = {a}: coverage - nominal (drift)",
                             f"mean {100 * (x['coverage'].mean() - (1 - a)):+.2f} pp; windows below nominal - 2 pp: "
                             f"{int((x['coverage'] < 1 - a - 0.02).sum())} of {len(x)}", "reported"))
    gamma = cfg["conformal"]["aci_gamma"]
    for a in cfg["alphas"]:
        x = M[(M.variant == v) & (M.method == "aci_ipcw") & np.isclose(M.alpha, a) & (M.region == "all")]
        if len(x):
            Tn = x["n_all"].sum()                       # number of per-order updates (applied in daily batches)
            mis = 1 - np.average(x["coverage_all"], weights=x["n_all"])
            bound = (max(a, 1 - a) + gamma) / (gamma * Tn)
            rows.append(_row("C6", f"ACI long-run miscoverage, alpha = {a} (all orders, undelivered = missed)",
                             f"{mis:.4f} (|gap| {abs(mis - a):.4f}, bound {bound:.4f}, T = {Tn} resolved orders)",
                             "|gap| <= bound", bool(abs(mis - a) <= bound)))

    # C7 / C8
    Cv = pd.read_csv(t / "rq3_convexity.csv")
    rows.append(_row("C7", "rho_i(P) non-increasing in P (Lemma 4)", f"{Cv['share_nonincreasing'].mean():.4f} of orders", "= 1",
                     bool(Cv["share_nonincreasing"].min() >= 0.999)))
    rows.append(_row("C7", "rho_i(P) convex in P (premise of the threshold form, Remark 7)",
                     f"{Cv['share_convex'].mean():.4f} of orders over P in [1, 75]; "
                     f"{Cv['share_convex_P15_40'].mean():.4f} over [15, 40]", "reported"))
    W = pd.read_csv(t / "rq3_policy_windows.csv")
    mono = []
    for k, g in W[W.policy == "optimal"].groupby("k"):
        g = g.sort_values("param")
        mono.append(bool((np.diff(g["mean_promise"].to_numpy()) <= 1e-9).all()))
    rows.append(_row("C8", "mean promise of the Lagrangian solution non-increasing in lambda (Proposition 5, part 2)",
                     f"{sum(mono)} of {len(mono)} windows", "all windows", all(mono)))

    # C10 / C11
    if (t / "rq3_duality_gap.csv").is_file():
        Gp = pd.read_csv(t / "rq3_duality_gap.csv")
        rows.append(_row("C10", "Lagrangian sweep near-optimal for every mean length L in 14..32 days (Proposition 5, "
                         "part 3, Algorithm 3): feasible repair minus dual bound",
                         f"mean {100 * Gp['gap'].mean():.4f} pp, max {100 * Gp['gap'].max():.4f} pp over "
                         f"{len(Gp)} window x L cases", "max gap < 0.1 pp", bool(Gp["gap"].max() < 0.001)))
    if (t / "rq3_curve_calibration.csv").is_file():
        Cc = pd.read_csv(t / "rq3_curve_calibration.csv")
        o = Cc[Cc["out_of_sample"]]
        bias = float((o["predicted_fit2017"] - o["observed"]).mean())
        mae = float((o["predicted_fit2017"] - o["observed"]).abs().mean())
        cor = float(np.corrcoef(o["predicted_fit2017"], o["observed"])[0, 1])
        rows.append(_row("C11", "response curve fitted on 2017 predicts the low-rating rate of the 2018 windows "
                         "under Olist's promise (out of time)",
                         f"mean bias {100 * bias:+.2f} pp, MAE {100 * mae:.2f} pp, correlation {cor:.3f} "
                         f"({len(o)} windows)", "|mean bias| <= 1 pp and correlation >= 0.7",
                         bool(abs(bias) <= 0.01 and cor >= 0.7)))

    # C9
    G = pd.read_csv(t / "rq2_response_curve.csv"); g = G[G.group == "all"].sort_values("S")
    rows.append(_row("C9", "adjusted response curve r(s) non-decreasing in s (Eq. 10)",
                     f"max decrease {max(0.0, -np.diff(g['r']).min()):.2e}", "<= 1e-9",
                     bool(np.diff(g["r"]).min() >= -1e-9)))
    RD = pd.read_csv(t / "rq2_rdd.csv")
    pl = RD[(RD.subset == "placebo") & np.isclose(RD.M, RD.M_rule_of_thumb)]
    cover0 = ((pl["ci_lo"] <= 0) & (pl["ci_hi"] >= 0)).mean()
    rows.append(_row("C9", "placebo RDD cut-offs (S = -6, -3, 3, 6): honest intervals (Lemma 3) cover 0",
                     f"{100 * cover0:.0f}% of {len(pl)} placebo estimates", "reported (>= 80% expected)"))
    A = pd.read_csv(t / "rq2_aipw_bins.csv")
    rows.append(_row("C9", "AIPW overlap: smallest propensity before clipping at 0.01, by bin (XGBoost)",
                     "; ".join(f"{r['label']}: p1 {r['ps_p01']:.3f}" for _, r in A[A.learner == "xgb"].iterrows()),
                     "reported"))
    if (t / "rq2_aipw_overlap_sensitivity.csv").is_file():
        Ov = pd.read_csv(t / "rq2_aipw_overlap_sensitivity.csv")
        base = Ov[(Ov.analysis == "clip 0.01")].set_index(["learner", "bin"])["rr"]
        dev = Ov.assign(d=(Ov.set_index(["learner", "bin"]).index.map(base.to_dict())))
        dev["rel"] = (dev["rr"] / dev["d"] - 1).abs()
        for lrn, name in (("xgb", "XGBoost (main)"), ("lr", "logistic regression")):
            dc = dev[(dev.learner == lrn) & dev.analysis.str.startswith("clip") & (dev.bin != cfg["rq2"]["ref_bin"])]
            m = dc.loc[dc["rel"].idxmax()]
            rows.append(_row("C9", f"AIPW risk ratios stable to the propensity clip 0.005-0.05, nuisance {name}",
                             f"largest relative change {100 * m['rel']:.1f}% ({m['analysis']}, bin {m['label']})",
                             "reported"))
        dt = dev[dev.analysis.str.startswith("trim") & (dev.bin != cfg["rq2"]["ref_bin"])]
        m = dt.loc[dt["rel"].idxmax()]
        late = Ov[Ov.analysis.str.startswith("trim") & (Ov.bin > cfg["rq2"]["ref_bin"])]
        full = Ov[(Ov.analysis == "clip 0.01") & (Ov.bin > cfg["rq2"]["ref_bin"])].groupby("learner")["n"].sum()
        drop = 1 - late.groupby(["learner", "analysis"])["n"].sum() / late.groupby(["learner", "analysis"])["learner"].first().map(full)
        rows.append(_row("C9", "AIPW trimming of orders with own propensity < 0.01 / 0.02 (Crump et al., 2009): changes the "
                         "target population, not only the estimator",
                         f"largest relative change of a risk ratio {100 * m['rel']:.1f}% ({m['analysis']}, {m['learner']}, "
                         f"bin {m['label']}); late orders removed up to {100 * drop.max():.0f}%", "reported"))
        ok_all = True
        for (lrn, an), g in Ov.groupby(["learner", "analysis"]):
            g = g.set_index("bin")
            late_bins = [k for k in g.index if k > cfg["rq2"]["ref_bin"]]
            rr = g.loc[late_bins, "rr"].to_numpy()
            ok_all &= bool(np.all(np.diff(rr) > 0) and np.all(g.loc[late_bins, "rr_lo"] > 1)
                           and np.all(g.loc[late_bins, "e_value_ci"] > 2) and g.loc[0, "rr_hi"] < 1)
        rows.append(_row("C9", "qualitative RQ2 conclusions hold in every overlap specification (2 learners x 4 clips + "
                         "2 trims): risk ratios increasing in lateness, every late bin RR CI above 1 with E-value(CI) > 2, "
                         "early-delivery bin RR CI below 1", f"{Ov.groupby(['learner', 'analysis']).ngroups} specifications",
                         "all specifications", ok_all))
    if (t / "rq1_tuning_choices.csv").is_file():
        Tc = pd.read_csv(t / "rq1_tuning_choices.csv"); tt = Tc[Tc.tuned_here]
        desc = lambda c: ", ".join(f"{k} x{v}" for k, v in tt[c].astype(str).value_counts().items())
        rows.append(_row("C13", "hyper-parameters chosen on the inner validation block (tuned origins): distribution of "
                         "the choices and share on the edge of the grid",
                         f"{len(tt)} tuned origins; XGBoost max_depth {desc('xgbq_max_depth')} (edge {tt.xgbq_edge.mean():.0%}); "
                         f"QRF min_samples_leaf {desc('qrf_min_samples_leaf')} (edge {tt.qrf_edge.mean():.0%}); "
                         f"AFT scale {desc('aft_scale')} (edge {tt.aft_scale_edge.mean():.0%})", "reported"))
    if (t / "rq1_tuning_extended.csv").is_file():          # re-tuning on grids extended at the edges (sens_fit)
        Te = pd.read_csv(t / "rq1_tuning_extended.csv")
        ch = Te[Te["changed"]]
        G = pd.read_csv(t / "rq1_grid_extended.csv") if (t / "rq1_grid_extended.csv").is_file() else pd.DataFrame()
        eff = ""
        if "coverage_extended" in G.columns:
            eff = (f"; refitted origins: coverage {100 * (G['coverage_extended'] - G['coverage_main']).abs().max():.2f} pp "
                   f"and mean promise {(G['mean_promise_extended'] - G['mean_promise_main']).abs().max():.2f} d at most "
                   "from the main fit (split IPCW, alpha = 0.10)")
            if "diverged_extended" in G.columns:
                eff += f"; refits whose 0.9-quantile leaves [1 hour, 10 years]: {int((G['diverged_extended'] > 0).sum())} of {len(G)}"
        rows.append(_row("C13", "grid edges: XGBoost QR depth and AFT scale re-tuned on grids extended at both ends",
                         f"{len(Te) // 2} tuned origins; choices changed: XGBoost {int((ch.variant == 'xgbq').sum())}, "
                         f"AFT {int((ch.variant == 'aft').sum())}; on the edge of the extended grids: XGBoost "
                         f"{Te[Te.variant == 'xgbq'].on_edge_extended.mean():.0%}, AFT {Te[Te.variant == 'aft'].on_edge_extended.mean():.0%}" + eff,
                         "reported"))
    if (t / "rq1_ensemble_selection.csv").is_file():
        Es = pd.read_csv(t / "rq1_ensemble_selection.csv")
        sel = Es[Es.selected]
        rows.append(_row("C12", "quantile average: the configured member set is the one selected by the pre-specified "
                         "rule (shortest promise at Olist's coverage over the 2017 origins)",
                         "; ".join(f"{r['candidate']} 2017 {r['length_gap_2017']:+.2f} d, 2018 {r['length_gap_2018']:+.2f} d"
                                   for _, r in Es.iterrows()) + f"; selected {', '.join(sel['candidate'])}",
                         "configured = selected", bool(len(sel) == 1 and sel["configured"].all())))
    R = pd.DataFrame(rows)
    R.to_csv(t / "theory_checks.csv", index=False)
    log("  theory checks: " + ", ".join(f"{c} {s}" for c, s in zip(R["check"], R["status"])))
    return R
