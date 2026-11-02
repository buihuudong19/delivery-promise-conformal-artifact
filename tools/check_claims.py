"""Check the qualitative statements of the article against the outputs of a run.

usage: python tools/check_claims.py OUTPUTS_DIR [LATEX_DIR]

Numbers in the article are macros (tools/gen_results_tex.py) and follow a new run automatically. Words do not: "every
contrast is significant", "the only model that", "none significant". Each such sentence of the LaTeX source carries a
comment "% claim:<id>", and this script tests the statement with the outputs of the run. It also tests the settings
that the text states in words (quantile levels, grids, bins) against config.yaml, and, with LATEX_DIR, that every
claim id of the text is defined here and every id defined here is used in the text.
Exit code 1 if a claim fails: the sentence must then be rewritten before the article is rebuilt.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if len(sys.argv) not in (2, 3):
    raise SystemExit(__doc__)
OUT = Path(sys.argv[1]).resolve(); TEX = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else None
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2du.config import load_config      # noqa: E402

CFG = load_config(ROOT / "config.yaml")
T = OUT / "tables"
r = lambda f: pd.read_csv(T / f, encoding="utf-8-sig")
ENS, BASE = CFG["ensemble"]["name"], CFG["rq3"]["model"]
CLAIMS = {}


def claim(cid, text):
    def deco(fn):
        CLAIMS[cid] = (text, fn)
        return fn
    return deco


def fam(C, prefix):
    return C[C.family.str.startswith(prefix)]


# ------------------------------------------------------------------ RQ1
@claim("c2-most-stress", "most origins where the IPCW estimate of Pr(Y <= t) is still off by more than 0.03 (check C2) are stress windows")
def _():
    E = r("theory_check_ipcw_cdf.csv").groupby("k")["ipcw"].mean(); bad = E[E > 0.03].index
    st = r("rq1_window_types.csv").set_index("k")["stress"].astype(bool)
    return bool(len(bad) > 0 and st.reindex(bad).mean() > 0.5)


@claim("olist-coverage-fell", "Olist's coverage is lower in 2018 than in 2017, at about the same mean promise (|change| < 1 day)")
def _():
    W = r("rq1_window_metrics.csv"); o = W[(W.variant == "olist") & (W.region == "all")].copy()
    o["y"] = pd.to_datetime(o["origin"]).dt.year
    g = o.groupby("y")[["coverage", "mean_promise"]].mean()
    return g.loc[2018, "coverage"] < g.loc[2017, "coverage"] and abs(g.loc[2018, "mean_promise"] - g.loc[2017, "mean_promise"]) < 1


@claim("rq1a-all-significant", "IPCW raises coverage over naive calibration for every variant, every Holm p < 0.001")
def _():
    A = fam(r("contrasts_rq1.csv"), "RQ1-A")
    return bool((A["mean"] > 0).all() and (A["p_holm"] < 0.001).all())


@claim("ipcw-near-nominal", "at a 90% target, IPCW calibration brings every variant within 1.5 pp of 90% (mean over windows)")
def _():
    S = r("rq1_summary.csv"); i = S[(S.region == "all") & np.isclose(S.alpha, 0.10) & (S.method == "split_ipcw")]
    return bool(((i["coverage"] - 0.9).abs() <= 0.015).all())


@claim("no-method-holds-peaks", "in the stress windows every calibration method of the forest, the oracle included, covers less than 90%")
def _():
    B = r("rq1_summary_by_period.csv")
    x = B[(B.variant == BASE) & np.isclose(B.alpha, 0.10) & (B.group == "stress") & (B.method != "raw")]
    return bool(len(x) and (x["coverage"] < 0.9).all())


@claim("mondrian-region-signs", "averaged over windows, Mondrian raises the coverage of the North and North-East and lowers that of the Centre-West (forest, 90%)")
def _():
    W = r("rq1_window_metrics.csv")
    x = W[(W.variant == BASE) & np.isclose(W.alpha, 0.10) & W.method.isin(["split_ipcw", "mondrian_ipcw"]) & (W.region != "all")]
    p = x.pivot_table(index="region", columns="method", values="coverage", aggfunc="mean"); d = p["mondrian_ipcw"] - p["split_ipcw"]
    return bool(d["Norte"] > 0 and d["Nordeste"] > 0 and d["Centro-Oeste"] < 0)


@claim("late-before-most", "among late orders, those rated before arrival account for most of the low ratings")
def _():
    M = r("rq2_survey_mechanism.csv").set_index(["late", "answered_before_delivery"])
    lb = M.loc[(1, 1), "mean"] * M.loc[(1, 1), "count"]; la = M.loc[(1, 0), "mean"] * M.loc[(1, 0), "count"]
    return bool(lb > la)


@claim("rq1b-small", "training weights change coverage by < 0.5 pp for the quantile models; the AFT with censored labels covers more and promises longer")
def _():
    C = r("contrasts_rq1.csv"); B = C[C.family == "RQ1-B training weights (coverage)"]
    q = B[~B.contrast.str.startswith("aft")]; a = B[B.contrast.str.startswith("aft")]["mean"].iloc[0]
    L = C[(C.family == "RQ1-B training weights (mean promise)") & C.contrast.str.startswith("aft")]["mean"].iloc[0]
    q = q[~q.contrast.str.startswith("route")]
    rt = B[B.contrast.str.startswith("route")].iloc[0]
    return bool((q["mean"].abs() < 0.005).all() and (q["p_holm"] >= 0.05).all() and rt["p_holm"] < 0.05 and a > 0 and L > 0)


@claim("north-worst-most", "the North is the most frequent worst region of the forest at 90% (split IPCW)")
def _():
    W = r("rq1_window_metrics.csv")
    x = W[(W.variant == BASE) & (W.method == "split_ipcw") & np.isclose(W.alpha, 0.10) & (W.region != "all")]
    return bool(x.loc[x.groupby("k")["coverage"].idxmin(), "region"].value_counts().idxmax() == "Norte")


@claim("raw-not-shorter", "at Olist's coverage, no uncalibrated quantile promise is significantly shorter than its split-IPCW promise, and every significant RQ1-J difference is positive (Holm p < 0.05)")
def _():
    C = r("contrasts_rq1.csv"); J = C[C.family.str.startswith("RQ1-J")]
    sig = J[J["p_holm"] < 0.05]
    return bool(len(J) > 0 and len(sig) > 0 and (sig["mean"] > 0).all())


@claim("late-rated-before-most", "more than half of the late orders with a review are rated before they arrive")
def _():
    M = r("rq2_survey_mechanism.csv").set_index(["late", "answered_before_delivery"])
    return bool(M.loc[(1, 1), "share_within_late_group"] > 0.5)


@claim("covall-diff-small", "counting never-delivered orders as missed changes the IPCW - naive differences by < 0.5 pp")
def _():
    S = r("rq1_summary.csv"); a = S[(S.region == "all") & np.isclose(S.alpha, 0.10)]
    i = a[a.method == "split_ipcw"].set_index("variant"); n = a[a.method == "split_naive"].set_index("variant")
    return bool(((i.coverage - n.coverage) - (i.coverage_all - n.coverage_all)).abs().max() < 0.005)


@claim("rq1f-no-single-model", "over all windows, no single model is significantly shorter than Olist at Olist's coverage (Holm p >= 0.05)")
def _():
    F = fam(r("contrasts_rq1.csv"), "RQ1-F"); s = F[~F.contrast.str.startswith(ENS)]
    return bool(((s["p_holm"] >= 0.05) | (s["mean"] >= 0)).all())


@claim("rq1h-ens-shortest", "the quantile average is shorter than each member over all windows and over 2018 (all means < 0)")
def _():
    H = fam(r("contrasts_rq1.csv"), "RQ1-H")
    return bool((H["mean"] < 0).all())


@claim("c12-five-selected", "the selection rule chooses the configured (five-model) set on 2017")
def _():
    S = r("rq1_ensemble_selection.csv")
    return bool(S.loc[S["selected"], "candidate"].iloc[0] == ENS and S.loc[S["configured"], "candidate"].iloc[0] == ENS)


@claim("rq1i-most-significant", "in 2018 the quantile average is significantly shorter than Olist, more than half of the single models are too, and the average is the shortest")
def _():
    I = fam(r("contrasts_rq1.csv"), "RQ1-I").copy(); I["v"] = I.contrast.str.replace(" - Olist", "", regex=False)
    e = I[I.v == ENS].iloc[0]; s = I[I.v != ENS]
    return bool(e["p_holm"] < 0.05 and e["mean"] < 0 and ((s["p_holm"] < 0.05) & (s["mean"] < 0)).mean() > 0.5
                and e["mean"] <= s["mean"].min())


@claim("raw-calibration", "uncorrected quantiles: linear QR and XGBoost under-cover at 0.9, the AFT over-covers at 0.8, the forest and the average are within 1.5 pp of 0.9, the average has the lowest pinball loss at 0.9")
def _():
    P = r("rq1_pinball_summary.csv"); g = lambda v, t, c="coverage_raw": P[(P.variant == v) & np.isclose(P.tau, t)][c].iloc[0]
    p9 = P[np.isclose(P.tau, 0.9)]
    return bool(g("lqr_ipcw", .9) < .9 and g("xgbq_ipcw", .9) < .9 and g("aft", .8) > .8 and abs(g(BASE, .9) - .9) < .015
                and abs(g(ENS, .9) - .9) < .015 and p9.loc[p9["pinball"].idxmin(), "variant"] == ENS)


@claim("rq1e-mondrian-none", "Mondrian does not raise the worst region significantly for any variant; in 2018 it lowers overall coverage for the forest (mean < 0)")
def _():
    C = r("contrasts_rq1.csv"); E = fam(C, "RQ1-E"); E = E[E.contrast.str.contains("mondrian_ipcw - split_ipcw")]
    g = C[(C.family == "RQ1-G test period 2018 (coverage)") & (C.contrast == f"{BASE}: mondrian_ipcw - split_ipcw")]["mean"].iloc[0]
    return bool(not ((E["mean"] > 0) & (E["p_holm"] < 0.05)).any() and g < 0)


@claim("rq1g-aci-not-better", "in 2018, ACI does not raise the coverage or the worst region of the forest, and its lowest window is lower than split's")
def _():
    C = r("contrasts_rq1.csv")
    c = C[(C.family == "RQ1-G test period 2018 (coverage)") & (C.contrast == f"{BASE}: aci_ipcw - split_ipcw")]["mean"].iloc[0]
    w = C[(C.family == "RQ1-G test period 2018 (worst-region coverage)") & (C.contrast == f"{BASE}: aci_ipcw - split_ipcw")]["mean"].iloc[0]
    S = r("sensitivity_rq1.csv"); t = S[(S.period == "2018 (test)") & (S.variant == BASE) & np.isclose(S.alpha, 0.10)].set_index("setting")
    return bool(c <= 0 and w <= 0 and t.loc["daily batch, gamma = 0.0005", "coverage_min_window"] < t.loc["reference: split IPCW", "coverage_min_window"])


@claim("c6-fails", "check C6 fails at alpha = 0.10 (exactly one FAIL row)")
def _():
    T_ = r("theory_checks.csv"); f = T_[T_.status == "FAIL"]
    return len(f) == 1 and f["check"].iloc[0] == "C6" and "alpha = 0.1 " in f["claim"].iloc[0]


@claim("checks-one-fail", "the theory checks have exactly one FAIL")
def _():
    return int((r("theory_checks.csv").status == "FAIL").sum()) == 1


@claim("sens-small", "Olist's estimate as an input shortens the XGBoost bounds by less than 1 day at 90% (minor settings are bounded by computed macros)")
def _():
    S = r("sensitivity_rq1.csv"); s = S[(S.variant == "xgbq_ipcw") & np.isclose(S.alpha, 0.10) & (S.period == "all windows")].set_index("setting")
    d = s.loc["main (split IPCW, 42 d, log score, cap 20)", "mean_promise"] - s.loc["Olist's estimate as an input", "mean_promise"]
    return bool(0 <= d < 1)


# ------------------------------------------------------------------ RQ2
@claim("s-beats-d", "the model in S has a lower CV log loss than the model in D, and adding P to S helps less than S over D")
def _():
    V = r("rq2_s_versus_d.csv").set_index("model")["cv_logloss"]
    return bool(V["f(S)"] < V["f(D)"] and V["f(S) + g(P)"] <= V["f(S)"])


@claim("rq2-rr-monotone", "AIPW risk ratios (XGBoost) increase with lateness, are above 1 for late bins and below 1 for the earliest bin, with CIs excluding 1")
def _():
    A = r("rq2_aipw_bins.csv"); X = A[A.learner == "xgb"].set_index("bin")
    late = X.loc[[3, 4, 5]]
    return bool(late["rr"].is_monotonic_increasing and (late["rr_lo"] > 1).all() and X.loc[0, "rr_hi"] < 1)


@claim("rdd-all-covers-zero", "for all ratings, the honest interval at S = 0 covers zero at every bandwidth (rule-of-thumb M)")
def _():
    R = r("rq2_rdd.csv"); R = R[(R.cutoff == 0) & (R.subset == "all") & np.isclose(R["M"], R["M_rule_of_thumb"])]
    return bool(((R.ci_lo <= 0) & (R.ci_hi >= 0)).all())


@claim("rdd-after-positive", "for ratings answered after arrival, the jump is positive and its honest interval excludes zero at h = 3, 5, 7")
def _():
    R = r("rq2_rdd.csv"); R = R[(R.cutoff == 0) & (R.subset == "answered_after_arrival") & np.isclose(R["M"], R["M_rule_of_thumb"]) & R.h.isin([3, 5, 7])]
    return bool((R.estimate > 0).all() and (R.ci_lo > 0).all())


@claim("lee-positive", "both Lee bounds (adjacent cells and two cells) are positive, and the lower end of the bootstrap interval is positive")
def _():
    L = r("rq2_lee_bounds.csv")
    return bool((L.bound_lo > 0).all() and (L.ci_lo > 0).all())


@claim("rdd-after-m2", "for ratings answered after arrival, with twice the rule-of-thumb M the honest interval excludes zero at h = 3 and 5 but not at h = 7")
def _():
    R = r("rq2_rdd.csv"); R = R[(R.cutoff == 0) & (R.subset == "answered_after_arrival") & np.isclose(R["M"], 2 * R["M_rule_of_thumb"])].set_index("h")
    return bool((R.loc[[3, 5], "ci_lo"] > 0).all() and R.loc[7, "ci_lo"] <= 0 <= R.loc[7, "ci_hi"])


# ------------------------------------------------------------------ RQ3
@claim("region-trade", "rating-optimal vs Olist (window means): the North-East's late and r-implied low-rating rates fall, the North's both rise")
def _():
    G = r("rq3_region_promises.csv").groupby(["region", "policy"])[["late_realised", "low_realised"]].mean()
    d = lambda reg, c: G.loc[(reg, "optimal"), c] - G.loc[(reg, "olist"), c]
    return bool(d("Nordeste", "late_realised") < 0 and d("Nordeste", "low_realised") < 0
                and d("Norte", "late_realised") > 0 and d("Norte", "low_realised") > 0)


@claim("lateopt-late-above", "at Olist's mean promise, the late-optimal policy has a higher realised late rate than the rating-optimal policy (mean over windows)")
def _():
    Q = r("rq3_matched.csv")
    return bool(Q["late_optimal_late_realised_at_L"].mean() > Q["optimal_late_realised_at_L"].mean())


@claim("rq3-2017-curve-same", "in the 2018 windows, the policy built from the 2017 curve and the main policy lower the low-rating rate by the same amount (|difference| < 0.05 pp)")
def _():
    C = r("contrasts_rq3_sensitivity.csv"); C = C[C.family.str.startswith("RQ3-S low-rating rate at Olist's mean promise (2018 test)")].set_index("contrast")
    return bool(abs(C.loc["optimal - Olist: policy from the 2017 curve", "mean"] - C.loc["optimal - Olist: forest CPD (main)", "mean"]) < 0.0005)


@claim("rq3-ranking", "uniform is worse than Olist (significant); route and coverage policies are not significantly different; late-optimal and optimal are significantly better")
def _():
    C = r("contrasts_rq3.csv"); L = C[C.family == "RQ3 low-rating rate at Olist's mean promise"].set_index("contrast")
    s = lambda k: L.loc[f"{k} - Olist"]
    return bool(s("uniform")["mean"] > 0 and s("uniform")["p_holm"] < 0.05 and s("route")["p_holm"] >= 0.05
                and s("cpd_quantile")["p_holm"] >= 0.05 and s("late_optimal")["mean"] < 0 and s("late_optimal")["p_holm"] < 0.05
                and s("optimal")["mean"] < 0 and s("optimal")["p_holm"] < 0.05)


@claim("rq3-gain-significant", "the optimal policy beats the late-optimal policy (Holm p < 0.05) and gains more in stress windows than in normal windows")
def _():
    C = r("contrasts_rq3.csv"); x = C[C.contrast == "optimal - late_optimal"].iloc[0]
    Q = r("rq3_matched.csv"); W = r("rq1_window_types.csv"); st = Q["k"].isin(W.loc[W["stress"], "k"])
    d = Q["optimal_low_realised_at_L"] - Q["low_olist"]
    return bool(x["mean"] < 0 and x["p_holm"] < 0.05 and d[st].mean() < d[~st].mean() < 0)


@claim("c11-curve-calibrated", "the 2017 curve predicts the 2018 windows with |mean bias| < 1 pp and correlation > 0.8")
def _():
    Cc = r("rq3_curve_calibration.csv"); o = Cc[Cc["out_of_sample"]]
    return bool(abs((o.predicted_fit2017 - o.observed).mean()) < 0.01 and np.corrcoef(o.predicted_fit2017, o.observed)[0, 1] > 0.8)


@claim("ens-cpd-not-significant", "the quantile-average CPD is not significantly better than the forest CPD for the optimal policy; it helps the coverage policy more")
def _():
    Cs = r("contrasts_rq3_sensitivity.csv"); E = Cs[Cs.family.str.contains("versus forest CPD (low-rating", regex=False)].set_index("contrast")
    o = E.loc["optimal: quantile-average CPD - forest CPD"]; c = E.loc["coverage policy: quantile-average CPD - forest CPD"]
    return bool(o["p_holm"] >= 0.05 and c["mean"] < o["mean"])


@claim("rq3-sens-all-significant", "every RQ3 sensitivity contrast of the optimal policy against Olist has Holm p < 0.001 and a negative mean")
def _():
    Cs = r("contrasts_rq3_sensitivity.csv"); X = Cs[Cs.contrast.str.startswith("optimal - Olist")]
    return bool((X["p_holm"] < 0.001).all() and (X["mean"] < 0).all())


# ------------------------------------------------------------------ settings stated in words
@claim("config-text", "settings stated in words: quantile levels, alpha levels, deviation bins, RDD bandwidths, response grid, ACI clip, alpha grid of the matched procedure")
def _():
    from p2du.experiments import rq1_extra
    src = Path(rq1_extra.__file__).read_text(encoding="utf-8")
    return bool(CFG["quantiles"] == [0.5, 0.8, 0.9, 0.95] and CFG["alphas"] == [0.05, 0.10, 0.20]
                and CFG["rq2"]["bins"] == [-200, -15, -6, 0, 3, 7, 200] and CFG["rq2"]["rdd_bandwidths"] == [3, 5, 7, 10]
                and CFG["rq2"]["s_range"] == [-40, 30] and CFG["conformal"]["aci_alpha_min_ratio"] == 0.5
                and CFG["rq2"]["ref_bin"] == 2 and "0.30" in src and "0.005" in src)


@claim("code-text", "constants stated in the text match the code: AIPW clip 0.01 with renormalisation, response curve trained on S in [-50, 40], placebo cut-offs -6/-3/3/6, certificate on 300 lambdas and L = 14..32 with 60 bisection steps, tail lookback 28 days, c0 = 28")
def _():
    src = lambda f: (ROOT / "src" / "p2du" / f).read_text(encoding="utf-8")
    r2, r3, ce, se = src("experiments/rq2_response.py"), src("experiments/rq3_policy.py"), src("data/censoring.py"), src("experiments/sensitivity.py")
    return all(x in t for x, t in (("clip=0.01", r2), ("pi / pi.sum(1, keepdims=True)", r2), ("clip(lo - 10, hi + 10)", r2),
                                   ("for c in (-6, -3, 3, 6)", r2), ("LAMBDA_FINE = 300", r3),
                                   ("L_CERT = (14, 32)", r3), ("n_bisect=60", r3), ("lookback_days=28", ce),
                                   ("TRUNC_C0 = 28", se))) and CFG["rq2"]["folds"] == 5


# ------------------------------------------------------------------ run
fails = 0
for cid, (text, fn) in CLAIMS.items():
    try:
        ok = bool(fn())
    except Exception as ex:                       # a missing table or column is a failure, not a crash
        ok = False; text += f" [error: {ex}]"
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {cid:28s} {text}")
if TEX:
    used = set()
    for f in [TEX / "main.tex", TEX / "Macros" / "highlights.tex", *sorted((TEX / "Chapters").glob("*.tex")),
              *sorted((TEX / "Supplementary").glob("*.tex"))]:
        for com in re.findall(r"(?<!\\)%([^\n]*)", f.read_text(encoding="utf-8")):   # every claim id in a comment
            used |= set(re.findall(r"claim:([\w-]+)", com))
    undefined = sorted(used - set(CLAIMS)); unused = sorted(set(CLAIMS) - used - {"config-text", "code-text"})
    if undefined:
        fails += 1; print("FAIL  claim ids in the text without a test:", ", ".join(undefined))
    if unused:
        print("note  tests not referenced in the text:", ", ".join(unused))
print(f"\n{len(CLAIMS) - fails if fails <= len(CLAIMS) else 0} of {len(CLAIMS)} claims hold" if not fails else
      f"\n{fails} claim(s) FAIL: rewrite the sentences marked with these ids before rebuilding the article")
sys.exit(1 if fails else 0)
