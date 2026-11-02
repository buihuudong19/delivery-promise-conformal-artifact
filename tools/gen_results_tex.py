"""Write the LaTeX number macros and table bodies of the article from the outputs of a full run.

usage: python tools/gen_results_tex.py OUTPUTS_DIR MACROS_DIR

MACROS_DIR receives
  results.tex   one \\newcommand per number printed in the article (readable names, a comment with the source)
  tables.tex    the bodies of the tables (\\TabSampleRows, \\TabModelsRows, ...), used inside tabularx
Every value is computed from OUTPUTS_DIR/tables/*.csv, OUTPUTS_DIR/manuscript/table*.csv, OUTPUTS_DIR/data/
features_purchase.pkl, OUTPUTS_DIR/manifest.json, OUTPUTS_DIR/logs/pipeline.log (run time), outputs_extra (cross-check),
config.yaml and the code (number of unit tests). Nothing is typed by hand: after a new run, regenerate both files and
recompile the article; then run tools/check_claims.py, which tests the qualitative statements of the text.
Organisation and conventions follow tools/gen_results_tex.py of the first article.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if len(sys.argv) != 3:
    raise SystemExit(__doc__)
OUT = Path(sys.argv[1]).resolve(); MAC = Path(sys.argv[2]).resolve()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2du.config import load_config                      # noqa: E402
from p2du.data.censoring import tail_share               # noqa: E402
from p2du.evaluation.origins import rolling_windows      # noqa: E402

CFG = load_config(ROOT / "config.yaml")
T = OUT / "tables"; MS = OUT / "manuscript"
r = lambda f: pd.read_csv(T / f, encoding="utf-8-sig")


# ------------------------------------------------------------------ formatting (as in the first article)
def num(x, d=2):
    s = f"{abs(x):.{d}f}"
    return f"$-${s}" if x < 0 and float(s) != 0 else s


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def pts(x, d=2):
    """A difference of proportions in percentage points, signed."""
    return num(100 * x, d)


def ci(lo, hi, d=2, scale=1.0):
    return f"[{num(lo * scale, d)}, {num(hi * scale, d)}]"


def big(n):
    return f"{int(round(n)):,}".replace(",", "{,}")


def holm(p):
    """Relation and value for '$p \\Macro$' in the text."""
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def one(df, what):
    """The single row of a filter; a filter that matches 0 or several rows stops the script (no silent first match)."""
    if len(df) != 1:
        raise SystemExit(f"gen_results_tex: {what} matched {len(df)} rows (expected 1)")
    return df.iloc[0]


def absn(x, d=2):
    return f"{abs(x):.{d}f}"


blocks = []


def block(title):
    blocks.append((title, []))


def m(name, value, note=""):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    blocks[-1][1].append((name, str(value), note))


VAR = {"route_plain": "Route table (plain)", "route_ipcw": "Route table (IPCW)", "lqr_plain": "Linear QR (plain)",
       "lqr_ipcw": "Linear QR (IPCW)", "xgbq_plain": "XGBoost QR (plain)", "xgbq_ipcw": "XGBoost QR (IPCW)",
       "qrf_plain": "Quantile forest (plain)", "qrf_ipcw": "Quantile forest (IPCW)", "aft_naive": "AFT (delivered only)",
       "aft": "AFT (censored labels)", "ens_ipcw": "Quantile average (5 models)"}
CODE = {"route_plain": "RoutePlain", "route_ipcw": "Route", "lqr_plain": "LqrPlain", "lqr_ipcw": "Lqr",
        "xgbq_plain": "XgbPlain", "xgbq_ipcw": "Xgb", "qrf_plain": "QrfPlain", "qrf_ipcw": "Qrf", "aft_naive": "AftNaive",
        "aft": "Aft", "ens_ipcw": "Ens"}
METH = {"raw": "Raw", "split_naive": "Naive", "split_ipcw": "Ipcw", "split_oracle": "Oracle", "mondrian_naive": "MondrianNaive",
        "mondrian_ipcw": "MondrianIpcw", "aci_ipcw": "Aci", "mondrian_aci_ipcw": "MondrianAci", "actual": "Actual"}
ALPHA = {0.05: "Five", 0.10: "Ten", 0.20: "Twenty"}
W2 = {"2017": "Sel", "2018": "Test"}
ens = CFG["ensemble"]["name"]; base3 = CFG["rq3"]["model"]

# ------------------------------------------------------------------ configuration (the text quotes these values)
block("configuration (config.yaml) and the rolling protocol (evaluation/origins.py)")
win = rolling_windows(CFG)
m("NOrigins", len(win)); m("OriginStep", CFG["rolling"]["step_days"])
m("FirstOrigin", f"{win[0]['origin'].day} {win[0]['origin']:%B %Y}")
m("LastOrigin", f"{win[-1]['origin'].day} {win[-1]['origin']:%B %Y}", "last origin produced by the step (not last_origin)")
m("CalDays", CFG["rolling"]["calib_days"]); m("HorizonDays", CFG["rolling"]["horizon_days"])
m("InnerDays", CFG["tuning"]["inner_val_days"]); m("TuneEvery", CFG["tuning"]["retune_every"])
m("TuneEveryWord", {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}[CFG["tuning"]["retune_every"]])
m("RouteShrink", CFG["models"]["route"]["shrink_m"]); m("WeightCap", int(CFG["censoring"]["weight_cap"]))
m("MondrianMinN", CFG["conformal"]["mondrian_min_n"]); m("PromiseCap", CFG["conformal"]["promise_cap_days"])
m("AciGamma", f"{CFG['conformal']['aci_gamma']:g}")
from p2du.experiments.sensitivity import K_SUBS, TRUNC_C0, ACI_GAMMAS     # noqa: E402
m("AciGammaGrid", "\\{" + ", ".join(f"{g:g}" for g in ACI_GAMMAS) + "\\}", "sensitivity.ACI_GAMMAS")
m("GridXgb", "max depth \\{" + ", ".join(str(g["max_depth"]) for g in CFG["tuning"]["xgbq_grid"] if len(g) == 1) + "\\}"
  + "".join(f", plus depth {g['max_depth']} with learning rate {g['learning_rate']}" for g in CFG["tuning"]["xgbq_grid"] if len(g) > 1))
m("GridQrf", "minimum leaf size \\{" + ", ".join(str(g["min_samples_leaf"]) for g in CFG["tuning"]["qrf_grid"]) + "\\}")
m("GridAft", "\\{" + ", ".join(f"{s:g}" for s in CFG["models"]["aft"]["scales"]) + "\\}")
from p2du.experiments.rq3_policy import D_MAX, LAMBDA_FINE, L_CERT    # noqa: E402
import inspect                                                          # noqa: E402
m("DMax", D_MAX); m("TailOffset", CFG["rq3"]["tail_offset_days"])
m("TailDay", CFG["rolling"]["calib_days"] + CFG["rq3"]["tail_offset_days"])
m("PgridMax", CFG["rq3"]["promise_grid"][1]); m("LambdaGrid", CFG["rq3"]["lambda_grid"] + 1, "0 and lambda_grid log-spaced values")
m("LambdaFine", LAMBDA_FINE, "rq3_policy.LAMBDA_FINE"); m("CertLLo", L_CERT[0]); m("CertLHi", L_CERT[1])
m("TailLookback", inspect.signature(tail_share).parameters["lookback_days"].default, "censoring.tail_share: lookback_days")
m("NBoot", big(CFG["bootstrap"]["n_boot"])); m("BlockLen", CFG["bootstrap"]["block_len"])
m("TruncCzero", TRUNC_C0, "sensitivity.TRUNC_C0 (Corollary 1)")
mem = CFG["ensemble"]["members"]
m("EnsNMembers", len(mem)); m("EnsNMembersWord", {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(len(mem), str(len(mem))))
LBL = {"route_ipcw": "the route table", "lqr_ipcw": "linear QR", "xgbq_ipcw": "XGBoost QR", "qrf_ipcw": "the quantile forest",
       "aft": "the AFT with censored labels"}
names_ = [LBL[x] for x in mem]
m("EnsMembersList", (", ".join(names_[:-1]) + " and " + names_[-1]) if len(names_) > 1 else names_[0], "config ensemble.members")

# ------------------------------------------------------------------ sample
block("sample (sample_flow.csv, features_purchase.pkl)")
fl = r("sample_flow.csv")
st_ = fl.set_index("step")["n_orders"]
m("NOrdersAll", big(one(fl[fl.step.str.startswith("All orders")], "sample_flow all")["n_orders"]))
m("NRuleOne", big(one(fl[fl.step.str.startswith("R1")], "sample_flow R1")["n_orders"]))
m("NSample", big(one(fl[fl.step.str.startswith("R2")], "sample_flow R2")["n_orders"]))
m("NCancelled", big(one(fl[fl.step.str.startswith("R1")], "sample_flow R1")["n_orders"]
                    - one(fl[fl.step.str.startswith("R2")], "sample_flow R2")["n_orders"]), "orders removed as cancelled or unavailable (R1 - R2)")
F = pd.read_pickle(OUT / "data" / "features_purchase.pkl")
dl = F[F["delivered"]]
m("NDeliveredAll", big(len(dl))); m("NNeverDelivered", big((~F["delivered"]).sum()))
rev = dl["review_score"].notna()
m("NReviewed", big(rev.sum())); m("ReviewShare", pct(rev.mean()))
m("YNinetyAll", f"{dl['Y'].quantile(0.9):.0f}", "90th percentile of the delivery time, delivered orders")
P = r("eda_promise_checks.csv").set_index("check")["value"]
m("OlistLenAll", f"{P['promise P in days: mean']:.1f}")
m("OlistLateAll", pct(P["late rate (S > 0) among delivered orders"]))
m("OlistQuantileAll", pct(1 - P["late rate (S > 0) among delivered orders"]))
m("OlistSlackAll", f"{P['median slack P - D among on-time orders (days)']:.0f}")
m("CorrApproval", f"{P['within seller-route-month correlation of P with approval_h']:.3f}")
m("CorrHandover", f"{P['within seller-route-month correlation of P with handover_d']:.3f}")

# ------------------------------------------------------------------ Olist in the evaluation windows
block("Olist's promise in the evaluation windows (rq1_window_metrics.csv)")
W = r("rq1_window_metrics.csv")
o = W[(W.variant == "olist") & (W.region == "all")].copy(); o["y"] = pd.to_datetime(o["origin"]).dt.year.astype(str)
m("OlistLenEval", f"{o['mean_promise'].mean():.1f}"); m("OlistCovEval", pct(o["coverage"].mean()))
m("OlistLateEval", pct(1 - o["coverage"].mean())); m("OlistCovAllEval", pct(o["coverage_all"].mean()))
m("OlistWorstEval", pct(W[(W.variant == "olist") & (W.region != "all")].groupby("k")["coverage"].min().mean()))
for y, g in o.groupby("y"):
    m(f"OlistCovYear{W2[y]}", pct(g["coverage"].mean())); m(f"OlistLenYear{W2[y]}", f"{g['mean_promise'].mean():.1f}")
Wt = r("rq1_window_types.csv")
m("NStressWindows", int(Wt["stress"].sum())); m("StressThreshold", pct(Wt["threshold"].iloc[0]))
m("NTestWindows", int(Wt["period"].str.startswith("2018").sum())); m("NSelWindows", int(Wt["period"].str.startswith("2017").sum()))

# ------------------------------------------------------------------ censoring and the tail share
block("censoring and the tail share (rq1_censoring_by_origin.csv, rq1_calibration_weights.csv, censoring.tail_share)")
Cb = r("rq1_censoring_by_origin.csv")
m("CalObsShare", pct(Cb["cal_observed_share"].mean()))
K = CFG["rolling"]["calib_days"]
obs_s, up_s, fin_s = [], [], []
for w in win:
    o_ = w["origin"]
    obs_s.append(tail_share(F, o_, K)[0]); up_s.append(tail_share(F, o_, K, mode="upper")[0])
    mk = (F["purchase_ts"] >= o_ - pd.Timedelta(days=K + 28)) & (F["purchase_ts"] < o_ - pd.Timedelta(days=K))
    X = F[mk & F["delivered"]]; fin_s.append(float((X["Y"] >= K).mean()))
m("TailObsMean", pct(np.mean(obs_s), 2)); m("TailFinMean", pct(np.mean(fin_s), 2)); m("TailUpMean", pct(np.mean(up_s), 1))
m("TailGapPoints", f"{100 * (np.mean(fin_s) - np.mean(obs_s)):.1f}", "final-snapshot minus observed tail share (pp)")
Cw = r("rq1_calibration_weights.csv")
m("CalNMean", big(Cw["n_cal"].mean())); m("CalNeffMean", big(Cw["n_eff"].mean()))

# ------------------------------------------------------------------ RQ1 summary
block("RQ1 coverage, length and worst region (rq1_summary.csv); naming Cov|Len|Worst|CovAll + variant + method + alpha")
S = r("rq1_summary.csv"); Sa = S[S.region == "all"]
for _, x in Sa.iterrows():
    if x["variant"] not in CODE or pd.isna(x["alpha"]):
        continue
    nm = CODE[x["variant"]] + METH[x["method"]] + ALPHA[round(x["alpha"], 2)]
    m("Cov" + nm, pct(x["coverage"])); m("Len" + nm, f"{x['mean_promise']:.1f}"); m("CovAll" + nm, pct(x["coverage_all"]))
    if pd.notna(x.get("worst_region")):
        m("Worst" + nm, pct(x["worst_region"]))
a10 = Sa[np.isclose(Sa.alpha, 0.10)]
nv = a10[a10.method == "split_naive"].set_index("variant"); iv = a10[a10.method == "split_ipcw"].set_index("variant")
m("NaiveShortLo", f"{100 * (0.9 - nv['coverage'].max()):.1f}"); m("NaiveShortHi", f"{100 * (0.9 - nv['coverage'].min()):.1f}")
m("IpcwCovLo", pct(iv["coverage"].min())); m("IpcwCovHi", pct(iv["coverage"].max()))
m("OracleGapQrf", f"{100 * (a10[(a10.variant == 'qrf_ipcw') & (a10.method == 'split_oracle')]['coverage'].iloc[0] - iv.loc['qrf_ipcw', 'coverage']):.1f}")
dall = (a10.set_index(["variant", "method"])["coverage"] - a10.set_index(["variant", "method"])["coverage_all"])
dall = dall[dall.index.get_level_values(1) != "actual"]
m("CovAllDrop", f"{100 * dall.mean():.1f}", "mean drop when never-delivered orders count as missed (90%)")
gd = (iv["coverage"] - nv["coverage"]) - (iv["coverage_all"] - nv["coverage_all"])
m("CovAllDiffMax", f"{np.ceil(100 * 100 * gd.abs().max()) / 100:.2f}", "upper bound (pp, 0.01 steps) of the change of IPCW - naive")

# ------------------------------------------------------------------ RQ1 contrasts
block("RQ1 contrasts (contrasts_rq1.csv); differences in pp for coverage, days for length")
C = r("contrasts_rq1.csv")


def ctr(fam_prefix, contrast):
    x = C[C.family.str.startswith(fam_prefix) & (C.contrast == contrast)]
    assert len(x) == 1, (fam_prefix, contrast, len(x))
    return x.iloc[0]


A = C[C.family.str.startswith("RQ1-A")]
m("RqOneIpcwGainLo", f"{100 * A['mean'].min():.1f}"); m("RqOneIpcwGainHi", f"{100 * A['mean'].max():.1f}")
Bc = C[C.family == "RQ1-B training weights (coverage)"]; Bl = C[C.family == "RQ1-B training weights (mean promise)"]
ml = Bc[~Bc.contrast.str.startswith(("aft", "route"))]
m("TrainWeightMaxAbs", f"{np.ceil(10000 * ml['mean'].abs().max()) / 100:.2f}", "bound (pp, 0.01 steps) on |IPCW - plain| coverage, quantile models")
m("TrainRouteHolm", holm(ctr("RQ1-B training weights (coverage)", "route: ipcw - plain")["p_holm"]))
m("TrainRouteCov", pts(ctr("RQ1-B training weights (coverage)", "route: ipcw - plain")["mean"]))
m("TrainRouteLen", num(ctr("RQ1-B training weights (mean promise)", "route: ipcw - plain")["mean"]))
m("AftCensCov", absn(100 * ctr("RQ1-B training weights (coverage)", "aft: censored - naive")["mean"]), "magnitude; sign checked by claim rq1b-small")
m("AftCensLen", absn(ctr("RQ1-B training weights (mean promise)", "aft: censored - naive")["mean"]), "magnitude; sign checked by claim rq1b-small")
Ff = C[C.family.str.startswith("RQ1-F")].copy(); Ff["v"] = Ff.contrast.str.replace(" - Olist", "", regex=False)
single = Ff[Ff.v != ens].sort_values("mean").iloc[0]
m("RqOneFBestLabel", VAR[single["v"]].replace("(plain)", "trained without weights").replace("(IPCW)", "with IPCW"),
  "single model with the shortest promise at Olist's coverage")
m("RqOneFBest", num(single["mean"])); m("RqOneFBestCi", ci(single["ci_lo"], single["ci_hi"])); m("RqOneFBestHolm", holm(single["p_holm"]))
e = Ff[Ff.v == ens].iloc[0]
m("RqOneEnsGap", num(e["mean"])); m("RqOneEnsGapAbs", f"{abs(e['mean']):.1f}"); m("RqOneEnsGapCi", ci(e["ci_lo"], e["ci_hi"]))
m("RqOneEnsGapHolm", holm(e["p_holm"]))
Fi = C[C.family.str.startswith("RQ1-I")].copy(); Fi["v"] = Fi.contrast.str.replace(" - Olist", "", regex=False); Fi = Fi.set_index("v")
m("RqOneEnsGapTest", num(Fi.loc[ens, "mean"])); m("RqOneEnsGapTestAbs", f"{abs(Fi.loc[ens, 'mean']):.1f}")
m("RqOneEnsGapTestCi", ci(Fi.loc[ens, "ci_lo"], Fi.loc[ens, "ci_hi"])); m("RqOneEnsGapTestHolm", holm(Fi.loc[ens, "p_holm"]))
for v in ("xgbq_ipcw", "lqr_ipcw", "qrf_ipcw"):
    m("RqOneI" + CODE[v], num(Fi.loc[v, "mean"]))
h = ctr("RQ1-H quantile average vs members, promise", f"{ens} - xgbq_ipcw")
ht = ctr("RQ1-H quantile average vs members, 2018", f"{ens} - xgbq_ipcw")
m("RqOneHXgb", num(h["mean"])); m("RqOneHXgbTest", num(ht["mean"])); m("RqOneHXgbTestCi", ci(ht["ci_lo"], ht["ci_hi"]))
E = C[C.family.str.startswith("RQ1-E") & C.contrast.str.contains("mondrian_ipcw - split_ipcw")]
m("RqOneEMondrianLo", num(100 * E["mean"].min(), 1)); m("RqOneEMondrianHi", num(100 * E["mean"].max(), 1))
g = ctr("RQ1-G test period 2018 (coverage)", "qrf_ipcw: mondrian_ipcw - split_ipcw")
m("RqOneGMondrianQrf", absn(100 * g["mean"]), "magnitude; sign checked by claim rq1e-mondrian-none"); m("RqOneGMondrianQrfHolm", holm(g["p_holm"]))
g = ctr("RQ1-G test period 2018 (coverage)", "qrf_ipcw: aci_ipcw - split_ipcw")
m("RqOneGAciQrf", pts(g["mean"], 1)); m("RqOneGAciQrfCi", ci(g["ci_lo"], g["ci_hi"], 1, 100)); m("RqOneGAciQrfHolm", holm(g["p_holm"]))
g = ctr("RQ1-G test period 2018 (worst-region coverage)", "qrf_ipcw: aci_ipcw - split_ipcw")
m("RqOneGAciQrfWorst", pts(g["mean"], 1))
Sel = r("rq1_ensemble_selection.csv").set_index("candidate")
m("SelGapFive", num(Sel.loc["ens_ipcw", "length_gap_2017"])); m("SelGapThree", num(Sel.loc["ens_xgbq_qrf_lqr", "length_gap_2017"]))
m("SelGapTwo", num(Sel.loc["ens_xgbq_qrf", "length_gap_2017"]))
Pin = r("rq1_pinball_summary.csv")
for v, tau, nm in (("lqr_ipcw", 0.9, "LqrNinety"), ("xgbq_ipcw", 0.9, "XgbNinety"), ("aft", 0.8, "AftEighty"),
                   ("qrf_ipcw", 0.9, "QrfNinety"), (ens, 0.9, "EnsNinety")):
    m("RawCov" + nm, pct(Pin[(Pin.variant == v) & np.isclose(Pin.tau, tau)]["coverage_raw"].iloc[0]))

# ------------------------------------------------------------------ periods, Mondrian, ACI
block("RQ1 periods (rq1_summary_by_period.csv), Mondrian by region, ACI")
Bp = r("rq1_summary_by_period.csv")
GK = {"2017 (selection)": "Sel", "2018 (test)": "Test", "stress": "Stress", "normal": "Normal"}
for _, x in Bp[(Bp.variant == base3) & np.isclose(Bp.alpha, 0.10) & Bp.group.isin(GK)].iterrows():
    nm = "Per" + CODE[base3] + METH[x["method"]] + GK[x["group"]]
    m(nm + "Cov", pct(x["coverage"])); m(nm + "Min", pct(x["coverage_min"])); m(nm + "Len", f"{x['mean_promise']:.1f}")
x = W[(W.variant == base3) & np.isclose(W.alpha, 0.10) & W.method.isin(["split_ipcw", "mondrian_ipcw"]) & (W.region != "all")]
pv = x.pivot_table(index="region", columns="method", values="coverage", aggfunc="mean")
for reg, nm in (("Norte", "North"), ("Nordeste", "Northeast"), ("Centro-Oeste", "Centrewest")):
    if reg not in pv.index:
        raise SystemExit(f"gen_results_tex: region {reg} missing from rq1_window_metrics.csv")
    m("MondrianDiff" + nm, absn(100 * (pv.loc[reg, "mondrian_ipcw"] - pv.loc[reg, "split_ipcw"]), 1), "magnitude; signs checked by claim mondrian-region-signs")
Th = r("theory_checks.csv")
c4 = Th[Th["claim"].str.contains("share of origins", regex=False)]["value"].iloc[0]
own = dict(re.findall(r"([\w-]+) ([0-9.]+)", c4.split("|")[0]))
if "Norte" not in own:
    raise SystemExit("gen_results_tex: the C4 INFO row of theory_checks.csv has no share for Norte")
m("MondrianOwnNorth", pct(float(own["Norte"]), 0))
Wr = W[(W.variant == base3) & (W.method == "split_ipcw") & np.isclose(W.alpha, 0.10) & (W.region != "all")]
worst_ = Wr.loc[Wr.groupby("k")["coverage"].idxmin(), "region"].value_counts(normalize=True)
m("MondrianWorstNorth", pct(worst_.get("Norte", 0.0), 0), "share of windows where the North is the worst region (claim north-worst-most)")
Sn = r("sensitivity_rq1.csv")
s1 = lambda setting, v, a, period="all windows": one(Sn[(Sn.setting == setting) & (Sn.variant == v) & np.isclose(Sn.alpha, a) & (Sn.period == period)], f"sensitivity {setting}/{v}/{a}/{period}")
m("AciMinTest", pct(s1(f"daily batch, gamma = {CFG['conformal']['aci_gamma']}", base3, 0.10, "2018 (test)")["coverage_min_window"]))
m("SplitMinTest", pct(s1("reference: split IPCW", base3, 0.10, "2018 (test)")["coverage_min_window"]))
onl = Sn[Sn.setting.str.startswith("online per-order") & (Sn.variant == base3) & np.isclose(Sn.alpha, 0.10)]
m("AciOnlineCapAll", pct(one(onl[onl.period == "all windows"], "online rule, all windows")["share_capped"]))
m("AciOnlineCapTest", pct(one(onl[onl.period == "2018 (test)"], "online rule, 2018")["share_capped"]))
gam = CFG["conformal"]["aci_gamma"]          # C6 and C5 recomputed as in experiments/theory_checks.py
xa = W[(W.variant == base3) & (W.method == "aci_ipcw") & np.isclose(W.alpha, 0.10) & (W.region == "all")]
mis = 1 - np.average(xa["coverage_all"], weights=xa["n_all"]); bnd = (max(0.1, 0.9) + gam) / (gam * xa["n_all"].sum())
m("AciMiscovTen", f"{mis:.3f}"); m("AciGapTen", f"{abs(mis - 0.1):.3f}"); m("AciBoundTen", f"{bnd:.3f}")
xo = W[(W.variant == base3) & (W.method == "split_oracle") & np.isclose(W.alpha, 0.10) & (W.region == "all")]
m("OracleBelowTen", int((xo["coverage"] < 0.9 - 0.02).sum()))

# ------------------------------------------------------------------ RQ1 sensitivity
block("RQ1 sensitivity (sensitivity_rq1.csv)")
MAIN = "main (split IPCW, 42 d, log score, cap 20)"
for v, nm in ((base3, "Qrf"), ("xgbq_ipcw", "Xgb")):
    for a, an in ((0.05, "Five"), (0.10, "Ten")):
        x = s1(MAIN, v, a)
        m(f"Main{nm}Cov{an}", pct(x["coverage"])); m(f"Main{nm}Len{an}", f"{x['mean_promise']:.1f}")
cm_ = s1(MAIN, base3, 0.10)["share_capped"]
m("CapMain", "no capped promise" if cm_ == 0 else "fewer than 0.1\\% of promises capped" if cm_ < 0.0005
  else f"{pct(cm_)} of promises capped")
WORDS = {14: "Fourteen", 21: "TwentyOne", 28: "TwentyEight"}
assert tuple(K_SUBS) == (14, 21, 28), "the text names the windows 14, 21 and 28 days"
for k_ in K_SUBS:
    m("CapWin" + WORDS[k_], pct(s1(f"calibration window {k_} d", base3, 0.10)["share_capped"], 1))
TU = "tail share upper bound (undelivered at the origin counted as tail)"
m("SensTailUpQrfCov", pct(s1(TU, base3, 0.10)["coverage"]))
m("TailUpCovGainQrf", f"{100 * (s1(TU, base3, 0.10)['coverage'] - s1(MAIN, base3, 0.10)['coverage']):.1f}")
m("TailUpLenGainQrf", f"{s1(TU, base3, 0.10)['mean_promise'] - s1(MAIN, base3, 0.10)['mean_promise']:.1f}")
m("SensTailUpQrfCapFive", pct(s1(TU, base3, 0.05)["share_capped"])); m("SensTailUpQrfLenFive", f"{s1(TU, base3, 0.05)['mean_promise']:.1f}")
NE = "level with the Kish effective size, (1 + 1/n_eff)"; TM = "level reserving the largest test weight, (1 + wbar / sum w)"
dn = [abs(s1(NE, v, a)["coverage"] - s1(MAIN, v, a)["coverage"]) for v in (base3, "xgbq_ipcw") for a in CFG["alphas"]]
m("NeffCovChange", f"{max(0.01, np.ceil(10000 * max(dn)) / 100):.2f}", "bound (pp, 0.01 steps)")
dt = [s1(TM, v, a)["coverage"] - s1(MAIN, v, a)["coverage"] for v in (base3, "xgbq_ipcw") for a in CFG["alphas"]]
m("TestMassCovLo", f"{100 * min(dt):.1f}"); m("TestMassCovHi", f"{100 * max(dt):.1f}")
m("FiniteCorrMax", f"{np.ceil(10 * 100 * max(max(abs(v) for v in dt), max(dn))) / 10:.1f}", "largest |coverage change| of the two corrections (pp)")
MINOR = ["additive score", "IPCW cap 10", "IPCW cap 50", "Mondrian by state"]
ref_m = Sn[(Sn.setting == "Mondrian by region (main)") & (Sn.variant == base3) & np.isclose(Sn.alpha, 0.10) & (Sn.period == "all windows")]
dc, dl_ = [], []
for st in MINOR:
    ref = one(ref_m, "Mondrian main") if st == "Mondrian by state" else s1(MAIN, base3, 0.10)
    x = s1(st, base3, 0.10); dc.append(abs(x["coverage"] - ref["coverage"])); dl_.append(abs(x["mean_promise"] - ref["mean_promise"]))
m("SensMinorCov", f"{np.ceil(1000 * max(dc)) / 10:.1f}"); m("SensMinorLen", f"{np.ceil(10 * max(dl_)) / 10:.1f}")
TR = f"truncated split conformal, follow-up >= {TRUNC_C0} d (finite-sample valid)"
m("TruncCov", pct(s1(TR, base3, 0.10)["coverage"])); m("TruncCovAll", pct(s1(TR, base3, 0.10)["coverage_all"]))
m("TruncLen", f"{s1(TR, base3, 0.10)['mean_promise']:.1f}"); m("TruncCap", pct(s1(TR, base3, 0.10)["share_capped"], 0))
m("SensOlistInputXgbLen", f"{s1(chr(79) + 'list' + chr(39) + 's estimate as an input', 'xgbq_ipcw', 0.10)['mean_promise']:.1f}")

# ------------------------------------------------------------------ RQ2
block("RQ2 (rq2_*.csv)")
V = r("rq2_s_versus_d.csv").set_index("model")["cv_logloss"]
m("CvllS", f"{V['f(S)']:.4f}"); m("CvllD", f"{V['f(D)']:.4f}"); m("CvllSP", f"{V['f(S) + g(P)']:.4f}"); m("CvllP", f"{V['g(P)']:.4f}")
Ab = r("rq2_aipw_bins.csv"); X = Ab[Ab.learner == "xgb"].set_index("bin")
m("AipwRefRisk", pct(X.loc[CFG["rq2"]["ref_bin"], "risk"]))
for b, nm in ((0, "Zero"), (3, "Three"), (4, "Four"), (5, "Five")):
    m("RRBin" + nm, f"{X.loc[b, 'rr']:.2f}")
m("RqTwoRRFirstLate", f"{X.loc[3, 'rr']:.1f}"); m("EvalueMax", f"{Ab['e_value'].max():.2f}")
ov = Th[(Th.check == "C9") & Th.claim.str.contains("every overlap specification")]
m("OverlapSpecs", re.findall(r"(\d+) specifications", ov["value"].iloc[0])[0])
Mm = r("rq2_survey_mechanism.csv").set_index(["late", "answered_before_delivery"])
m("LateBeforeShare", pct(Mm.loc[(1, 1), "share_within_late_group"])); m("LateLowBefore", pct(Mm.loc[(1, 1), "mean"]))
m("LateLowAfter", pct(Mm.loc[(1, 0), "mean"])); m("OnTimeBeforeShare", pct(Mm.loc[(0, 1), "share_within_late_group"]))
R = r("rq2_rdd.csv"); R0 = R[(R.cutoff == 0) & np.isclose(R["M"], R["M_rule_of_thumb"])]
for sub, nm in (("all", "All"), ("answered_after_arrival", "After")):
    x = R0[(R0.subset == sub) & (R0.h == 5)].iloc[0]
    m(f"Rdd{nm}HFive", num(100 * x["estimate"], 1)); m(f"Rdd{nm}HFiveCi", ci(x["ci_lo"], x["ci_hi"], 1, 100))
    if nm == "All":
        m("RddAllHFiveHalf", f"{50 * (x['ci_hi'] - x['ci_lo']):.0f}")
aa = R0[(R0.subset == "answered_after_arrival") & R0.h.isin([3, 5, 7])]["estimate"]
m("RddAfterLo", f"{100 * aa.min():.1f}"); m("RddAfterHi", f"{100 * aa.max():.1f}")
Pl = R[(R.cutoff != 0) & np.isclose(R["M"], R["M_rule_of_thumb"])]
m("RddPlaceboCover", pct(((Pl.ci_lo <= 0) & (Pl.ci_hi >= 0)).mean(), 0)); m("RddPlaceboN", len(Pl))
Lb = r("rq2_lee_bounds.csv").set_index("left")
Lb = Lb.loc[["(0,)", "(-1, 0)"]].reset_index()
m("LeeShareLeft", pct(Lb.loc[0, "share_after_left"])); m("LeeShareRight", pct(Lb.loc[0, "share_after_right"]))
m("LeeCellTrim", pct(Lb.loc[0, "trim_share"])); m("LeeCellBounds", f"{100 * Lb.loc[0, 'bound_lo']:.1f} to {100 * Lb.loc[0, 'bound_hi']:.1f}")
m("LeeCellCi", f"{100 * Lb.loc[0, 'ci_lo']:.1f} to {100 * Lb.loc[0, 'ci_hi']:.1f}")
m("LeeTwoCellsBounds", f"{100 * Lb.loc[1, 'bound_lo']:.1f} to {100 * Lb.loc[1, 'bound_hi']:.1f}")
Rg = r("rq2_aipw_bins_by_region.csv"); l4 = Rg[(Rg.learner == "lr") & (Rg["bin"] >= 4)]
m("RegionRdLateMin", f"{100 * l4['rd'].min():.0f}"); m("RegionRdLateMax", f"{100 * l4['rd'].max():.0f}")

# ------------------------------------------------------------------ RQ3
block("RQ3 (rq3_matched.csv, contrasts_rq3.csv, rq3_region_promises.csv, checks)")
Q = r("rq3_matched.csv"); C3 = r("contrasts_rq3.csv")
c3 = lambda fam, con: one(C3[C3.family.str.startswith(fam) & (C3.contrast == con)], f"RQ3 {fam}/{con}")
m("RqThreeOlistLen", f"{Q['L_olist'].mean():.1f}"); m("RqThreeOlistLow", pct(Q["low_olist"].mean(), 2))
m("RqThreeOlistLate", pct(Q["late_olist"].mean(), 2))
x = c3("RQ3 low-rating rate", "optimal - Olist")
m("RqThreeGainAbs", f"{abs(100 * x['mean']):.2f}"); m("RqThreeGainCi", ci(x["ci_lo"], x["ci_hi"], 2, 100))
m("RqThreeGainAbsCi", ci(-x["ci_hi"], -x["ci_lo"], 2, 100), "interval of the reduction (sign flipped)")
m("RqThreeLateGainAbs", f"{abs(100 * c3('RQ3 late rate', 'optimal - Olist')['mean']):.2f}")
m("RqThreeLenGainAbs", f"{abs(c3('RQ3 promise length', 'optimal - Olist')['mean']):.2f}")
x = c3("RQ3 rating-aware", "optimal - late_optimal")
m("RqThreeVsLate", pts(x["mean"])); m("RqThreeVsLateCi", ci(x["ci_lo"], x["ci_hi"], 2, 100)); m("RqThreeVsLateHolm", holm(x["p_holm"]))
m("RqThreeVsCpd", pts(c3("RQ3 rating-aware", "optimal - cpd_quantile")["mean"]))
d = Q["optimal_low_realised_at_L"] - Q["low_olist"]; st = Q["k"].isin(Wt.loc[Wt["stress"], "k"])
m("RqThreeWinShare", pct((d < 0).mean(), 0)); m("RqThreeGainStress", absn(100 * d[st].mean())); m("RqThreeGainNormal", absn(100 * d[~st].mean()))
m("RqThreeAtMax", pct(Q["optimal_share_at_max_P"].mean(), 1)); m("OlistAboveMax", pct(Q["olist_share_above_max_P"].mean(), 2))
Rp = r("rq3_region_promises.csv").groupby(["region", "policy"])[["mean_promise", "late_realised", "low_realised"]].mean()
for reg, nm in (("Nordeste", "Nordeste"), ("Norte", "Norte")):
    for pol, pn in (("olist", "Olist"), ("optimal", "Optimal")):
        m(f"RegionPromise{nm}{pn}", f"{Rp.loc[(reg, pol), 'mean_promise']:.1f}")
        m(f"RegionLate{nm}{pn}", pct(Rp.loc[(reg, pol), "late_realised"]))
        m(f"RegionLow{nm}{pn}", pct(Rp.loc[(reg, pol), "low_realised"]))
G = r("rq3_duality_gap.csv"); m("DualityGapMax", f"{100 * G['gap'].max():.4f}"); m("DualityCases", len(G))
# The policy tables interpolate the coarse sweep (rq3_policy_windows.csv) linearly between adjacent points: the model
# value of that randomised mix, minus the dual bound LB(L) of the fine sweep, at the integer lengths of check C10.
_Wo = r("rq3_policy_windows.csv"); _Wo = _Wo[_Wo.policy == "optimal"]
_ig = []
for _k, _g in G.groupby("k"):
    _o = _Wo[_Wo.k == _k].sort_values("mean_promise")
    for _L, _lb in zip(_g["L"], _g["LB"]):
        if _o["mean_promise"].min() <= _L <= _o["mean_promise"].max():
            _ig.append(np.interp(_L, _o["mean_promise"], _o["low_expected"]) - _lb)
_ig = 100 * np.array(_ig)
m("InterpGapMax", f"{_ig.max():.2f}", "coarse-sweep interpolation minus dual bound, max over window x L (pp)")
m("InterpGapMean", f"{_ig.mean():.2f}", "same, mean (pp)")
Cv = r("rq3_convexity.csv"); m("ConvexAll", pct(Cv["share_convex"].mean(), 0)); m("ConvexMid", pct(Cv["share_convex_P15_40"].mean(), 0))
Cc = r("rq3_curve_calibration.csv"); oos = Cc[Cc["out_of_sample"]]
m("CurveOosBias", f"{100 * (oos['predicted_fit2017'] - oos['observed']).mean():+.2f}".replace("-", "$-$"))
m("CurveOosCorr", f"{np.corrcoef(oos['predicted_fit2017'], oos['observed'])[0, 1]:.2f}")

block("RQ3 sensitivity (contrasts_rq3_sensitivity.csv)")
Cs = r("contrasts_rq3_sensitivity.csv")
cs = lambda fam, con: one(Cs[Cs.family.str.contains(fam, regex=False) & (Cs.contrast == con)], f"RQ3-S {fam}/{con}")
x = cs("low-rating rate at Olist's mean promise (all windows)", "optimal - Olist: quantile-average CPD")
m("SensRqThreeEnsAbs", f"{abs(100 * x['mean']):.2f}")
x = cs("versus forest CPD (low-rating", "optimal: quantile-average CPD - forest CPD")
m("SensRqThreeEnsVsForest", pts(x["mean"])); m("SensRqThreeEnsVsForestHolm", holm(x["p_holm"]))
m("SensRqThreeEnsVsForestCoverage", pts(cs("versus forest CPD (low-rating", "coverage policy: quantile-average CPD - forest CPD")["mean"]))
x = cs("low-rating rate at Olist's mean promise (2018 test)", "optimal - Olist: policy from the 2017 curve")
m("RqThreeGainTestAbs", f"{abs(100 * x['mean']):.2f}"); m("RqThreeGainTestAbsCi", ci(-x["ci_hi"], -x["ci_lo"], 2, 100))
x = cs("low-rating rate at Olist's mean promise (2018 test)", "optimal - Olist: forest CPD (main)")
m("RqThreeGainTestMainAbs", f"{abs(100 * x['mean']):.2f}", "main policy (full-sample curve), 2018 windows")
Q7 = r("rq3_matched_fit2017.csv"); t7 = pd.to_datetime(Q7["origin"]).dt.year == 2018
m("RqThreeTestWinShare", pct(((Q7["optimal_low_realised_at_L"] - Q7["low_olist"])[t7] < 0).mean(), 0))
m("SensRqThreeNeutral", pts(cs("low-rating rate at Olist's mean promise (all windows)", "optimal - Olist: three-star ratings as not low")["mean"]))
tl = Cs[Cs.family.str.startswith("RQ3-S low-rating rate at Olist's mean promise (all windows)") & Cs.contrast.str.contains("tail mass")]
if len(tl) != 4:
    raise SystemExit("gen_results_tex: expected 4 tail-location rows in contrasts_rq3_sensitivity.csv")
m("SensRqThreeTailLo", absn(100 * tl["mean"].abs().min())); m("SensRqThreeTailHi", absn(100 * tl["mean"].abs().max()))

# ------------------------------------------------------------------ checks, tests, run
block("checks, tests and the run (theory_checks.csv, tests, outputs_extra, manifest.json, logs/pipeline.log)")
m("ChecksRows", len(Th)); m("NChecks", Th["check"].nunique())
m("ChecksPass", int((Th.status == "PASS").sum())); m("ChecksInfo", int((Th.status == "INFO").sum()))
m("ChecksFail", {0: "none", 1: "one", 2: "two", 3: "three"}.get(int((Th.status == "FAIL").sum()), str(int((Th.status == "FAIL").sum()))))
c1 = Th[Th.check == "C1"]["value"].iloc[0]; n1 = re.findall(r"(\d+) violations in (\d+) orders", c1)[0]
m("CheckOneViol", n1[0]); m("CheckOneN", big(int(n1[1])))
c2 = Th[(Th.check == "C2") & (Th.status == "PASS")]["value"].iloc[0]
m("CheckTwoShare", re.findall(r"closer in (\d+)%", c2)[0] + "\\%")
m("NUnitTests", len(re.findall(r"^def test_", (ROOT / "tests" / "test_p2du.py").read_text(encoding="utf-8"), re.M)))
cx = ROOT / "outputs_extra" / "tables" / "crosscheck_summary.csv"
if not cx.is_file():
    print("WARNING: outputs_extra/tables/crosscheck_summary.csv not found; \\NCrosscheck is --")
m("NCrosscheck", int(pd.read_csv(cx)["n"].sum()) if cx.is_file() else "--")
man = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
m("PythonVersion", man["python"])
for k_, nm in (("numpy", "Numpy"), ("pandas", "Pandas"), ("scikit-learn", "Sklearn"), ("xgboost", "Xgboost"),
               ("statsmodels", "Statsmodels"), ("quantile-forest", "QuantileForest")):
    m("Ver" + nm, man["libs"][k_])
plat = man["platform"]
log = OUT / "logs" / "pipeline.log"
rt, nproc = "--", None
if log.is_file():
    raw = log.read_bytes()
    txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
    # the last full run: "Pipeline finished in X min" (pipeline.py); for an older log, the sum of the stage durations.
    # Time stamps are not used directly, because the parallel fitting processes print them out of order.
    fin = re.findall(r"Pipeline finished in ([0-9.]+) min", txt)
    dur = re.findall(r"Stage \w+: done \(([0-9.]+) min\)", txt)
    rt = f"{max(map(float, fin)):.0f}" if fin else (f"{sum(map(float, dur)):.0f}" if dur else "--")
    npr = re.findall(r"Pipeline finished in ([0-9.]+) min \(\d+ stages, (\d+) fitting process", txt)
    nproc = int(max(npr, key=lambda t: float(t[0]))[1]) if npr else None   # the full run: the longest one in the log
if rt == "--":
    print("WARNING: no full-run pipeline.log; \\RunTimeMin is -- (run run_pipeline.bat without arguments)")
m("RunTimeMin", rt, "wall time of the logged run" if rt != "--" else "no pipeline.log: fill after the full run")
os_ = ("Windows 11" if plat.startswith("Windows-11") else "Windows") if plat.lower().startswith("windows") \
    else "Linux"
words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
m("RunMachine", f"a {os_} computer" + (f" with {words.get(nproc, nproc)} parallel fitting process"
                                        + ("es" if nproc != 1 else "") if nproc else ""), plat)
m("VerScipy", man["libs"].get("scipy", "--"))
m("RunOS", os_)
m("RunWorkers", words.get(nproc, nproc) if nproc else "--", "parallel fitting processes (pipeline.log)")
m("RunThreadsModel", words.get(CFG.get("n_jobs", 2), CFG.get("n_jobs", 2)), "config n_jobs: threads per XGBoost model or forest")
m("RunSeed", CFG["seed"])
_st = {}
if log.is_file():
    for nm_, mn_ in re.findall(r"Stage (\w+): done \(([0-9.]+) min\)", txt):
        _st[nm_] = float(mn_)                       # the last occurrence of each stage wins
m("RunStages", len(_st) if _st else "--", "stages of the logged run")
m("RunFitMin", f"{_st.get('rq1_fit', np.nan) + _st.get('sens_fit', np.nan):.0f}" if _st else "--",
  "rq1_fit + sens_fit (min)")

# ---- computational budget (Section 5.5, Table tab:budget): number of model fits, from config.yaml and the code
from p2du.experiments.rq1_fit import VARIANTS as _VARIANTS                       # noqa: E402
from p2du.experiments.sensitivity import SENS_VARIANTS as _SV, ROUTE_M as _RM, HIGH_TAUS as _HT  # noqa: E402
_K = len(rolling_windows(CFG))
_TC = r("rq1_tuning_choices.csv"); _ntuned = int(_TC["tuned_here"].sum())
_gx, _gq = len(CFG["tuning"]["xgbq_grid"]), len(CFG["tuning"]["qrf_grid"])
_ga = len(CFG["models"]["aft"]["distributions"]) * len(CFG["models"]["aft"]["scales"])
_members = CFG["ensemble"]["members"]
_fold = CFG["rq2"]["folds"]
_regions = r("rq2_aipw_bins_by_region.csv")["region"].nunique()
budget = {
    "Main": _K * len(_VARIANTS),
    "Tune": _ntuned * (_gx + _gq + _ga),
    "Promise": _K * len(_SV) + _ntuned * (_gx + _gq),
    "Route": _K * len(_RM),
    "Highq": _K * len(_members),
    "Svd": 5 * (5 + 1),
    "Aipw": 2 * _fold * 2 + _regions * _fold * 2,
    "Curve": 3,
}
budget["RqOne"] = budget["Main"] + budget["Tune"] + budget["Promise"] + budget["Route"] + budget["Highq"]
budget["RqTwo"] = budget["Svd"] + budget["Aipw"] + budget["Curve"]
budget["Total"] = budget["RqOne"] + budget["RqTwo"]
for k_, v_ in budget.items():
    m("NFits" + k_, f"{v_:,}".replace(",", "{,}"))
m("RqTwoFolds", _fold); m("NVariants", len(_VARIANTS)); m("NTunedOrigins", _ntuned); m("NGridAft", _ga)
m("NGridXgbN", _gx); m("NGridQrfN", _gq); m("NHighTaus", len(_HT)); m("NRegions", _regions)
m("NRqThreeRuns", len(list(T.glob("rq3_policy_windows*.csv"))), "main policy run and its sensitivity runs")
m("NLambdaCoarse", CFG["rq3"]["lambda_grid"] + 1)
_c = [r(f) for f in ("contrasts_rq1.csv", "contrasts_rq3.csv", "contrasts_rq3_sensitivity.csv")]
m("NContrasts", sum(len(x) for x in _c)); m("NFamilies", sum(x["family"].nunique() for x in _c))
from p2du.experiments.rq2_response import lee_bounds as _lee                     # noqa: E402
from p2du.reporting.inference import N_BOOT_P as _nbp                          # noqa: E402
m("NBootP", big(_nbp), "bootstrap replicates of the p-values (reporting/inference.py)")
m("NBootLee", f"{inspect.signature(_lee).parameters['n_boot'].default:,}".replace(",", "{,}"), "rq2_response.lee_bounds")

# ------------------------------------------------------------------ RQ1-J and route shrinkage (stage sens_fit + sensitivity)
block("RQ1-J: uncalibrated quantile promise against split IPCW at Olist's coverage; route shrinkage")
TAB_RAW = TAB_ROUTE = None
Cj = C[C.family.str.startswith("RQ1-J")]
if len(Cj) and (T / "rq1_raw_matched.csv").is_file():
    Rm = r("rq1_raw_matched.csv"); Mo_ = r("rq1_matched_olist.csv")
    olen = Mo_.groupby("k")["olist_length"].first()
    rows_ = []
    for v in [x for x in CFG["ensemble"]["members"]] + [ens]:
        x = one(Cj[Cj.contrast == f"{v}: raw - split_ipcw"], f"RQ1-J {v}")
        g = Rm[Rm.variant == v].set_index("k")
        gap_o = (g["raw_length_at_olist_coverage"] - olen.reindex(g.index)).mean()
        rows_.append((VAR[v], int(x["n_windows"]), num(gap_o), f"{num(x['mean'])} {ci(x['ci_lo'], x['ci_hi'])}",
                      "$<$0.001" if x["p_holm"] < 0.001 else f"{x['p_holm']:.3f}"))
        m("RawVsSplit" + CODE[v], num(x["mean"])); m("RawVsSplit" + CODE[v] + "Abs", absn(x["mean"]))
        m("RawVsSplit" + CODE[v] + "Ci", ci(x["ci_lo"], x["ci_hi"])); m("RawVsSplit" + CODE[v] + "Holm", holm(x["p_holm"]))
        m("RawReach" + CODE[v], int(x["n_windows"]))
    mm_ = Cj["mean"]
    m("RawVsSplitLo", num(mm_.min())); m("RawVsSplitHi", num(mm_.max()))
    m("RawVsSplitAbsLo", absn(mm_.abs().min())); m("RawVsSplitAbsHi", absn(mm_.abs().max()))
    m("RawNSig", int((Cj["p_holm"] < 0.05).sum())); m("RawNModels", len(Cj))
    m("RawReachMin", int(Cj["n_windows"].min()))
    TAB_RAW = "\n".join(" & ".join(map(str, rw)) + " \\\\" for rw in rows_)
else:
    print("WARNING: rq1_raw_matched.csv or family RQ1-J missing: run the stages sens_fit, sensitivity and inference")
    for nm_ in ("RawVsSplitEns", "RawVsSplitEnsAbs", "RawVsSplitEnsCi", "RawVsSplitEnsHolm", "RawVsSplitLo", "RawVsSplitHi",
                "RawVsSplitAbsLo", "RawVsSplitAbsHi", "RawNSig", "RawNModels", "RawReachMin"):
        m(nm_, "--")
if (T / "rq1_route_shrinkage.csv").is_file():
    Rs = r("rq1_route_shrinkage.csv")
    m0 = CFG["models"]["route"]["shrink_m"]
    rows_ = []
    for mv, g in Rs.groupby("shrink_m", sort=True):
        g = g.set_index(g["alpha"].round(2))
        cells = []
        for a_ in (0.05, 0.10, 0.20):
            cells += [f"{100 * g.loc[a_, 'coverage']:.1f}", f"{g.loc[a_, 'mean_promise']:.1f}", f"{100 * g.loc[a_, 'worst_region']:.1f}"]
        rows_.append(" & ".join([f"$\\kappa = {int(mv)}$" + (" (main)" if mv == m0 else "")] + cells + [num(g["length_gap_at_olist_coverage"].iloc[0])]) + " \\\\")
    TAB_ROUTE = "\n".join(rows_)
    t10 = Rs[np.isclose(Rs.alpha, 0.10)].set_index("shrink_m")
    if m0 not in t10.index:
        raise SystemExit("gen_results_tex: the configured route shrinkage is missing from rq1_route_shrinkage.csv")
    m("RouteShrinkCovMax", f"{np.ceil(1000 * (t10['coverage'] - t10.loc[m0, 'coverage']).abs().max()) / 10:.1f}", "pp, 90% target")
    m("RouteShrinkLenMax", f"{np.ceil(10 * (t10['mean_promise'] - t10.loc[m0, 'mean_promise']).abs().max()) / 10:.1f}", "days, 90% target")
    m("RouteShrinkGapLo", num(t10["length_gap_at_olist_coverage"].min())); m("RouteShrinkGapHi", num(t10["length_gap_at_olist_coverage"].max()))
    m("RouteShrinkMs", ", ".join(str(int(x)) for x in sorted(t10.index)))
else:
    print("WARNING: rq1_route_shrinkage.csv missing: run the stages sens_fit and sensitivity")
    for nm_ in ("RouteShrinkCovMax", "RouteShrinkLenMax", "RouteShrinkGapLo", "RouteShrinkGapHi", "RouteShrinkMs"):
        m(nm_, "--")

# ------------------------------------------------------------------ graphical abstract (plain numbers for TikZ)
block("Graphical abstract: plain numbers (no % sign) used for bar heights in Figs/ga.tex")
xq = lambda meth: W[(W.variant == base3) & (W.method == meth) & np.isclose(W.alpha, 0.10) & (W.region == "all")]["coverage"].mean()
m("GaCovNaive", f"{100 * xq('split_naive'):.1f}"); m("GaCovIpcw", f"{100 * xq('split_ipcw'):.1f}"); m("GaCovOracle", f"{100 * xq('split_oracle'):.1f}")
for b_, nm in ((0, "Zero"), (1, "One"), (2, "Two"), (3, "Three"), (4, "Four"), (5, "Five")):
    m("GaRRBin" + nm, f"{X.loc[b_, 'rr']:.2f}")
for fam_, nm in (("uniform", "Uniform"), ("route", "Route"), ("cpd_quantile", "Cpd"), ("late_optimal", "LateOpt"), ("optimal", "Opt")):
    x = c3("RQ3 low-rating rate", f"{fam_} - Olist")
    m("GaLow" + nm, f"{100 * x['mean']:.2f}")
m("GaLateBefore", f"{100 * Mm.loc[(1, 1), 'share_within_late_group']:.1f}"); m("GaLateLowBefore", f"{100 * Mm.loc[(1, 1), 'mean']:.1f}")
m("GaLateLowAfter", f"{100 * Mm.loc[(1, 0), 'mean']:.1f}")

# ------------------------------------------------------------------ write results.tex
MAC.mkdir(parents=True, exist_ok=True)
names = [n for _, b in blocks for n, _, _ in b]
dup = {n for n in names if names.count(n) > 1}
if dup:
    raise SystemExit(f"duplicate macro names: {sorted(dup)}")
lines = ["%% =====================================================================",
         "%%  results.tex -- every number printed in the manuscript. Cite the macro, never a value.",
         f"%%  Generated by tools/gen_results_tex.py from {OUT.name}/ of the run recorded in manifest.json",
         f"%%  ({man['time']}, Python {man['python']}, {man['platform']}, code {man['code_sha256'][:12]}). Do not edit.",
         "%% ====================================================================="]
for title, items in blocks:
    lines += ["", f"%% ---- {title}"]
    for n, v, note in items:
        lines.append(f"\\newcommand{{\\{n}}}{{{v}}}" + (f"  % {note}" if note else ""))
(MAC / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ table bodies
BOLD = "\x01"      # a cell that starts with this mark is set in bold (rules in the functions below)


def tex_cell(x):
    s = "" if pd.isna(x) else str(x)
    if s.startswith(BOLD):
        return "\\textbf{" + tex_cell(s[1:]) + "}"
    s = s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
    s = s.replace("<=", "$\\le$").replace(">=", "$\\ge$").replace("<0.001", "$<$0.001")
    s = re.sub(r"(?<![\w.])-(?=\d)", "$-$", s)          # minus signs of numbers
    return s


def rows_from(csv, transform=None, cols=None):
    D_ = pd.read_csv(MS / csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if transform:
        D_ = transform(D_)
    if cols:
        D_ = D_[cols]
    return "\n".join(" & ".join(tex_cell(v) for v in row) + " \\\\" for row in D_.itertuples(index=False))


def contrasts_readable(D_):
    rep = {f"{k}:": f"{VAR[k]}:" for k in VAR}
    D_ = D_.copy()
    D_["Contrast"] = D_["Contrast"].map(lambda c: next((c.replace(k, v, 1) for k, v in rep.items() if c.startswith(k)), c))
    D_["Contrast"] = D_["Contrast"].map(lambda c: next((c.replace(k, VAR[k], 1) for k in VAR if c.startswith(k + " ")), c))
    for k_, v_ in (("mondrian_aci_ipcw", "Mondrian ACI"), ("mondrian_ipcw", "Mondrian IPCW"), ("aci_ipcw", "ACI"),
                   ("split_ipcw", "split IPCW"), ("ipcw - plain", "IPCW $-$ plain"), ("IPCW - naive", "IPCW $-$ naive"),
                   ("censored - naive", "censored $-$ delivered only"), (" - Olist", " $-$ Olist"), (" - split", " $-$ split")):
        D_["Contrast"] = D_["Contrast"].str.replace(k_, v_, regex=False)
    D_["Family"] = D_["Family"].str.replace(r"alpha=(0\.\d+)", lambda mm: f"target {100 * (1 - float(mm.group(1))):.0f}%", regex=True)
    return D_


# The claims of theory_checks.csv name the results by number; in the table they become cross-references to the
# labels of the article (\Mref: the supplementary reads them from main.aux), so a renumbering cannot leave them stale.
THEORY_REFS = (
    ("Y $\\le$ U implies a kept promise floor(U)", "$Y \\le U$ implies a kept promise $\\lfloor U \\rfloor$"),
    ("Pr(Y $\\le$ t)", "$\\Pr(Y \\le t)$"),
    ("rho\\_i(P)", "$\\rho_i(P)$"),
    ("r(s)", "$r(s)$"),
    ("(Proposition 5, part 3, Algorithm 3)", "(Proposition~\\Mref{prop:lagrange}, part~3; Algorithm~\\Mref{alg:sweep})"),
    ("(Proposition 5, part 2)", "(Proposition~\\Mref{prop:lagrange}, part~2)"),
    ("(Proposition 1)", "(Proposition~\\Mref{prop:promise})"),
    ("(Proposition 2)", "(Proposition~\\Mref{prop:ident})"),
    ("(Lemma 2 within regions)", "(Lemma~\\Mref{lem:split} within regions)"),
    ("(Lemma 2)", "(Lemma~\\Mref{lem:split})"),
    ("(Lemma 3)", "(Lemma~\\Mref{lem:rdd})"),
    ("(Lemma 4)", "(Lemma~\\Mref{lem:mono})"),
    ("(Eq. 10)", "(Eq.~\\Meqref{eq:gformula})"),
    ("premise of the threshold form, Remark 7", "premise of marginal allocation, Section~\\Mref{sec:policy}"),
    (" (Crump et al., 2009)", "~\\citep{Crump2009}"),
    ("selected by the pre-specified rule", "selected by the rule fixed before the reported run"),
)


def theory_refs(body):
    for a, b in THEORY_REFS:
        body = body.replace(a, b)
    body = re.sub(r"alpha = (0\.\d+)", r"$\\alpha = \1$", body)
    return body


def sens_readable(D_):
    """Table 9 shows the base model of RQ3 (the quantile forest); the XGBoost rows stay in table7_sensitivity.csv."""
    D_ = D_[D_["Model"] == "Quantile forest"].drop(columns="Model")
    # Corollary 1 holds under exchangeability (and Assumption 1); the label of the run omits the condition
    return D_.apply(lambda c: c.str.replace("(finite-sample valid)", "(finite-sample valid under exchangeability)",
                                            regex=False))


# ---- bold cells: each rule is stated in the caption of its table
def _lead(v):
    """Leading number of a cell such as '-0.96 [-1.68, -0.47]' or '12.7'; NaN when there is none."""
    m_ = re.match(r"\s*(-?\d[\d,]*\.?\d*)", str(v))
    return float(m_.group(1).replace(",", "")) if m_ else np.nan


def _bold_extreme(D_, cols, how):
    D_ = D_.copy()
    for c_ in cols:
        v = D_[c_].map(_lead)
        if v.notna().any():
            best = v.max() if how == "max" else v.min()
            D_.loc[v == best, c_] = BOLD + D_.loc[v == best, c_]
    return D_


def sample_bold(D_):
    """Table 5: the largest delivery times, late rate and low-rating rate among the regions."""
    return _bold_extreme(D_, [c for c in D_.columns if c.startswith(("Median Y", "90th pct", "Late", "Low"))], "max")


def models_bold(D_):
    """Table 6: the shortest promise at Olist's coverage (most negative difference), all windows and 2018."""
    return _bold_extreme(D_, [c for c in D_.columns if "Olist" in c or c.startswith("Same")], "min")


def methods_bold(D_):
    """Table 7: stress-window coverage of split IPCW and of the oracle (both below the 90% target, see the text)."""
    D_ = D_.copy(); c_ = "stress windows: cov."
    m_ = D_["Method"].isin(["Split, IPCW", "Split, oracle (infeasible)"])
    D_.loc[m_, c_] = BOLD + D_.loc[m_, c_]
    return D_


def rq2_bold(D_):
    """Table 8: risk ratios of the late bins (S > 0)."""
    D_ = D_.copy(); c_ = [c for c in D_.columns if c.startswith("RR")][0]
    m_ = D_.iloc[:, 0].str.match(r"^(\d|S >= )")
    D_.loc[m_, c_] = BOLD + D_.loc[m_, c_]
    return D_


def rq3_bold(D_):
    """Table 9: the best policy value in each column (lowest rates, shortest promise, largest reduction)."""
    return _bold_extreme(D_, list(D_.columns[1:5]), "min")


def contrasts_bold(D_):
    """Table S6: Holm-adjusted p-values below 0.05."""
    D_ = contrasts_readable(D_); v = D_["Holm p"].map(lambda x: 0.0 if str(x).startswith("<") else _lead(x))
    D_.loc[v < 0.05, "Holm p"] = BOLD + D_.loc[v < 0.05, "Holm p"]
    return D_


def theory_bold(D_):
    """Table S5: failed checks."""
    D_ = D_.copy(); m_ = D_["Status"] == "FAIL"
    D_.loc[m_, "Status"] = BOLD + D_.loc[m_, "Status"]
    return D_


tables = {
    "TabSampleRows": rows_from("table1_sample.csv", sample_bold),
    "TabModelsRows": rows_from("table2_rq1_models.csv", models_bold),
    "TabMethodsRows": rows_from("table3_rq1_methods.csv", methods_bold),
    "TabContrastsRows": rows_from("table4_rq1_contrasts.csv", contrasts_bold),
    "TabRqTwoRows": rows_from("table5_rq2.csv", rq2_bold),
    "TabRqThreeRows": rows_from("table6_rq3.csv", rq3_bold),
    "TabSensRows": rows_from("table7_sensitivity.csv", sens_readable),
    "TabSensRqThreeRows": rows_from("table7b_sensitivity_rq3.csv"),
    "TabTheoryRows": theory_refs(rows_from("table8_theory.csv", theory_bold, cols=["Check", "Claim", "Value", "Status"])),
    "TabRawRows": TAB_RAW or "\\multicolumn{5}{l}{Not available: run the stages sens\\_fit, sensitivity and inference.} \\\\",
    "TabRouteRows": TAB_ROUTE or "\\multicolumn{11}{l}{Not available: run the stages sens\\_fit and sensitivity.} \\\\",
}
tl = ["%% tables.tex -- bodies of the tables of the manuscript (used inside tabularx).",
      f"%% Generated by tools/gen_results_tex.py from {OUT.name}/manuscript/table*.csv. Do not edit."]
for n, body in tables.items():
    tl += ["", f"\\newcommand{{\\{n}}}{{%", body, "}"]
(MAC / "tables.tex").write_text("\n".join(tl) + "\n", encoding="utf-8")
print(f"results.tex: {len(names)} macros; tables.tex: {len(tables)} table bodies -> {MAC}")
