"""RQ1 additions: promise length at Olist's coverage, results by period and by type of window, drift episodes.

  rq1_matched_olist.csv    for every window and model variant, the mean promise that split conformal (IPCW) needs to
                           reach Olist's realised coverage of that window, found on a fine grid of miscoverage levels
                           (alpha in 0.01..0.30; the base quantile model is the fitted level closest to 1 - alpha).
                           The comparison is descriptive: the level is matched ex post, as in the matched comparisons
                           of RQ3.
  rq1_summary_by_period.csv window means for the selection period (origins in 2017, used to choose the ACI step) and
                           the test period (origins in 2018), and for stress versus normal windows
  rq1_window_types.csv     stress windows: Olist's late rate in the window at least twice the median over windows
                           (a model-free definition; it picks the Black Friday 2017 and the Feb-Mar 2018 windows)
  eda_drift_weekly.csv     weekly deliveries, purchases, late rate and mean promise (cause of the Feb-Mar 2018 dip)
  rq1_tuning_choices.csv   hyper-parameters chosen at every origin (re-selected every `tuning.retune_every` origins), and
                           whether the choice lies on the edge of its grid
  rq1_ensemble_selection.csv  the pre-specified choice of the quantile average: mean promise at Olist's coverage of
                           each candidate member set (config `ensemble.candidates`) over the 2017 origins (selection)
                           and the 2018 origins (test); the set with the shortest 2017 promise is selected
"""
import numpy as np
import pandas as pd

from ..conformal.one_sided import make_ecdf, level, promise_from_bound
from ..data.censoring import tail_share
from .rq1_coverage import _cal_block, group_col
from .rq1_fit import load_fits, add_ensembles

SPLIT_YEAR = pd.Timestamp("2018-01-01")
ALPHA_GRID = np.round(np.arange(0.01, 0.3001, 0.005), 4)


def window_types(M):
    o = M[(M.variant == "olist") & (M.region == "all")][["k", "origin", "coverage"]].copy()
    o["olist_late"] = 1 - o["coverage"]
    thr = 2 * o["olist_late"].median()
    o["stress"] = o["olist_late"] >= thr
    o["period"] = np.where(pd.to_datetime(o["origin"]) < SPLIT_YEAR, "2017 (selection)", "2018 (test)")
    o["threshold"] = thr
    return o


def _base_tau(a, taus):
    up = [t for t in taus if t >= 0.8]
    return min(up, key=lambda t: abs(t - (1 - a)))


def matched_olist(D, cfg, out, variants=None, fits=None):
    fits = fits if fits is not None else load_fits(out / "data" / "parts", "main", cfg)
    taus = cfg["quantiles"]; cap = cfg["conformal"]["promise_cap_days"]; K = cfg["rolling"]["calib_days"]
    rows = []
    for r in fits:
        E = D.iloc[r["eval_idx"]]; ok = E["delivered"].to_numpy(); Dv = E["D"].to_numpy()[ok]
        Po = E["P_olist"].to_numpy()[ok].astype(float)
        cov_o, len_o = float((Dv <= Po).mean()), float(Po.mean())
        tails = tail_share(D, r["origin"], K, groups=group_col(cfg))
        for v in (variants or list(r["pred"])):
            curve = []
            for a in ALPHA_GRID:
                ti = taus.index(_base_tau(a, taus))
                cal = _cal_block(r, D, v, ti, cfg, tails)
                e, n = make_ecdf(cal["score"], cal["obs"], cal["w"], "ipcw", cal["tail"])
                P = promise_from_bound(r["pred"][v]["eval"][ok, ti].astype(float), float(e.quantile(level(a, n))), cap)
                curve.append((float((Dv <= P).mean()), float(P.mean())))
            c = np.array(sorted(curve))
            L_at = float(np.interp(cov_o, c[:, 0], c[:, 1])) if c[0, 0] <= cov_o <= c[-1, 0] else np.nan
            d = np.array(sorted(curve, key=lambda x: x[1]))
            C_at = float(np.interp(len_o, d[:, 1], d[:, 0])) if d[0, 1] <= len_o <= d[-1, 1] else np.nan
            rows.append(dict(k=r["k"], origin=r["origin"], variant=v, olist_coverage=cov_o, olist_length=len_o,
                             length_at_olist_coverage=L_at, coverage_at_olist_length=C_at,
                             length_gap=L_at - len_o, coverage_gap=C_at - cov_o))
    return pd.DataFrame(rows)


def tuning_choices(cfg, out):
    """Hyper-parameters chosen at each origin and whether each lies on the edge of its grid (one row per origin)."""
    tc = cfg["tuning"]; a = cfg["models"]["aft"]
    depths = sorted({g.get("max_depth", cfg["models"]["xgbq"]["max_depth"]) for g in tc["xgbq_grid"]})
    leaves = sorted({g.get("min_samples_leaf", cfg["models"]["qrf"]["min_samples_leaf"]) for g in tc["qrf_grid"]})
    scales = sorted(a["scales"])
    rows = []
    for r in load_fits(out / "data" / "parts", "main"):
        ch = r["chosen"]; x, q, f = ch.get("xgbq", {}), ch.get("qrf", {}), ch.get("aft", {})
        rows.append(dict(k=r["k"], origin=r["origin"], tuned_here=ch.get("_k") == r["k"], tuned_at=ch.get("_k"),
                         xgbq_max_depth=x.get("max_depth"), xgbq_learning_rate=x.get("learning_rate"),
                         qrf_min_samples_leaf=q.get("min_samples_leaf"), aft_dist=f.get("dist"), aft_scale=f.get("scale"),
                         xgbq_edge=x.get("max_depth") in (depths[0], depths[-1]),
                         qrf_edge=q.get("min_samples_leaf") in (leaves[0], leaves[-1]),
                         aft_scale_edge=f.get("scale") in (scales[0], scales[-1])))
    return pd.DataFrame(rows)


def ensemble_selection(D, cfg, out):
    """Apply the selection rule of the quantile average: the candidate member set with the shortest mean promise at
    Olist's coverage over the selection period (2017 origins); 2018 is reported as the test."""
    E = cfg.get("ensemble") or {}
    cands = E.get("candidates") or {}
    if not cands:
        return pd.DataFrame()
    fits = add_ensembles(load_fits(out / "data" / "parts", "main"), {f"cand::{k}": v for k, v in cands.items()})
    Mo = matched_olist(D, cfg, out, [f"cand::{k}" for k in cands], fits=fits)
    Mo["period"] = np.where(pd.to_datetime(Mo["origin"]) < SPLIT_YEAR, "2017", "2018")
    g = Mo.groupby(["variant", "period"])["length_gap"].mean().unstack()
    rows = []
    for k, members in cands.items():
        x = g.loc[f"cand::{k}"]
        rows.append(dict(candidate=k, members=" + ".join(members), n_members=len(members),
                         length_gap_2017=float(x.get("2017", np.nan)), length_gap_2018=float(x.get("2018", np.nan)),
                         length_gap_all=float(Mo.loc[Mo.variant == f"cand::{k}", "length_gap"].mean()),
                         configured=list(members) == list(E["members"])))
    R = pd.DataFrame(rows)
    R["selected"] = R["length_gap_2017"] == R["length_gap_2017"].min()
    return R


def by_period(M, W):
    X = M.merge(W[["k", "stress", "period"]], on="k", how="left")
    X["window_type"] = np.where(X["stress"], "stress", "normal")
    out = []
    for keys in (["period"], ["window_type"]):
        g = X[X.region == "all"].groupby(["variant", "method", "alpha"] + keys, dropna=False)
        s = g.agg(n_windows=("k", "nunique"), coverage=("coverage", "mean"), coverage_min=("coverage", "min"),
                  mean_promise=("mean_promise", "mean"), coverage_all=("coverage_all", "mean")).reset_index()
        reg = X[X.region != "all"].groupby(["variant", "method", "alpha", "k"] + keys, dropna=False)["coverage"].min()
        reg = reg.groupby(level=[0, 1, 2, 4], dropna=False).mean().rename("worst_region").reset_index()
        s = s.merge(reg, on=["variant", "method", "alpha"] + keys, how="left")
        s = s.rename(columns={keys[0]: "group"}); s.insert(3, "split_by", keys[0])
        out.append(s)
    return pd.concat(out, ignore_index=True)


def drift_weekly(D):
    dl = D[D["delivered"]]
    wk_del = dl.set_index("delivered_ts").resample("W-SUN").size().rename("deliveries")
    p = D.set_index("purchase_ts")
    wk_pur = p.resample("W-SUN").size().rename("purchases")
    lt = dl.set_index("purchase_ts").resample("W-SUN").agg({"late_olist": "mean", "P_olist": "mean", "Y": "mean"})
    W = pd.concat([wk_del, wk_pur, lt], axis=1).reset_index().rename(columns={"index": "week"})
    W = W.rename(columns={W.columns[0]: "week", "late_olist": "late_rate_by_purchase_week",
                          "P_olist": "mean_promise_by_purchase_week", "Y": "mean_delivery_time_by_purchase_week"})
    return W[(W["week"] >= "2017-01-01") & (W["week"] <= "2018-08-31")]


def run_rq1_extra(D, cfg, out, log=print):
    t = out / "tables"
    M = pd.read_csv(t / "rq1_window_metrics.csv")
    W = window_types(M)
    W.to_csv(t / "rq1_window_types.csv", index=False)
    by_period(M, W).to_csv(t / "rq1_summary_by_period.csv", index=False)
    Mo = matched_olist(D, cfg, out)
    Mo.to_csv(t / "rq1_matched_olist.csv", index=False)
    drift_weekly(D).to_csv(t / "eda_drift_weekly.csv", index=False)
    Tc = tuning_choices(cfg, out)
    Tc.to_csv(t / "rq1_tuning_choices.csv", index=False)
    tt = Tc[Tc.tuned_here]
    log(f"  rq1 extra: tuning choices at the grid edge over {len(tt)} tuned origins: xgbq {int(tt.xgbq_edge.sum())}, "
        f"qrf {int(tt.qrf_edge.sum())}, aft scale {int(tt.aft_scale_edge.sum())}")
    Es = ensemble_selection(D, cfg, out)
    if len(Es):
        Es.to_csv(t / "rq1_ensemble_selection.csv", index=False)
        sel = Es.loc[Es.selected, "candidate"].tolist()
        log(f"  rq1 extra: ensemble selected on 2017 {sel}; configured {cfg['ensemble']['name']} "
            f"{'matches' if bool(Es.loc[Es.selected, 'configured'].all()) else 'DOES NOT MATCH'} the rule")
    s = Mo.groupby("variant")["length_gap"].mean().round(2).to_dict()
    log(f"  rq1 extra: stress windows {W.loc[W.stress, 'k'].tolist()}; length gap at Olist's coverage {s}")
