"""Sensitivity analyses (one table).

Without refitting (stored predictions of rq1_fit):
  calibration window   the last 14, 21 or 28 days before the origin instead of 42 (more recent, more censored)
  score                additive score Y - q(x) instead of the multiplicative score log Y - log q(x)
  tail share           upper bound of the tail share (orders undelivered at the origin counted as tail) instead of the
                       share among delivered cohort orders (a lower bound); see data/censoring.tail_share
  IPCW cap             10 and 50 instead of 20
  finite-sample level  beta = (1 - alpha)(1 + 1/n_eff) with the Kish effective size of the IPCW weights, and
                       beta = (1 - alpha)(1 + wbar / sum w), which reserves the largest possible test weight (the cap)
                       as in weighted conformal prediction (Tibshirani et al., 2019); both instead of (1 + 1/|C|)
  truncated split      exact finite-sample variant (Corollary 1 of the article): only the calibration orders with a
                       follow-up of at least c0 days, scores of min(Y, c0), unweighted, no tail term; a bound >= c0 is
                       replaced by the cap (c0 = 28 days)
  Mondrian unit        customer state (states with enough calibration orders) instead of region
  ACI update           gamma in {0.0001, 0.0002, 0.0005, 0.001}; no clipping; online per-order updates as in
                       Gibbs and Candes (2021)
  undelivered orders   counted as missed promises (column coverage_all of the RQ1 tables)
With refitting (stage sens_fit):
  Olist's estimate     the platform's promised days added as an input of XGBoost and QRF (both IPCW)
  route shrinkage      route table (IPCW) with shrinkage m = 20 and 100 instead of 50 (rq1_route_shrinkage.csv)
  uncalibrated promise the members of the quantile average refitted at the extra levels HIGH_TAUS with the
                       hyper-parameters of the main fit; promise floor(q_tau) on a fine grid of tau, the level read
                       off at Olist's coverage of each window (rq1_raw_matched.csv), against the split-IPCW promise
                       at the same coverage (rq1_matched_olist.csv): what the conformal step adds at matched coverage
  grid edges           at the tuned origins, the XGBoost QR depth and the AFT scale are re-tuned on grids extended
                       by EXT_XGB_DEPTHS and EXT_AFT_SCALES (only the new points are fitted; the loss of the main
                       choice is read from the stored fit); if a choice changes, the variant is refitted on the
                       origins that use it and split IPCW at alpha = 0.10 is compared with the main fit
                       (rq1_tuning_extended.csv, rq1_grid_extended.csv)
RQ2 / RQ3:
  three-star ratings counted as not low; response curve fitted on 2017 orders only (policies chosen with it and
  evaluated with the full-sample curve)
"""
import numpy as np
import pandas as pd

from ..conformal.one_sided import ScoreECDF, level, aci_promises, make_ecdf
from ..data.censoring import ipcw_weights, tail_share
from .rq1_coverage import _rows, run_rq1
from .rq1_fit import load_fits, run_rq1_fit

DAY = pd.Timedelta(days=1)
SENS_VARIANTS = ("xgbq_ipcw", "qrf_ipcw")
K_SUBS = (14, 21, 28)                       # shorter calibration windows (days)
TRUNC_C0 = 28                               # c0 of the truncated split conformal (Corollary 1)
ACI_GAMMAS = (0.0001, 0.0002, 0.0005, 0.001)  # grid of the ACI step (the configured step is chosen on 2017)
ROUTE_M = (20, 100)                         # shrinkage of the route table besides the configured m
HIGH_TAUS = (0.97, 0.98, 0.99, 0.995)       # extra quantile levels of the uncalibrated benchmark
RAW_TAU_GRID = np.round(np.arange(0.50, 0.99501, 0.0025), 4)
EXT_XGB_DEPTHS = (1, 8)                     # added to the XGBoost QR depth grid (grid-edge check)
EXT_AFT_SCALES = (0.15, 0.2)                # added to the AFT scale grid


def grid_extension(D, vocab, cfg, out, log=print):
    """Re-tune the XGBoost QR depth and the AFT scale at every tuned origin on grids extended at the edges, and refit
    the variants whose choice changes (rq1_tuning_extended.csv, rq1_grid_extended.csv)."""
    import pickle
    from ..data.purchase_features import feature_set
    from ..evaluation.origins import rolling_windows, origin_masks, inner_masks
    from .rq1_fit import _tune, _tune_aft, fit_origin
    from .rq1_coverage import _cal_block, group_col
    from ..conformal.one_sided import static_promises
    pdir = out / "data" / "parts"; t = out / "tables"
    num, cat = feature_set(cfg, None)
    W = {w["k"]: w for w in rolling_windows(cfg)}
    fits = {r["k"]: r for r in load_fits(pdir, "main")}
    xgb_grid = [g for g in cfg["tuning"]["xgbq_grid"]]
    ext_xgb = [{"max_depth": d} for d in EXT_XGB_DEPTHS]
    full_depths = sorted({g.get("max_depth") for g in xgb_grid} | set(EXT_XGB_DEPTHS))
    full_scales = sorted(set(cfg["models"]["aft"]["scales"]) | set(EXT_AFT_SCALES))
    rows = []
    for k, r in sorted(fits.items()):
        ch = r["chosen"]
        if ch.get("_k") != k or "xgbq" not in r["inner_loss"]:
            continue                                    # not a tuned origin
        w = W[k]
        M = origin_masks(D, w, cfg)
        tr_in, va_in = inner_masks(D, M["train"], w, cfg)
        base = dict(cfg["models"]["xgbq"])
        bx, lx = _tune("xgbq", True, D, tr_in, va_in, w["origin"], num, cat, vocab, cfg, base, ext_xgb)
        new_x = bx if lx < r["inner_loss"]["xgbq"] else ch["xgbq"]
        rows.append(dict(k=k, origin=w["origin"], variant="xgbq", main_choice=f"max_depth={ch['xgbq']['max_depth']}, "
                         f"learning_rate={ch['xgbq']['learning_rate']}", main_loss=float(r["inner_loss"]["xgbq"]),
                         best_new_point=f"max_depth={bx['max_depth']}", new_point_loss=float(lx),
                         extended_choice=f"max_depth={new_x['max_depth']}, learning_rate={new_x['learning_rate']}",
                         changed=bool(lx < r["inner_loss"]["xgbq"]),
                         on_edge_extended=new_x["max_depth"] in (full_depths[0], full_depths[-1]), _params=new_x))
        c2 = dict(cfg); c2["models"] = dict(cfg["models"]); c2["models"]["aft"] = dict(cfg["models"]["aft"], scales=list(EXT_AFT_SCALES))
        ba, la, _ = _tune_aft(D, tr_in, va_in, w["origin"], num, cat, vocab, c2)
        new_a = dict(dist=ba[0], scale=ba[1]) if la < r["inner_loss"]["aft"] else ch["aft"]
        rows.append(dict(k=k, origin=w["origin"], variant="aft", main_choice=f"{ch['aft']['dist']}, scale={ch['aft']['scale']}",
                         main_loss=float(r["inner_loss"]["aft"]), best_new_point=f"{ba[0]}, scale={ba[1]}",
                         new_point_loss=float(la), extended_choice=f"{new_a['dist']}, scale={new_a['scale']}",
                         changed=bool(la < r["inner_loss"]["aft"]),
                         on_edge_extended=new_a["scale"] in (full_scales[0], full_scales[-1]), _params=new_a))
        log(f"  grid extension: origin {k:02d} xgbq {rows[-2]['extended_choice']} ({'changed' if rows[-2]['changed'] else 'same'}), "
            f"aft {rows[-1]['extended_choice']} ({'changed' if rows[-1]['changed'] else 'same'})")
    T = pd.DataFrame(rows)
    T.drop(columns="_params").to_csv(t / "rq1_tuning_extended.csv", index=False)
    # refit the variants whose choice changed, on the origins that carry that choice (until the next tuned origin)
    res = []
    tuned = sorted(T["k"].unique())
    for _, x in T[T["changed"]].iterrows():
        kind = x["variant"]; v = "xgbq_ipcw" if kind == "xgbq" else "aft"
        nxt = [kk for kk in tuned if kk > x["k"]]
        for k in [kk for kk in sorted(fits) if x["k"] <= kk < (nxt[0] if nxt else 10 ** 6)]:
            f = pdir / f"rq1fit__gridext_{kind}__k{k:02d}.pkl"
            if f.is_file():
                with open(f, "rb") as fh:
                    r2 = pickle.load(fh)
            else:
                ch2 = {kk: vv for kk, vv in fits[k]["chosen"].items() if kk != "_k"}
                ch2[kind] = x["_params"]
                r2 = fit_origin(D, W[k], num, cat, vocab, cfg, (v,), fixed_chosen=ch2)
                with open(f, "wb") as fh:
                    pickle.dump(r2, fh, protocol=pickle.HIGHEST_PROTOCOL)
            r = fits[k]; E = D.iloc[r["eval_idx"]]
            tails = tail_share(D, r["origin"], cfg["rolling"]["calib_days"], groups=group_col(cfg))
            ti = cfg["quantiles"].index(0.9); ge = E[group_col(cfg)].fillna("NA").to_numpy()
            for lab, rr in (("main", r), ("extended", r2)):
                cal = _cal_block(rr, D, v, ti, cfg, tails)
                P = static_promises(rr["pred"][v]["eval"][:, ti].astype(float), ge, cal, 0.10, "split", "ipcw", cfg)
                x0 = _rows(k, r["origin"], v, "split_ipcw", 0.10, E, P)[0]
                # a fit has diverged when its 0.9-quantile of the delivery time leaves [1 hour, 10 years] for some order
                lq = rr["pred"][v]["eval"][:, ti].astype(float)
                div = bool((~np.isfinite(lq)).any() or (lq > np.log(3650.0)).any() or (lq < np.log(1.0 / 24)).any())
                res.append(dict(k=k, variant=v, fit=lab, coverage=x0["coverage"], mean_promise=x0["mean_promise"],
                                diverged=float(div)))
            log(f"  grid extension: refitted {v} at origin {k:02d}")
    G = pd.DataFrame(res)
    if len(G):
        G = G.pivot_table(index=["k", "variant"], columns="fit", values=["coverage", "mean_promise", "diverged"]).reset_index()
        G.columns = ["_".join(c).strip("_") for c in G.columns]
    else:
        G = pd.DataFrame([dict(note="no tuning choice changed on the extended grids")])
    G.to_csv(t / "rq1_grid_extended.csv", index=False)


def run_sens_fit(D, vocab, cfg, out, log=print, workers=1):
    pdir, fp = out / "data" / "parts", out / "data" / "features_purchase.pkl"
    run_rq1_fit(D, vocab, cfg, pdir, variants=SENS_VARIANTS, tag="promise", promise=True, log=log,
                workers=workers, feat_path=fp)
    for m in ROUTE_M:
        c2 = dict(cfg); c2["models"] = dict(cfg["models"]); c2["models"]["route"] = dict(cfg["models"]["route"], shrink_m=m)
        run_rq1_fit(D, vocab, c2, pdir, variants=("route_ipcw",), tag=f"route_m{m}", log=log, workers=workers,
                    feat_path=fp, chosen_from="main")
    c3 = dict(cfg); c3["quantiles"] = list(HIGH_TAUS)
    run_rq1_fit(D, vocab, c3, pdir, variants=tuple(cfg["ensemble"]["members"]), tag="highq", log=log,
                workers=workers, feat_path=fp, chosen_from="main")
    grid_extension(D, vocab, cfg, out, log)


def raw_quantile_matched(D, cfg, out):
    """Uncalibrated benchmark: the promise floor(q_tau(x)) of the stored log-quantiles (main levels and HIGH_TAUS,
    rearranged, linearly interpolated in tau on RAW_TAU_GRID) and the mean promise at Olist's coverage of each window,
    read off the coverage-length curve as in rq1_extra.matched_olist; NaN when no level on the grid reaches it."""
    pdir = out / "data" / "parts"; cap = cfg["conformal"]["promise_cap_days"]
    hi = {r["k"]: r for r in load_fits(pdir, "highq", cfg)}
    if not hi:
        return None
    taus = np.array(list(cfg["quantiles"]) + list(HIGH_TAUS))
    order = np.argsort(taus); taus = taus[order]
    rows = []
    for r in load_fits(pdir, "main", cfg):
        h = hi.get(r["k"])
        if h is None:
            continue
        E = D.iloc[r["eval_idx"]]; ok = E["delivered"].to_numpy(); Dv = E["D"].to_numpy()[ok]
        Po = E["P_olist"].to_numpy()[ok].astype(float); cov_o = float((Dv <= Po).mean())
        for v in h["pred"]:
            Q = np.hstack([np.asarray(r["pred"][v]["eval"], float), np.asarray(h["pred"][v]["eval"], float)])[ok][:, order]
            Q = np.sort(Q, axis=1)                  # rearrangement across the two fits
            j = np.clip(np.searchsorted(taus, RAW_TAU_GRID, side="right") - 1, 0, len(taus) - 2)
            f = (RAW_TAU_GRID - taus[j]) / (taus[j + 1] - taus[j])
            cov, ln = [], []
            for jj, ff in zip(j, f):
                P = np.clip(np.floor(np.exp(Q[:, jj] + ff * (Q[:, jj + 1] - Q[:, jj]))), 1, cap)
                cov.append(float((Dv <= P).mean())); ln.append(float(P.mean()))
            cov, ln = np.maximum.accumulate(np.array(cov)), np.maximum.accumulate(np.array(ln))
            reach = cov[0] <= cov_o <= cov[-1]
            L_at = float(np.interp(cov_o, cov, ln)) if reach else np.nan
            rows.append(dict(k=r["k"], origin=r["origin"], variant=v, olist_coverage=cov_o,
                             raw_coverage_max=float(cov[-1]), raw_length_at_olist_coverage=L_at, reached=bool(reach)))
    R = pd.DataFrame(rows)
    R.to_csv(out / "tables" / "rq1_raw_matched.csv", index=False)
    return R


def route_shrinkage(D, cfg, out):
    """Route table (IPCW, split IPCW at every alpha and matched to Olist's coverage) for m = 20, 50 (main), 100."""
    from .rq1_extra import matched_olist
    pdir = out / "data" / "parts"; m0 = cfg["models"]["route"]["shrink_m"]
    rows = []
    for m, tag in [(m0, "main")] + [(m, f"route_m{m}") for m in ROUTE_M]:
        fits = load_fits(pdir, tag)
        if not fits:
            continue
        Mo = matched_olist(D, cfg, out, variants=["route_ipcw"], fits=fits)
        for a in cfg["alphas"]:
            rr = []
            for r in fits:
                E, P = _static(D, r, "route_ipcw", a, cfg)
                rr += _rows(r["k"], r["origin"], "route_ipcw", "split_ipcw", a, E, P, cap=cfg["conformal"]["promise_cap_days"])
            s = _summ(rr, "route shrinkage", f"route table, m = {m}" + (" (main)" if tag == "main" else ""), "route_ipcw", a)
            s.update(shrink_m=m, length_gap_at_olist_coverage=float(Mo["length_gap"].mean()))
            rows.append(s)
    R = pd.DataFrame(rows)
    R.to_csv(out / "tables" / "rq1_route_shrinkage.csv", index=False)
    return R


def _promise(base_q, Q, cap, additive):
    with np.errstate(over="ignore", invalid="ignore"):
        U = base_q + Q if additive else base_q * np.exp(Q)
    return np.clip(np.floor(np.where(np.isfinite(U), U, cap)), 1, cap)


def _beta(a, n, w, rule, wbar):
    if rule == "n":
        return level(a, n)
    if rule == "neff":
        return level(a, w.sum() ** 2 / max((w ** 2).sum(), 1e-12))
    if rule == "test_mass":
        return (1 - a) * (1 + wbar / max(w.sum(), 1e-12))
    raise ValueError(rule)


def truncated_split_bound(y, obs, c, q_cal, q_eval, alpha, c0):
    """Corollary 1: one-sided split conformal on min(Y, c0) over the calibration orders with follow-up c >= c0.
    For these orders min(Y, c0) is known at the origin (Y if delivered, else Y >= c >= c0); under independent
    censoring they are an unweighted sample of the calibration population, so Lemma 2 applies to the truncated
    outcome. Returns the bound U (np.inf where U >= c0, since a bound of c0 days or more is not identified)."""
    keep = c >= c0
    yt = np.where(obs[keep], np.minimum(np.where(obs[keep], y[keep], c0), c0), c0)
    n = int(keep.sum())
    e = ScoreECDF(np.log(yt / q_cal[keep]), np.full(n, 1.0 / max(n, 1)))
    U = q_eval * np.exp(float(e.quantile(level(alpha, n))))
    return np.where(U < c0, U, np.inf)


def _truncated(r, D, v, a, cfg, c0):
    taus = cfg["quantiles"]; ti = taus.index(round(1 - a, 4)); capP = cfg["conformal"]["promise_cap_days"]
    C = D.iloc[r["cal_idx"]]; E = D.iloc[r["eval_idx"]]
    c = ((r["origin"] - C["purchase_day"]) / DAY).to_numpy()
    q = np.exp(r["pred"][v]["cal"][:, ti].astype(float)); qe = np.exp(r["pred"][v]["eval"][:, ti].astype(float))
    U = truncated_split_bound(r["cal_y"], r["cal_obs"], c, q, qe, a, c0)
    return E, np.clip(np.floor(np.where(np.isfinite(U), U, capP)), 1, capP)


def _static(D, r, v, a, cfg, K_sub=None, additive=False, cap_w=None, unit="customer_region", mondrian=False,
            tail_mode="observed", beta_rule="n", trunc_c0=None):
    if trunc_c0:
        return _truncated(r, D, v, a, cfg, trunc_c0)
    taus = cfg["quantiles"]; ti = taus.index(round(1 - a, 4)); capP = cfg["conformal"]["promise_cap_days"]
    C = D.iloc[r["cal_idx"]]; E = D.iloc[r["eval_idx"]]
    keep = np.ones(len(C), bool)
    if K_sub:
        keep = (C["purchase_ts"] >= r["origin"] - pd.Timedelta(days=K_sub)).to_numpy()
    C = C[keep]
    obs = r["cal_obs"][keep]; y = r["cal_y"][keep]
    q = np.exp(r["pred"][v]["cal"][keep, ti].astype(float)); qe = np.exp(r["pred"][v]["eval"][:, ti].astype(float))
    c = ((r["origin"] - C["purchase_day"]) / DAY).to_numpy()
    w, _ = ipcw_weights(obs, c, y, cap_w or cfg["censoring"]["weight_cap"])
    score = np.where(obs, (np.where(obs, y, 1.0) - q) if additive else np.log(np.where(obs, y, 1.0) / q), np.nan)
    K = K_sub or cfg["rolling"]["calib_days"]
    tail, tail_g = tail_share(D, r["origin"], K, groups=unit, mode=tail_mode)
    if not mondrian:
        e, n = make_ecdf(score, obs, w, "ipcw", tail)
        b = _beta(a, n, w[obs], beta_rule, cap_w or cfg["censoring"]["weight_cap"])
        return E, _promise(qe, float(e.quantile(b)), capP, additive)
    gc = C[unit].fillna("NA").to_numpy(); ge = E[unit].fillna("NA").to_numpy()
    pooled = make_ecdf(score, obs, w, "ipcw", tail)
    Q = np.empty(len(E))
    for g in pd.unique(ge):
        m = gc == g
        e, n = make_ecdf(score[m], obs[m], w[m], "ipcw", tail_g.get(g, tail)) \
            if m.sum() >= cfg["conformal"]["mondrian_min_n"] else pooled
        Q[ge == g] = e.quantile(level(a, n))
    return E, _promise(qe, Q, capP, additive)


def _summ(rows, analysis, setting, v, a, period="all windows"):
    M = pd.DataFrame(rows)
    allr = M[M.region == "all"]; reg = M[M.region != "all"].groupby("k")["coverage"].min()
    return dict(analysis=analysis, setting=setting, period=period, variant=v, alpha=a, coverage=allr["coverage"].mean(),
                coverage_min_window=allr["coverage"].min(), mean_promise=allr["mean_promise"].mean(),
                worst_region=reg.mean(), coverage_all=allr["coverage_all"].mean(), n_windows=allr["k"].nunique(),
                share_capped=allr["share_capped"].mean() if "share_capped" in allr else np.nan)


def run_sensitivity(D, cfg, out, log=print):
    t = out / "tables"
    fits = load_fits(out / "data" / "parts", "main")
    res = []
    for v in SENS_VARIANTS:
        for a in cfg["alphas"]:
            specs = [("main (split IPCW, 42 d, log score, cap 20)", dict())]
            specs += [(f"calibration window {k} d", dict(K_sub=k)) for k in K_SUBS]
            specs += [("additive score", dict(additive=True)), ("IPCW cap 10", dict(cap_w=10)),
                      ("IPCW cap 50", dict(cap_w=50)),
                      ("tail share upper bound (undelivered at the origin counted as tail)", dict(tail_mode="upper")),
                      ("level with the Kish effective size, (1 + 1/n_eff)", dict(beta_rule="neff")),
                      ("level reserving the largest test weight, (1 + wbar / sum w)", dict(beta_rule="test_mass")),
                      (f"truncated split conformal, follow-up >= {TRUNC_C0} d (finite-sample valid)", dict(trunc_c0=TRUNC_C0))]
            for name, kw in specs:
                rows = []
                for r in fits:
                    E, P = _static(D, r, v, a, cfg, **kw)
                    rows += _rows(r["k"], r["origin"], v, "split_ipcw", a, E, P, cap=cfg["conformal"]["promise_cap_days"])
                res.append(_summ(rows, "conformal (split)", name, v, a))
            for name, unit in (("Mondrian by region (main)", "customer_region"), ("Mondrian by state", "customer_state")):
                rows = []
                for r in fits:
                    E, P = _static(D, r, v, a, cfg, unit=unit, mondrian=True)
                    rows += _rows(r["k"], r["origin"], v, "mondrian_ipcw", a, E, P, cap=cfg["conformal"]["promise_cap_days"])
                res.append(_summ(rows, "conformal (Mondrian)", name, v, a))
        log(f"  sensitivity: static variants of {v} done")
    # ACI step size and update rule (pooled ACI, IPCW), alpha = 0.10
    aci_specs = [(f"daily batch, gamma = {g}", dict(aci_step="day", aci_gamma=g)) for g in ACI_GAMMAS]
    aci_specs += [("daily batch, gamma = 0.0005, no clipping", dict(aci_step="day", aci_gamma=0.0005, aci_alpha_min_ratio=-1e9)),
                  ("online per-order update (Gibbs and Candes), gamma = 0.0005, no clipping",
                   dict(aci_step="order", aci_gamma=0.0005, aci_alpha_min_ratio=-1e9))]
    for name, kw in aci_specs:
        c2 = dict(cfg); c2["conformal"] = dict(cfg["conformal"], **kw)
        M, _ = run_rq1(D, c2, out, log=lambda *_: None, methods=("aci_ipcw",), write=False, variants=SENS_VARIANTS)
        for v in SENS_VARIANTS:
            Mv = M[(M.variant == v) & (M.method == "aci_ipcw") & np.isclose(M.alpha, 0.10)]
            res.append(_summ(Mv.to_dict("records"), "conformal (ACI)", name, v, 0.10))
            Mt = Mv[pd.to_datetime(Mv["origin"]) >= pd.Timestamp("2018-01-01")]
            res.append(_summ(Mt.to_dict("records"), "conformal (ACI)", name, v, 0.10, "2018 (test)"))
    # the split-conformal reference on the same 2018 windows
    Ms = pd.read_csv(t / "rq1_window_metrics.csv")
    for v in SENS_VARIANTS:
        Mt = Ms[(Ms.variant == v) & (Ms.method == "split_ipcw") & np.isclose(Ms.alpha, 0.10) &
                (pd.to_datetime(Ms["origin"]) >= pd.Timestamp("2018-01-01"))]
        res.append(_summ(Mt.to_dict("records"), "conformal (ACI)", "reference: split IPCW", v, 0.10, "2018 (test)"))
    log("  sensitivity: ACI done")
    # Olist's estimate as an input (refitted models)
    if load_fits(out / "data" / "parts", "promise"):
        M, _ = run_rq1(D, cfg, out, log=lambda *_: None, tag="promise", methods=("raw", "split_ipcw", "mondrian_ipcw"),
                       write=True)
        for v in SENS_VARIANTS:
            for a in cfg["alphas"]:
                Mv = M[(M.variant == v) & (M.method == "split_ipcw") & (M.alpha == a)]
                res.append(_summ(Mv.to_dict("records"), "features", "Olist's estimate as an input", v, a))
    S = pd.DataFrame(res)
    S.to_csv(t / "sensitivity_rq1.csv", index=False)
    if route_shrinkage(D, cfg, out) is not None:
        log("  sensitivity: route shrinkage done")
    if raw_quantile_matched(D, cfg, out) is not None:
        log("  sensitivity: uncalibrated quantile benchmark done")
    # size of the calibration sets and of their IPCW weights (Remark 3: Kish effective size)
    cw = []
    for r in fits:
        w = r["cal_w"][r["cal_obs"]]
        cw.append(dict(k=r["k"], n_cal=len(r["cal_obs"]), n_obs=int(r["cal_obs"].sum()), sum_w=float(w.sum()),
                       n_eff=float(w.sum() ** 2 / (w ** 2).sum()), w_max=float(w.max())))
    pd.DataFrame(cw).to_csv(t / "rq1_calibration_weights.csv", index=False)
    # RQ3 with alternative response curves
    from .rq3_policy import run_rq3
    rows = []
    for tag, curve, ev in (("fit2017", "rq2_response_curve_fit2017.csv", "rq2_response_curve.csv"),
                           ("neutral0", "rq2_response_curve_neutral0.csv", None)):
        if (t / curve).is_file():
            _, Mt = run_rq3(D, cfg, out, log=lambda *_: None, curve=curve, eval_curve=ev, tag=tag)
            rows.append(dict(analysis="RQ3 response curve", setting=tag, low_olist=Mt["low_olist"].mean(),
                             low_optimal_at_L=Mt["optimal_low_realised_at_L"].mean(),
                             low_cpd_at_L=Mt["cpd_quantile_low_realised_at_L"].mean(),
                             L_olist=Mt["L_olist"].mean(), L_optimal_at_low=Mt["optimal_L_at_low"].mean()))
    # RQ3 with the quantile average of RQ1 as the base model of the predictive distribution (main response curve)
    ens = (cfg.get("ensemble") or {}).get("name")
    if ens:
        _, Mt = run_rq3(D, cfg, out, log=lambda *_: None, tag="cpd_ens", model=ens)
        rows.append(dict(analysis="RQ3 predictive distribution", setting=f"cpd_ens ({ens})", low_olist=Mt["low_olist"].mean(),
                         low_optimal_at_L=Mt["optimal_low_realised_at_L"].mean(),
                         low_cpd_at_L=Mt["cpd_quantile_low_realised_at_L"].mean(),
                         L_olist=Mt["L_olist"].mean(), L_optimal_at_low=Mt["optimal_L_at_low"].mean()))
        log("  sensitivity: RQ3 with the quantile-average CPD done")
    # RQ3 with other locations of the mass the predictive distribution cannot place (main: K + 7 = 49 days)
    K = cfg["rolling"]["calib_days"]
    for dt in (K, K + 14, K + 28, 120):
        _, Mt = run_rq3(D, cfg, out, log=lambda *_: None, tag=f"tail{dt}", tail_day=dt)
        rows.append(dict(analysis="RQ3 tail location", setting=f"tail{dt}", low_olist=Mt["low_olist"].mean(),
                         low_optimal_at_L=Mt["optimal_low_realised_at_L"].mean(),
                         low_cpd_at_L=Mt["cpd_quantile_low_realised_at_L"].mean(),
                         L_olist=Mt["L_olist"].mean(), L_optimal_at_low=Mt["optimal_L_at_low"].mean()))
    log("  sensitivity: RQ3 tail locations done")
    # checks of the response curve: cross-fitting (optimise with one half's curve, score with the other's, both ways),
    # scoring with the AIPW step curve, and a curve fitted on ratings answered after arrival
    runs = {}
    for tag, curve, ev in (("xfitAB", "rq2_response_curve_xfitA.csv", "rq2_response_curve_xfitB.csv"),
                           ("xfitBA", "rq2_response_curve_xfitB.csv", "rq2_response_curve_xfitA.csv"),
                           ("evalaipw", "rq2_response_curve.csv", "rq2_response_curve_aipwstep.csv"),
                           ("after", "rq2_response_curve_after.csv", None)):
        if (t / curve).is_file() and (ev is None or (t / ev).is_file()):
            _, runs[tag] = run_rq3(D, cfg, out, log=lambda *_: None, curve=curve, eval_curve=ev, tag=tag)
    if "xfitAB" in runs and "xfitBA" in runs:
        a_, b_ = runs["xfitAB"].set_index("k"), runs["xfitBA"].set_index("k").loc[runs["xfitAB"]["k"]]
        num = a_.select_dtypes("number").columns
        X = ((a_[num] + b_[num]) / 2).reset_index()
        X.insert(1, "origin", a_["origin"].to_numpy())
        X.to_csv(t / "rq3_matched_xfit.csv", index=False)
        runs["xfit"] = X
    for tag in ("xfit", "evalaipw", "after"):
        if tag in runs:
            Mt = runs[tag]
            rows.append(dict(analysis="RQ3 response curve check", setting=tag, low_olist=Mt["low_olist"].mean(),
                             low_optimal_at_L=Mt["optimal_low_realised_at_L"].mean(),
                             low_cpd_at_L=Mt["cpd_quantile_low_realised_at_L"].mean(),
                             L_olist=Mt["L_olist"].mean(), L_optimal_at_low=Mt["optimal_L_at_low"].mean()))
    log("  sensitivity: RQ3 curve checks done")
    pd.DataFrame(rows).to_csv(t / "sensitivity_rq3.csv", index=False)
    # RQ2: selection into answering after arrival at the promised day (Lee bounds)
    from .rq2_response import rq2_frame, lee_bounds
    R2 = rq2_frame(D, cfg)
    pd.DataFrame([lee_bounds(R2, (0,), (1,), seed=cfg["seed"]), lee_bounds(R2, (-1, 0), (1, 2), seed=cfg["seed"])]
                 ).to_csv(t / "rq2_lee_bounds.csv", index=False)
    log("  sensitivity: RQ2 Lee bounds done")
    log("  sensitivity: done")
