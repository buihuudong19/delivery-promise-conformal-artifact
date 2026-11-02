"""RQ1, step 1: fit the delivery-time models at every rolling origin and store their quantile predictions.

For each origin the stored file holds, for every model variant, the predicted log-quantiles of the calibration and
evaluation orders, the hyper-parameters chosen on the inner validation block (re-selected every `tuning.retune_every`
origins and kept in between), and the censoring status of the
calibration orders at the origin. The conformal step (rq1_coverage.py) works on these files only, so that the
corrections can be varied without refitting.
"""
import pickle
import time

import numpy as np
import pandas as pd

from ..data.censoring import status_at, ipcw_weights
from ..data.purchase_features import feature_set
from ..evaluation.origins import rolling_windows, origin_masks, inner_masks
from ..evaluation.metrics import pinball
from ..models.quantile_models import RouteTable, LinearQR, XGBQuantile, QRF, XGBAFT

VARIANTS = ("route_plain", "route_ipcw", "lqr_plain", "lqr_ipcw", "xgbq_plain", "xgbq_ipcw", "qrf_plain", "qrf_ipcw",
            "aft", "aft_naive")
UPPER = (0.8, 0.9, 0.95)


def _upper_loss(logy, Q, taus, w=None):
    return np.mean([pinball(logy, Q[:, taus.index(t)], t, w) for t in UPPER if t in taus])


def _training_view(D, mask, origin, cap):
    T = D[mask]
    obs, c, y = status_at(T, origin)
    w, _ = ipcw_weights(obs, c, y, cap)
    return T, obs, c, y, w


def _make(kind, num, cat, vocab, cfg, params):
    taus = cfg["quantiles"]; seed = cfg["seed"]
    if kind == "route":
        return RouteTable(taus, cfg["models"]["route"]["shrink_m"])
    if kind == "lqr":
        p = cfg["models"]["lqr"]
        return LinearQR(num, cat, taus, p["max_iter"], p["min_frequency"])
    if kind == "xgbq":
        return XGBQuantile(num, cat, vocab, taus, seed=seed, n_jobs=cfg.get("n_jobs", 2), **params)
    if kind == "qrf":
        return QRF(num, cat, taus, seed=seed, n_jobs=cfg.get("n_jobs", 2), **params)
    raise ValueError(kind)


def _fit_quantile(kind, weighted, T, obs, y, w, num, cat, vocab, cfg, params):
    m = _make(kind, num, cat, vocab, cfg, params)
    return m.fit(T[obs], np.log(y[obs]), w[obs] if weighted else None)


def _tune(kind, weighted, D, tr_in, va_in, origin, num, cat, vocab, cfg, base, grid):
    """Choose the hyper-parameters on the inner block (pinball loss at the upper levels, IPCW-weighted if weighted)."""
    cap = cfg["censoring"]["weight_cap"]
    Ti, oi, ci, yi, wi = _training_view(D, tr_in, origin, cap)
    Tv, ov, cv, yv, wv = _training_view(D, va_in, origin, cap)
    best, best_loss = None, np.inf
    for g in grid:
        p = dict(base); p.update(g)
        m = _fit_quantile(kind, weighted, Ti, oi, yi, wi, num, cat, vocab, cfg, p)
        Q = m.predict(Tv[ov])
        loss = _upper_loss(np.log(yv[ov]), Q, cfg["quantiles"], wv[ov] if weighted else None)
        if loss < best_loss:
            best, best_loss = p, loss
    return best, best_loss


def _aft_bounds(obs, c, y):
    lo = np.where(obs, y, np.maximum(c, 1e-3)); hi = np.where(obs, y, np.inf)
    return lo, hi


def _tune_aft(D, tr_in, va_in, origin, num, cat, vocab, cfg):
    cap = cfg["censoring"]["weight_cap"]; a = dict(cfg["models"]["aft"])
    dists, scales = a.pop("distributions"), a.pop("scales")
    Ti, oi, ci, yi, _ = _training_view(D, tr_in, origin, cap)
    Tv, ov, cv, yv, wv = _training_view(D, va_in, origin, cap)
    lo, hi = _aft_bounds(oi, ci, yi)
    best, best_loss = None, np.inf
    for d in dists:
        for s in scales:
            m = XGBAFT(num, cat, vocab, cfg["quantiles"], dist=d, scale=s, seed=cfg["seed"], n_jobs=cfg.get("n_jobs", 2), **a).fit(Ti, lo, hi)
            loss = _upper_loss(np.log(yv[ov]), m.predict(Tv[ov]), cfg["quantiles"], wv[ov])
            if loss < best_loss:
                best, best_loss = (d, s), loss
    return best, best_loss, a


def fit_origin(D, w, num, cat, vocab, cfg, variants=VARIANTS, calib_days=None, prev_chosen=None, fixed_chosen=None):
    """Fit the variants at one origin. With fixed_chosen (the hyper-parameters of another fit of the same origin),
    no tuning is done: used by the refits that only change the quantile levels or a fixed setting."""
    t_start = time.time()
    cap = cfg["censoring"]["weight_cap"]
    M = origin_masks(D, w, cfg, calib_days)
    o = w["origin"]
    T, obs, c, y, wt = _training_view(D, M["train"], o, cap)
    C = D[M["cal"]]; E = D[M["eval"]]
    c_obs, c_c, c_y = status_at(C, o)
    c_w, c_g = ipcw_weights(c_obs, c_c, c_y, cap)
    tr_in, va_in = inner_masks(D, M["train"], w, cfg, calib_days)
    res = dict(k=w["k"], origin=o, end=w["end"], cal_idx=np.flatnonzero(M["cal"]), eval_idx=np.flatnonzero(M["eval"]),
               cal_obs=c_obs, cal_y=c_y, cal_w=c_w, cal_g=c_g, n_train=int(len(T)), n_train_obs=int(obs.sum()),
               train_censored_share=float(1 - obs.mean()), pred={}, chosen={}, inner_loss={}, seconds={})
    tc = cfg["tuning"]
    tune_now = prev_chosen is None or (w["k"] - prev_chosen.get("_k", 0)) >= tc["retune_every"]
    can_tune = va_in.sum() > 200 and tr_in.sum() > 1000
    chosen = {} if (tune_now and can_tune) or prev_chosen is None else dict(prev_chosen)
    if fixed_chosen is not None:
        chosen, tune_now, can_tune = dict(fixed_chosen), False, False
    for v in variants:
        t0 = time.time()
        kind, _, mode = v.partition("_")
        if kind in ("route", "lqr", "xgbq", "qrf"):
            weighted = mode == "ipcw"
            base = dict(cfg["models"].get(kind, {})) if kind in ("xgbq", "qrf") else {}
            grid = tc.get(f"{kind}_grid")
            if grid:
                # the plain and IPCW variants share the hyper-parameters (chosen with the IPCW-weighted inner loss),
                # so that the two differ only in the weighting of the training orders
                if kind not in chosen and can_tune:
                    chosen[kind], res["inner_loss"][kind] = _tune(kind, True, D, tr_in, va_in, o, num, cat, vocab, cfg,
                                                                  base, grid)
                base = chosen.get(kind, base)
            m = _fit_quantile(kind, weighted, T, obs, y, wt, num, cat, vocab, cfg, base)
        else:                                       # aft (censored labels) and aft_naive (delivered only)
            if "aft" not in chosen:
                if can_tune:
                    ch, loss, _ = _tune_aft(D, tr_in, va_in, o, num, cat, vocab, cfg)
                    res["inner_loss"]["aft"] = loss
                else:
                    ch = ("normal", 1.0)
                chosen["aft"] = dict(dist=ch[0], scale=ch[1])
            a = dict(cfg["models"]["aft"]); a.pop("distributions"); a.pop("scales")
            m = XGBAFT(num, cat, vocab, cfg["quantiles"], dist=chosen["aft"]["dist"], scale=chosen["aft"]["scale"],
                       seed=cfg["seed"], n_jobs=cfg.get("n_jobs", 2), **a)
            if mode == "naive":
                m.fit(T[obs], y[obs], y[obs])
            else:
                lo, hi = _aft_bounds(obs, c, y)
                m.fit(T, lo, hi)
        res["pred"][v] = dict(cal=m.predict(C).astype(np.float32), eval=m.predict(E).astype(np.float32))
        res["seconds"][v] = time.time() - t0
    if tune_now and can_tune:
        chosen["_k"] = w["k"]
    res["chosen"] = chosen
    res["seconds"]["total"] = time.time() - t_start
    return res


def _fixed(pdir, tag, k):
    """Hyper-parameters chosen by the fit `tag` at origin k (None if that fit does not exist)."""
    f = pdir / f"rq1fit__{tag}__k{k:02d}.pkl"
    if not f.is_file():
        return None
    with open(f, "rb") as fh:
        return pickle.load(fh)["chosen"]


def _run_block(D, vocab, cfg, pdir, windows, variants, tag, calib_days, promise, log, chosen_from=None):
    """Fit a block of consecutive origins; the hyper-parameters are tuned at the first origin of a block (or taken
    from the last fitted origin before it when resuming) and carried over within the block."""
    num, cat = feature_set(cfg, promise)
    prev = None
    first_k = windows[0]["k"]
    for f in sorted(pdir.glob(f"rq1fit__{tag}__k*.pkl")):
        kk = int(f.stem.split("__k")[-1])
        if kk < first_k and (first_k - kk) < cfg["tuning"]["retune_every"]:
            with open(f, "rb") as fh:
                prev = pickle.load(fh)["chosen"]
    for w in windows:
        f = pdir / f"rq1fit__{tag}__k{w['k']:02d}.pkl"
        if f.is_file():
            with open(f, "rb") as fh:
                prev = pickle.load(fh)["chosen"]
            continue
        K = pd.Timedelta(days=calib_days or cfg["rolling"]["calib_days"])
        if (D["purchase_ts"] < w["origin"] - K).sum() < cfg["rolling"]["min_train"]:
            continue
        fx = _fixed(pdir, chosen_from, w["k"]) if chosen_from else None
        if chosen_from and fx is None:
            continue                                  # the reference fit skipped this origin too
        r = fit_origin(D, w, num, cat, vocab, cfg, variants, calib_days, prev, fixed_chosen=fx)
        prev = r["chosen"]
        with open(f, "wb") as fh:
            pickle.dump(r, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"  rq1 fit [{tag}] origin {w['k']:02d} {w['origin'].date()}: n_train={r['n_train']}, "
            f"{r['seconds']['total']:.0f}s " + " ".join(f"{k}={v:.0f}" for k, v in r["seconds"].items() if k != "total"))


def _worker(args):
    feat_path, vocab, cfg, pdir, windows, variants, tag, calib_days, promise, chosen_from = args
    D = pd.read_pickle(feat_path)
    _run_block(D, vocab, cfg, pdir, windows, variants, tag, calib_days, promise,
               log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True), chosen_from=chosen_from)
    return windows[0]["k"]


def run_rq1_fit(D, vocab, cfg, pdir, variants=VARIANTS, tag="main", calib_days=None, promise=None, log=print,
                workers=1, feat_path=None, chosen_from=None):
    """Fit all origins. With workers > 1, blocks of `tuning.retune_every` consecutive origins run in parallel
    processes; because the hyper-parameters are re-selected at the start of every block, the tuning schedule does not
    depend on the number of workers."""
    pdir.mkdir(parents=True, exist_ok=True)
    W = rolling_windows(cfg)
    B = cfg["tuning"]["retune_every"]
    blocks = [W[i:i + B] for i in range(0, len(W), B)]
    if workers <= 1 or feat_path is None:
        for blk in blocks:
            _run_block(D, vocab, cfg, pdir, blk, variants, tag, calib_days, promise, log, chosen_from)
        return
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(str(feat_path), vocab, cfg, pdir, blk, variants, tag, calib_days, promise, chosen_from) for blk in blocks]
    # "spawn" is the only start method on Windows; using it everywhere keeps the behaviour identical across platforms
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        for k in ex.map(_worker, jobs):
            log(f"  rq1 fit [{tag}] block starting at origin {k:02d} done")


def add_ensembles(fits, specs):
    """Add derived variants: for each name -> member list in `specs`, the mean of the members' stored log-quantiles
    (calibration and evaluation orders). Quantile averaging (Lichtendahl et al., 2013); the mean of non-crossing
    quantile vectors is non-crossing. Fits that lack a member (e.g. the sensitivity refits) are left unchanged."""
    for r in fits:
        for name, members in specs.items():
            if name in r["pred"] or not all(m in r["pred"] for m in members):
                continue
            r["pred"][name] = {s: np.mean([np.asarray(r["pred"][m][s], dtype=np.float64) for m in members], axis=0)
                               for s in ("cal", "eval")}
    return fits


def load_fits(pdir, tag="main", cfg=None):
    """Stored fits of all origins; with cfg, the configured quantile-average variant (`ensemble`) is added."""
    out = []
    for f in sorted(pdir.glob(f"rq1fit__{tag}__k*.pkl")):
        with open(f, "rb") as fh:
            out.append(pickle.load(fh))
    if cfg is not None and cfg.get("ensemble"):
        add_ensembles(out, {cfg["ensemble"]["name"]: cfg["ensemble"]["members"]})
    return out
