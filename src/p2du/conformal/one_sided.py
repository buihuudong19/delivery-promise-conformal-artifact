"""One-sided conformal corrections of an upper quantile of the delivery time, with censored calibration data.

Score of a calibration order j for a base quantile model q: s_j = log Y_j - log q(x_j) (multiplicative correction,
so the bound widens with the predicted scale). The calibrated bound is U(x) = q(x) exp(Q), where Q is a quantile of the
scores, and the promise is P(x) = floor(U(x)) days (data/delivery_sample.py explains why floor keeps the guarantee).

Calibration weighting
  naive   only the calibration orders delivered by the origin, unweighted (the common choice; it ignores that the
          slow orders of the calibration window are the ones still on their way, and so under-covers)
  ipcw    masses proportional to w_j = 1 / G(Y_j) over the observed orders (data/censoring.py), scaled to 1 - pi,
          where pi is the share of orders whose delivery time exceeds the observable horizon of the calibration window
          (measured on older orders, data/censoring.tail_share); the mass pi is placed at +infinity, so the delivery
          times that the calibration window cannot show make the correction conservative rather than optimistic

Variants
  split      one correction for all orders (split conformal; Vovk et al., 2005; Romano et al., 2019, one-sided)
  mondrian   one correction per group (customer region or state); group-conditional validity (Vovk, 2013)
  aci        adaptive conformal inference (Gibbs and Candes, 2021) with delayed feedback: the level alpha_t is updated
             with the outcomes of earlier promises as they become known (delivery, or the promised day passing):
             alpha <- alpha + gamma (alpha_target - err); main version with daily updates (see aci_promises)
  mondrian_aci  ACI run separately within each group
"""
import heapq

import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)


class ScoreECDF:
    """Weighted empirical distribution of calibration scores; quantile(beta) returns +inf if beta is not reached."""

    def __init__(self, scores, masses):
        o = np.argsort(scores, kind="stable")
        self.s = np.asarray(scores, float)[o]
        self.cm = np.cumsum(np.asarray(masses, float)[o])
        self.n = len(scores)

    def quantile(self, beta):
        beta = np.asarray(beta, float)
        idx = np.searchsorted(self.cm, beta - 1e-12, side="left")
        out = np.where(idx < len(self.s), self.s[np.minimum(idx, max(len(self.s) - 1, 0))], np.inf)
        return np.where(beta <= 0, -np.inf, out)

    def cdf_below(self, t):
        """Mass of scores strictly below t (vectorised)."""
        i = np.searchsorted(self.s, t, side="left")
        return np.where(i > 0, self.cm[np.maximum(i - 1, 0)], 0.0)


def make_ecdf(score, obs, w, weighting, tail=0.0):
    s = score[obs]
    if weighting == "naive":
        return ScoreECDF(s, np.full(len(s), 1.0 / max(len(s), 1))), len(s)
    ww = w[obs]
    return ScoreECDF(s, (1.0 - tail) * ww / max(ww.sum(), 1e-12)), len(obs)


def level(alpha, n):
    """Finite-sample level of the split-conformal quantile, (1 - alpha)(1 + 1/n)."""
    return (1 - alpha) * (1 + 1.0 / max(n, 1))


def group_ecdfs(score, obs, w, groups, weighting, min_n, tail=0.0, tail_g=None):
    pooled = make_ecdf(score, obs, w, weighting, tail)
    out = {}
    for g in pd.unique(groups):
        m = groups == g
        if m.sum() >= min_n:
            out[g] = make_ecdf(score[m], obs[m], w[m], weighting, (tail_g or {}).get(g, tail))
    return pooled, out


def promise_from_bound(logq, Q, cap):
    with np.errstate(over="ignore", invalid="ignore"):
        U = np.exp(logq + Q)
    P = np.floor(np.where(np.isfinite(U), U, cap))
    return np.clip(P, 1, cap)


def static_promises(logq_eval, groups_eval, cal, alpha, method, weighting, cfg):
    """Promises of the split and Mondrian variants (and of the uncorrected model, method 'raw')."""
    cap = cfg["conformal"]["promise_cap_days"]
    if method == "raw":
        return promise_from_bound(logq_eval, 0.0, cap)
    pooled, by_g = group_ecdfs(cal["score"], cal["obs"], cal["w"], cal["group"], weighting,
                               cfg["conformal"]["mondrian_min_n"], cal.get("tail", 0.0), cal.get("tail_g"))
    if method == "split":
        e, n = pooled
        return promise_from_bound(logq_eval, float(e.quantile(level(alpha, n))), cap)
    Q = np.empty(len(logq_eval))
    for g in pd.unique(groups_eval):
        m = groups_eval == g
        e, n = by_g.get(g, pooled)
        Q[m] = e.quantile(level(alpha, n))
    return promise_from_bound(logq_eval, Q, cap)


def aci_promises(seq, alpha, gamma, by_group, cap, step="day", alpha_min=0.01):
    """Adaptive conformal inference with delayed feedback over the whole sequence of evaluation orders.

    The outcome of a promise becomes known at the delivery or, if the promised day passes first, at the end of that
    day (a miss). step = "day": the per-order updates of the promises resolved the day before are applied once a day
    as a batch, alpha <- alpha + gamma * sum_j (alpha_target - err_j) (a per-day mean would over-weight days with few
    deliveries, such as Sundays, when the resolved outcomes are mostly misses), clipped below at alpha_min, and the
    quantile level is
    capped at the largest level the calibration distribution reaches, so the bound never exceeds the largest observed
    calibration score (main version; the platform updates its promise rule daily, and the feedback of a demand peak
    arrives in bursts). step = "order":
    the original per-observation update of Gibbs and Candes (2021) without clipping (sensitivity; with bursty, delayed
    feedback the level can turn negative, which gives an infinite bound, here capped at `cap` days).

    seq: DataFrame sorted by purchase time with columns purchase_ts, purchase_day, logq, k (origin), group,
         D (floor of the delivery time; NaN if never delivered), delivered_ts; seq.attrs["ecdfs"]: dict
         (k, group or None) -> (ECDF, n)
    Returns the promises and the level alpha_t used for each order.
    """
    ecdfs = seq.attrs["ecdfs"]
    a = {}
    heap, pending = [], {}
    P = np.empty(len(seq)); at = np.empty(len(seq))
    pt = seq["purchase_ts"].to_numpy(); pday = seq["purchase_day"].to_numpy()
    logq = seq["logq"].to_numpy(); kk = seq["k"].to_numpy(); gg = seq["group"].to_numpy()
    Dv = seq["D"].to_numpy(); dts = seq["delivered_ts"].to_numpy()
    one_day = np.timedelta64(1, "D")
    for i in range(len(seq)):
        if step == "order":
            while heap and heap[0][0] <= pt[i]:
                _, _, g, err = heapq.heappop(heap)
                a[g] = a.get(g, alpha) + gamma * (alpha - err)
        else:
            today = pday[i]
            while heap and heap[0][0] < today:                  # resolved before today's midnight
                tres, _, g, err = heapq.heappop(heap)
                d = tres.astype("datetime64[D]")
                pending.setdefault((d, g), []).append(err)
            for (d, g) in sorted(pending, key=lambda x: x[0]):
                e_d = np.asarray(pending[(d, g)], float)
                a[g] = max(alpha_min, a.get(g, alpha) + gamma * float(np.sum(alpha - e_d)))
            pending.clear()
        g = gg[i] if by_group else None
        ai = a.get(g, alpha)
        e, n = ecdfs.get((kk[i], g), ecdfs[(kk[i], None)])
        if ai >= 1:
            Q = -np.inf
        else:
            lev = level(max(ai, -1.0), n)
            if step == "day":              # keep the bound finite: at most the largest observed calibration score
                lev = min(lev, float(e.cm[-1]) if len(e.cm) else lev)
            Q = float(e.quantile(lev))
        p = float(promise_from_bound(np.array([logq[i]]), Q, cap)[0])
        P[i], at[i] = p, ai
        end = pday[i] + np.timedelta64(int(p) + 1, "D")
        delivered = not np.isnan(Dv[i])
        if delivered and dts[i] < end:
            heapq.heappush(heap, (dts[i], i, g, 0.0))
        else:
            heapq.heappush(heap, (end, i, g, 1.0))
    return P, at


def cpd_pmf(logm, ecdf, d_max, tail_day):
    """Conformal predictive distribution of the integer delivery day D = floor(Y) for each order.

    Y = m(x) exp(s), s drawn from the weighted calibration scores of the median model. Returns an array (n, d_max + 2):
    columns 0..d_max are Pr(D = d); the last column holds the mass beyond d_max plus the unassigned (censored) mass,
    to be placed at tail_day."""
    d = np.arange(d_max + 1)
    thr = np.log(d + 1.0)[None, :] - np.asarray(logm)[:, None]
    F = ecdf.cdf_below(thr)                         # Pr(Y < d + 1) = Pr(D <= d)
    F = np.minimum(np.maximum.accumulate(F, axis=1), 1.0)
    pmf = np.diff(np.concatenate([np.zeros((len(F), 1)), F], axis=1), axis=1)
    tail = np.clip(1.0 - F[:, -1], 0, 1)
    return np.concatenate([pmf, tail[:, None]], axis=1)
