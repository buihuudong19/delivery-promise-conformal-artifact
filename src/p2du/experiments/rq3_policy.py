"""RQ3: rating-aware promise policies.

Problem: choose an integer promise P_i for each order to minimise sum_i rho_i(P_i) subject to mean(P_i) <= L, with
rho_i(P) = E[ r(D_i - P) ], the expectation taken over the predictive distribution of D_i and r(s) the response
curve of RQ2. Lagrangian relaxation: for a multiplier lambda each order takes P_i(lambda) = argmin_P rho_i(P) +
lambda P (one-dimensional search); sweeping lambda traces the frontier of promise length against the low-rating rate.
Proposition 5 of the article: every point of the sweep is optimal for its own mean length L(lambda) (Everett, 1963),
L(lambda) is non-increasing, and these points are the vertices of the lower convex hull of the value function V(L).
Only if every rho_i is convex in P does the solution take the threshold (marginal-allocation) form for every budget
(Fox, 1966). Convexity fails on the data (rq3_convexity.csv), so for L between two vertices the program reports the
Lagrangian dual bound LB(L) <= V(L) and a feasible repair UB(L) >= V(L) of the vertex below L (greedy allocation of
the remaining days); the gap UB - LB bounds the loss of using the sweep (rq3_duality_gap.csv).

Predictive distribution: conformal predictive distribution of the base model's median (config rq3.model) with the
IPCW masses of the calibration orders of each origin, per region (data/censoring.py, conformal/one_sided.py).
Sensitivity (experiments/sensitivity.py, tag cpd_ens): the same procedure with the quantile average of RQ1
(config ensemble) as the base model, whose median is the mean of the members' log-medians.

Policies compared on the same evaluation orders (delivered orders, outcomes from the final snapshot)
  olist        Olist's promise; olist_shift: Olist's promise minus k days
  uniform      one promise c for every order
  route        route quantile at level tau (route table with IPCW weights, refitted at each origin)
  cpd_quantile the (1 - alpha)-quantile of the conformal predictive distribution (coverage-driven policy)
  late_optimal the Lagrangian solution for the expected late rate instead of the low-rating rate (a
               coverage-oriented allocation with the same predictive distributions)
  optimal      the Lagrangian solution for the expected low-rating rate (rating-aware)
  region budget  the same solution within each customer region at Olist's regional mean promise, so that no promise
               days move between regions (rq3_matched_regionbudget.csv, rq3_region_budget_regions.csv)
Metrics: mean promise; realised late rate 1{D > P} (model-free); realised low-rating rate mean r(D - P) (actual D,
estimated r); expected low-rating rate mean rho_i(P_i) (model-based).
"""
import numpy as np
import pandas as pd

from ..conformal.one_sided import make_ecdf, cpd_pmf
from ..data.censoring import status_at, ipcw_weights, tail_share
from ..evaluation.origins import origin_masks
from ..models.quantile_models import RouteTable
from .rq1_fit import load_fits

D_MAX = 120
LAMBDA_FINE = 300          # log-spaced multipliers of the duality certificate (plus 0)
L_CERT = (14, 32)          # integer mean lengths covered by the certificate


def load_curve(path, group="all"):
    C = pd.read_csv(path)
    C = C[C["group"] == group].sort_values("S")
    return C["S"].to_numpy(), C["r"].to_numpy()


def r_of(s, S, r):
    return np.interp(np.clip(s, S[0], S[-1]), S, r)


def predictive_pmfs(r, D, v, cfg):
    """(n_eval, D_MAX + 2) matrix of Pr(D = d) for the evaluation orders of origin record r; last column = tail."""
    E = D.iloc[r["eval_idx"]]; C = D.iloc[r["cal_idx"]]
    logm_c = r["pred"][v]["cal"][:, 0].astype(float); logm_e = r["pred"][v]["eval"][:, 0].astype(float)
    obs = r["cal_obs"]
    score = np.where(obs, np.log(np.where(obs, r["cal_y"], 1.0)) - logm_c, np.nan)
    gc = C["customer_region"].fillna("NA").to_numpy(); ge = E["customer_region"].fillna("NA").to_numpy()
    tail, tail_g = tail_share(D, r["origin"], cfg["rolling"]["calib_days"], groups="customer_region")
    pooled = make_ecdf(score, obs, r["cal_w"], "ipcw", tail)[0]
    out = np.zeros((len(E), D_MAX + 2))
    for g in pd.unique(ge):
        m = ge == g; mc = gc == g
        e = make_ecdf(score[mc], obs[mc], r["cal_w"][mc], "ipcw", tail_g.get(g, tail))[0] \
            if mc.sum() >= cfg["conformal"]["mondrian_min_n"] else pooled
        out[m] = cpd_pmf(logm_e[m], e, D_MAX, None)
    return out


def rho_matrix(pmf, Pgrid, S, r, tail_day):
    d = np.append(np.arange(D_MAX + 1), tail_day)
    Rm = r_of(d[:, None] - Pgrid[None, :], S, r)            # (d, P)
    return pmf @ Rm


def late_matrix(pmf, Pgrid, tail_day):
    d = np.append(np.arange(D_MAX + 1), tail_day)
    return pmf @ (d[:, None] > Pgrid[None, :]).astype(float)


def _policy_rows(k, family, param, P, Dv, S, r, rho, Pgrid, late_mod, region):
    j = np.clip(P.astype(int) - Pgrid[0], 0, len(Pgrid) - 1)
    ii = np.arange(len(P))
    row = dict(k=k, policy=family, param=float(param), mean_promise=float(P.mean()),
               late_realised=float((Dv > P).mean()), low_realised=float(r_of(Dv - P, S, r).mean()),
               low_expected=float(rho[ii, j].mean()), late_expected=float(late_mod[ii, j].mean()), n=len(P))
    return row


def _interp_at(df, x_col, y_col, x0):
    d = df.sort_values(x_col)
    x, y = d[x_col].to_numpy(), d[y_col].to_numpy()
    if x0 < x.min() or x0 > x.max():
        return np.nan
    return float(np.interp(x0, x, y))


def run_rq3(D, cfg, out, log=print, curve="rq2_response_curve.csv", eval_curve=None, tag="main", model=None,
            tail_day=None):
    """model: base model of the predictive distribution; None = config rq3.model. A derived variant (the quantile
    average) is built from the stored fits when it is requested."""
    t = out / "tables"
    v = model or cfg["rq3"]["model"]
    fits = load_fits(out / "data" / "parts", "main", cfg if model is not None else None)
    if v not in fits[0]["pred"]:
        raise KeyError(f"rq3: base model {v} is not among the stored fits")
    S_opt, r_opt = load_curve(t / curve)
    S_ev, r_ev = load_curve(t / (eval_curve or curve))
    lo, hi = cfg["rq3"]["promise_grid"]
    Pgrid = np.arange(lo, hi + 1)
    tail_day = tail_day or cfg["rolling"]["calib_days"] + cfg["rq3"]["tail_offset_days"]
    lambdas = np.concatenate([[0.0], np.logspace(-5, -0.7, cfg["rq3"]["lambda_grid"])])
    alphas = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    route_taus = np.array([0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99])
    rows, matched, reg_rows, conv_rows, gap_rows, rb_rows, rbr_rows = [], [], [], [], [], [], []
    for rec in fits:
        k = rec["k"]
        E = D.iloc[rec["eval_idx"]]
        ok = E["delivered"].to_numpy()
        pmf_all = predictive_pmfs(rec, D, v, cfg)
        pmf = pmf_all[ok]; Ed = E[ok]; Dv = Ed["D"].to_numpy(); reg = Ed["customer_region"].fillna("NA").to_numpy()
        rho = rho_matrix(pmf, Pgrid, S_opt, r_opt, tail_day)
        rho_ev = rho if eval_curve is None else rho_matrix(pmf, Pgrid, S_ev, r_ev, tail_day)
        late_mod = late_matrix(pmf, Pgrid, tail_day)
        # Lemma 4 (monotonicity) and Remark 7 (discrete convexity) of rho_i over the promise grid
        d1 = np.diff(rho, axis=1); d2 = np.diff(rho, 2, axis=1)
        conv_rows.append(dict(k=k, share_nonincreasing=float((d1 <= 1e-9).all(1).mean()),
                              share_convex=float((d2 >= -1e-9).all(1).mean()),
                              share_convex_P10_40=float((d2[:, 9:39] >= -1e-9).all(1).mean()),
                              share_convex_P15_40=float((d2[:, 14:39] >= -1e-9).all(1).mean()),
                              mean_rho_at_olist=float(rho[np.arange(len(Dv)), np.clip(Ed["P_olist"].to_numpy() - lo, 0, len(Pgrid) - 1)].mean())))
        P_ol = Ed["P_olist"].to_numpy().astype(float)
        W = []
        W.append(_policy_rows(k, "olist", 0, P_ol, Dv, S_ev, r_ev, rho_ev, Pgrid, late_mod, reg))
        for sh in range(-5, 16):
            W.append(_policy_rows(k, "olist_shift", sh, np.clip(P_ol - sh, lo, hi), Dv, S_ev, r_ev, rho_ev, Pgrid, late_mod, reg))
        for c in range(3, 46):
            W.append(_policy_rows(k, "uniform", c, np.full(len(Dv), float(c)), Dv, S_ev, r_ev, rho_ev, Pgrid, late_mod, reg))
        # route table refitted at the origin with the IPCW-weighted training orders
        M = origin_masks(D, dict(origin=rec["origin"], end=rec["end"]), cfg)
        Tr = D[M["train"]]
        obs, c, y = status_at(Tr, rec["origin"])
        w, _ = ipcw_weights(obs, c, y, cfg["censoring"]["weight_cap"])
        rt = RouteTable(route_taus, cfg["models"]["route"]["shrink_m"]).fit(Tr[obs], np.log(y[obs]), w[obs])
        RQ = np.exp(rt.predict(Ed))
        for j, tau in enumerate(route_taus):
            W.append(_policy_rows(k, "route", tau, np.clip(np.floor(RQ[:, j]), lo, hi), Dv, S_ev, r_ev, rho_ev, Pgrid, late_mod, reg))
        cdf = np.cumsum(pmf[:, :D_MAX + 1], axis=1)
        for a in alphas:
            P = np.argmax(cdf >= 1 - a, axis=1).astype(float)
            P[~(cdf >= 1 - a).any(1)] = hi
            W.append(_policy_rows(k, "cpd_quantile", a, np.clip(P, lo, hi), Dv, S_ev, r_ev, rho_ev, Pgrid, late_mod, reg))
        Popt = {}
        for lam in lambdas:
            P = Pgrid[np.argmin(rho + lam * Pgrid[None, :], axis=1)].astype(float)
            Popt[lam] = P
            W.append(_policy_rows(k, "optimal", lam, P, Dv, S_ev, r_ev, rho_ev, Pgrid, late_mod, reg))
        for lam in lambdas:
            P = Pgrid[np.argmin(late_mod + lam * Pgrid[None, :], axis=1)].astype(float)
            W.append(_policy_rows(k, "late_optimal", lam, P, Dv, S_ev, r_ev, rho_ev, Pgrid, late_mod, reg))
        Wd = pd.DataFrame(W); rows.append(Wd)
        if tag == "main":
            lam_fine = np.concatenate([[0.0], np.logspace(-5, -0.3, LAMBDA_FINE)])
            for g in duality_gap(rho, Pgrid, np.arange(float(L_CERT[0]), L_CERT[1] + 0.5, 1.0), lam_fine):
                gap_rows.append(dict(k=k, **g))
        # matched comparisons at Olist's mean promise of this window
        L0 = float(P_ol.mean()); olr = Wd[Wd.policy == "olist"].iloc[0]
        mrow = dict(k=k, origin=rec["origin"], L_olist=L0, low_olist=olr["low_realised"], late_olist=olr["late_realised"],
                    low_expected_olist=olr["low_expected"],
                    low_actual_olist=float(Ed["low"].mean()), n_reviewed=int(Ed["low"].notna().sum()))
        for fam in ("uniform", "route", "cpd_quantile", "late_optimal", "optimal", "olist_shift"):
            F = Wd[Wd.policy == fam]
            for col in ("low_realised", "late_realised", "low_expected"):
                mrow[f"{fam}_{col}_at_L"] = _interp_at(F, "mean_promise", col, L0)
            # promise length that reaches Olist's realised low-rating rate
            mrow[f"{fam}_L_at_low"] = _interp_at(F, "low_realised", "mean_promise", olr["low_realised"])
        matched.append(mrow)
        # promise by region under each policy at the matched length (optimal: closest lambda)
        Fo = Wd[Wd.policy == "optimal"].reset_index(drop=True)
        lam_star = Fo.loc[(Fo["mean_promise"] - L0).abs().idxmin(), "param"]
        Pst = Popt[lam_star]
        mrow["optimal_share_at_max_P"] = float((Pst >= hi).mean())       # does the grid limit bind?
        mrow["olist_share_above_max_P"] = float((P_ol > hi).mean())
        Fc = Wd[Wd.policy == "cpd_quantile"].reset_index(drop=True)
        a_star = Fc.loc[(Fc["mean_promise"] - L0).abs().idxmin(), "param"]
        Pc = np.argmax(cdf >= 1 - a_star, axis=1).astype(float); Pc[~(cdf >= 1 - a_star).any(1)] = hi
        for g in sorted(pd.unique(reg)):
            m = reg == g
            for pol, P in (("olist", P_ol), ("optimal", Pst), ("cpd_quantile", np.clip(Pc, lo, hi))):
                reg_rows.append(dict(k=k, region=g, policy=pol, n=int(m.sum()), mean_promise=float(P[m].mean()),
                                     late_realised=float((Dv[m] > P[m]).mean()),
                                     low_realised=float(r_of(Dv[m] - P[m], S_ev, r_ev).mean())))
        if tag == "main":
            lo_rb, la_rb, L_rb, regs = region_budget_policy(rho, Pgrid, lambdas, Dv, S_ev, r_ev, reg, P_ol)
            rb_rows.append(dict(k=k, origin=rec["origin"], L_olist=L0, low_olist=olr["low_realised"],
                                late_olist=olr["late_realised"], optimal_low_realised_at_L=lo_rb,
                                optimal_late_realised_at_L=la_rb, optimal_L_at_low=L_rb,
                                single_low_realised_at_L=mrow["optimal_low_realised_at_L"],
                                single_late_realised_at_L=mrow["optimal_late_realised_at_L"]))
            for x in regs:
                m_ = reg == x["region"]
                rbr_rows.append(dict(k=k, region=x["region"], n=x["n"], olist_mean_promise=float(P_ol[m_].mean()),
                                     olist_low_realised=float(r_of(Dv[m_] - P_ol[m_], S_ev, r_ev).mean()),
                                     olist_late_realised=float((Dv[m_] > P_ol[m_]).mean()),
                                     budget_mean_promise=x["mean_promise"], budget_low_realised=x["low_realised"],
                                     budget_late_realised=x["late_realised"]))
        if tag == "main" and k == fits[len(fits) // 2]["k"]:
            # example orders for the illustration of rho_i(P)
            rng = np.random.default_rng(cfg["seed"]); pick = rng.choice(len(Dv), 6, replace=False)
            ex = pd.DataFrame([dict(order=int(i), P=int(p), rho=float(rho[i, j]), P_olist=float(P_ol[i]),
                                    region=reg[i]) for i in pick for j, p in enumerate(Pgrid)])
            ex.to_csv(t / "rq3_rho_examples.csv", index=False)
        log(f"  rq3 [{tag}] origin {k:02d}: Olist L={L0:.1f} low={olr['low_realised']:.4f}; optimal at L: "
            f"{mrow['optimal_low_realised_at_L']:.4f}")
    suffix = "" if tag == "main" else "_" + tag
    A = pd.concat(rows, ignore_index=True)
    A.to_csv(t / f"rq3_policy_windows{suffix}.csv", index=False)
    Mt = pd.DataFrame(matched); Mt.to_csv(t / f"rq3_matched{suffix}.csv", index=False)
    if tag == "main":
        pd.DataFrame(reg_rows).to_csv(t / "rq3_region_promises.csv", index=False)
        pd.DataFrame(conv_rows).to_csv(t / "rq3_convexity.csv", index=False)
        pd.DataFrame(gap_rows).to_csv(t / "rq3_duality_gap.csv", index=False)
        curve_calibration(D, fits, cfg, t).to_csv(t / "rq3_curve_calibration.csv", index=False)
        frontier(A).to_csv(t / "rq3_frontier.csv", index=False)
        pd.DataFrame(rb_rows).to_csv(t / "rq3_matched_regionbudget.csv", index=False)
        pd.DataFrame(rbr_rows).to_csv(t / "rq3_region_budget_regions.csv", index=False)
    return A, Mt


def region_budget_policy(rho, Pgrid, lambdas, Dv, S, r, reg, P_ol, shifts=np.arange(-10.0, 10.01, 0.25)):
    """Rating-aware promises solved separately within each region, with the region's mean promise fixed at Olist's
    mean promise in that region (no promise days move between regions). For each region the Lagrangian sweep gives
    the frontier (mean promise, realised low-rating sum, late sum); values at a budget are interpolated on it.
    Returns the window values at Olist's regional budgets, the promise length at Olist's realised low-rating rate when
    every regional budget moves by the same shift, and the per-region values."""
    n = len(Dv); fr = {}
    for g in pd.unique(reg):
        m = reg == g
        pts = []
        for lam in lambdas:
            P = Pgrid[np.argmin(rho[m] + lam * Pgrid[None, :], axis=1)].astype(float)
            pts.append((P.mean(), float(r_of(Dv[m] - P, S, r).mean()), float((Dv[m] > P).mean())))
        pts = np.array(sorted(pts))
        fr[g] = (int(m.sum()), float(P_ol[m].mean()), pts)

    def at(g, L):
        ng, _, pts = fr[g]
        if L < pts[0, 0] or L > pts[-1, 0]:
            return np.nan, np.nan
        return float(np.interp(L, pts[:, 0], pts[:, 1])), float(np.interp(L, pts[:, 0], pts[:, 2]))

    def window(shift):
        lo_, la_ = 0.0, 0.0
        for g, (ng, Lg, _) in fr.items():
            a, b = at(g, Lg + shift)
            lo_ += ng * a; la_ += ng * b
        return lo_ / n, la_ / n

    low0, late0 = window(0.0)
    regions = [dict(region=g, n=ng, mean_promise=Lg, low_realised=at(g, Lg)[0], late_realised=at(g, Lg)[1])
               for g, (ng, Lg, _) in fr.items()]
    olist_low = float(r_of(Dv - P_ol, S, r).mean())
    curve = [(float(P_ol.mean() + sh), window(sh)[0]) for sh in shifts]
    curve = np.array([c for c in curve if np.isfinite(c[1])])
    L_at_low = np.nan
    if len(curve) > 1:
        o = np.argsort(curve[:, 1])
        if curve[:, 1].min() <= olist_low <= curve[:, 1].max():
            L_at_low = float(np.interp(olist_low, curve[o, 1], curve[o, 0]))
    return low0, late0, L_at_low, regions


def duality_gap(rho, Pgrid, L_grid, lambdas, n_bisect=60):
    """Dual lower bound and feasible upper bound of min mean rho_i(P_i) s.t. mean P_i <= L, for each L in L_grid.

    For each L the multiplier is refined by bisection (on log lambda, starting from the bracket in `lambdas`) until the
    two Lagrangian vertices adjacent to L are found: L(lam_hi) <= L <= L(lam_lo). The dual function is concave in lambda
    with subgradient L(lambda) - L, so its maximum lies in that bracket (LB = max over every evaluated lambda). The
    feasible repair starts from the vertex just below L and spends the remaining integer budget greedily on the largest
    one-day decrease of rho_i (UB). LB <= V(L) <= UB, so UB - LB bounds the loss of the Lagrangian policy."""
    import heapq
    n = rho.shape[0]; ii = np.arange(n)
    Pf = Pgrid.astype(float)

    def solve(lam):
        v = rho + lam * Pf[None, :]
        jj = v.argmin(1)
        return jj, float(v[ii, jj].mean()), float(Pf[jj].mean()), float(rho[ii, jj].mean())

    base = [solve(lam) for lam in lambdas]
    out = []
    for L in L_grid:
        evals = list(zip(lambdas, base))
        LB = max(mv - lam * L for lam, (_, mv, _, _) in evals)
        feas = [(lam, r) for lam, r in evals if r[2] <= L + 1e-12]
        if not feas:
            out.append(dict(L=float(L), LB=LB, UB=np.nan, gap=np.nan, vertex_L=np.nan)); continue
        lam_hi, r_hi = min(feas, key=lambda e: e[0])                # smallest feasible multiplier: longest feasible vertex
        infeas = [(lam, r) for lam, r in evals if r[2] > L + 1e-12 and lam < lam_hi]
        if infeas:
            lam_lo, r_lo = max(infeas, key=lambda e: e[0])
            for _ in range(n_bisect):
                mid = np.sqrt(lam_lo * lam_hi) if lam_lo > 0 else lam_hi / 2.0
                r = solve(mid)
                LB = max(LB, r[1] - mid * L)
                if r[2] <= L + 1e-12:
                    lam_hi, r_hi = mid, r
                else:
                    lam_lo, r_lo = mid, r
                if lam_hi - lam_lo <= 1e-12 * max(lam_hi, 1e-12):
                    break
        jj = r_hi[0].copy()
        budget = int(np.floor(n * L - Pf[jj].sum() + 1e-9))
        h = [(-(rho[i, jj[i]] - rho[i, jj[i] + 1]), i) for i in range(n) if jj[i] + 1 < len(Pgrid)]
        heapq.heapify(h)
        while budget > 0 and h:
            dlt, i = heapq.heappop(h)
            if -dlt <= 0:
                break
            jj[i] += 1; budget -= 1
            if jj[i] + 1 < len(Pgrid):
                heapq.heappush(h, (-(rho[i, jj[i]] - rho[i, jj[i] + 1]), i))
        UB = min(float(rho[ii, jj].mean()), r_hi[3])
        out.append(dict(L=float(L), LB=float(LB), UB=UB, gap=UB - float(LB), vertex_L=r_hi[2]))
    return out


def curve_calibration(D, fits, cfg, t):
    """Out-of-time check of the response curve: r(s) fitted on 2017 orders (and the full-sample curve) predicts the
    low-rating rate of the reviewed orders of every evaluation window under Olist's actual promise; compared with the
    observed rate. The 2018 windows are out of sample for the 2017 curve."""
    S7, r7 = load_curve(t / "rq2_response_curve_fit2017.csv")
    Sf, rf = load_curve(t / "rq2_response_curve.csv")
    rows = []
    for rec in fits:
        E = D.iloc[rec["eval_idx"]]
        R = E[E["delivered"] & E["low"].notna()]
        s = R["S_olist"].to_numpy()
        rows.append(dict(k=rec["k"], origin=rec["origin"], n=len(R), observed=float(R["low"].mean()),
                         predicted_fit2017=float(r_of(s, S7, r7).mean()), predicted_full=float(r_of(s, Sf, rf).mean()),
                         out_of_sample=bool(rec["origin"] >= pd.Timestamp("2018-01-01"))))
    return pd.DataFrame(rows)


def frontier(A):
    """Pooled frontier: the mean over windows of each policy parameter (same parameter in every window)."""
    return A.groupby(["policy", "param"]).agg(mean_promise=("mean_promise", "mean"),
                                              late_realised=("late_realised", "mean"),
                                              low_realised=("low_realised", "mean"),
                                              low_expected=("low_expected", "mean"),
                                              late_expected=("late_expected", "mean"),
                                              n_windows=("k", "nunique")).reset_index()
