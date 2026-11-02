"""Tests of the P2DU pipeline: leakage guards on real data and property tests of the methods on synthetic data.

    python -m pytest tests -q          (or: python tests/test_p2du.py)

The data tests need the feature file (python run_pipeline.py --stage data) and are skipped without it; the tests of the
fitted origins also need at least one file of the stage rq1_fit.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2du.config import load_config                                          # noqa: E402
from p2du.conformal.one_sided import ScoreECDF, make_ecdf, level, promise_from_bound, cpd_pmf  # noqa: E402
from p2du.data.censoring import status_at, ipcw_weights, censoring_survival  # noqa: E402
from p2du.data.purchase_features import feature_set, OUTCOME_COLS            # noqa: E402
from p2du.evaluation.origins import rolling_windows, origin_masks            # noqa: E402

CFG = load_config()
FEAT = CFG["out_dir"] / "data" / "features_purchase.pkl"
F = pd.read_pickle(FEAT) if FEAT.is_file() else None


def _skip():
    return F is None


# ----------------------------------------------------------------------------------------------- leakage guards
def test_feature_set_has_no_outcome():
    for promise in (False, True):
        num, cat = feature_set(CFG, promise)
        used = set(num + cat)
        assert not used & set(OUTCOME_COLS), used & set(OUTCOME_COLS)
        assert not any(c.startswith("p_") for c in used)
        assert ("f_olist_promise" in used) == promise


def test_asof_seller_delivery_history_matches_bruteforce():
    if _skip():
        return
    dev = F[F["delivered"]][["seller_id", "delivered_ts"]]
    rng = np.random.default_rng(0)
    for i in rng.choice(len(F), 300, replace=False):
        r = F.iloc[i]
        n = int(((dev["seller_id"] == r["seller_id"]) & (dev["delivered_ts"] < r["purchase_ts"])).sum())
        assert n == r["h_seller_deliv_n"], (r["order_id"], n, r["h_seller_deliv_n"])


def test_asof_route_history_uses_only_past_deliveries():
    if _skip():
        return
    rng = np.random.default_rng(1)
    for i in rng.choice(len(F), 200, replace=False):
        r = F.iloc[i]
        past = F[(F["route"] == r["route"]) & F["delivered"] & (F["delivered_ts"] < r["purchase_ts"])]
        assert len(past) == r["h_route_n"]


def test_origin_sets_and_censoring():
    if _skip():
        return
    for w in rolling_windows(CFG)[::5]:
        M = origin_masks(F, w, CFG)
        K = pd.Timedelta(days=CFG["rolling"]["calib_days"])
        assert (F.loc[M["train"], "purchase_ts"] < w["origin"] - K).all()
        assert ((F.loc[M["cal"], "purchase_ts"] >= w["origin"] - K) & (F.loc[M["cal"], "purchase_ts"] < w["origin"])).all()
        assert (F.loc[M["eval"], "purchase_ts"] >= w["origin"]).all()
        assert not (M["train"] & M["cal"]).any() and not (M["cal"] & M["eval"]).any()
        C = F[M["cal"]]
        obs, c, y = status_at(C, w["origin"])
        # no calibration order contributes an outcome that is known only after the origin
        assert (C.loc[obs, "delivered_ts"] < w["origin"]).all()
        assert np.all(np.isnan(y[~obs]))
        assert np.all(y[obs] < c[obs] + 1e-9)
        wt, _ = ipcw_weights(obs, c, y, CFG["censoring"]["weight_cap"])
        assert np.all(wt[~obs] == 0) and np.all(wt[obs] >= 1)


def test_stored_fits_respect_the_origin():
    from p2du.experiments.rq1_fit import load_fits
    if _skip():
        return
    fits = load_fits(CFG["out_dir"] / "data" / "parts")
    for r in fits[:3]:
        C = F.iloc[r["cal_idx"]]
        assert (C.loc[r["cal_obs"], "delivered_ts"] < r["origin"]).all()
        assert (F.iloc[r["eval_idx"]]["purchase_ts"] >= r["origin"]).all()


# ----------------------------------------------------------------------------------------------- methods
def test_proposition1_floor_keeps_the_bound():
    rng = np.random.default_rng(2)
    U = rng.uniform(1, 40, 10000); Y = U - rng.exponential(2, 10000)
    Y = np.maximum(Y, 0.01)
    P = promise_from_bound(np.log(U), 0.0, 90)
    assert np.all(np.floor(Y) <= P)           # Y <= U implies a kept promise


def test_proposition2_ipcw_is_unbiased_under_independent_censoring():
    rng = np.random.default_rng(3)
    n = 200000
    Y = rng.lognormal(2.3, 0.5, n); C = rng.uniform(0, 60, n)
    obs = Y < C
    w, _ = ipcw_weights(obs, C, np.where(obs, Y, np.nan), cap=1e9)
    for t in (5, 10, 20, 30):
        est = (w[obs] * (Y[obs] <= t)).sum() / n
        assert abs(est - (Y <= t).mean()) < 0.01, (t, est, (Y <= t).mean())
    naive = (Y[obs] <= 20).mean()
    assert naive > (Y <= 20).mean() + 0.01   # ignoring censoring is optimistic about short delivery times


def test_split_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(4)
    cov = []
    for rep in range(200):
        q = np.full(1000, np.log(10.0))
        y = rng.lognormal(2.3, 0.4, 1000)
        e, n = make_ecdf(np.log(y) - q, np.ones(1000, bool), np.ones(1000), "naive")
        Q = float(e.quantile(level(0.1, n)))
        yt = rng.lognormal(2.3, 0.4, 500)
        cov.append(np.mean(np.log(yt) <= np.log(10.0) + Q))
    assert np.mean(cov) >= 0.895


def test_cpd_pmf_is_a_distribution():
    rng = np.random.default_rng(5)
    s = rng.normal(0, 0.4, 500)
    e = ScoreECDF(s, np.full(500, 0.97 / 500))
    pmf = cpd_pmf(np.log([5.0, 12.0, 30.0]), e, 120, None)
    assert np.allclose(pmf.sum(1), 1.0) and (pmf >= -1e-12).all()
    assert np.allclose(pmf[:, -1], 0.03, atol=1e-6)


def test_lagrangian_monotone_in_lambda():
    rng = np.random.default_rng(6)
    P = np.arange(1, 61)
    rho = np.exp(-np.outer(rng.uniform(0.05, 0.3, 300), P)) * 0.6 + 0.08
    means = [P[np.argmin(rho + lam * P[None, :], axis=1)].mean() for lam in np.logspace(-5, -0.7, 30)]
    assert np.all(np.diff(means) <= 1e-9)


def test_duality_gap_bounds():
    from p2du.experiments.rq3_policy import duality_gap
    rng = np.random.default_rng(7)
    P = np.arange(1, 41)
    # S-shaped (non-convex) expected low-rating curves, as on the data
    c = rng.uniform(8, 25, 400)[:, None]
    rho = 0.08 + 0.7 / (1 + np.exp((P[None, :] - c) / 2.0))
    lam = np.concatenate([[0.0], np.logspace(-5, -0.3, 200)])
    G = duality_gap(rho, P, np.arange(10.0, 30.0, 2.0), lam)
    for g in G:
        assert g["LB"] <= g["UB"] + 1e-12          # weak duality and feasibility
        assert g["vertex_L"] <= g["L"] + 1e-9      # the repaired vertex is feasible


def test_aci_daily_batch_stays_finite():
    from p2du.conformal.one_sided import ScoreECDF, aci_promises
    rng = np.random.default_rng(8)
    n = 3000
    t0 = pd.Timestamp("2018-01-01")
    pt = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 60, n)), unit="D")
    Y = rng.lognormal(2.3, 0.5, n)
    Y[(pt > t0 + pd.Timedelta(days=20)) & (pt < t0 + pd.Timedelta(days=30))] *= 2.5   # a demand peak
    S = pd.DataFrame(dict(purchase_ts=pt, purchase_day=pt.normalize(), logq=np.log(np.full(n, 15.0)), k=1, group="x",
                          D=np.floor(Y + (pt - pt.normalize()) / pd.Timedelta(days=1)),
                          delivered_ts=pt.normalize() + pd.to_timedelta(Y, unit="D")))
    sc = rng.normal(0, 0.4, 2000)
    S.attrs["ecdfs"] = {(1, None): (ScoreECDF(sc, np.full(2000, 0.98 / 2000)), 2000)}
    P, at = aci_promises(S, 0.1, 0.0005, False, 90, "day", 0.05)
    assert np.isfinite(P).all() and P.max() < 90 and at.min() >= 0.05 - 1e-12


def test_quantile_average_is_mean_of_members_and_non_crossing():
    from p2du.experiments.rq1_fit import add_ensembles
    rng = np.random.default_rng(9)
    def q(n):   # non-crossing log-quantiles (rows increasing)
        return np.cumsum(np.abs(rng.normal(0.3, 0.1, (n, 4))), axis=1).astype(np.float32)
    fits = [dict(pred={m: dict(cal=q(50), eval=q(30)) for m in ("a", "b", "c")}),
            dict(pred={m: dict(cal=q(50), eval=q(30)) for m in ("a", "b")})]          # member "c" missing: skipped
    add_ensembles(fits, {"ens": ["a", "b", "c"]})
    e = fits[0]["pred"]["ens"]
    for s_ in ("cal", "eval"):
        ref = (fits[0]["pred"]["a"][s_].astype(float) + fits[0]["pred"]["b"][s_] + fits[0]["pred"]["c"][s_]) / 3
        assert np.allclose(e[s_], ref) and (np.diff(e[s_], axis=1) >= 0).all()
    assert "ens" not in fits[1]["pred"]
    add_ensembles(fits, {"ens": ["a", "b"]})                                            # never overwrites
    assert np.allclose(fits[0]["pred"]["ens"]["eval"], e["eval"])


def test_tail_share_bounds():
    from p2du.data.censoring import tail_share
    if _skip():
        return
    for w in rolling_windows(CFG)[::6]:
        o, K = w["origin"], CFG["rolling"]["calib_days"]
        lo, _ = tail_share(F, o, K); hi, _ = tail_share(F, o, K, mode="upper")
        m = (F["purchase_ts"] >= o - pd.Timedelta(days=K + 28)) & (F["purchase_ts"] < o - pd.Timedelta(days=K))
        X = F[m]; truth = float((X.loc[X["delivered"], "Y"] >= K).mean())
        old = X[X["delivered_ts"] < o]; ref = float((old["Y"] >= K).mean())
        assert abs(lo - ref) < 1e-12                     # the main (observed) share is unchanged
        assert lo <= truth + 1e-12 <= hi + 2e-12         # lower and upper bound of the final-snapshot share


def test_rq3_cpd_of_quantile_average_is_a_distribution():
    """The predictive distribution built from the quantile average (RQ3 sensitivity) is a proper pmf, and its median
    input is the mean of the members' log-medians."""
    from p2du.experiments.rq1_fit import load_fits
    from p2du.experiments.rq3_policy import predictive_pmfs
    if _skip():
        return
    pdir = CFG["out_dir"] / "data" / "parts"
    if not list(pdir.glob("rq1fit__main__k*.pkl")):
        return
    rec = load_fits(pdir, "main", CFG)[0]
    ens = CFG["ensemble"]["name"]
    ref = np.mean([rec["pred"][m]["eval"][:, 0].astype(float) for m in CFG["ensemble"]["members"]], axis=0)
    assert np.allclose(rec["pred"][ens]["eval"][:, 0], ref)
    pmf = predictive_pmfs(rec, F, ens, CFG)
    assert (pmf >= -1e-12).all() and np.allclose(pmf.sum(1), 1.0, atol=1e-9)


def test_truncated_split_conformal_exact_coverage():
    """Corollary 1: with independent administrative censoring, split conformal on min(Y, c0) over the orders with
    follow-up >= c0 covers Y at level 1 - alpha in finite samples (Monte Carlo over repetitions)."""
    from p2du.experiments.sensitivity import truncated_split_bound
    rng = np.random.default_rng(3)
    alpha, c0, K, hits = 0.1, 28.0, 42.0, []
    for _ in range(400):
        n = 300
        x = rng.uniform(0, 1, n + 1)
        y = np.exp(2.3 + 0.5 * x + rng.normal(0, 0.6, n + 1))            # delivery times, heavy right tail
        c = rng.uniform(0, K, n)                                         # follow-up, independent of (x, y)
        obs = y[:n] < c
        q = np.exp(2.3 + 0.5 * x)                                        # a (deliberately miscalibrated) median
        U = truncated_split_bound(np.where(obs, y[:n], np.nan), obs, c, q[:n], q[n:], alpha, c0)
        hits.append(float(y[n] <= U[0]))
    assert np.mean(hits) >= 1 - alpha - 0.03, np.mean(hits)


def test_lee_bounds_collapse_without_selection_and_bracket_the_truth():
    """Lee bounds equal the raw contrast when both cells answer after arrival at the same rate, and bracket the true
    jump among always-responders under monotone selection."""
    from p2du.experiments.rq2_response import lee_bounds
    rng = np.random.default_rng(5)
    n = 20000
    S = rng.choice([0, 1], n)
    y = rng.binomial(1, np.where(S == 1, 0.25, 0.15))                    # true jump 0.10 for everyone
    R = pd.DataFrame(dict(S=S, low=y, answered_before_delivery=np.zeros(n, int)))
    b = lee_bounds(R, (0,), (1,), n_boot=50)
    assert abs(b["bound_lo"] - b["raw_jump"]) < 1e-12 and abs(b["bound_hi"] - b["raw_jump"]) < 1e-12
    before = (S == 1) & (y == 1) & (rng.uniform(size=n) < 0.3)            # unhappy late customers answer early
    R["answered_before_delivery"] = before.astype(int)
    b = lee_bounds(R, (0,), (1,), n_boot=50)
    assert b["bound_lo"] - 0.02 <= 0.10 <= b["bound_hi"] + 0.02


def test_censoring_survival():
    G = censoring_survival([1, 2, 3, 4])
    assert G(0.5) == 1.0 and G(2) == 0.5 and G(4) == 0.0


def test_xfit_halves_partition_balanced_by_period():
    from p2du.experiments.rq2_response import xfit_halves
    rng = np.random.default_rng(3)
    ts = pd.Timestamp("2017-01-01") + pd.to_timedelta(rng.uniform(0, 400, 5000), unit="D")
    R = pd.DataFrame(dict(purchase_ts=ts))
    A = xfit_halves(R, seed=42)
    assert A.dtype == bool and len(A) == len(R) and (A | ~A).all() and not (A & ~A).any()
    per = ((R["purchase_ts"] - R["purchase_ts"].min()) // pd.Timedelta(days=14)).to_numpy()
    for p in np.unique(per):
        m = per == p
        assert abs(int(A[m].sum()) - int((~A[m]).sum())) <= 1
    assert (xfit_halves(R, seed=42) == A).all()


def test_cluster_se_equals_iid_se_with_singleton_clusters():
    from p2du.experiments.rq2_response import cluster_se
    rng = np.random.default_rng(4)
    x = rng.normal(size=500)
    assert abs(cluster_se(x - x.mean(), np.arange(500)) - x.std(ddof=1) / np.sqrt(500)) < 1e-12
    # clustering duplicated observations widens the interval
    xx = np.repeat(x, 2)
    assert cluster_se(xx - xx.mean(), np.repeat(np.arange(500), 2)) > xx.std(ddof=1) / np.sqrt(1000)


def test_notail_equals_ipcw_when_tail_share_is_zero():
    from p2du.conformal.one_sided import static_promises
    rng = np.random.default_rng(5)
    n = 800
    score = rng.normal(size=n); obs = rng.uniform(size=n) < 0.8; w = rng.uniform(1, 3, n)
    cal = dict(score=np.where(obs, score, np.nan), obs=obs, w=w, group=np.array(["a"] * n), tail=0.0, tail_g={})
    lq = rng.normal(2.5, 0.3, 300); ge = np.array(["a"] * 300)
    p_ipcw = static_promises(lq, ge, cal, 0.10, "split", "ipcw", CFG)
    p_notail = static_promises(lq, ge, dict(cal, tail=0.0, tail_g={}), 0.10, "split", "ipcw", CFG)
    assert (p_ipcw == p_notail).all()
    # with a positive tail share the tail term can only lengthen the promises
    p_tail = static_promises(lq, ge, dict(cal, tail=0.03), 0.10, "split", "ipcw", CFG)
    assert (p_tail >= p_notail).all()


def test_region_budget_policy_respects_each_regional_budget():
    from p2du.experiments.rq3_policy import region_budget_policy
    rng = np.random.default_rng(6)
    n, Pg = 600, np.arange(1, 61)
    S = np.arange(-40, 31); r = 0.08 + 0.4 / (1 + np.exp(-(S - 2) / 2.0))
    reg = rng.choice(["N", "S", "E"], n)
    mu = np.where(reg == "N", 30, np.where(reg == "S", 15, 22))
    Dv = np.clip(np.round(rng.gamma(4, mu / 4)), 0, 80)
    pmf = np.zeros((n, 122))
    for i in range(n):
        d = np.clip(np.round(rng.gamma(4, mu[i] / 4, 200)), 0, 120).astype(int)
        pmf[i, :121] = np.bincount(d, minlength=121)[:121] / 200
    d = np.append(np.arange(121), 49)
    rho = pmf @ np.interp(np.clip(d[:, None] - Pg[None, :], S[0], S[-1]), S, r)
    P_ol = np.clip(np.round(mu * 1.1), 1, 60).astype(float)
    lambdas = np.concatenate([[0.0], np.logspace(-5, -0.7, 60)])
    low, late, L_at_low, regs = region_budget_policy(rho, Pg, lambdas, Dv, S, r, reg, P_ol)
    for x in regs:
        assert abs(x["mean_promise"] - P_ol[reg == x["region"]].mean()) < 1e-9
    assert np.isfinite(low) and np.isfinite(late) and r.min() <= low <= r.max() and 0 <= late <= 1
    # the window value is the order-weighted mean of the regional values
    tot = sum(x["n"] * x["low_realised"] for x in regs) / n
    assert abs(tot - low) < 1e-12


def test_block_bootstrap_gives_every_window_the_same_weight():
    # circular blocks: the bootstrap mean is centred on the sample mean even when the extreme windows sit at the ends
    from p2du.reporting.inference import boot_means
    from p2du.evaluation.metrics import moving_block_bootstrap_mean
    x = np.r_[5.0, np.zeros(14), 5.0]
    for b in (2, 3, 5, 6):
        m = boot_means(x, b, n_boot=20000, seed=1)
        assert abs(m.mean() - x.mean()) < 0.02, (b, m.mean(), x.mean())
        _, _, _, m2 = moving_block_bootstrap_mean(x, 20000, b, 1)
        assert abs(m2.mean() - x.mean()) < 0.02, (b, m2.mean(), x.mean())



if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK", name)
