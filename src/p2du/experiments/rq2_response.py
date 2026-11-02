"""RQ2: how the probability of a low rating responds to the deviation from the promise, S = D - P.

Population: delivered orders with a review; low = 1-2 stars, 4-5 stars = 0, 3 stars left out (as in the first
article; included as 0 in the sensitivity analysis).

  (1) crude curve      low-rating rate by integer S with Wilson intervals; the same split by whether the review was
                       answered before or after the parcel arrived (the survey is sent at delivery or on the promised
                       date, so a late order can be rated before it arrives)
  (2) S versus D       cross-validated log loss of logistic models in natural splines of S, of D, and of either plus
                       the promise P: if customers react to the deviation, f(S) should beat f(D) and P should add
                       little to f(S)
  (3) AIPW             doubly robust risk of each deviation bin (multi-valued treatment; Robins et al., 1994), with
                       5-fold cross-fitting, logistic and XGBoost nuisance models, influence-function intervals,
                       risk differences and risk ratios against the reference bin, and E-values (VanderWeele and
                       Ding, 2017) for the risk ratios
  (4) g-formula curve  r(s) = mean_i mu(s, X_i) from an XGBoost outcome model with a monotone (non-decreasing)
                       constraint in s; this curve feeds the promise policies of RQ3 (overall, by region, and fitted
                       on 2017 orders only for the out-of-time check)
  (5) RDD              jump of the low-rating rate between S = 0 (delivered on the promised day) and S = 1, local
                       linear on each side with a triangular kernel over the cells of the discrete running variable.
                       Inference: honest confidence intervals for the Taylor (Hoelder) class with a bound M on the
                       second derivative, worst-case bias (M / 2) sum_i |w_i| x_i^2 of the linear estimator and the
                       folded-normal critical value (Armstrong and Kolesar, 2020). With a discrete running variable
                       the bias of the extrapolation to the cut-off is not identified without such a bound, which is
                       the point of Kolesar and Rothe (2018); this implementation is the bounded-second-derivative
                       version, not their bounded-misspecification (BME) intervals. M: rule of thumb from quadratic
                       fits on each side, and a grid.
Confounders X (all known at purchase): the purchase-time features of RQ1, the promise P, and the as-of low-rating
rates of the seller and the category (reviews answered before the purchase).
"""
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# A category of the evaluation or calibration orders that is absent from the training orders (a new seller state or
# route) is mapped to the infrequent category by design (handle_unknown="infrequent_if_exist"); scikit-learn warns at
# every such transform. The warning carries no information here and floods the log, so it is silenced (no effect on
# any number).
warnings.filterwarnings("ignore", message="Found unknown categories in columns", category=UserWarning)

from ..data.order_attributes import _asof_cum
from ..data.purchase_features import feature_set

DAY = pd.Timedelta(days=1)


def _slog(x):
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))


def wilson(k, n, z=1.96):
    p = k / np.maximum(n, 1)
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def e_value(rr):
    rr = np.asarray(rr, float)
    r = np.where(rr < 1, 1 / rr, rr)
    return r + np.sqrt(r * (r - 1))


def rq2_frame(D, cfg, neutral_as_zero=False):
    """Delivered, reviewed orders with the outcome, S, the bins and the as-of low-rating histories."""
    R = D[D["delivered"] & D["review_score"].notna()].copy()
    R["_qid"] = np.arange(len(R))
    ev = D[D["review_score"].notna()][["seller_id", "category", "last_answer_ts", "review_score"]].copy()
    ev["lowe"] = (ev["review_score"] <= 2).astype(float)
    m = cfg["features"]["smoothing_m"]
    for key, name in (("seller_id", "seller"), ("category", "cat")):
        n, s = _asof_cum(R, ev, key, "purchase_ts", "last_answer_ts", "lowe")
        R[f"h_{name}_lowrate"] = (s + m * 0.13) / (n + m)
    if neutral_as_zero:
        R["low"] = (R["review_score"] <= 2).astype(float)
    R = R[R["low"].notna()].copy()
    R["S"] = R["S_olist"].astype(int)
    R["P"] = R["P_olist"].astype(int)
    R["Dd"] = R["D"].astype(int)
    edges = cfg["rq2"]["bins"]
    R["bin"] = pd.cut(R["S"], edges, labels=False, right=True).astype(int)
    R["answered_before_delivery"] = (R["first_answer_ts"] < R["delivered_ts"]).astype(int)
    R["region"] = R["customer_region"]
    return R.drop(columns=["_qid"]).reset_index(drop=True)


def bin_labels(cfg):
    e = cfg["rq2"]["bins"]
    lab = []
    for a, b in zip(e[:-1], e[1:]):
        lo, hi = a + 1, b
        if a <= -100:
            lab.append(f"S <= {hi}")
        elif b >= 100:
            lab.append(f"S >= {lo}")
        else:
            lab.append(f"{lo} to {hi}")
    return lab


def confounders(cfg):
    num, cat = feature_set(cfg, promise=False)
    return num + ["P", "h_seller_lowrate", "h_cat_lowrate"], cat


# ----------------------------------------------------------------------------------------------- (1) crude curve
def crude_curve(R, lo=-40, hi=30, by=None):
    X = R.assign(Sc=R["S"].clip(lo, hi))
    keys = ["Sc"] + ([by] if by else [])
    g = X.groupby(keys)["low"].agg(["sum", "count"]).reset_index()
    g["rate"] = g["sum"] / g["count"]
    g["lo"], g["hi"] = wilson(g["sum"].to_numpy(), g["count"].to_numpy())
    return g.rename(columns={"Sc": "S", "count": "n", "sum": "n_low"})


# ----------------------------------------------------------------------------------------------- (2) S versus D
def _ns(x, knots):
    """Natural cubic spline basis (truncated power form) with the given knots."""
    x = np.asarray(x, float); k = np.asarray(knots, float); K = len(k)
    def d(j):
        return (np.maximum(x - k[j], 0) ** 3 - np.maximum(x - k[-1], 0) ** 3) / (k[-1] - k[j])
    cols = [x] + [d(j) - d(K - 2) for j in range(K - 2)]
    return np.column_stack(cols)


def s_versus_d(R, cfg, n_knots=6):
    y = R["low"].to_numpy()
    S, Dd, P = R["S"].clip(-40, 40).to_numpy(), R["Dd"].clip(0, 80).to_numpy(), R["P"].clip(0, 70).to_numpy()
    def kn(x):
        return np.quantile(x, np.linspace(0.05, 0.95, n_knots))
    specs = {"f(D)": [_ns(Dd, kn(Dd))], "f(S)": [_ns(S, kn(S))], "f(D) + g(P)": [_ns(Dd, kn(Dd)), _ns(P, kn(P))],
             "f(S) + g(P)": [_ns(S, kn(S)), _ns(P, kn(P))], "g(P)": [_ns(P, kn(P))]}
    skf = StratifiedKFold(5, shuffle=True, random_state=cfg["seed"])
    rows = []
    for name, blocks in specs.items():
        X = StandardScaler().fit_transform(np.column_stack(blocks))
        ll = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            m = LogisticRegression(C=1e4, max_iter=5000).fit(X[tr], y[tr])
            p = np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
            ll[te] = -(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
        mfull = LogisticRegression(C=1e4, max_iter=5000).fit(X, y)
        p = np.clip(mfull.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        llf = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        k = X.shape[1] + 1
        rows.append(dict(model=name, cv_logloss=float(ll.mean()), cv_logloss_se=float(ll.std() / np.sqrt(len(ll))),
                         aic=2 * k - 2 * llf, n_params=k, n=len(y)))
    base = np.mean(-(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean())))
    out = pd.DataFrame(rows)
    out["mcfadden_r2_cv"] = 1 - out["cv_logloss"] / base
    return out


def surface(R, d_edges=(0, 5, 10, 15, 20, 25, 30, 40, 60, 300), p_edges=(0, 10, 15, 20, 25, 30, 35, 40, 200)):
    X = R.assign(db=pd.cut(R["Dd"], d_edges, right=False), pb=pd.cut(R["P"], p_edges, right=False))
    g = X.groupby(["pb", "db"], observed=False)["low"].agg(["mean", "count"]).reset_index()
    g["pb"] = g["pb"].astype(str); g["db"] = g["db"].astype(str)
    return g.rename(columns={"mean": "rate", "count": "n"})


# ----------------------------------------------------------------------------------------------- (3) AIPW
def _lr_pre(num, cat):
    numt = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                     ("log", FunctionTransformer(_slog)), ("sc", StandardScaler())])
    return ColumnTransformer([("n", numt, num),
                              ("c", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100), cat)])


def _xgb_X(df, num, cat, vocab):
    X = df[num].astype(float).copy()
    for c in cat:
        X[c] = pd.Categorical(df[c], categories=vocab[c])
    return X


def aipw_nuisances(R, vocab, cfg, learner="lr", log=print):
    """Cross-fitted outcome model mu_b(X) and generalised propensity pi_b(X) for every deviation bin b."""
    num, cat = confounders(cfg)
    B = R["bin"].to_numpy(); y = R["low"].to_numpy(); nb = int(B.max()) + 1
    mu = np.zeros((len(R), nb)); pi = np.zeros((len(R), nb))
    skf = StratifiedKFold(cfg["rq2"]["folds"], shuffle=True, random_state=cfg["seed"])
    for f, (tr, te) in enumerate(skf.split(np.zeros(len(R)), B * 2 + y)):
        A, T = R.iloc[tr], R.iloc[te]
        if learner == "lr":
            pre = _lr_pre(num, cat).fit(A[num + cat])
            XA, XT = pre.transform(A[num + cat]), pre.transform(T[num + cat])
            ps = LogisticRegression(C=0.5, max_iter=3000).fit(XA, B[tr])
            pi[te] = ps.predict_proba(XT)
            oh = np.eye(nb)
            om = LogisticRegression(C=0.5, max_iter=3000).fit(_hstack(XA, oh[B[tr]]), y[tr])
            for b in range(nb):
                mu[te, b] = om.predict_proba(_hstack(XT, np.tile(oh[b], (len(te), 1))))[:, 1]
        else:
            XA, XT = _xgb_X(A, num, cat, vocab), _xgb_X(T, num, cat, vocab)
            prm = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                       min_child_weight=20, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
                       random_state=cfg["seed"], n_jobs=cfg.get("n_jobs", 2))
            pi[te] = xgb.XGBClassifier(objective="multi:softprob", **prm).fit(XA, B[tr]).predict_proba(XT)
            XA2 = XA.assign(_bin=B[tr]); om = xgb.XGBClassifier(objective="binary:logistic", **prm).fit(XA2, y[tr])
            for b in range(nb):
                mu[te, b] = om.predict_proba(XT.assign(_bin=b))[:, 1]
        log(f"    aipw {learner} fold {f + 1}")
    return dict(mu=mu, pi=pi, B=B, y=y, learner=learner)


def aipw_estimate(nu, cfg, clip=0.01, keep=None):
    """AIPW risks by bin from stored nuisances; propensities clipped at `clip` (and renormalised); `keep` restricts
    the sample (trimming)."""
    mu, pi, B, y = nu["mu"], nu["pi"], nu["B"], nu["y"]
    if keep is not None:
        mu, pi, B, y = mu[keep], pi[keep], B[keep], y[keep]
    nb = mu.shape[1]
    raw_own = pi[np.arange(len(B)), B]
    pi = np.clip(pi, clip, 1.0); pi = pi / pi.sum(1, keepdims=True)
    phi = mu + (np.eye(nb)[B] * (y[:, None] - mu)) / pi
    psi = phi.mean(0); n = len(y)
    ref = cfg["rq2"]["ref_bin"]
    rows = []
    labels = bin_labels(cfg)
    for b in range(nb):
        se = phi[:, b].std(ddof=1) / np.sqrt(n)
        d = phi[:, b] - phi[:, ref]; se_d = d.std(ddof=1) / np.sqrt(n)
        # log risk ratio, delta method on the influence functions
        ifl = phi[:, b] / psi[b] - phi[:, ref] / psi[ref]; se_l = ifl.std(ddof=1) / np.sqrt(n)
        rr = psi[b] / psi[ref]
        rr_lo, rr_hi = rr * np.exp(-1.96 * se_l), rr * np.exp(1.96 * se_l)
        bound = rr_lo if rr >= 1 else rr_hi
        rows.append(dict(learner=nu["learner"], clip=clip, n_used=int(n), bin=b, label=labels[b], n=int((B == b).sum()),
                         crude=float(y[B == b].mean()), risk=float(psi[b]), risk_lo=psi[b] - 1.96 * se,
                         risk_hi=psi[b] + 1.96 * se, rd=float(psi[b] - psi[ref]), rd_lo=psi[b] - psi[ref] - 1.96 * se_d,
                         rd_hi=psi[b] - psi[ref] + 1.96 * se_d, rr=float(rr), rr_lo=float(rr_lo), rr_hi=float(rr_hi),
                         e_value=float(e_value(rr)) if b != ref else np.nan,
                         e_value_ci=float(e_value(bound)) if (b != ref and ((rr >= 1 and rr_lo > 1) or (rr < 1 and rr_hi < 1))) else 1.0,
                         p_value=float(2 * stats.norm.sf(abs(psi[b] - psi[ref]) / se_d)) if b != ref else np.nan,
                         ps_min=float(nu["pi"][:, b].min()), ps_p01=float(np.quantile(nu["pi"][:, b], 0.01)),
                         share_own_ps_below_clip=float((raw_own[B == b] < clip).mean())))
    return pd.DataFrame(rows)


def aipw_bins(R, vocab, cfg, learner="lr", log=print, clip=0.01):
    return aipw_estimate(aipw_nuisances(R, vocab, cfg, learner, log), cfg, clip)


def overlap_sensitivity(nus, cfg):
    """AIPW under other propensity clips and with trimming of the orders whose propensity for their own bin is
    below 0.01 or 0.02 (Crump et al., 2009, adapted to a multi-valued exposure; the estimand becomes the trimmed
    population)."""
    out = []
    for nu in nus:
        for c in (0.005, 0.01, 0.02, 0.05):
            out.append(aipw_estimate(nu, cfg, clip=c).assign(analysis=f"clip {c}"))
        own = nu["pi"][np.arange(len(nu["B"])), nu["B"]]
        for thr in (0.01, 0.02):
            out.append(aipw_estimate(nu, cfg, clip=0.01, keep=own >= thr).assign(analysis=f"trim own propensity < {thr}"))
    return pd.concat(out, ignore_index=True)


def _hstack(a, b):
    import scipy.sparse as sp
    return sp.hstack([a, sp.csr_matrix(b)]).tocsr() if sp.issparse(a) else np.hstack([a, b])


# ----------------------------------------------------------------------------------------------- (4) g-formula curve
def gformula_curve(R, vocab, cfg, fit_mask=None, pop_mask=None, groups=None):
    num, cat = confounders(cfg)
    lo, hi = cfg["rq2"]["s_range"]
    F = R if fit_mask is None else R[fit_mask]
    X = _xgb_X(F, num, cat, vocab).assign(_S=F["S"].clip(lo - 10, hi + 10).to_numpy())
    cons = tuple([0] * (X.shape[1] - 1) + [1])
    m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=20, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
                          monotone_constraints=cons, random_state=cfg["seed"], n_jobs=cfg.get("n_jobs", 2)).fit(X, F["low"].to_numpy())
    Pop = R if pop_mask is None else R[pop_mask]
    XP = _xgb_X(Pop, num, cat, vocab)
    grid = np.arange(lo, hi + 1)
    rows = []
    gser = Pop[groups].to_numpy() if groups else None
    for s in grid:
        p = m.predict_proba(XP.assign(_S=s))[:, 1]
        rows.append(dict(S=int(s), group="all", r=float(p.mean())))
        if groups:
            for g in pd.unique(gser):
                rows.append(dict(S=int(s), group=g, r=float(p[gser == g].mean())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- (5) RDD
def _ll_weights(x, n, h, side):
    """Weights (one per cell) of the local linear intercept at x = 0 with a triangular kernel."""
    k = np.maximum(1 - np.abs(x) / (h + 0.5), 0) * n
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(k)
    A = np.linalg.inv(X.T @ W @ X) @ X.T @ W
    return A[0]


def rdd_discrete(R, cfg, cutoff=0, M_grid=None, subset=None):
    """Jump between S = cutoff and S = cutoff + 1; x = S - cutoff - 0.5 so that the jump is evaluated at x = 0."""
    X = R if subset is None else R[subset]
    cells = X.groupby("S")["low"].agg(["mean", "count"])
    rows = []
    # rule-of-thumb M: largest |second derivative| of global quadratic fits on each side (within 15 days)
    def rot():
        out = []
        for side in (-1, 1):
            c = cells.loc[(cells.index <= cutoff) & (cells.index >= cutoff - 14)] if side < 0 else \
                cells.loc[(cells.index > cutoff) & (cells.index <= cutoff + 15)]
            if len(c) < 4:
                continue
            co = np.polyfit(c.index.to_numpy() - cutoff - 0.5, c["mean"].to_numpy(), 2, w=np.sqrt(c["count"].to_numpy()))
            out.append(abs(2 * co[0]))
        return max(out) if out else 0.01
    M0 = rot()
    Ms = M_grid or [M0, 2 * M0, 0.005, 0.01]
    for h in cfg["rq2"]["rdd_bandwidths"]:
        L = cells.loc[(cells.index <= cutoff) & (cells.index > cutoff - h)]
        Rt = cells.loc[(cells.index > cutoff) & (cells.index <= cutoff + h)]
        if len(L) < 2 or len(Rt) < 2:
            continue
        xl, xr = L.index.to_numpy() - cutoff - 0.5, Rt.index.to_numpy() - cutoff - 0.5
        wl, wr = _ll_weights(xl, L["count"].to_numpy(), h, -1), _ll_weights(xr, Rt["count"].to_numpy(), h, 1)
        est = float(wr @ Rt["mean"].to_numpy() - wl @ L["mean"].to_numpy())
        var = float(np.sum(wr ** 2 * Rt["mean"] * (1 - Rt["mean"]) / Rt["count"]) +
                    np.sum(wl ** 2 * L["mean"] * (1 - L["mean"]) / L["count"]))
        se = np.sqrt(var)
        for M in Ms:
            bias = M / 2 * (np.sum(np.abs(wr) * xr ** 2) + np.sum(np.abs(wl) * xl ** 2))
            cv = np.sqrt(stats.ncx2.ppf(0.95, 1, (bias / se) ** 2)) if se > 0 else np.nan
            rows.append(dict(cutoff=cutoff, h=h, M=float(M), M_rule_of_thumb=float(M0), estimate=est, se=float(se),
                             max_bias=float(bias), ci_lo=est - cv * se, ci_hi=est + cv * se,
                             ci_lo_naive=est - 1.96 * se, ci_hi_naive=est + 1.96 * se,
                             n_left=int(L["count"].sum()), n_right=int(Rt["count"].sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- stage
def lee_bounds(R, left=(0,), right=(1,), n_boot=500, seed=0):
    """Lee (2009) bounds on the jump in the low-rating rate between deviation cells `left` and `right`, among the
    customers who would answer after arrival whatever the cell ("always after-arrival responders"). Lateness can only
    lower the probability of answering after arrival (monotone selection), so the after-arrival group of the left cell
    is trimmed by the share p = (s_left - s_right) / s_left, from the top (lower bound of its mean) or the bottom (upper
    bound). With a binary outcome, trimming removes ones or zeros. Percentile bootstrap over orders for the interval.
    Returns a dict with the shares, the raw contrast, the bounds and the bootstrap interval of the bounds."""
    after = (R["answered_before_delivery"] == 0).to_numpy()
    S = R["S"].to_numpy(); y = R["low"].to_numpy().astype(float)
    L = np.isin(S, left); Rr = np.isin(S, right)

    def one(idx):
        l, r = idx[L[idx]], idx[Rr[idx]]
        sl, sr = after[l].mean(), after[r].mean()
        yl, yr = y[l][after[l]], y[r][after[r]]
        p = max((sl - sr) / sl, 0.0)
        m = yl.mean()
        lo_l, hi_l = max(m - p, 0.0) / (1 - p), min(m / (1 - p), 1.0)      # trim ones / trim zeros
        return sl, sr, yr.mean() - m, yr.mean() - hi_l, yr.mean() - lo_l, p

    base = one(np.arange(len(R)))
    rng = np.random.default_rng(seed)
    bs = np.array([one(rng.integers(0, len(R), len(R))) for _ in range(n_boot)])
    return dict(left=str(left), right=str(right), share_after_left=base[0], share_after_right=base[1],
                trim_share=base[5], raw_jump=base[2], bound_lo=base[3], bound_hi=base[4],
                ci_lo=float(np.quantile(bs[:, 3], 0.025)), ci_hi=float(np.quantile(bs[:, 4], 0.975)))


def run_rq2(D, vocab, cfg, out, log=print):
    t = out / "tables"
    R = rq2_frame(D, cfg)
    log(f"  rq2: {len(R)} orders, low-rating rate {R['low'].mean():.4f}")
    crude_curve(R).to_csv(t / "rq2_crude_curve.csv", index=False)
    crude_curve(R, by="answered_before_delivery").to_csv(t / "rq2_crude_curve_by_answer_timing.csv", index=False)
    crude_curve(R, by="region").to_csv(t / "rq2_crude_curve_by_region.csv", index=False)
    late = R[R["S"] > 0]
    mech = R.assign(late=(R["S"] > 0).astype(int)).groupby(["late", "answered_before_delivery"])["low"].agg(
        ["mean", "count"]).reset_index()
    mech["share_within_late_group"] = mech["count"] / mech.groupby("late")["count"].transform("sum")
    mech.to_csv(t / "rq2_survey_mechanism.csv", index=False)
    log(f"  rq2: late orders {len(late)}, answered before arrival {late['answered_before_delivery'].mean():.3f}")
    s_versus_d(R, cfg).to_csv(t / "rq2_s_versus_d.csv", index=False)
    surface(R).to_csv(t / "rq2_surface_P_D.csv", index=False)
    nus = [aipw_nuisances(R, vocab, cfg, "lr", log), aipw_nuisances(R, vocab, cfg, "xgb", log)]
    A = pd.concat([aipw_estimate(nu, cfg, clip=0.01) for nu in nus], ignore_index=True)
    A.to_csv(t / "rq2_aipw_bins.csv", index=False)
    overlap_sensitivity(nus, cfg).to_csv(t / "rq2_aipw_overlap_sensitivity.csv", index=False)
    reg = []
    for g, Rg in R.groupby("region"):
        a = aipw_bins(Rg.reset_index(drop=True), vocab, cfg, "lr", log=lambda *_: None)
        reg.append(a.assign(region=g))
    pd.concat(reg, ignore_index=True).to_csv(t / "rq2_aipw_bins_by_region.csv", index=False)
    G = gformula_curve(R, vocab, cfg, groups="region")
    G.to_csv(t / "rq2_response_curve.csv", index=False)
    pre = (R["purchase_ts"] < pd.Timestamp("2018-01-01")).to_numpy()
    gformula_curve(R, vocab, cfg, fit_mask=pre).to_csv(t / "rq2_response_curve_fit2017.csv", index=False)
    R3 = rq2_frame(D, cfg, neutral_as_zero=True)
    gformula_curve(R3, vocab, cfg).to_csv(t / "rq2_response_curve_neutral0.csv", index=False)
    rd = [rdd_discrete(R, cfg, 0)]
    rd.append(rdd_discrete(R, cfg, 0, subset=(R["answered_before_delivery"] == 0)).assign(subset="answered_after_arrival"))
    for c in (-6, -3, 3, 6):                                        # placebo cut-offs
        rd.append(rdd_discrete(R, cfg, c).assign(subset="placebo"))
    RD = pd.concat(rd, ignore_index=True); RD["subset"] = RD["subset"].fillna("all")
    RD.to_csv(t / "rq2_rdd.csv", index=False)
    log("  rq2: done")
