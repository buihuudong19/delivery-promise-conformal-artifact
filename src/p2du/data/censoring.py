"""Right censoring at an origin and inverse-probability-of-censoring weights (IPCW).

At an origin o, an order bought before o is observed if it was delivered before o; otherwise its delivery time is only
known to exceed the follow-up c_i = o - purchase day. Censoring here is administrative: c_i depends on the purchase
time alone and is known for every order. Under the assumption that the follow-up is independent of the features and
the delivery time within the set (C independent of (X, Y), Assumption 1 of the article), the probability that order
i is observed is G(Y_i) = Pr(C > Y_i), where C is the follow-up of an order drawn from the same set. G is estimated by the share of the set whose follow-up
exceeds t. Weighting each observed order by 1 / G(Y_i) gives an unbiased estimate of any risk over the whole set
(Proposition 2 of the article); the weights are capped (config: censoring.weight_cap).
"""
import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)


def status_at(D, origin):
    """Observed flag, follow-up c (days from the purchase day to the origin) and the observed delivery time."""
    obs = (D["delivered_ts"] < origin).to_numpy()
    c = ((origin - D["purchase_day"]) / DAY).to_numpy()
    y = np.where(obs, D["Y"].to_numpy(), np.nan)
    return obs, c, y


def censoring_survival(c_all):
    """t -> share of the set with follow-up strictly greater than t."""
    cs = np.sort(np.asarray(c_all, float))
    n = len(cs)
    return lambda t: (n - np.searchsorted(cs, np.asarray(t, float), side="right")) / max(n, 1)


def ipcw_weights(obs, c, y, cap=20.0):
    """Weights 1 / G(Y_i) for observed orders (0 for censored orders) and the estimated G at each observed Y."""
    G = censoring_survival(c)
    w = np.zeros(len(obs))
    g = np.full(len(obs), np.nan)
    g[obs] = G(y[obs])
    w[obs] = 1.0 / np.maximum(g[obs], 1.0 / cap)
    return w, g


def tail_share(D, origin, horizon_days, lookback_days=28, groups=None, mode="observed"):
    """Share of delivered orders with a delivery time of at least `horizon_days`, measured on the orders bought in
    [origin - horizon - lookback, origin - horizon) and delivered by the origin. For these orders the event
    {Y >= horizon} is known at the origin (apart from the few that are still on their way after 42 to 70 days, most of
    which are never delivered). The calibration set of an origin cannot show such orders (their follow-up is shorter
    than the horizon); this share is the mass placed beyond the largest observable score. Orders that are never
    delivered are not part of the coverage target (they are counted as missed promises in a sensitivity analysis), so
    they do not enter the share.
    mode = "observed" (main): the share among the cohort orders delivered by the origin. Cohort orders still on their
    way at the origin are left out although they have Y >= horizon, so this share is a lower bound on the tail share
    of the delivered population (Section 4.3 of the article; mean 1.07% against 1.62% in the final snapshot).
    mode = "upper" (sensitivity): orders not delivered by the origin are counted as tail. Because the never-delivered
    orders also enter, this is an upper bound on the tail share of the delivered population, which makes the
    correction conservative (Lemma 1 and Proposition 3 of the article).
    Returns the overall share and, if groups is a column name, a dict of shares by group."""
    H = pd.Timedelta(days=horizon_days)
    m = (D["purchase_ts"] >= origin - H - pd.Timedelta(days=lookback_days)) & (D["purchase_ts"] < origin - H)
    done = (D["delivered_ts"] < origin)
    if mode == "observed":
        m = m & done
    elif mode != "upper":
        raise ValueError(mode)
    X = D[m]
    tail = ((X["Y"] >= horizon_days) | ~(X["delivered_ts"] < origin)).astype(float)
    overall = float(tail.mean()) if len(X) else 0.0
    if groups is None:
        return overall, {}
    return overall, tail.groupby(X[groups].fillna("NA")).mean().to_dict()
