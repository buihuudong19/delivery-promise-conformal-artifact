"""Phase-1 checks of the outline (data exploration before the main analyses).

  (a) meaning of the estimated delivery date: day resolution; whether the promise depends on information that
      arrives only after the purchase (payment-approval delay, carrier hand-over delay) once the route and the
      month are held fixed. A promise recomputed later would track these delays; a promise fixed at purchase cannot.
  (b) censoring at the origins: share of the calibration orders still undelivered at each origin
  (c) survey timing: when the satisfaction survey is created relative to the delivery and to the promised date
  (d) distributions of the delivery time Y, the promise P and the deviation S by region, and their monthly path
"""
import numpy as np
import pandas as pd

from ..evaluation.origins import rolling_windows, origin_masks

DAY = pd.Timedelta(days=1)


def _within(x, keys):
    return x - x.groupby(keys).transform("mean")


def promise_checks(D):
    X = D[D["approved_ts"].notna()].copy()
    X["approval_h"] = (X["approved_ts"] - X["purchase_ts"]) / pd.Timedelta(hours=1)
    X["handover_d"] = (X["t0"] - X["purchase_ts"]) / DAY
    X["month"] = X["purchase_ts"].dt.to_period("M").astype(str)
    keys = [X["route"], X["month"]]
    pw = _within(X["P_olist"].astype(float), keys)
    rows = []
    for c in ("approval_h", "handover_d", "Y"):
        m = X[c].notna()
        rows.append(dict(check=f"within route-month correlation of P with {c}",
                         value=float(np.corrcoef(pw[m], _within(X.loc[m, c], [X.loc[m, "route"], X.loc[m, "month"]]))[0, 1]),
                         n=int(m.sum())))
    # the same within seller x route x month cells: a seller's declared handling time is known at purchase, so a
    # correlation that survives inside these cells would point to a promise updated after the purchase
    k3 = [X["seller_id"], X["route"], X["month"]]
    pw3 = _within(X["P_olist"].astype(float), k3)
    for c in ("approval_h", "handover_d"):
        m = X[c].notna()
        rows.append(dict(check=f"within seller-route-month correlation of P with {c}",
                         value=float(np.corrcoef(pw3[m], _within(X.loc[m, c], [X.loc[m, "seller_id"], X.loc[m, "route"],
                                                                              X.loc[m, "month"]]))[0, 1]), n=int(m.sum())))
    rows.append(dict(check="estimated date stored at midnight (share)",
                     value=float((D["estimated_ts"] == D["estimated_ts"].dt.normalize()).mean()), n=len(D)))
    rows.append(dict(check="promise P in days: mean", value=float(D["P_olist"].mean()), n=len(D)))
    rows.append(dict(check="promise P in days: share of orders with P <= 7", value=float((D["P_olist"] <= 7).mean()), n=len(D)))
    dl = D[D["delivered"]]
    rows.append(dict(check="late rate (S > 0) among delivered orders", value=float((dl["S_olist"] > 0).mean()), n=len(dl)))
    rows.append(dict(check="Olist promise as a quantile of the delivery time: share of D <= P (= 1 - late rate)",
                     value=float((dl["S_olist"] <= 0).mean()), n=len(dl)))
    rows.append(dict(check="median slack P - D among on-time orders (days)",
                     value=float((-dl.loc[dl["S_olist"] <= 0, "S_olist"]).median()), n=int((dl["S_olist"] <= 0).sum())))
    return pd.DataFrame(rows)


def censoring_by_origin(D, cfg):
    rows = []
    for w in rolling_windows(cfg):
        M = origin_masks(D, w, cfg)
        C = D[M["cal"]]
        und = ~(C["delivered_ts"] < w["origin"])
        age = ((w["origin"] - C["purchase_day"]) / DAY)
        rows.append(dict(k=w["k"], origin=w["origin"], n_cal=len(C), share_undelivered=float(und.mean()),
                         share_undelivered_last7d=float(und[age <= 7].mean()),
                         share_undelivered_older21d=float(und[age > 21].mean()),
                         n_eval=int(M["eval"].sum()), n_train=int(M["train"].sum())))
    return pd.DataFrame(rows)


def censoring_by_age(D, cfg):
    """Share of calibration orders not yet delivered at the origin, by purchase age (days before the origin)."""
    rows = []
    for w in rolling_windows(cfg):
        M = origin_masks(D, w, cfg)
        C = D[M["cal"]]
        age = ((w["origin"] - C["purchase_day"]) / DAY).astype(int)
        und = (~(C["delivered_ts"] < w["origin"])).astype(float)
        g = und.groupby(age).agg(["mean", "size"])
        for a, r in g.iterrows():
            rows.append(dict(k=w["k"], age=int(a), share_undelivered=float(r["mean"]), n=int(r["size"])))
    return pd.DataFrame(rows)


def survey_timing(D):
    X = D[D["delivered"] & D["review_created_ts"].notna()].copy()
    X["created_vs_delivery_d"] = (X["review_created_ts"] - X["delivered_ts"].dt.normalize()) / DAY
    X["created_vs_promise_d"] = (X["review_created_ts"] - X["estimated_ts"]) / DAY
    X["late"] = X["S_olist"] > 0
    X["created_before_delivery"] = X["review_created_ts"] < X["delivered_ts"].dt.normalize()
    X["answered_before_delivery"] = X["first_answer_ts"] < X["delivered_ts"]
    g = X.groupby("late").agg(n=("order_id", "size"), created_before_delivery=("created_before_delivery", "mean"),
                              answered_before_delivery=("answered_before_delivery", "mean"),
                              created_vs_delivery_median=("created_vs_delivery_d", "median"),
                              created_vs_promise_median=("created_vs_promise_d", "median")).reset_index()
    return g


def region_profile(D):
    dl = D[D["delivered"]]
    g = dl.groupby("customer_region").agg(n=("order_id", "size"), Y_median=("Y", "median"),
                                          Y_p90=("Y", lambda x: x.quantile(0.9)), Y_p95=("Y", lambda x: x.quantile(0.95)),
                                          P_mean=("P_olist", "mean"), late=("late_olist", "mean"),
                                          S_median=("S_olist", "median"),
                                          low=("low", "mean")).reset_index()
    return g


def monthly(D):
    X = D.assign(month=D["purchase_ts"].dt.to_period("M").dt.to_timestamp())
    dl = X[X["delivered"]]
    g = dl.groupby("month").agg(n=("order_id", "size"), Y_mean=("Y", "mean"), Y_p90=("Y", lambda x: x.quantile(0.9)),
                                P_mean=("P_olist", "mean"), late=("late_olist", "mean"), low=("low", "mean")).reset_index()
    return g


def run_eda(D, cfg, out):
    t = out / "tables"
    promise_checks(D).to_csv(t / "eda_promise_checks.csv", index=False)
    censoring_by_origin(D, cfg).to_csv(t / "eda_censoring_by_origin.csv", index=False)
    censoring_by_age(D, cfg).to_csv(t / "eda_censoring_by_age.csv", index=False)
    survey_timing(D).to_csv(t / "eda_survey_timing.csv", index=False)
    region_profile(D).to_csv(t / "eda_region_profile.csv", index=False)
    monthly(D).to_csv(t / "eda_monthly.csv", index=False)
    dl = D[D["delivered"]]
    S = dl.assign(Sc=dl["S_olist"].clip(-40, 20)).groupby(["customer_region", "Sc"]).size().rename("n").reset_index()
    S.to_csv(t / "eda_S_by_region.csv", index=False)
