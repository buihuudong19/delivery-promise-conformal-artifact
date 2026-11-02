"""Features admissible at the purchase time tau_i (the moment the delivery date is shown to the customer).

Admissibility follows the definition of the first article: a feature of order i may use an event of another order
only if that event happened strictly before tau_i. The events and the times at which they become known are
  delivery event      delivered timestamp            value Y (delivery time from the purchase day), log Y
  late event          min(delivery, promised day + 1) value 1{late against Olist's promise}
  hand-over event     carrier hand-over timestamp     value processing days (hand-over - purchase)
  purchase event      purchase timestamp             (counts: seller volume, customer history, platform load)

Feature groups
  BASE   order attributes known at purchase (basket, price, freight, weight, volume, payment, distance, calendar)
  HIST   as-of histories of the seller, route (seller state x customer state), customer state and category, and the
         recent platform-wide delivery time and order load (drift signals)
  PROMISE Olist's own estimate (only in the sensitivity analysis: it is known at purchase, but using it would turn
         the models into post-processors of the platform's estimate)
"""
import numpy as np
import pandas as pd

from .order_attributes import _asof_cum, _asof_global

DAY = pd.Timedelta(days=1)

BASE_NUM = ["f_purchase_dow", "f_purchase_hour", "f_purchase_month", "f_freight_ratio", "f_same_state", "f_n_items",
            "f_n_products", "f_n_sellers", "f_price_sum", "f_price_max", "f_freight_sum", "f_weight_g", "f_volume_cm3",
            "f_photos_qty", "f_pay_value", "f_n_payments", "f_installments", "f_distance_km"]
BASE_CAT = ["c_category", "c_customer_state", "c_seller_state", "c_pay_type"]
HIST = ["h_seller_deliv_n", "h_seller_deliv_mean", "h_seller_late_n", "h_seller_laterate", "h_seller_proc_n",
        "h_seller_proc_mean", "h_seller_vol30", "h_route_n", "h_route_deliv_mean", "h_route_deliv_sd",
        "h_route_laterate", "h_cstate_deliv_mean", "h_cat_deliv_mean", "h_platform_deliv_mean30",
        "h_platform_vol14", "h_customer_prior_orders"]
PROMISE = ["f_olist_promise"]
# variables that are known only after the purchase; none of them may be an input (checked by the tests)
OUTCOME_COLS = ["Y", "D", "S_olist", "late_olist", "delivered_ts", "t0", "approved_ts", "review_score", "low",
                "review_created_ts", "first_answer_ts", "last_answer_ts", "delivered"]


def feature_set(cfg=None, promise=None):
    use_p = cfg["features"].get("include_platform_promise", False) if promise is None else promise
    return BASE_NUM + HIST + (PROMISE if use_p else []), BASE_CAT


def _asof_mean(S, ev, key, etime, value, prior, m):
    n, s = _asof_cum(S, ev, key, "purchase_ts", etime, value)
    return n, (s + m * prior) / (n + m)


def build_purchase_features(A, cfg):
    m = cfg["features"]["smoothing_m"]
    S = A.copy()
    S["_qid"] = np.arange(len(S))
    S["route"] = S["seller_state"].astype(str) + ">" + S["customer_state"].astype(str)

    # ---------------- BASE
    S["f_purchase_dow"] = S["purchase_ts"].dt.dayofweek
    S["f_purchase_hour"] = S["h_frac"] * 24
    S["f_purchase_month"] = S["purchase_ts"].dt.month
    S["f_freight_ratio"] = S["freight_sum"] / S["price_sum"].replace(0, np.nan)
    S["f_same_state"] = (S["customer_state"] == S["seller_state"]).astype(int)
    for c in ["n_items", "n_products", "n_sellers", "price_sum", "price_max", "freight_sum", "weight_g", "volume_cm3",
              "photos_qty", "pay_value", "n_payments", "installments", "distance_km"]:
        S["f_" + c] = S[c]
    for c in ["category", "customer_state", "seller_state", "pay_type"]:
        S["c_" + c] = S[c].fillna("unknown").astype(str)
    S["f_olist_promise"] = S["P_olist"].astype(float)

    # ---------------- events
    dev = S.loc[S["delivered"], ["seller_id", "route", "customer_state", "category", "delivered_ts", "Y"]].copy()
    dev["Y2"] = dev["Y"] ** 2
    lev = S[["seller_id", "route", "delivered_ts", "promise_end_ts", "late_olist"]].copy()
    lev["late"] = lev["late_olist"].fillna(1.0)          # an open order is known to be late once its promised day passes
    lev["late_time"] = lev["delivered_ts"].where(lev["delivered_ts"] < lev["promise_end_ts"], lev["promise_end_ts"])
    pev = S.loc[S["t0"].notna(), ["seller_id", "t0", "purchase_ts"]].copy()
    pev["proc"] = (pev["t0"] - pev["purchase_ts"]) / DAY

    # platform-wide as-of mean delivery time (prior of the m-estimates) and its recent 30-day value
    gn, gs = _asof_global(S, dev, "purchase_ts", "delivered_ts", "Y")
    prior = np.where(gn > 0, gs / np.maximum(gn, 1), 12.0)
    W = cfg["features"]["platform_window_days"]
    S2 = S[["_qid"]].copy(); S2["q"] = S["purchase_ts"] - pd.Timedelta(days=W)
    gn_w, gs_w = _asof_global(S2, dev, "q", "delivered_ts", "Y")
    nrec = gn - gn_w
    S["h_platform_deliv_mean30"] = np.where(nrec > 20, (gs - gs_w) / np.maximum(nrec, 1), prior)

    n, S["h_cstate_deliv_mean"] = _asof_mean(S, dev, "customer_state", "delivered_ts", "Y", prior, m)
    n, S["h_cat_deliv_mean"] = _asof_mean(S, dev, "category", "delivered_ts", "Y", prior, m)
    S["h_route_n"], S["h_route_deliv_mean"] = _asof_mean(S, dev, "route", "delivered_ts", "Y",
                                                         S["h_cstate_deliv_mean"].to_numpy(), m)
    n2, s2 = _asof_cum(S, dev, "route", "purchase_ts", "delivered_ts", "Y2")
    n1, s1 = _asof_cum(S, dev, "route", "purchase_ts", "delivered_ts", "Y")
    var = np.where(n1 > 1, (s2 - s1 ** 2 / np.maximum(n1, 1)) / np.maximum(n1 - 1, 1), np.nan)
    S["h_route_deliv_sd"] = np.sqrt(np.clip(var, 0, None))
    S["h_seller_deliv_n"], S["h_seller_deliv_mean"] = _asof_mean(S, dev, "seller_id", "delivered_ts", "Y",
                                                                 S["h_route_deliv_mean"].to_numpy(), m)
    n, s = _asof_cum(S, lev, "seller_id", "purchase_ts", "late_time", "late")
    S["h_seller_late_n"], S["h_seller_laterate"] = n, (s + 5 * 0.08) / (n + 5)
    n, s = _asof_cum(S, lev, "route", "purchase_ts", "late_time", "late")
    S["h_route_laterate"] = (s + m * 0.08) / (n + m)
    n, s = _asof_cum(S, pev, "seller_id", "purchase_ts", "t0", "proc")
    S["h_seller_proc_n"] = n
    S["h_seller_proc_mean"] = np.where(n > 0, s / np.maximum(n, 1), np.nan)

    pe = S[["seller_id", "purchase_ts"]].copy()
    n_now, _ = _asof_cum(S, pe, "seller_id", "purchase_ts", "purchase_ts")
    S3 = S[["_qid", "seller_id"]].copy()
    S3["q30"] = S["purchase_ts"] - pd.Timedelta(days=cfg["features"]["seller_volume_window_days"])
    n_30, _ = _asof_cum(S3, pe, "seller_id", "q30", "purchase_ts")
    S["h_seller_vol30"] = n_now - n_30
    pe1 = S[["purchase_ts"]].copy(); pe1["one"] = 1.0
    a, _ = _asof_global(S, pe1, "purchase_ts", "purchase_ts", "one")
    S4 = S[["_qid"]].copy(); S4["q"] = S["purchase_ts"] - pd.Timedelta(days=14)
    b, _ = _asof_global(S4, pe1, "q", "purchase_ts", "one")
    S["h_platform_vol14"] = a - b
    ce = S[["customer_uid", "purchase_ts"]].copy()
    S["h_customer_prior_orders"], _ = _asof_cum(S, ce, "customer_uid", "purchase_ts", "purchase_ts")
    S = S.drop(columns=["_qid"]).copy()
    return S
