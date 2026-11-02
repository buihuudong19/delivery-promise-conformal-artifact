"""Order-level delivery sample of the P2DU study and its sample-flow table.

Notation (days, counted from midnight of the purchase day)
  Y_i      delivery time: delivered timestamp - purchase day (continuous, > 0)
  D_i      floor(Y_i): calendar days between the purchase day and the delivery day
  P_i      Olist's promise: estimated delivery date - purchase day (integer; the estimate has day resolution)
  S_i      D_i - P_i, the deviation from the promise; the order is late if and only if S_i > 0

With this convention a promise of P days is kept exactly when floor(Y) <= P, so an upper bound U on Y gives the
integer promise floor(U), and Y <= U implies that the promise is kept (Proposition 1 of the article).

Rules
  R1  purchase time and estimated delivery date recorded, at least one item (route and seller known)
  R2  cancelled or unavailable orders are removed (no delivery to promise; status known soon after purchase)
  R3  the remaining orders without a delivery timestamp are kept as 'open': censored in training, and counted as
      missed promises only in the sensitivity analysis
"""
import numpy as np
import pandas as pd

from .order_attributes import order_attributes

DAY = pd.Timedelta(days=1)


def aggregate_reviews(reviews):
    """Worst score and survey timestamps of each order (the worst score is kept when an order has several reviews)."""
    g = reviews.groupby("order_id")
    return pd.DataFrame({
        "n_reviews": g.size(),
        "review_score": g["review_score"].min(),
        "review_created_ts": g["review_creation_date"].min(),
        "first_answer_ts": g["review_answer_timestamp"].min(),
        "last_answer_ts": g["review_answer_timestamp"].max(),
    })


def build_delivery_sample(raw, cfg):
    A = order_attributes(raw)
    flow = [("All orders in the orders table", len(A))]
    A = A[A["purchase_ts"].notna() & A["estimated_ts"].notna() & A["seller_id"].notna()]
    flow.append(("R1: purchase time, promised date and at least one item", len(A)))
    cancelled = A["order_status"].isin(cfg["sample"]["cancelled_status"]) & A["delivered_ts"].isna()
    A = A[~cancelled]
    flow.append(("R2: cancelled / unavailable orders removed", len(A)))
    A = A.copy()
    A["purchase_day"] = A["purchase_ts"].dt.normalize()
    A["h_frac"] = (A["purchase_ts"] - A["purchase_day"]) / DAY
    A["P_olist"] = ((A["estimated_ts"].dt.normalize() - A["purchase_day"]) / DAY).round().astype(int)
    A["Y"] = (A["delivered_ts"] - A["purchase_day"]) / DAY
    A["delivered"] = A["delivered_ts"].notna()
    A["D"] = np.floor(A["Y"])
    A["S_olist"] = A["D"] - A["P_olist"]
    A["late_olist"] = (A["S_olist"] > 0).astype(float).where(A["delivered"])
    # the moment the outcome of a promise P becomes known: the delivery, or the end of the promised day if it passes
    A["promise_end_ts"] = A["estimated_ts"].dt.normalize() + DAY
    flow.append(("   of which delivered", int(A["delivered"].sum())))
    flow.append(("   of which open (never delivered, kept as censored)", int((~A["delivered"]).sum())))
    rv = aggregate_reviews(raw["reviews"])
    A = A.join(rv, on="order_id")
    sc = cfg["sample"]
    A["low"] = np.where(A["review_score"].isin(sc["low_scores"]), 1.0,
                        np.where(A["review_score"].isin(sc["high_scores"]), 0.0, np.nan))
    A["is_neutral"] = (A["review_score"] == 3).astype(int)
    flow.append(("   delivered with a review (RQ2 population, three-star included)",
                 int((A["delivered"] & A["review_score"].notna()).sum())))
    A = A.sort_values("purchase_ts", kind="stable").reset_index(drop=True)
    flow_df = pd.DataFrame(flow, columns=["step", "n_orders"])
    return A, flow_df
