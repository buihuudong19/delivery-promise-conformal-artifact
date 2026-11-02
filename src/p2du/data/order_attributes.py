# Reused from earlier code released under the MIT licence: the order attributes, the region map and the two as-of
# counters are copied unchanged. The feature definitions of that code (prediction at the carrier hand-over) are not
# used here.
"""Order-level attributes of all Olist orders and the as-of counting helpers used by the purchase-time features.

_asof_cum(queries, events, key, qtime, etime, value) returns, for each query row, the number of events of the same key
that happened strictly before the query time, and the sum of their values. Only events with an event time before the
query time enter, which is what makes a feature admissible at that time.
"""
import numpy as np
import pandas as pd

REGION = {**{s: "Norte" for s in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
          **{s: "Nordeste" for s in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
          **{s: "Centro-Oeste" for s in ["DF", "GO", "MT", "MS"]},
          **{s: "Sudeste" for s in ["ES", "MG", "RJ", "SP"]},
          **{s: "Sul" for s in ["PR", "RS", "SC"]}}
DAY = pd.Timedelta(days=1)


def _haversine(lat1, lon1, lat2, lon2):
    r1, r2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = r2 - r1, np.radians(lon2 - lon1)
    h = np.sin(dlat / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def order_attributes(raw):
    """Order-level attributes for ALL orders (keys for the history events and inputs to the features)."""
    o = raw["orders"].rename(columns={"order_purchase_timestamp": "purchase_ts", "order_approved_at": "approved_ts",
                                      "order_delivered_carrier_date": "t0", "order_delivered_customer_date": "delivered_ts",
                                      "order_estimated_delivery_date": "estimated_ts"})
    it = raw["items"].merge(raw["products"], on="product_id", how="left") \
        .merge(raw["translation"], on="product_category_name", how="left") \
        .merge(raw["sellers"], on="seller_id", how="left") \
        .sort_values(["order_id", "order_item_id"], kind="stable")
    it["volume_cm3"] = it["product_length_cm"] * it["product_height_cm"] * it["product_width_cm"]
    g = it.groupby("order_id")
    first = g.first()
    items = pd.DataFrame({
        "n_items": g.size(), "n_products": g["product_id"].nunique(), "n_sellers": g["seller_id"].nunique(),
        "price_sum": g["price"].sum(), "price_max": g["price"].max(), "freight_sum": g["freight_value"].sum(),
        "seller_id": first["seller_id"], "seller_state": first["seller_state"],
        "seller_zip": first["seller_zip_code_prefix"],
        "category": first["product_category_name_english"].fillna("unknown"),
        "weight_g": first["product_weight_g"], "volume_cm3": first["volume_cm3"],
        "photos_qty": first["product_photos_qty"], "name_len": first["product_name_lenght"],
        "desc_len": first["product_description_lenght"],
    })
    pay = raw["payments"]
    pg = pay.groupby("order_id")
    main_type = pay.sort_values(["order_id", "payment_value"], ascending=[True, False]).groupby("order_id")["payment_type"].first()
    pays = pd.DataFrame({"pay_value": pg["payment_value"].sum(), "n_payments": pg.size(),
                         "installments": pg["payment_installments"].max(), "pay_type": main_type})
    c = raw["customers"].set_index("customer_id")
    df = o.set_index("order_id").join(items, how="left").join(pays, how="left")
    df["customer_state"] = df["customer_id"].map(c["customer_state"])
    df["customer_zip"] = df["customer_id"].map(c["customer_zip_code_prefix"])
    df["customer_uid"] = df["customer_id"].map(c["customer_unique_id"])
    geo = raw["geo"].groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()
    df["distance_km"] = _haversine(df["customer_zip"].map(geo["geolocation_lat"]), df["customer_zip"].map(geo["geolocation_lng"]),
                                   df["seller_zip"].map(geo["geolocation_lat"]), df["seller_zip"].map(geo["geolocation_lng"]))
    df["customer_region"] = df["customer_state"].map(REGION)
    df["seller_region"] = df["seller_state"].map(REGION)
    return df.reset_index()


def _asof_cum(queries, events, key, qtime, etime, value=None):
    """For each query, the number of events of the same key strictly before the query time and the sum of their values."""
    ev = events[[key, etime] + ([value] if value else [])].dropna(subset=[key, etime]).sort_values(etime, kind="stable").copy()
    ev["_n"] = ev.groupby(key).cumcount() + 1
    ev["_s"] = ev.groupby(key)[value].cumsum() if value else 0.0
    q = queries[["_qid", key, qtime]].copy()
    q[key] = q[key].fillna("__NA__")
    ev[key] = ev[key].astype(q[key].dtype) if ev[key].dtype != q[key].dtype else ev[key]
    q = q.dropna(subset=[qtime]).sort_values(qtime, kind="stable")
    m = pd.merge_asof(q, ev[[key, etime, "_n", "_s"]].rename(columns={etime: "_et"}), left_on=qtime, right_on="_et",
                      by=key, direction="backward", allow_exact_matches=False)
    m = m.set_index("_qid").reindex(queries["_qid"])
    return m["_n"].fillna(0).to_numpy(), m["_s"].fillna(0).to_numpy()


def _asof_global(queries, events, qtime, etime, value):
    ev = events[[etime, value]].dropna().sort_values(etime, kind="stable").copy()
    ev["_n"] = np.arange(1, len(ev) + 1)
    ev["_s"] = ev[value].cumsum()
    q = queries[["_qid", qtime]].dropna().sort_values(qtime, kind="stable")
    m = pd.merge_asof(q, ev[[etime, "_n", "_s"]].rename(columns={etime: "_et"}), left_on=qtime, right_on="_et",
                      direction="backward", allow_exact_matches=False).set_index("_qid").reindex(queries["_qid"])
    return m["_n"].fillna(0).to_numpy(), m["_s"].fillna(0).to_numpy()
