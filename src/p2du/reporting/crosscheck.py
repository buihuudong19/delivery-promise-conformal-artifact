"""Independent cross-checks of the reported numbers (outputs_extra/).

Every quantity below is recomputed directly from the nine raw CSV files with plain pandas: this module imports nothing
from the p2du package except the configuration, so an error in the feature, sample or experiment code would show up
as a mismatch. The results are compared with the tables of the pipeline.

  C-01..03  sample counts (all orders, after the rules, delivered)
  C-04..05  Olist's mean promise and late rate
  C-06      late rate, mean promise and low-rating rate by region
  C-07      crude low-rating rate and size of every deviation bin of RQ2
  C-08      share of late orders rated before arrival and their low-rating rate
  C-09      Olist's coverage and mean promise in every evaluation window of RQ1
  C-10      coverage and mean promise of the stored per-order promises (main model and quantile average, split /
            Mondrian IPCW, alpha 0.10)
            recomputed against the raw delivery dates
  C-11      Olist's realised low-rating rate in every RQ3 window (raw deviations + response curve)
Outputs: outputs_extra/tables/crosscheck_results.csv (one row per compared number), crosscheck_summary.csv,
outputs_extra/results_extra.tex (macros of the checks) and outputs_extra/README.md.
"""
import time

import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)
CHECKS = {"C-01": "all orders in the orders table", "C-02": "orders after rules R1-R2", "C-03": "delivered orders",
          "C-04": "Olist's mean promise", "C-05": "Olist's late rate", "C-06": "late rate, mean promise, low-rating rate by region",
          "C-07": "RQ2 deviation bins: size and crude low-rating rate", "C-08": "late orders rated before arrival",
          "C-09": "RQ1: Olist's coverage and mean promise per window",
          "C-10": "RQ1: stored per-order promises of the main model and the quantile average re-scored on the raw delivery dates",
          "C-11": "RQ3: Olist's realised low-rating rate per window"}


def _raw(cfg):
    d = cfg["data_dir"]
    rd = lambda f: pd.read_csv(d / f, encoding="utf-8-sig")
    o = rd("olist_orders_dataset.csv")
    for c in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        o[c] = pd.to_datetime(o[c])
    it = rd("olist_order_items_dataset.csv")
    cu = rd("olist_customers_dataset.csv")
    rv = rd("olist_order_reviews_dataset.csv")
    for c in ["review_answer_timestamp", "review_creation_date"]:
        rv[c] = pd.to_datetime(rv[c])
    return o, it, cu, rv


REG = {**{s: "Norte" for s in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
       **{s: "Nordeste" for s in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
       **{s: "Centro-Oeste" for s in ["DF", "GO", "MT", "MS"]},
       **{s: "Sudeste" for s in ["ES", "MG", "RJ", "SP"]}, **{s: "Sul" for s in ["PR", "RS", "SC"]}}


def independent_frame(cfg):
    o, it, cu, rv = _raw(cfg)
    n_all = len(o)
    o = o[o["order_id"].isin(it["order_id"]) & o["order_purchase_timestamp"].notna() &
          o["order_estimated_delivery_date"].notna()]
    n_r1 = len(o)
    canc = o["order_status"].isin(cfg["sample"]["cancelled_status"]) & o["order_delivered_customer_date"].isna()
    o = o[~canc].copy()
    pday = o["order_purchase_timestamp"].dt.floor("D")
    o["P"] = (o["order_estimated_delivery_date"].dt.floor("D") - pday).dt.days
    o["Yd"] = (o["order_delivered_customer_date"] - pday).dt.total_seconds() / 86400
    o["Dd"] = np.floor(o["Yd"])
    o["S"] = o["Dd"] - o["P"]
    o["region"] = o["customer_id"].map(cu.set_index("customer_id")["customer_state"]).map(REG)
    g = rv.groupby("order_id")
    o["score"] = o["order_id"].map(g["review_score"].min())
    o["first_answer"] = o["order_id"].map(g["review_answer_timestamp"].min())
    return o, dict(n_all=n_all, n_r1=n_r1, n_r2=len(o), n_delivered=int(o["order_delivered_customer_date"].notna().sum()))


def _cmp(rows, cid, item, pipe, indep, tol=1e-9):
    d = abs(float(pipe) - float(indep))
    rows.append(dict(check=cid, item=item, pipeline=float(pipe), independent=float(indep), abs_diff=d,
                     status="match" if d <= tol else "MISMATCH"))


def run_crosscheck(cfg, out):
    t = out / "tables"; ex = out.parent / "outputs_extra"; (ex / "tables").mkdir(parents=True, exist_ok=True)
    o, n = independent_frame(cfg)
    rows = []
    fl = pd.read_csv(t / "sample_flow.csv")["n_orders"].to_numpy()
    _cmp(rows, "C-01", "all orders", fl[0], n["n_all"]); _cmp(rows, "C-02", "after R1-R2", fl[2], n["n_r2"])
    _cmp(rows, "C-03", "delivered", fl[3], n["n_delivered"])
    P = pd.read_csv(t / "eda_promise_checks.csv").set_index("check")["value"]
    dl = o[o["order_delivered_customer_date"].notna()]
    _cmp(rows, "C-04", "Olist mean promise (days)", P["promise P in days: mean"], o["P"].mean())
    _cmp(rows, "C-05", "Olist late rate", P["late rate (S > 0) among delivered orders"], (dl["S"] > 0).mean())
    rp = pd.read_csv(t / "eda_region_profile.csv").set_index("customer_region")
    lowi = np.where(dl["score"] <= 2, 1.0, np.where(dl["score"] >= 4, 0.0, np.nan))
    for gname, x in dl.assign(lowi=lowi).groupby("region"):
        _cmp(rows, "C-06", f"late rate, {gname}", rp.loc[gname, "late"], (x["S"] > 0).mean())
        _cmp(rows, "C-06", f"mean promise, {gname}", rp.loc[gname, "P_mean"], x["P"].mean())
        _cmp(rows, "C-06", f"low-rating rate, {gname}", rp.loc[gname, "low"], np.nanmean(x["lowi"]))
    R = dl[dl["score"].notna() & (dl["score"] != 3)].copy()
    R["low"] = (R["score"] <= 2).astype(float)
    R["bin"] = pd.cut(R["S"], cfg["rq2"]["bins"], labels=False, right=True)
    A = pd.read_csv(t / "rq2_aipw_bins.csv"); A = A[A.learner == "xgb"].set_index("bin")
    for b, x in R.groupby("bin"):
        _cmp(rows, "C-07", f"bin {A.loc[b, 'label']}: orders", A.loc[b, "n"], len(x))
        _cmp(rows, "C-07", f"bin {A.loc[b, 'label']}: crude low-rating rate", A.loc[b, "crude"], x["low"].mean())
    late = R[R["S"] > 0]
    before = late["first_answer"] < late["order_delivered_customer_date"]
    Mm = pd.read_csv(t / "rq2_survey_mechanism.csv"); ml = Mm[Mm.late == 1].set_index("answered_before_delivery")
    _cmp(rows, "C-08", "late orders rated before arrival (share)", ml.loc[1, "share_within_late_group"], before.mean())
    _cmp(rows, "C-08", "low-rating rate, late and rated before arrival", ml.loc[1, "mean"], late.loc[before, "low"].mean())
    M = pd.read_csv(t / "rq1_window_metrics.csv", parse_dates=["origin"])
    H = pd.Timedelta(days=cfg["rolling"]["horizon_days"])
    olw = M[(M.variant == "olist") & (M.region == "all")].set_index("k")
    for k, r in olw.iterrows():
        x = dl[(dl["order_purchase_timestamp"] >= r["origin"]) & (dl["order_purchase_timestamp"] < r["origin"] + H)]
        _cmp(rows, "C-09", f"window {k}: Olist coverage", r["coverage"], (x["S"] <= 0).mean())
        _cmp(rows, "C-09", f"window {k}: Olist mean promise", r["mean_promise"], x["P"].mean())
    # C-10: stored promises against the raw delivery dates
    feat = pd.read_pickle(out / "data" / "features_purchase.pkl")[["order_id"]]
    St = pd.read_csv(out / "data" / "rq1_promises.csv.gz")
    St["order_id"] = feat["order_id"].to_numpy()[St["pos"].to_numpy()]
    Dmap = o.set_index("order_id")["Dd"]
    ens = (cfg.get("ensemble") or {}).get("name")
    for v in [cfg["rq3"]["model"]] + ([ens] if ens and (St.variant == ens).any() else []):
        for meth in ("split_ipcw", "mondrian_ipcw"):
            x = St[(St.variant == v) & (St.method == meth)].copy()
            x["D"] = x["order_id"].map(Dmap)
            x = x[x["D"].notna()]
            pm = M[(M.variant == v) & (M.method == meth) & np.isclose(M.alpha, 0.10) & (M.region == "all")].set_index("k")
            for k, g in x.groupby("k"):
                _cmp(rows, "C-10", f"window {k}: {v} {meth} coverage", pm.loc[k, "coverage"], (g["D"] <= g["P"]).mean())
                _cmp(rows, "C-10", f"window {k}: {v} {meth} mean promise", pm.loc[k, "mean_promise"], g["P"].mean())
    # C-11: Olist's realised low-rating rate in the RQ3 windows
    C = pd.read_csv(t / "rq2_response_curve.csv"); C = C[C.group == "all"].sort_values("S")
    rf = lambda s: np.interp(np.clip(s, C["S"].iloc[0], C["S"].iloc[-1]), C["S"], C["r"])
    Q = pd.read_csv(t / "rq3_matched.csv", parse_dates=["origin"]).set_index("k")
    for k, r in Q.iterrows():
        x = dl[(dl["order_purchase_timestamp"] >= r["origin"]) & (dl["order_purchase_timestamp"] < r["origin"] + H)]
        _cmp(rows, "C-11", f"window {k}: Olist realised low-rating rate", r["low_olist"], rf(x["S"].to_numpy()).mean())
    Rr = pd.DataFrame(rows)
    Rr.to_csv(ex / "tables" / "crosscheck_results.csv", index=False)
    Sm = Rr.groupby("check").agg(n=("item", "size"), mismatches=("status", lambda s: int((s != "match").sum())),
                                 max_abs_diff=("abs_diff", "max")).reset_index()
    Sm.to_csv(ex / "tables" / "crosscheck_summary.csv", index=False)
    tex = ["% Generated by the P2DU pipeline (stage crosscheck). Do not edit by hand."]
    tex.append(f"\\newcommand{{\\pdXcheckItems}}{{{len(Rr):,}}}")
    tex.append(f"\\newcommand{{\\pdXcheckMismatches}}{{{int((Rr.status != 'match').sum())}}}")
    tex.append(f"\\newcommand{{\\pdXcheckMaxDiff}}{{{Rr['abs_diff'].max():.1e}}}")
    (ex / "results_extra.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    readme = ["# Independent cross-checks (outputs_extra)", "",
              f"Written by the stage `crosscheck` ({time.strftime('%Y-%m-%d %H:%M')}). Each number is recomputed from the "
              "raw Olist CSV files with plain pandas (src/p2du/reporting/crosscheck.py imports nothing from the pipeline "
              "code) and compared with the pipeline tables.", "",
              f"Result: {len(Rr)} numbers compared, {int((Rr.status != 'match').sum())} mismatches, largest absolute "
              f"difference {Rr['abs_diff'].max():.2e}.", "", "| Check | Numbers | Mismatches | Max abs. diff. |",
              "|---|---|---|---|---|"]
    readme[-2] = "| Check | What | Numbers | Mismatches | Max abs. diff. |"
    readme += [f"| {r['check']} | {CHECKS.get(r['check'], '')} | {r['n']} | {r['mismatches']} | {r['max_abs_diff']:.2e} |"
               for _, r in Sm.iterrows()]
    readme += ["", "Files: `tables/crosscheck_results.csv` (every comparison), `tables/crosscheck_summary.csv`, "
               "`results_extra.tex` (macros)."]
    (ex / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return Rr, Sm
