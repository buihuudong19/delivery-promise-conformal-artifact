"""RQ1, step 2: conformal corrections of the stored quantile predictions; coverage and promise length.

For every origin, model variant, miscoverage level alpha and conformal method, the promises of the evaluation orders
are scored on the orders delivered in the final snapshot (coverage = share of promises kept). The undelivered
orders are counted as missed promises in the column coverage_all (sensitivity). Olist's own promise is scored on the
same orders. Methods:
  raw              the model's (1 - alpha)-quantile, no correction
  split_naive      split conformal on the calibration orders delivered by the origin (censoring ignored)
  split_ipcw       split conformal with IPCW masses
  mondrian_naive   per-region correction, censoring ignored
  mondrian_ipcw    per-region correction with IPCW masses
  aci_ipcw         adaptive conformal inference over the sequence of purchases, delayed feedback
  mondrian_aci_ipcw  ACI run within each region
  split_oracle     split conformal with the final-snapshot outcomes of the calibration orders (not feasible at the
                   origin; it separates the censoring bias of the calibration window from the effect of drift)
"""
import numpy as np
import pandas as pd

from ..conformal.one_sided import (static_promises, aci_promises, make_ecdf, promise_from_bound)
from ..data.censoring import tail_share
from ..evaluation.metrics import pinball
from .rq1_fit import load_fits

METHODS = ("raw", "split_naive", "split_ipcw", "mondrian_naive", "mondrian_ipcw", "aci_ipcw", "mondrian_aci_ipcw",
           "split_oracle")
STORE_METHODS = ("raw", "split_ipcw", "mondrian_ipcw", "mondrian_aci_ipcw")   # per-order promises kept at alpha = 0.10
STORE_ALPHA = 0.10


def group_col(cfg):
    return "customer_region" if cfg["conformal"]["mondrian_unit"] == "region" else "customer_state"


def _cal_block(r, D, v, ti, cfg, tails=None):
    C = D.iloc[r["cal_idx"]]
    logq = r["pred"][v]["cal"][:, ti].astype(float)
    obs = r["cal_obs"]
    score = np.where(obs, np.log(np.where(obs, r["cal_y"], 1.0)) - logq, np.nan)
    tail, tail_g = tails if tails else (0.0, {})
    return dict(score=score, obs=obs, w=r["cal_w"], group=C[group_col(cfg)].fillna("NA").to_numpy(), tail=tail,
                tail_g=tail_g)


def _rows(k, origin, variant, method, alpha, E, P, cap=None):
    """Per-window and per-region metrics; with `cap`, also the share of promises at the cap (bound not finite)."""
    out = []
    ok = E["delivered"].to_numpy()
    Dv = E["D"].to_numpy()
    kept_all = np.where(ok, Dv <= P, False)
    for reg, m in [("all", np.ones(len(E), bool))] + [(g, (E["customer_region"] == g).to_numpy())
                                                     for g in sorted(E["customer_region"].dropna().unique())]:
        mm = m & ok
        if mm.sum() == 0:
            continue
        kept = Dv[mm] <= P[mm]
        out.append(dict(k=k, origin=origin, variant=variant, method=method, alpha=alpha, region=reg, n=int(mm.sum()),
                        coverage=float(kept.mean()), mean_promise=float(P[mm].mean()),
                        median_promise=float(np.median(P[mm])),
                        slack_kept=float((P[mm] - Dv[mm])[kept].mean()) if kept.any() else np.nan,
                        coverage_all=float(kept_all[m].mean()), n_all=int(m.sum())))
        if cap is not None:
            out[-1]["share_capped"] = float((P[m] >= cap - 1e-9).mean())
    return out


def run_rq1(D, cfg, out, log=print, tag="main", methods=METHODS, write=True, calib_days=None, variants=None):
    fits = load_fits(out / "data" / "parts", tag, cfg)
    if not fits:
        raise FileNotFoundError("no RQ1 fits; run the stage rq1_fit first")
    taus = cfg["quantiles"]; cap = cfg["conformal"]["promise_cap_days"]; gamma = cfg["conformal"]["aci_gamma"]
    variants = [v for v in fits[0]["pred"] if variants is None or v in variants]
    rows, loss_rows, cens_rows, store = [], [], [], []
    seq_parts = {}
    K = calib_days or cfg["rolling"]["calib_days"]
    for r in fits:
        E = D.iloc[r["eval_idx"]]
        ok = E["delivered"].to_numpy()
        tails = tail_share(D, r["origin"], K, groups=group_col(cfg))
        rows += _rows(r["k"], r["origin"], "olist", "actual", np.nan, E, E["P_olist"].to_numpy().astype(float))
        store.append(pd.DataFrame(dict(pos=r["eval_idx"], k=r["k"], variant="olist", method="actual",
                                       alpha=np.nan, P=E["P_olist"].to_numpy().astype(float))))
        cens_rows.append(dict(k=r["k"], origin=r["origin"], n_train=r["n_train"],
                              train_censored_share=r["train_censored_share"], n_cal=len(r["cal_obs"]),
                              cal_observed_share=float(r["cal_obs"].mean()),
                              ht_mass=float(r["cal_w"].sum() / max(len(r["cal_obs"]), 1)),
                              cal_weight_mean=float(r["cal_w"][r["cal_obs"]].mean()),
                              cal_weight_max=float(r["cal_w"].max()), tail_share=tails[0]))
        for v in variants:
            QE = r["pred"][v]["eval"].astype(float)
            ly = np.log(E["Y"].to_numpy()[ok])
            for j, t in enumerate(taus):
                loss_rows.append(dict(k=r["k"], origin=r["origin"], variant=v, tau=t, pinball=pinball(ly, QE[ok, j], t),
                                      coverage_raw=float((ly <= QE[ok, j]).mean())))
            for a in cfg["alphas"]:
                ti = taus.index(round(1 - a, 4))
                cal = _cal_block(r, D, v, ti, cfg, tails)
                ge = E[group_col(cfg)].fillna("NA").to_numpy()
                for meth in methods:
                    if "aci" in meth:
                        continue
                    base, _, wt = meth.partition("_")
                    if wt == "oracle":
                        # oracle: final outcomes of the delivered calibration orders, the population of the
                        # coverage target (never-delivered orders are not part of it)
                        Cd = D.iloc[r["cal_idx"]]
                        dl = Cd["delivered"].to_numpy()
                        yo = np.where(dl, Cd["Y"].to_numpy(), 1.0)
                        lq = r["pred"][v]["cal"][:, ti].astype(float)
                        oc = dict(score=np.log(yo) - lq, obs=dl, w=np.ones(len(yo)), group=cal["group"])
                        P = static_promises(QE[:, ti], ge, oc, a, base, "naive", cfg)
                    else:
                        P = static_promises(QE[:, ti], ge, cal, a, base, wt or "ipcw", cfg)
                    rows += _rows(r["k"], r["origin"], v, meth, a, E, P, cap=cfg["conformal"]["promise_cap_days"])
                    if meth in STORE_METHODS and abs(a - STORE_ALPHA) < 1e-9:
                        store.append(pd.DataFrame(dict(pos=r["eval_idx"], k=r["k"], variant=v,
                                                       method=meth, alpha=a, P=P)))
                # ingredients of the ACI runs (sequential over all windows)
                pooled = make_ecdf(cal["score"], cal["obs"], cal["w"], "ipcw", cal["tail"])
                ecd = {(r["k"], None): pooled}
                for g in pd.unique(cal["group"]):
                    m = cal["group"] == g
                    if m.sum() >= cfg["conformal"]["mondrian_min_n"]:
                        ecd[(r["k"], g)] = make_ecdf(cal["score"][m], cal["obs"][m], cal["w"][m], "ipcw",
                                                     cal["tail_g"].get(g, cal["tail"]))
                sp = seq_parts.setdefault((v, a), dict(frames=[], ecdfs={}))
                sp["frames"].append(pd.DataFrame(dict(pos=r["eval_idx"], k=r["k"], logq=QE[:, ti], group=ge)))
                sp["ecdfs"].update(ecd)
    aci_rows, traj = [], []
    for (v, a), sp in seq_parts.items():
        S = pd.concat(sp["frames"], ignore_index=True)
        X = D.iloc[S["pos"]].reset_index(drop=True)
        S = pd.concat([S, X[["order_id", "purchase_ts", "purchase_day", "D", "delivered_ts", "delivered",
                             "customer_region", "Y"]]], axis=1).sort_values("purchase_ts", kind="stable")
        S.attrs["ecdfs"] = sp["ecdfs"]
        for meth, by_g in (("aci_ipcw", False), ("mondrian_aci_ipcw", True)):
            if meth not in methods:
                continue
            P, at = aci_promises(S, a, gamma, by_g, cap, cfg["conformal"].get("aci_step", "day"),
                                 a * cfg["conformal"].get("aci_alpha_min_ratio", 0.5))
            S2 = S.assign(P=P, alpha_t=at)
            for k, g in S2.groupby("k"):
                aci_rows += _rows(k, g["purchase_ts"].min().normalize(), v, meth, a, g, g["P"].to_numpy(),
                                  cap=cfg["conformal"]["promise_cap_days"])
                if meth in STORE_METHODS and abs(a - STORE_ALPHA) < 1e-9:
                    store.append(pd.DataFrame(dict(pos=g["pos"].to_numpy(), k=k, variant=v, method=meth,
                                                   alpha=a, P=g["P"].to_numpy())))
            if meth == "aci_ipcw":
                d = S2.set_index("purchase_ts")["alpha_t"].resample("D").last().dropna()
                traj.append(pd.DataFrame(dict(day=d.index, alpha_t=d.to_numpy(), variant=v, alpha=a)))
    # origin dates of the ACI rows are taken from the fit records
    org = {r["k"]: r["origin"] for r in fits}
    for x in aci_rows:
        x["origin"] = org[x["k"]]
    M = pd.DataFrame(rows + aci_rows)
    L = pd.DataFrame(loss_rows); Cn = pd.DataFrame(cens_rows)
    T = pd.concat(traj, ignore_index=True) if traj else pd.DataFrame()
    St = pd.concat(store, ignore_index=True)
    if write:
        t = out / "tables"
        M.to_csv(t / f"rq1_window_metrics{'' if tag == 'main' else '_' + tag}.csv", index=False)
        L.to_csv(t / f"rq1_pinball{'' if tag == 'main' else '_' + tag}.csv", index=False)
        if tag == "main":
            Cn.to_csv(t / "rq1_censoring_by_origin.csv", index=False)
            T.to_csv(t / "rq1_aci_alpha_trajectory.csv", index=False)
            St.to_csv(out / "data" / "rq1_promises.csv.gz", index=False, compression="gzip")
            summarise(M, cfg).to_csv(t / "rq1_summary.csv", index=False)
            summarise_loss(L).to_csv(t / "rq1_pinball_summary.csv", index=False)
    log(f"  rq1 [{tag}]: {len(fits)} origins, {len(M)} metric rows")
    return M, L


def summarise(M, cfg):
    g = M.groupby(["variant", "method", "alpha", "region"], dropna=False)
    S = g.agg(n_windows=("k", "nunique"), n=("n", "sum"), coverage=("coverage", "mean"),
              coverage_min=("coverage", "min"), mean_promise=("mean_promise", "mean"),
              coverage_all=("coverage_all", "mean"), slack_kept=("slack_kept", "mean")).reset_index()
    S["target"] = 1 - S["alpha"]
    M2 = M.assign(_under=(M["coverage"] < 1 - M["alpha"] - 0.02).astype(float).where(M["alpha"].notna()))
    S["under_windows"] = M2.groupby(["variant", "method", "alpha", "region"], dropna=False)["_under"].mean().to_numpy()
    reg = M[M["region"] != "all"].groupby(["variant", "method", "alpha", "k"], dropna=False)["coverage"].agg(["min", "max"])
    reg = reg.assign(gap=reg["max"] - reg["min"]).groupby(level=[0, 1, 2], dropna=False).agg(
        region_gap=("gap", "mean"), worst_region=("min", "mean")).reset_index()
    return S.merge(reg, on=["variant", "method", "alpha"], how="left")


def summarise_loss(L):
    return L.groupby(["variant", "tau"]).agg(pinball=("pinball", "mean"), coverage_raw=("coverage_raw", "mean"),
                                             n_windows=("k", "nunique")).reset_index()
