"""Pilot (not part of the pipeline): does a wider tuning grid, or averaging stored quantiles across models, shorten
the conformal promise at the same coverage?

Part A  ensemble of the stored predictions (all 30 origins, no refit): mean of the log-quantiles of xgbq_ipcw and
        qrf_ipcw (and of xgbq, qrf, lqr), then split IPCW conformal.
Part B  wider grids on six origins: each candidate is fitted exactly as the pipeline does (inner-block loss for the
        selection, full fit for the promises), then split IPCW conformal on the evaluation window.
Writes CSV files to docs/evidence/pilot_tuning_2026-10-31/. Usage: python tools/pilot_tuning_ensemble.py [A|B|AB]
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2du.config import load_config                                   # noqa: E402
from p2du.pipeline import load_frame                                  # noqa: E402
from p2du.experiments import rq1_coverage, rq1_extra                  # noqa: E402
from p2du.experiments.rq1_fit import load_fits, fit_origin            # noqa: E402
from p2du.data.purchase_features import feature_set                   # noqa: E402
from p2du.evaluation.origins import rolling_windows                   # noqa: E402

HERE = ROOT / "docs" / "evidence" / "pilot_tuning_2026-10-31"
HERE.mkdir(parents=True, exist_ok=True)
part = sys.argv[1] if len(sys.argv) > 1 else "AB"
cfg = load_config()
out = cfg["out_dir"]
D, vocab = load_frame(out)
fits = load_fits(out / "data" / "parts", "main")


def matched(recs, variants):
    real = rq1_extra.load_fits
    rq1_extra.load_fits = lambda *a, **k: recs
    try:
        return rq1_extra.matched_olist(D, cfg, out, variants)
    finally:
        rq1_extra.load_fits = real


def evaluate(recs, variants):
    real = rq1_coverage.load_fits
    rq1_coverage.load_fits = lambda *a, **k: recs
    try:
        M, _ = rq1_coverage.run_rq1(D, cfg, out, log=lambda *_: None, methods=("split_ipcw",), write=False,
                                    variants=variants)
    finally:
        rq1_coverage.load_fits = real
    M = M[(M.region == "all") & (M.method == "split_ipcw")]
    return M[["k", "variant", "alpha", "coverage", "mean_promise"]]


if "A" in part:
    t0 = time.time()
    recs = []
    for r in fits:
        r = dict(r); r["pred"] = dict(r["pred"])
        for name, members in (("ens_xgbq_qrf", ("xgbq_ipcw", "qrf_ipcw")),
                              ("ens_xgbq_qrf_lqr", ("xgbq_ipcw", "qrf_ipcw", "lqr_ipcw")),
                              ("ens_all5", ("route_ipcw", "lqr_ipcw", "xgbq_ipcw", "qrf_ipcw", "aft"))):
            r["pred"][name] = {s: np.mean([r["pred"][m][s].astype(float) for m in members], axis=0)
                               for s in ("cal", "eval")}
        recs.append(r)
    A = evaluate(recs, ["xgbq_ipcw", "qrf_ipcw", "lqr_ipcw", "aft", "ens_xgbq_qrf", "ens_xgbq_qrf_lqr", "ens_all5"])
    A.to_csv(HERE / "pilot_A_ensemble.csv", index=False)
    matched(recs, ["xgbq_ipcw", "qrf_ipcw", "lqr_ipcw", "aft", "ens_xgbq_qrf", "ens_xgbq_qrf_lqr",
                   "ens_all5"]).to_csv(HERE / "pilot_A_matched.csv", index=False)
    print(f"part A done in {time.time() - t0:.0f}s", flush=True)

if "B" in part:
    W = {w["k"]: w for w in rolling_windows(cfg)}
    KS = [5, 10, 15, 20, 25, 30]
    num, cat = feature_set(cfg, False)
    cands = {
        "xgbq": [dict(max_depth=2), dict(max_depth=3), dict(max_depth=4),
                 dict(max_depth=4, min_child_weight=50), dict(max_depth=3, learning_rate=0.1),
                 dict(max_depth=4, reg_lambda=10.0)],
        "qrf": [dict(min_samples_leaf=3), dict(min_samples_leaf=5), dict(min_samples_leaf=10),
                dict(min_samples_leaf=5, max_features=0.33), dict(min_samples_leaf=5, max_samples=0.8)],
        "aft": [("extreme", 0.3), ("extreme", 0.45), ("extreme", 0.6), ("normal", 0.45)],
    }
    rows = []
    for k in KS:
        w = W[k]
        base = next(r for r in fits if r["k"] == k)
        for kind, lst in cands.items():
            for j, c in enumerate(lst):
                t0 = time.time()
                c2 = copy.deepcopy(cfg)
                if kind == "aft":
                    c2["models"]["aft"]["distributions"] = [c[0]]; c2["models"]["aft"]["scales"] = [c[1]]
                    v = "aft"; label = f"{c[0]}, scale {c[1]}"
                else:
                    c2["tuning"][f"{kind}_grid"] = [c]
                    v = f"{kind}_ipcw"; label = ", ".join(f"{a}={b}" for a, b in c.items())
                r = fit_origin(D, w, num, cat, vocab, c2, variants=(v,), prev_chosen=None)
                name = f"{kind}#{j}"
                rec = dict(base); rec["pred"] = {name: r["pred"][v]}
                M = evaluate([rec], [name])
                Lm = matched([rec], [name]).iloc[0]
                for _, m in M.iterrows():
                    rows.append(dict(k=k, kind=kind, cand=j, label=label, inner_loss=r["inner_loss"].get(kind),
                                     alpha=m["alpha"], coverage=m["coverage"], mean_promise=m["mean_promise"],
                                     length_gap_vs_olist=Lm["length_gap"], seconds=time.time() - t0))
                print(f"k={k} {kind} {label}: inner {r['inner_loss'].get(kind):.5f} "
                      f"{time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(HERE / "pilot_B_grid.csv", index=False)
    print("part B done", flush=True)
