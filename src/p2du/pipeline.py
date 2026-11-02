"""Pipeline stages of the P2DU study and their orchestration (structure adapted from the first article's pipeline).

Stages
  data         delivery sample, sample-flow table and the purchase-time feature file
  eda          Phase-1 checks: meaning of the estimated date, censoring at the origins, survey timing, S by region
  rq1_fit      delivery-time models at every rolling origin (heavy; resumable, one file per origin)
  rq1          conformal corrections, coverage and promise length by model, variant, region and window
  rq2          response of the low-rating probability to the deviation from the promise (AIPW, RDD, E-values)
  rq3          promise policies: frontier of promise length versus low-rating rate; Olist's distance to it
  sens_fit     refits needed by the sensitivity analyses (Olist's estimate as an input; route shrinkage m = 20, 100;
               the members of the quantile average at the extra levels 0.97-0.995 for the uncalibrated benchmark)
  theory       checks of the propositions and of the premises of the RQs on the real data (tables/theory_checks.csv)
  sensitivity  sensitivity table (calibration length, score, Mondrian unit, ACI step, undelivered orders, ...)
  inference    paired window-level contrasts with block-bootstrap intervals and Holm adjustment
  figures      figures of the article (vector PDF and 600-dpi PNG)
  macros       LaTeX macros with every number quoted in the text (outputs/manuscript/results_macros.tex)
  tables       ready-to-use tables of the article (outputs/manuscript/table*.tex and .csv)
  catalog      index of all output tables and a digest of the main results (outputs/manuscript/*.md)
  crosscheck   independent recomputation of the main numbers from the raw CSV files (outputs_extra/)
  manifest     SHA-256 of every output file, code hashes and library versions
"""
import argparse
import hashlib
import json
import platform
import time
from importlib import metadata

import pandas as pd

from .config import ROOT, load_config
from .data.io import load_raw, sha256
from .data.delivery_sample import build_delivery_sample
from .data.purchase_features import build_purchase_features, BASE_CAT

ALL_STAGES = ("data", "eda", "rq1_fit", "rq1", "rq2", "rq3", "sens_fit", "sensitivity", "theory", "inference",
              "figures", "macros", "tables", "catalog", "crosscheck", "manifest")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_frame(out):
    p = out / "data" / "features_purchase.pkl"
    if not p.is_file():
        raise FileNotFoundError(f"{p} is missing. Run: python run_pipeline.py --stage data")
    D = pd.read_pickle(p)
    return D, {c: sorted(D[c].unique()) for c in BASE_CAT}


def stage_data(cfg, out, args):
    raw, _ = load_raw(cfg["data_dir"])
    A, flow = build_delivery_sample(raw, cfg)
    F = build_purchase_features(A, cfg)
    flow.to_csv(out / "tables" / "sample_flow.csv", index=False, encoding="utf-8-sig")
    F.to_pickle(out / "data" / "features_purchase.pkl")
    print(flow.to_string(index=False))


def stage_eda(cfg, out, args):
    from .experiments import eda
    D, _ = load_frame(out)
    eda.run_eda(D, cfg, out)


def _quick(cfg, args):
    if args.quick:
        cfg["rolling"]["step_days"] = 112
    return cfg


def stage_rq1_fit(cfg, out, args):
    from .experiments.rq1_fit import run_rq1_fit
    D, vocab = load_frame(out)
    run_rq1_fit(D, vocab, _quick(cfg, args), out / "data" / "parts", log=log, workers=args.workers,
                feat_path=out / "data" / "features_purchase.pkl")


def stage_rq1(cfg, out, args):
    from .experiments import rq1_coverage, rq1_extra
    D, _ = load_frame(out)
    cfg = _quick(cfg, args)
    rq1_coverage.run_rq1(D, cfg, out, log=log)
    rq1_extra.run_rq1_extra(D, cfg, out, log=log)


def stage_rq2(cfg, out, args):
    from .experiments import rq2_response
    D, vocab = load_frame(out)
    rq2_response.run_rq2(D, vocab, cfg, out, log=log)


def stage_rq3(cfg, out, args):
    from .experiments import rq3_policy
    D, _ = load_frame(out)
    rq3_policy.run_rq3(D, _quick(cfg, args), out, log=log)


def stage_sens_fit(cfg, out, args):
    from .experiments import sensitivity
    D, vocab = load_frame(out)
    sensitivity.run_sens_fit(D, vocab, _quick(cfg, args), out, log=log, workers=args.workers)


def stage_sensitivity(cfg, out, args):
    from .experiments import sensitivity
    D, _ = load_frame(out)
    sensitivity.run_sensitivity(D, _quick(cfg, args), out, log=log)


def stage_theory(cfg, out, args):
    from .experiments import theory_checks
    D, _ = load_frame(out)
    theory_checks.run_theory_checks(D, cfg, out, log=log)


def stage_inference(cfg, out, args):
    from .reporting import contrasts
    contrasts.run_contrasts(cfg, out)


def stage_figures(cfg, out, args):
    from .reporting import figures
    figures.make_all(cfg, out)


def stage_macros(cfg, out, args):
    from .reporting import macros
    macros.write_macros(cfg, out)


def stage_tables(cfg, out, args):
    from .reporting import manuscript_tables
    manuscript_tables.write_tables(cfg, out)


def stage_catalog(cfg, out, args):
    from .reporting import catalog
    catalog.write_catalog(cfg, out)


def stage_crosscheck(cfg, out, args):
    from .reporting import crosscheck
    R, S = crosscheck.run_crosscheck(cfg, out)
    log(f"  crosscheck: {len(R)} numbers, {int((R.status != 'match').sum())} mismatches")


def stage_manifest(cfg, out, args):
    libs = {}
    for lib in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "statsmodels", "quantile-forest", "matplotlib",
                "pyyaml"):
        try:
            libs[lib] = metadata.version(lib)
        except Exception:
            libs[lib] = "n/a"
    files = [dict(file=p.relative_to(out).as_posix(), sha256=sha256(p)) for p in sorted(out.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and "parts" not in p.parts
             and "logs" not in p.parts]
    src = sorted([*(ROOT / "src").rglob("*.py"), ROOT / "run_pipeline.py", ROOT / "config.yaml",
                  ROOT / "requirements-lock.txt"])
    code = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in src if p.is_file()}
    code_all = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    man = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(), platform=platform.platform(),
               libs=libs, seed=cfg["seed"], quick=args.quick, code_sha256=code_all, code_files=code, outputs=files)
    (out / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")


STAGES = {name: globals()[f"stage_{name}"] for name in ALL_STAGES}


def main():
    ap = argparse.ArgumentParser(description="P2DU pipeline")
    ap.add_argument("--stage", default="all", help="all, or a comma-separated list of: " + ", ".join(ALL_STAGES))
    ap.add_argument("--quick", action="store_true", help="smoke test on a few origins")
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=None, help="parallel processes of rq1_fit / sens_fit (default: config workers)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = cfg["out_dir"]
    if args.workers is None:
        args.workers = int(cfg.get("workers", 1))
    if args.quick:                      # a smoke test never touches the outputs of the reported run
        out = out.parent / (out.name + "_quick")
    for d in ("tables", "figures", "data", "logs", "manuscript"):
        (out / d).mkdir(parents=True, exist_ok=True)
    stages = ALL_STAGES if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s}")
    t_run = time.time()
    for s in stages:
        t = time.time(); log(f"Stage {s}: start")
        STAGES[s](load_config(args.config) if s != "manifest" else cfg, out, args)
        log(f"Stage {s}: done ({(time.time() - t) / 60:.1f} min)")
    log(f"Pipeline finished in {(time.time() - t_run) / 60:.1f} min ({len(stages)} stages, {args.workers} fitting processes)")
