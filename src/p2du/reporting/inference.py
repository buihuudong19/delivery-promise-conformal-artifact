"""Paired window-level inference: block-bootstrap intervals, Holm adjustment within each family of comparisons and a
Newey-West (HAC) check that does not resample.

paired_summary(), hac_p() and holm() are reused unchanged from earlier code released under the MIT licence;
boot_means() is adapted from it so that the blocks wrap around the end of the series (circular block bootstrap) and
every window is drawn equally often. The families of comparisons are those of the present study (see contrasts()).
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

N_BOOT_P = 20000
BLOCKS = (1, 2, 3, 5, 6)


def boot_means(x, block, n_boot=2000, seed=42):
    x = np.asarray(x, float); n = len(x); block = min(block, n)
    rng = np.random.default_rng(seed); nb = math.ceil(n / block)
    st = rng.integers(0, n, size=(n_boot, nb))          # circular blocks: every window has the same weight
    return x[((st[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]].mean(1)


def paired_summary(d, block=3, n_boot=2000, seed=42):
    """Mean, 95% percentile interval and bootstrap p-value of a paired window-level difference."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) < 3:                        # too few windows for an interval (smoke tests only)
        return dict(mean=float(d.mean()) if len(d) else np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
    m = boot_means(d, block, n_boot, seed)
    mp = boot_means(d, block, N_BOOT_P, seed)
    p = max(2 * min((mp <= 0).mean(), (mp >= 0).mean()), 1 / N_BOOT_P)
    return dict(mean=float(d.mean()), ci_lo=float(np.percentile(m, 2.5)), ci_hi=float(np.percentile(m, 97.5)), p=float(p))


def hac_p(d, lag=2):
    """Two-sided p-value of a Newey-West t-test that the mean of an autocorrelated sequence is zero (Bartlett kernel,
    t distribution with n - 1 degrees of freedom); lag 2 matches the block length of 3 windows."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 3:
        return np.nan
    e = d - d.mean(); v = e @ e / n
    for j in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / n
    if v <= 0:
        return np.nan
    return float(2 * stats.t.sf(abs(d.mean()) / math.sqrt(v / n), n - 1))


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def contrast_row(family, label, d, cfg):
    bs = cfg["bootstrap"]
    r = dict(family=family, contrast=label, n_windows=int(np.sum(~np.isnan(np.asarray(d, float)))),
             **paired_summary(d, bs["block_len"], bs["n_boot"], cfg["seed"]))
    r["p_hac"] = hac_p(d, bs["block_len"] - 1)
    return r


def adjust(df, by="family"):
    df = df.copy(); df["p_holm"] = np.nan
    for _, idx in df.groupby(by).groups.items():
        ok = df.loc[idx, "p"].notna()
        ii = [j for j, o in zip(idx, ok) if o]
        if ii:
            df.loc[ii, "p_holm"] = holm(df.loc[ii, "p"])
    return df
