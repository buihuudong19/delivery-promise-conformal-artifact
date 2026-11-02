"""Evaluation metrics of the delivery-time bounds and the moving-block bootstrap.

moving_block_bootstrap_mean() and block_indices() are adapted from earlier code released under the MIT licence: the
blocks wrap around the end of the series (circular block bootstrap), so that the first and last windows are drawn as
often as the others.
"""
import math

import numpy as np


def pinball(y, q, tau, w=None):
    """Mean pinball (quantile) loss of the predicted tau-quantile q for outcomes y, optionally weighted."""
    u = np.asarray(y, float) - np.asarray(q, float)
    loss = np.maximum(tau * u, (tau - 1) * u)
    return float(np.average(loss, weights=w)) if w is not None else float(loss.mean())


def promise_metrics(D_int, P):
    """Coverage (share of promises kept, floor(Y) <= P), late rate, mean and median promise, mean slack of kept ones."""
    D_int = np.asarray(D_int, float); P = np.asarray(P, float)
    kept = D_int <= P
    return dict(n=len(P), coverage=float(kept.mean()), late=float(1 - kept.mean()), mean_promise=float(P.mean()),
                median_promise=float(np.median(P)), slack_kept=float((P - D_int)[kept].mean()) if kept.any() else np.nan)


def moving_block_bootstrap_mean(x, n_boot=2000, block=3, seed=42):
    """Mean of a time-ordered series with a 95% moving-block bootstrap interval."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.array([])
    block = min(block, n)
    rng = np.random.default_rng(seed)
    nb = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, nb))      # circular blocks: every window has the same weight
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), means


def block_indices(n, n_boot=2000, block=3, seed=42):
    rng = np.random.default_rng(seed)
    nb = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, nb))
    return ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
