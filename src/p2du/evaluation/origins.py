"""Rolling origins on the purchase time (a rolling-origin protocol adapted to a purchase-time decision).

rolling_windows() is reused from earlier code released under the MIT licence.
At an origin o, with calibration length K and inner validation length V:
  training set      orders bought before o - K; outcomes as known at o (orders not delivered by o are censored)
  calibration set   orders bought in [o - K, o); outcomes as known at o (heavily censored for recent purchases)
  evaluation set    orders bought in [o, o + H); promises are made at purchase with the model of origin o, and they
                    are scored with the outcomes recorded in the final snapshot of the data
  inner split       the last V days of the training set validate the hyper-parameters of origin o
"""
import pandas as pd


def rolling_windows(cfg):
    r = cfg["rolling"]
    origins = pd.date_range(r["first_origin"], r["last_origin"], freq=f"{r['step_days']}D")
    H = pd.Timedelta(days=r["horizon_days"])
    return [dict(k=i + 1, origin=a, end=a + H) for i, a in enumerate(origins)]


def origin_masks(D, w, cfg, calib_days=None):
    K = pd.Timedelta(days=calib_days or cfg["rolling"]["calib_days"])
    o = w["origin"]
    t = D["purchase_ts"]
    return dict(train=(t < o - K).to_numpy(), cal=((t >= o - K) & (t < o)).to_numpy(),
                eval=((t >= o) & (t < w["end"])).to_numpy(), cal_start=o - K)


def inner_masks(D, train_mask, w, cfg, calib_days=None):
    K = pd.Timedelta(days=calib_days or cfg["rolling"]["calib_days"])
    V = pd.Timedelta(days=cfg["tuning"]["inner_val_days"])
    t = D["purchase_ts"]
    cut = w["origin"] - K - V
    return (train_mask & (t < cut).to_numpy()), (train_mask & (t >= cut).to_numpy())
