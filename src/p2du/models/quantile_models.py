"""Models of the delivery time at purchase: two non-ML benchmarks and four machine-learning models.

Every model predicts conditional quantiles of log Y at the levels of config `quantiles`; predictions are
exponentiated by the caller. Quantile regression is equivariant to monotone transformations, so a quantile of log Y
is the log of the quantile of Y. Crossing quantiles are rearranged (sorted across levels), which can only reduce the
pinball loss (Chernozhukov, Fernandez-Val and Galichon, 2010).

  route   empirical quantiles by route (seller state x customer state), shrunk towards the customer state and the
          whole platform for routes with few orders (benchmark, no ML)
  lqr     linear quantile regression (statsmodels QuantReg). Weights enter through the positive homogeneity of the
          check function, rho_tau(w u) = w rho_tau(u): a weighted fit is an unweighted fit on (w y, w X)
  xgbq    XGBoost with the quantile (pinball) objective, all levels in one multi-output model
  qrf     quantile regression forest (Meinshausen, 2006; package quantile-forest); the weights enter both the tree
          growing and the weighted leaf quantiles
  aft     XGBoost accelerated failure time model: log Y = f(x) + sigma Z, trained on observed AND censored orders
          (interval labels [c, inf) for orders not delivered by the origin); the quantiles follow from the
          distribution of Z. The distribution and sigma are chosen inside each origin.

Variants: *_plain uses only the orders delivered by the origin, unweighted (common practice); *_ipcw weights them by
1 / G(Y) (see data/censoring.py). aft uses the censored orders directly; aft_naive drops them.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from quantile_forest import RandomForestQuantileRegressor
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

# A category of the evaluation or calibration orders that is absent from the training orders (a new seller state or
# route) is mapped to the infrequent category by design (handle_unknown="infrequent_if_exist"); scikit-learn warns at
# every such transform. The warning carries no information here and floods the log, so it is silenced (no effect on
# any number).
warnings.filterwarnings("ignore", message="Found unknown categories in columns", category=UserWarning)


def _slog(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def weighted_quantile(v, q, w=None):
    """Quantiles of v at levels q (array) with weights w (left-continuous inverse of the weighted ECDF)."""
    v = np.asarray(v, float); q = np.atleast_1d(q)
    if len(v) == 0:
        return np.full(len(q), np.nan)
    w = np.ones(len(v)) if w is None else np.asarray(w, float)
    o = np.argsort(v, kind="stable"); v, w = v[o], w[o]
    cw = np.cumsum(w) / w.sum()
    idx = np.searchsorted(cw, q - 1e-12, side="left")
    return v[np.minimum(idx, len(v) - 1)]


def rearrange(Q):
    return np.sort(Q, axis=1)


# ----------------------------------------------------------------------------------------------- benchmark
class RouteTable:
    name = "route"

    def __init__(self, taus, shrink_m=50):
        self.taus, self.m = np.asarray(taus), shrink_m

    def fit(self, df, logy, w=None):
        w = np.ones(len(df)) if w is None else w
        d = pd.DataFrame({"route": df["route"].to_numpy(), "cs": df["customer_state"].to_numpy(), "y": logy, "w": w})
        self.glob = weighted_quantile(d["y"], self.taus, d["w"])
        self.cs = {}
        for k, g in d.groupby("cs"):
            n = len(g)
            self.cs[k] = (n * weighted_quantile(g["y"], self.taus, g["w"]) + self.m * self.glob) / (n + self.m)
        self.rt = {}
        for k, g in d.groupby("route"):
            n = len(g); par = self.cs.get(g["cs"].iloc[0], self.glob)
            self.rt[k] = (n * weighted_quantile(g["y"], self.taus, g["w"]) + self.m * par) / (n + self.m)
        return self

    def predict(self, df):
        out = np.empty((len(df), len(self.taus)))
        for i, (r, c) in enumerate(zip(df["route"].to_numpy(), df["customer_state"].to_numpy())):
            out[i] = self.rt.get(r, self.cs.get(c, self.glob))
        return rearrange(out)


# ----------------------------------------------------------------------------------------------- linear
class LinearQR:
    name = "lqr"

    def __init__(self, num, cat, taus, max_iter=2000, min_frequency=200):
        self.num, self.cat, self.taus = num, cat, taus
        numt = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                         ("log", FunctionTransformer(_slog)), ("sc", StandardScaler())])
        catt = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency, drop="first",
                             sparse_output=False)
        self.pre = ColumnTransformer([("n", numt, num), ("c", catt, cat)])
        self.max_iter = max_iter

    def fit(self, df, logy, w=None):
        X = self.pre.fit_transform(df[self.num + self.cat])
        X = np.column_stack([np.ones(len(X)), X])
        keep = X.std(0) > 0; keep[0] = True
        self.keep = keep; X = X[:, keep]
        w = np.ones(len(X)) if w is None else np.asarray(w, float)
        Xw, yw = X * w[:, None], logy * w
        self.beta = []
        for t in self.taus:
            res = sm.QuantReg(yw, Xw).fit(q=t, max_iter=self.max_iter, p_tol=1e-5)
            self.beta.append(res.params)
        self.beta = np.column_stack(self.beta)
        return self

    def predict(self, df):
        X = self.pre.transform(df[self.num + self.cat])
        X = np.column_stack([np.ones(len(X)), X])[:, self.keep]
        return rearrange(X @ self.beta)


# ----------------------------------------------------------------------------------------------- trees
def _xgb_frame(df, num, cat, vocab):
    X = df[num].astype(float).copy()
    for c in cat:
        X[c] = pd.Categorical(df[c], categories=vocab[c])
    return X


class XGBQuantile:
    name = "xgbq"

    def __init__(self, num, cat, vocab, taus, seed=42, n_jobs=2, **params):
        self.num, self.cat, self.vocab, self.taus = num, cat, vocab, np.asarray(taus)
        self.model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=self.taus, tree_method="hist",
                                      enable_categorical=True, max_cat_to_onehot=1, random_state=seed,
                                      n_jobs=n_jobs, **params)

    def fit(self, df, logy, w=None):
        self.model.fit(_xgb_frame(df, self.num, self.cat, self.vocab), logy, sample_weight=w)
        return self

    def predict(self, df):
        return rearrange(np.asarray(self.model.predict(_xgb_frame(df, self.num, self.cat, self.vocab))).reshape(len(df), -1))


class QRF:
    name = "qrf"

    def __init__(self, num, cat, taus, seed=42, n_jobs=2, **params):
        self.num, self.cat, self.taus, self.seed = num, cat, list(taus), seed
        self.pre = ColumnTransformer([("n", SimpleImputer(strategy="median", add_indicator=True), num),
                                      ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat)])
        self.model = RandomForestQuantileRegressor(random_state=seed, n_jobs=n_jobs, **params)

    def fit(self, df, logy, w=None):
        X = self.pre.fit_transform(df[self.num + self.cat])
        self.model.fit(X, logy, sample_weight=w)     # quantile-forest uses the weights in the leaf quantiles too
        return self

    def predict(self, df):
        X = self.pre.transform(df[self.num + self.cat])
        return rearrange(np.asarray(self.model.predict(X, quantiles=self.taus)).reshape(len(df), -1))


def aft_z(dist, taus):
    taus = np.asarray(taus, float)
    if dist == "normal":
        return stats.norm.ppf(taus)
    if dist == "logistic":
        return np.log(taus / (1 - taus))
    if dist == "extreme":                       # XGBoost's extreme-value (minimum Gumbel) distribution
        return np.log(-np.log(1 - taus))
    raise ValueError(dist)


class XGBAFT:
    name = "aft"

    def __init__(self, num, cat, vocab, taus, dist="normal", scale=1.0, seed=42, n_jobs=2, n_estimators=400, **params):
        self.num, self.cat, self.vocab, self.taus = num, cat, vocab, np.asarray(taus)
        self.dist, self.scale, self.rounds = dist, scale, n_estimators
        self.params = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution=dist,
                           aft_loss_distribution_scale=scale, tree_method="hist", max_cat_to_onehot=1, seed=seed,
                           nthread=n_jobs, **params)

    def fit(self, df, lower, upper):
        """lower = upper = Y for observed orders; lower = c, upper = inf for censored orders (original scale)."""
        dm = xgb.DMatrix(_xgb_frame(df, self.num, self.cat, self.vocab), enable_categorical=True)
        dm.set_float_info("label_lower_bound", np.asarray(lower, float))
        dm.set_float_info("label_upper_bound", np.asarray(upper, float))
        self.booster = xgb.train(self.params, dm, num_boost_round=self.rounds)
        return self

    def predict(self, df):
        dm = xgb.DMatrix(_xgb_frame(df, self.num, self.cat, self.vocab), enable_categorical=True)
        mu = self.booster.predict(dm, output_margin=True)
        return rearrange(mu[:, None] + self.scale * aft_z(self.dist, self.taus)[None, :])
