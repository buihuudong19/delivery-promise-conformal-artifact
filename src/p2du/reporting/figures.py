"""Figures of the P2DU article (vector PDF and 600-dpi PNG), drawn at their printed size in the two-column Elsevier
layout (column 3.30 in, text width 6.84 in), so that the font sizes are the printed sizes (6.5 to 8 pt). Figures 2, 3,
4, 5, 9 and 11 are drawn for one column (panels stacked); the others need the full text width (a wide diagram, a dense
time axis, three panels side by side or insets).

The visual style (fonts, Okabe-Ito palette, constrained layout, panel titles, save routine) is reused from
src/leakaware/reporting/figures.py of the first article (leakaware v1.0.2, MIT); the figures themselves are new.
Every number drawn comes from a table in outputs/tables written by the pipeline.

  fig01_framework            decision pipeline of the study and where each RQ sits
  fig02_data_context         monthly delivery time and promise; deviation S by region
  fig03_protocol_censoring   rolling origins with calibration windows; censoring of the calibration orders by age
  fig04_rq1_efficiency       coverage against mean promise length (models, benchmarks, Olist)
  fig05_rq1_censoring        coverage and promise length under naive, IPCW and oracle calibration
  fig06_rq1_time             coverage by window (split, Mondrian, ACI) and the ACI level alpha_t
  fig07_rq1_regions          coverage and promise length by region
  fig08_rq1_quantile_calib   empirical coverage of the uncorrected quantiles and pinball loss
  fig09_rq2_response         low-rating rate by deviation S: crude, g-formula curve, AIPW bins
  fig10_rq2_s_vs_d           low-rating surface over promise and delivery time; S versus D model comparison
  fig11_rq2_survey           survey timing: ratings given before and after arrival
  fig12_rq2_effects          AIPW risk ratios with E-values, by region, and RDD estimates
  fig13_rq3_frontier         frontier of promise length against the low-rating and late rates
  fig14_rq3_anatomy          rho_i(P) of example orders, promises by region, window-level gains
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from ..evaluation.metrics import moving_block_bootstrap_mean

COL, FULL = 3.30, 6.84
INK, INK2, MUTED, GRID, LIGHT = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#e3e3e3", "#f3f3f3"
OI = dict(blue="#0072B2", vermillion="#D55E00", green="#009E73", orange="#E69F00", sky="#56B4E9",
          purple="#CC79A7", yellow="#F0E442", black="#000000")
KINDS = ("route", "lqr", "xgbq", "qrf", "aft")
K_COLOR = {"route": "#6b6b6b", "lqr": OI["vermillion"], "xgbq": OI["blue"], "qrf": OI["green"], "aft": OI["purple"],
           "ens": OI["orange"], "olist": INK}
K_MARK = {"route": "D", "lqr": "s", "xgbq": "o", "qrf": "^", "aft": "v", "ens": "P", "olist": "*"}
K_LABEL = {"route": "Route table", "lqr": "Linear QR", "xgbq": "XGBoost QR", "qrf": "Quantile forest",
           "aft": "XGBoost AFT", "ens": "Quantile average", "olist": "Olist promise"}
M_COLOR = {"split_naive": OI["vermillion"], "split_ipcw": OI["blue"], "split_oracle": MUTED,
           "mondrian_ipcw": OI["orange"], "aci_ipcw": OI["green"], "mondrian_aci_ipcw": OI["purple"], "raw": INK2}
M_LABEL = {"raw": "Uncorrected", "split_naive": "Split, naive", "split_ipcw": "Split, IPCW",
           "split_oracle": "Split, oracle", "mondrian_naive": "Mondrian, naive", "mondrian_ipcw": "Mondrian, IPCW",
           "aci_ipcw": "ACI, IPCW", "mondrian_aci_ipcw": "Mondrian ACI, IPCW"}
P_COLOR = {"uniform": MUTED, "route": "#6b6b6b", "cpd_quantile": OI["sky"], "late_optimal": OI["blue"],
           "optimal": OI["vermillion"],
           "olist_shift": OI["orange"], "olist": INK}
P_LABEL = {"uniform": "Uniform promise", "route": "Route quantile", "cpd_quantile": "Coverage policy (CPD quantile)",
           "late_optimal": "Late-rate optimal",
           "optimal": "Rating-aware optimal", "olist_shift": "Olist promise shifted", "olist": "Olist promise"}
REG_ORDER = ["Sudeste", "Sul", "Centro-Oeste", "Nordeste", "Norte"]
REG_EN = {"Sudeste": "Southeast", "Sul": "South", "Centro-Oeste": "Centre-West", "Nordeste": "Northeast",
          "Norte": "North"}
EVENTS = [("2017-11-20", "2017-12-08", "Black Friday"), ("2018-02-10", "2018-02-18", "Carnival"),
          ("2018-05-21", "2018-06-04", "Truckers' strike")]


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Inter", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 5,
        "axes.titlecolor": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
        "legend.handletextpad": 0.4, "legend.columnspacing": 1.2, "legend.borderaxespad": 0.2,
        "lines.linewidth": 1.1, "lines.markersize": 3.5, "errorbar.capsize": 0,
        "figure.dpi": 150, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": True, "mathtext.default": "regular"})


def _subplots(*args, **kw):
    fig, axes = plt.subplots(*args, layout="constrained", **kw)
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.015, wspace=0.03, hspace=0.04)
    return fig, axes


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"{name}.{ext}", facecolor="white")
    plt.close(fig)


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5); ax.set_axisbelow(True)


def _panel(ax, i, title=""):
    ax.set_title(f"({'abcdefgh'[i]}) {title}".rstrip())


def _read(out, name):
    p = out / "tables" / name
    return pd.read_csv(p) if p.is_file() else None


def _kind(v):
    return v.split("_")[0]


def _main_variants(cfg):
    """The five models in their censoring-aware versions, plus the quantile average when it is configured."""
    ens = (cfg.get("ensemble") or {}).get("name")
    return ("route_ipcw", "lqr_ipcw", "xgbq_ipcw", "qrf_ipcw", "aft") + ((ens,) if ens else ())


def _boot(x, cfg):
    bs = cfg["bootstrap"]
    m, lo, hi, _ = moving_block_bootstrap_mean(np.asarray(x, float), bs["n_boot"], bs["block_len"], cfg["seed"])
    return m, lo, hi


def _shade_events(ax, label=True):
    for j, (a, b, name) in enumerate(EVENTS):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=LIGHT, zorder=0, lw=0)
        if label:                                   # staggered heights, so that adjacent labels do not overlap;
            lo_, hi_ = ax.get_xlim()                  # a label in the right quarter ends at its event instead
            right = (mdates.date2num(pd.Timestamp(a)) - lo_) / (hi_ - lo_) > 0.75
            if ax.figure.get_figwidth() < 4:          # one-column figure: short names
                name = {"Truckers' strike": "Strike"}.get(name, name)
            ax.text(pd.Timestamp(b if right else a), 1.0 - 0.06 * (j % 2), name, transform=ax.get_xaxis_transform(),
                    fontsize=5.5, color=MUTED, va="top", ha="right" if right else "left")


def _lines_box(ax, lines, loc, xy, align="left", frame=False, fontsize=5.4, color=INK2):
    """Text block of several lines in which single lines can be bold: lines = [(text, bold, colour or None)].
    The headline number of a panel is set in bold, as in the text and the tables."""
    kids = [TextArea(t, textprops=dict(fontsize=fontsize, color=c or color, fontweight="bold" if bd else "normal"))
            for t, bd, c in lines]
    box = AnchoredOffsetbox(loc=loc, child=VPacker(children=kids, align=align, pad=0, sep=1.2), frameon=frame,
                            bbox_to_anchor=xy, bbox_transform=ax.transAxes, borderpad=0, pad=0.3)
    if frame:
        box.patch.set_boxstyle("round,pad=0.3"); box.patch.set_facecolor("white")
        box.patch.set_edgecolor(GRID); box.patch.set_linewidth(0.5)
    ax.add_artist(box)


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


# ----------------------------------------------------------------------------------------------- Fig. 1
def fig_framework(out, cfg):
    fig, ax = plt.subplots(figsize=(FULL, 2.3))
    ax.set_xlim(0, 100); ax.set_ylim(0.5, 37.6); ax.axis("off")

    def box(x, y, w, h, title, body, color, tag=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc="white", ec=color, lw=0.9))
        ax.add_patch(Rectangle((x - 0.25, y + h - 4.2), w + 0.5, 4.45, fc=color, ec="none", alpha=0.15))
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="center", fontsize=6.1, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + (h - 4.2) / 2, body, ha="center", va="center", fontsize=5.5, color=INK2, linespacing=1.4)
        if tag:
            ax.text(x + w - 0.6, y + 0.9, tag, ha="right", va="bottom", fontsize=5.6, fontweight="bold", color=color)

    def arrow(x0, y0, x1, y1, text=None, dy=1.8, dx=0.0):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0))
        if text:
            ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=6, color=INK2)

    y0, h = 19, 18
    box(0.4, y0, 14.6, h, "Order at purchase", "basket, price, freight\nroute, distance\nas-of seller and route\nhistories, platform load", MUTED)
    box(19.8, y0, 16.6, h, "Delivery-time model", "route table, linear QR\nXGBoost QR, QRF\nXGBoost AFT\nplain / IPCW / censored", OI["blue"], "RQ1")
    box(41.6, y0, 17.6, h, "One-sided conformal", "split, Mondrian (region),\nadaptive (ACI)\nIPCW masses for the\ncensored calibration set", OI["sky"], "RQ1")
    box(64.8, y0, 15.2, h, "Promise policy", "minimise $\\Sigma_i\\,\\rho_i(P_i)$\nsubject to $\\frac{1}{n}\\Sigma_i P_i \\leq L$\nLagrangian sweep", OI["vermillion"], "RQ3")
    box(85.4, y0, 14.2, h, "Outcomes", "late rate\nlow-rating rate\nmean promise", INK2)
    ym = y0 + (h - 4.2) / 2
    arrow(15.3, ym, 19.5, ym); arrow(36.7, ym, 41.3, ym, "$\\hat q_\\tau(x)$"); arrow(59.5, ym, 64.5, ym, "$\\hat p_i, U(x)$")
    arrow(80.3, ym, 85.1, ym, "$P_i$")
    box(33, 1.2, 56, 12.5, "Customer response to the deviation $S = D - P$",
        "low-rating probability $r(s)$: AIPW by deviation bins, monotone g-formula curve,\n"
        "regression discontinuity at $S = 0$, E-values, survey timing", OI["orange"], "RQ2")
    arrow(72.4, 13.9, 72.4, 18.8, "$r(s)$", dy=-0.6, dx=2.6)
    ax.text(0.6, 9.5, "Decision at purchase: the delivery\ndate shown to the customer.\n"
            "Evaluated on 30 rolling origins\n(2017-06 to 2018-07), Olist data.", fontsize=5.8, color=INK2, va="center",
            linespacing=1.4)
    _save(fig, out, "fig01_framework")


# ----------------------------------------------------------------------------------------------- Fig. 2
def fig_data_context(out, cfg):
    Mo = _read(out, "eda_monthly.csv"); S = _read(out, "eda_S_by_region.csv")
    if Mo is None or S is None:
        return
    Mo["month"] = pd.to_datetime(Mo["month"]); Mo = Mo[(Mo["n"] >= 500)]
    fig, (a, b) = _subplots(2, 1, figsize=(COL, 4.5))
    a.plot(Mo["month"], Mo["P_mean"], color=INK, marker="*", ms=4, label="Olist promise, mean")
    a.plot(Mo["month"], Mo["Y_p90"], color=OI["vermillion"], marker="o", ms=2.5, label="Delivery time, 90th pct.")
    a.plot(Mo["month"], Mo["Y_mean"], color=OI["blue"], marker="o", ms=2.5, label="Delivery time, mean")
    a.set_ylabel("Days from purchase"); _date_axis(a); _grid(a); _shade_events(a)
    a2 = a.twinx(); a2.bar(Mo["month"], Mo["late"] * 100, width=20, color=OI["orange"], alpha=0.35, label="Late rate (%)")
    a2.set_ylabel("Late against the promise (%)", color=OI["orange"]); a2.spines["right"].set_visible(True)
    a2.set_ylim(0, max(25, Mo["late"].max() * 130)); a.set_zorder(a2.get_zorder() + 1); a.patch.set_visible(False)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.set_ylim(0, Mo[["P_mean", "Y_p90"]].to_numpy().max() * 1.6)
    a.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, bbox_to_anchor=(0, 0.88), fontsize=5.8, columnspacing=0.8)
    _panel(a, 0, "Promise and delivery time by purchase month")
    for g in REG_ORDER:
        x = S[S.customer_region == g].sort_values("Sc")
        p = x["n"] / x["n"].sum()
        b.plot(x["Sc"], p * 100, lw=1.0, label=REG_EN[g], color={"Sudeste": OI["blue"], "Sul": OI["sky"],
               "Centro-Oeste": OI["green"], "Nordeste": OI["orange"], "Norte": OI["vermillion"]}[g])
    b.axvline(0.5, color=INK2, lw=0.6, ls="--"); b.text(1.2, b.get_ylim()[1] * 0.92, "late", fontsize=6, color=INK2)
    b.set_xlabel("Deviation S = D - P (days; clipped at -40 and 20)"); b.set_ylabel("Share of delivered orders (%)")
    b.legend(loc="upper left", fontsize=6); _grid(b); _panel(b, 1, "Deviation from the promise by region")
    _save(fig, out, "fig02_data_context")


# ----------------------------------------------------------------------------------------------- Fig. 3
def fig_protocol_censoring(out, cfg):
    A = _read(out, "eda_censoring_by_age.csv")
    if A is None:
        return
    fig, (a, b) = _subplots(2, 1, figsize=(COL, 4.0), gridspec_kw=dict(height_ratios=[1, 1.25]))
    K, H, step = cfg["rolling"]["calib_days"], cfg["rolling"]["horizon_days"], cfg["rolling"]["step_days"]
    for i in range(4):
        o = 150 + i * step; y = 3 - i
        a.barh(y, o - K, left=0, height=0.55, color=OI["blue"], alpha=0.35, lw=0)
        a.barh(y, K, left=o - K, height=0.55, color=OI["sky"], alpha=0.9, lw=0, hatch="////", edgecolor="white")
        a.barh(y, H, left=o, height=0.55, color=OI["vermillion"], alpha=0.85, lw=0)
        a.plot([o, o], [y - 0.38, y + 0.38], color=INK, lw=0.9)
        a.text(o + H + 2, y, f"origin {i + 1}", va="center", fontsize=6, color=INK2)
    a.set_yticks([]); a.spines["left"].set_visible(False); a.set_xlim(0, 250)
    a.set_xlabel("Purchase time (days, schematic)", labelpad=1)
    hs = [Rectangle((0, 0), 1, 1, color=OI["blue"], alpha=0.35), Rectangle((0, 0), 1, 1, fc=OI["sky"], hatch="////", ec="white"),
          Rectangle((0, 0), 1, 1, color=OI["vermillion"], alpha=0.85)]
    a.legend(hs, ["Training: bought before o - 42 d", "Calibration: last 42 d (censored)", "Evaluation: next 14 d"],
             loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=5.8, columnspacing=0.8)
    _panel(a, 0, "Rolling origins every 14 days (30 origins)")
    g = A.groupby("age").apply(lambda x: np.average(x["share_undelivered"], weights=x["n"]), include_groups=False)
    lo = A.groupby("age")["share_undelivered"].quantile(0.1); hi = A.groupby("age")["share_undelivered"].quantile(0.9)
    b.fill_between(g.index, lo * 100, hi * 100, color=OI["sky"], alpha=0.35, lw=0, label="10th-90th pct. over origins")
    b.plot(g.index, g * 100, color=OI["blue"], label="Mean over origins")
    b.set_xlabel("Purchase age at the origin (days)"); b.set_ylabel("Not yet delivered (%)")
    b.set_xlim(0, K); b.set_ylim(0, 102); b.legend(loc="upper right"); _grid(b)
    _panel(b, 1, "Censoring of the calibration orders")
    _save(fig, out, "fig03_protocol_censoring")


# ----------------------------------------------------------------------------------------------- Fig. 4
def fig_rq1_efficiency(out, cfg):
    S = _read(out, "rq1_summary.csv")
    if S is None:
        return
    S = S[S.region == "all"]
    # two panels side by side across the text width: the figure is shown in the supplementary material
    fig, axes = _subplots(1, 2, figsize=(FULL, 2.55), sharey=False)
    for ax, (meth, title) in zip(axes, (("split_ipcw", "After split conformal (IPCW)"), ("raw", "Uncorrected quantiles"))):
        for v in _main_variants(cfg):
            x = S[(S.variant == v) & (S.method == meth)].sort_values("alpha")
            k = _kind(v)
            ax.plot(x["mean_promise"], x["coverage"] * 100, color=K_COLOR[k], marker=K_MARK[k], ms=3.6, lw=0.9,
                    label=K_LABEL[k])
        o = S[S.variant == "olist"].iloc[0]
        ax.scatter([o["mean_promise"]], [o["coverage"] * 100], marker="*", s=70, color=INK, zorder=5, label="Olist promise")
        for a in cfg["alphas"]:
            ax.axhline((1 - a) * 100, color=MUTED, lw=0.5, ls=":")
            ax.text(0.01, (1 - a) * 100 + 0.3, f"target {int((1 - a) * 100)}%", fontsize=5.5, color=MUTED,
                    transform=ax.get_yaxis_transform())
        ax.set_xlabel("Mean promise (days)"); ax.set_ylabel("Coverage = promises kept (%)"); _grid(ax, "both")
        _panel(ax, list(axes).index(ax), title)
    axes[1].legend(loc="lower right", fontsize=6)
    C = _read(out, "contrasts_rq1.csv")
    if C is not None:
        F = C[C.family.str.startswith("RQ1-F")].set_index("contrast")
        lines = [("Mean promise at Olist's coverage,", False, None), ("minus Olist (days, 95% CI):", False, None)]
        ens = (cfg.get("ensemble") or {}).get("name")
        for v in _main_variants(cfg):
            key = f"{v} - Olist"
            if key in F.index:
                r = F.loc[key]
                lines.append((f"{K_LABEL[_kind(v)]}: {r['mean']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]", v == ens, None))
        _lines_box(axes[0], lines, "lower right", (0.98, 0.03), align="right", frame=True, fontsize=5.2)
    _save(fig, out, "fig04_rq1_efficiency")


# ----------------------------------------------------------------------------------------------- Fig. 5
def fig_rq1_censoring(out, cfg):
    M = _read(out, "rq1_window_metrics.csv")
    if M is None:
        return
    M = M[(M.region == "all") & np.isclose(M.alpha, 0.10)]
    vs = ["route_plain", "route_ipcw", "lqr_plain", "lqr_ipcw", "xgbq_plain", "xgbq_ipcw", "qrf_plain", "qrf_ipcw",
          "aft_naive", "aft"]
    lab = {v: f"{K_LABEL[_kind(v)]}, " + ("censored labels" if v == "aft" else "delivered only" if v == "aft_naive" else
                                          "IPCW" if v.endswith("ipcw") else "plain") for v in vs}
    meths = ["split_naive", "split_ipcw", "split_oracle"]
    fig, (a, b) = _subplots(2, 1, figsize=(COL, 5.4), sharey=True)
    y = np.arange(len(vs))[::-1]
    for j, m in enumerate(meths):
        for i, v in enumerate(vs):
            x = M[(M.variant == v) & (M.method == m)].sort_values("k")
            if x.empty:
                continue
            for ax, col, sc in ((a, "coverage", 100), (b, "mean_promise", 1)):
                mu, lo, hi = _boot(x[col] * sc, cfg)
                yy = y[i] + (1 - j) * 0.22
                ax.hlines(yy, lo, hi, color=M_COLOR[m], lw=0.9)
                ax.scatter([mu], [yy], s=11, color=M_COLOR[m], marker="o", zorder=3,
                           label=M_LABEL[m] if i == 0 and ax is a else None)
    a.axvline(90, color=INK2, lw=0.6, ls="--"); a.set_yticks(y); a.set_yticklabels([lab[v] for v in vs])
    b.tick_params(axis="y", labelleft=True)
    a.set_xlabel("Coverage at target 90% (%)"); b.set_xlabel("Mean promise (days)")
    _grid(a, "x"); _grid(b, "x")
    fig.legend(*a.get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=6.2)
    _panel(a, 0, "Coverage by calibration weighting"); _panel(b, 1, "Promise length")
    _save(fig, out, "fig05_rq1_censoring")


# ----------------------------------------------------------------------------------------------- Fig. 6
def fig_rq1_time(out, cfg):
    M = _read(out, "rq1_window_metrics.csv"); T = _read(out, "rq1_aci_alpha_trajectory.csv")
    if M is None:
        return
    v = cfg["rq3"]["model"]
    M = M[(M.region == "all")]; M["origin"] = pd.to_datetime(M["origin"])
    fig, (a, b) = _subplots(2, 1, figsize=(FULL, 3.3), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    for m in ("split_naive", "split_ipcw", "mondrian_ipcw", "aci_ipcw"):
        x = M[(M.variant == v) & (M.method == m) & np.isclose(M.alpha, 0.10)].sort_values("origin")
        a.plot(x["origin"] + pd.Timedelta(days=7), x["coverage"] * 100, color=M_COLOR[m], marker="o", ms=2.4, lw=0.9,
               label=M_LABEL[m])
    o = M[M.variant == "olist"].sort_values("origin")
    a.plot(o["origin"] + pd.Timedelta(days=7), o["coverage"] * 100, color=INK, ls="--", lw=0.8, label="Olist promise")
    a.axhline(90, color=INK2, lw=0.6, ls=":"); a.set_ylabel("Coverage (%)"); _grid(a)
    a.set_ylim(min(60, a.get_ylim()[0]), 101)
    Wt = _read(out, "rq1_window_types.csv")
    if Wt is not None:
        for _, w in Wt[Wt["stress"]].iterrows():
            o0 = pd.Timestamp(w["origin"])
            for ax in (a, b):
                ax.axvspan(o0, o0 + pd.Timedelta(days=cfg["rolling"]["horizon_days"]), color="#fde9d9", lw=0, zorder=0)
    for d0, lab in (("2017-11-24", "Black Friday"), ("2018-02-12", "Carnival backlog"), ("2018-05-21", "Truckers' strike")):
        a.annotate(lab, (pd.Timestamp(d0), 0.02), xycoords=("data", "axes fraction"), fontsize=5.5, color=INK2,
                   ha="center", va="bottom")
    for ax in (a, b):
        ax.axvline(pd.Timestamp("2018-01-01"), color=INK2, lw=0.6, ls="-.")
    b.text(pd.Timestamp("2017-12-28"), 0.97, "2017: ACI step chosen  ", transform=b.get_xaxis_transform(), ha="right",
           va="top", fontsize=5.5, color=INK2)
    b.text(pd.Timestamp("2018-01-05"), 0.97, "  2018: test period", transform=b.get_xaxis_transform(), ha="left",
           va="top", fontsize=5.5, color=INK2)
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=5, fontsize=6)
    _panel(a, 0, f"Coverage by evaluation window, {K_LABEL[_kind(v)]}, target 90%")
    if T is not None and not T.empty:
        t = T[(T.variant == v) & np.isclose(T.alpha, 0.10)]
        b.plot(pd.to_datetime(t["day"]), t["alpha_t"], color=OI["green"], lw=0.9)
        b.axhline(0.10, color=INK2, lw=0.6, ls=":")
    b.set_ylabel(r"ACI level $\alpha_t$"); _date_axis(b); _grid(b)
    _panel(b, 1, "Adaptive level of ACI (daily batches of delayed feedback); shaded: stress windows")
    _save(fig, out, "fig06_rq1_time")


# ----------------------------------------------------------------------------------------------- Fig. 7
def fig_rq1_regions(out, cfg):
    M = _read(out, "rq1_window_metrics.csv")
    if M is None:
        return
    v = cfg["rq3"]["model"]
    M = M[np.isclose(M.alpha.fillna(-1), 0.10) | (M.variant == "olist")]
    meths = ["split_naive", "split_ipcw", "mondrian_ipcw", "aci_ipcw", "mondrian_aci_ipcw"]
    fig, (a, b) = _subplots(1, 2, figsize=(FULL, 2.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    y = np.arange(len(REG_ORDER))[::-1]
    off = (2 - np.arange(len(meths))) * 0.14
    for j, m in enumerate(meths):
        for i, g in enumerate(REG_ORDER):
            x = M[(M.variant == v) & (M.method == m) & (M.region == g)].sort_values("k")
            if x.empty:
                continue
            mu, lo, hi = _boot(x["coverage"] * 100, cfg)
            a.hlines(y[i] + off[j], lo, hi, color=M_COLOR[m], lw=0.9)
            a.scatter([mu], [y[i] + off[j]], s=10, color=M_COLOR[m], zorder=3, label=M_LABEL[m] if i == 0 else None)
    a.axvline(90, color=INK2, lw=0.6, ls="--"); a.set_yticks(y); a.set_yticklabels([REG_EN[g] for g in REG_ORDER])
    a.set_xlabel("Coverage at target 90% (%)"); _grid(a, "x")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=5.8)
    _panel(a, 0, f"Coverage by customer region, {K_LABEL[_kind(v)]}")
    w = 0.26
    for j, (vv, m, lab, col) in enumerate((("olist", "actual", "Olist promise", INK),
                                           (v, "split_ipcw", "Split, IPCW", OI["blue"]),
                                           (v, "mondrian_ipcw", "Mondrian, IPCW", OI["orange"]))):
        vals = [M[(M.variant == vv) & (M.method == m) & (M.region == g)]["mean_promise"].mean() for g in REG_ORDER]
        b.barh(y + (1 - j) * w, vals, height=w, color=col, alpha=0.85, label=lab)
    b.set_yticks(y); b.set_yticklabels([REG_EN[g] for g in REG_ORDER]); b.set_xlabel("Mean promise (days)")
    _grid(b, "x"); b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=5.8)
    _panel(b, 1, "Promise length by region")
    _save(fig, out, "fig07_rq1_regions")


# ----------------------------------------------------------------------------------------------- Fig. 8
def fig_rq1_quantile_calib(out, cfg):
    L = _read(out, "rq1_pinball.csv")
    if L is None:
        return
    fig, (a, b) = _subplots(1, 2, figsize=(FULL, 2.3))
    vs = list(_main_variants(cfg))
    for v in vs:
        x = L[L.variant == v].groupby("tau")["coverage_raw"].mean()
        k = _kind(v)
        a.plot(x.index * 100, (x.values - x.index) * 100, color=K_COLOR[k], marker=K_MARK[k], ms=3.4, label=K_LABEL[k])
    a.axhline(0, color=INK2, lw=0.6, ls="--"); a.set_xlabel(r"Nominal level $\tau$ (%)")
    a.set_ylabel("Empirical minus nominal coverage (pp)"); _grid(a); a.legend(loc="best", fontsize=6)
    _panel(a, 0, "Calibration of the uncorrected quantiles")
    allv = ["route_plain", "route_ipcw", "lqr_plain", "lqr_ipcw", "xgbq_plain", "xgbq_ipcw", "qrf_plain", "qrf_ipcw",
            "aft_naive", "aft"] + [v for v in _main_variants(cfg)[5:] if (L.variant == v).any()]
    ref = L[(L.variant == "route_ipcw") & np.isclose(L.tau, 0.9)].set_index("k")["pinball"]
    yy = np.arange(len(allv))[::-1]
    for i, v in enumerate(allv):
        x = L[(L.variant == v) & np.isclose(L.tau, 0.9)].set_index("k")["pinball"]
        d = 100 * (x / ref.reindex(x.index) - 1)
        mu, lo, hi = _boot(d.to_numpy(), cfg)
        k = _kind(v); filled = v.endswith("ipcw") or v == "aft"
        b.hlines(yy[i], lo, hi, color=K_COLOR[k], lw=0.9)
        b.scatter([mu], [yy[i]], s=14, marker=K_MARK[k], facecolor=K_COLOR[k] if filled else "white",
                  edgecolor=K_COLOR[k], zorder=3)
    b.axvline(0, color=INK2, lw=0.6, ls="--"); b.set_yticks(yy)
    b.set_yticklabels([f"{K_LABEL[_kind(v)]} ({'censored' if v == 'aft' else 'naive' if v == 'aft_naive' else v.split('_')[1]})" for v in allv])
    b.set_xlabel(r"Pinball loss at $\tau$ = 0.9 relative to the route table (%)"); _grid(b, "x")
    _panel(b, 1, "Accuracy of the 90% quantile")
    _save(fig, out, "fig08_rq1_quantile_calib")


# ----------------------------------------------------------------------------------------------- Fig. 9
def fig_rq2_response(out, cfg):
    C = _read(out, "rq2_crude_curve.csv"); G = _read(out, "rq2_response_curve.csv"); A = _read(out, "rq2_aipw_bins.csv")
    if C is None or G is None:
        return
    fig, ax = _subplots(1, 1, figsize=(COL, 2.5))
    C = C[(C.S >= -35) & (C.S <= 20)]
    ax.vlines(C["S"], C["lo"] * 100, C["hi"] * 100, color=MUTED, lw=0.6)
    ax.scatter(C["S"], C["rate"] * 100, s=7, color=MUTED, zorder=2, label="Crude rate (Wilson 95% CI)")
    g = G[(G.group == "all") & (G.S >= -35) & (G.S <= 20)]
    ax.plot(g["S"], g["r"] * 100, color=OI["blue"], lw=1.4, label="Adjusted curve r(s), g-formula")
    if A is not None:
        e = cfg["rq2"]["bins"]
        a = A[A.learner == "xgb"]
        for _, r in a.iterrows():
            lo, hi = max(e[int(r["bin"])] + 1, -35), min(e[int(r["bin"]) + 1], 20)
            ax.add_patch(Rectangle((lo - 0.5, r["risk_lo"] * 100), hi - lo + 1, (r["risk_hi"] - r["risk_lo"]) * 100,
                                   color=OI["vermillion"], alpha=0.18, lw=0))
            ax.hlines(r["risk"] * 100, lo - 0.5, hi + 0.5, color=OI["vermillion"], lw=1.2,
                      label="AIPW risk by deviation bin" if r["bin"] == 0 else None)
    ax.axvline(0.5, color=INK2, lw=0.6, ls="--"); ax.text(1.2, 95, "late", fontsize=6, color=INK2)
    ax.set_xlabel("Deviation from the promise S = D - P (days)"); ax.set_ylabel("Low rating (1-2 stars, %)")
    ax.set_ylim(0, 100); _grid(ax); ax.legend(loc="upper left", fontsize=5.8)
    _save(fig, out, "fig09_rq2_response")


# ----------------------------------------------------------------------------------------------- Fig. 10
def fig_rq2_s_vs_d(out, cfg):
    Sf = _read(out, "rq2_surface_P_D.csv"); V = _read(out, "rq2_s_versus_d.csv")
    if Sf is None or V is None:
        return
    fig, (a, b) = _subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    pb = list(dict.fromkeys(Sf["pb"])); db = list(dict.fromkeys(Sf["db"]))
    Z = Sf.pivot_table(index="pb", columns="db", values="rate").reindex(index=pb, columns=db)
    N = Sf.pivot_table(index="pb", columns="db", values="n").reindex(index=pb, columns=db)
    Z = Z.where(N >= 30)
    im = a.imshow(Z.to_numpy() * 100, origin="lower", aspect="auto", cmap="YlOrRd", vmin=0, vmax=80)
    a.set_xticks(range(len(db))); a.set_xticklabels([s.replace("[", "").replace(")", "").replace(", ", "-") for s in db], rotation=0)
    a.set_yticks(range(len(pb))); a.set_yticklabels([s.replace("[", "").replace(")", "").replace(", ", "-") for s in pb])
    for i in range(len(pb)):
        for j in range(len(db)):
            if np.isfinite(Z.iloc[i, j]):
                a.text(j, i, f"{Z.iloc[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.2,
                       color="white" if Z.iloc[i, j] > 0.45 else INK)
    a.set_xlabel("Delivery time D (days)"); a.set_ylabel("Olist promise P (days)")
    cb = fig.colorbar(im, ax=a, shrink=0.85, pad=0.01); cb.set_label("Low rating (%)"); cb.outline.set_linewidth(0.4)
    _panel(a, 0, "Low-rating rate by promise and delivery time")
    order = ["g(P)", "f(D)", "f(D) + g(P)", "f(S)", "f(S) + g(P)"]
    V = V.set_index("model").reindex(order)
    cols = [MUTED, OI["sky"], OI["sky"], OI["blue"], OI["blue"]]
    b.barh(range(len(order))[::-1], V["mcfadden_r2_cv"] * 100, color=cols, height=0.6)
    for i, (m, r) in enumerate(V.iterrows()):
        b.text(r["mcfadden_r2_cv"] * 100 + 0.2, len(order) - 1 - i, f"{r['cv_logloss']:.4f}", va="center", fontsize=5.8, color=INK2)
    b.set_yticks(range(len(order))[::-1]); b.set_yticklabels(order)
    b.set_xlabel("Cross-validated McFadden pseudo-R$^2$ (%)\n(labels: cross-validated log loss)"); _grid(b, "x")
    _panel(b, 1, "Deviation versus absolute delivery time")
    _save(fig, out, "fig10_rq2_s_vs_d")


# ----------------------------------------------------------------------------------------------- Fig. 11
def fig_rq2_survey(out, cfg):
    C = _read(out, "rq2_crude_curve_by_answer_timing.csv"); Ctot = _read(out, "rq2_crude_curve.csv")
    if C is None:
        return
    fig, (a, b) = _subplots(2, 1, figsize=(COL, 4.2))
    for flag, col, lab in ((0, OI["blue"], "Answered after arrival"), (1, OI["vermillion"], "Answered before arrival")):
        x = C[(C.answered_before_delivery == flag) & (C.n >= 25) & (C.S >= -30) & (C.S <= 20)]
        a.fill_between(x["S"], x["lo"] * 100, x["hi"] * 100, color=col, alpha=0.2, lw=0)
        a.plot(x["S"], x["rate"] * 100, color=col, marker="o", ms=2.2, lw=0.9, label=lab)
    a.axvline(0.5, color=INK2, lw=0.6, ls="--"); a.set_xlabel("Deviation S (days)"); a.set_ylabel("Low rating (%)")
    a.legend(loc="upper left", fontsize=6); _grid(a); _panel(a, 0, "Low-rating rate by survey timing")
    piv = C.pivot_table(index="S", columns="answered_before_delivery", values="n").fillna(0)
    sh = piv.get(1, 0) / piv.sum(1)
    sh = sh[(sh.index >= -10) & (sh.index <= 20)]
    b.bar(sh.index, sh.values * 100, color=OI["vermillion"], alpha=0.8, width=0.8)
    b.axvline(0.5, color=INK2, lw=0.6, ls="--"); b.set_xlabel("Deviation S (days)")
    b.set_ylabel("Answered before arrival (%)"); _grid(b); _panel(b, 1, "Share rated before the parcel arrived")
    _save(fig, out, "fig11_rq2_survey")


# ----------------------------------------------------------------------------------------------- Fig. 12
def fig_rq2_effects(out, cfg):
    A = _read(out, "rq2_aipw_bins.csv"); Rg = _read(out, "rq2_aipw_bins_by_region.csv"); RD = _read(out, "rq2_rdd.csv")
    if A is None:
        return
    fig, axes = _subplots(3, 1, figsize=(COL, 6.2), gridspec_kw=dict(height_ratios=[1, 1.05, 1.15]))
    a, b, c = axes
    ref = cfg["rq2"]["ref_bin"]
    labs = A[A.learner == "lr"].sort_values("bin")["label"].tolist()
    y = np.arange(len(labs))[::-1]
    for j, (ln, col) in enumerate((("lr", OI["vermillion"]), ("xgb", OI["blue"]))):
        x = A[A.learner == ln].sort_values("bin")
        yy = y + (0.5 - j) * 0.25
        a.hlines(yy, x["rr_lo"], x["rr_hi"], color=col, lw=0.9)
        a.scatter(x["rr"], yy, s=12, color=col, zorder=3, label={"lr": "Logistic nuisance", "xgb": "XGBoost nuisance"}[ln])
        if ln == "xgb":
            for yi, (_, r) in zip(yy, x.iterrows()):
                if r["bin"] != ref:
                    a.text(max(r["rr_hi"], r["rr"]) * 1.15, yi + 0.12, f"E = {r['e_value']:.1f}", fontsize=5.4, va="center", color=INK2)
    a.set_xscale("log"); a.axvline(1, color=INK2, lw=0.6, ls="--"); a.set_yticks(y); a.set_yticklabels(labs)
    a.set_xticks([0.5, 1, 2, 4, 8]); a.set_xticklabels(["0.5", "1", "2", "4", "8"]); a.minorticks_off()
    a.set_xlabel("Risk ratio vs S in -5..0 (log scale)"); a.set_xlim(0.35, 16); _grid(a, "x")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=5.6)
    _panel(a, 0, "AIPW risk ratios")
    if Rg is not None:
        late_bins = sorted(set(Rg["bin"]) - set(range(ref + 1)))
        for j, bb in enumerate(late_bins):
            x = Rg[Rg["bin"] == bb].set_index("region").reindex(REG_ORDER)
            yy = np.arange(len(REG_ORDER))[::-1] + (1 - j) * 0.22
            col = [OI["orange"], OI["vermillion"], "#7a1f00"][j % 3]
            b.hlines(yy, x["rd_lo"] * 100, x["rd_hi"] * 100, color=col, lw=0.9)
            b.scatter(x["rd"] * 100, yy, s=10, color=col, zorder=3, label=labs[bb].replace("S >= 8", "8 or more"))
        b.set_yticks(np.arange(len(REG_ORDER))[::-1]); b.set_yticklabels([REG_EN[g] for g in REG_ORDER])
        b.set_xlabel("Risk difference vs S in -5..0 (pp)"); _grid(b, "x")
        b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=5.6, title="Late by (days)", title_fontsize=5.8)
        _panel(b, 1, "Late-delivery effect by region")
    Ct = _read(out, "rq2_crude_curve_by_answer_timing.csv"); Ca = _read(out, "rq2_crude_curve.csv")
    if RD is not None and Ct is not None:
        h = 5
        def _ll(cells, side):
            x = cells["S"].to_numpy() - 0.5; n = cells["n"].to_numpy(); yv = cells["rate"].to_numpy()
            k = np.maximum(1 - np.abs(x) / (h + 0.5), 0) * n
            Xd = np.column_stack([np.ones_like(x), x]); W = np.diag(k)
            beta = np.linalg.solve(Xd.T @ W @ Xd, Xd.T @ W @ yv)
            xs = np.linspace(x.min(), 0.0, 20) if side < 0 else np.linspace(0.0, x.max(), 20)
            return xs + 0.5, (beta[0] + beta[1] * xs) * 100
        for data, col, lab, mk in ((Ca, MUTED, "all ratings", "o"),
                                   (Ct[Ct.answered_before_delivery == 0], OI["green"], "rated after arrival", "o")):
            x = data[(data.S >= -10) & (data.S <= 10)]
            c.vlines(x["S"], x["lo"] * 100, x["hi"] * 100, color=col, lw=0.6)
            c.scatter(x["S"], x["rate"] * 100, s=9, color=col, marker=mk, zorder=3, label=lab)
            L, Rr = x[(x.S <= 0) & (x.S > -h)], x[(x.S > 0) & (x.S <= h)]
            for cells, side in ((L, -1), (Rr, 1)):
                if len(cells) >= 2:
                    xs, ys = _ll(cells, side)
                    c.plot(xs, ys, color=col, lw=1.0)
        c.axvline(0.5, color=INK2, lw=0.6, ls="--")
        txt = [("Jump at S = 0 to 1", False, None), ("(h = 5, honest 95% CI)", False, None)]
        for sub, lab in (("answered_after_arrival", "after arrival"), ("all", "all ratings")):
            r = RD[(RD.subset == sub) & (RD.cutoff == 0) & (RD.h == h) & np.isclose(RD.M, RD.M_rule_of_thumb)]
            if len(r):
                r = r.iloc[0]
                bd = sub == "answered_after_arrival"
                txt += [(f"{lab}:", bd, None),
                        (f"  {100 * r['estimate']:+.1f} pp [{100 * r['ci_lo']:.1f}, {100 * r['ci_hi']:.1f}]", bd, None)]
        _lines_box(c, txt, "upper left", (0.02, 0.98))
        c.set_xlabel("Deviation S (days)"); c.set_ylabel("Low rating (%)"); c.set_xlim(-10.5, 10.5)
        _grid(c); c.legend(loc="lower left", fontsize=5.6, bbox_to_anchor=(0.0, 0.16))
        _panel(c, 2, "RDD at the promised day")
    _save(fig, out, "fig12_rq2_effects")


# ----------------------------------------------------------------------------------------------- Fig. 13
def fig_rq3_frontier(out, cfg):
    F = _read(out, "rq3_frontier.csv")
    if F is None:
        return
    fig, (a, b) = _subplots(1, 2, figsize=(FULL, 3.0))
    for pol in ("uniform", "route", "olist_shift", "cpd_quantile", "late_optimal", "optimal"):
        x = F[F.policy == pol].sort_values("mean_promise")
        x = x[(x.mean_promise >= 8) & (x.mean_promise <= 32)]
        for ax, col in ((a, "low_realised"), (b, "late_realised")):
            ax.plot(x["mean_promise"], x[col] * 100, color=P_COLOR[pol], lw=1.5 if pol == "optimal" else 1.0,
                    ls="-" if pol in ("optimal", "late_optimal", "cpd_quantile") else "--", label=P_LABEL[pol])
    o = F[F.policy == "olist"].iloc[0]
    for ax, col in ((a, "low_realised"), (b, "late_realised")):
        ax.scatter([o["mean_promise"]], [o[col] * 100], marker="*", s=80, color=INK, zorder=5, label="Olist promise")
        ax.set_xlabel("Mean promise (days)"); _grid(ax, "both")
        # zoom on the neighbourhood of Olist's promise, where the policies differ by fractions of a point
        ins = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
        for pol in ("route", "olist_shift", "cpd_quantile", "late_optimal", "optimal"):
            x = F[F.policy == pol].sort_values("mean_promise")
            x = x[(x.mean_promise >= o["mean_promise"] - 2.5) & (x.mean_promise <= o["mean_promise"] + 2.5)]
            ins.plot(x["mean_promise"], x[col] * 100, color=P_COLOR[pol], lw=1.2 if pol == "optimal" else 0.8,
                     ls="-" if pol in ("optimal", "late_optimal", "cpd_quantile") else "--")
        ins.scatter([o["mean_promise"]], [o[col] * 100], marker="*", s=40, color=INK, zorder=5)
        ins.tick_params(labelsize=5); ins.grid(color=GRID, lw=0.4)
        for sp in ins.spines.values():
            sp.set_linewidth(0.5)
    a.set_ylabel("Low-rating rate implied by r(s)\nat the actual delivery time (%)"); b.set_ylabel("Late rate at the actual delivery time (%)")
    fig.legend(*a.get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=6)
    _panel(a, 0, "Promise length and low ratings"); _panel(b, 1, "Promise length and lateness")
    _save(fig, out, "fig13_rq3_frontier")


# ----------------------------------------------------------------------------------------------- Fig. 14
def fig_rq3_anatomy(out, cfg):
    E = _read(out, "rq3_rho_examples.csv"); Rg = _read(out, "rq3_region_promises.csv"); Q = _read(out, "rq3_matched.csv")
    if E is None or Rg is None or Q is None:
        return
    fig, (a, b, c) = _subplots(3, 1, figsize=(COL, 6.0), gridspec_kw=dict(height_ratios=[1, 1.05, 1.1]))
    cols = [OI["blue"], OI["vermillion"], OI["green"], OI["orange"], OI["purple"], OI["sky"]]
    for i, (o, x) in enumerate(E.groupby("order")):
        x = x[x.P <= 45]
        a.plot(x["P"], x["rho"] * 100, color=cols[i % 6], lw=0.9)
        a.scatter([x["P_olist"].iloc[0]], [np.interp(x["P_olist"].iloc[0], x["P"], x["rho"]) * 100], marker="*",
                  s=28, color=cols[i % 6], zorder=3)
    a.set_xlabel("Promise P (days)"); a.set_ylabel(r"Expected low rating $\rho_i(P)$ (%)"); _grid(a)
    a.text(0.98, 0.95, "stars: Olist's promise", transform=a.transAxes, ha="right", va="top", fontsize=5.8, color=INK2)
    _panel(a, 0, r"$\rho_i(P)$ of six orders")
    y = np.arange(len(REG_ORDER))[::-1]; w = 0.26
    for j, pol in enumerate(("olist", "cpd_quantile", "optimal")):
        vals = [Rg[(Rg.policy == pol) & (Rg.region == g)]["mean_promise"].mean() for g in REG_ORDER]
        b.barh(y + (1 - j) * w, vals, height=w, color=P_COLOR[pol], alpha=0.85, label=P_LABEL[pol].replace(" (CPD quantile)", ""))
    b.set_yticks(y); b.set_yticklabels([REG_EN[g] for g in REG_ORDER]); b.set_xlabel("Mean promise (days)")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=5.6); _grid(b, "x")
    _panel(b, 1, "Promise by region, equal mean")
    Q["origin"] = pd.to_datetime(Q["origin"])
    d = (Q["optimal_low_realised_at_L"] - Q["low_olist"]) * 100
    dc = (Q["cpd_quantile_low_realised_at_L"] - Q["low_olist"]) * 100
    c.bar(Q["origin"] + pd.Timedelta(days=7), d, width=10, color=P_COLOR["optimal"], alpha=0.85, label="Optimal - Olist")
    c.plot(Q["origin"] + pd.Timedelta(days=7), dc, color=P_COLOR["cpd_quantile"], marker="o", ms=2.2, lw=0.8, label="Coverage policy - Olist")
    mu, lo, hi = _boot(d.to_numpy(), cfg)
    c.axhline(mu, color=P_COLOR["optimal"], lw=0.8, ls="--")
    _lines_box(c, [(f"optimal - Olist: mean {mu:.2f} pp", True, None), (f"95% block-bootstrap CI [{lo:.2f}, {hi:.2f}]", False, None)],
               "lower left", (0.02, 0.03), fontsize=5.6, color=P_COLOR["optimal"])
    c.axhline(0, color=INK2, lw=0.6); _date_axis(c); c.set_ylabel("Change in r(s)-implied\nlow-rating rate (pp)"); _grid(c)
    c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=5.6); _panel(c, 2, "Gain by window, equal mean")
    _save(fig, out, "fig14_rq3_anatomy")


def fig_s1_aci_sensitivity(out, cfg):
    """Dot plot of the ACI specifications (rows) against mean promise, lowest-window coverage and share of capped
    promises; marker shape = model, filled = all windows, hollow = 2018 (test period, gamma chosen on 2017)."""
    S = _read(out, "sensitivity_rq1.csv")
    if S is None or "period" not in S.columns:
        return
    X = S[(S.analysis == "conformal (ACI)")].copy()
    g0 = cfg["conformal"]["aci_gamma"]
    order = [f"daily batch, gamma = {g}" for g in (0.0001, 0.0002, 0.0005, 0.001)]
    order += ["daily batch, gamma = 0.0005, no clipping",
              "online per-order update (Gibbs and Candes), gamma = 0.0005, no clipping", "reference: split IPCW"]
    names = {f"daily batch, gamma = {g0}": f"daily batch, \u03b3 = {g0} (main)"}
    lab = [names.get(o, o.replace("gamma", "\u03b3").replace("online per-order update (Gibbs and Candes)",
                                                                 "per-order (Gibbs & Cand\u00e8s)")
                     .replace("reference: split IPCW", "split IPCW (reference)")) for o in order]
    cols = [("mean_promise", "Mean promise (days)", 1.0), ("coverage_min_window", "Lowest window coverage (%)", 100.0),
            ("share_capped", "Promises at the 90-day cap (%)", 100.0)]
    cols = [c for c in cols if c[0] in X.columns]
    fig, axes = _subplots(1, len(cols), figsize=(FULL, 2.6), sharey=True)
    offs = {("xgbq_ipcw", "all windows"): -0.24, ("qrf_ipcw", "all windows"): -0.08,
            ("xgbq_ipcw", "2018 (test)"): 0.08, ("qrf_ipcw", "2018 (test)"): 0.24}
    for ax, (c, xl, sc) in zip(axes, cols):
        for (v, per), dy in offs.items():
            mk = "o" if v.startswith("xgbq") else "^"
            x = X[(X.variant == v) & (X.period == per)].set_index("setting")
            for i, o in enumerate(order):
                if o not in x.index or pd.isna(x.loc[o, c]):
                    continue
                col = INK if o.startswith("reference") else (OI["vermillion"] if "no clipping" in o else OI["green"])
                ax.scatter(x.loc[o, c] * sc, i + dy, s=14, marker=mk, zorder=3,
                           color=col if per == "all windows" else "white", edgecolors=col, linewidths=0.8)
        ax.set_xlabel(xl); _grid(ax, "x")
        if c == "coverage_min_window":
            ax.axvline(90, color=MUTED, lw=0.5, ls=":")
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(lab, fontsize=5.6); axes[0].invert_yaxis()
    i0 = order.index(f"daily batch, gamma = {g0}") if f"daily batch, gamma = {g0}" in order else None
    for ax in axes:
        if i0 is not None:
            ax.axhspan(i0 - 0.45, i0 + 0.45, color=OI["green"], alpha=0.07, lw=0)
    for k, ax in enumerate(axes):
        _panel(ax, k, ["Promise length", "Worst window", "Capped promises"][k])
    hs = [Line2D([], [], ls="", marker="o", color=MUTED, label="XGBoost QR"),
          Line2D([], [], ls="", marker="^", color=MUTED, label="Quantile forest"),
          Line2D([], [], ls="", marker="o", color=MUTED, label="all windows"),
          Line2D([], [], ls="", marker="o", mfc="white", color=MUTED, label="2018 (test)"),
          Line2D([], [], ls="", marker="s", color=OI["green"], label="ACI, clipped at \u03b1/2"),
          Line2D([], [], ls="", marker="s", color=OI["vermillion"], label="ACI, no clipping"),
          Line2D([], [], ls="", marker="s", color=INK, label="split IPCW")]
    fig.legend(handles=hs, loc="outside lower center", ncol=7, fontsize=5.8)
    _save(fig, out, "figS1_aci_sensitivity")


def fig_s2_rq3_checks(out, cfg):
    Cc = _read(out, "rq3_curve_calibration.csv"); Gp = _read(out, "rq3_duality_gap.csv")
    if Cc is None or Gp is None:
        return
    fig, (a, b) = _subplots(1, 2, figsize=(FULL, 2.5))
    for oos, col, lab in ((False, MUTED, "2017 windows (in sample)"), (True, OI["blue"], "2018 windows (out of sample)")):
        x = Cc[Cc.out_of_sample == oos]
        a.scatter(x["predicted_fit2017"] * 100, x["observed"] * 100, s=14, color=col, label=lab, zorder=3)
    lim = [min(Cc[["observed", "predicted_fit2017"]].min()) * 100 - 1, max(Cc[["observed", "predicted_fit2017"]].max()) * 100 + 1]
    a.plot(lim, lim, color=INK2, lw=0.6, ls="--"); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("Predicted by r(s) fitted on 2017 (%)"); a.set_ylabel("Observed low-rating rate (%)")
    a.legend(loc="upper left", fontsize=6); _grid(a, "both"); _panel(a, 0, "Out-of-time check of the response curve")
    g = Gp.groupby("L")["gap"].agg(["mean", "max"]) * 100
    b.plot(g.index, g["max"], color=OI["vermillion"], marker="o", ms=2.5, label="max over windows")
    b.plot(g.index, g["mean"], color=OI["blue"], marker="o", ms=2.5, label="mean over windows")
    b.set_xlabel("Mean promise length L (days)"); b.set_ylabel("Feasible repair - dual bound (pp)")
    b.legend(loc="upper right", fontsize=6); _grid(b); _panel(b, 1, "Optimality gap of the Lagrangian sweep")
    _save(fig, out, "figS2_rq3_checks")


FIGURES = (fig_framework, fig_data_context, fig_protocol_censoring, fig_rq1_efficiency, fig_rq1_censoring, fig_rq1_time,
           fig_rq1_regions, fig_rq1_quantile_calib, fig_rq2_response, fig_rq2_s_vs_d, fig_rq2_survey, fig_rq2_effects,
           fig_rq3_frontier, fig_rq3_anatomy, fig_s1_aci_sensitivity, fig_s2_rq3_checks)


def make_all(cfg, out, only=None):
    _style()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for f in FIGURES:
        if only and f.__name__ not in only:
            continue
        f(out, cfg)
