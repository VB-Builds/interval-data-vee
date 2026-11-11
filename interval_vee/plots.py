"""README figure (1600x900) and dark LinkedIn figure (960x700) from the demo results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DARK = {"bg": "#16213A", "grid": "#2A3750", "text": "#F8FAFC", "muted": "#A9B6CB", "accent": "#38BDF8",
        "green": "#34D399", "amber": "#FBBF24", "red": "#F87171", "grey": "#64748B", "violet": "#A78BFA"}
LIGHT = {"bg": "#FFFFFF", "grid": "#E5E7EB", "text": "#111827", "muted": "#6B7280", "accent": "#0284C7",
         "green": "#059669", "amber": "#D97706", "red": "#DC2626", "grey": "#94A3B8", "violet": "#7C3AED"}
LEN_LABELS = {1: "15 min", 2: "30 min", 4: "1 h", 8: "2 h", 16: "4 h", 32: "8 h", 96: "1 d", 288: "3 d", 672: "7 d"}
METHODS = [("linear", "linear interpolation", "amber", "o"), ("profile", "similar-day profile", "green", "s"),
           ("regression", "temperature regression", "violet", "D"), ("vee+register", "VEE + daily register read", "accent", "^")]
RULE_SHORT = {"GAP": "gap", "ZERO_RUN": "zero run", "SPIKE": "spike", "NEGATIVE": "negative", "STUCK": "stuck",
              "DUPLICATE": "duplicate", "CLOCK_SHIFT": "clock shift", "REGISTER_MISMATCH": "register"}
SRC = "Data: UCI household power dataset (CC BY 4.0), Sceaux, France, 2008-01 to 2010-10; weather: Open-Meteo.com (CC BY 4.0)"


def _font():
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Segoe UI" if "Segoe UI" in names else "DejaVu Sans"


def _style(ax, pal, grid_axis="y"):
    ax.set_facecolor(pal["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(pal["grid"])
    ax.tick_params(colors=pal["muted"])
    ax.xaxis.label.set_color(pal["muted"])
    ax.yaxis.label.set_color(pal["muted"])
    ax.grid(axis=grid_axis, color=pal["grid"], lw=0.8)
    ax.set_axisbelow(True)


def _curve_panel(ax, summ, pal, fs, col, methods, floor=0.005, ylabel="", lw=2.5, ms=6):
    lengths = sorted(summ["length"].unique())
    x = np.arange(len(lengths))
    for m, label, c, mk in methods:
        y = [summ[(summ["length"] == L) & (summ["method"] == m)][col].iloc[0] for L in lengths]
        ax.plot(x, np.maximum(y, floor), color=pal[c], lw=lw, marker=mk, ms=ms, label=label)
    ax.set_xticks(x, [LEN_LABELS.get(L, str(L)) for L in lengths], fontsize=fs)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_xlabel("gap length", fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)


def _pick_window(f, truth):
    """Two local days with the most different repairable faults (no gross spike, so the y axis stays readable)."""
    kinds_ok = {"gap", "zero_run", "stuck", "negative", "spike", "clock_shift"}
    best, best_score = None, -1
    for e in f.events:
        if e.kind not in kinds_ok or (e.kind == "spike" and e.detail.get("gross")):
            continue
        a = max(0, e.start - 48)
        b = min(len(truth), a + 192)
        inside = [x for x in f.events if x.start < b and x.end > a]
        if any(x.kind == "spike" and x.detail.get("gross") for x in inside) or np.isnan(f.truth[a:b]).any():
            continue
        score = len({x.kind for x in inside if x.kind in kinds_ok}) + 0.001 * sum(min(x.length, 96) for x in inside)
        if score > best_score:
            best, best_score = (a, b), score
    return best


def _example_panel(ax, f, res, truth, pal, fs):
    a, b = _pick_window(f, truth)
    t = truth.index[a:b].tz_convert("Europe/Paris").tz_localize(None)
    tv = f.truth[a:b]
    ax.plot(t, tv, color=pal["grey"], lw=4, alpha=0.55, label="hidden truth", drawstyle="steps-post")
    rec = res.received[a:b]
    bad = (res.flags[a:b] & 0b11111) > 0
    ax.plot(t, np.where(~bad, rec, np.nan), color=pal["text"], lw=1.2, drawstyle="steps-post", label="meter data as received")
    ax.scatter(t[bad & ~np.isnan(rec)], rec[bad & ~np.isnan(rec)], color=pal["red"], s=22, zorder=3, label="rejected reading")
    est = np.where(res.status[a:b] == "E", res.value[a:b], np.nan)
    ax.plot(t, est, color=pal["accent"], lw=2.2, drawstyle="steps-post", label="estimate")
    hi = np.nanmax(tv) * 1.35
    ax.set_ylim(min(0, np.nanmin(np.where(np.isnan(rec), 0, rec)) * 1.1), hi)
    kinds = sorted({x.kind.replace("_", " ") for x in f.events if x.start < b and x.end > a})
    ax.set_ylabel("kWh per 15 min", fontsize=fs)
    ax.tick_params(labelsize=fs - 1)
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    return kinds, t[0]


def make_figures(results: dict, summ: pd.DataFrame, res, f, truth: pd.Series, docs: Path) -> None:
    docs.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = _font()
    readme_figure(results, summ, res, f, truth, docs / "figure.png")
    post_figure(results, summ, docs / "post-figure.png")


def readme_figure(results, summ, res, f, truth, path: Path) -> None:
    pal = LIGHT
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=pal["bg"])
    gs = fig.add_gridspec(2, 2, left=0.095, right=0.985, top=0.855, bottom=0.075, hspace=0.45, wspace=0.17)
    mo = results["monthly"]
    lim = results["longest_gap_within_limit"]
    fig.text(0.055, 0.95, "VEE for 15-minute meter data: faults found, gaps filled, error measured against real data",
             fontsize=21, fontweight="bold", color=pal["text"])
    fig.text(0.055, 0.915, SRC, fontsize=11.5, color=pal["muted"])
    ax1 = fig.add_subplot(gs[0, 0])
    _style(ax1, pal)
    _curve_panel(ax1, summ, pal, 11, "month_p90_pct", METHODS, ylabel="% of the month's energy (p90)")
    ax1.set_yscale("log")
    ax1.axhline(results["billing_limit_pct"], color=pal["red"], lw=1.5, ls="--")
    ax1.text(len(LEN_LABELS) - 1, results["billing_limit_pct"] * 0.62, f"{results['billing_limit_pct']:g} % own limit",
             color=pal["red"], fontsize=10.5, ha="right")
    ax1.legend(frameon=False, fontsize=10, labelcolor=pal["text"], loc="upper left", ncol=2)
    ax1.set_ylim(0.004, 60)
    ax1.text(len(LEN_LABELS) - 1, 0.0065, "with register: < 0.01 % at every length", color=pal["accent"], fontsize=10.5, ha="right")
    ax1.set_title(f"Error one gap adds to the monthly kWh (p90): < 1 % up to {lim['vee']} from intervals alone",
                  fontsize=12.5, color=pal["text"], loc="left")
    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2, pal)
    _curve_panel(ax2, summ, pal, 11, "wape_pct", METHODS, floor=0, ylabel="interval error, WAPE %")
    ax2.set_ylim(0, 100)
    w = summ[(summ["method"] == "vee+register") & (summ["length"] >= 96)]["wape_pct"]
    ax2.set_title(f"15-minute shape error (WAPE): {w.min():.0f}-{w.max():.0f} % from 1 day, even with register",
                  fontsize=12.5, color=pal["text"], loc="left")
    ax3 = fig.add_subplot(gs[1, 0])
    _style(ax3, pal, grid_axis="x")
    det = pd.DataFrame(results["detection"])
    y = np.arange(len(det))
    ax3.barh(y - 0.2, det["recall"] * 100, height=0.38, color=pal["accent"], label="recall")
    ax3.barh(y + 0.2, det["precision"] * 100, height=0.38, color=pal["green"], label="precision")
    for i, (r, p) in enumerate(zip(det["recall"], det["precision"], strict=True)):
        ax3.text(r * 100 + 0.5, i - 0.2, f"{r * 100:.0f}", va="center", fontsize=9.5, color=pal["text"])
        ax3.text(p * 100 + 0.5, i + 0.2, f"{p * 100:.0f}", va="center", fontsize=9.5, color=pal["text"])
    lab = [f"{RULE_SHORT[r]} {d}/{e}" for r, d, e in zip(det["rule"], det["detected"], det["events"], strict=True)]
    ax3.set_yticks(y, lab, fontsize=10.5)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 112)
    ax3.set_xlabel("%", fontsize=11)
    ax3.legend(frameon=False, fontsize=10, labelcolor=pal["text"], loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    n_f = sum(results["faults_injected"].values())
    ax3.set_title(f"Detection: {n_f} injected faults + {results['source_gaps']['runs']} real gaps",
                  fontsize=12.5, color=pal["text"], loc="left")
    ax4 = fig.add_subplot(gs[1, 1])
    _style(ax4, pal)
    kinds, t0 = _example_panel(ax4, f, res, truth, pal, 11)
    ax4.legend(frameon=False, fontsize=9.5, labelcolor=pal["text"], loc="upper right", ncol=2)
    ax4.set_title(f"Example, 2 days from {t0:%d %b %Y}: {', '.join(kinds)}", fontsize=12.5, color=pal["text"], loc="left")
    fig.text(0.985, 0.012, f"Monthly energy after VEE: max {mo['vee_abs_max_pct']:.2f} % off over {mo['months']} months "
             f"(as received: up to {mo['raw_abs_max_pct']:.0f} %)", fontsize=10.5, color=pal["muted"], ha="right")
    fig.savefig(path, dpi=100, facecolor=pal["bg"])
    plt.close(fig)


def post_figure(results, summ, path: Path) -> None:
    pal = DARK
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=pal["bg"])
    lim = results["longest_gap_within_limit"]
    fig.text(0.04, 0.948, "Smart-meter gaps: how long an estimate stays within 1 %", fontsize=14, fontweight="bold",
             color=pal["text"])
    fig.text(0.04, 0.912, "Open data: one real household, 2008-2010 (UCI, CC BY 4.0); faults simulated | weather: Open-Meteo.com",
             fontsize=12, color=pal["muted"])
    ax1 = fig.add_axes([0.105, 0.50, 0.865, 0.33])
    _style(ax1, pal)
    meth = [("vee", "intervals only", "amber", "o"), ("vee+register", "+ daily register read", "accent", "^")]
    _curve_panel(ax1, summ, pal, 12, "month_p90_pct", meth, ylabel="% of month (p90)", lw=3.5, ms=8)
    ax1.set_yscale("log")
    ax1.set_ylim(0.004, 40)
    ax1.set_xlabel("")
    ax1.axhline(results["billing_limit_pct"], color=pal["red"], lw=2, ls="--")
    ax1.text(0.0, results["billing_limit_pct"] * 1.3, "1 % (own yardstick)", color=pal["red"], fontsize=12)
    ax1.text(len(LEN_LABELS) - 1, 0.0075, "with register: < 0.01 %", color=pal["accent"], fontsize=12, ha="right")
    ax1.legend(frameon=False, fontsize=12, labelcolor=pal["text"], loc="upper left", bbox_to_anchor=(0.12, 1.02))
    ax1.set_title(f"Error one gap adds to the monthly kWh: below 1 % up to {lim['vee']}", fontsize=13,
                  color=pal["text"], loc="left")
    ax2 = fig.add_axes([0.105, 0.08, 0.865, 0.31])
    _style(ax2, pal)
    det = pd.DataFrame(results["detection"])
    x = np.arange(len(det))
    ax2.bar(x, det["recall"] * 100, color=[pal["green"] if r >= 0.95 else pal["amber"] for r in det["recall"]], width=0.68)
    for i, r in enumerate(det["recall"]):
        ax2.text(i, r * 100 + 3, f"{r * 100:.0f}", ha="center", fontsize=12, color=pal["text"])
    ax2.set_xticks(x, [RULE_SHORT[r].replace(" ", "\n") for r in det["rule"]], fontsize=12)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.set_ylim(0, 118)
    ax2.set_yticks([0, 50, 100])
    ax2.set_title(f"Recall per rule, {sum(results['faults_injected'].values())} simulated faults injected (%)", fontsize=13,
                  color=pal["text"], loc="left")
    fig.savefig(path, dpi=100, facecolor=pal["bg"])
    plt.close(fig)
