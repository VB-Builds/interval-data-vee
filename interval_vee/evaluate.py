"""Score VEE output against the hidden truth and the answer key of injected faults."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .estimate import Context, linear, profile, regression
from .faults import Event
from .vee import FLAGS, VeeResult, runs

KIND_FLAG = {"gap": "GAP", "zero_run": "ZERO_RUN", "spike": "SPIKE", "negative": "NEGATIVE", "stuck": "STUCK",
             "duplicate": "DUPLICATE", "clock_shift": "CLOCK_SHIFT", "register_mismatch": "REGISTER_MISMATCH"}
VALUE_KINDS = ["gap", "zero_run", "stuck", "spike", "negative"]


def _subtype(e: Event) -> str:
    d = e.detail
    if e.kind == "spike":
        return "gross" if d.get("gross") else "subtle (+0.5..2.5 kWh)"
    if e.kind == "duplicate":
        return "conflicting" if d.get("conflicting") else "identical"
    if e.kind == "gap":
        return "source" if d.get("origin") == "source" else "injected"
    if e.kind in ("zero_run", "stuck"):
        return f"{e.length} slots"
    if e.kind == "clock_shift":
        return d.get("cause", "")
    if e.kind == "register_mismatch":
        f = d.get("factor", 1.0)
        return "|error| < 5 %" if abs(f - 1) < 0.05 else "|error| >= 5 %"
    return ""


def detection(events: list[Event], flags: np.ndarray) -> pd.DataFrame:
    """Event-level recall and precision per rule.

    recall: injected events with at least one interval flagged by the matching rule.
    precision: runs of flagged intervals that overlap an injected event of that kind. For REGISTER_MISMATCH a flagged
    day also counts as correct when it holds any other injected fault: the register check exists to catch what the
    interval rules missed, so those days are reported separately as `caught_other`.
    """
    n = len(flags)
    rows = []
    for kind, name in KIND_FLAG.items():
        bit = FLAGS[name]
        evs = [e for e in events if e.kind == kind]
        mask = np.zeros(n, dtype=bool)
        for e in evs:
            mask[e.start:e.end] = True
        other = np.zeros(n, dtype=bool)
        if kind == "register_mismatch":
            for e in events:
                if e.kind != kind and e.detail.get("origin") != "source":
                    other[e.start:e.end] = True
        hit = [bool(np.any(flags[e.start:e.end] & bit)) for e in evs]
        fr = runs((flags & bit) > 0)
        tp = sum(bool(mask[s:s + m].any()) for s, m in fr)
        caught = sum(bool(other[s:s + m].any() and not mask[s:s + m].any()) for s, m in fr)
        rows.append({"rule": name, "events": len(evs), "detected": int(sum(hit)),
                     "recall": sum(hit) / len(evs) if evs else np.nan, "flagged_runs": len(fr), "caught_other": caught,
                     "false_runs": len(fr) - tp - caught, "precision": (tp + caught) / len(fr) if fr else np.nan})
    return pd.DataFrame(rows)


def detection_by_subtype(events: list[Event], flags: np.ndarray) -> pd.DataFrame:
    rows = []
    for e in events:
        bit = FLAGS[KIND_FLAG[e.kind]]
        rows.append({"kind": e.kind, "subtype": _subtype(e), "hit": bool(np.any(flags[e.start:e.end] & bit))})
    df = pd.DataFrame(rows)
    return df.groupby(["kind", "subtype"]).agg(events=("hit", "size"), detected=("hit", "sum")).reset_index()


def repair_accuracy(events: list[Event], res: VeeResult, truth: np.ndarray) -> pd.DataFrame:
    """Error of the edited values against the truth, per injected fault kind."""
    rows = []
    for e in events:
        if e.detail.get("origin") == "source":
            continue
        sl = slice(e.start, e.end)
        tv, vv = truth[sl], res.value[sl]
        ok = ~np.isnan(tv)
        if not ok.any():
            continue
        filled = ~np.isnan(vv[ok])
        err = np.where(filled, vv[ok] - tv[ok], np.nan)
        rows.append({"kind": e.kind, "length": e.length, "true_kwh": float(tv[ok].sum()),
                     "repaired": bool(filled.all()), "abs_err_kwh": float(np.nansum(np.abs(err))),
                     "net_err_kwh": float(np.nansum(err)), "exact": bool(np.nanmax(np.abs(err)) < 0.0006) if filled.any() else False})
    return pd.DataFrame(rows)


def monthly_energy(res: VeeResult, truth: np.ndarray, raw_slot: np.ndarray, raw_value: np.ndarray, tz: str) -> pd.DataFrame:
    """Monthly energy over intervals with known truth: raw rows summed as received vs after VEE."""
    months = res.grid.tz_convert(tz).tz_localize(None).to_period("M")
    codes, uniq = pd.factorize(months)
    known = ~np.isnan(truth)
    w_true = np.bincount(codes, weights=np.where(known, truth, 0), minlength=len(uniq))
    w_vee = np.bincount(codes, weights=np.where(known & ~np.isnan(res.value), res.value, 0), minlength=len(uniq))
    raw_ok = known[raw_slot]
    w_raw = np.bincount(codes[raw_slot[raw_ok]], weights=raw_value[raw_ok], minlength=len(uniq))
    df = pd.DataFrame({"month": [str(u) for u in uniq], "true_kwh": w_true, "raw_kwh": w_raw, "vee_kwh": w_vee})
    df["raw_err_pct"] = 100 * (df["raw_kwh"] - df["true_kwh"]) / df["true_kwh"]
    df["vee_err_kwh"] = df["vee_kwh"] - df["true_kwh"]
    df["vee_err_pct"] = 100 * df["vee_err_kwh"] / df["true_kwh"]
    return df


def benchmark_gaps(truth: pd.Series, temperature: pd.Series | None, start: str, end: str, lengths: list[int],
                   per_length: int = 60, seed: int = 11, tz: str = "Europe/Paris", linear_max: int = 16,
                   regression_min: int = 96) -> pd.DataFrame:
    """Cut clean gaps of fixed length into the truth and score each estimation method on the same gaps."""
    rng = np.random.default_rng(seed)
    base0 = truth.to_numpy(dtype=float)
    ctx = Context(truth.index, tz, temperature)
    totals = ctx.daily_totals(base0)
    reg = np.round(totals, 2)
    month = truth.index.tz_convert(tz).tz_localize(None).to_period("M")
    mcode, _ = pd.factorize(month)
    month_kwh = np.bincount(mcode, weights=np.nan_to_num(base0), minlength=mcode.max() + 1)
    lo = int(truth.index.searchsorted(pd.Timestamp(start, tz=tz)))
    hi = int(truth.index.searchsorted(pd.Timestamp(end, tz=tz)))
    rows = []
    for L in lengths:
        done = 0
        tries = 0
        while done < per_length and tries < 50 * per_length:
            tries += 1
            a = int(rng.integers(lo + 1, hi - L - 1))
            b = a + L
            days = np.unique(ctx.day_code[a:b])
            if np.isnan(base0[a - 1:b + 1]).any() or np.isnan(totals[days]).any():
                continue
            base = base0.copy()
            base[a:b] = np.nan
            tv = base0[a:b]
            ests = {"linear": linear(base, a, b), "profile": profile(base, a, b, ctx),
                    "regression": regression(base, a, b, ctx) if ctx.temp is not None else np.full(L, np.nan)}
            use_reg = L >= regression_min and not np.isnan(ests["regression"]).any()
            ests["vee"] = ests["linear"] if L <= linear_max else (ests["regression"] if use_reg else ests["profile"])
            ests["vee+register"] = _prorate(ests["vee"], base, a, b, ctx, reg)
            for m, est in ests.items():
                if np.isnan(est).any():
                    continue
                err = est - tv
                rows.append({"length": L, "method": m, "mae_kwh": float(np.mean(np.abs(err))),
                             "mape_pct": float(100 * np.mean(np.abs(err) / tv)),
                             "wape_pct": float(100 * np.abs(err).sum() / tv.sum()),
                             "energy_err_pct": float(100 * err.sum() / tv.sum()), "gap_true_kwh": float(tv.sum()),
                             "err_kwh": float(err.sum()), "month_share_pct": float(100 * abs(err.sum()) / month_kwh[mcode[a]])})
            done += 1
    return pd.DataFrame(rows)


def _prorate(est: np.ndarray, base: np.ndarray, a: int, b: int, ctx: Context, reg: np.ndarray) -> np.ndarray:
    out = est.copy()
    codes = ctx.day_code[a:b]
    for d in np.unique(codes):
        lo, hi = ctx.day_start[d], ctx.day_end[d]
        actual = float(np.nansum(base[lo:hi]))
        sel = codes == d
        s = out[sel].sum()
        target = max(0.0, reg[d] - actual)
        out[sel] = out[sel] * target / s if s > 0 else target / sel.sum()
    return out


def summarize_benchmark(bench: pd.DataFrame) -> pd.DataFrame:
    g = bench.groupby(["length", "method"])
    out = g.agg(n=("mae_kwh", "size"), mae_kwh=("mae_kwh", "mean"), mape_pct=("mape_pct", "mean"),
                wape_pct=("wape_pct", "mean"),
                energy_abs_med_pct=("energy_err_pct", lambda x: float(np.median(np.abs(x)))),
                energy_abs_p90_pct=("energy_err_pct", lambda x: float(np.quantile(np.abs(x), 0.9))),
                month_p90_pct=("month_share_pct", lambda x: float(np.quantile(x, 0.9))),
                month_max_pct=("month_share_pct", "max"),
                err_kwh=("err_kwh", "sum"), true_kwh=("gap_true_kwh", "sum")).reset_index()
    out["energy_bias_pct"] = 100 * out.pop("err_kwh") / out.pop("true_kwh")
    return out
