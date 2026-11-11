"""Reproduce every number in the README: gap benchmark + full fault mix on 34 months of real household data."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import export
from .evaluate import benchmark_gaps, detection, detection_by_subtype, monthly_energy, repair_accuracy, summarize_benchmark
from .faults import inject, source_gap_events
from .meterio import ingest, read_register, read_rows, write_register
from .source import load_temperature, load_truth
from .vee import VeeConfig, run_vee

TZ = "Europe/Paris"
EVAL_START, EVAL_END = "2008-01-01", "2010-11-01"  # thresholds were chosen on 2007 only
LENGTHS = [1, 2, 4, 8, 16, 32, 96, 288, 672]
BILLING_LIMIT_PCT = 1.0  # own criterion: one gap may move the month's energy by at most 1 % (p90)


def _slice(s: pd.Series, start: str, end: str) -> pd.Series:
    lo = s.index.searchsorted(pd.Timestamp(start, tz=TZ))
    hi = s.index.searchsorted(pd.Timestamp(end, tz=TZ))
    return s.iloc[lo:hi]


def _hours(slots: int) -> str:
    h = slots / 4
    return f"{h:g} h" if h < 24 else f"{h / 24:g} d"


def run_demo(out: Path, docs: Path | None = None, figures: bool = True, per_length: int = 60, examples: Path | None = None,
             seed: int = 7) -> dict:
    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    truth_all = load_truth()
    temp = load_temperature()
    truth = _slice(truth_all, EVAL_START, EVAL_END)
    known = truth.notna()
    data = {"dataset": "UCI Individual household electric power consumption (CC BY 4.0), one house in Sceaux, France",
            "period": f"{EVAL_START} to {EVAL_END} (local), {len(truth):,} quarter-hours, {int(known.sum()):,} with known truth",
            "temperature": "Open-Meteo historical weather API, daily mean 2 m temperature (CC BY 4.0)"}

    # 1. gap benchmark: clean gaps, every method on the same gaps
    bench = benchmark_gaps(truth_all, temp, EVAL_START, EVAL_END, LENGTHS, per_length=per_length, seed=11)
    summ = summarize_benchmark(bench)
    summ.to_csv(out / "gap_benchmark.csv", index=False)
    t1 = time.perf_counter()

    # 2. full fault mix through the meter file
    f = inject(truth, seed=seed, tz=TZ)
    meter_csv, reg_csv = out / "meter_faulty.csv", out / "register_daily.csv"
    f.rows.to_csv(meter_csv, index=False)
    write_register(f.register, reg_csv)
    key = [{"kind": e.kind, "start_utc": str(truth.index[e.start]), "slots": e.length, **e.detail} for e in f.events]
    (out / "answer_key.json").write_text(json.dumps(key, indent=1, default=str))
    rows = read_rows(meter_csv)
    register = read_register(reg_csv)
    ing = ingest(rows, tz=TZ, grid=truth.index)
    cfg = VeeConfig(tz=TZ)
    t2 = time.perf_counter()
    res = run_vee(ing, cfg, register=register, temperature=temp)
    t_vee = time.perf_counter() - t2
    res_noreg = run_vee(ing, replace(cfg, use_register=False), register=None, temperature=temp)  # no register data at all
    exp_csv, aud_csv = export.write(res, out, TZ)

    events = f.events + source_gap_events(f.truth)
    det = detection(events, res.flags)
    sub = detection_by_subtype(events, res.flags)
    acc = repair_accuracy(f.events, res, f.truth)
    acc_nr = repair_accuracy(f.events, res_noreg, f.truth)
    mon = monthly_energy(res, f.truth, ing.slot, ing.value, TZ)
    mon_nr = monthly_energy(res_noreg, f.truth, ing.slot, ing.value, TZ)
    det.to_csv(out / "detection.csv", index=False)
    sub.to_csv(out / "detection_by_subtype.csv", index=False)
    mon.assign(vee_noreg_kwh=mon_nr["vee_kwh"], vee_noreg_err_pct=mon_nr["vee_err_pct"]).to_csv(out / "monthly_energy.csv", index=False)

    need = (res.flags & 0b11111) > 0  # GAP, ZERO_RUN, SPIKE, NEGATIVE, STUCK need a value
    local = truth.index.tz_convert(TZ)
    day_counts = pd.Series(1, index=local.tz_localize(None).normalize()).groupby(level=0).size()
    dst_days = {str(d.date()): int(c) for d, c in day_counts.items() if c != 96}

    def bench_row(L, m):
        r = summ[(summ["length"] == L) & (summ["method"] == m)]
        return {k: float(r[k].iloc[0]) for k in ("mae_kwh", "mape_pct", "wape_pct", "energy_abs_med_pct", "energy_abs_p90_pct",
                                                  "month_p90_pct", "month_max_pct", "energy_bias_pct")} if len(r) else None

    limit_ok = {m: max([L for L in LENGTHS if (b := bench_row(L, m)) and b["month_p90_pct"] <= BILLING_LIMIT_PCT], default=0)
                for m in ("vee", "vee+register")}
    first_fail = {m: min([L for L in LENGTHS if (b := bench_row(L, m)) and b["month_p90_pct"] > BILLING_LIMIT_PCT], default=None)
                  for m in ("vee", "vee+register")}

    def acc_kind(a):
        g = a.groupby("kind")
        return {k: {"events": int(len(d)), "repaired_pct": 100 * float(d["repaired"].mean()), "exact_pct": 100 * float(d["exact"].mean()),
                    "abs_err_kwh": float(d["abs_err_kwh"].sum()), "net_err_kwh": float(d["net_err_kwh"].sum()),
                    "true_kwh": float(d["true_kwh"].sum())} for k, d in g}

    conflicting = [e for e in f.events if e.kind == "duplicate" and e.detail.get("conflicting")]
    dup_ok = sum(abs(res.value[e.start] - f.truth[e.start]) < 0.0006 for e in conflicting)
    results = {
        "data": data,
        "rows": ing.issues,
        "faults_injected": {k: int(sum(e.kind == k for e in f.events)) for k in sorted({e.kind for e in f.events})},
        "source_gaps": {"runs": int(sum(1 for e in events if e.detail.get("origin") == "source")),
                        "slots": int((~known).sum())},
        "detection": det.to_dict(orient="records"),
        "detection_by_subtype": sub.to_dict(orient="records"),
        "intervals_needing_value": int(need.sum()),
        "intervals_repaired": int((need & ~np.isnan(res.value)).sum()),
        "repaired_pct": 100 * float((need & ~np.isnan(res.value)).sum() / max(1, need.sum())),
        "repair_accuracy": acc_kind(acc),
        "repair_accuracy_no_register": acc_kind(acc_nr),
        "conflicting_duplicates_resolved": [int(dup_ok), len(conflicting)],
        "clock_shifts_found": res.summary["clock_shifts"],
        "register_days_checked": res.summary["register_days_checked"],
        "monthly": {
            "months": len(mon),
            "raw_abs_max_pct": float(mon["raw_err_pct"].abs().max()),
            "raw_abs_median_pct": float(mon["raw_err_pct"].abs().median()),
            "vee_abs_max_pct": float(mon["vee_err_pct"].abs().max()),
            "vee_abs_max_kwh": float(mon["vee_err_kwh"].abs().max()),
            "vee_abs_mean_pct": float(mon["vee_err_pct"].abs().mean()),
            "vee_noreg_abs_max_pct": float(mon_nr["vee_err_pct"].abs().max()),
            "vee_noreg_abs_max_kwh": float(mon_nr["vee_err_kwh"].abs().max()),
            "vee_noreg_abs_mean_pct": float(mon_nr["vee_err_pct"].abs().mean()),
            "total_true_kwh": float(mon["true_kwh"].sum()),
            "total_vee_err_kwh": float(mon["vee_err_kwh"].sum()),
            "total_vee_noreg_err_kwh": float(mon_nr["vee_err_kwh"].sum()),
        },
        "benchmark": {f"{L}|{m}": bench_row(L, m) for L in LENGTHS for m in ("linear", "profile", "regression", "vee", "vee+register")},
        "benchmark_per_length": per_length,
        "billing_limit_pct": BILLING_LIMIT_PCT,
        "longest_gap_within_limit": {m: _hours(v) if v else "none" for m, v in limit_ok.items()},
        "first_gap_over_limit": {m: _hours(v) if v else "none" for m, v in first_fail.items()},
        "dst_days_intervals": dst_days,
        "timing_s": {"benchmark": round(t1 - t0, 1), "vee_full_run": round(t_vee, 2), "total": None},
        "outputs": {"export": str(exp_csv), "audit": str(aud_csv), "audit_rows": int(len(res.audit))},
    }
    results["timing_s"]["total"] = round(time.perf_counter() - t0, 1)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    if examples is not None:
        write_examples(examples, rows, register, res, f, truth)
    if figures and docs is not None:
        from .plots import make_figures

        make_figures(results, summ, res, f, truth, docs)
        slim = {k: v for k, v in results.items() if k not in ("outputs", "timing_s")}
        (docs / "results.json").write_text(json.dumps(slim, indent=1, default=float))
    return results


def write_examples(path: Path, rows: pd.DataFrame, register: pd.Series, res, f, truth: pd.Series) -> None:
    """One month (October 2009: autumn DST change) of input and output, small enough to commit."""
    path.mkdir(parents=True, exist_ok=True)
    m = rows["timestamp"].str.startswith("2009-10")
    rows[m].to_csv(path / "meter_2009-10.csv", index=False)
    reg = register[[str(d).startswith("2009-10") for d in register.index]]
    write_register(reg, path / "register_2009-10.csv")
    df = export.to_frame(res, TZ)
    df[df["timestamp_local"].str.startswith("2009-10")].to_csv(path / "vee_export_2009-10.csv", index=False)
    a = export.audit_frame(res)
    loc = pd.to_datetime(a["timestamp_utc"], utc=True).dt.tz_convert(TZ)
    a[(loc.dt.year == 2009) & (loc.dt.month == 10)].to_csv(path / "vee_audit_2009-10.csv", index=False)
