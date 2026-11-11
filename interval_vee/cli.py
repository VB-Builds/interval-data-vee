"""Command line: demo, run, inject, benchmark, prepare-data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import export
from .meterio import ingest, read_register, read_rows
from .vee import VeeConfig, run_vee


def _demo(a) -> int:
    from .demo import run_demo

    r = run_demo(Path(a.out), Path(a.docs), figures=not a.no_figures, per_length=a.per_length,
                 examples=None if a.no_examples else Path(a.examples))
    print(json.dumps({k: r[k] for k in ("faults_injected", "repaired_pct", "monthly", "longest_gap_within_limit",
                                        "first_gap_over_limit", "timing_s")}, indent=1, default=float))
    for d in r["detection"]:
        print(f"{d['rule']:<18} recall {d['recall']:.3f}  precision {d['precision']:.3f}  ({d['detected']}/{d['events']})")
    return 0


def _run(a) -> int:
    from .source import load_temperature

    rows = read_rows(a.readings)
    ing = ingest(rows, tz=a.tz)
    reg = read_register(a.register) if a.register else None
    temp = load_temperature(Path(a.temperature)) if a.temperature else None
    cfg = VeeConfig(tz=a.tz, use_register=not a.no_register_prorate, long_method=a.long_method,
                    max_interval_kwh=a.max_interval_kwh)
    res = run_vee(ing, cfg, register=reg, temperature=temp)
    p1, p2 = export.write(res, Path(a.out), a.tz)
    print(json.dumps(res.summary, indent=1, default=int))
    print(f"wrote {p1} and {p2}")
    return 0


def _inject(a) -> int:
    import pandas as pd

    from .faults import inject
    from .meterio import write_register
    from .source import load_truth

    s = load_truth()
    s = s[(s.index >= pd.Timestamp(a.start, tz=a.tz)) & (s.index < pd.Timestamp(a.end, tz=a.tz))]
    f = inject(s, seed=a.seed, tz=a.tz)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f.rows.to_csv(out / "meter_faulty.csv", index=False)
    write_register(f.register, out / "register_daily.csv")
    key = [{"kind": e.kind, "start_utc": str(s.index[e.start]), "slots": e.length, **e.detail} for e in f.events]
    (out / "answer_key.json").write_text(json.dumps(key, indent=1, default=str))
    print(f"{len(f.rows)} rows, {len(f.events)} faults -> {out}")
    return 0


def _benchmark(a) -> int:
    import pandas as pd

    from .demo import LENGTHS
    from .evaluate import benchmark_gaps, summarize_benchmark
    from .source import load_temperature, load_truth

    b = summarize_benchmark(benchmark_gaps(load_truth(), load_temperature(), a.start, a.end, LENGTHS, per_length=a.per_length,
                                           seed=a.seed))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(b.round(2).to_string(index=False))
    return 0


def _prepare(a) -> int:
    from .source import prepare

    print(prepare(Path(a.cache), Path(__file__).parent / "data"))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="interval-vee", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="reproduce the README numbers")
    d.add_argument("--out", default="out/demo")
    d.add_argument("--docs", default="docs")
    d.add_argument("--examples", default="examples")
    d.add_argument("--per-length", type=int, default=60)
    d.add_argument("--no-figures", action="store_true")
    d.add_argument("--no-examples", action="store_true")
    d.set_defaults(fn=_demo)
    r = sub.add_parser("run", help="VEE on your own meter file")
    r.add_argument("--readings", required=True, help="CSV with timestamp,kwh (interval start; offset or naive local time)")
    r.add_argument("--register", help="CSV with date,kwh (daily register advance)")
    r.add_argument("--temperature", help="CSV with date,t_mean (daily mean temperature)")
    r.add_argument("--tz", default="Europe/Zurich")
    r.add_argument("--long-method", choices=["regression", "profile"], default="regression")
    r.add_argument("--max-interval-kwh", type=float, default=3.0)
    r.add_argument("--no-register-prorate", action="store_true")
    r.add_argument("--out", default="out/run")
    r.set_defaults(fn=_run)
    i = sub.add_parser("inject", help="write a faulty meter file + answer key from the open dataset")
    i.add_argument("--start", default="2009-10-01")
    i.add_argument("--end", default="2009-11-01")
    i.add_argument("--seed", type=int, default=7)
    i.add_argument("--tz", default="Europe/Paris")
    i.add_argument("--out", default="out/inject")
    i.set_defaults(fn=_inject)
    b = sub.add_parser("benchmark", help="gap-length benchmark on a period")
    b.add_argument("--start", default="2007-01-01")
    b.add_argument("--end", default="2008-01-01")
    b.add_argument("--per-length", type=int, default=40)
    b.add_argument("--seed", type=int, default=3)
    b.set_defaults(fn=_benchmark)
    q = sub.add_parser("prepare-data", help="rebuild the shipped data from the original downloads (internet)")
    q.add_argument("--cache", default="cache/raw")
    q.set_defaults(fn=_prepare)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
