import json

import numpy as np
import pandas as pd
import pytest
from conftest import TZ, synthetic

from interval_vee import export
from interval_vee.cli import main
from interval_vee.estimate import Context
from interval_vee.evaluate import benchmark_gaps, detection, monthly_energy, summarize_benchmark
from interval_vee.faults import KINDS, Event, daily_register, inject, source_gap_events
from interval_vee.meterio import ingest
from interval_vee.source import load_temperature, load_truth
from interval_vee.vee import FLAGS, VeeConfig, run_vee

SMALL = {"gap": (6, [1, 8, 96]), "zero_run": (3, [4, 8]), "stuck": (3, [6, 8]), "spike": (4, [1]), "negative": (3, [1]),
         "duplicate": (4, [1]), "clock_shift": (2, [4, 8]), "register_mismatch": (2, [0])}


@pytest.fixture(scope="module")
def mix():
    s = synthetic(days=56, start="2009-10-05", noise=0.03, seed=5)
    f = inject(s, seed=2, plan=SMALL, tz=TZ)
    res = run_vee(ingest(f.rows, TZ, grid=s.index), VeeConfig(tz=TZ, edited_at="t"), register=f.register)
    return s, f, res


def test_inject_is_deterministic_and_covers_every_kind(mix):
    s, f, _ = mix
    f2 = inject(s, seed=2, plan=SMALL, tz=TZ)
    assert f.rows.equals(f2.rows)
    assert {e.kind for e in f.events} == set(KINDS)
    assert any(e.detail.get("cause") == "missed DST switch" for e in f.events)


def test_injected_events_do_not_overlap(mix):
    _, f, _ = mix
    spans = sorted((e.start, e.end) for e in f.events)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:], strict=False))


def test_register_is_the_true_daily_energy(mix):
    s, f, _ = mix
    ctx = Context(s.index, TZ)
    lo, hi = ctx.day_bounds(f.register.index[3])
    assert f.register.iloc[3] == pytest.approx(s.to_numpy()[lo:hi].sum(), abs=0.005)


def test_full_mix_detects_structural_faults_exactly(mix):
    s, f, res = mix
    det = detection(f.events + source_gap_events(f.truth), res.flags).set_index("rule")
    for rule in ("GAP", "NEGATIVE", "DUPLICATE", "CLOCK_SHIFT", "ZERO_RUN", "STUCK"):
        assert det.loc[rule, "recall"] == 1.0, rule
        assert det.loc[rule, "precision"] == 1.0, rule
    for e in f.events:
        if e.kind == "clock_shift":
            np.testing.assert_allclose(res.value[e.start:e.end], s.to_numpy()[e.start:e.end])


def test_every_rejected_or_missing_interval_gets_a_value(mix):
    _, _, res = mix
    need = (res.flags & 0b11111) > 0
    assert need.any() and not np.isnan(res.value[need]).any()
    assert set(res.status[need]) <= {"E", "C"}


def test_detection_counts_misses_and_false_runs():
    flags = np.zeros(100, dtype=np.int64)
    flags[10] = FLAGS["SPIKE"]
    flags[50] = FLAGS["SPIKE"]
    ev = [Event("spike", 10, 1), Event("spike", 30, 1)]
    d = detection(ev, flags).set_index("rule").loc["SPIKE"]
    assert d["recall"] == 0.5 and d["precision"] == 0.5 and d["false_runs"] == 1


def test_register_flag_on_a_day_with_another_fault_counts_as_caught():
    flags = np.zeros(200, dtype=np.int64)
    flags[96:192] = FLAGS["REGISTER_MISMATCH"]
    d = detection([Event("spike", 120, 1)], flags).set_index("rule").loc["REGISTER_MISMATCH"]
    assert d["caught_other"] == 1 and d["false_runs"] == 0


def test_monthly_energy_raw_vs_vee(mix):
    s, f, res = mix
    ing = ingest(f.rows, TZ, grid=s.index)
    m = monthly_energy(res, f.truth, ing.slot, ing.value, TZ)
    assert list(m["month"]) == ["2009-10", "2009-11"]
    assert (m["vee_err_pct"].abs() < 2).all()
    assert m["raw_err_pct"].abs().max() > m["vee_err_pct"].abs().max()


def test_benchmark_on_shipped_data_small():
    b = benchmark_gaps(load_truth(), load_temperature(), "2009-01-01", "2009-03-01", [1, 96], per_length=3, seed=1)
    s = summarize_benchmark(b)
    assert set(s["method"]) == {"linear", "profile", "regression", "vee", "vee+register"}
    reg = s[s["method"] == "vee+register"]
    assert reg.set_index("length").loc[96, "energy_abs_p90_pct"] < 0.5  # day-long gap: register fixes the energy


def test_shipped_data_spans_the_source_period():
    s = load_truth()
    assert str(s.index[0]) == "2006-12-16 16:30:00+00:00"
    assert len(s) == 138350
    assert s.notna().sum() == 136551
    assert s.max() < 3.0 and s.min() > 0
    t = load_temperature()
    assert len(t) >= 1440 and t.between(-20, 40).all()


def test_export_columns_and_flag_text(mix, tmp_path):
    _, _, res = mix
    p1, p2 = export.write(res, tmp_path, TZ)
    df = pd.read_csv(p1, dtype=str, keep_default_na=False)
    assert list(df.columns) == ["timestamp_utc", "timestamp_local", "kwh", "status", "flags", "method"]
    assert df["timestamp_local"].iloc[0].endswith("+02:00")
    assert any("CLOCK_SHIFT" in x for x in df["flags"])
    aud = pd.read_csv(p2, dtype=str, keep_default_na=False)
    assert list(aud.columns) == ["timestamp_utc", "rule", "method", "original_kwh", "new_kwh", "edited_at", "note"]
    assert (aud["edited_at"] == "t").all()


def test_cli_inject_then_run(tmp_path, capsys):
    assert main(["inject", "--start", "2009-10-01", "--end", "2009-10-15", "--out", str(tmp_path / "inj")]) == 0
    key = json.loads((tmp_path / "inj" / "answer_key.json").read_text())
    assert len(key) > 0
    assert main(["run", "--readings", str(tmp_path / "inj" / "meter_faulty.csv"), "--register",
                 str(tmp_path / "inj" / "register_daily.csv"), "--tz", "Europe/Paris", "--out", str(tmp_path / "run")]) == 0
    out = pd.read_csv(tmp_path / "run" / "vee_export.csv")
    assert out["kwh"].notna().all()
    assert "wrote" in capsys.readouterr().out


def test_daily_register_skips_incomplete_days():
    s = synthetic(days=5)
    v = s.to_numpy().copy()
    v[200] = np.nan
    reg = daily_register(v, Context(s.index, TZ))
    assert len(reg) == 4
