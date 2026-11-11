import numpy as np
import pandas as pd
from conftest import TZ, rows_from, synthetic

from interval_vee.estimate import Context
from interval_vee.faults import daily_register
from interval_vee.meterio import ingest
from interval_vee.vee import FLAGS, VeeConfig, find_clock_shifts, run_vee, runs

CFG = VeeConfig(tz=TZ, edited_at="2026-01-01T00:00:00Z")


def _run(series, rows=None, register=None, cfg=CFG):
    rows = rows_from(series) if rows is None else rows
    return run_vee(ingest(rows, TZ, grid=series.index), cfg, register=register)


def _set(rows, i, value):
    rows = rows.copy()
    rows.loc[i, "kwh"] = value
    return rows


def test_runs_helper():
    assert runs([0, 1, 1, 0, 1]) == [(1, 2), (4, 1)]
    assert runs([]) == []


def test_clean_data_passes_unchanged(week6):
    res = _run(week6)
    assert (res.flags == 0).all()
    assert (res.status == "A").all()
    np.testing.assert_allclose(res.value, week6.to_numpy())
    assert len(res.audit) == 0


def test_negative_value_is_rejected_and_estimated(week6):
    res = _run(week6, _set(rows_from(week6), 500, "-0.200"))
    assert res.flags[500] & FLAGS["NEGATIVE"]
    assert res.status[500] == "E" and res.method[500] == "linear"
    assert res.value[500] > 0


def test_gross_spike_above_physical_limit(week6):
    res = _run(week6, _set(rows_from(week6), 900, "57.000"))
    assert res.flags[900] & FLAGS["SPIKE"]
    assert abs(res.value[900] - week6.iloc[900]) < 0.05


def test_relative_spike_rule_and_its_threshold(week6):
    v = week6.iloc[1000]
    res = _run(week6, _set(rows_from(week6), 1000, f"{v + 1.5:.3f}"))
    assert res.flags[1000] & FLAGS["SPIKE"]
    res = _run(week6, _set(rows_from(week6), 1000, f"{v + 0.3:.3f}"))
    assert not res.flags[1000] & FLAGS["SPIKE"]


def test_zero_run_needs_four_intervals(week6):
    rows = rows_from(week6)
    for i in range(200, 203):
        rows = _set(rows, i, "0.000")
    assert not (_run(week6, rows).flags & FLAGS["ZERO_RUN"]).any()
    rows = _set(rows, 203, "0.000")
    res = _run(week6, rows)
    assert list(np.flatnonzero(res.flags & FLAGS["ZERO_RUN"])) == [200, 201, 202, 203]
    assert (res.status[200:204] == "E").all()


def test_stuck_value_flags_repeats_but_keeps_first_reading(week6):
    rows = rows_from(week6)
    first = rows.loc[299, "kwh"]
    for i in range(300, 306):
        rows = _set(rows, i, first)
    res = _run(week6, rows)
    assert list(np.flatnonzero(res.flags & FLAGS["STUCK"])) == list(range(300, 306))
    assert res.status[299] == "A"


def test_four_repeats_are_not_stuck(week6):
    rows = rows_from(week6)
    for i in range(300, 304):
        rows = _set(rows, i, rows.loc[299, "kwh"])
    assert not (_run(week6, rows).flags & FLAGS["STUCK"]).any()


def test_identical_duplicate_is_dropped_without_change(week6):
    rows = rows_from(week6)
    rows = pd.concat([rows, rows.iloc[[700]]], ignore_index=True)
    res = _run(week6, rows)
    assert res.flags[700] & FLAGS["DUPLICATE"]
    assert res.value[700] == week6.iloc[700]
    assert res.audit["method"].tolist() == ["duplicate_dropped"]


def test_conflicting_duplicate_keeps_reading_closest_to_neighbours(week6):
    rows = rows_from(week6)
    rows = pd.concat([rows.iloc[[800]].assign(kwh="1.900"), rows], ignore_index=True)  # wrong row first in file
    res = _run(week6, rows)
    assert res.value[800] == week6.iloc[800]
    assert res.method[800] == "duplicate_resolved"
    assert "1.900" in res.audit["original_kwh"].iloc[0]


def test_gap_is_flagged_and_filled(week6):
    rows = rows_from(week6).drop(index=range(1200, 1208))
    res = _run(week6, rows)
    assert (res.flags[1200:1208] & FLAGS["GAP"]).all()
    assert (res.status[1200:1208] == "E").all()
    assert set(res.method[1200:1208]) == {"linear"}


def test_long_gap_uses_profile_without_temperature(week6):
    rows = rows_from(week6).drop(index=range(2000, 2096))
    res = _run(week6, rows)
    assert set(res.method[2000:2096]) == {"profile"}
    np.testing.assert_allclose(res.value[2000:2096], week6.to_numpy()[2000:2096], atol=0.002)


def _shifted_rows(series, a, b, s):
    """Block [a, b) labelled s slots off, rows in emission (true time) order."""
    lab = series.index.tz_convert(TZ).strftime("%Y-%m-%d %H:%M")
    emit = []
    for t in range(len(series)):
        label = t + s if a <= t < b else t
        emit.append((label, series.iloc[t], t))
    emit = [e for e in emit if 0 <= e[0] < len(series)]
    emit.sort(key=lambda e: e[2])
    return pd.DataFrame({"timestamp": [lab[e[0]] for e in emit], "kwh": [f"{e[1]:.3f}" for e in emit]})


def test_clock_ahead_block_is_moved_back_exactly():
    s = synthetic(noise=0.03)
    res = _run(s, _shifted_rows(s, 1500, 1500 + 192, 4))
    assert res.summary["clock_shifts"] == [{"shift": 4, "start": 1500, "end": 1692, "gap": 1500, "dup": 1692}]
    np.testing.assert_allclose(res.value, s.to_numpy())
    assert (res.status[1500:1692] == "C").all()
    assert (res.flags[1500:1696] & FLAGS["CLOCK_SHIFT"]).all()


def test_clock_behind_block_is_moved_forward_exactly():
    s = synthetic(noise=0.03)
    res = _run(s, _shifted_rows(s, 2008, 2008 + 288, -8))
    assert [x["shift"] for x in res.summary["clock_shifts"]] == [-8]
    np.testing.assert_allclose(res.value, s.to_numpy())
    assert res.method[2100] == "time_shift_+120min"


def test_find_clock_shifts_ignores_lengths_that_are_not_whole_hours():
    count = np.ones(2000, dtype=int)
    count[100:103] = 0
    count[400:403] = 2
    assert find_clock_shifts(count, CFG) == []
    count[100:104] = 0
    count[400:404] = 2
    assert find_clock_shifts(count, CFG)[0]["shift"] == 4


def test_register_mismatch_day_is_prorated_to_register():
    s = synthetic()
    ctx = Context(s.index, TZ)
    reg = daily_register(s.to_numpy(), ctx)
    lo, hi = ctx.day_bounds(pd.Timestamp("2009-10-14").date())
    rows = rows_from(s)
    for i in range(lo, hi):
        rows.loc[i, "kwh"] = f"{s.iloc[i] * 0.8:.3f}"
    res = _run(s, rows, register=reg)
    assert (res.flags[lo:hi] & FLAGS["REGISTER_MISMATCH"]).all()
    assert (res.method[lo:hi] == "register_prorate").all()
    assert abs(res.value[lo:hi].sum() - reg.loc[pd.Timestamp("2009-10-14").date()]) < 1e-9
    assert res.summary["register_days_mismatch"] == 1


def test_register_within_tolerance_is_not_flagged():
    s = synthetic()
    ctx = Context(s.index, TZ)
    reg = daily_register(s.to_numpy(), ctx) * 1.01
    res = _run(s, register=reg)
    assert not (res.flags & FLAGS["REGISTER_MISMATCH"]).any()
    assert res.summary["register_days_checked"] == len(reg) - 0


def test_estimates_are_scaled_to_register_on_gap_days():
    s = synthetic(noise=0.03, seed=3)
    ctx = Context(s.index, TZ)
    reg = daily_register(s.to_numpy(), ctx)
    rows = rows_from(s).drop(index=range(1000, 1040))
    res = _run(s, rows, register=reg)
    d = ctx.day_code[1000]
    lo, hi = ctx.day_start[d], ctx.day_end[d]
    assert abs(res.value[lo:hi].sum() - reg.iloc[d]) < 1e-9
    assert res.method[1010].endswith("+register")
    res2 = _run(s, rows, register=reg, cfg=VeeConfig(tz=TZ, use_register=False))
    assert not res2.method[1010].endswith("+register")


def test_audit_row_has_original_new_rule_method_and_time(week6):
    res = _run(week6, _set(rows_from(week6), 500, "-0.200"))
    row = res.audit.iloc[0]
    assert row["rule"] == "NEGATIVE" and row["method"] == "linear"
    assert row["original_kwh"] == -0.2
    assert row["edited_at"] == "2026-01-01T00:00:00Z"
    assert abs(row["new_kwh"] - res.value[500]) < 1e-12
