import numpy as np
import pandas as pd
import pytest
from conftest import TZ, synthetic

from interval_vee.estimate import Context, estimate_run, linear, profile, profile_raw, regression


def test_linear_is_a_straight_line_between_anchors():
    base = np.array([1.0, np.nan, np.nan, np.nan, 2.0])
    np.testing.assert_allclose(linear(base, 1, 4), [1.25, 1.5, 1.75])


def test_linear_needs_both_anchors():
    assert np.isnan(linear(np.array([np.nan, np.nan, 1.0]), 0, 2)).all()
    assert np.isnan(linear(np.array([1.0, np.nan, np.nan]), 1, 3)).all()


def test_context_counts_dst_days():
    s = synthetic(days=7, start="2009-10-22")
    ctx = Context(s.index, TZ)
    d = ctx.day_index[pd.Timestamp("2009-10-25").date()]
    assert ctx.day_expected[d] == 100
    lo, hi = ctx.day_bounds(pd.Timestamp("2009-10-25").date())
    assert hi - lo == 100


def test_partial_day_at_grid_edge_has_no_bounds():
    s = synthetic(days=3)[10:]
    ctx = Context(s.index, TZ)
    assert ctx.day_bounds(pd.Timestamp("2009-10-05").date()) == (None, None)


def test_daily_totals_nan_for_incomplete_days(week6):
    v = week6.to_numpy().copy()
    v[5] = np.nan
    tot = Context(week6.index, TZ).daily_totals(v)
    assert np.isnan(tot[0]) and not np.isnan(tot[1])


def test_profile_reproduces_a_weekly_pattern_across_dst(week6):
    base = week6.to_numpy().copy()
    a, b = 1900, 2100  # around the autumn change on 25 Oct 2009
    base[a:b] = np.nan
    ctx = Context(week6.index, TZ)
    np.testing.assert_allclose(profile(base, a, b, ctx), week6.to_numpy()[a:b], atol=1e-9)


def test_profile_level_follows_neighbouring_data(week6):
    base = week6.to_numpy().copy() * 1.0
    a, b = 2000, 2096
    base[a - 96:b + 96] *= 1.5  # house uses 50 % more around the gap than in other weeks
    truth = base[a:b].copy()
    base[a:b] = np.nan
    est = profile(base, a, b, Context(week6.index, TZ))
    assert abs(est.sum() / truth.sum() - 1) < 0.02


def test_profile_raw_without_history_is_nan():
    s = synthetic(days=3)
    assert np.isnan(profile_raw(s.to_numpy(), np.arange(10), Context(s.index, TZ))).all()


def test_regression_recovers_heating_slope():
    s = synthetic(days=70, start="2009-10-05")
    ctx0 = Context(s.index, TZ)
    rng = np.random.default_rng(1)
    temp = pd.Series(rng.uniform(-2, 18, len(ctx0.days)), index=ctx0.days)
    hdd = np.maximum(0, 15 - temp.to_numpy())
    v = s.to_numpy().copy()
    for d in range(len(ctx0.days)):  # day energy = shape scaled to 8 + 0.6 * HDD kWh
        lo, hi = ctx0.day_start[d], ctx0.day_end[d]
        v[lo:hi] = v[lo:hi] / v[lo:hi].sum() * (8 + 0.6 * hdd[d])
    ctx = Context(s.index, TZ, temperature=temp)
    d = 40
    lo, hi = ctx.day_start[d], ctx.day_end[d]
    base = v.copy()
    base[lo:hi] = np.nan
    est = regression(base, lo, hi, ctx)
    assert est.sum() == pytest.approx(8 + 0.6 * hdd[d], rel=0.03)


def test_regression_without_temperature_returns_nan(week6):
    base = week6.to_numpy().copy()
    base[1000:1096] = np.nan
    assert np.isnan(regression(base, 1000, 1096, Context(week6.index, TZ))).all()


@pytest.mark.parametrize("length,expected", [(1, "linear"), (16, "linear"), (17, "profile"), (95, "profile"), (96, "profile")])
def test_estimate_run_method_by_length(week6, length, expected):
    base = week6.to_numpy().copy()
    base[2000:2000 + length] = np.nan
    _, meth = estimate_run(base, 2000, 2000 + length, Context(week6.index, TZ))
    assert meth == expected  # 96 falls back to profile: no temperature given
