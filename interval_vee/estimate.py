"""Estimation methods for missing or rejected intervals.

- linear: straight line between the last good value before and the first good value after the gap.
- profile: mean of the same local clock time on the same weekday, 1..K weeks before and after, scaled to the level of
  the 24 h of good data around the gap (factor clipped to 0.5..2).
- regression: daily energy from a least-squares fit on nearby complete days (heating degree days + weekend), spread
  over the day with the profile shape. Needs daily mean temperature; falls back to profile otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEK_NS = 7 * 86400 * 10**9
DAY_SLOTS = 96


class Context:
    """Calendar lookups for a UTC grid: local days, local clock times and weekdays."""

    def __init__(self, grid: pd.DatetimeIndex, tz: str, temperature: pd.Series | None = None, profile_weeks: int = 4,
                 hdd_base: float = 15.0):
        self.grid = grid
        local = grid.tz_convert(tz)
        naive = local.tz_localize(None)
        self.local_ns = naive.as_unit("ns").asi8
        uniq = pd.Index(self.local_ns)
        keep = ~uniq.duplicated()
        self._lookup = pd.Index(self.local_ns[keep])
        self._lookup_pos = np.flatnonzero(keep)
        days = naive.normalize()
        codes, uniques = pd.factorize(days)
        self.day_code = codes
        self.days = [d.date() for d in uniques]
        self.day_index = {d: i for i, d in enumerate(self.days)}
        starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
        ends = np.r_[starts[1:], len(codes)]
        expected = []
        for d in uniques:
            t0 = pd.Timestamp(d).tz_localize(tz)
            t1 = (pd.Timestamp(d) + pd.Timedelta(days=1)).tz_localize(tz)
            expected.append(int((t1 - t0) / pd.Timedelta(minutes=15)))
        self.day_start, self.day_end, self.day_expected = starts, ends, np.array(expected)
        self.weekend = np.array([d.weekday() >= 5 for d in self.days])
        self.profile_weeks = profile_weeks
        self.hdd_base = hdd_base
        self.temp = None
        if temperature is not None:
            t = pd.Series(temperature)
            t.index = [pd.Timestamp(x).date() for x in t.index]
            self.temp = np.array([t.get(d, np.nan) for d in self.days], dtype=float)

    def day_bounds(self, day) -> tuple[int | None, int | None]:
        i = self.day_index.get(day)
        if i is None or self.day_end[i] - self.day_start[i] != self.day_expected[i]:
            return None, None
        return int(self.day_start[i]), int(self.day_end[i])

    def shifted(self, slots: np.ndarray, weeks: int) -> np.ndarray:
        pos = self._lookup.get_indexer(self.local_ns[slots] + weeks * WEEK_NS)
        return np.where(pos >= 0, self._lookup_pos[np.maximum(pos, 0)], -1)

    def daily_totals(self, base: np.ndarray) -> np.ndarray:
        """Energy per local day, NaN unless the day is complete."""
        good = ~np.isnan(base)
        s = np.bincount(self.day_code, weights=np.where(good, base, 0.0), minlength=len(self.days))
        c = np.bincount(self.day_code, weights=good.astype(float), minlength=len(self.days))
        size = self.day_end - self.day_start
        return np.where((c == self.day_expected) & (size == self.day_expected), s, np.nan)


def linear(base: np.ndarray, a: int, b: int) -> np.ndarray:
    n = b - a
    if a == 0 or b >= len(base) or np.isnan(base[a - 1]) or np.isnan(base[b]):
        return np.full(n, np.nan)
    w = np.arange(1, n + 1) / (n + 1)
    return base[a - 1] + (base[b] - base[a - 1]) * w


def profile_raw(base: np.ndarray, slots: np.ndarray, ctx: Context) -> np.ndarray:
    acc = np.zeros(len(slots))
    cnt = np.zeros(len(slots))
    for k in range(1, ctx.profile_weeks + 1):
        for w in (-k, k):
            pos = ctx.shifted(slots, w)
            v = np.where(pos >= 0, base[np.maximum(pos, 0)], np.nan)
            ok = ~np.isnan(v)
            acc[ok] += v[ok]
            cnt[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, acc / cnt, np.nan)


def profile(base: np.ndarray, a: int, b: int, ctx: Context, window: int = DAY_SLOTS) -> np.ndarray:
    est = profile_raw(base, np.arange(a, b), ctx)
    nb = np.r_[np.arange(max(0, a - window), a), np.arange(b, min(len(base), b + window))]
    nb = nb[~np.isnan(base[nb])]
    if len(nb):
        p = profile_raw(base, nb, ctx)
        ok = ~np.isnan(p)
        if ok.sum() >= 8 and p[ok].sum() > 0:
            est = est * float(np.clip(base[nb][ok].sum() / p[ok].sum(), 0.5, 2.0))
    return est


def regression(base: np.ndarray, a: int, b: int, ctx: Context, window_days: int = 42, min_days: int = 14) -> np.ndarray:
    n = b - a
    out = np.full(n, np.nan)
    if ctx.temp is None:
        return out
    totals = ctx.daily_totals(base)
    hdd = np.maximum(0.0, ctx.hdd_base - ctx.temp)
    slots = np.arange(a, b)
    codes = ctx.day_code[slots]
    for d in np.unique(codes):
        if np.isnan(hdd[d]):
            return np.full(n, np.nan)
        lo, hi = max(0, d - window_days), min(len(ctx.days), d + window_days + 1)
        idx = np.arange(lo, hi)
        ok = ~np.isnan(totals[idx]) & ~np.isnan(hdd[idx])
        idx = idx[ok]
        if len(idx) < min_days:
            return np.full(n, np.nan)
        X = np.c_[np.ones(len(idx)), hdd[idx], ctx.weekend[idx].astype(float)]
        coef, *_ = np.linalg.lstsq(X, totals[idx], rcond=None)
        e_day = max(0.0, float(coef @ np.array([1.0, hdd[d], float(ctx.weekend[d])])))
        day_slots = np.arange(ctx.day_start[d], ctx.day_end[d])
        shape = profile_raw(base, day_slots, ctx)
        if np.all(np.isnan(shape)):
            return np.full(n, np.nan)
        shape = np.where(np.isnan(shape), np.nanmean(shape), shape)
        if shape.sum() <= 0:
            return np.full(n, np.nan)
        in_day = codes == d
        out[in_day] = shape[slots[in_day] - ctx.day_start[d]] * e_day / shape.sum()
    return out


def estimate_run(base: np.ndarray, a: int, b: int, ctx: Context, linear_max: int = 16, long_method: str = "regression",
                 regression_min: int = 96) -> tuple[np.ndarray, str]:
    """Estimate slots [a, b). Returns (values, method name); NaN where nothing could be estimated."""
    if b - a <= linear_max:
        est = linear(base, a, b)
        if not np.isnan(est).any():
            return est, "linear"
    if long_method == "regression" and b - a >= regression_min:
        est = regression(base, a, b, ctx)
        if not np.isnan(est).any():
            return est, "regression"
    est = profile(base, a, b, ctx)
    if np.isnan(est).any():
        lin = linear(base, a, b)
        if not np.isnan(lin).any():
            return lin, "linear"
    return est, "profile"
