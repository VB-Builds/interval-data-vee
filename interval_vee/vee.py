"""Validation, estimation and editing of 15-minute interval data.

Order of work:
1. duplicates and clock shifts (needs all rows of a slot), 2. value rules (negative, spike, zero run, stuck),
3. gaps, 4. register-vs-interval check on complete days, 5. estimation of every missing or rejected interval,
6. optional: estimates of a day scaled so the day matches the register advance.
Every changed interval gets an audit row: original value, new value, rule, method, timestamp of the edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .estimate import Context, estimate_run, profile_raw
from .meterio import Ingested

FLAGS = {"GAP": 1, "ZERO_RUN": 2, "SPIKE": 4, "NEGATIVE": 8, "STUCK": 16, "DUPLICATE": 32, "CLOCK_SHIFT": 64,
         "REGISTER_MISMATCH": 128}
REJECT = FLAGS["ZERO_RUN"] | FLAGS["SPIKE"] | FLAGS["NEGATIVE"] | FLAGS["STUCK"]


@dataclass
class VeeConfig:
    tz: str = "Europe/Paris"
    max_interval_kwh: float = 3.0  # physical ceiling: 12 kW for 15 minutes
    spike_rise_kwh: float = 0.9  # value above both neighbours by more than this
    zero_run_min: int = 4  # >= 1 h of exact zeros
    stuck_repeats: int = 5  # >= 6 identical non-zero values in a row
    max_shift_slots: int = 8  # clock shifts of 1 h or 2 h
    max_shift_block_days: float = 7.0
    register_tol_kwh: float = 0.05
    register_tol_rel: float = 0.02
    linear_max: int = 16  # gaps up to 4 h: linear interpolation (chosen on 2007, see README)
    long_method: str = "regression"  # "regression" (temperature) or "profile" for gaps of a day or more
    regression_min: int = 96  # shorter gaps above linear_max use the profile
    profile_weeks: int = 4
    hdd_base: float = 15.0
    use_register: bool = True  # prorate estimates to the daily register advance
    edited_at: str | None = None


@dataclass
class VeeResult:
    grid: pd.DatetimeIndex
    value: np.ndarray
    received: np.ndarray  # value as received at the slot label (first row), NaN if none
    status: np.ndarray  # A actual, C corrected, E estimated, M missing
    flags: np.ndarray
    method: np.ndarray
    audit: pd.DataFrame
    summary: dict = field(default_factory=dict)

    def flag_names(self, i: int) -> str:
        return ";".join(k for k, b in FLAGS.items() if self.flags[i] & b)


def runs(mask) -> list[tuple[int, int]]:
    """(start, length) of every run of True values."""
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    d = np.diff(np.r_[0, m, 0])
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1)
    return list(zip(s.tolist(), (e - s).tolist(), strict=True))


class _Cands:
    def __init__(self, ing: Ingested):
        o = np.lexsort((ing.order, ing.slot))
        self.slot, self.value = ing.slot[o], ing.value[o]
        n = len(ing.grid)
        self.count = np.bincount(self.slot, minlength=n)
        self.first = np.searchsorted(self.slot, np.arange(n))

    def get(self, t: int) -> np.ndarray:
        return self.value[self.first[t]:self.first[t] + self.count[t]]


def _shift_evidence(c: _Cands, ref: np.ndarray, ctx: Context, g0: int, d0: int, n: int, sign: int) -> float:
    """> 0 when the block's values fit the usual weekly profile better at their corrected time than at their label."""
    if sign > 0:
        true = np.arange(g0, d0)
        x = np.array([c.get(u)[0] if c.count[u] else np.nan for u in range(g0 + n, d0 + n)])
    else:
        true = np.arange(d0 + n, g0 + n)
        x = np.array([(c.get(u)[1] if u < d0 + n else c.get(u)[0]) if c.count[u] else np.nan for u in range(d0, g0)])
    r_true = profile_raw(ref, true, ctx)
    r_label = profile_raw(ref, true + sign * n, ctx)
    ok = ~np.isnan(x) & ~np.isnan(r_true) & ~np.isnan(r_label)
    if ok.sum() < 8:
        return 0.0
    return float(np.sum((x[ok] - r_label[ok]) ** 2) - np.sum((x[ok] - r_true[ok]) ** 2))


def _search(comp, i, ug, ud, total, picked, best) -> None:
    if total > best[0]:
        best[0], best[1] = total, list(picked)
    for j in range(i, len(comp)):
        w, g, d = comp[j]
        if g not in ug and d not in ud:
            picked.append(comp[j])
            _search(comp, j + 1, ug | {g}, ud | {d}, total + w, picked, best)
            picked.pop()


def _best_matching(edges: list[tuple[float, int, int]]) -> list[tuple[float, int, int]]:
    """Maximum-weight set of (weight, gap, dup) edges with every gap and every dup used at most once."""
    chosen: list[tuple[float, int, int]] = []
    left = sorted(edges, reverse=True)
    while left:  # split into connected components
        comp, nodes = [left[0]], {("g", left[0][1]), ("d", left[0][2])}
        grown = True
        while grown:
            grown = False
            for e in left:
                if e not in comp and (("g", e[1]) in nodes or ("d", e[2]) in nodes):
                    comp.append(e)
                    nodes |= {("g", e[1]), ("d", e[2])}
                    grown = True
        left = [e for e in left if e not in comp]
        if len(comp) > 16:  # large components: greedy by weight
            ug, ud = set(), set()
            for e in sorted(comp, reverse=True):
                if e[1] not in ug and e[2] not in ud:
                    chosen.append(e)
                    ug.add(e[1])
                    ud.add(e[2])
            continue
        best: list = [0.0, []]
        _search(comp, 0, frozenset(), frozenset(), 0.0, [], best)
        chosen.extend(best[1])
    return chosen


def find_clock_shifts(count: np.ndarray, cfg: VeeConfig, c: _Cands | None = None, ref: np.ndarray | None = None,
                      ctx: Context | None = None) -> list[dict]:
    """Pair a gap of L slots with a run of L double readings: the footprint of a block whose clock was off by L slots.

    Each candidate pair is weighted by how much better its block fits the usual weekly profile after the shift
    (summed over the block); the set of pairs with the largest total weight wins (each gap and duplicate run used
    once). A pair without profile evidence keeps a tiny weight, nearer pairs slightly more, so a lone footprint is
    still repaired.
    """
    gaps = [(s, n) for s, n in runs(count == 0) if n % 4 == 0 and n <= cfg.max_shift_slots]
    dups = [(s, n) for s, n in runs(count >= 2) if n % 4 == 0 and n <= cfg.max_shift_slots and np.all(count[s:s + n] == 2)]
    maxblock = int(cfg.max_shift_block_days * 96)
    pairs = []
    for g0, n in gaps:
        for d0, dn in dups:
            if dn != n:
                continue
            if d0 >= g0 + n and d0 - g0 <= maxblock:
                dist, sign = d0 - (g0 + n), +1
            elif d0 + n <= g0 and g0 - d0 <= maxblock:
                dist, sign = g0 - (d0 + n), -1
            else:
                continue
            ev = _shift_evidence(c, ref, ctx, g0, d0, n, sign) if c is not None and ctx is not None else 0.0
            pairs.append((max(ev, 0.0) + 1e-6 / (1 + dist), g0, d0, n, sign))
    info = {(g0, d0): (n, sign) for _, g0, d0, n, sign in pairs}
    out = []
    for _, g0, d0 in _best_matching([(w, g0, d0) for w, g0, d0, _, _ in pairs]):
        n, sign = info[(g0, d0)]
        if sign > 0:  # clock ahead: block [g0, d0) was labelled n slots late
            out.append({"shift": n, "start": g0, "end": d0, "gap": g0, "dup": d0})
        else:  # clock behind: block [d0+n, g0+n) was labelled n slots early
            out.append({"shift": -n, "start": d0 + n, "end": g0 + n, "gap": g0, "dup": d0})
    return sorted(out, key=lambda x: x["start"])


def run_vee(ing: Ingested, cfg: VeeConfig | None = None, register: pd.Series | None = None,
            temperature: pd.Series | None = None) -> VeeResult:
    cfg = cfg or VeeConfig()
    edited_at = cfg.edited_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    grid = ing.grid
    n = len(grid)
    c = _Cands(ing)
    count = c.count.copy()
    val = np.full(n, np.nan)
    received = np.full(n, np.nan)
    has = count >= 1
    received[has] = c.value[c.first[has]]
    single = count == 1
    val[single] = received[single]
    flags = np.zeros(n, dtype=np.int64)
    status = np.full(n, "A", dtype="<U1")
    method = np.full(n, "", dtype=object)
    audit: list[tuple] = []

    def log(t, rule, meth, orig, new, note=""):
        audit.append((t, rule, meth, orig, new, edited_at, note))

    ctx = Context(grid, cfg.tz, temperature, cfg.profile_weeks, cfg.hdd_base)

    # 1a. clock shifts: move the block back to its true slots
    shifts = find_clock_shifts(count, cfg, c, val, ctx)
    new = val.copy()
    for sh in shifts:
        s, a, b = sh["shift"], sh["start"], sh["end"]
        if s > 0:
            moves = [(t, t - s, 0) for t in range(a + s, b)] + [(t, t - s, 0) for t in range(b, b + s)]
            for t in range(b, b + s):
                new[t] = c.get(t)[1]
            lo, hi = a, b + s
        else:
            k = -s
            moves = [(t, t + k, 1) for t in range(a - k, a)] + [(t, t + k, 0) for t in range(a, b - k)]
            for t in range(a - k, a):
                new[t] = c.get(t)[0]
            lo, hi = a - k, b
        for src, dst, which in moves:
            cs = c.get(src)
            new[dst] = cs[which] if len(cs) > which else np.nan
        count[lo:hi] = 1
        flags[lo:hi] |= FLAGS["CLOCK_SHIFT"]
        for t in range(a, b):
            status[t] = "C"
            method[t] = f"time_shift_{-s * 15:+d}min"
            log(t, "CLOCK_SHIFT", method[t], received[t], new[t], f"reading labelled {s * 15:+d} min off")
    val = new

    # 1b. remaining duplicates
    for t in np.flatnonzero(count >= 2):
        cs = c.get(t)
        flags[t] |= FLAGS["DUPLICATE"]
        if np.all(cs == cs[0]):
            val[t] = cs[0]
            log(t, "DUPLICATE", "duplicate_dropped", cs[0], cs[0], f"{len(cs)} identical rows")
            continue
        nb = [val[u] for u in (t - 1, t + 1) if 0 <= u < n and count[u] == 1 and not np.isnan(val[u])]
        ref = float(np.mean(nb)) if nb else float(np.median(cs))
        val[t] = cs[np.argmin(np.abs(cs - ref))]
        method[t] = "duplicate_resolved"
        log(t, "DUPLICATE", "duplicate_resolved", "|".join(f"{x:.3f}" for x in cs), val[t], "closest to neighbours kept")

    # 2. value rules
    have = ~np.isnan(val)
    neg = have & (val < 0)
    prev = np.r_[np.nan, val[:-1]]
    nxt = np.r_[val[1:], np.nan]
    with np.errstate(invalid="ignore"):
        nb = np.fmax(prev, nxt)
        spike = have & ((val > cfg.max_interval_kwh) | (val - nb > cfg.spike_rise_kwh))
    zero = np.zeros(n, dtype=bool)
    for s, m in runs(have & (val == 0)):
        if m >= cfg.zero_run_min:
            zero[s:s + m] = True
    stuck = np.zeros(n, dtype=bool)
    for s, m in runs(have & (val != 0) & (val == prev)):
        if m >= cfg.stuck_repeats:
            stuck[s:s + m] = True
    flags[neg] |= FLAGS["NEGATIVE"]
    flags[spike] |= FLAGS["SPIKE"]
    flags[zero] |= FLAGS["ZERO_RUN"]
    flags[stuck] |= FLAGS["STUCK"]
    original = val.copy()
    rejected = (flags & REJECT) > 0
    val[rejected] = np.nan

    # 3. gaps
    gap = ~have
    flags[gap] |= FLAGS["GAP"]

    reg = _register_by_day(ctx, register)

    # 4. register check on complete, unflagged days
    reg_checked = reg_mismatch = 0
    for d, r in reg.items():
        lo, hi = ctx.day_bounds(d)
        if lo is None or np.any(flags[lo:hi]) or np.any(np.isnan(val[lo:hi])):
            continue
        reg_checked += 1
        ssum = float(val[lo:hi].sum())
        if abs(ssum - r) > cfg.register_tol_kwh + cfg.register_tol_rel * r:
            reg_mismatch += 1
            flags[lo:hi] |= FLAGS["REGISTER_MISMATCH"]
            if cfg.use_register and ssum > 0:
                f = r / ssum
                for t in range(lo, hi):
                    log(t, "REGISTER_MISMATCH", "register_prorate", val[t], val[t] * f, f"day sum {ssum:.3f} vs register {r:.2f}")
                val[lo:hi] *= f
                status[lo:hi] = "C"
                method[lo:hi] = "register_prorate"

    # 5. estimation
    base = val.copy()
    need = np.isnan(base)
    est_mask = np.zeros(n, dtype=bool)
    for a, m in runs(need):
        est, meth = estimate_run(base, a, a + m, ctx, cfg.linear_max, cfg.long_method, cfg.regression_min)
        ok = ~np.isnan(est)
        val[a:a + m][ok] = est[ok]
        est_mask[a:a + m] = ok
        method[a:a + m] = np.where(ok, meth, "")
        status[a:a + m] = np.where(ok, "E", "M")

    # 6. scale estimates of a day to the register advance
    if cfg.use_register:
        for d, r in reg.items():
            lo, hi = ctx.day_bounds(d)
            if lo is None:
                continue
            e = est_mask[lo:hi]
            if not e.any() or np.any(np.isnan(val[lo:hi])):
                continue
            actual = float(val[lo:hi][~e].sum())
            esum = float(val[lo:hi][e].sum())
            target = r - actual
            if target < -cfg.register_tol_kwh:
                flags[lo:hi] |= FLAGS["REGISTER_MISMATCH"]
                continue
            target = max(target, 0.0)
            seg = val[lo:hi]
            seg[e] = seg[e] * (target / esum) if esum > 0 else target / e.sum()
            meth = method[lo:hi]
            meth[e] = [x + "+register" for x in meth[e]]

    for t in np.flatnonzero(est_mask | (need & ~est_mask)):
        rule = ";".join(k for k, b in FLAGS.items() if flags[t] & b and k != "REGISTER_MISMATCH") or "GAP"
        log(t, rule, method[t] or "none", original[t], val[t])

    aud = pd.DataFrame(audit, columns=["slot", "rule", "method", "original_kwh", "new_kwh", "edited_at", "note"])
    summary = {"intervals": n, "rows": ing.issues, "clock_shifts": [{k: int(v) for k, v in s.items()} for s in shifts],
               "register_days_checked": reg_checked, "register_days_mismatch": reg_mismatch,
               "estimated": int(est_mask.sum()), "not_estimated": int((need & ~est_mask).sum()),
               "flag_counts": {k: int(((flags & b) > 0).sum()) for k, b in FLAGS.items()}}
    return VeeResult(grid, val, received, status, flags, method, aud, summary)


def _register_by_day(ctx: Context, register: pd.Series | None) -> dict:
    if register is None or len(register) == 0:
        return {}
    out = {}
    for d, r in register.items():
        if np.isnan(r):
            continue
        key = pd.Timestamp(d).date()
        if key in ctx.day_index:
            out[key] = float(r)
    return out
