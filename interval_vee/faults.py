"""Turn a clean 15-minute series into a faulty meter file with known faults (the answer key)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .estimate import Context
from .meterio import local_labels

KINDS = ["gap", "zero_run", "spike", "negative", "stuck", "duplicate", "clock_shift", "register_mismatch"]

DEFAULT_PLAN = {
    "gap": (120, [1, 2, 4, 8, 16, 32, 96, 288]),
    "zero_run": (40, [2, 4, 8, 16, 96]),
    "stuck": (40, [4, 8, 16, 48, 96]),
    "spike": (80, [1]),
    "negative": (40, [1, 2, 4]),
    "duplicate": (60, [1]),
    "clock_shift": (12, [4, 8]),
    "register_mismatch": (24, [0]),
}


@dataclass
class Event:
    kind: str
    start: int  # first affected slot
    length: int
    detail: dict = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass
class Faulty:
    rows: pd.DataFrame  # timestamp (naive local), kwh; in file order
    register: pd.Series  # daily register advance by local date
    events: list[Event]
    grid: pd.DatetimeIndex
    truth: np.ndarray


class _Space:
    def __init__(self, truth: np.ndarray, buffer: int, rng: np.random.Generator):
        self.free = ~np.isnan(truth)
        self.buffer = buffer
        self.rng = rng

    def take(self, length: int, lo: int = 0, hi: int | None = None, tries: int = 4000) -> int | None:
        n = len(self.free)
        hi = n if hi is None else hi
        b = self.buffer
        for _ in range(tries):
            s = int(self.rng.integers(max(lo, b), max(lo + 1, hi - length - b)))
            if self.free[s - b:s + length + b].all():
                self.mark(s - b, s + length + b)
                return s
        return None

    def check(self, s: int, e: int) -> bool:
        return 0 <= s and e <= len(self.free) and bool(self.free[s:e].all())

    def mark(self, s: int, e: int) -> None:
        self.free[max(0, s):e] = False


def daily_register(truth: np.ndarray, ctx: Context) -> pd.Series:
    """Register advance per complete local day, rounded to 0.01 kWh (days with unknown truth have no read)."""
    tot = ctx.daily_totals(truth)
    keep = ~np.isnan(tot)
    return pd.Series(np.round(tot[keep], 2), index=[d for d, k in zip(ctx.days, keep, strict=True) if k], name="register_kwh")


def inject(truth: pd.Series, seed: int = 7, plan: dict | None = None, tz: str = "Europe/Paris", buffer: int = 16) -> Faulty:
    plan = DEFAULT_PLAN if plan is None else plan
    rng = np.random.default_rng(seed)
    grid = truth.index
    tv = truth.to_numpy(dtype=float)
    n = len(tv)
    ctx = Context(grid, tz)
    space = _Space(tv, buffer, rng)
    meter = tv.copy()
    events: list[Event] = []
    shifted: dict[int, int] = {}  # slot -> label shift for rows of clock-shift blocks
    extra: list[tuple[int, float, float]] = []  # (label slot, value, emission key)
    register = daily_register(tv, ctx)

    # register mismatch: whole complete days with scaled intervals
    cnt = plan.get("register_mismatch", (0, []))[0]
    days = [d for d in register.index if ctx.day_bounds(d)[0] is not None]
    for d in rng.permutation(len(days)):
        if cnt <= 0:
            break
        lo, hi = ctx.day_bounds(days[d])
        if not space.check(lo - buffer, hi + buffer):
            continue
        space.mark(lo - buffer, hi + buffer)
        f = float(rng.choice([rng.uniform(0.70, 0.97), rng.uniform(1.03, 1.30)]))
        meter[lo:hi] = tv[lo:hi] * f
        events.append(Event("register_mismatch", lo, hi - lo, {"factor": round(f, 4), "day": str(days[d])}))
        cnt -= 1

    # clock shifts: missed autumn DST switch (clock stays 1 h ahead), then random +-1 h / +-2 h blocks
    cnt, lengths = plan.get("clock_shift", (0, []))
    local = grid.tz_convert(tz)
    offs = np.asarray([o.total_seconds() for o in local.map(lambda x: x.utcoffset())])
    autumn = np.flatnonzero(np.diff(offs) < 0) + 1  # first winter-time slot after each autumn change
    for a in autumn:
        if cnt <= 0:
            break
        blk = int(rng.integers(1, 4)) * 96
        if space.check(a - buffer, a + blk + 4 + buffer):
            space.mark(a - buffer, a + blk + 4 + buffer)
            for t in range(a, a + blk):
                shifted[t] = 4
            events.append(Event("clock_shift", a, blk + 4, {"shift": 4, "block": [a, a + blk], "cause": "missed DST switch"}))
            cnt -= 1
    while cnt > 0:
        s = int(rng.choice(lengths)) * int(rng.choice([-1, 1]))
        blk = int(rng.integers(1, 5)) * 96
        a = space.take(blk + abs(s))
        if a is None:
            break
        a0 = a + abs(s) if s < 0 else a  # keep the shifted labels inside the reserved span
        for t in range(a0, a0 + blk):
            shifted[t] = s
        events.append(Event("clock_shift", a, blk + abs(s), {"shift": s, "block": [a0, a0 + blk], "cause": "clock offset"}))
        cnt -= 1

    def place(kind, apply):
        c, lens = plan.get(kind, (0, []))
        for i in range(c):
            m = int(lens[i % len(lens)])
            s = space.take(m)
            if s is None:
                return
            detail = apply(s, m) or {}
            events.append(Event(kind, s, m, detail))

    def gap(s, m):
        meter[s:s + m] = np.nan

    def zero_run(s, m):
        meter[s:s + m] = 0.0

    def stuck(s, m):
        meter[s:s + m] = np.round(tv[s - 1], 3)

    def spike(s, m):
        gross = rng.random() < 0.5
        add = rng.uniform(10, 500) if gross else rng.uniform(0.5, 2.5)
        meter[s] = tv[s] + add
        return {"added_kwh": round(float(add), 3), "gross": bool(gross)}

    def negative(s, m):
        flip = rng.random() < 0.5
        meter[s:s + m] = -tv[s:s + m] if flip else -rng.uniform(0.001, 0.05, m)
        return {"sign_flip": bool(flip)}

    def duplicate(s, m):
        conflict = rng.random() < 0.5
        v = tv[s - 96] if conflict and not np.isnan(tv[s - 96]) and abs(tv[s - 96] - tv[s]) > 0.0015 else tv[s]
        conflict = bool(v != tv[s])
        key = s + (rng.choice([-0.4, 0.4]) if conflict else 0.4)
        extra.append((s, float(v), float(key)))
        return {"conflicting": conflict, "other_kwh": round(float(v), 3)}

    for kind, fn in [("gap", gap), ("zero_run", zero_run), ("stuck", stuck), ("spike", spike), ("negative", negative),
                     ("duplicate", duplicate)]:
        place(kind, fn)

    # emission list: (label slot, value, emission key)
    keep = ~np.isnan(meter)
    slots = np.flatnonzero(keep)
    labels = slots + np.array([shifted.get(int(t), 0) for t in slots])
    emit = [(int(lab), float(v), float(t)) for lab, v, t in zip(labels, meter[keep], slots, strict=True)] + extra
    emit = [e for e in emit if 0 <= e[0] < n]
    emit.sort(key=lambda e: e[2])
    lab_text = local_labels(grid, tz)
    rows = pd.DataFrame({"timestamp": [lab_text[e[0]] for e in emit], "kwh": [f"{e[1]:.3f}" for e in emit]})
    events.sort(key=lambda e: e.start)
    return Faulty(rows=rows, register=register, events=events, grid=grid, truth=tv)


def source_gap_events(truth: np.ndarray) -> list[Event]:
    m = np.isnan(truth).astype(np.int8)
    d = np.diff(np.r_[0, m, 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [Event("gap", int(s), int(e - s), {"origin": "source"}) for s, e in zip(starts, ends, strict=True)]
