"""Read meter files (timestamp,kwh) onto a UTC 15-minute grid, keeping every row so duplicates stay visible.

Timestamps are interval starts. They may carry a UTC offset ("2009-10-25T02:15:00+02:00", "...Z") or be naive local
wall-clock time. Naive local time is resolved like this: in the repeated autumn hour the first occurrence in file order
is summer time and the second is winter time; times that do not exist (the skipped spring hour) are rejected and counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SLOT = pd.Timedelta(minutes=15)
_OFFSET = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


@dataclass
class Ingested:
    grid: pd.DatetimeIndex  # UTC interval starts
    slot: np.ndarray  # grid position of every accepted row
    value: np.ndarray  # kWh of every accepted row
    order: np.ndarray  # file order of every accepted row
    issues: dict = field(default_factory=dict)

    @property
    def counts(self) -> np.ndarray:
        return np.bincount(self.slot, minlength=len(self.grid))


def read_rows(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = {c.lower().strip(): c for c in df.columns}
    ts = cols.get("timestamp") or cols.get("timestamp_local") or cols.get("timestamp_utc")
    kwh = cols.get("kwh")
    if ts is None or kwh is None:
        raise ValueError("meter file needs the columns 'timestamp' and 'kwh'")
    return pd.DataFrame({"timestamp": df[ts].str.strip(), "kwh": df[kwh].str.strip()})


def to_utc(stamps: pd.Series, tz: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Parse timestamps to UTC. Returns (utc, nonexistent_mask); unparseable stamps become NaT."""
    stamps = stamps.astype(str).reset_index(drop=True)
    n = len(stamps)
    has_offset = stamps.str.contains(_OFFSET)
    out = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns, UTC]")
    nonexistent = np.zeros(n, dtype=bool)
    if has_offset.any():
        out[has_offset] = pd.to_datetime(stamps[has_offset], utc=True, errors="coerce", format="ISO8601")
    naive_mask = ~has_offset.to_numpy()
    if naive_mask.any():
        naive = pd.DatetimeIndex(pd.to_datetime(stamps[naive_mask], errors="coerce", format="ISO8601"))
        ok = ~naive.isna()
        dst = naive.tz_localize(tz, ambiguous=np.ones(len(naive), dtype=bool), nonexistent="NaT")
        std = naive.tz_localize(tz, ambiguous=np.zeros(len(naive), dtype=bool), nonexistent="NaT")
        ambiguous = ok & ~dst.isna() & (dst != std)
        key = pd.Series(naive.as_unit("ns").asi8)
        occ = key.where(ambiguous).groupby(key).cumcount().to_numpy()
        use_std = ambiguous & (occ >= 1)
        local = dst.where(~use_std, std)
        nonexistent[np.flatnonzero(naive_mask)] = ok & dst.isna()
        out[naive_mask] = local.tz_convert("UTC")
    return pd.DatetimeIndex(out), nonexistent


def ingest(rows: pd.DataFrame, tz: str = "Europe/Paris", grid: pd.DatetimeIndex | None = None) -> Ingested:
    """Map rows onto the UTC 15-minute grid. Rejected rows are counted in `issues`."""
    rows = rows.reset_index(drop=True)
    utc, nonexistent = to_utc(rows["timestamp"], tz)
    vals = pd.to_numeric(rows["kwh"], errors="coerce").to_numpy(dtype=float)
    bad_ts = np.asarray(utc.isna()) & ~nonexistent
    unparseable = np.isnan(vals) & ~np.asarray(utc.isna())
    ok = ~np.asarray(utc.isna()) & ~np.isnan(vals)
    ns = utc.as_unit("ns").asi8
    off_grid = ok & (ns % SLOT.value != 0)
    ok &= ~off_grid
    if grid is None:
        if not ok.any():
            raise ValueError("no usable rows")
        grid = pd.date_range(pd.Timestamp(ns[ok].min(), tz="UTC"), pd.Timestamp(ns[ok].max(), tz="UTC"), freq=SLOT)
    slot = np.full(len(rows), -1, dtype=np.int64)
    slot[ok] = (ns[ok] - grid[0].as_unit("ns").value) // SLOT.value
    outside = ok & ((slot < 0) | (slot >= len(grid)))
    ok &= ~outside
    issues = {"rows": int(len(rows)), "accepted": int(ok.sum()), "nonexistent_local_time": int(nonexistent.sum()),
              "bad_timestamp": int(bad_ts.sum()), "unparseable_value": int(unparseable.sum()),
              "off_grid": int(off_grid.sum()), "outside_period": int(outside.sum())}
    idx = np.flatnonzero(ok)
    return Ingested(grid=grid, slot=slot[idx], value=vals[idx], order=idx, issues=issues)


def local_labels(grid: pd.DatetimeIndex, tz: str) -> np.ndarray:
    """Naive local wall-clock labels ("YYYY-MM-DD HH:MM") of UTC slots."""
    return np.asarray(grid.tz_convert(tz).strftime("%Y-%m-%d %H:%M"))


def read_register(path: Path | str) -> pd.Series:
    """Daily register advance (kWh between two midnight register reads), indexed by local date."""
    df = pd.read_csv(path)
    return pd.Series(df["kwh"].to_numpy(dtype=float), index=pd.to_datetime(df["date"]).dt.date, name="register_kwh")


def write_register(reg: pd.Series, path: Path | str) -> None:
    pd.DataFrame({"date": [str(d) for d in reg.index], "kwh": np.round(reg.to_numpy(dtype=float), 2)}).to_csv(path, index=False)
