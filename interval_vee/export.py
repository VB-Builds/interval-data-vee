"""Plain CSV export with quality flags, and the audit trail of every edit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .vee import FLAGS, VeeResult

STATUS = {"A": "actual", "C": "corrected", "E": "estimated", "M": "missing"}


def flags_text(flags: np.ndarray) -> list[str]:
    names = list(FLAGS.items())
    return [";".join(k for k, b in names if f & b) for f in flags]


def to_frame(res: VeeResult, tz: str) -> pd.DataFrame:
    local = res.grid.tz_convert(tz)
    kwh = np.where(np.isnan(res.value), "", np.char.mod("%.3f", np.nan_to_num(res.value)))
    return pd.DataFrame({
        "timestamp_utc": res.grid.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timestamp_local": [t.isoformat() for t in local],
        "kwh": kwh,
        "status": res.status,
        "flags": flags_text(res.flags),
        "method": res.method.astype(str),
    })


def audit_frame(res: VeeResult) -> pd.DataFrame:
    a = res.audit.copy()
    a.insert(0, "timestamp_utc", res.grid[a["slot"].to_numpy(dtype=int)].strftime("%Y-%m-%dT%H:%M:%SZ"))
    a = a.drop(columns="slot")
    for col in ("original_kwh", "new_kwh"):
        a[col] = [x if isinstance(x, str) else ("" if x is None or np.isnan(x) else f"{x:.3f}") for x in a[col]]
    return a


def write(res: VeeResult, out: Path, tz: str, name: str = "vee") -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    p1, p2 = out / f"{name}_export.csv", out / f"{name}_audit.csv"
    to_frame(res, tz).to_csv(p1, index=False)
    audit_frame(res).to_csv(p2, index=False)
    return p1, p2
