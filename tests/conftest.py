import numpy as np
import pandas as pd
import pytest

TZ = "Europe/Paris"


def synthetic(days: int = 42, start: str = "2009-10-05", seed: int = 0, noise: float = 0.0) -> pd.Series:
    """Household-like 15-min kWh: base load + morning and evening peaks, same every week (+ optional noise)."""
    t0 = pd.Timestamp(start, tz=TZ).tz_convert("UTC")
    t1 = (pd.Timestamp(start) + pd.Timedelta(days=days)).tz_localize(TZ).tz_convert("UTC")
    idx = pd.date_range(t0, t1, freq="15min", inclusive="left")
    local = idx.tz_convert(TZ)
    h = local.hour + local.minute / 60
    wd = local.day_of_week
    q = local.hour * 4 + local.minute // 15
    ripple = 0.006 * np.sin(2 * np.pi * q / 7)  # keeps neighbours different, repeats every local day
    v = 0.08 + 0.25 * np.exp(-((h - 7.5) ** 2) / 1.5) + 0.4 * np.exp(-((h - 19.5) ** 2) / 3) + 0.05 * (wd >= 5) + ripple
    if noise:
        v = v + np.random.default_rng(seed).normal(0, noise, len(v)).clip(-0.05, None)
    return pd.Series(np.round(np.asarray(v, dtype=float), 3), index=idx, name="kwh")


@pytest.fixture
def week6():
    return synthetic()


def rows_from(series: pd.Series, tz: str = TZ) -> pd.DataFrame:
    labels = series.index.tz_convert(tz).strftime("%Y-%m-%d %H:%M")
    return pd.DataFrame({"timestamp": labels, "kwh": [f"{x:.3f}" for x in series.to_numpy()]})
