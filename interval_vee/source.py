"""Hidden truth: an open household dataset aggregated to 15-minute energy, plus daily mean temperature.

The 1-minute source file is read from `cache/` (gitignored). The repo ships the derived 15-minute series
(`data/household_15min.csv.gz`) and daily temperatures (`data/temperature_daily.csv`) so the demo runs offline.
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption"
# Sceaux (Hauts-de-Seine), the location given for the household
LAT, LON = 48.78, 2.29
METEO_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
             "&start_date={start}&end_date={end}&daily=temperature_2m_mean&timezone=Europe%2FParis")
LOCAL_TZ = "Europe/Paris"
# The source has 1,440 minutes on every day, DST days included, so its clock is read as fixed UTC+1.
SOURCE_OFFSET = pd.Timedelta(hours=1)


def download_source(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    z = cache / "household.zip"
    if not z.exists():
        urllib.request.urlretrieve(UCI_URL, z)  # noqa: S310 - fixed https URL
    txt = cache / "household_power_consumption.txt"
    if not txt.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extract("household_power_consumption.txt", cache)
    return txt


def minutes_to_15min(txt: Path) -> pd.DataFrame:
    """1-minute mean kW -> 15-minute kWh (UTC). An interval is known only if all 15 minutes are present."""
    df = pd.read_csv(txt, sep=";", usecols=["Date", "Time", "Global_active_power"], na_values=["?"],
                     dtype={"Global_active_power": "float64"})
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, format="%d/%m/%Y %H:%M:%S")
    utc = (ts - SOURCE_OFFSET).dt.tz_localize("UTC")
    s = pd.Series(df["Global_active_power"].to_numpy(), index=utc)
    s = s[~s.index.duplicated()]
    full = pd.date_range(s.index.min().ceil("15min"), s.index.max().floor("15min"), freq="1min", tz="UTC", inclusive="left")
    s = s.reindex(full)
    g = s.groupby(s.index.floor("15min"))
    kwh = g.sum(min_count=1) / 60.0
    kwh[g.count() < 15] = np.nan
    return pd.DataFrame({"timestamp_utc": kwh.index, "kwh": kwh.round(3).to_numpy()})


def load_truth(path: Path | None = None) -> pd.Series:
    """15-minute energy in kWh on a UTC index; NaN where the source has missing minutes."""
    if path is None:
        path = resources.files("interval_vee") / "data" / "household_15min.csv.gz"
    df = pd.read_csv(path, compression="gzip")
    idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp_utc"], utc=True))
    return pd.Series(df["kwh"].to_numpy(dtype=float), index=idx, name="kwh").asfreq("15min")


def fetch_temperature(start: str, end: str) -> pd.DataFrame:
    url = METEO_URL.format(lat=LAT, lon=LON, start=start, end=end)
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed https URL
        js = json.loads(r.read().decode())
    return pd.DataFrame({"date": js["daily"]["time"], "t_mean": js["daily"]["temperature_2m_mean"]})


def load_temperature(path: Path | None = None) -> pd.Series:
    """Daily mean air temperature (deg C) indexed by local date."""
    if path is None:
        path = resources.files("interval_vee") / "data" / "temperature_daily.csv"
    df = pd.read_csv(path)
    return pd.Series(df["t_mean"].to_numpy(dtype=float), index=pd.to_datetime(df["date"]).dt.date, name="t_mean")


def prepare(cache: Path, data_dir: Path) -> dict:
    """Rebuild the committed data files from the original downloads (internet)."""
    txt = download_source(cache)
    df = minutes_to_15min(txt)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "household_15min.csv.gz"
    df.assign(timestamp_utc=df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_csv(
        out, index=False, compression={"method": "gzip", "mtime": 0})
    first = df["timestamp_utc"].iloc[0].tz_convert(LOCAL_TZ).date()
    last = df["timestamp_utc"].iloc[-1].tz_convert(LOCAL_TZ).date()
    t = fetch_temperature(str(first), str(last))
    t.to_csv(data_dir / "temperature_daily.csv", index=False)
    return {"intervals": len(df), "known": int(df["kwh"].notna().sum()), "first": str(first), "last": str(last),
            "temperature_days": len(t)}
