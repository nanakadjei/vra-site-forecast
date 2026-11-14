"""Thin Open-Meteo client with on-disk caching.

Three sources, all free and key-less:
  * forecast API            -> current NWP forecast, up to 16 days
  * archive API (ERA5)      -> multi-year "truth" used for climatology and scoring
  * historical-forecast API -> archived day-N-ahead forecasts, used to learn bias corrections

Everything is returned as daily pandas DataFrames in local time (Ghana = GMT) with columns:
  ghi_mj  daily global horizontal irradiation, MJ/m2/day
  tmean   daily mean 2 m temperature, deg C
  tmax    daily max 2 m temperature, deg C
  rain    daily precipitation, mm
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTFC_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

VARS = ["ghi_mj", "tmean", "tmax", "rain"]
W_M2_HOURLY_TO_MJ = 0.0036  # 1 W/m2 for 1 h = 0.0036 MJ/m2


def _get_json(url: str, params: dict, ttl_hours: float) -> dict:
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
        return json.loads(path.read_text())
    last_err = None
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, timeout=90)
            if r.status_code == 429:  # rate limited (more likely from shared CI runners): back off longer
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            path.write_text(json.dumps(data))
            return data
        except requests.RequestException as e:  # network / HTTP error: retry with backoff
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo request failed: {url}: {last_err}")


def _daily_frame(d: dict) -> pd.DataFrame:
    df = pd.DataFrame(d["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index.name = "date"
    return df.rename(columns={
        "shortwave_radiation_sum": "ghi_mj",
        "temperature_2m_mean": "tmean",
        "temperature_2m_max": "tmax",
        "precipitation_sum": "rain",
    })[VARS]


def fetch_era5(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """ERA5 reanalysis daily data. ERA5 lags real time by ~5 days, so `end` is clipped."""
    end = min(end, date.today() - timedelta(days=6))
    p = dict(latitude=lat, longitude=lon, start_date=start.isoformat(), end_date=end.isoformat(),
             daily="shortwave_radiation_sum,temperature_2m_mean,temperature_2m_max,precipitation_sum",
             timezone="GMT")
    df = _daily_frame(_get_json(ARCHIVE_URL, p, ttl_hours=24))
    return df.dropna()


def fetch_nwp(lat: float, lon: float, days: int = 16) -> pd.DataFrame:
    """Current NWP forecast (today .. today+days-1) from the forecast API."""
    p = dict(latitude=lat, longitude=lon, forecast_days=days, timezone="GMT",
             daily="shortwave_radiation_sum,temperature_2m_mean,temperature_2m_max,precipitation_sum")
    return _daily_frame(_get_json(FORECAST_URL, p, ttl_hours=1))


def fetch_hist_forecasts(lat: float, lon: float, start: date, end: date, leads=range(1, 8)) -> dict[int, pd.DataFrame]:
    """Archived forecasts issued `lead` days earlier, aggregated to daily. Returns {lead: DataFrame}."""
    end = min(end, date.today() - timedelta(days=2))
    out: dict[int, pd.DataFrame] = {}
    for lead in leads:
        p = dict(latitude=lat, longitude=lon, start_date=start.isoformat(), end_date=end.isoformat(),
                 hourly=",".join(f"{v}_previous_day{lead}" for v in
                                 ("shortwave_radiation", "temperature_2m", "precipitation")),
                 timezone="GMT")
        h = _get_json(HISTFC_URL, p, ttl_hours=24 * 7)["hourly"]
        df = pd.DataFrame(h)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        df.columns = ["ghi_w", "temp", "rain"]
        g = df.resample("D")
        daily = pd.DataFrame({
            "ghi_mj": g["ghi_w"].sum(min_count=20) * W_M2_HOURLY_TO_MJ,
            "tmean": g["temp"].mean(),
            "tmax": g["temp"].max(),
            "rain": g["rain"].sum(min_count=20),
        })
        daily.index.name = "date"
        out[lead] = daily.dropna()
    return out
