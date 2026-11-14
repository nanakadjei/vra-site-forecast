"""Glue: load a site, fit/cached model, produce the JSON payload used by the CLI and web app."""
from __future__ import annotations

import json
import pickle
import time
from datetime import date
from pathlib import Path

from . import openmeteo as om
from .backtest import evaluate
from .models import Model, fit_model, full_forecast, summarise

ROOT = Path(__file__).resolve().parent.parent
SITES_FILE = ROOT / "sites.json"
HISTORY_START = date(2005, 1, 1)   # truth record for climatology
HISTFC_START = date(2022, 1, 1)    # archived forecasts start here
SPLIT = date(2024, 12, 31)          # honest backtest: fit <= split, score after
MODEL_TTL_H = 24


SOURCES = [
    {"id": "nwp", "name": "Weather-model forecast (days 0-15)",
     "provider": "Open-Meteo Forecast API: blend of national weather-service models",
     "url": "https://open-meteo.com/en/docs", "licence": "CC BY 4.0",
     "used_for": "Today and the next 15 days, before bias correction"},
    {"id": "histfc", "name": "Archived forecasts (2022 to date)",
     "provider": "Open-Meteo Historical Forecast API",
     "url": "https://open-meteo.com/en/docs/historical-forecast-api", "licence": "CC BY 4.0",
     "used_for": "Learning the site-specific bias correction (days 1-7)"},
    {"id": "era5", "name": "ERA5 reanalysis (2005 to date)",
     "provider": "Copernicus Climate Change Service / ECMWF, via Open-Meteo Archive API",
     "url": "https://open-meteo.com/en/docs/historical-weather-api", "licence": "CC BY 4.0 (Copernicus licence)",
     "used_for": "Climatology, observation proxy for training and backtest scoring"},
    {"id": "osm", "name": "Base map",
     "provider": "OpenStreetMap contributors, rendered with Leaflet",
     "url": "https://www.openstreetmap.org/copyright", "licence": "ODbL",
     "used_for": "Map tiles"},
    {"id": "sites", "name": "Site coordinates",
     "provider": "sites.json in this project (approximate, not VRA survey data)",
     "url": "", "licence": "",
     "used_for": "Locating each forecast point"},
]


def load_sites() -> list[dict]:
    return json.loads(SITES_FILE.read_text())["sites"]


def get_site(site_id: str) -> dict:
    for s in load_sites():
        if s["id"] == site_id:
            return s
    raise KeyError(f"Unknown site '{site_id}'. Known: {[s['id'] for s in load_sites()]}")


def _model_path(site_id: str) -> Path:
    return om.CACHE_DIR / f"model_{site_id}.pkl"


def build_site(site: dict, force: bool = False) -> tuple[Model, dict]:
    """Fit the production model (all data up to the latest ERA5 day) and the honest backtest metrics."""
    p = _model_path(site["id"])
    if p.exists() and not force and (time.time() - p.stat().st_mtime) < MODEL_TTL_H * 3600:
        return pickle.loads(p.read_bytes())
    lat, lon = site["lat"], site["lon"]
    truth = om.fetch_era5(lat, lon, HISTORY_START, date.today())
    hist = om.fetch_hist_forecasts(lat, lon, HISTFC_START, date.today())
    metrics, _ = evaluate(truth, hist, SPLIT)
    model = fit_model(truth, hist)
    p.write_bytes(pickle.dumps((model, metrics)))
    return model, metrics


def site_forecast(site_id: str, force: bool = False) -> dict:
    site = get_site(site_id)
    model, metrics = build_site(site, force)
    nwp = om.fetch_nwp(site["lat"], site["lon"], 16)
    today = date.today()
    daily = full_forecast(model, nwp, today, 35)
    recs = []
    for d, r in daily.iterrows():
        recs.append({"date": d.strftime("%Y-%m-%d"), "lead": int(r["lead"]), "source": r["source"],
                     **{k: round(float(r[k]), 2) for k in daily.columns if k not in ("lead", "source")}})
    return {
        "site": site,
        "generated": today.isoformat(),
        "model_trained_on": model.trained_on,
        "era5_last_day": model.last_obs.isoformat(),
        "sources": SOURCES,
        "daily": recs,
        "summary": summarise(daily),
        "backtest": metrics,
        "notes": [
            "Truth proxy is ERA5 reanalysis, not plant instruments; recalibrate on pyranometer data when available.",
            "Days 0-7: NWP with site bias correction. Days 8-15: relaxes to climatology. Days 16-35: climatology plus "
            "decaying anomaly persistence, i.e. a seasonal-tendency outlook, not a day-by-day weather forecast.",
        ],
    }
