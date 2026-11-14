"""Forecast models.

Horizon bands (days ahead of today):
  0-7    NWP forecast, bias-corrected per lead with   truth ~ a*nwp + b*climatology + c   (OLS)
  8-15   the lead-7 corrected forecast, shrunk linearly toward climatology
  16-35  climatology + decaying persistence of the recent anomaly (regression, fitted per lead)

"Truth" is ERA5, a proxy for site observations. Replace with VRA plant pyranometer / met-mast
data when available: `fit_model` only needs a daily DataFrame with the same columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

CONT_VARS = ["ghi_mj", "tmean", "tmax", "rain"]
RAIN_DAY_MM = 1.0
Z80 = 1.2816  # half-width of an 80% interval (P10..P90) for a normal error
SHORT_LEADS = range(1, 8)
BLEND_END = 16  # lead at which the corrected NWP forecast has fully relaxed to climatology
EXT_LEADS = range(BLEND_END, 36)
ANOM_WINDOW = 14  # days of recent anomaly used as the persistence predictor


# ----------------------------------------------------------------------------- climatology
def _doy(idx) -> np.ndarray:
    d = pd.DatetimeIndex(idx).dayofyear.values.copy()
    d[d == 366] = 365
    return d


def _circular_smooth(values: np.ndarray, half: int = 7) -> np.ndarray:
    pad = np.concatenate([values[-half:], values, values[:half]])
    k = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(pad, k, mode="valid")


def build_climatology(truth: pd.DataFrame) -> pd.DataFrame:
    """Smoothed day-of-year mean / std per variable and rain-day frequency. Index 1..365."""
    doy = _doy(truth.index)
    out = {}
    for v in CONT_VARS:
        g = truth[v].groupby(doy)
        out[f"{v}_mean"] = _circular_smooth(g.mean().reindex(range(1, 366)).interpolate().values)
        out[f"{v}_std"] = _circular_smooth(g.std().reindex(range(1, 366)).interpolate().values)
    wet = (truth["rain"] >= RAIN_DAY_MM).astype(float)
    out["rain_prob"] = _circular_smooth(wet.groupby(doy).mean().reindex(range(1, 366)).interpolate().values, half=10)
    return pd.DataFrame(out, index=range(1, 366))


def clim_for(clim: pd.DataFrame, dates) -> pd.DataFrame:
    return clim.loc[_doy(dates)].set_axis(pd.DatetimeIndex(dates))


# ----------------------------------------------------------------------------- model object
@dataclass
class Model:
    clim: pd.DataFrame
    short: dict = field(default_factory=dict)   # {(var, lead): (a, b, c, sigma)}
    rain_clf: dict = field(default_factory=dict)  # {lead: LogisticRegression}
    ext: dict = field(default_factory=dict)     # {(var, lead_from_last_obs): (slope, sigma)}
    last_obs: date | None = None                # last day of the truth series
    anom_now: dict = field(default_factory=dict)  # {var: mean anomaly of the last ANOM_WINDOW days}
    trained_on: str = ""


def _ols(y: np.ndarray, nwp: np.ndarray, clim: np.ndarray):
    X = np.column_stack([nwp, clim, np.ones_like(nwp)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(coef[0]), float(coef[1]), float(coef[2]), float(resid.std(ddof=3))


def fit_model(truth: pd.DataFrame, hist_fc: dict[int, pd.DataFrame], clim: pd.DataFrame | None = None) -> Model:
    """Fit all components. `truth` = daily observations/reanalysis, `hist_fc` = {lead: archived NWP daily}."""
    clim = clim if clim is not None else build_climatology(truth)
    m = Model(clim=clim, last_obs=truth.index[-1].date(),
              trained_on=f"{truth.index[0].date()}..{truth.index[-1].date()}")

    # --- short range: per-lead bias correction
    for lead, fc in hist_fc.items():
        idx = fc.index.intersection(truth.index)
        c = clim_for(clim, idx)
        wet_x, wet_y = None, None
        for v in CONT_VARS:
            if len(idx) < 60:
                m.short[(v, lead)] = (1.0, 0.0, 0.0, float(truth[v].std()))
            else:
                m.short[(v, lead)] = _ols(truth.loc[idx, v].values, fc.loc[idx, v].values, c[f"{v}_mean"].values)
        if len(idx) >= 60:
            wet_x = np.column_stack([np.log1p(fc.loc[idx, "rain"].values), c["rain_prob"].values])
            wet_y = (truth.loc[idx, "rain"].values >= RAIN_DAY_MM).astype(int)
            if wet_y.min() != wet_y.max():
                m.rain_clf[lead] = LogisticRegression(C=5.0).fit(wet_x, wet_y)

    # --- extended range: anomaly persistence
    anom = pd.DataFrame({v: truth[v].values - clim_for(clim, truth.index)[f"{v}_mean"].values
                         for v in CONT_VARS}, index=truth.index)
    roll = anom.rolling(ANOM_WINDOW).mean()
    for v in CONT_VARS:
        m.anom_now[v] = float(roll[v].iloc[-1])
        for L in range(BLEND_END - 6, 46):  # lead measured from the last observed day
            x = roll[v].values[:-L]
            y = anom[v].values[L:]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 200:
                m.ext[(v, L)] = (0.0, float(anom[v].std()))
                continue
            slope = float(np.cov(x[ok], y[ok])[0, 1] / np.var(x[ok], ddof=1))
            slope = max(0.0, slope)
            if v == "rain":
                slope = 0.0  # backtest shows anomaly persistence has no skill for daily rain: use climatology
            sigma = float((y[ok] - slope * x[ok]).std())
            m.ext[(v, L)] = (slope, sigma)
    return m


# ----------------------------------------------------------------------------- prediction
def _bands(mean, sigma, v):
    lo, hi = mean - Z80 * sigma, mean + Z80 * sigma
    if v in ("ghi_mj", "rain"):
        lo, hi, mean = max(lo, 0.0), max(hi, 0.0), max(mean, 0.0)
    return mean, lo, hi


def _short_pred(m: Model, v: str, lead: int, nwp_val: float, c_mean: float):
    a, b, c, sigma = m.short[(v, min(max(lead, 1), 7))]
    return a * nwp_val + b * c_mean + c, sigma


def predict_short(m: Model, nwp: pd.DataFrame, today: date) -> pd.DataFrame:
    """Corrected daily forecast for every row of `nwp` (lead 0..15)."""
    rows = []
    c = clim_for(m.clim, nwp.index)
    for d in nwp.index:
        lead = (d.date() - today).days
        row = {"date": d, "lead": lead, "source": "nwp+bias-correction" if lead <= 7 else "nwp->climatology blend"}
        s = 0.0 if lead <= 7 else min((lead - 7) / (BLEND_END - 7), 1.0)  # shrink weight toward climatology
        for v in CONT_VARS:
            mean7, sigma7 = _short_pred(m, v, lead, nwp.loc[d, v], c.loc[d, f"{v}_mean"])
            mean = (1 - s) * mean7 + s * c.loc[d, f"{v}_mean"]
            sigma = (1 - s) * sigma7 + s * c.loc[d, f"{v}_std"]
            mean, lo, hi = _bands(mean, sigma, v)
            row[v], row[f"{v}_p10"], row[f"{v}_p90"] = mean, lo, hi
        clf = m.rain_clf.get(min(max(lead, 1), 7))
        p_clim = c.loc[d, "rain_prob"]
        if clf is not None:
            p7 = float(clf.predict_proba([[np.log1p(nwp.loc[d, "rain"]), p_clim]])[0, 1])
        else:
            p7 = float(p_clim)
        row["rain_prob"] = (1 - s) * p7 + s * float(p_clim)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def predict_extended(m: Model, start: date, end: date, today: date) -> pd.DataFrame:
    """Climatology + persisted anomaly for dates start..end (inclusive)."""
    dates = pd.date_range(start, end)
    c = clim_for(m.clim, dates)
    rows = []
    for d in dates:
        L = min((d.date() - m.last_obs).days, 45)
        row = {"date": d, "lead": (d.date() - today).days, "source": "climatology+anomaly"}
        for v in CONT_VARS:
            slope, sigma = m.ext.get((v, max(L, BLEND_END - 6)), (0.0, c.loc[d, f"{v}_std"]))
            mean, lo, hi = _bands(c.loc[d, f"{v}_mean"] + slope * m.anom_now[v], sigma, v)
            row[v], row[f"{v}_p10"], row[f"{v}_p90"] = mean, lo, hi
        row["rain_prob"] = float(c.loc[d, "rain_prob"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def full_forecast(m: Model, nwp: pd.DataFrame, today: date | None = None, horizon_days: int = 35) -> pd.DataFrame:
    today = today or date.today()
    short = predict_short(m, nwp[nwp.index >= pd.Timestamp(today)], today)
    last_short = short.index[-1].date()
    ext = predict_extended(m, last_short + timedelta(days=1), today + timedelta(days=horizon_days - 1), today)
    return pd.concat([short, ext])


def summarise(daily: pd.DataFrame) -> dict:
    """Week-ahead (days 0-6) and month-ahead (days 0-29) aggregates, with anomaly vs climatology."""
    def agg(df):
        return {
            "start": df.index[0].strftime("%Y-%m-%d"), "end": df.index[-1].strftime("%Y-%m-%d"), "days": len(df),
            "ghi_mj_mean": round(float(df["ghi_mj"].mean()), 2),
            "ghi_kwh_m2_mean": round(float(df["ghi_mj"].mean() / 3.6), 2),
            "tmean": round(float(df["tmean"].mean()), 1),
            "tmax": round(float(df["tmax"].mean()), 1),
            "rain_total_mm": round(float(df["rain"].sum()), 1),
            "expected_rain_days": round(float(df["rain_prob"].sum()), 1),
        }
    return {"week": agg(daily.iloc[:7]), "month": agg(daily.iloc[:30])}
