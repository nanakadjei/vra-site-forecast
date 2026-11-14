"""Offline tests on synthetic data (no network)."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vra_forecast.backtest import evaluate, score_submission
from vra_forecast.models import build_climatology, fit_model, full_forecast, summarise


def make_truth(years=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=365 * years)
    doy = idx.dayofyear.values
    ghi = 20 + 3 * np.sin(2 * np.pi * (doy - 80) / 365) + rng.normal(0, 2, len(idx))
    t = 28 + 2 * np.sin(2 * np.pi * (doy - 60) / 365) + rng.normal(0, 1, len(idx))
    wet = rng.random(len(idx)) < 0.2 * (1 + np.sin(2 * np.pi * (doy - 120) / 365)) / 1.0
    rain = np.where(wet, rng.gamma(2, 4, len(idx)), 0.0)
    return pd.DataFrame({"ghi_mj": ghi.clip(2), "tmean": t, "tmax": t + 7, "rain": rain}, index=idx)


def fake_hist_fc(truth, lead, seed=1):
    """A biased, noisy 'NWP' that gets noisier with lead."""
    rng = np.random.default_rng(seed + lead)
    f = truth.copy()
    f["ghi_mj"] = truth["ghi_mj"] * 1.1 + rng.normal(0, 0.6 * lead, len(f))
    f["tmean"] = truth["tmean"] + 1.5 + rng.normal(0, 0.3 * lead, len(f))
    f["tmax"] = truth["tmax"] + 1.5 + rng.normal(0, 0.3 * lead, len(f))
    f["rain"] = (truth["rain"] * rng.uniform(0.5, 1.5, len(f))).clip(0)
    return f.iloc[-1200:]


@pytest.fixture(scope="module")
def setup():
    truth = make_truth()
    hist = {L: fake_hist_fc(truth, L) for L in range(1, 8)}
    return truth, hist


def test_climatology_shape_and_seasonality():
    c = build_climatology(make_truth())
    assert len(c) == 365 and not c.isna().any().any()
    assert c.loc[170, "ghi_mj_mean"] > c.loc[350, "ghi_mj_mean"]  # seasonal cycle recovered
    assert ((c["rain_prob"] >= 0) & (c["rain_prob"] <= 1)).all()


def test_bias_correction_removes_bias(setup):
    truth, hist = setup
    m = fit_model(truth, hist)
    a, b, c, sigma = m.short[("tmean", 1)]
    assert sigma < 1.5
    nwp_val = 30.0 + 1.5
    pred = a * nwp_val + b * m.clim["tmean_mean"].mean() + c
    assert abs(pred - 30.0) < 1.5  # the +1.5 C bias is pulled back


def test_corrected_beats_raw_in_backtest(setup):
    truth, hist = setup
    metrics, _ = evaluate(truth, hist, date(2022, 1, 1))
    for row in metrics["short_range"]:
        assert row["ghi_mj_mae_model"] < row["ghi_mj_mae_raw_nwp"]
        assert row["tmean_mae_model"] < row["tmean_mae_raw_nwp"]


def test_full_forecast_continuous_and_bounded(setup):
    truth, hist = setup
    m = fit_model(truth, hist)
    today = date(2023, 6, 1)
    nwp = hist[1].iloc[:16].copy()
    nwp.index = pd.date_range(today, periods=16)
    daily = full_forecast(m, nwp, today, 35)
    assert len(daily) == 35 and list(daily["lead"]) == list(range(35))
    assert (daily["ghi_mj"] >= 0).all() and (daily["rain"] >= 0).all()
    assert (daily["ghi_mj_p10"] <= daily["ghi_mj"]).all() and (daily["ghi_mj"] <= daily["ghi_mj_p90"]).all()
    assert daily["rain_prob"].between(0, 1).all()
    s = summarise(daily)
    assert s["week"]["days"] == 7 and s["month"]["days"] == 30


def test_scorer_ranks_climatology_at_zero_and_truth_at_one(setup):
    truth, _ = setup
    clim = build_climatology(truth)
    te = truth.iloc[-100:]
    perfect = te[["ghi_mj", "tmean", "rain"]].reset_index().rename(columns={"index": "date"})
    r = score_submission(perfect, truth, clim)
    assert r["composite_score"] == pytest.approx(1.0) and r["wet_dry_accuracy"] == 1.0
    from vra_forecast.models import clim_for
    c = clim_for(clim, te.index)
    base = pd.DataFrame({"date": te.index, "ghi_mj": c["ghi_mj_mean"].values,
                         "tmean": c["tmean_mean"].values, "rain": c["rain_mean"].values})
    assert score_submission(base, truth, clim)["composite_score"] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        score_submission(perfect.drop(columns=["rain"]), truth, clim)
