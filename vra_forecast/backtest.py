"""Out-of-sample backtest and hackathon scorer.

`evaluate` fits on data up to `split` and scores on everything after it, so the reported skill is honest.
Baselines every entry must beat:
  climatology   day-of-year mean of the training years
  persistence   "tomorrow looks like today" (value observed `lead` days before)
  raw NWP       the uncorrected public forecast (what a phone app shows)
Skill = 1 - MAE_model / MAE_baseline  (positive = better than that baseline).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .models import (CONT_VARS, EXT_LEADS, RAIN_DAY_MM, SHORT_LEADS, ANOM_WINDOW, Model, build_climatology,
                     clim_for, fit_model)

SCORE_VARS = ["ghi_mj", "tmean", "rain"]


def _mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def evaluate(truth: pd.DataFrame, hist_fc: dict[int, pd.DataFrame], split: date) -> tuple[dict, Model]:
    """Fit on <= split, score on > split. Returns (metrics dict, fitted model)."""
    sp = pd.Timestamp(split)
    tr = truth[truth.index <= sp]
    te = truth[truth.index > sp]
    clim = build_climatology(tr)
    m = fit_model(tr, {L: f[f.index <= sp] for L, f in hist_fc.items()}, clim)

    short_rows = []
    for lead in SHORT_LEADS:
        fc = hist_fc[lead]
        idx = fc.index.intersection(te.index)
        if len(idx) < 30:
            continue
        c = clim_for(clim, idx)
        row = {"lead": lead, "n": int(len(idx))}
        for v in SCORE_VARS:
            a, b, k, _ = m.short[(v, lead)]
            corr = a * fc.loc[idx, v].values + b * c[f"{v}_mean"].values + k
            if v in ("ghi_mj", "rain"):
                corr = np.maximum(corr, 0)
            y = te.loc[idx, v].values
            pers_idx = idx - pd.Timedelta(days=lead)
            pers = truth[v].reindex(pers_idx).values
            ok = ~np.isnan(pers)
            row[f"{v}_mae_model"] = _mae(y, corr)
            row[f"{v}_mae_raw_nwp"] = _mae(y, fc.loc[idx, v].values)
            row[f"{v}_mae_clim"] = _mae(y, c[f"{v}_mean"].values)
            row[f"{v}_mae_persistence"] = _mae(y[ok], pers[ok])
            row[f"{v}_skill_vs_clim"] = 1 - row[f"{v}_mae_model"] / row[f"{v}_mae_clim"]
            row[f"{v}_skill_vs_raw_nwp"] = 1 - row[f"{v}_mae_model"] / row[f"{v}_mae_raw_nwp"]
        # rain / no-rain hit rate (the "phone forecast is right more than half the time" benchmark)
        wet = te.loc[idx, "rain"].values >= RAIN_DAY_MM
        clf = m.rain_clf.get(lead)
        if clf is not None:
            p = clf.predict_proba(np.column_stack([np.log1p(fc.loc[idx, "rain"].values), c["rain_prob"].values]))[:, 1]
            row["wet_dry_acc_model"] = float(np.mean((p >= 0.5) == wet))
        row["wet_dry_acc_raw_nwp"] = float(np.mean((fc.loc[idx, "rain"].values >= RAIN_DAY_MM) == wet))
        row["wet_dry_acc_always_dry"] = float(np.mean(~wet))
        short_rows.append(row)

    # --- extended range: origins every 5 days through the test period
    anom = pd.DataFrame({v: truth[v].values - clim_for(clim, truth.index)[f"{v}_mean"].values for v in CONT_VARS},
                        index=truth.index)
    roll = anom.rolling(ANOM_WINDOW).mean()
    ext_rows = []
    origins = te.index[::5]
    for lo, hi in [(10, 20), (21, 30), (31, 40)]:
        errs = {v: {"model": [], "clim": []} for v in SCORE_VARS}
        for t in origins:
            for L in range(lo, hi + 1):
                d = t + pd.Timedelta(days=L)
                if d not in truth.index:
                    continue
                for v in SCORE_VARS:
                    slope = m.ext.get((v, L), (0.0, 0.0))[0]
                    cm = float(clim_for(clim, [d])[f"{v}_mean"].iloc[0])
                    pred = cm + slope * roll.loc[t, v]
                    if v in ("ghi_mj", "rain"):
                        pred = max(pred, 0.0)
                    errs[v]["model"].append(abs(truth.loc[d, v] - pred))
                    errs[v]["clim"].append(abs(truth.loc[d, v] - cm))
        row = {"lead_from_last_obs": f"{lo}-{hi}"}
        for v in SCORE_VARS:
            if errs[v]["model"]:
                row[f"{v}_mae_model"] = float(np.mean(errs[v]["model"]))
                row[f"{v}_mae_clim"] = float(np.mean(errs[v]["clim"]))
                row[f"{v}_skill_vs_clim"] = 1 - row[f"{v}_mae_model"] / row[f"{v}_mae_clim"]
        ext_rows.append(row)

    metrics = {"split": str(split), "test_start": str(te.index[0].date()), "test_end": str(te.index[-1].date()),
               "truth": "ERA5 reanalysis (proxy for site observations)", "short_range": short_rows, "extended": ext_rows}
    return metrics, m


# ----------------------------------------------------------------------------- hackathon scorer
def score_submission(submission: pd.DataFrame, truth: pd.DataFrame, clim: pd.DataFrame) -> dict:
    """Score a participant's forecasts.

    `submission` columns: date, ghi_mj, tmean, rain (daily forecasts; one row per target date).
    Returns MAE per variable, skill vs climatology, wet/dry accuracy, and a composite 0-1+ score
    (mean skill over the three variables; > 0 beats climatology).
    """
    sub = submission.copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.set_index("date").sort_index()
    idx = sub.index.intersection(truth.index)
    if len(idx) == 0:
        raise ValueError("No overlap between submission dates and truth")
    c = clim_for(clim, idx)
    out: dict = {"n_days": int(len(idx))}
    skills = []
    for v in SCORE_VARS:
        if v not in sub.columns:
            raise ValueError(f"Submission is missing column '{v}'")
        mae = _mae(truth.loc[idx, v], sub.loc[idx, v])
        base = _mae(truth.loc[idx, v], c[f"{v}_mean"])
        out[f"{v}_mae"], out[f"{v}_mae_clim"] = mae, base
        out[f"{v}_skill"] = 1 - mae / base
        skills.append(out[f"{v}_skill"])
    wet = truth.loc[idx, "rain"] >= RAIN_DAY_MM
    out["wet_dry_accuracy"] = float(((sub.loc[idx, "rain"] >= RAIN_DAY_MM) == wet).mean())
    out["composite_score"] = float(np.mean(skills))
    return out
