# Brief 3: Short-term weather and solar-resource forecasting tool

Site-specific day-, week- and month-ahead forecasts of solar irradiation, temperature and rainfall for VRA generation sites, with an honest backtest against the baselines a hackathon entry must beat.

## Run

```bash
pip install -r requirements.txt
python app.py                 # web app at http://127.0.0.1:5003 (first load per site downloads ~20 years of data: 1-4 min)
python cli.py forecast lawra  # 35-day forecast table
python cli.py build           # refit every site and print backtest skill
python -m pytest tests -q     # offline unit tests
```

Data: [Open-Meteo](https://open-meteo.com) (free, no key): NWP forecast, ERA5 reanalysis, archived day-1..7 forecasts. Responses are cached in `.cache/`. Open-Meteo's free tier is for non-commercial use; check terms (or use the commercial plan) before operational deployment.

## Hosting on GitHub Pages (daily rebuild)

`.github/workflows/daily-forecast.yml` rebuilds and publishes a static copy of the app every day at 05:00 UTC (= 05:00 Ghana time, which is on GMT). It installs the dependencies, runs the tests, runs `python build_static.py` (downloads data, fits every site model, writes `_site/data/<site>.json`) and deploys `_site/` to GitHub Pages. A full build takes up to about 30-40 minutes. If any site fails the deployment is skipped and the previous version stays online.

One-time setup:

1. Create a GitHub repository and push this folder to its `main` branch.
2. In the repository: **Settings > Pages > Build and deployment > Source: GitHub Actions**.
3. Run the workflow once from the **Actions** tab (**Run workflow**), or push to `main`. The site appears at `https://<user>.github.io/<repo>/`.

Notes:

- Free GitHub Pages needs a **public** repository (private needs a paid plan). The code, site list and approximate coordinates are then public.
- GitHub pauses scheduled workflows in a public repository after 60 days without repository activity. A `keepalive` job in the workflow re-enables the workflow through the API on every run to reset that timer. This is a widely used workaround rather than a documented guarantee, so if the daily build ever stops, re-enable it from the Actions tab or make a small commit.
- The static site has no live **Refresh model** button; the daily build replaces it. To test the static build locally: `python build_static.py --site lawra`, then serve `_site/` with `python -m http.server --directory _site`.
- Check Open-Meteo's terms (non-commercial free tier) before relying on a public deployment.

## How it works

| Horizon | Method |
|---|---|
| Days 0-7 | NWP forecast, bias-corrected per lead: `truth ~ a*NWP + b*climatology + c`, fitted on archived forecasts vs ERA5. Rain chance from a per-lead logistic model. |
| Days 8-15 | Lead-7 corrected forecast relaxing linearly to climatology. |
| Days 16-35 | Climatology + decaying persistence of the last 14 days' anomaly (irradiation, temperature). Rain uses climatology only. This is a seasonal-tendency outlook, not a weather forecast. |

Week-ahead and month-ahead summaries aggregate the daily series.

## Backtest (honest)
Models are fitted on data up to 2024-12-31 and scored on 2025 onward. Baselines: raw NWP (what a phone app shows), climatology, persistence. Skill = 1 - MAE/MAE_baseline. Shown in the web app per site and returned by `/api/forecast/<site>`.

Example (Akosombo, 2025-01..2026-11): irradiation error 14-26% lower than raw NWP at days 1-7; 11-23% lower than climatology; beyond day 15 only ~4-8% better than climatology for irradiation. Wet/dry accuracy is about the same as raw NWP (73-76%), so no claim of improvement there. Run `python cli.py build` for all sites.

## Hackathon scoring
`python cli.py score entry.csv lawra` scores a CSV with columns `date,ghi_mj,tmean,rain` (daily forecasts) against ERA5 truth. Composite score = mean skill over the three variables vs climatology (0 = climatology, 1 = perfect). Note: ERA5 is the truth proxy; for a real competition use VRA plant pyranometer / met-mast data and a held-out period participants have not seen.

## Known limits and to-do before operational use
- **Site coordinates in `sites.json` are approximate and unverified.** Replace with surveyed plant coordinates.
- Truth is ERA5, not plant instruments. Swap in site measurements (pass a daily DataFrame with `ghi_mj, tmean, tmax, rain` to `fit_model`) to calibrate to the actual site.
- Irradiation is global horizontal. Plane-of-array / power forecasts need module tilt, bifaciality and plant model (next step, links to Brief 2 dashboard).
- No ensemble or satellite nowcasting; intra-day (hourly) forecasts are not modelled.
