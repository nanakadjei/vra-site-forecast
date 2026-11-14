"""Command line tools.

  python cli.py forecast lawra            # print the 35-day forecast
  python cli.py build                     # (re)fit all sites and print backtest skill
  python cli.py score entry.csv lawra     # score a hackathon submission (date,ghi_mj,tmean,rain)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

import pandas as pd

from vra_forecast import openmeteo as om
from vra_forecast.backtest import score_submission
from vra_forecast.models import build_climatology
from vra_forecast.service import HISTORY_START, SPLIT, build_site, get_site, load_sites, site_forecast


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast"); f.add_argument("site")
    b = sub.add_parser("build"); b.add_argument("--site")
    s = sub.add_parser("score"); s.add_argument("csv"); s.add_argument("site")
    a = ap.parse_args(argv)

    if a.cmd == "forecast":
        r = site_forecast(a.site)
        df = pd.DataFrame(r["daily"])[["date", "lead", "ghi_mj", "tmean", "tmax", "rain", "rain_prob", "source"]]
        print(df.to_string(index=False)); print(json.dumps(r["summary"], indent=2))
    elif a.cmd == "build":
        for site in ([get_site(a.site)] if a.site else load_sites()):
            _, m = build_site(site, force=True)
            d1, d7 = m["short_range"][0], m["short_range"][-1]
            print(f"{site['id']:14s} ghi skill vs raw NWP d1 {d1['ghi_mj_skill_vs_raw_nwp']:+.0%}  d7 {d7['ghi_mj_skill_vs_raw_nwp']:+.0%}"
                  f"   vs clim d7 {d7['ghi_mj_skill_vs_clim']:+.0%}")
    elif a.cmd == "score":
        site = get_site(a.site)
        sub_df = pd.read_csv(a.csv)
        truth = om.fetch_era5(site["lat"], site["lon"], HISTORY_START, date.today())
        clim = build_climatology(truth[truth.index <= pd.Timestamp(SPLIT)])  # climatology baseline excludes test years
        print(json.dumps(score_submission(sub_df, truth, clim), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
