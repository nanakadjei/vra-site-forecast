"""Build the static site for GitHub Pages into ./_site

    python build_static.py              # all sites
    python build_static.py --site lawra # one site (development)

Output:
    _site/index.html        the web page, switched to static mode
    _site/data/sites.json   site list
    _site/data/<id>.json    the full forecast payload for each site (same shape as /api/forecast/<id>)
    _site/.nojekyll

Exits non-zero if any site fails, so the scheduled workflow does not publish a half-built site and
the previous good deployment stays online.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from vra_forecast.service import load_sites, site_forecast

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "_site"
STATIC_FLAG = "<script>window.VRA_STATIC = true;</script>\n</head>"


def write_json(path: Path, obj) -> None:
    # allow_nan=False: raise instead of writing NaN, which browsers cannot parse as JSON
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def check_payload(p: dict) -> None:
    d = p["daily"]
    assert len(d) == 35, f"expected 35 daily rows, got {len(d)}"
    assert [r["lead"] for r in d] == list(range(35)), "lead sequence broken"
    assert all(r["ghi_mj"] >= 0 and r["rain"] >= 0 for r in d), "negative irradiation or rain"


def build(only: str | None) -> int:
    sites = load_sites()
    if only:
        sites = [s for s in sites if s["id"] == only]
        if not sites:
            print(f"Unknown site {only!r}", file=sys.stderr)
            return 2
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data").mkdir(parents=True)
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    failed = []
    for s in sites:
        print(f"building {s['id']} ...", flush=True)
        try:
            payload = site_forecast(s["id"])
            payload["built_at"] = built_at
            check_payload(payload)
            write_json(OUT / "data" / f"{s['id']}.json", payload)
        except Exception as e:  # keep going so one failure shows every problem in the log
            print(f"  FAILED {s['id']}: {e}", file=sys.stderr)
            failed.append(s["id"])

    write_json(OUT / "data" / "sites.json", sites)
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert "</head>" in html
    (OUT / "index.html").write_text(html.replace("</head>", STATIC_FLAG, 1), encoding="utf-8")
    (OUT / ".nojekyll").write_text("")

    if failed:
        print(f"{len(failed)} site(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"done: {len(sites)} site(s) -> {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site")
    sys.exit(build(ap.parse_args().site))
