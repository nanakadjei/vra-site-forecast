"""Web app + JSON API.  Run:  python app.py   then open http://127.0.0.1:5003"""
from __future__ import annotations

import threading

from flask import Flask, jsonify, render_template, request

from vra_forecast.service import load_sites, site_forecast

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
_lock = threading.Lock()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/sites")
def api_sites():
    return jsonify(load_sites())


@app.get("/api/forecast/<site_id>")
def api_forecast(site_id: str):
    try:
        with _lock:  # one model build at a time; results are cached on disk
            return jsonify(site_forecast(site_id, force=request.args.get("refresh") == "1"))
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except RuntimeError as e:  # upstream data service unavailable
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=False)
