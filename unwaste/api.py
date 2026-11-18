"""
Step 10 · Serve the screen and the JSON outputs.

WHAT THIS FILE DOES (say this to a judge):
  "A small FastAPI app. It serves the manager screen and exposes the nightly outputs
   as JSON endpoints: /api/scoreboard, /api/actions, /api/demand-truth. In production
   the store-ops app would call these same endpoints."

Run:  uvicorn unwaste.api:app --port 8000     then open http://localhost:8000
Owner: Aniz.
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

WEB = Path(__file__).resolve().parent.parent / "web"
app = FastAPI(title="Unwaste API")


def _read(name: str):
    f = WEB / "data" / f"{name}.json"
    if not f.exists():
        raise HTTPException(404, f"{name}.json not found: run python -m unwaste.pipeline first")
    return json.loads(f.read_text())


@app.get("/api/scoreboard")
def scoreboard():
    return _read("scoreboard")


@app.get("/api/actions")
def actions(store: str | None = None, night: str | None = None):
    data = _read("actions")
    rows = [a for a in data["actions"]
            if (store is None or a["store_name"] == store) and (night is None or a["date"] == night)]
    return {"source": data["source"], "count": len(rows), "actions": rows}


@app.get("/api/demand-truth")
def demand_truth():
    return _read("demand_truth")


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
