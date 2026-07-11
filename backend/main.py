"""FastAPI backend for the TPEx Monte-Carlo backtest web app.

Run with:
    uvicorn backend.main:app --port 8600
"""
from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import backtest as bt
from . import monte_carlo as mc
from . import tpex_data

app = FastAPI(title="TPEx Monte-Carlo Backtest API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ---------------------------------------------------------------- lists

@app.get("/api/categories")
def categories():
    return {"categories": tpex_data.list_categories()}


@app.get("/api/industries")
def industries(category: str):
    return {"industries": tpex_data.list_industries(category)}


@app.get("/api/securities")
def securities(category: str, industry: str | None = None):
    return {"securities": tpex_data.list_securities(category, industry)}


@app.get("/api/coverage")
def coverage():
    return tpex_data.coverage()


# ---------------------------------------------------------------- backtest

class BacktestRequest(BaseModel):
    codes: list[str] = Field(min_length=1)
    start: str
    end: str
    strategy: str  # "ma" | "rsi"
    params: dict = {}
    capital: float = 1_000_000
    risk_pct: float = 10.0
    leverage: float = 1.0


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    results = {}
    errors = {}
    for code in req.codes:
        info = tpex_data.security_info(code)
        if info is None:
            errors[code] = "查無此標的（僅支援上櫃／興櫃）"
            continue
        prices = tpex_data.load_prices(code, info["market"], req.start, req.end)
        if prices.empty:
            errors[code] = "此區間無價格資料"
            continue
        try:
            result = bt.run_backtest(
                prices, req.strategy, req.params,
                capital=req.capital, risk_pct=req.risk_pct, leverage=req.leverage,
            )
        except ValueError as exc:
            errors[code] = str(exc)
            continue
        result["code"] = code
        result["name"] = info["name"]
        result["category"] = info["category"]
        results[code] = result
    if not results:
        raise HTTPException(400, detail={"message": "所有標的回測失敗", "errors": errors})
    return {"results": results, "errors": errors}


# ---------------------------------------------------------------- monte carlo (async job)

class MonteCarloRequest(BaseModel):
    trade_returns: list[float] = Field(min_length=2)
    n_sims: int = Field(default=1000, ge=10, le=100_000)
    mode: str = "reshuffle"
    capital: float = 1_000_000


JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _run_job(job_id: str, req: MonteCarloRequest) -> None:
    def cb(p: float) -> None:
        with JOBS_LOCK:
            JOBS[job_id]["progress"] = p

    try:
        result = mc.run_monte_carlo(
            req.trade_returns, n_sims=req.n_sims, mode=req.mode,
            capital=req.capital, progress_cb=cb,
        )
        with JOBS_LOCK:
            JOBS[job_id].update(status="done", progress=1.0, result=result)
    except Exception as exc:  # noqa: BLE001
        with JOBS_LOCK:
            JOBS[job_id].update(status="error", error=str(exc))


@app.post("/api/montecarlo")
def start_monte_carlo(req: MonteCarloRequest):
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "progress": 0.0}
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/montecarlo/{job_id}")
def poll_monte_carlo(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return dict(job)


# ---------------------------------------------------------------- frontend

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
