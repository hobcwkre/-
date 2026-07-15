"""FastAPI backend for the TPEx Monte-Carlo backtest web app.

Run with:
    uvicorn backend.main:app --port 8600
"""
from __future__ import annotations

import io
import math
import threading
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import backtest as bt
from . import monte_carlo as mc
from . import tpex_data

def _json_safe(o):
    """Replace non-finite floats (NaN/inf) with None so responses stay valid JSON."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


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


@app.get("/api/benchmark")
def benchmark(start: str, end: str):
    """櫃買指數 daily closes for the benchmark comparison."""
    s = tpex_data.load_benchmark(start, end)
    return {
        "dates": [str(d.date()) for d in s.index],
        "closes": [float(v) for v in s.values],
    }


# ---------------------------------------------------------------- custom data upload

_DATE_COLS = {"date", "日期", "交易日期", "time", "年月日"}
_CLOSE_COLS = {"close", "adj close", "adj_close", "adjclose", "收盤", "收盤價", "收盤價(元)", "price"}


def _parse_price_csv(content: bytes) -> pd.DataFrame:
    text = None
    for enc in ("utf-8-sig", "utf-8", "cp950", "big5"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("無法解讀檔案編碼（支援 UTF-8 / Big5）")
    df = pd.read_csv(io.StringIO(text))
    cols = {str(c).strip().lower(): c for c in df.columns}
    date_col = next((cols[k] for k in cols if k in _DATE_COLS), None)
    close_col = next((cols[k] for k in cols if k in _CLOSE_COLS), None)
    if date_col is None or close_col is None:
        raise ValueError(
            f"找不到日期／收盤價欄位。偵測到的欄位：{list(df.columns)}；"
            "日期欄需為 date/日期，收盤欄需為 close/Adj Close/收盤/收盤價 之一"
        )

    def norm_date(s: str) -> str | None:
        s = str(s).strip().replace("/", "-").replace(".", "-")
        parts = s.split("-")
        if len(parts) == 3 and parts[0].isdigit():
            y = int(parts[0])
            if y < 1911:  # 民國年
                y += 1911
            try:
                return f"{y:04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
            except ValueError:
                return None
        return None

    out = pd.DataFrame(
        {
            "date": df[date_col].map(norm_date),
            "close": pd.to_numeric(
                df[close_col].astype(str).str.replace(",", "").str.strip(), errors="coerce"
            ),
        }
    ).dropna()
    out = out[out["close"] > 0].drop_duplicates(subset="date").sort_values("date")
    if len(out) < 5:
        raise ValueError(f"有效資料列不足（僅 {len(out)} 列），至少需要 5 個交易日")
    return out


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...), name: str = Form(default="")):
    content = await file.read()
    if len(content) > 10_000_000:
        raise HTTPException(400, "檔案過大（上限 10 MB）")
    try:
        df = _parse_price_csv(content)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    ds_name = (name or (file.filename or "custom").rsplit(".", 1)[0]).strip()[:60]
    info = tpex_data.add_custom_dataset(ds_name, df)
    return info


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
    return _json_safe({"results": results, "errors": errors})


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
            JOBS[job_id].update(status="done", progress=1.0, result=_json_safe(result))
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
