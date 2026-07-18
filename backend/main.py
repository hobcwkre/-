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
from . import warrant_pricing as wp
from .memlog import log_mem

# ---- memory-budget guards (Render free tier: 512 MB) ----
MAX_BACKTEST_CODES = 20          # codes per request (processed sequentially)
MAX_BACKTEST_SPAN_DAYS = 4000    # ~11 years of daily data per code
MC_MAX_CELLS = 20_000_000        # n_sims × n_trades ceiling; above this n_sims is clamped
MAX_JOBS_KEPT = 10               # completed MC results retained in memory

import numpy as _np


def _json_safe(o):
    """Convert numpy scalars to native types and replace non-finite floats
    with None. Needed because np.float32 (unlike np.float64) is NOT a
    subclass of Python float, so FastAPI's encoder rejects it outright."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    if isinstance(o, _np.integer):
        return int(o)
    if isinstance(o, (_np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


app = FastAPI(title="TPEx Monte-Carlo Backtest API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def no_cache_html(request, call_next):
    """Stale cached index.html + freshly deployed API = broken pages (the
    front/back end ship together, so the HTML must revalidate every load)."""
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


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


@app.get("/api/memstat")
def memstat():
    """Current process memory (for verifying the 512MB-tier fixes in prod)."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        return {"rss_mb": round(info.rss / 1048576, 1), "vms_mb": round(info.vms / 1048576, 1)}
    except ImportError:
        return {"rss_mb": None, "error": "psutil not installed"}


_SCOPE_CATEGORIES = {"上櫃", "ETF", "債券ETF", "權證", "自訂資料"}


@app.get("/api/security/{code}")
def security_lookup(code: str):
    """Direct code lookup (for the type-a-code add flow), scoped to this app."""
    info = tpex_data.security_info(code.strip())
    if info is None or info["category"] not in _SCOPE_CATEGORIES:
        raise HTTPException(404, "查無此代碼（本系統僅支援上櫃股票／ETF／權證與自訂資料）")
    info["board"] = "自訂" if info["market"] == "custom" else "上櫃"
    return info


@app.get("/api/benchmark")
def benchmark(start: str, end: str):
    """櫃買指數 daily closes for the benchmark comparison."""
    s = tpex_data.load_benchmark(start, end)
    return {
        "dates": [str(d.date()) for d in s.index],
        "closes": [float(v) for v in s.values],
    }


# ---------------------------------------------------------------- warrant valuation

@app.get("/api/warrant/{code}/pricing")
def warrant_pricing(code: str, start: str, end: str, risk_free_rate: float = 1.6, hv_window: int = 60):
    """Theoretical price (via underlying HV) + implied vol, day by day, for one warrant."""
    terms = tpex_data.get_warrant_terms(code)
    if terms is None:
        raise HTTPException(404, "查無此權證的發行條款資料")
    if not terms["type"] or not terms["style"]:
        raise HTTPException(400, "此權證條款資料不完整（缺少買賣權或美/歐式別）")

    underlying_info = tpex_data.security_info(terms["underlying_code"])
    underlying_market = underlying_info["market"] if underlying_info else "otc"
    underlying_px = tpex_data.load_prices(terms["underlying_code"], underlying_market, start, end)
    warrant_px = tpex_data.load_prices(code, "otc", start, end)
    if underlying_px.empty or warrant_px.empty:
        raise HTTPException(400, "標的股或權證在此區間查無價格資料")

    result = wp.build_pricing_series(
        terms, underlying_px["close"].dropna(), warrant_px["close"].dropna(),
        risk_free_rate=risk_free_rate / 100, hv_window=hv_window,
    )
    result["code"] = code
    result["underlying_code"] = terms["underlying_code"]
    result["underlying_name"] = terms["underlying_name"]
    result["risk_free_rate"] = risk_free_rate
    result["hv_window"] = hv_window
    return _json_safe(result)


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
    strategy: str  # "ma" | "rsi" | "ma_multi"
    params: dict = {}
    capital: float = 1_000_000
    risk_pct: float = 10.0
    leverage: float = 1.0
    # per-code capital allocation in percent; missing codes share equally.
    # Normalized to 100% server-side so the client never has to be exact.
    weights: dict[str, float] | None = None


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    # -- request-size guard: keeps a single request's working set bounded
    if len(req.codes) > MAX_BACKTEST_CODES:
        raise HTTPException(400, f"一次最多回測 {MAX_BACKTEST_CODES} 檔標的（收到 {len(req.codes)} 檔）")
    try:
        from datetime import date as _d
        span = (_d.fromisoformat(req.end) - _d.fromisoformat(req.start)).days
    except ValueError:
        raise HTTPException(400, "日期格式錯誤（需 YYYY-MM-DD）")
    if span > MAX_BACKTEST_SPAN_DAYS:
        raise HTTPException(400, f"回測區間過長（{span} 天 > 上限 {MAX_BACKTEST_SPAN_DAYS} 天）")

    log_mem(f"backtest start codes={len(req.codes)} span={span}d")

    # ---- per-code capital allocation (weights normalized to sum to 100)
    raw_w = req.weights or {}
    weights = {c: max(float(raw_w.get(c, 0)), 0.0) for c in req.codes}
    if sum(weights.values()) <= 0:
        weights = {c: 1.0 for c in req.codes}  # default: equal split
    w_sum = sum(weights.values())
    weights = {c: w / w_sum for c, w in weights.items()}

    results = {}
    errors = {}
    # codes are processed one at a time; each code's price frame is released
    # (del) before the next is loaded, so peak memory is one series, not all
    for code in req.codes:
        info = tpex_data.security_info(code)
        if info is None:
            errors[code] = "查無此標的（僅支援上櫃／興櫃）"
            continue
        prices = tpex_data.load_prices(code, info["market"], req.start, req.end)
        if prices.empty:
            errors[code] = "此區間無價格資料"
            continue
        allocated = req.capital * weights[code]
        if allocated <= 0:
            errors[code] = "配置權重為 0，未執行回測"
            del prices
            continue
        try:
            result = bt.run_backtest(
                prices, req.strategy, req.params,
                capital=allocated, risk_pct=req.risk_pct, leverage=req.leverage,
            )
        except ValueError as exc:
            errors[code] = str(exc)
            continue
        finally:
            del prices
        result["code"] = code
        result["name"] = info["name"]
        result["category"] = info["category"]
        result["weight_pct"] = round(weights[code] * 100, 4)
        result["allocated_capital"] = allocated
        results[code] = result
    log_mem("backtest done")
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


def _run_job(job_id: str, req: MonteCarloRequest, n_sims: int, clamped: bool) -> None:
    def cb(p: float) -> None:
        with JOBS_LOCK:
            JOBS[job_id]["progress"] = p

    try:
        result = mc.run_monte_carlo(
            req.trade_returns, n_sims=n_sims, mode=req.mode,
            capital=req.capital, progress_cb=cb,
        )
        result["clamped"] = clamped
        result["requested_n_sims"] = req.n_sims
        with JOBS_LOCK:
            JOBS[job_id].update(status="done", progress=1.0, result=_json_safe(result))
            # evict oldest finished jobs so results don't accumulate for the
            # life of the process (each holds the 200 sampled paths)
            while len(JOBS) > MAX_JOBS_KEPT:
                oldest = next(iter(JOBS))
                if oldest == job_id or JOBS[oldest].get("status") == "running":
                    break
                del JOBS[oldest]
    except Exception as exc:  # noqa: BLE001
        with JOBS_LOCK:
            JOBS[job_id].update(status="error", error=str(exc))


@app.post("/api/montecarlo")
def start_monte_carlo(req: MonteCarloRequest):
    # -- simulation-budget guard: n_sims × n_trades bounded; auto-clamp n_sims
    n_trades = len(req.trade_returns)
    n_sims = req.n_sims
    clamped = False
    if n_sims * n_trades > MC_MAX_CELLS:
        n_sims = max(MC_MAX_CELLS // n_trades, 100)
        clamped = True
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "progress": 0.0}
    threading.Thread(target=_run_job, args=(job_id, req, n_sims, clamped), daemon=True).start()
    return {"job_id": job_id, "n_sims": n_sims, "clamped": clamped}


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
