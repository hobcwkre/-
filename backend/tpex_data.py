"""Security lists (cascading dropdown data) and price loading for the MC web app.

All data is strictly TPEx (上櫃 + 興櫃) — the SQLite universe is built from
TPEx-only sources (see src/crawler), so no TWSE-listed security can appear.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage import db  # noqa: E402

# TPEx official industry classification codes (as used by the TPEx quote pages)
INDUSTRY_NAMES = {
    "02": "食品工業", "03": "塑膠工業", "04": "紡織纖維", "05": "電機機械",
    "06": "電器電纜", "08": "玻璃陶瓷", "10": "鋼鐵工業", "11": "橡膠工業",
    "14": "建材營造", "15": "航運業", "16": "觀光餐旅", "17": "金融保險",
    "20": "其他", "21": "化學工業", "22": "生技醫療", "23": "油電燃氣",
    "24": "半導體", "25": "電腦及週邊設備", "26": "光電", "27": "通信網路",
    "28": "電子零組件", "29": "電子通路", "30": "資訊服務", "31": "其他電子",
    "32": "文化創意", "33": "農業科技", "34": "電子商務", "35": "綠能環保",
    "36": "數位雲端", "37": "運動休閒", "38": "居家生活", "80": "管理股票",
}

# market-category tier (first dropdown); 股票 spans both TPEx boards
STOCK_CATS = {"上櫃", "興櫃"}
CATEGORY_TIERS = ["股票", "ETF", "債券ETF", "ETN"]


def _conn():
    conn = db.get_conn()
    db.init_db(conn)
    return conn


def list_categories() -> list[str]:
    companies = db.load_companies(_conn())
    present = set(companies["category"])
    tiers = []
    if present & STOCK_CATS:
        tiers.append("股票")
    for cat in ["ETF", "債券ETF", "ETN"]:
        if cat in present:
            tiers.append(cat)
    return tiers


def list_industries(category: str) -> list[dict]:
    """Industry tier — only meaningful for 股票."""
    if category != "股票":
        return []
    companies = db.load_companies(_conn())
    stocks = companies[companies["category"].isin(STOCK_CATS)]
    codes = sorted({c for c in stocks["industry_code"] if c and c in INDUSTRY_NAMES})
    return [{"code": c, "name": INDUSTRY_NAMES[c]} for c in codes]


def list_securities(category: str, industry: str | None = None) -> list[dict]:
    companies = db.load_companies(_conn())
    if category == "股票":
        subset = companies[companies["category"].isin(STOCK_CATS)]
        if industry:
            subset = subset[subset["industry_code"] == industry]
    else:
        subset = companies[companies["category"] == category]
    subset = subset.sort_values("code")
    return [
        {
            "code": r["code"],
            "name": r["name"],
            "market": r["market"],
            "category": r["category"],
            "board": "上櫃" if r["market"] == "otc" else "興櫃",
        }
        for _, r in subset.iterrows()
    ]


def load_prices(code: str, market: str, start: str, end: str) -> pd.DataFrame:
    return db.load_price_series(_conn(), code, market, start, end)


def load_benchmark(start: str, end: str) -> pd.Series:
    return db.load_index_series(_conn(), start, end)


def coverage() -> dict:
    conn = _conn()
    lo_otc, hi_otc = db.covered_date_range(conn, "otc")
    lo_esb, hi_esb = db.covered_date_range(conn, "esb")
    return {"otc": [lo_otc, hi_otc], "esb": [lo_esb, hi_esb]}


def security_info(code: str) -> dict | None:
    companies = db.load_companies(_conn())
    hit = companies[companies["code"] == code]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {"code": r["code"], "name": r["name"], "market": r["market"], "category": r["category"]}
