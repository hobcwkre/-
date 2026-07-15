"""Security lists (cascading dropdown data) and price loading for the MC web app.

Scope: 上櫃 only — 股票 / ETF(含債券ETF) / 權證, plus user-uploaded custom
datasets (自訂資料). The SQLite universe is built from TPEx-only sources
(see src/crawler), so no TWSE-listed security can appear.

Warrants are NOT bulk-synced (≈9k live codes would balloon the DB); their
price history is fetched per-code on demand via afterTrading/tradingStock
(one month per request) and cached into daily_quotes + fetched_months.
"""
from __future__ import annotations

import sys
import threading
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crawler.client import TpexClient  # noqa: E402
from src.crawler.daily_quotes import fetch_stock_month  # noqa: E402
from src.crawler.market_index import month_starts  # noqa: E402
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

CUSTOM_CATEGORY = "自訂資料"
CUSTOM_PREFIX = "U"  # custom dataset codes look like U1, U2, ...

_client_lock = threading.Lock()
_client: TpexClient | None = None


def _get_client() -> TpexClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = TpexClient(delay=0.15)
        return _client


def _conn():
    conn = db.get_conn()
    db.init_db(conn)
    return conn


def list_categories() -> list[str]:
    conn = _conn()
    companies = db.load_companies(conn)
    present = set(companies["category"])
    tiers = []
    if "上櫃" in present:
        tiers.append("股票")
    if present & {"ETF", "債券ETF"}:
        tiers.append("ETF")
    if "權證" in present:
        tiers.append("權證")
    if len(db.list_custom_datasets(conn)):
        tiers.append(CUSTOM_CATEGORY)
    return tiers


def list_industries(category: str) -> list[dict]:
    """Industry tier — only meaningful for 股票 (上櫃)."""
    if category != "股票":
        return []
    companies = db.load_companies(_conn())
    stocks = companies[companies["category"] == "上櫃"]
    codes = sorted({c for c in stocks["industry_code"] if c and c in INDUSTRY_NAMES})
    return [{"code": c, "name": INDUSTRY_NAMES[c]} for c in codes]


def list_securities(category: str, industry: str | None = None) -> list[dict]:
    conn = _conn()
    if category == CUSTOM_CATEGORY:
        ds = db.list_custom_datasets(conn)
        return [
            {
                "code": f"{CUSTOM_PREFIX}{r['id']}",
                "name": r["name"],
                "market": "custom",
                "category": CUSTOM_CATEGORY,
                "board": "自訂",
            }
            for _, r in ds.iterrows()
        ]
    companies = db.load_companies(conn)
    if category == "股票":
        subset = companies[companies["category"] == "上櫃"]
        if industry:
            subset = subset[subset["industry_code"] == industry]
    elif category == "ETF":
        subset = companies[companies["category"].isin(["ETF", "債券ETF"])]
    elif category == "權證":
        subset = companies[companies["category"] == "權證"]
    else:
        subset = companies.iloc[0:0]
    subset = subset.sort_values("code")
    return [
        {
            "code": r["code"],
            "name": r["name"],
            "market": r["market"],
            "category": r["category"],
            "board": "上櫃",
        }
        for _, r in subset.iterrows()
    ]


def security_info(code: str) -> dict | None:
    conn = _conn()
    if code.startswith(CUSTOM_PREFIX) and code[len(CUSTOM_PREFIX):].isdigit():
        ds = db.list_custom_datasets(conn)
        hit = ds[ds["id"] == int(code[len(CUSTOM_PREFIX):])]
        if not hit.empty:
            return {
                "code": code, "name": hit.iloc[0]["name"],
                "market": "custom", "category": CUSTOM_CATEGORY,
            }
        return None
    companies = db.load_companies(conn)
    hit = companies[companies["code"] == code]
    if hit.empty:
        return None
    # this app is 上櫃-scoped: prefer the otc row when a code exists on both boards
    otc = hit[hit["market"] == "otc"]
    r = (otc if not otc.empty else hit).iloc[0]
    return {"code": r["code"], "name": r["name"], "market": r["market"], "category": r["category"]}


def _ensure_on_demand(code: str, start: str, end: str) -> None:
    """Fetch missing months of a warrant's history from TPEx and cache them."""
    conn = _conn()
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    current_month = date.today().strftime("%Y-%m")
    client = _get_client()
    for m in month_starts(s, e):
        key = m.strftime("%Y-%m")
        if key != current_month and db.month_fetched(conn, code, key):
            continue
        try:
            df = fetch_stock_month(client, code, m)
        except Exception:  # noqa: BLE001 - a failed month just stays uncached
            continue
        if not df.empty:
            db.upsert_quotes(conn, df)
        db.mark_month_fetched(conn, code, key)


def load_prices(code: str, market: str, start: str, end: str) -> pd.DataFrame:
    conn = _conn()
    if market == "custom":
        ds_id = int(code[len(CUSTOM_PREFIX):])
        return db.load_custom_series(conn, ds_id, start, end)
    info = security_info(code)
    if info and info["category"] == "權證":
        _ensure_on_demand(code, start, end)
    return db.load_price_series(conn, code, market, start, end)


def load_benchmark(start: str, end: str) -> pd.Series:
    return db.load_index_series(_conn(), start, end)


def coverage() -> dict:
    conn = _conn()
    lo_otc, hi_otc = db.covered_date_range(conn, "otc")
    return {"otc": [lo_otc, hi_otc]}


def add_custom_dataset(name: str, df: pd.DataFrame) -> dict:
    conn = _conn()
    ds_id = db.add_custom_dataset(conn, name, df)
    return {
        "code": f"{CUSTOM_PREFIX}{ds_id}",
        "name": name,
        "rows": len(df),
        "start": df["date"].min(),
        "end": df["date"].max(),
    }
