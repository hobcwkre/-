"""Fetch the tradable-security universe from TPEx.

Three sources combine into the `companies` table (each row carries a
`category` label used by the portfolio UI):

  - OpenAPI mopsfin_t187ap03_O : 上櫃 company fundamentals  -> category 上櫃
  - OpenAPI mopsfin_t187ap03_R : 興櫃 company fundamentals  -> category 興櫃
  - POST afterTrading/otc type=EE / EN : the daily quote board filtered to
    ETFs / ETNs. Used only to harvest code+name lists; a bond ETF is
    labelled 債券ETF (code suffix B/C or 債 in the name), others ETF / ETN.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .client import TpexClient, to_query_date

# mopsfin_t187ap03_O = 上櫃股票基本資料, mopsfin_t187ap03_R = 興櫃公司基本資料
_ENDPOINTS = {
    "otc": ("mopsfin_t187ap03_O", "上櫃"),
    "esb": ("mopsfin_t187ap03_R", "興櫃"),
}

_COLUMNS = ["code", "market", "name", "industry_code", "listing_date", "category"]


def fetch_companies(client: TpexClient, market: str) -> pd.DataFrame:
    """market: 'otc' or 'esb'."""
    if market not in _ENDPOINTS:
        raise ValueError(f"unknown market: {market}")
    endpoint, category = _ENDPOINTS[market]
    rows = client.get_openapi(endpoint)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=_COLUMNS)
    out = pd.DataFrame(
        {
            "code": df["SecuritiesCompanyCode"].str.strip(),
            "market": market,
            "name": df["CompanyAbbreviation"].str.strip(),
            "industry_code": df.get("SecuritiesIndustryCode", "").astype(str).str.strip(),
            "listing_date": df.get("DateOfListing", "").astype(str).str.strip(),
            "category": category,
        }
    )
    return out


def _etf_category(code: str, name: str) -> str:
    if code.endswith(("B", "C")) or "債" in name:
        return "債券ETF"
    return "ETF"


def fetch_otc_products(client: TpexClient, d: date) -> pd.DataFrame:
    """ETF/ETN universe as listed on the OTC daily quote board for date `d`.

    Returns an empty frame on non-trading days.
    """
    frames = []
    for type_code, kind in (("EE", "ETF"), ("EN", "ETN"), ("WW", "權證")):
        body = {"date": to_query_date(d), "type": type_code, "id": "", "response": "json"}
        payload = client.post_query("afterTrading/otc", body)
        tables = payload.get("tables") or []
        rows = tables[0].get("data") if tables else []
        for r in rows:
            code, name = r[0].strip(), r[1].strip()
            category = _etf_category(code, name) if kind == "ETF" else kind
            frames.append(
                {
                    "code": code,
                    "market": "otc",
                    "name": name,
                    "industry_code": "",
                    "listing_date": "",
                    "category": category,
                }
            )
    return pd.DataFrame(frames, columns=_COLUMNS)


def fetch_otc_products_recent(client: TpexClient, lookback_days: int = 10) -> pd.DataFrame:
    """Walk back from today until we hit a trading day with product data."""
    d = date.today()
    for _ in range(lookback_days):
        df = fetch_otc_products(client, d)
        if not df.empty:
            return df
        d -= timedelta(days=1)
    return pd.DataFrame(columns=_COLUMNS)
