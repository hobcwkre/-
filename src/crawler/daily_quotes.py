"""Fetch one trading day's full-market quotes for OTC (上櫃) and emerging (興櫃) stocks.

Both endpoints below return every listed security for a single date in one
call, which is far cheaper than querying stock-by-stock:

  OTC  : POST /www/zh-tw/afterTrading/otc   body: date, type=EW, id, response=json
         type=EW = "所有證券(不含權證、牛熊證)" (all securities excl. warrants/bull-bear
         certs). Still includes bond ETFs etc; caller should filter to the
         known company-code universe.

  ESB  : POST /www/zh-tw/emerging/des010    body: date, id, response=json
         "日行情表(電腦議價點選成交)" - the standard computer-matched quote board,
         which covers the large majority of emerging-stock trading.

Reverse-engineered from https://www.tpex.org.tw (public pages), 2026-07-11.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .client import TpexClient, to_query_date


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if s in ("", "-", "--"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_otc_daily(client: TpexClient, d: date) -> pd.DataFrame:
    body = {"date": to_query_date(d), "type": "EW", "id": "", "response": "json"}
    payload = client.post_query("afterTrading/otc", body)
    return _parse_otc(payload, d)


def _parse_otc(payload: dict, d: date) -> pd.DataFrame:
    cols = [
        "code", "market", "date", "open", "high", "low", "close",
        "avg_price", "volume", "amount", "transactions", "change",
    ]
    tables = payload.get("tables") or []
    rows = tables[0].get("data") if tables else []
    if not rows:
        return pd.DataFrame(columns=cols)
    out = []
    for r in rows:
        out.append(
            {
                "code": r[0].strip(),
                "market": "otc",
                "date": d.isoformat(),
                "open": _num(r[4]),
                "high": _num(r[5]),
                "low": _num(r[6]),
                "close": _num(r[2]),
                "avg_price": None,
                "volume": _num(r[7]),
                "amount": _num(r[8]),
                "transactions": _num(r[9]),
                "change": _num(r[3]),
            }
        )
    return pd.DataFrame(out, columns=cols)


def fetch_esb_daily(client: TpexClient, d: date) -> pd.DataFrame:
    body = {"date": to_query_date(d), "id": "", "response": "json"}
    payload = client.post_query("emerging/des010", body)
    return _parse_esb(payload, d)


def _parse_esb(payload: dict, d: date) -> pd.DataFrame:
    cols = [
        "code", "market", "date", "open", "high", "low", "close",
        "avg_price", "volume", "amount", "transactions", "change",
    ]
    tables = payload.get("tables") or []
    rows = tables[0].get("data") if tables else []
    if not rows:
        return pd.DataFrame(columns=cols)
    out = []
    for r in rows:
        out.append(
            {
                "code": r[0].strip(),
                "market": "esb",
                "date": d.isoformat(),
                "open": None,  # 興櫃 is dealer-quote driven; no exchange "open" price
                "high": _num(r[8]),
                "low": _num(r[9]),
                "close": _num(r[10]),
                "avg_price": _num(r[4]),
                "volume": _num(r[11]),
                "amount": _num(r[12]),
                "transactions": _num(r[13]),
                "change": _num(r[6]),
            }
        )
    return pd.DataFrame(out, columns=cols)


FETCHERS = {"otc": fetch_otc_daily, "esb": fetch_esb_daily}
