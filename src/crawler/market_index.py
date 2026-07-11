"""Fetch the TPEx 櫃買指數 (OTC market index) daily history.

POST /www/zh-tw/afterTrading/tradingIndex with any date inside a month
returns that whole month's daily rows:
  [日期(民國), 成交張數, 金額(仟元), 筆數, 櫃買指數, 漲/跌]
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .client import TpexClient, to_query_date


def _roc_to_iso(s: str) -> str:
    y, m, d = s.strip().split("/")
    return f"{int(y) + 1911:04d}-{int(m):02d}-{int(d):02d}"


def _num(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if s in ("", "-", "--"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_index_month(client: TpexClient, d: date) -> pd.DataFrame:
    """All daily index rows for the month containing `d`. Columns: date, close, change."""
    body = {"date": to_query_date(d.replace(day=1)), "id": "", "response": "json"}
    payload = client.post_query("afterTrading/tradingIndex", body)
    tables = payload.get("tables") or []
    rows = tables[0].get("data") if tables else []
    out = [
        {"date": _roc_to_iso(r[0]), "close": _num(r[4]), "change": _num(r[5])}
        for r in rows
    ]
    df = pd.DataFrame(out, columns=["date", "close", "change"])
    return df.dropna(subset=["close"])


def month_starts(start: date, end: date) -> list[date]:
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months
