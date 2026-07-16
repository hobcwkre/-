"""Fetch OTC warrant static issuance terms from TPEx OpenAPI.

Endpoint: tpex_warrant_issue (上櫃權證發行基本資料) — a single GET returns every
live warrant's contract terms in one call (~9k rows), so this is bulk-synced
like the company list, unlike warrant price history which is fetched
per-code on demand (see daily_quotes.fetch_stock_month).
"""
from __future__ import annotations

import pandas as pd

from .client import TpexClient

_TYPE_MAP = {"認購": "call", "認售": "put"}
_STYLE_MAP = {"歐式": "european", "美式": "american"}


def _date_to_iso(s: str) -> str | None:
    """ListedDate/ExpiryDate in this endpoint are already AD, 8-digit YYYYMMDD
    (unlike the 7-digit ROC 'Date' field elsewhere in the same response)."""
    s = (s or "").strip()
    if len(s) != 8 or not s.isdigit():
        return None
    y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
    if not (1900 < y < 2100 and 1 <= m <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


def fetch_warrant_terms(client: TpexClient) -> pd.DataFrame:
    rows = client.get_openapi("tpex_warrant_issue")
    out = []
    for r in rows:
        try:
            strike = float(r["LatestExercisePrice"])
            ratio = float(r["Latest ExerciseRatio"])
        except (KeyError, ValueError):
            continue
        out.append(
            {
                "code": r["Code"].strip(),
                "underlying_code": r["UnderlyingStockCode"].strip(),
                "underlying_name": r["UnderlyingStock"].strip(),
                "type": _TYPE_MAP.get(r.get("Type", "").strip(), ""),
                "style": _STYLE_MAP.get(r.get("American/European", "").strip(), ""),
                "strike": strike,
                "ratio": ratio,
                "listed_date": _date_to_iso(r.get("ListedDate", "")),
                "expiry_date": _date_to_iso(r.get("ExpiryDate", "")),
            }
        )
    df = pd.DataFrame(out)
    return df[(df["strike"] > 0) & (df["ratio"] > 0) & df["expiry_date"].notna()]
