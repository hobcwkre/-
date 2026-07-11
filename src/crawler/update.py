"""Orchestrate crawling: sync company lists, sync a range of trading days into the DB."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pandas as pd
from tqdm import tqdm

from ..storage import db
from .client import TpexClient
from .company_list import fetch_companies, fetch_otc_products_recent
from .daily_quotes import FETCHERS
from .market_index import fetch_index_month, month_starts

MARKETS = ("otc", "esb")


def sync_companies(client: TpexClient, conn: sqlite3.Connection) -> dict[str, int]:
    counts = {}
    for market in MARKETS:
        companies = fetch_companies(client, market)
        db.upsert_companies(conn, companies)
        counts[market] = len(companies)
    products = fetch_otc_products_recent(client)
    db.upsert_companies(conn, products)
    counts["etf_etn"] = len(products)
    return counts


def sync_index_range(client: TpexClient, conn: sqlite3.Connection, start: date, end: date) -> int:
    """Fetch 櫃買指數 daily closes for every month overlapping [start, end]."""
    total = 0
    for month in month_starts(start, end):
        try:
            df = fetch_index_month(client, month)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] index {month:%Y-%m} failed: {exc}")
            continue
        db.upsert_index_quotes(conn, df)
        total += len(df)
    return total


def _daterange(start: date, end: date):
    d = start
    one_day = timedelta(days=1)
    while d <= end:
        if d.weekday() < 5:  # skip Sat/Sun; national holidays just return empty and are skipped
            yield d
        d += one_day


def sync_daily_range(
    client: TpexClient,
    conn: sqlite3.Connection,
    market: str,
    start: date,
    end: date,
    valid_codes: set[str] | None = None,
    show_progress: bool = True,
) -> dict[str, int]:
    """Fetch every trading day in [start, end] for `market` and upsert into the DB.

    If valid_codes is given, rows whose code isn't in it are dropped (filters
    out ETFs/bonds/warrants that ride along in the OTC "all securities" feed).
    """
    fetch = FETCHERS[market]
    days = list(_daterange(start, end))
    iterator = tqdm(days, desc=f"sync {market}") if show_progress else days

    days_with_data = 0
    rows_written = 0
    for d in iterator:
        try:
            df = fetch(client, d)
        except Exception as exc:  # noqa: BLE001 - keep crawling past transient errors
            print(f"[warn] {market} {d.isoformat()} failed: {exc}")
            continue
        if df.empty:
            continue
        if valid_codes is not None:
            df = df[df["code"].isin(valid_codes)]
        if df.empty:
            continue
        db.upsert_quotes(conn, df)
        db.set_last_date(conn, market, d.isoformat())
        days_with_data += 1
        rows_written += len(df)

    return {"days_with_data": days_with_data, "rows_written": rows_written}
