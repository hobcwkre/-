"""Build the shareable single-file web app: embed SQLite data into web/template.html.

Usage:
    python export_web.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--min-days 5]

Output: web/index.html — fully self-contained (data + backtest engine in JS),
suitable for publishing as a static page. Re-run after syncing new data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.storage import db

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "web" / "template.html"
OUTPUT = ROOT / "web" / "index.html"


def build_payload(start: str | None, end: str | None, min_days: int) -> dict:
    conn = db.get_conn()
    db.init_db(conn)

    where, params = "", []
    if start:
        where += " AND date >= ?"
        params.append(start)
    if end:
        where += " AND date <= ?"
        params.append(end)

    dates = [
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT date FROM daily_quotes WHERE 1=1{where} ORDER BY date", params
        )
    ]
    date_pos = {d: i for i, d in enumerate(dates)}

    companies = db.load_companies(conn)
    meta = companies.set_index(["code", "market"])

    quotes = pd.read_sql_query(
        f"SELECT code, market, date, close FROM daily_quotes WHERE close IS NOT NULL{where}",
        conn,
        params=params,
    )

    securities = []
    for (code, market), grp in quotes.groupby(["code", "market"]):
        if len(grp) < min_days:
            continue
        try:
            info = meta.loc[(code, market)]
        except KeyError:
            continue
        prices: list[float | None] = [None] * len(dates)
        for d, c in zip(grp["date"], grp["close"]):
            prices[date_pos[d]] = round(float(c), 4)
        securities.append(
            {
                "c": code,
                "n": str(info["name"]),
                "k": str(info["category"]),
                "m": market,
                "p": prices,
            }
        )
    securities.sort(key=lambda s: (s["m"], s["c"]))

    idx = db.load_index_series(conn, start, end)
    index_closes: list[float | None] = [None] * len(dates)
    for d, v in idx.items():
        key = d.date().isoformat()
        if key in date_pos:
            index_closes[date_pos[key]] = round(float(v), 2)

    return {"dates": dates, "index": index_closes, "securities": securities}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--min-days", type=int, default=5)
    args = parser.parse_args()

    payload = build_payload(args.start, args.end, args.min_days)
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    blob = blob.replace("</", "<\\/")  # keep the inline <script> intact

    html = TEMPLATE.read_text(encoding="utf-8")
    if "__DATA_JSON__" not in html:
        raise SystemExit("template missing __DATA_JSON__ placeholder")
    OUTPUT.write_text(html.replace("__DATA_JSON__", blob), encoding="utf-8")

    n_dates = len(payload["dates"])
    n_sec = len(payload["securities"])
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"wrote {OUTPUT} — {n_sec} securities × {n_dates} days, {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
