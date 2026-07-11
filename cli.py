"""Command-line entrypoint for the TPEx OTC/emerging backtest system.

Usage:
    python cli.py init-db
    python cli.py sync-companies
    python cli.py sync-quotes --market otc --start 2024-01-01 --end 2026-07-11
    python cli.py sync-quotes --market esb --start 2024-01-01 --end 2026-07-11
    python cli.py status
"""
from __future__ import annotations

import argparse
from datetime import date, datetime

from src.crawler.client import TpexClient
from src.crawler.update import sync_companies, sync_daily_range, sync_index_range
from src.storage import db


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def cmd_init_db(args: argparse.Namespace) -> None:
    conn = db.get_conn()
    db.init_db(conn)
    print(f"initialized {db.DEFAULT_DB_PATH}")


def cmd_sync_companies(args: argparse.Namespace) -> None:
    conn = db.get_conn()
    db.init_db(conn)
    client = TpexClient(delay=args.delay)
    counts = sync_companies(client, conn)
    print(f"companies synced: {counts}")


def cmd_sync_quotes(args: argparse.Namespace) -> None:
    conn = db.get_conn()
    db.init_db(conn)
    client = TpexClient(delay=args.delay)

    companies = db.load_companies(conn, args.market)
    if companies.empty:
        print(f"no {args.market} companies in DB yet; run sync-companies first "
              f"(continuing without code filtering)")
        valid_codes = None
    else:
        valid_codes = set(companies["code"])

    start = args.start or _parse_date(db.get_last_date(conn, args.market) or "2024-01-01")
    end = args.end or date.today()
    print(f"syncing {args.market} quotes {start} -> {end} (delay={args.delay}s/request)")
    result = sync_daily_range(client, conn, args.market, start, end, valid_codes=valid_codes)
    print(f"done: {result}")


def cmd_sync_index(args: argparse.Namespace) -> None:
    conn = db.get_conn()
    db.init_db(conn)
    client = TpexClient(delay=args.delay)
    start = args.start or _parse_date("2024-01-01")
    end = args.end or date.today()
    print(f"syncing 櫃買指數 {start} -> {end}")
    n = sync_index_range(client, conn, start, end)
    print(f"done: {n} index rows")


def cmd_status(args: argparse.Namespace) -> None:
    conn = db.get_conn()
    db.init_db(conn)
    for market in ("otc", "esb"):
        n_companies = len(db.load_companies(conn, market))
        lo, hi = db.covered_date_range(conn, market)
        print(f"{market}: {n_companies} securities, quotes {lo or '-'} .. {hi or '-'}")
    idx = db.load_index_series(conn)
    if len(idx):
        print(f"index: {len(idx)} rows, {idx.index.min().date()} .. {idx.index.max().date()}")
    else:
        print("index: no data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db").set_defaults(func=cmd_init_db)

    p = sub.add_parser("sync-companies")
    p.add_argument("--delay", type=float, default=0.4)
    p.set_defaults(func=cmd_sync_companies)

    p = sub.add_parser("sync-quotes")
    p.add_argument("--market", choices=["otc", "esb"], required=True)
    p.add_argument("--start", type=_parse_date, default=None, help="YYYY-MM-DD (default: resume from last sync)")
    p.add_argument("--end", type=_parse_date, default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--delay", type=float, default=0.4)
    p.set_defaults(func=cmd_sync_quotes)

    p = sub.add_parser("sync-index")
    p.add_argument("--start", type=_parse_date, default=None, help="YYYY-MM-DD (default: 2024-01-01)")
    p.add_argument("--end", type=_parse_date, default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--delay", type=float, default=0.4)
    p.set_defaults(func=cmd_sync_index)

    sub.add_parser("status").set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
