"""SQLite storage for TPEx company metadata and daily quotes."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "tpex.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    code TEXT NOT NULL,
    market TEXT NOT NULL CHECK(market IN ('otc', 'esb')),
    name TEXT,
    industry_code TEXT,
    listing_date TEXT,
    category TEXT,
    PRIMARY KEY (code, market)
);

CREATE TABLE IF NOT EXISTS index_quotes (
    date TEXT PRIMARY KEY,
    close REAL,
    change REAL
);

CREATE TABLE IF NOT EXISTS daily_quotes (
    code TEXT NOT NULL,
    market TEXT NOT NULL CHECK(market IN ('otc', 'esb')),
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    avg_price REAL,
    volume REAL,
    amount REAL,
    transactions REAL,
    change REAL,
    PRIMARY KEY (code, market, date)
);
CREATE INDEX IF NOT EXISTS idx_quotes_lookup ON daily_quotes(market, code, date);

CREATE TABLE IF NOT EXISTS crawl_state (
    market TEXT PRIMARY KEY,
    last_date TEXT
);

-- months already fetched on demand per code (warrants), to avoid refetching
-- months where the security legitimately had zero trades
CREATE TABLE IF NOT EXISTS fetched_months (
    code TEXT NOT NULL,
    month TEXT NOT NULL,
    PRIMARY KEY (code, month)
);

-- warrant static issuance terms (BS pricing inputs), bulk-synced from
-- tpex_warrant_issue; refreshed periodically since strike/ratio can be
-- adjusted (ex-dividend, cash capital increase) over a warrant's life
CREATE TABLE IF NOT EXISTS warrant_terms (
    code TEXT PRIMARY KEY,
    underlying_code TEXT NOT NULL,
    underlying_name TEXT,
    type TEXT NOT NULL,        -- 'call' or 'put'
    style TEXT NOT NULL,       -- 'european' or 'american'
    strike REAL NOT NULL,
    ratio REAL NOT NULL,       -- 履約比例: shares of underlying per warrant unit
    listed_date TEXT,
    expiry_date TEXT NOT NULL,
    synced_at TEXT NOT NULL
);

-- user-uploaded price datasets
CREATE TABLE IF NOT EXISTS custom_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS custom_quotes (
    dataset_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    close REAL NOT NULL,
    PRIMARY KEY (dataset_id, date)
);
"""


def get_conn(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: Streamlit reruns callbacks on different threads
    # than the one that created the cached connection; access here stays
    # effectively sequential (one script rerun at a time), so this is safe.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    # migrate pre-category databases in place
    cols = {row[1] for row in conn.execute("PRAGMA table_info(companies)")}
    if "category" not in cols:
        conn.execute("ALTER TABLE companies ADD COLUMN category TEXT")
    conn.commit()


def upsert_companies(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    rows = df[["code", "market", "name", "industry_code", "listing_date", "category"]].values.tolist()
    conn.executemany(
        """INSERT INTO companies (code, market, name, industry_code, listing_date, category)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(code, market) DO UPDATE SET
             name=excluded.name,
             industry_code=excluded.industry_code,
             listing_date=excluded.listing_date,
             category=excluded.category""",
        rows,
    )
    conn.commit()


def upsert_quotes(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    cols = ["code", "market", "date", "open", "high", "low", "close",
            "avg_price", "volume", "amount", "transactions", "change"]
    rows = df[cols].where(pd.notna(df[cols]), None).values.tolist()
    conn.executemany(
        f"""INSERT INTO daily_quotes ({", ".join(cols)})
            VALUES ({", ".join(["?"] * len(cols))})
            ON CONFLICT(code, market, date) DO UPDATE SET
              open=excluded.open, high=excluded.high, low=excluded.low,
              close=excluded.close, avg_price=excluded.avg_price,
              volume=excluded.volume, amount=excluded.amount,
              transactions=excluded.transactions, change=excluded.change""",
        rows,
    )
    conn.commit()


def get_last_date(conn: sqlite3.Connection, market: str) -> str | None:
    row = conn.execute("SELECT last_date FROM crawl_state WHERE market=?", (market,)).fetchone()
    return row[0] if row else None


def set_last_date(conn: sqlite3.Connection, market: str, date_str: str) -> None:
    conn.execute(
        """INSERT INTO crawl_state (market, last_date) VALUES (?, ?)
           ON CONFLICT(market) DO UPDATE SET last_date=excluded.last_date""",
        (market, date_str),
    )
    conn.commit()


def load_companies(conn: sqlite3.Connection, market: str | None = None) -> pd.DataFrame:
    base = (
        "SELECT code, market, name, industry_code, listing_date, "
        "COALESCE(category, CASE market WHEN 'esb' THEN '興櫃' ELSE '上櫃' END) AS category "
        "FROM companies"
    )
    if market:
        return pd.read_sql_query(base + " WHERE market=? ORDER BY code", conn, params=(market,))
    return pd.read_sql_query(base + " ORDER BY market, code", conn)


def search_companies(conn: sqlite3.Connection, market: str, keyword: str, limit: int = 20) -> pd.DataFrame:
    like = f"%{keyword}%"
    return pd.read_sql_query(
        """SELECT * FROM companies WHERE market=? AND (code LIKE ? OR name LIKE ?)
           ORDER BY code LIMIT ?""",
        conn,
        params=(market, like, like, limit),
    )


_PRICE_COLUMNS = ("open", "high", "low", "close", "avg_price", "volume",
                  "amount", "transactions", "change")


def load_price_series(
    conn: sqlite3.Connection,
    code: str,
    market: str,
    start: str | None = None,
    end: str | None = None,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Daily rows for one security. Pass columns=("close",) to load only what
    a computation needs instead of the full OHLCV row (10x narrower frame)."""
    cols = columns or _PRICE_COLUMNS
    assert all(c in _PRICE_COLUMNS for c in cols)
    query = f"SELECT date, {', '.join(cols)} " \
            "FROM daily_quotes WHERE market=? AND code=?"
    params: list = [market, code]
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)
    query += " ORDER BY date"
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    return df.set_index("date")


def covered_date_range(conn: sqlite3.Connection, market: str) -> tuple[str | None, str | None]:
    row = conn.execute(
        "SELECT MIN(date), MAX(date) FROM daily_quotes WHERE market=?", (market,)
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def upsert_warrant_terms(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    from datetime import datetime

    now = datetime.now().isoformat(timespec="seconds")
    cols = ["code", "underlying_code", "underlying_name", "type", "style",
            "strike", "ratio", "listed_date", "expiry_date"]
    rows = [tuple(r) + (now,) for r in df[cols].itertuples(index=False, name=None)]
    conn.executemany(
        f"""INSERT INTO warrant_terms ({", ".join(cols)}, synced_at)
            VALUES ({", ".join(["?"] * (len(cols) + 1))})
            ON CONFLICT(code) DO UPDATE SET
              underlying_code=excluded.underlying_code, underlying_name=excluded.underlying_name,
              type=excluded.type, style=excluded.style, strike=excluded.strike,
              ratio=excluded.ratio, listed_date=excluded.listed_date,
              expiry_date=excluded.expiry_date, synced_at=excluded.synced_at""",
        rows,
    )
    conn.commit()


def get_warrant_terms(conn: sqlite3.Connection, code: str) -> dict | None:
    row = conn.execute(
        """SELECT code, underlying_code, underlying_name, type, style, strike, ratio,
                  listed_date, expiry_date, synced_at
           FROM warrant_terms WHERE code=?""",
        (code,),
    ).fetchone()
    if row is None:
        return None
    cols = ["code", "underlying_code", "underlying_name", "type", "style", "strike",
            "ratio", "listed_date", "expiry_date", "synced_at"]
    return dict(zip(cols, row))


def warrant_terms_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM warrant_terms").fetchone()[0]


def month_fetched(conn: sqlite3.Connection, code: str, month: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM fetched_months WHERE code=? AND month=?", (code, month)
    ).fetchone() is not None


def mark_month_fetched(conn: sqlite3.Connection, code: str, month: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO fetched_months (code, month) VALUES (?, ?)", (code, month)
    )
    conn.commit()


# ---------------------------------------------------------------- custom datasets

def add_custom_dataset(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> int:
    """df columns: date (ISO str), close. Replaces an existing dataset of the same name."""
    from datetime import datetime

    row = conn.execute("SELECT id FROM custom_datasets WHERE name=?", (name,)).fetchone()
    if row:
        ds_id = row[0]
        conn.execute("DELETE FROM custom_quotes WHERE dataset_id=?", (ds_id,))
    else:
        cur = conn.execute(
            "INSERT INTO custom_datasets (name, created) VALUES (?, ?)",
            (name, datetime.now().isoformat(timespec="seconds")),
        )
        ds_id = cur.lastrowid
    conn.executemany(
        "INSERT OR REPLACE INTO custom_quotes (dataset_id, date, close) VALUES (?, ?, ?)",
        [(ds_id, d, float(c)) for d, c in zip(df["date"], df["close"])],
    )
    conn.commit()
    return ds_id


def list_custom_datasets(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT d.id, d.name, d.created, COUNT(q.date) AS rows,
                  MIN(q.date) AS start, MAX(q.date) AS end
           FROM custom_datasets d LEFT JOIN custom_quotes q ON q.dataset_id = d.id
           GROUP BY d.id ORDER BY d.id""",
        conn,
    )


def load_custom_series(
    conn: sqlite3.Connection, ds_id: int, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    query = "SELECT date, close FROM custom_quotes WHERE dataset_id=?"
    params: list = [ds_id]
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)
    query += " ORDER BY date"
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    return df.set_index("date")


def upsert_index_quotes(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    rows = df[["date", "close", "change"]].where(pd.notna(df), None).values.tolist()
    conn.executemany(
        """INSERT INTO index_quotes (date, close, change) VALUES (?, ?, ?)
           ON CONFLICT(date) DO UPDATE SET close=excluded.close, change=excluded.change""",
        rows,
    )
    conn.commit()


def load_index_series(
    conn: sqlite3.Connection, start: str | None = None, end: str | None = None
) -> pd.Series:
    query = "SELECT date, close FROM index_quotes WHERE 1=1"
    params: list = []
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)
    query += " ORDER BY date"
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    return df.set_index("date")["close"]
