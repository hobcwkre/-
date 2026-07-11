"""Performance metrics for a backtest equity curve and trade log."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {}

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_days = (equity.index[-1] - equity.index[0]).days
    years = n_days / 365.25 if n_days > 0 else np.nan
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years and years > 0 else np.nan

    daily_ret = equity.pct_change().dropna()
    ann_vol = daily_ret.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (daily_ret.mean() * TRADING_DAYS_PER_YEAR) / ann_vol if ann_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    if trades is not None and not trades.empty:
        wins = trades[trades["return"] > 0]["return"]
        losses = trades[trades["return"] < 0]["return"]
        win_rate = len(wins) / len(trades)
        profit_factor = (wins.sum() / -losses.sum()) if losses.sum() < 0 else np.nan
        avg_trade_return = trades["return"].mean()
        num_trades = len(trades)
    else:
        win_rate = np.nan
        profit_factor = np.nan
        avg_trade_return = np.nan
        num_trades = 0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_return": avg_trade_return,
        "num_trades": num_trades,
    }
