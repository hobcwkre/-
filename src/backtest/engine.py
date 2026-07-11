"""Vectorized single-asset daily backtest engine with Taiwan-style trading costs."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import compute_metrics
from .strategy import Strategy

# Defaults reflect typical Taiwan retail brokerage terms: 0.1425% commission
# (often discounted), 0.3% securities transaction tax on sells only.
DEFAULT_COMMISSION_RATE = 0.001425
DEFAULT_TAX_RATE = 0.003


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict


class Backtester:
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        commission_discount: float = 1.0,
        tax_rate: float = DEFAULT_TAX_RATE,
        slippage_bp: float = 0.0,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate * commission_discount
        self.tax_rate = tax_rate
        self.slippage_bp = slippage_bp

    def run(self, price_df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        price_df = price_df.dropna(subset=["close"]).copy()
        if price_df.empty:
            raise ValueError("price_df has no usable close prices")

        signals = strategy.generate_signals(price_df)
        # Trade on the bar AFTER the signal is known, avoiding lookahead bias.
        positions = signals.shift(1).fillna(0).astype(int)

        close = price_df["close"]
        daily_ret = close.pct_change().fillna(0)

        pos_change = positions.diff().fillna(positions.iloc[0])
        buy_cost = pos_change.clip(lower=0) * self.commission_rate
        sell_cost = (-pos_change.clip(upper=0)) * (self.commission_rate + self.tax_rate)
        slippage_cost = pos_change.abs() * (self.slippage_bp / 10_000)
        cost = buy_cost + sell_cost + slippage_cost

        strat_ret = positions * daily_ret - cost
        equity = self.initial_capital * (1 + strat_ret).cumprod()

        trades = self._extract_trades(close, positions)
        metrics = compute_metrics(equity, trades)
        metrics["strategy"] = strategy.name

        return BacktestResult(equity_curve=equity, positions=positions, trades=trades, metrics=metrics)

    @staticmethod
    def _extract_trades(close: pd.Series, positions: pd.Series) -> pd.DataFrame:
        pos_change = positions.diff().fillna(positions.iloc[0])
        trades = []
        entry_date = None
        entry_price = None
        for dt, chg in pos_change.items():
            if chg > 0 and entry_date is None:
                entry_date, entry_price = dt, close.loc[dt]
            elif chg < 0 and entry_date is not None:
                exit_price = close.loc[dt]
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": dt,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return": exit_price / entry_price - 1,
                        "open": False,
                    }
                )
                entry_date, entry_price = None, None
        if entry_date is not None:
            # Still holding at the end of the window: mark-to-market against
            # the last close so metrics (win rate, etc.) reflect it too.
            exit_price = close.iloc[-1]
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": close.index[-1],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": exit_price / entry_price - 1,
                    "open": True,
                }
            )
        return pd.DataFrame(trades, columns=["entry_date", "exit_date", "entry_price", "exit_price", "return", "open"])
