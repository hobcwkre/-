"""Trading strategies: each turns a price DataFrame into a target position series.

A strategy only sees data up to and including bar t when deciding the
position for bar t (no lookahead). The backtest engine is responsible for
shifting the resulting signal by one bar before applying it, so a signal
computed from today's close is executed at tomorrow's price.

Position values are 0 (flat) or 1 (fully long). Short-selling is out of scope.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):
    name: str = "strategy"

    @abstractmethod
    def generate_signals(self, price_df: pd.DataFrame) -> pd.Series:
        """Return a 0/1 position series indexed like price_df."""


class MovingAverageCross(Strategy):
    """Long while the short moving average is above the long moving average."""

    def __init__(self, short_window: int = 5, long_window: int = 20):
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self.short_window = short_window
        self.long_window = long_window
        self.name = f"MA Cross ({short_window}/{long_window})"

    def generate_signals(self, price_df: pd.DataFrame) -> pd.Series:
        close = price_df["close"]
        ma_short = close.rolling(self.short_window).mean()
        ma_long = close.rolling(self.long_window).mean()
        signal = (ma_short > ma_long).astype(int)
        signal[ma_long.isna()] = 0
        return signal


class RSIThreshold(Strategy):
    """Enter long when RSI drops below `lower`, exit when it rises above `upper`."""

    def __init__(self, period: int = 14, lower: float = 30.0, upper: float = 70.0):
        self.period = period
        self.lower = lower
        self.upper = upper
        self.name = f"RSI({period}) {lower}/{upper}"

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def generate_signals(self, price_df: pd.DataFrame) -> pd.Series:
        rsi = self._rsi(price_df["close"], self.period)
        position = pd.Series(0, index=price_df.index, dtype=int)
        in_position = False
        for dt, value in rsi.items():
            if not in_position and value < self.lower:
                in_position = True
            elif in_position and value > self.upper:
                in_position = False
            position.loc[dt] = int(in_position)
        return position


class BollingerBand(Strategy):
    """Enter long on a close below the lower band, exit on a close above the upper band."""

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std
        self.name = f"Bollinger({window}, {num_std}sigma)"

    def generate_signals(self, price_df: pd.DataFrame) -> pd.Series:
        close = price_df["close"]
        mid = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        lower = mid - self.num_std * std
        upper = mid + self.num_std * std

        position = pd.Series(0, index=price_df.index, dtype=int)
        in_position = False
        for dt in price_df.index:
            c, lo, hi = close.loc[dt], lower.loc[dt], upper.loc[dt]
            if pd.isna(lo) or pd.isna(hi):
                position.loc[dt] = 0
                continue
            if not in_position and c < lo:
                in_position = True
            elif in_position and c > hi:
                in_position = False
            position.loc[dt] = int(in_position)
        return position


STRATEGIES: dict[str, type[Strategy]] = {
    "ma_cross": MovingAverageCross,
    "rsi": RSIThreshold,
    "bollinger": BollingerBand,
}
