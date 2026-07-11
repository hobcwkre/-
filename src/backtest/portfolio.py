"""Multi-asset portfolio backtest: weighted buy & hold with optional rebalancing,
benchmarked against the TPEx 櫃買指數.

Prices are aligned on the union of trading dates and forward-filled (a stock
that didn't trade on a given day keeps its last close). The portfolio starts
on the first date where every asset has a price.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS_PER_YEAR, compute_metrics

REBALANCE_FREQ = {"none": None, "monthly": "MS", "quarterly": "QS"}


def compute_risk_metrics(equity: pd.Series, benchmark_equity: pd.Series | None) -> dict:
    """Risk statistics for the portfolio, incl. an OLS market-model regression
    of portfolio daily returns on benchmark (櫃買指數) daily returns:
        r_p = alpha + beta * r_m + e
    Reports coef / std err / t / p / 95% CI for both alpha and beta.
    """
    r = equity.pct_change().dropna()
    out: dict = {}
    if len(r) >= 2:
        downside = r.clip(upper=0)
        downside_dev = float(np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR))
        out["downside_dev"] = downside_dev
        out["sortino"] = (r.mean() * TRADING_DAYS_PER_YEAR / downside_dev) if downside_dev > 0 else np.nan
        out["var95"] = float(r.quantile(0.05))

    if benchmark_equity is None or len(benchmark_equity) < 3:
        return out
    b = benchmark_equity.pct_change().dropna()
    aligned = pd.concat([r.rename("p"), b.rename("m")], axis=1, join="inner").dropna()
    n = len(aligned)
    if n < 3:
        return out
    rp, rm = aligned["p"].to_numpy(), aligned["m"].to_numpy()
    x_bar, y_bar = rm.mean(), rp.mean()
    sxx = ((rm - x_bar) ** 2).sum()
    if sxx <= 0:
        return out
    beta = ((rm - x_bar) * (rp - y_bar)).sum() / sxx
    alpha = y_bar - beta * x_bar
    resid = rp - (alpha + beta * rm)
    dof = n - 2
    sigma2 = (resid**2).sum() / dof if dof > 0 else np.nan
    se_beta = math.sqrt(sigma2 / sxx)
    se_alpha = math.sqrt(sigma2 * (1 / n + x_bar**2 / sxx))

    def _p_value(t_stat: float) -> float:
        # two-sided, normal approximation (n is daily-return sample size)
        return math.erfc(abs(t_stat) / math.sqrt(2))

    ss_tot = ((rp - y_bar) ** 2).sum()
    out["regression"] = {
        "n": n,
        "r2": 1 - (resid**2).sum() / ss_tot if ss_tot > 0 else np.nan,
        "rows": [
            {
                "name": "Alpha（截距，日）",
                "coef": alpha,
                "se": se_alpha,
                "t": alpha / se_alpha if se_alpha > 0 else np.nan,
                "p": _p_value(alpha / se_alpha) if se_alpha > 0 else np.nan,
                "lo": alpha - 1.96 * se_alpha,
                "hi": alpha + 1.96 * se_alpha,
            },
            {
                "name": "Beta（櫃買指數）",
                "coef": beta,
                "se": se_beta,
                "t": beta / se_beta if se_beta > 0 else np.nan,
                "p": _p_value(beta / se_beta) if se_beta > 0 else np.nan,
                "lo": beta - 1.96 * se_beta,
                "hi": beta + 1.96 * se_beta,
            },
        ],
    }
    out["beta"] = beta
    out["alpha_annual"] = alpha * TRADING_DAYS_PER_YEAR
    out["r2"] = out["regression"]["r2"]
    out["correlation"] = float(np.corrcoef(rp, rm)[0, 1])
    return out


@dataclass
class PortfolioResult:
    equity: pd.Series                # portfolio value over time
    benchmark_equity: pd.Series | None  # same capital tracking the index
    asset_values: pd.DataFrame       # per-asset market value over time
    asset_returns: pd.Series         # per-asset total return over the window
    metrics: dict


def run_portfolio(
    prices: pd.DataFrame,
    weights: pd.Series,
    initial_capital: float = 1_000_000,
    rebalance: str = "none",
    benchmark: pd.Series | None = None,
) -> PortfolioResult:
    """prices: columns = asset codes, index = dates, values = close prices.
    weights: indexed by the same codes; will be normalized to sum to 1.
    """
    if rebalance not in REBALANCE_FREQ:
        raise ValueError(f"rebalance must be one of {list(REBALANCE_FREQ)}")
    prices = prices.sort_index().ffill()
    prices = prices.dropna(how="any")  # start once every asset has a price
    if len(prices) < 2:
        raise ValueError("重疊的交易日不足（需要至少 2 天所有標的皆有價格）")

    weights = weights.reindex(prices.columns).fillna(0.0)
    if weights.sum() <= 0:
        raise ValueError("權重總和必須大於 0")
    weights = weights / weights.sum()

    freq = REBALANCE_FREQ[rebalance]
    if freq is None:
        rebalance_dates = {prices.index[0]}
    else:
        # first trading day of each period, plus the very first day
        period_first = prices.groupby(prices.index.to_period(freq[0])).head(1).index
        rebalance_dates = set(period_first) | {prices.index[0]}

    values = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    shares = None
    equity_now = initial_capital
    for dt in prices.index:
        if dt in rebalance_dates or shares is None:
            shares = (equity_now * weights) / prices.loc[dt]
        row_values = shares * prices.loc[dt]
        values.loc[dt] = row_values
        equity_now = row_values.sum()

    equity = values.sum(axis=1)

    asset_returns = prices.iloc[-1] / prices.iloc[0] - 1

    benchmark_equity = None
    bench_return = np.nan
    if benchmark is not None and len(benchmark):
        bench = benchmark.sort_index().reindex(prices.index).ffill().dropna()
        if len(bench) >= 2:
            benchmark_equity = initial_capital * bench / bench.iloc[0]
            bench_return = bench.iloc[-1] / bench.iloc[0] - 1

    metrics = compute_metrics(equity, pd.DataFrame())
    metrics["benchmark_return"] = bench_return
    metrics["excess_return"] = (
        metrics["total_return"] - bench_return if pd.notna(bench_return) else np.nan
    )
    metrics.update(compute_risk_metrics(equity, benchmark_equity))

    return PortfolioResult(
        equity=equity,
        benchmark_equity=benchmark_equity,
        asset_values=values,
        asset_returns=asset_returns,
        metrics=metrics,
    )
