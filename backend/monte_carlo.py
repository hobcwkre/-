"""Monte-Carlo analysis of a trade-return sequence.

Input: per-trade returns on equity (fractions, e.g. +0.042 / -0.018), taken
from the backtest trade log. Equity compounds multiplicatively per trade.

Modes
-----
reshuffle : permute trade order (np.random.shuffle). Order changes drawdown
            but not total return.
bootstrap : resample with replacement (np.random.choice). Both total return
            and drawdown vary.

Per simulation we record: max drawdown, total return, and trades-to-recover
(number of trades from the max-drawdown trough until equity regains its prior
peak; None if it never recovers within the sequence).
"""
from __future__ import annotations

import numpy as np

MAX_PATHS_RETURNED = 200  # spaghetti-chart sample
PATH_POINTS = 400         # downsample very long paths


def _simulate_path(returns: np.ndarray, capital: float) -> tuple[np.ndarray, float, float, int | None]:
    equity = capital * np.cumprod(1.0 + returns)
    equity = np.concatenate(([capital], equity))
    peaks = np.maximum.accumulate(equity)
    dd = equity / peaks - 1.0
    max_dd = float(dd.min())
    trough = int(dd.argmin())
    recover: int | None = None
    peak_before = peaks[trough]
    for j in range(trough + 1, len(equity)):
        if equity[j] >= peak_before:
            recover = j - trough
            break
    total = float(equity[-1] / capital - 1.0)
    return equity, max_dd, total, recover


def run_monte_carlo(
    trade_returns: list[float],
    n_sims: int = 1000,
    mode: str = "reshuffle",
    capital: float = 1_000_000,
    seed: int | None = None,
    progress_cb=None,
) -> dict:
    base = np.asarray(trade_returns, dtype=float)
    if len(base) < 2:
        raise ValueError("交易筆數不足（至少需要 2 筆交易才能模擬）")
    if mode not in ("reshuffle", "bootstrap"):
        raise ValueError("mode 必須是 reshuffle 或 bootstrap")
    rng = np.random.default_rng(seed)

    orig_equity, orig_dd, orig_total, orig_recover = _simulate_path(base, capital)

    drawdowns = np.empty(n_sims)
    totals = np.empty(n_sims)
    recovers: list[int | None] = []
    paths: list[list[float]] = []

    for k in range(n_sims):
        if mode == "reshuffle":
            sample = rng.permutation(base)
        else:
            sample = rng.choice(base, size=len(base), replace=True)
        equity, max_dd, total, recover = _simulate_path(sample, capital)
        drawdowns[k] = max_dd
        totals[k] = total
        recovers.append(recover)
        if k < MAX_PATHS_RETURNED:
            if len(equity) > PATH_POINTS:
                idx = np.linspace(0, len(equity) - 1, PATH_POINTS).astype(int)
                equity = equity[idx]
            paths.append([round(float(v), 2) for v in equity])
        if progress_cb and (k + 1) % max(1, n_sims // 50) == 0:
            progress_cb((k + 1) / n_sims)

    def pct_dd(q: float) -> float:
        return float(np.percentile(drawdowns, 100 - q))  # deeper drawdown = lower value

    recover_arr = np.array([r if r is not None else np.nan for r in recovers], dtype=float)

    def pct_recover(q: float) -> float | None:
        valid = recover_arr[~np.isnan(recover_arr)]
        if len(valid) == 0:
            return None
        return float(np.percentile(valid, q))

    percentiles = [
        {"pct": q, "drawdown": pct_dd(q), "recover_trades": pct_recover(q)}
        for q in (50, 75, 90, 95, 99)
    ]

    return {
        "mode": mode,
        "n_sims": n_sims,
        "n_trades": int(len(base)),
        "original": {
            "max_drawdown": orig_dd,
            "total_return": orig_total,
            "recover_trades": orig_recover,
            "equity": [round(float(v), 2) for v in orig_equity],
        },
        "drawdowns": [float(v) for v in drawdowns],
        "totals": [float(v) for v in totals],
        "never_recovered_pct": float(np.mean(np.isnan(recover_arr))),
        "percentiles": percentiles,
        "paths": paths,
        "dd_p95": pct_dd(95),
        "underestimated": bool(pct_dd(95) < orig_dd),  # MC 95th deeper than original
    }
