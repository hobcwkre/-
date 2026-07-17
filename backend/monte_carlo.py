"""Monte-Carlo analysis of a trade-return sequence — memory-bounded version.

Designed for a 512 MB container (Render free tier):

- Simulations run in CHUNKS (default 1000 paths at a time), fully vectorized
  in float32. Each chunk's (chunk × n_trades) matrices are released before
  the next chunk starts, so peak memory is O(chunk_size × n_trades), not
  O(n_sims × n_trades).
- Per-simulation we keep only three float32/int32 scalars (max drawdown,
  total return, trades-to-recover). The raw path matrix is never retained.
- The response does NOT carry raw per-simulation arrays anymore. The
  drawdown distribution is summarized server-side into a fixed 40-bin
  histogram plus percentiles (exact — computed from the full scalar arrays,
  which cost only n_sims × 4 bytes each). At 100k sims this shrinks the
  JSON payload (and the copy retained by the jobs store) from ~8 MB to a
  few KB with zero loss for what the UI actually renders.

Modes
-----
reshuffle : permute trade order. Order changes drawdown, not total return.
bootstrap : resample with replacement. Both total return and drawdown vary.
"""
from __future__ import annotations

import numpy as np

from .memlog import log_mem

MAX_PATHS_RETURNED = 200   # spaghetti-chart sample (≤200 × ≤400 points)
PATH_POINTS = 400          # downsample very long paths
DEFAULT_CHUNK = 1000
HIST_BINS = 40


def _simulate_original(returns: np.ndarray, capital: float) -> tuple[np.ndarray, float, float, int | None]:
    equity = capital * np.cumprod(1.0 + returns.astype(np.float64))
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
    return equity, max_dd, float(equity[-1] / capital - 1.0), recover


def _downsample(path: np.ndarray) -> list[float]:
    if len(path) > PATH_POINTS:
        idx = np.linspace(0, len(path) - 1, PATH_POINTS).astype(int)
        path = path[idx]
    return [round(float(v), 2) for v in path]


def run_monte_carlo(
    trade_returns: list[float],
    n_sims: int = 1000,
    mode: str = "reshuffle",
    capital: float = 1_000_000,
    seed: int | None = None,
    progress_cb=None,
    chunk_size: int = DEFAULT_CHUNK,
) -> dict:
    base = np.asarray(trade_returns, dtype=np.float32)
    n_trades = len(base)
    if n_trades < 2:
        raise ValueError("交易筆數不足（至少需要 2 筆交易才能模擬）")
    if mode not in ("reshuffle", "bootstrap"):
        raise ValueError("mode 必須是 reshuffle 或 bootstrap")
    rng = np.random.default_rng(seed)
    log_mem(f"mc start n_sims={n_sims} n_trades={n_trades} mode={mode}")

    orig_equity, orig_dd, orig_total, orig_recover = _simulate_original(base, capital)

    # per-simulation scalars only: 4 bytes × n_sims each (100k sims ≈ 1.2 MB total)
    drawdowns = np.empty(n_sims, dtype=np.float32)
    totals = np.empty(n_sims, dtype=np.float32)
    recovers = np.full(n_sims, -1, dtype=np.int32)
    paths: list[list[float]] = []

    col_idx = np.arange(n_trades + 1, dtype=np.int32)

    for start in range(0, n_sims, chunk_size):
        m = min(chunk_size, n_sims - start)
        if mode == "reshuffle":
            sample = np.tile(base, (m, 1))
            rng.permuted(sample, axis=1, out=sample)
        else:
            sample = rng.choice(base, size=(m, n_trades), replace=True)

        growth = np.cumprod(1.0 + sample, axis=1, dtype=np.float32)
        eq = np.empty((m, n_trades + 1), dtype=np.float32)
        eq[:, 0] = capital
        eq[:, 1:] = capital * growth
        del sample, growth

        peaks = np.maximum.accumulate(eq, axis=1)
        ddm = eq / peaks - 1.0
        drawdowns[start:start + m] = ddm.min(axis=1)
        totals[start:start + m] = eq[:, -1] / capital - 1.0

        trough = ddm.argmin(axis=1)
        peak_at_trough = peaks[np.arange(m), trough]
        ok = (eq >= peak_at_trough[:, None]) & (col_idx[None, :] > trough[:, None])
        has = ok.any(axis=1)
        first = ok.argmax(axis=1)
        recovers[start:start + m] = np.where(has, first - trough, -1).astype(np.int32)

        while len(paths) < MAX_PATHS_RETURNED and (len(paths) - start) < m:
            paths.append(_downsample(eq[len(paths) - start]))

        del eq, peaks, ddm, ok  # chunk buffers released before the next iteration
        if progress_cb:
            progress_cb(min(start + m, n_sims) / n_sims)

    log_mem("mc chunks done")

    def pct_dd(q: float) -> float:
        return float(np.percentile(drawdowns, 100 - q))  # deeper drawdown = lower value

    valid_rec = recovers[recovers >= 0].astype(np.float64)

    def pct_recover(q: float) -> float | None:
        return float(np.percentile(valid_rec, q)) if len(valid_rec) else None

    percentiles = [
        {
            "pct": q,
            "drawdown": pct_dd(q),
            "recover_trades": pct_recover(q),
            "total_return": float(np.percentile(totals, 100 - q)),
        }
        for q in (50, 75, 90, 95, 99)
    ]

    counts, edges = np.histogram(drawdowns, bins=HIST_BINS)
    result = {
        "mode": mode,
        "n_sims": n_sims,
        "n_trades": int(n_trades),
        "original": {
            "max_drawdown": orig_dd,
            "total_return": orig_total,
            "recover_trades": orig_recover,
            "equity": _downsample(orig_equity),
        },
        "dd_hist": {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]},
        "never_recovered_pct": float(np.mean(recovers < 0)),
        "percentiles": percentiles,
        "paths": paths,
        "dd_p95": pct_dd(95),
        "underestimated": bool(pct_dd(95) < orig_dd),
    }
    log_mem("mc result built")
    return result
