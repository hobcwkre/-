"""Trade-based backtest for the Monte-Carlo web app.

Model
-----
- Signals (均線交叉 / RSI) are computed on daily closes; a signal decided on
  bar t-1 is executed at the close of bar t (no lookahead).
- 槓桿倍數 scales exposure: daily equity change = leverage × asset return.
- 單筆風險% is a per-trade stop-loss on equity: the trade is force-closed at
  the first close where the trade's equity loss reaches that percentage.
- Taiwan costs: 0.1425% commission each way + 0.3% tax on sells, applied to
  the levered notional.

Outputs per run: trade log (the Monte-Carlo input), daily equity curve,
performance metrics, and an OLS regression of strategy daily returns on the
asset's own close-to-close returns (rows: Const, Adj close).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

COMMISSION = 0.001425
SELL_TAX = 0.003
TRADING_DAYS = 252


# ---------------------------------------------------------------- signals

def signal_ma(close: pd.Series, short: int, long: int) -> pd.Series:
    ma_s = close.rolling(short).mean()
    ma_l = close.rolling(long).mean()
    sig = (ma_s > ma_l).astype(int)
    sig[ma_l.isna()] = 0
    return sig


def signal_rsi(close: pd.Series, period: int, lower: float, upper: float) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)
    sig = pd.Series(0, index=close.index, dtype=int)
    in_pos = False
    for dt, v in rsi.items():
        if not in_pos and v < lower:
            in_pos = True
        elif in_pos and v > upper:
            in_pos = False
        sig[dt] = int(in_pos)
    return sig


def build_signal(close: pd.Series, strategy: str, params: dict) -> tuple[pd.Series, str]:
    if strategy == "ma":
        s, l = int(params.get("short", 5)), int(params.get("long", 20))
        if s >= l:
            raise ValueError("短期均線需小於長期均線")
        return signal_ma(close, s, l), f"均線交叉 ({s}/{l})"
    if strategy == "rsi":
        p = int(params.get("period", 14))
        lo, hi = float(params.get("lower", 30)), float(params.get("upper", 70))
        return signal_rsi(close, p, lo, hi), f"RSI({p}) {lo:g}/{hi:g}"
    raise ValueError(f"未知策略: {strategy}")


# ---------------------------------------------------------------- fractional engine (多均線分批)

def run_backtest_ma_multi(
    price_df: pd.DataFrame,
    params: dict,
    capital: float = 1_000_000,
    leverage: float = 1.0,
) -> dict:
    """多均線分批策略（對照 TEJ 簡報做法）：

    - 期初以全額買入（部位比例 f = 1）。
    - 三條均線（預設 20/60/120，可調）兩兩構成三組配對；任一組發生
      死亡交叉（短均線跌破長均線）→ 賣出「原始投資金額」的 X%（部位 −X%）；
      黃金交叉 → 買回 X%（部位 +X%）。部位比例夾在 [0, 1]。
    - 訊號於次一交易日收盤執行；交叉僅在兩條均線皆有值時判定。
    - 成本：加碼收手續費、減碼收手續費＋證交稅，按變動的名目金額計。

    勝率／賺賠比對分批策略無明確定義（單筆進出不成對），回傳 NaN；
    蒙地卡羅請以每日報酬序列為輸入（前端已配合）。
    """
    close = price_df["close"].dropna()
    windows = sorted({int(params.get("w1", 20)), int(params.get("w2", 60)), int(params.get("w3", 120))})
    if len(windows) < 2:
        raise ValueError("至少需要兩條不同週期的均線")
    tranche = float(params.get("tranche", 20)) / 100.0
    if not (0 < tranche <= 1):
        raise ValueError("進出場買賣%數需介於 0 與 100 之間")
    if len(close) < max(windows) + 5:
        raise ValueError(f"價格資料不足（最長均線 {max(windows)} 日需要至少 {max(windows)+5} 個交易日）")

    mas = {w: close.rolling(w).mean() for w in windows}
    pairs = [(a, b) for i, a in enumerate(windows) for b in windows[i+1:]]

    n = len(close)
    dates = close.index
    # target position decided on bar t-1, executed at close of bar t
    target = np.empty(n)
    f = 1.0  # 期初全額買入
    pending_delta = 0.0
    deltas_at = np.zeros(n)
    for i in range(n):
        target[i] = f = min(1.0, max(0.0, f + pending_delta))
        deltas_at[i] = pending_delta
        pending_delta = 0.0
        for a, b in pairs:
            ma_a, ma_b = mas[a].iloc[i], mas[b].iloc[i]
            if i == 0 or np.isnan(ma_a) or np.isnan(ma_b):
                continue
            pa, pb = mas[a].iloc[i-1], mas[b].iloc[i-1]
            if np.isnan(pa) or np.isnan(pb):
                continue
            if pa >= pb and ma_a < ma_b:      # 死亡交叉 → 賣出一份
                pending_delta -= tranche
            elif pa <= pb and ma_a > ma_b:    # 黃金交叉 → 買回一份
                pending_delta += tranche

    equity = capital
    equity_curve = np.empty(n)
    trades: list[dict] = []
    prev_f = 0.0
    prev_price = close.iloc[0]
    for i in range(n):
        price = close.iloc[i]
        ret = price / prev_price - 1 if i > 0 else 0.0
        delta_f = target[i] - prev_f
        cost = 0.0
        if delta_f > 1e-12:
            cost = leverage * delta_f * COMMISSION
        elif delta_f < -1e-12:
            cost = leverage * (-delta_f) * (COMMISSION + SELL_TAX)
        equity = equity * (1 + leverage * prev_f * ret - cost)
        if abs(delta_f) > 1e-12:
            trades.append({
                "entry_date": str(dates[i].date()), "exit_date": "",
                "entry_price": round(float(price), 4), "exit_price": None,
                "return": None, "pnl": None,
                "reason": f"{'買進' if delta_f > 0 else '賣出'} {abs(delta_f)*100:.0f}%（部位 → {target[i]*100:.0f}%）",
            })
        equity_curve[i] = equity
        prev_f = target[i]
        prev_price = price

    eq = pd.Series(equity_curve, index=dates)
    daily_ret = eq.pct_change().dropna()
    asset_ret = close.pct_change().dropna()
    total_return = equity / capital - 1
    days = (dates[-1] - dates[0]).days
    years = days / 365.25 if days > 0 else float("nan")
    cagr = (equity / capital) ** (1 / years) - 1 if years and years > 0 else float("nan")
    vol = float(daily_ret.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = float(daily_ret.mean() * TRADING_DAYS / vol) if vol > 0 else float("nan")
    running_max = eq.cummax()
    max_dd = float((eq / running_max - 1).min())

    aligned = pd.concat([daily_ret.rename("y"), asset_ret.rename("x")], axis=1).dropna()
    regression = ols_summary(aligned["y"].to_numpy(), aligned["x"].to_numpy(), ("Const", "Adj close"))

    label = f"多均線分批 ({'/'.join(str(w) for w in windows)}, 每次 {tranche*100:.0f}%)"
    return {
        "strategy_label": label,
        "ma_windows": windows,
        "dates": [str(d.date()) for d in dates],
        "equity_curve": [round(float(v), 2) for v in equity_curve],
        "positions": [round(float(v), 4) for v in target],
        "close": [float(c) for c in close],
        "trades": trades,
        "metrics": {
            "total_return": total_return, "cagr": cagr, "volatility": vol,
            "sharpe": sharpe, "max_drawdown": max_dd,
            "win_rate": float("nan"), "payoff_ratio": float("nan"),
            "num_trades": len(trades),
        },
        "regression": regression,
    }


# ---------------------------------------------------------------- OLS

def ols_summary(y: np.ndarray, x: np.ndarray, names: tuple[str, str]) -> dict:
    """OLS y = b0 + b1*x with normal-approximation p-values."""
    n = len(y)
    if n < 3 or np.var(x) == 0:
        return {}
    x_bar, y_bar = x.mean(), y.mean()
    sxx = ((x - x_bar) ** 2).sum()
    beta = ((x - x_bar) * (y - y_bar)).sum() / sxx
    alpha = y_bar - beta * x_bar
    resid = y - (alpha + beta * x)
    dof = n - 2
    sigma2 = (resid**2).sum() / dof
    se_b = math.sqrt(sigma2 / sxx)
    se_a = math.sqrt(sigma2 * (1 / n + x_bar**2 / sxx))
    ss_tot = ((y - y_bar) ** 2).sum()

    def row(name, coef, se):
        t = coef / se if se > 0 else float("nan")
        p = math.erfc(abs(t) / math.sqrt(2)) if np.isfinite(t) else float("nan")
        return {
            "name": name, "coef": coef, "std_err": se, "t": t, "p": p,
            "ci_low": coef - 1.96 * se, "ci_high": coef + 1.96 * se,
        }

    return {
        "n": n,
        "r2": 1 - (resid**2).sum() / ss_tot if ss_tot > 0 else float("nan"),
        "rows": [row(names[0], alpha, se_a), row(names[1], beta, se_b)],
    }


# ---------------------------------------------------------------- engine

def run_backtest(
    price_df: pd.DataFrame,
    strategy: str,
    params: dict,
    capital: float = 1_000_000,
    risk_pct: float = 10.0,
    leverage: float = 1.0,
) -> dict:
    if strategy == "ma_multi":
        return run_backtest_ma_multi(price_df, params, capital=capital, leverage=leverage)
    close = price_df["close"].dropna()
    if len(close) < 5:
        raise ValueError("價格資料不足（少於 5 個交易日）")
    signal, strategy_label = build_signal(close, strategy, params)
    executed = signal.shift(1).fillna(0).astype(int)  # trade the bar after the signal

    stop_frac = max(risk_pct, 0.01) / 100.0  # per-trade equity stop-loss
    dates = close.index
    n = len(close)

    equity = capital
    equity_curve = np.empty(n)
    position_flags = np.zeros(n, dtype=int)  # 1 = holding at that day's close
    trades: list[dict] = []
    in_pos = False
    entry_price = entry_equity = 0.0
    entry_date = None

    def close_trade(i: int, reason: str) -> None:
        nonlocal equity, in_pos
        price = close.iloc[i]
        gross = leverage * (price / entry_price - 1)
        cost = leverage * (2 * COMMISSION + SELL_TAX)
        trade_ret = gross - cost
        pnl = entry_equity * trade_ret
        equity = entry_equity * (1 + trade_ret)
        trades.append({
            "entry_date": str(entry_date.date()), "exit_date": str(dates[i].date()),
            "entry_price": round(float(entry_price), 4), "exit_price": round(float(price), 4),
            "return": trade_ret, "pnl": pnl, "reason": reason,
        })
        in_pos = False

    for i in range(n):
        price = close.iloc[i]
        if in_pos:
            unrealized = leverage * (price / entry_price - 1)
            if unrealized <= -stop_frac:
                close_trade(i, "停損")
            elif executed.iloc[i] == 0:
                close_trade(i, "訊號出場")
            else:
                equity = entry_equity * (1 + unrealized)
        if not in_pos and executed.iloc[i] == 1 and i < n - 1:
            in_pos = True
            entry_price = price
            entry_equity = equity
            entry_date = dates[i]
        equity_curve[i] = equity
        position_flags[i] = 1 if in_pos else 0

    if in_pos:
        close_trade(n - 1, "期末平倉")
        equity_curve[-1] = equity
        position_flags[-1] = 0

    eq = pd.Series(equity_curve, index=dates)
    daily_ret = eq.pct_change().dropna()
    asset_ret = close.pct_change().dropna()

    total_return = equity / capital - 1
    days = (dates[-1] - dates[0]).days
    years = days / 365.25 if days > 0 else float("nan")
    cagr = (equity / capital) ** (1 / years) - 1 if years and years > 0 else float("nan")
    vol = float(daily_ret.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = float(daily_ret.mean() * TRADING_DAYS / vol) if vol > 0 else float("nan")
    running_max = eq.cummax()
    max_dd = float((eq / running_max - 1).min())

    rets = [t["return"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    win_rate = len(wins) / len(rets) if rets else float("nan")
    payoff = (np.mean(wins) / abs(np.mean(losses))) if wins and losses else float("nan")

    aligned = pd.concat([daily_ret.rename("y"), asset_ret.rename("x")], axis=1).dropna()
    regression = ols_summary(
        aligned["y"].to_numpy(), aligned["x"].to_numpy(), ("Const", "Adj close")
    )

    return {
        "strategy_label": strategy_label,
        "dates": [str(d.date()) for d in dates],
        "equity_curve": [round(float(v), 2) for v in equity_curve],
        "positions": [int(v) for v in position_flags],
        "close": [float(c) for c in close],
        "trades": trades,
        "metrics": {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "payoff_ratio": payoff,
            "num_trades": len(trades),
        },
        "regression": regression,
    }
