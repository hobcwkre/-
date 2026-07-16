"""Black-Scholes theoretical pricing for TPEx (上櫃) warrants.

Two independent things are computed, deliberately kept separate (see the
architecture discussion this module implements):

1. Theoretical price, using the UNDERLYING'S HISTORICAL volatility (HV) fed
   into Black-Scholes. Compared against the warrant's actual market price,
   this tells you whether the market is pricing the warrant rich or cheap
   in *price* terms.
2. Implied volatility (IV), solved backwards from the market price via
   Newton-Raphson (bisection fallback). Comparing IV to HV tells you
   whether the market is pricing the warrant rich or cheap in *volatility*
   terms — a genuinely different question. Using IV to compute the
   "theoretical price" would be circular (BS(IV) reproduces the market
   price by construction), so it never feeds back into (1).

Caveat: Taiwan warrants can be American-style, but Black-Scholes prices
European options. For a non-dividend-paying underlying this is a standard,
widely-used approximation (early exercise of a call is never optimal
without dividends); it is flagged in the API response so the frontend can
disclose it rather than presenting American-style warrants as exact.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_SQRT_2PI = math.sqrt(2 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bs_price_greeks(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> dict:
    """Per-underlying-share Black-Scholes price and Greeks (before ratio scaling)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return {"price": intrinsic, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc = math.exp(-r * T)
    if is_call:
        price = S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        theta = (-S * _norm_pdf(d1) * sigma / (2 * sqrt_t) - r * K * disc * _norm_cdf(d2)) / TRADING_DAYS
    else:
        price = K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0
        theta = (-S * _norm_pdf(d1) * sigma / (2 * sqrt_t) + r * K * disc * _norm_cdf(-d2)) / TRADING_DAYS
    gamma = _norm_pdf(d1) / (S * sigma * sqrt_t)
    vega = S * _norm_pdf(d1) * sqrt_t / 100  # per 1 vol point (0.01)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def implied_vol(
    market_price: float, S: float, K: float, T: float, r: float, ratio: float, is_call: bool
) -> float | None:
    """Solve sigma such that ratio * BS(sigma) == market_price. None if unsolvable."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0 or ratio <= 0:
        return None
    target = market_price / ratio

    sigma = 0.4
    for _ in range(50):
        g = bs_price_greeks(S, K, T, r, sigma, is_call)
        diff = g["price"] - target
        vega_per_unit = g["vega"] * 100  # undo the /100 scaling for the derivative
        if abs(diff) < 1e-6:
            return sigma
        if vega_per_unit <= 1e-8:
            break
        sigma -= diff / vega_per_unit
        if sigma <= 0 or sigma > 5:
            break
    else:
        return max(sigma, 1e-4)

    # Newton failed to converge (or stepped out of bounds) — bisection fallback
    lo, hi = 1e-4, 5.0
    f_lo = bs_price_greeks(S, K, T, r, lo, is_call)["price"] - target
    f_hi = bs_price_greeks(S, K, T, r, hi, is_call)["price"] - target
    if f_lo * f_hi > 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        f_mid = bs_price_greeks(S, K, T, r, mid, is_call)["price"] - target
        if abs(f_mid) < 1e-6:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def historical_vol_series(underlying_close: pd.Series, window: int = 60) -> pd.Series:
    """Rolling annualized historical volatility of the underlying's daily returns."""
    rets = underlying_close.pct_change()
    return rets.rolling(window).std() * math.sqrt(TRADING_DAYS)


def build_pricing_series(
    terms: dict,
    underlying_close: pd.Series,
    warrant_close: pd.Series,
    risk_free_rate: float,
    hv_window: int = 60,
) -> dict:
    """Aligned daily series: theoretical price (HV-based), market price, IV, mispricing, greeks."""
    is_call = terms["type"] == "call"
    K, ratio = terms["strike"], terms["ratio"]
    expiry = date.fromisoformat(terms["expiry_date"])

    hv = historical_vol_series(underlying_close, hv_window)
    idx = underlying_close.index.intersection(warrant_close.index)
    idx = idx.sort_values()

    rows = []
    for d in idx:
        S = float(underlying_close.loc[d])
        wprice = float(warrant_close.loc[d])
        sigma_hv = hv.loc[d] if d in hv.index else np.nan
        T = max((expiry - d.date()).days, 0) / 365.0
        theo = None
        if pd.notna(sigma_hv) and sigma_hv > 0 and T > 0:
            g = bs_price_greeks(S, K, T, risk_free_rate, float(sigma_hv), is_call)
            theo = g["price"] * ratio
        iv = implied_vol(wprice, S, K, T, risk_free_rate, ratio, is_call) if T > 0 else None
        greeks = None
        if iv is not None:
            g = bs_price_greeks(S, K, T, risk_free_rate, iv, is_call)
            # per_share = standard textbook Greeks (as if you held the underlying option
            # directly); per_warrant = scaled by 履約比例, i.e. how much one warrant unit
            # actually moves — the two are easy to conflate, so both are reported labeled.
            greeks = {
                "delta_per_share": g["delta"], "delta_per_warrant": g["delta"] * ratio,
                "gamma_per_share": g["gamma"], "gamma_per_warrant": g["gamma"] * ratio,
                "vega_per_share": g["vega"], "vega_per_warrant": g["vega"] * ratio,
                "theta_per_share": g["theta"], "theta_per_warrant": g["theta"] * ratio,
            }
        rows.append({
            "date": str(d.date()),
            "underlying_close": S,
            "market_price": wprice,
            "theoretical_price": theo,
            "mispricing_pct": (wprice / theo - 1) if theo and theo > 0 else None,
            "hv": float(sigma_hv) if pd.notna(sigma_hv) else None,
            "iv": iv,
            "days_to_expiry": int((expiry - d.date()).days),
            "greeks": greeks,
        })
    return {"rows": rows, "expiry": terms["expiry_date"], "strike": K, "ratio": ratio,
            "type": terms["type"], "style": terms["style"]}
