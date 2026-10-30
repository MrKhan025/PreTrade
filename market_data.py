"""
Market data engine for PreTrade.

Fetches live quotes, watchlist batches, nearest-expiry option chains, ATM implied
volatility, 13-week T-bill risk-free rates, daily OHLCV history, and intraday bars
via `yfinance`, with automatic deterministic fallback when offline or in test environments.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd

from black_scholes import bs_price_dividend

logger = logging.getLogger(__name__)

_FALLBACK_TICKERS: dict[str, dict[str, Any]] = {
    "AAPL": {"name": "Apple Inc.", "price": 228.50, "prev_close": 226.40, "iv": 24.5, "div": 0.0044},
    "MSFT": {"name": "Microsoft Corporation", "price": 425.00, "prev_close": 421.80, "iv": 22.0, "div": 0.0072},
    "SPY": {"name": "SPDR S&P 500 ETF Trust", "price": 582.00, "prev_close": 579.50, "iv": 15.2, "div": 0.0125},
    "QQQ": {"name": "Invesco QQQ Trust", "price": 498.00, "prev_close": 494.20, "iv": 19.8, "div": 0.0058},
    "NVDA": {"name": "NVIDIA Corporation", "price": 138.60, "prev_close": 135.20, "iv": 46.5, "div": 0.0003},
    "TSLA": {"name": "Tesla, Inc.", "price": 254.00, "prev_close": 248.90, "iv": 54.0, "div": 0.0},
    "AMZN": {"name": "Amazon.com, Inc.", "price": 210.40, "prev_close": 208.10, "iv": 28.5, "div": 0.0},
    "META": {"name": "Meta Platforms, Inc.", "price": 592.00, "prev_close": 586.00, "iv": 32.0, "div": 0.0034},
}


def _fallback_profile(symbol: str) -> dict[str, Any]:
    sym = str(symbol).strip().upper() or "AAPL"
    if sym in _FALLBACK_TICKERS:
        return dict(_FALLBACK_TICKERS[sym])
    seed = sum(ord(ch) * (i + 1) for i, ch in enumerate(sym)) % 10_000
    rng = np.random.default_rng(seed)
    price = round(float(rng.uniform(45.0, 350.0)), 2)
    chg_pct = float(rng.uniform(-2.2, 2.5))
    prev = round(price / (1.0 + chg_pct / 100.0), 2)
    iv = round(float(rng.uniform(18.0, 42.0)), 1)
    return {"name": f"{sym} Corp.", "price": price, "prev_close": prev, "iv": iv, "div": 0.005}


def fetch_risk_free_rate(default: float = 0.045) -> float:
    """Fetch the 13-week US Treasury Bill yield (`^IRX`) via yfinance, falling back to `default`."""
    try:
        import yfinance as yf

        tbill = yf.Ticker("^IRX")
        hist = tbill.history(period="5d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            last_yield_pct = float(hist["Close"].dropna().iloc[-1])
            if 0.0 <= last_yield_pct <= 25.0:
                return round(last_yield_pct / 100.0, 4)
    except Exception as exc:
        logger.debug("yfinance fallback triggered: %s", exc)
    return float(default)


def fetch_quote(symbol: str) -> dict[str, Any]:
    """Fetch a live equity/ETF quote via yfinance with safe fallback."""
    sym = str(symbol).strip().upper()
    if not sym:
        raise ValueError("Ticker symbol must not be empty.")

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        hist = ticker.history(period="5d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            closes = hist["Close"].dropna()
            if not closes.empty:
                price = float(closes.iloc[-1])
                prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else price
                change_pct = ((price - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
                info: dict[str, Any] = {}
                try:
                    info = ticker.info or {}
                except Exception:
                    info = {}
                div_yield = float(info.get("dividendYield") or 0.0)
                if div_yield > 1.0:
                    div_yield /= 100.0
                name = str(info.get("shortName") or info.get("longName") or sym)
                return {
                    "symbol": sym,
                    "name": name,
                    "price": round(price, 2),
                    "prev_close": round(prev_close, 2),
                    "change_pct": round(change_pct, 2),
                    "dividend_yield": round(max(div_yield, 0.0), 4),
                    "currency": str(info.get("currency") or "USD"),
                    "as_of": now_str,
                    "source": "yfinance",
                }
    except Exception as exc:
        logger.debug("yfinance fallback triggered: %s", exc)

    prof = _fallback_profile(sym)
    price = float(prof["price"])
    prev_close = float(prof["prev_close"])
    change_pct = ((price - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
    return {
        "symbol": sym,
        "name": str(prof["name"]),
        "price": round(price, 2),
        "prev_close": round(prev_close, 2),
        "change_pct": round(change_pct, 2),
        "dividend_yield": float(prof["div"]),
        "currency": "USD",
        "as_of": now_str,
        "source": "fallback",
    }


def fetch_quotes_batch(symbols: Sequence[str]) -> list[dict[str, Any]]:
    """Fetch quotes for a batch of watchlist symbols."""
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for raw in symbols:
        sym = str(raw).strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        try:
            results.append(fetch_quote(sym))
        except Exception:
            continue
    return results


def _build_synthetic_chain_summary(
    symbol: str,
    spot: float,
    base_iv_pct: float = 24.0,
    target_dte: int = 30,
    rate: float = 0.045,
    dividend_yield: float = 0.0,
) -> dict[str, Any]:
    """Generate a realistic ATM option chain preview when live options data is unavailable."""
    dte = max(int(target_dte), 7)
    expiry_date = (datetime.now(timezone.utc) + timedelta(days=dte)).strftime("%Y-%m-%d")
    step = 5.0 if spot >= 100 else (2.5 if spot >= 30 else 1.0)
    atm_strike = round(spot / step) * step
    strikes = [max(1.0, atm_strike + i * step) for i in range(-4, 5)]
    base_sigma = max(base_iv_pct / 100.0, 0.05)

    rows: list[dict[str, Any]] = []
    for k in strikes:
        skew = 1.0 + 0.25 * ((spot - k) / max(spot, 1.0))
        iv_dec = float(np.clip(base_sigma * skew, 0.05, 2.50))
        for opt_type in ("Call", "Put"):
            theo = bs_price_dividend(
                S=spot,
                K=k,
                T=float(dte),
                sigma=iv_dec,
                r=rate,
                dividend_yield=dividend_yield,
                option_type=opt_type.lower(),
            )
            spread = max(0.05, round(theo * 0.03, 2))
            bid = max(0.01, round(theo - 0.5 * spread, 2))
            ask = round(bid + spread, 2)
            rows.append(
                {
                    "type": opt_type,
                    "strike": float(k),
                    "lastPrice": round(theo, 2),
                    "bid": bid,
                    "ask": ask,
                    "impliedVolatility": round(iv_dec, 4),
                    "volume": int(120 + abs(atm_strike - k) * 5),
                    "openInterest": int(850 + abs(atm_strike - k) * 20),
                }
            )
    df = pd.DataFrame(rows)
    return {
        "symbol": symbol.upper(),
        "nearest_expiry": expiry_date,
        "dte": dte,
        "atm_iv": round(base_iv_pct, 2),
        "chain_preview": df,
        "error": None,
        "source": "fallback",
    }


def fetch_option_chain_summary(
    symbol: str,
    spot: float | None = None,
    target_dte: int = 30,
) -> dict[str, Any]:
    """Fetch nearest option expiration, ATM implied volatility, and ATM chain slice."""
    sym = str(symbol).strip().upper()
    quote = fetch_quote(sym)
    spot_val = float(spot if spot is not None and spot > 0 else quote["price"])
    prof = _fallback_profile(sym)

    try:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        expirations = list(getattr(ticker, "options", []) or [])
        if expirations:
            today = datetime.now(timezone.utc).date()
            best_exp = expirations[0]
            best_dte = max((datetime.strptime(best_exp, "%Y-%m-%d").date() - today).days, 1)
            for exp_str in expirations:
                d_days = max((datetime.strptime(exp_str, "%Y-%m-%d").date() - today).days, 1)
                if d_days >= 7:
                    best_exp = exp_str
                    best_dte = d_days
                    break

            chain = ticker.option_chain(best_exp)
            calls = chain.calls.copy() if chain and chain.calls is not None else pd.DataFrame()
            puts = chain.puts.copy() if chain and chain.puts is not None else pd.DataFrame()
            if not calls.empty:
                calls["type"] = "Call"
                calls["dist"] = (calls["strike"] - spot_val).abs()
                atm_calls = calls.nsmallest(5, "dist").sort_values("strike")
                atm_iv_raw = float(atm_calls["impliedVolatility"].dropna().median() or 0.20)
                atm_iv_pct = round(atm_iv_raw * 100.0 if atm_iv_raw <= 3.0 else atm_iv_raw, 2)

                preview_frames = [atm_calls]
                if not puts.empty:
                    puts["type"] = "Put"
                    puts["dist"] = (puts["strike"] - spot_val).abs()
                    preview_frames.append(puts.nsmallest(5, "dist").sort_values("strike"))
                combined = pd.concat(preview_frames, ignore_index=True)
                keep_cols = [
                    c
                    for c in ["type", "strike", "lastPrice", "bid", "ask", "impliedVolatility", "volume", "openInterest"]
                    if c in combined.columns
                ]
                return {
                    "symbol": sym,
                    "nearest_expiry": best_exp,
                    "dte": int(best_dte),
                    "atm_iv": float(np.clip(atm_iv_pct, 1.0, 300.0)),
                    "chain_preview": combined[keep_cols].reset_index(drop=True),
                    "error": None,
                    "source": "yfinance",
                }
    except Exception as exc:
        logger.debug("yfinance fallback triggered: %s", exc)

    return _build_synthetic_chain_summary(
        symbol=sym,
        spot=spot_val,
        base_iv_pct=float(prof["iv"]),
        target_dte=target_dte,
        dividend_yield=float(quote.get("dividend_yield", 0.0)),
    )


def load_market_defaults(symbol: str) -> dict[str, Any]:
    """Load a complete set of option model defaults for a ticker symbol."""
    sym = str(symbol).strip().upper()
    quote = fetch_quote(sym)
    rate = fetch_risk_free_rate(default=0.045)
    chain_info = fetch_option_chain_summary(sym, spot=quote["price"])
    return {
        "symbol": sym,
        "name": quote["name"],
        "spot": float(quote["price"]),
        "prev_close": float(quote["prev_close"]),
        "change_pct": float(quote["change_pct"]),
        "iv": float(chain_info["atm_iv"]),
        "dte": int(chain_info["dte"]),
        "rate": float(rate),
        "dividend_yield": float(quote["dividend_yield"]),
        "as_of": quote["as_of"],
        "chain": chain_info,
        "source": quote["source"],
    }


def fetch_price_history(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical OHLCV bars via yfinance with deterministic synthetic fallback."""
    sym = str(symbol).strip().upper() or "AAPL"
    try:
        import yfinance as yf

        hist = yf.Ticker(sym).history(period=period, interval=interval)
        if hist is not None and not hist.empty:
            df = hist.reset_index()
            if "Datetime" in df.columns and "Date" not in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            return df
    except Exception as exc:
        logger.debug("yfinance fallback triggered: %s", exc)

    prof = _fallback_profile(sym)
    period_map = {"1d": 2, "5d": 5, "1mo": 22, "3mo": 65, "6mo": 126, "1y": 252, "2y": 504}
    n_bars = period_map.get(period, 126)
    seed = sum(ord(ch) * (i + 1) for i, ch in enumerate(sym)) % 10_000
    rng = np.random.default_rng(seed)

    daily_vol = (float(prof["iv"]) / 100.0) / np.sqrt(252.0)
    returns = rng.normal(0.0003, daily_vol, size=n_bars)
    closes = float(prof["price"]) * np.exp(np.cumsum(returns) - np.sum(returns))
    opens = np.roll(closes, 1)
    opens[0] = closes[0] * (1.0 - returns[0])
    spread = np.abs(rng.normal(0.0, daily_vol * 0.75, size=n_bars)) * closes
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread
    volumes = rng.integers(15_000_000, 65_000_000, size=n_bars)
    end_dt = datetime.now(timezone.utc).date()
    dates = pd.bdate_range(end=end_dt, periods=n_bars)

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": np.round(opens, 2),
            "High": np.round(highs, 2),
            "Low": np.round(lows, 2),
            "Close": np.round(closes, 2),
            "Volume": volumes,
        }
    )


def fetch_intraday_bars(
    symbol: str,
    interval: str = "5m",
    period: str = "5d",
) -> pd.DataFrame:
    """Fetch intraday OHLCV bars (1m / 5m) with synthetic intraday fallback."""
    sym = str(symbol).strip().upper() or "AAPL"
    try:
        import yfinance as yf

        hist = yf.Ticker(sym).history(period=period, interval=interval)
        if hist is not None and not hist.empty:
            df = hist.reset_index()
            if "Datetime" in df.columns and "Date" not in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            return df
    except Exception as exc:
        logger.debug("yfinance fallback triggered: %s", exc)

    prof = _fallback_profile(sym)
    n_bars = 120
    step_min = 1 if "1m" in interval else 5
    seed = (sum(ord(ch) for ch in sym) + step_min) % 10_000
    rng = np.random.default_rng(seed)

    bar_vol = (float(prof["iv"]) / 100.0) / np.sqrt(252.0 * 78.0)
    rets = rng.normal(0.0, bar_vol, size=n_bars)
    closes = float(prof["price"]) * np.exp(np.cumsum(rets) - np.sum(rets))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    wiggle = np.abs(rng.normal(0.0, bar_vol * 0.65, size=n_bars)) * closes
    highs = np.maximum(opens, closes) + wiggle
    lows = np.minimum(opens, closes) - wiggle
    volumes = rng.integers(80_000, 450_000, size=n_bars)
    end_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    dates = [end_ts - timedelta(minutes=step_min * (n_bars - 1 - i)) for i in range(n_bars)]

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": np.round(opens, 2),
            "High": np.round(highs, 2),
            "Low": np.round(lows, 2),
            "Close": np.round(closes, 2),
            "Volume": volumes,
        }
    )
