from __future__ import annotations

import numpy as np
from scipy.stats import norm


def normalize_dte(days_to_expiry: float) -> float:
    days = float(days_to_expiry)
    if not np.isfinite(days):
        raise ValueError("Days to expiry must be finite.")
    if days <= 0:
        return 1e-9
    return max(days / 365.0, 1e-9)


def bs_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes price for a European option."""
    S = float(S)
    K = float(K)
    sigma = float(sigma)
    r = float(r)
    option_type = option_type.lower()

    if S <= 0 or K <= 0:
        raise ValueError("Spot and strike must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility must be positive.")

    time = normalize_dte(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * time) / (sigma * np.sqrt(time))
    d2 = d1 - sigma * np.sqrt(time)

    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * time) * norm.cdf(d2))
    if option_type == "put":
        return float(K * np.exp(-r * time) * norm.cdf(-d2) - S * norm.cdf(-d1))

    raise ValueError("option_type must be 'call' or 'put'.")


def bs_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    option_type: str = "call",
) -> dict:
    """Exact Black-Scholes Greeks for vanilla options."""
    S = float(S)
    K = float(K)
    sigma = float(sigma)
    r = float(r)
    option_type = option_type.lower()

    if S <= 0 or K <= 0:
        raise ValueError("Spot and strike must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility must be positive.")

    time = normalize_dte(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * time) / (sigma * np.sqrt(time))
    d2 = d1 - sigma * np.sqrt(time)
    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = norm.cdf(d1)
        gamma = pdf_d1 / (S * sigma * np.sqrt(time))
        theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(time)) - r * K * np.exp(-r * time) * norm.cdf(d2)
        theta_daily = theta / 365.0
        vega = S * pdf_d1 * np.sqrt(time)
        price = bs_price(S, K, T, sigma, r, option_type="call")
    elif option_type == "put":
        delta = norm.cdf(d1) - 1.0
        gamma = pdf_d1 / (S * sigma * np.sqrt(time))
        theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(time)) + r * K * np.exp(-r * time) * norm.cdf(-d2)
        theta_daily = theta / 365.0
        vega = S * pdf_d1 * np.sqrt(time)
        price = bs_price(S, K, T, sigma, r, option_type="put")
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "theta_daily": float(theta_daily),
        "vega": float(vega),
    }


def simulate_theta_decay(
    S: float,
    K: float,
    sigma: float,
    r: float,
    option_type: str = "call",
    dte_min: int = 1,
    dte_max: int = 365,
    steps: int = 80,
):
    dte_values = np.linspace(dte_min, dte_max, steps)
    output = []
    for dte in dte_values:
        metrics = bs_greeks(S, K, dte, sigma, r, option_type=option_type)
        output.append({"dte": float(dte), "price": metrics["price"], "delta": metrics["delta"]})
    return output
