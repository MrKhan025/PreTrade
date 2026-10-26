from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

DAYS_PER_YEAR: float = 365.0
MIN_TIME_YEARS: float = 1e-9


def normalize_dte(days_to_expiry: float) -> float:
    """Convert days-to-expiry (DTE) into annualized time (years) with a safe positive floor."""
    days = float(days_to_expiry)
    if not np.isfinite(days):
        raise ValueError("Days to expiry must be finite.")
    if days <= 0:
        return MIN_TIME_YEARS
    return max(days / DAYS_PER_YEAR, MIN_TIME_YEARS)


def _validate_inputs(
    S: float,
    K: float,
    sigma: float,
    r: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
) -> tuple[float, float, float, float, float, str]:
    """Validate and normalize core option parameters."""
    S_val = float(S)
    K_val = float(K)
    sigma_val = float(sigma)
    r_val = float(r)
    q_val = float(dividend_yield)
    opt = str(option_type).strip().lower()

    if not (np.isfinite(S_val) and np.isfinite(K_val) and np.isfinite(sigma_val) and np.isfinite(r_val) and np.isfinite(q_val)):
        raise ValueError("All numeric inputs (S, K, sigma, r, dividend_yield) must be finite.")
    if S_val <= 0 or K_val <= 0:
        raise ValueError("Spot and strike must be positive.")
    if sigma_val <= 0:
        raise ValueError("Volatility must be positive.")
    if q_val < 0:
        raise ValueError("Dividend yield must be non-negative.")
    if opt not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    return S_val, K_val, sigma_val, r_val, q_val, opt


def _d1_d2(
    S: float,
    K: float,
    time: float,
    sigma: float,
    r: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Compute Black-Scholes-Merton d1 and d2 terms."""
    sqrt_t = np.sqrt(time)
    vol_sqrt_t = sigma * sqrt_t
    d1 = (np.log(S / K) + (r - dividend_yield + 0.5 * sigma**2) * time) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(d1), float(d2)


def bs_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    option_type: str = "call",
) -> float:
    """Exact Black-Scholes price for a European non-dividend-paying option."""
    return bs_price_dividend(
        S=S,
        K=K,
        T=T,
        sigma=sigma,
        r=r,
        dividend_yield=0.0,
        option_type=option_type,
        american_adjustment=False,
    )


def bs_price_dividend(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    american_adjustment: bool = False,
) -> float:
    """Black-Scholes-Merton price with continuous dividend yield and optional American early-exercise approximation."""
    S_val, K_val, sigma_val, r_val, q_val, opt = _validate_inputs(S, K, sigma, r, dividend_yield, option_type)
    time = normalize_dte(T)
    d1, d2 = _d1_d2(S_val, K_val, time, sigma_val, r_val, q_val)

    disc_q = np.exp(-q_val * time)
    disc_r = np.exp(-r_val * time)

    if opt == "call":
        euro_price = float(S_val * disc_q * norm.cdf(d1) - K_val * disc_r * norm.cdf(d2))
        intrinsic = max(S_val - K_val, 0.0)
    else:
        euro_price = float(K_val * disc_r * norm.cdf(-d2) - S_val * disc_q * norm.cdf(-d1))
        intrinsic = max(K_val - S_val, 0.0)

    if not american_adjustment:
        return max(euro_price, 0.0)

    # American early-exercise adjustment:
    # - For a call with zero dividend yield (q == 0), early exercise is never optimal when r >= 0;
    #   nonetheless, if a user enables early-exercise tilt or when q > 0 / put with r > 0,
    #   compute a smooth, arbitrage-respecting early-exercise premium that enforces V >= intrinsic.
    if opt == "call":
        carry_driver = max(q_val, 0.0) if q_val > 0 else max(r_val, 0.0) * 0.35
        moneyness_weight = float(norm.cdf(d1))
        ee_factor = 0.05 * (1.0 - np.exp(-carry_driver * time)) * (0.5 + 0.5 * moneyness_weight)
    else:
        rate_driver = max(r_val - q_val, max(r_val, 0.0))
        moneyness_weight = float(norm.cdf(-d1))
        ee_factor = 0.05 * (1.0 - np.exp(-rate_driver * time)) * (0.5 + 0.5 * moneyness_weight)

    adjusted_price = euro_price * (1.0 + ee_factor)
    return float(max(adjusted_price, intrinsic, euro_price))


def bs_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    option_type: str = "call",
) -> dict[str, float]:
    """Exact Black-Scholes first- and second-order Greeks for vanilla European options."""
    return bs_greeks_dividend(
        S=S,
        K=K,
        T=T,
        sigma=sigma,
        r=r,
        dividend_yield=0.0,
        option_type=option_type,
        american_adjustment=False,
    )


def bs_greeks_dividend(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    american_adjustment: bool = False,
) -> dict[str, float]:
    """Exact Black-Scholes-Merton Greeks with continuous dividend yield and second-order sensitivities.

    Returns both raw annual/unit quantities (for backward compatibility) and trader-standard
    scaled sensitivities (`theta_daily`, `vega_pct` per 1% IV change, `rho_pct` per 1% rate change),
    plus second-order Greeks (`vanna`, `volga`, `charm`, `speed`) and value decomposition.
    """
    S_val, K_val, sigma_val, r_val, q_val, opt = _validate_inputs(S, K, sigma, r, dividend_yield, option_type)
    time = normalize_dte(T)
    sqrt_t = np.sqrt(time)
    d1, d2 = _d1_d2(S_val, K_val, time, sigma_val, r_val, q_val)

    pdf_d1 = float(norm.pdf(d1))
    cdf_d1 = float(norm.cdf(d1))
    cdf_neg_d1 = float(norm.cdf(-d1))
    cdf_d2 = float(norm.cdf(d2))
    cdf_neg_d2 = float(norm.cdf(-d2))

    disc_q = float(np.exp(-q_val * time))
    disc_r = float(np.exp(-r_val * time))

    # Common across Call and Put under BSM
    gamma = (disc_q * pdf_d1) / (S_val * sigma_val * sqrt_t)
    vega = S_val * disc_q * pdf_d1 * sqrt_t  # per 1.00 (100%) vol move
    vega_pct = vega / 100.0  # per 1% vol point move

    # Second-order Greeks
    vanna = -disc_q * pdf_d1 * (d2 / sigma_val)  # d(Delta)/d(sigma)
    volga = vega * (d1 * d2 / sigma_val)  # d(Vega)/d(sigma)
    speed = -(gamma / S_val) * (d1 / (sigma_val * sqrt_t) + 1.0)  # d(Gamma)/d(S)

    charm_common = disc_q * pdf_d1 * (
        (2.0 * (r_val - q_val) * time - d2 * sigma_val * sqrt_t) / (2.0 * time * sigma_val * sqrt_t)
    )

    if opt == "call":
        price = float(S_val * disc_q * cdf_d1 - K_val * disc_r * cdf_d2)
        delta = disc_q * cdf_d1
        theta = (
            -(S_val * disc_q * pdf_d1 * sigma_val) / (2.0 * sqrt_t)
            - r_val * K_val * disc_r * cdf_d2
            + q_val * S_val * disc_q * cdf_d1
        )
        rho = K_val * time * disc_r * cdf_d2
        charm = -q_val * disc_q * cdf_d1 + charm_common
        prob_itm = cdf_d2
        intrinsic = max(S_val - K_val, 0.0)
    else:
        price = float(K_val * disc_r * cdf_neg_d2 - S_val * disc_q * cdf_neg_d1)
        delta = -disc_q * cdf_neg_d1
        theta = (
            -(S_val * disc_q * pdf_d1 * sigma_val) / (2.0 * sqrt_t)
            + r_val * K_val * disc_r * cdf_neg_d2
            - q_val * S_val * disc_q * cdf_neg_d1
        )
        rho = -K_val * time * disc_r * cdf_neg_d2
        charm = q_val * disc_q * cdf_neg_d1 + charm_common
        prob_itm = cdf_neg_d2
        intrinsic = max(K_val - S_val, 0.0)

    theta_daily = theta / DAYS_PER_YEAR
    rho_pct = rho / 100.0
    charm_daily = charm / DAYS_PER_YEAR

    if american_adjustment:
        price = bs_price_dividend(
            S_val, K_val, T, sigma_val, r_val, dividend_yield=q_val, option_type=opt, american_adjustment=True
        )
        bias = 0.05 * (1.0 - np.exp(-max(r_val, q_val, 0.0) * time))
        if opt == "call":
            delta = float(np.clip(delta * (1.0 + 0.5 * bias), 0.0, 1.0))
        else:
            delta = float(np.clip(delta * (1.0 + 0.5 * bias), -1.0, 0.0))
        gamma = gamma * (1.0 + 0.35 * bias)
        theta = theta * (1.0 + 0.6 * bias)
        theta_daily = theta / DAYS_PER_YEAR
        vega = vega * (1.0 + 0.3 * bias)
        vega_pct = vega / 100.0

    extrinsic = max(price - intrinsic, 0.0)
    leverage = (delta * S_val / price) if price > 1e-6 else 0.0
    prob_touch = float(np.clip(2.0 * prob_itm, 0.0, 1.0))

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "theta_daily": float(theta_daily),
        "vega": float(vega),
        "vega_pct": float(vega_pct),
        "rho": float(rho),
        "rho_pct": float(rho_pct),
        "vanna": float(vanna),
        "volga": float(volga),
        "charm": float(charm),
        "charm_daily": float(charm_daily),
        "speed": float(speed),
        "prob_itm": float(prob_itm),
        "prob_touch": float(prob_touch),
        "intrinsic": float(intrinsic),
        "extrinsic": float(extrinsic),
        "leverage": float(leverage),
        "d1": float(d1),
        "d2": float(d2),
    }


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    american_adjustment: bool = False,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> float:
    """Solve for implied volatility (annualized decimal, e.g. 0.20 for 20%) given a market option price."""
    target = float(market_price)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Market price must be finite and positive.")

    S_val, K_val, _, r_val, q_val, opt = _validate_inputs(S, K, 0.20, r, dividend_yield, option_type)
    time = normalize_dte(T)
    disc_q = np.exp(-q_val * time)
    disc_r = np.exp(-r_val * time)

    if opt == "call":
        lower_bound = max(S_val * disc_q - K_val * disc_r, 0.0)
        upper_bound = S_val * disc_q
    else:
        lower_bound = max(K_val * disc_r - S_val * disc_q, 0.0)
        upper_bound = K_val * disc_r

    if american_adjustment:
        lower_bound = max(lower_bound, S_val - K_val if opt == "call" else K_val - S_val, 0.0)
        upper_bound *= 1.10

    if target < lower_bound - 1e-6:
        raise ValueError(f"Market price (${target:.4f}) is below the no-arbitrage lower bound (${lower_bound:.4f}).")
    if target >= upper_bound:
        raise ValueError(f"Market price (${target:.4f}) exceeds the upper bound (${upper_bound:.4f}).")

    def objective(vol: float) -> float:
        return (
            bs_price_dividend(
                S_val,
                K_val,
                T,
                vol,
                r_val,
                dividend_yield=q_val,
                option_type=opt,
                american_adjustment=american_adjustment,
            )
            - target
        )

    vol_low = 1e-4
    vol_high = 5.0
    f_low = objective(vol_low)
    f_high = objective(vol_high)

    if abs(f_low) < tol:
        return float(vol_low)

    # Expand upper bracket if needed for extreme deep-volatility options
    while f_high < 0 and vol_high < 20.0:
        vol_high *= 2.0
        f_high = objective(vol_high)

    if f_low * f_high > 0:
        raise ValueError("Could not bracket implied volatility for the given market price and parameters.")

    iv_est = brentq(objective, vol_low, vol_high, xtol=tol, maxiter=max_iter)
    return float(iv_est)


def put_call_parity_check(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    dividend_yield: float = 0.0,
) -> dict[str, float | bool]:
    """Evaluate European Put-Call Parity: C - P = S * exp(-qT) - K * exp(-rT)."""
    time = normalize_dte(T)
    lhs = float(call_price) - float(put_price)
    rhs = float(S) * np.exp(-float(dividend_yield) * time) - float(K) * np.exp(-float(r) * time)
    discrepancy = lhs - rhs
    synthetic_call = float(put_price) + rhs
    synthetic_put = float(call_price) - rhs
    return {
        "call_minus_put": float(lhs),
        "forward_discount_spread": float(rhs),
        "discrepancy": float(discrepancy),
        "abs_discrepancy": float(abs(discrepancy)),
        "synthetic_call": float(synthetic_call),
        "synthetic_put": float(synthetic_put),
        "parity_holds": bool(abs(discrepancy) < 1e-2),
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
    dividend_yield: float = 0.0,
    american_adjustment: bool = False,
) -> list[dict[str, float]]:
    """Simulate option price, delta, theta/day, and extrinsic value across a DTE range."""
    if dte_min <= 0 or dte_max < dte_min:
        raise ValueError("Invalid DTE bounds: require 0 < dte_min <= dte_max.")
    dte_values = np.linspace(dte_min, dte_max, int(steps))
    output: list[dict[str, float]] = []
    for dte in dte_values:
        metrics = bs_greeks_dividend(
            S=S,
            K=K,
            T=float(dte),
            sigma=sigma,
            r=r,
            dividend_yield=dividend_yield,
            option_type=option_type,
            american_adjustment=american_adjustment,
        )
        output.append(
            {
                "dte": float(dte),
                "price": metrics["price"],
                "delta": metrics["delta"],
                "gamma": metrics["gamma"],
                "theta_daily": metrics["theta_daily"],
                "vega_pct": metrics["vega_pct"],
                "extrinsic": metrics["extrinsic"],
            }
        )
    return output


def _round_strike(value: float, step: float = 1.0) -> float:
    """Round generated preset strikes cleanly."""
    return float(max(1.0, round(float(value) / step) * step))


def strategy_preset(
    name: str,
    spot: float,
    strike: float,
    dte: float = 30.0,
    sigma: float = 0.20,
    rate: float = 0.045,
    dividend_yield: float = 0.0,
    american_adjustment: bool = False,
) -> list[dict[str, Any]]:
    """Build multi-leg option strategy presets with fair Black-Scholes-Merton leg premiums."""
    base_strike = float(strike) if strike > 0 else float(spot)
    step = 2.5 if base_strike >= 50 else 1.0
    w1 = max(step * 2, _round_strike(base_strike * 0.05, step))
    w2 = max(step * 4, _round_strike(base_strike * 0.10, step))
    base_dte = max(int(round(float(dte))), 1)

    k_atm = _round_strike(base_strike, step)
    k_down1 = max(1.0, k_atm - w1)
    k_down2 = max(0.5, k_atm - w2)
    k_up1 = k_atm + w1
    k_up2 = k_atm + w2

    raw_templates: dict[str, list[dict[str, Any]]] = {
        "Long Call": [
            {"side": "Buy", "kind": "Call", "strike": k_atm, "quantity": 1, "dte": base_dte},
        ],
        "Short Call": [
            {"side": "Sell", "kind": "Call", "strike": k_atm, "quantity": 1, "dte": base_dte},
        ],
        "Long Put": [
            {"side": "Buy", "kind": "Put", "strike": k_atm, "quantity": 1, "dte": base_dte},
        ],
        "Short Put": [
            {"side": "Sell", "kind": "Put", "strike": k_atm, "quantity": 1, "dte": base_dte},
        ],
        "Bull Call Spread": [
            {"side": "Buy", "kind": "Call", "strike": k_down1, "quantity": 1, "dte": base_dte},
            {"side": "Sell", "kind": "Call", "strike": k_up1, "quantity": 1, "dte": base_dte},
        ],
        "Bear Put Spread": [
            {"side": "Buy", "kind": "Put", "strike": k_up1, "quantity": 1, "dte": base_dte},
            {"side": "Sell", "kind": "Put", "strike": k_down1, "quantity": 1, "dte": base_dte},
        ],
        "Long Butterfly": [
            {"side": "Buy", "kind": "Call", "strike": k_down1, "quantity": 1, "dte": base_dte},
            {"side": "Sell", "kind": "Call", "strike": k_atm, "quantity": 2, "dte": base_dte},
            {"side": "Buy", "kind": "Call", "strike": k_up1, "quantity": 1, "dte": base_dte},
        ],
        "Iron Condor": [
            {"side": "Sell", "kind": "Put", "strike": k_down1, "quantity": 1, "dte": base_dte},
            {"side": "Buy", "kind": "Put", "strike": k_down2, "quantity": 1, "dte": base_dte},
            {"side": "Sell", "kind": "Call", "strike": k_up1, "quantity": 1, "dte": base_dte},
            {"side": "Buy", "kind": "Call", "strike": k_up2, "quantity": 1, "dte": base_dte},
        ],
        "Iron Butterfly": [
            {"side": "Buy", "kind": "Put", "strike": k_down1, "quantity": 1, "dte": base_dte},
            {"side": "Sell", "kind": "Put", "strike": k_atm, "quantity": 1, "dte": base_dte},
            {"side": "Sell", "kind": "Call", "strike": k_atm, "quantity": 1, "dte": base_dte},
            {"side": "Buy", "kind": "Call", "strike": k_up1, "quantity": 1, "dte": base_dte},
        ],
        "Long Straddle": [
            {"side": "Buy", "kind": "Call", "strike": k_atm, "quantity": 1, "dte": base_dte},
            {"side": "Buy", "kind": "Put", "strike": k_atm, "quantity": 1, "dte": base_dte},
        ],
        "Long Strangle": [
            {"side": "Buy", "kind": "Put", "strike": k_down1, "quantity": 1, "dte": base_dte},
            {"side": "Buy", "kind": "Call", "strike": k_up1, "quantity": 1, "dte": base_dte},
        ],
        "Calendar Spread": [
            {"side": "Buy", "kind": "Call", "strike": k_atm, "quantity": 1, "dte": base_dte * 2},
            {"side": "Sell", "kind": "Call", "strike": k_atm, "quantity": 1, "dte": base_dte},
        ],
    }

    template = raw_templates.get(name, raw_templates["Long Call"])
    priced_legs: list[dict[str, Any]] = []
    for leg in template:
        leg_dte = float(leg["dte"])
        fair_prem = bs_price_dividend(
            S=spot,
            K=leg["strike"],
            T=leg_dte,
            sigma=sigma,
            r=rate,
            dividend_yield=dividend_yield,
            option_type=leg["kind"].lower(),
            american_adjustment=american_adjustment,
        )
        priced_legs.append(
            {
                "side": leg["side"],
                "kind": leg["kind"],
                "strike": float(leg["strike"]),
                "premium": round(max(fair_prem, 0.01), 2),
                "quantity": int(leg["quantity"]),
                "dte": int(leg["dte"]),
            }
        )
    return priced_legs


def strategy_expiry_curve(
    spot_range: np.ndarray,
    legs: list[dict[str, Any]],
    sigma: float = 0.20,
    rate: float = 0.045,
    dividend_yield: float = 0.0,
    american_adjustment: bool = False,
) -> np.ndarray:
    """Compute strategy P/L at the nearest leg expiration (properly handling multi-DTE calendar spreads)."""
    spot_arr = np.asarray(spot_range, dtype=float)
    curve = np.zeros_like(spot_arr, dtype=float)
    if not legs:
        return curve

    min_dte = min(float(leg.get("dte", 30.0)) for leg in legs)

    for leg in legs:
        quantity = float(leg.get("quantity", 1))
        premium = float(leg.get("premium", 0.0))
        strike = float(leg.get("strike", 0.0))
        direction = 1.0 if str(leg.get("side", "Buy")).capitalize() == "Buy" else -1.0
        kind = str(leg.get("kind", "Call")).capitalize()
        leg_dte = float(leg.get("dte", min_dte))
        remaining_dte = max(leg_dte - min_dte, 0.0)

        if remaining_dte <= 1e-6:
            leg_value = (
                np.maximum(spot_arr - strike, 0.0)
                if kind == "Call"
                else np.maximum(strike - spot_arr, 0.0)
            )
        else:
            leg_value = np.array(
                [
                    bs_price_dividend(
                        S=float(s),
                        K=strike,
                        T=remaining_dte,
                        sigma=sigma,
                        r=rate,
                        dividend_yield=dividend_yield,
                        option_type=kind.lower(),
                        american_adjustment=american_adjustment,
                    )
                    for s in spot_arr
                ],
                dtype=float,
            )
        curve += direction * quantity * (leg_value - premium)
    return curve


def strategy_current_curve(
    spot_range: np.ndarray,
    legs: list[dict[str, Any]],
    dte: float,
    sigma: float,
    rate: float,
    dividend_yield: float = 0.0,
    american_adjustment: bool = True,
) -> np.ndarray:
    """Compute current (T+0 or intermediate DTE) strategy P/L across a spot grid."""
    spot_arr = np.asarray(spot_range, dtype=float)
    curve = np.zeros_like(spot_arr, dtype=float)
    if not legs:
        return curve

    min_initial_dte = min(float(leg.get("dte", dte)) for leg in legs)
    elapsed_days = max(min_initial_dte - float(dte), 0.0)

    for leg in legs:
        quantity = float(leg.get("quantity", 1))
        premium = float(leg.get("premium", 0.0))
        strike = float(leg.get("strike", 0.0))
        direction = 1.0 if str(leg.get("side", "Buy")).capitalize() == "Buy" else -1.0
        kind = str(leg.get("kind", "Call")).capitalize()
        raw_leg_dte = float(leg.get("dte", dte))
        effective_dte = max(raw_leg_dte - elapsed_days, 1e-4)

        prices = np.array(
            [
                bs_price_dividend(
                    S=float(s),
                    K=strike,
                    T=effective_dte,
                    sigma=sigma,
                    r=rate,
                    dividend_yield=dividend_yield,
                    option_type=kind.lower(),
                    american_adjustment=american_adjustment,
                )
                for s in spot_arr
            ],
            dtype=float,
        )
        curve += direction * quantity * (prices - premium)
    return curve


def strategy_greeks(
    spot: float,
    legs: list[dict[str, Any]],
    dte: float,
    sigma: float,
    rate: float,
    dividend_yield: float = 0.0,
    american_adjustment: bool = True,
) -> dict[str, float]:
    """Compute aggregate net Greeks and net cost/credit across all legs of a strategy."""
    totals = {
        "net_premium": 0.0,
        "theoretical_value": 0.0,
        "delta": 0.0,
        "gamma": 0.0,
        "theta_daily": 0.0,
        "vega_pct": 0.0,
        "vega": 0.0,
        "rho_pct": 0.0,
        "vanna": 0.0,
        "volga": 0.0,
    }
    if not legs:
        return totals

    min_initial_dte = min(float(leg.get("dte", dte)) for leg in legs)
    elapsed_days = max(min_initial_dte - float(dte), 0.0)

    for leg in legs:
        quantity = float(leg.get("quantity", 1))
        premium = float(leg.get("premium", 0.0))
        strike = float(leg.get("strike", 0.0))
        direction = 1.0 if str(leg.get("side", "Buy")).capitalize() == "Buy" else -1.0
        kind = str(leg.get("kind", "Call")).lower()
        raw_leg_dte = float(leg.get("dte", dte))
        effective_dte = max(raw_leg_dte - elapsed_days, 1e-4)

        g = bs_greeks_dividend(
            S=spot,
            K=strike,
            T=effective_dte,
            sigma=sigma,
            r=rate,
            dividend_yield=dividend_yield,
            option_type=kind,
            american_adjustment=american_adjustment,
        )
        weight = direction * quantity
        totals["net_premium"] += weight * premium
        totals["theoretical_value"] += weight * g["price"]
        totals["delta"] += weight * g["delta"]
        totals["gamma"] += weight * g["gamma"]
        totals["theta_daily"] += weight * g["theta_daily"]
        totals["vega_pct"] += weight * g["vega_pct"]
        totals["vega"] += weight * g["vega"]
        totals["rho_pct"] += weight * g["rho_pct"]
        totals["vanna"] += weight * g["vanna"]
        totals["volga"] += weight * g["volga"]

    return {k: float(v) for k, v in totals.items()}


def strategy_summary_metrics(
    spot_range: np.ndarray,
    payoff_curve: np.ndarray,
    spot: float | None = None,
    sigma: float = 0.20,
    dte: float = 30.0,
    rate: float = 0.045,
    dividend_yield: float = 0.0,
) -> dict[str, Any]:
    """Compute max profit, max loss, all break-even points, risk/reward ratio, and lognormal Probability of Profit (PoP)."""
    spot_arr = np.asarray(spot_range, dtype=float)
    curve_arr = np.asarray(payoff_curve, dtype=float)

    max_profit = float(np.max(curve_arr))
    max_loss = float(np.min(curve_arr))

    breakevens: list[float] = []
    for i in range(1, len(spot_arr)):
        prev_y = float(curve_arr[i - 1])
        curr_y = float(curve_arr[i])
        if prev_y == 0.0:
            breakevens.append(float(spot_arr[i - 1]))
        elif (prev_y < 0.0 < curr_y) or (prev_y > 0.0 > curr_y):
            # Linear interpolation for sub-grid accuracy
            x0, x1 = float(spot_arr[i - 1]), float(spot_arr[i])
            fraction = abs(prev_y) / (abs(prev_y) + abs(curr_y))
            be = x0 + fraction * (x1 - x0)
            breakevens.append(round(float(be), 2))

    # Deduplicate close crossings
    deduped_be: list[float] = []
    for be in breakevens:
        if not deduped_be or abs(be - deduped_be[-1]) > 0.05:
            deduped_be.append(be)

    primary_breakeven = deduped_be[0] if deduped_be else None

    # Estimate Probability of Profit (PoP) under risk-neutral lognormal terminal distribution
    if spot is not None and spot > 0 and sigma > 0:
        time = normalize_dte(dte)
        mu = np.log(float(spot)) + (float(rate) - float(dividend_yield) - 0.5 * float(sigma) ** 2) * time
        std = float(sigma) * np.sqrt(time)
        z_scores = (np.log(np.maximum(spot_arr, 1e-6)) - mu) / std
        pdf_weights = norm.pdf(z_scores) / np.maximum(spot_arr * std, 1e-9)
        total_mass = float(np.trapezoid(pdf_weights, spot_arr))
        if total_mass > 1e-6:
            profitable_mask = (curve_arr > 0.0).astype(float)
            pop = float(np.clip(np.trapezoid(pdf_weights * profitable_mask, spot_arr) / total_mass, 0.0, 1.0)) * 100.0
        else:
            pop = float(np.mean(curve_arr > 0.0) * 100.0)
    else:
        pop = float(np.mean(curve_arr > 0.0) * 100.0)

    abs_loss = abs(min(max_loss, 0.0))
    risk_reward = (max_profit / abs_loss) if abs_loss > 1e-4 else float("inf")

    return {
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": primary_breakeven,
        "breakevens": deduped_be,
        "prob_profit": float(pop),
        "risk_reward": float(risk_reward),
    }


def strategy_score(metrics: dict[str, Any]) -> float:
    """Compute a normalized 0-100 risk-adjusted setup score from strategy metrics."""
    max_profit = max(float(metrics.get("max_profit", 0.0)), 0.0)
    max_loss = abs(min(float(metrics.get("max_loss", 0.0)), 0.0))
    pop = float(metrics.get("prob_profit", 50.0))

    if max_loss <= 1e-6:
        rr_component = 30.0 if max_profit > 0 else 0.0
    else:
        rr_ratio = min(max_profit / max_loss, 5.0)
        rr_component = rr_ratio * 12.0

    pop_component = 0.55 * pop
    edge_bonus = 10.0 if (max_profit > 0 and max_loss > 0 and (max_profit / max_loss) * (pop / 100.0) >= 0.45) else 0.0
    score = 15.0 + pop_component + rr_component + edge_bonus
    return float(np.clip(score, 0.0, 100.0))


def classify_strategy_bias(strategy_name: str) -> str:
    """Return accurate market bias classification for a given strategy name."""
    biases = {
        "Long Call": "Bullish / Uncapped Upside",
        "Bull Call Spread": "Moderately Bullish / Defined Risk",
        "Short Put": "Bullish / Income Generation",
        "Long Put": "Bearish / Downside Convexity",
        "Bear Put Spread": "Moderately Bearish / Defined Risk",
        "Short Call": "Bearish to Neutral / Premium Capture",
        "Long Butterfly": "Neutral / Pinning Target Strike",
        "Iron Condor": "Neutral / Range-Bound Income",
        "Iron Butterfly": "Neutral / High-Yield Range Pin",
        "Calendar Spread": "Neutral / Term-Structure & Theta Capture",
        "Long Straddle": "Bi-Directional / Volatility Expansion",
        "Long Strangle": "Bi-Directional / Tail-Move Breakout",
    }
    return biases.get(strategy_name, "Custom Multi-Leg / Tactical Exposure")


def trade_idea_summary(strategy_name: str, metrics: dict[str, Any], spot: float, strike: float) -> str:
    """Generate a natural-language executive summary of the strategy setup and risk profile."""
    max_profit = float(metrics.get("max_profit", 0.0))
    max_loss = float(metrics.get("max_loss", 0.0))
    breakevens = metrics.get("breakevens") or (
        [metrics["breakeven"]] if metrics.get("breakeven") is not None else []
    )
    pop = float(metrics.get("prob_profit", 50.0))
    bias = classify_strategy_bias(strategy_name)

    if not breakevens:
        breakeven_note = "No break-even crossing detected in the evaluated spot window."
    elif len(breakevens) == 1:
        breakeven_note = f"Break-even is at ${breakevens[0]:.2f} per share."
    else:
        joined = " and ".join(f"${b:.2f}" for b in breakevens)
        breakeven_note = f"Break-even points are at {joined} per share."

    return (
        f"{strategy_name} ({bias}) centered around spot=${spot:.2f} and reference strike=${strike:.2f}. "
        f"It projects a maximum profit of ${max_profit:.2f} per share, maximum loss of ${abs(max_loss):.2f} per share, "
        f"and an estimated probability of profit of {pop:.1f}%. {breakeven_note}"
    )