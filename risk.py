"""
Quantitative risk analytics for PreTrade:
- Lognormal / normal expected move over a DTE horizon
- Risk-neutral expected P/L, Probability of Profit (PoP), and reward/risk for multi-leg strategies
- Delta-Gamma-Vega portfolio Value-at-Risk (VaR) and Conditional VaR (CVaR)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from black_scholes import (
    DAYS_PER_YEAR,
    normalize_dte,
    strategy_expiry_curve,
    strategy_summary_metrics,
)

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def expected_move(
    spot: float,
    sigma: float,
    dte: float,
    confidence: float = 0.68,
) -> dict[str, float]:
    """Compute the option-implied expected move over `dte` days at a given confidence level."""
    S_val = float(spot)
    sigma_val = float(sigma)
    conf = float(np.clip(confidence, 0.01, 0.999))
    if S_val <= 0 or sigma_val <= 0:
        raise ValueError("Spot and volatility must be positive.")

    time = normalize_dte(dte)
    z_score = float(norm.ppf(0.5 + 0.5 * conf))
    move = S_val * sigma_val * float(np.sqrt(time)) * z_score
    lower = max(S_val - move, 0.0)
    upper = S_val + move
    pct_of_spot = (move / S_val) * 100.0

    return {
        "expected_move": float(move),
        "lower": float(lower),
        "upper": float(upper),
        "pct_of_spot": float(pct_of_spot),
        "confidence": conf,
        "z_score": z_score,
    }


def strategy_risk_metrics(
    spot: float,
    legs: list[dict[str, Any]],
    sigma: float,
    rate: float,
    dte: float,
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """Compute risk-neutral expected P/L, PoP, max profit/loss, and reward-to-risk for a strategy."""
    S_val = float(spot)
    sigma_val = float(sigma)
    r_val = float(rate)
    q_val = float(dividend_yield)
    if S_val <= 0 or sigma_val <= 0:
        raise ValueError("Spot and volatility must be positive.")

    min_dte = min((float(leg.get("dte", dte)) for leg in legs), default=float(dte))
    horizon_dte = max(min(min_dte, float(dte)), 1.0)
    time = normalize_dte(horizon_dte)
    std = sigma_val * float(np.sqrt(time))

    s_min = max(0.5, S_val * float(np.exp(-4.5 * std)))
    s_max = max(S_val * float(np.exp(4.5 * std)), S_val * 1.8)
    spot_grid = np.linspace(s_min, s_max, 600)

    pnl_curve = strategy_expiry_curve(
        spot_grid,
        legs,
        sigma=sigma_val,
        rate=r_val,
        dividend_yield=q_val,
        american_adjustment=False,
    )
    summary = strategy_summary_metrics(
        spot_grid,
        pnl_curve,
        spot=S_val,
        sigma=sigma_val,
        dte=horizon_dte,
        rate=r_val,
        dividend_yield=q_val,
    )

    mu = np.log(S_val) + (r_val - q_val - 0.5 * sigma_val**2) * time
    z_scores = (np.log(np.maximum(spot_grid, 1e-6)) - mu) / std
    pdf = norm.pdf(z_scores) / np.maximum(spot_grid * std, 1e-9)
    norm_mass = float(_trapz(pdf, spot_grid))

    if norm_mass > 1e-6:
        pdf_norm = pdf / norm_mass
        expected_pnl = float(_trapz(pnl_curve * pdf_norm, spot_grid))
        variance = float(max(_trapz(((pnl_curve - expected_pnl) ** 2) * pdf_norm, spot_grid), 0.0))
        pnl_std = float(np.sqrt(variance))
    else:
        expected_pnl = float(np.mean(pnl_curve))
        pnl_std = float(np.std(pnl_curve))

    max_profit = float(summary["max_profit"])
    max_loss = float(abs(summary["max_loss"]))
    reward_risk = (max_profit / max_loss) if max_loss > 1e-4 else float("inf")
    sharpe_proxy = (expected_pnl / pnl_std) if pnl_std > 1e-6 else 0.0

    return {
        "expected_pnl": expected_pnl,
        "pnl_std": pnl_std,
        "prob_profit": float(summary["prob_profit"]),
        "max_profit": max_profit,
        "max_loss": max_loss,
        "reward_risk": float(reward_risk),
        "sharpe_proxy": float(sharpe_proxy),
    }


def portfolio_var_delta_normal(
    net_delta: float,
    net_gamma: float,
    net_vega: float,
    spot: float,
    sigma: float,
    horizon_days: float = 1.0,
    confidence: float = 0.95,
    vol_of_vol: float = 0.03,
) -> dict[str, float]:
    """Compute parametric Delta-Gamma-Vega Value-at-Risk (VaR) over `horizon_days`."""
    S_val = float(spot)
    sigma_val = float(sigma)
    dt = max(float(horizon_days), 1e-3) / DAYS_PER_YEAR
    conf = float(np.clip(confidence, 0.50, 0.999))

    std_ds = S_val * sigma_val * float(np.sqrt(dt))
    std_div_pts = float(vol_of_vol) * 100.0 * float(np.sqrt(dt))

    expected_pnl = 0.5 * float(net_gamma) * (std_ds**2)
    var_delta = (float(net_delta) * std_ds) ** 2
    var_gamma = 0.5 * (float(net_gamma) ** 2) * (std_ds**4)
    var_vega = (float(net_vega) * std_div_pts) ** 2
    pnl_std = float(np.sqrt(max(var_delta + var_gamma + var_vega, 1e-12)))

    z_alpha = float(norm.ppf(conf))
    if pnl_std > 1e-6 and std_ds > 1e-6:
        third_moment = 3.0 * (float(net_delta) ** 2) * float(net_gamma) * (std_ds**4) + (
            float(net_gamma) ** 3
        ) * (std_ds**6)
        skew = float(np.clip(third_moment / (pnl_std**3), -1.5, 1.5))
    else:
        skew = 0.0

    z_cf = z_alpha - (skew / 6.0) * (z_alpha**2 - 1.0)
    quantile_loss = -(expected_pnl - z_cf * pnl_std)
    var_estimate = float(max(quantile_loss, 0.0))

    alpha_tail = 1.0 - conf
    es_multiplier = float(norm.pdf(z_alpha) / max(alpha_tail, 1e-6))
    cvar_estimate = float(max(-(expected_pnl - es_multiplier * pnl_std), var_estimate))

    return {
        "var": var_estimate,
        "cvar": cvar_estimate,
        "expected_pnl": float(expected_pnl),
        "pnl_std": pnl_std,
        "confidence": conf,
        "horizon_days": float(horizon_days),
    }


def position_sizing_recommendation(
    account_size: float,
    risk_pct: float,
    max_loss_per_share: float,
    max_profit_per_share: float,
    prob_profit: float,
    contract_multiplier: int = 100,
) -> dict[str, float | int]:
    """Compute position sizing (contracts, dollar risk, max reward, and fractional Kelly)
    given an account size, max risk % per trade, and strategy per-share metrics.
    """
    acct = max(float(account_size), 0.0)
    r_pct = float(np.clip(risk_pct, 0.05, 100.0))
    mult = max(int(contract_multiplier), 1)

    risk_budget_dollars = acct * (r_pct / 100.0)
    loss_per_share = abs(float(max_loss_per_share))
    profit_per_share = max(float(max_profit_per_share), 0.0)
    loss_per_contract = loss_per_share * mult
    profit_per_contract = profit_per_share * mult

    if loss_per_contract > 1e-6:
        raw_contracts = int(np.floor(risk_budget_dollars / loss_per_contract))
        contracts = max(raw_contracts, 0)
        b_odds = profit_per_share / loss_per_share
    else:
        contracts = 0
        b_odds = 0.0

    p_win = float(np.clip(prob_profit / 100.0, 0.0, 1.0))
    q_loss = 1.0 - p_win
    if b_odds > 1e-6:
        full_kelly = max((p_win * b_odds - q_loss) / b_odds, 0.0)
    else:
        full_kelly = 0.0
    half_kelly_pct = float(np.clip(0.5 * full_kelly * 100.0, 0.0, 25.0))

    total_risk = contracts * loss_per_contract
    total_max_profit = contracts * profit_per_contract

    return {
        "risk_budget_dollars": float(risk_budget_dollars),
        "loss_per_contract": float(loss_per_contract),
        "profit_per_contract": float(profit_per_contract),
        "recommended_contracts": int(contracts),
        "total_dollar_risk": float(total_risk),
        "total_dollar_max_profit": float(total_max_profit),
        "half_kelly_pct": half_kelly_pct,
    }
