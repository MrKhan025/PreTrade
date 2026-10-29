"""
Variance-reduced Monte Carlo option pricing engine using antithetic variates,
terminal spot control variates, and pathwise Greek estimators.
"""

from __future__ import annotations

import numpy as np

from black_scholes import _validate_inputs, bs_price_dividend, normalize_dte


def mc_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    n_paths: int = 25_000,
    seed: int | None = 42,
    use_control_variate: bool = True,
) -> dict[str, float]:
    """Price a European option via Monte Carlo with antithetic + control variates.

    Returns a dictionary containing `price`, `stderr`, `ci_lower`, `ci_upper`,
    `bsm_benchmark`, pathwise `delta`, `vega`, `vega_pct`, and `n_paths`.
    """
    S_val, K_val, sigma_val, r_val, q_val, opt = _validate_inputs(
        S, K, sigma, r, dividend_yield, option_type
    )
    paths = max(int(n_paths), 100)
    half_paths = max(paths // 2, 50)
    effective_paths = half_paths * 2

    time = normalize_dte(T)
    sqrt_t = float(np.sqrt(time))
    disc = float(np.exp(-r_val * time))
    drift = (r_val - q_val - 0.5 * sigma_val**2) * time
    diffusion = sigma_val * sqrt_t

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(half_paths)

    st_pos = S_val * np.exp(drift + diffusion * z)
    st_neg = S_val * np.exp(drift - diffusion * z)

    if opt == "call":
        payoff_pos = np.maximum(st_pos - K_val, 0.0)
        payoff_neg = np.maximum(st_neg - K_val, 0.0)
        ind_pos = (st_pos > K_val).astype(float)
        ind_neg = (st_neg > K_val).astype(float)
    else:
        payoff_pos = np.maximum(K_val - st_pos, 0.0)
        payoff_neg = np.maximum(K_val - st_neg, 0.0)
        ind_pos = -(st_pos < K_val).astype(float)
        ind_neg = -(st_neg < K_val).astype(float)

    disc_payoff_pair = 0.5 * disc * (payoff_pos + payoff_neg)

    if use_control_variate:
        cv_target = S_val * float(np.exp(-q_val * time))
        cv_pair = 0.5 * disc * (st_pos + st_neg)
        cov_matrix = np.cov(disc_payoff_pair, cv_pair, ddof=1)
        var_cv = float(cov_matrix[1, 1])
        if var_cv > 1e-12:
            beta = float(cov_matrix[0, 1] / var_cv)
            adjusted_samples = disc_payoff_pair - beta * (cv_pair - cv_target)
        else:
            adjusted_samples = disc_payoff_pair
    else:
        adjusted_samples = disc_payoff_pair

    price_est = float(max(np.mean(adjusted_samples), 0.0))
    stderr = float(np.std(adjusted_samples, ddof=1) / np.sqrt(half_paths))
    ci_half = 1.96 * stderr

    pw_delta_pos = disc * ind_pos * (st_pos / S_val)
    pw_delta_neg = disc * ind_neg * (st_neg / S_val)
    pw_delta = float(0.5 * np.mean(pw_delta_pos + pw_delta_neg))

    pw_vega_pos = disc * ind_pos * st_pos * (-sigma_val * time + sqrt_t * z)
    pw_vega_neg = disc * ind_neg * st_neg * (-sigma_val * time - sqrt_t * z)
    pw_vega = float(0.5 * np.mean(pw_vega_pos + pw_vega_neg))

    bsm_ref = bs_price_dividend(
        S_val, K_val, T, sigma_val, r_val, dividend_yield=q_val, option_type=opt, american_adjustment=False
    )

    return {
        "price": price_est,
        "stderr": stderr,
        "ci_lower": float(max(price_est - ci_half, 0.0)),
        "ci_upper": float(price_est + ci_half),
        "bsm_benchmark": float(bsm_ref),
        "delta": pw_delta,
        "vega": pw_vega,
        "vega_pct": float(pw_vega / 100.0),
        "n_paths": int(effective_paths),
    }
