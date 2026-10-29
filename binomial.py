"""
Cox-Ross-Rubinstein (CRR) Binomial Tree pricing and exact tree-node / finite-difference
Greeks for European and American options with continuous dividend yield.
"""

from __future__ import annotations

import numpy as np

from black_scholes import DAYS_PER_YEAR, _validate_inputs, normalize_dte


def _crr_tree_with_early_layers(
    S_val: float,
    K_val: float,
    T: float,
    sigma_val: float,
    r_val: float,
    q_val: float,
    opt: str,
    style_norm: str,
    n_steps: int,
) -> tuple[float, np.ndarray, np.ndarray, float, float, float]:
    """Run CRR backward induction and return (V0, V_step1, V_step2, u, d, dt)."""
    time = normalize_dte(T)
    dt = time / n_steps
    u = float(np.exp(sigma_val * np.sqrt(dt)))
    d = 1.0 / u
    disc = float(np.exp(-r_val * dt))
    growth = float(np.exp((r_val - q_val) * dt))
    p = float(np.clip((growth - d) / (u - d), 1e-9, 1.0 - 1e-9))
    one_minus_p = 1.0 - p

    j = np.arange(n_steps + 1, dtype=float)
    spot_terminal = S_val * (u**j) * (d ** (n_steps - j))

    if opt == "call":
        values = np.maximum(spot_terminal - K_val, 0.0)
    else:
        values = np.maximum(K_val - spot_terminal, 0.0)

    v_step2 = values.copy()
    v_step1 = values.copy()

    for step in range(n_steps - 1, -1, -1):
        cont_values = disc * (p * values[1 : step + 2] + one_minus_p * values[: step + 1])
        if style_norm == "american":
            idx = np.arange(step + 1, dtype=float)
            spot_nodes = S_val * (u**idx) * (d ** (step - idx))
            if opt == "call":
                exercise = np.maximum(spot_nodes - K_val, 0.0)
            else:
                exercise = np.maximum(K_val - spot_nodes, 0.0)
            values = np.maximum(cont_values, exercise)
        else:
            values = cont_values

        if step == 2:
            v_step2 = values.copy()
        elif step == 1:
            v_step1 = values.copy()

    return float(max(values[0], 0.0)), v_step1, v_step2, u, d, dt


def crr_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    style: str = "european",
    steps: int = 200,
) -> float:
    """Compute option value using a Cox-Ross-Rubinstein (CRR) binomial tree.

    Supports both `style="european"` and `style="american"` with continuous dividend yield.
    """
    S_val, K_val, sigma_val, r_val, q_val, opt = _validate_inputs(
        S, K, sigma, r, dividend_yield, option_type
    )
    style_norm = str(style).strip().lower()
    if style_norm not in ("european", "american"):
        raise ValueError("style must be 'european' or 'american'.")
    n_steps = int(steps)
    if n_steps < 2:
        raise ValueError("steps must be at least 2.")

    v0, _, _, _, _, _ = _crr_tree_with_early_layers(
        S_val, K_val, T, sigma_val, r_val, q_val, opt, style_norm, n_steps
    )
    return v0


def crr_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    style: str = "american",
    steps: int = 150,
) -> dict[str, float]:
    """Compute CRR binomial tree Greeks using interior tree layers for Delta/Gamma/Theta
    and central finite differences for Vega and Rho.
    """
    S_val, K_val, sigma_val, r_val, q_val, opt = _validate_inputs(
        S, K, sigma, r, dividend_yield, option_type
    )
    style_norm = str(style).strip().lower()
    if style_norm not in ("european", "american"):
        raise ValueError("style must be 'european' or 'american'.")
    n_steps = max(int(steps), 10)

    v0, v1, v2, u, d, dt = _crr_tree_with_early_layers(
        S_val, K_val, T, sigma_val, r_val, q_val, opt, style_norm, n_steps
    )

    # Step 1 nodes: index 0 is down (S*d), index 1 is up (S*u)
    s_u = S_val * u
    s_d = S_val * d
    delta = (float(v1[1]) - float(v1[0])) / (s_u - s_d)

    # Step 2 nodes: index 0 is dd (S*d^2), index 1 is ud (S), index 2 is uu (S*u^2)
    s_uu = S_val * (u**2)
    s_ud = S_val
    s_dd = S_val * (d**2)
    delta_u = (float(v2[2]) - float(v2[1])) / (s_uu - s_ud)
    delta_d = (float(v2[1]) - float(v2[0])) / (s_ud - s_dd)
    h = 0.5 * (s_uu - s_dd)
    gamma = (delta_u - delta_d) / h if h > 1e-12 else 0.0

    # Theta from middle node at step 2 (S_ud == S)
    theta_annual = (float(v2[1]) - v0) / (2.0 * dt) if dt > 1e-12 else 0.0
    theta_daily = theta_annual / DAYS_PER_YEAR

    # Vega and Rho via central difference in sigma and r
    dvol = 1e-3
    v_up = crr_price(S_val, K_val, T, sigma_val + dvol, r_val, q_val, opt, style_norm, n_steps)
    v_dn = crr_price(S_val, K_val, T, max(sigma_val - dvol, 1e-4), r_val, q_val, opt, style_norm, n_steps)
    vega = (v_up - v_dn) / (2.0 * dvol)
    vega_pct = vega / 100.0

    dr = 1e-4
    r_up = crr_price(S_val, K_val, T, sigma_val, r_val + dr, q_val, opt, style_norm, n_steps)
    r_dn = crr_price(S_val, K_val, T, sigma_val, r_val - dr, q_val, opt, style_norm, n_steps)
    rho = (r_up - r_dn) / (2.0 * dr)
    rho_pct = rho / 100.0

    return {
        "price": float(v0),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta_daily": float(theta_daily),
        "theta": float(theta_annual),
        "vega": float(vega),
        "vega_pct": float(vega_pct),
        "rho": float(rho),
        "rho_pct": float(rho_pct),
    }
