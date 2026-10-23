from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from black_scholes import bs_greeks, bs_price
from charts import make_heatmap_figure, make_payoff_figure, make_theta_decay_figure

st.set_page_config(page_title="Greek Lens", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .main { background: #0b1120; }
    .stApp { background: linear-gradient(180deg, #0f172a 0%, #111827 100%); }
    .block-container { padding-top: 1.8rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .metric-card { background: rgba(15, 23, 42, 0.72); border: 1px solid #1f2937; border-radius: 12px; padding: 12px; }
    .preset-btn button { border-radius: 12px; min-height: 62px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def strategy_preset(name: str, spot: float, strike: float):
    presets = {
        "Long Call": [{"side": "Buy", "kind": "Call", "strike": strike, "premium": 3.2, "quantity": 1, "dte": 30}],
        "Short Call": [{"side": "Sell", "kind": "Call", "strike": strike, "premium": 3.2, "quantity": 1, "dte": 30}],
        "Long Put": [{"side": "Buy", "kind": "Put", "strike": strike, "premium": 2.8, "quantity": 1, "dte": 30}],
        "Short Put": [{"side": "Sell", "kind": "Put", "strike": strike, "premium": 2.8, "quantity": 1, "dte": 30}],
        "Bull Call Spread": [
            {"side": "Buy", "kind": "Call", "strike": max(spot * 0.95, strike - 10), "premium": 4.4, "quantity": 1, "dte": 30},
            {"side": "Sell", "kind": "Call", "strike": max(spot * 1.05, strike + 10), "premium": 2.1, "quantity": 1, "dte": 30},
        ],
        "Bear Put Spread": [
            {"side": "Buy", "kind": "Put", "strike": max(spot * 1.05, strike + 10), "premium": 4.8, "quantity": 1, "dte": 30},
            {"side": "Sell", "kind": "Put", "strike": max(spot * 0.95, strike - 10), "premium": 2.3, "quantity": 1, "dte": 30},
        ],
        "Long Butterfly": [
            {"side": "Buy", "kind": "Call", "strike": max(spot * 0.95, strike - 10), "premium": 5.5, "quantity": 1, "dte": 30},
            {"side": "Sell", "kind": "Call", "strike": strike, "premium": 3.1, "quantity": 2, "dte": 30},
            {"side": "Buy", "kind": "Call", "strike": max(spot * 1.05, strike + 10), "premium": 1.9, "quantity": 1, "dte": 30},
        ],
        "Iron Condor": [
            {"side": "Sell", "kind": "Put", "strike": max(spot * 0.9, strike - 15), "premium": 1.8, "quantity": 1, "dte": 30},
            {"side": "Buy", "kind": "Put", "strike": max(spot * 0.85, strike - 25), "premium": 0.6, "quantity": 1, "dte": 30},
            {"side": "Sell", "kind": "Call", "strike": max(spot * 1.1, strike + 15), "premium": 1.7, "quantity": 1, "dte": 30},
            {"side": "Buy", "kind": "Call", "strike": max(spot * 1.15, strike + 25), "premium": 0.5, "quantity": 1, "dte": 30},
        ],
        "Calendar Spread": [
            {"side": "Buy", "kind": "Call", "strike": strike, "premium": 5.4, "quantity": 1, "dte": 60},
            {"side": "Sell", "kind": "Call", "strike": strike, "premium": 3.2, "quantity": 1, "dte": 30},
        ],
    }
    return presets.get(name, presets["Long Call"])


def strategy_expiry_curve(spot_range: np.ndarray, legs: list[dict]) -> np.ndarray:
    curve = np.zeros_like(spot_range, dtype=float)
    for leg in legs:
        quantity = leg.get("quantity", 1)
        premium = float(leg.get("premium", 0.0))
        strike = float(leg.get("strike", 0.0))
        direction = 1.0 if leg.get("side", "Buy") == "Buy" else -1.0
        kind = leg.get("kind", "Call")
        intrinsic = np.maximum(spot_range - strike, 0.0) if kind == "Call" else np.maximum(strike - spot_range, 0.0)
        curve += direction * quantity * (intrinsic - premium)
    return curve


def strategy_current_curve(spot_range: np.ndarray, legs: list[dict], dte: float, sigma: float, rate: float) -> np.ndarray:
    curve = np.zeros_like(spot_range, dtype=float)
    for leg in legs:
        quantity = leg.get("quantity", 1)
        premium = float(leg.get("premium", 0.0))
        strike = float(leg.get("strike", 0.0))
        direction = 1.0 if leg.get("side", "Buy") == "Buy" else -1.0
        kind = leg.get("kind", "Call")
        leg_dte = float(leg.get("dte", dte))
        prices = np.array([bs_price(s, strike, leg_dte, sigma, rate, option_type=kind.lower()) for s in spot_range])
        curve += direction * quantity * (prices - premium)
    return curve


def strategy_summary_metrics(spot_range: np.ndarray, payoff_curve: np.ndarray) -> dict:
    max_profit = float(np.max(payoff_curve))
    max_loss = float(np.min(payoff_curve))
    zero_crossings = []
    for i in range(1, len(spot_range)):
        prev = payoff_curve[i - 1]
        curr = payoff_curve[i]
        if (prev < 0 and curr >= 0) or (prev > 0 and curr <= 0):
            zero_crossings.append((spot_range[i - 1], spot_range[i]))
    breakeven = None
    if zero_crossings:
        breakeven = float((zero_crossings[0][0] + zero_crossings[0][1]) / 2.0)
    return {"max_profit": max_profit, "max_loss": max_loss, "breakeven": breakeven}


st.title("Greek Lens")
st.caption("Options Greeks • multi-leg payoff builder • theta decay • strategy analytics")

with st.sidebar:
    st.header("Live Greeks Calculator")
    spot = st.number_input("Spot Price", value=100.0, min_value=10.0, max_value=2000.0, step=5.0)
    strike = st.number_input("Strike Price", value=100.0, min_value=10.0, max_value=2000.0, step=5.0)
    dte = st.number_input("Days to Expiry", value=30, min_value=1, max_value=3650, step=1)
    iv = st.number_input("Implied Volatility (%)", value=20.0, min_value=0.5, max_value=300.0, step=0.5)
    rate = st.number_input("Risk-Free Rate (%)", value=4.5, min_value=-10.0, max_value=30.0, step=0.1) / 100.0
    sigma = iv / 100.0

    st.markdown("---")
    st.header("Option Snapshot")
    call_metrics = bs_greeks(spot, strike, dte, sigma, rate, option_type="call")
    put_metrics = bs_greeks(spot, strike, dte, sigma, rate, option_type="put")
    st.metric("Call Price", f"${call_metrics['price']:.2f}")
    st.metric("Put Price", f"${put_metrics['price']:.2f}")
    st.metric("Call Delta", f"{call_metrics['delta']:.4f}")
    st.metric("Put Delta", f"{put_metrics['delta']:.4f}")

    st.markdown("---")
    st.header("Preset Strategy")
    preset_names = [
        "Bull Call Spread",
        "Bear Put Spread",
        "Long Butterfly",
        "Iron Condor",
        "Calendar Spread",
        "Long Call",
        "Short Call",
        "Long Put",
        "Short Put",
    ]
    if "strategy_preset" not in st.session_state:
        st.session_state.strategy_preset = "Bull Call Spread"

    for name in preset_names:
        if st.button(name, key=f"preset_{name}", use_container_width=True):
            st.session_state.strategy_preset = name

st.subheader("Greeks Explorer")
price_tab, theta_tab, strategy_tab, heatmap_tab = st.tabs(["Live Greeks", "Theta Decay", "Strategy Builder", "Sensitivity Heatmap"])

with price_tab:
    metrics_df = pd.DataFrame(
        {
            "Price": [call_metrics["price"], put_metrics["price"]],
            "Delta": [call_metrics["delta"], put_metrics["delta"]],
            "Gamma": [call_metrics["gamma"], put_metrics["gamma"]],
            "Theta / day": [call_metrics["theta_daily"], put_metrics["theta_daily"]],
            "Vega": [call_metrics["vega"], put_metrics["vega"]],
        },
        index=["Call", "Put"],
    )
    st.dataframe(metrics_df.style.format({"Price": "${:.2f}", "Theta / day": "${:.4f}", "Vega": "${:.4f}"}), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Call")
        st.metric("Price", f"${call_metrics['price']:.2f}")
        st.metric("Delta", f"{call_metrics['delta']:.4f}")
        st.metric("Gamma", f"{call_metrics['gamma']:.5f}")
        st.metric("Theta/day", f"${call_metrics['theta_daily']:.4f}")
        st.metric("Vega", f"${call_metrics['vega']:.4f}")
    with c2:
        st.markdown("### Put")
        st.metric("Price", f"${put_metrics['price']:.2f}")
        st.metric("Delta", f"{put_metrics['delta']:.4f}")
        st.metric("Gamma", f"{put_metrics['gamma']:.5f}")
        st.metric("Theta/day", f"${put_metrics['theta_daily']:.4f}")
        st.metric("Vega", f"${put_metrics['vega']:.4f}")

with theta_tab:
    sim_dte = st.slider("Simulated DTE", min_value=1, max_value=365, value=30, step=1)
    sim_option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
    sim_days = np.linspace(1, sim_dte, 80)
    sim_values = []
    sim_deltas = []
    for day in sim_days:
        metrics = bs_greeks(spot, strike, day, sigma, rate, option_type=sim_option_type.lower())
        sim_values.append(metrics["price"])
        sim_deltas.append(metrics["delta"])
    st.plotly_chart(make_theta_decay_figure(sim_days, sim_values, sim_deltas, sim_option_type), use_container_width=True)

with strategy_tab:
    strategy_dte = st.slider("Strategy Curve DTE", min_value=1, max_value=365, value=30, step=1)
    selected_template = st.session_state.get("strategy_preset", "Bull Call Spread")
    st.info(f"Selected preset: {selected_template}")

    custom_mode = st.checkbox("Build custom strategy", value=False)
    if custom_mode:
        leg_count = st.slider("Number of legs", min_value=1, max_value=4, value=2)
        legs = []
        for idx in range(leg_count):
            st.markdown(f"### Leg {idx + 1}")
            cols = st.columns(6)
            with cols[0]:
                side = st.selectbox("Side", ["Buy", "Sell"], key=f"side_{idx}")
            with cols[1]:
                kind = st.selectbox("Type", ["Call", "Put"], key=f"type_{idx}")
            with cols[2]:
                leg_strike = st.number_input("Strike", value=strike + (idx - 1) * 5, min_value=10.0, max_value=2000.0, step=5.0, key=f"strike_{idx}")
            with cols[3]:
                premium = st.number_input("Premium", value=2.5 + idx * 0.75, min_value=0.0, max_value=200.0, step=0.25, key=f"premium_{idx}")
            with cols[4]:
                quantity = st.number_input("Qty", value=1, min_value=1, max_value=500, step=1, key=f"qty_{idx}")
            with cols[5]:
                leg_dte = st.number_input("DTE", value=strategy_dte, min_value=1, max_value=3650, step=1, key=f"dte_{idx}")
            legs.append({
                "side": side,
                "kind": kind,
                "strike": leg_strike,
                "premium": premium,
                "quantity": quantity,
                "dte": leg_dte,
            })
    else:
        legs = strategy_preset(selected_template, spot, strike)
        for i, leg in enumerate(legs, start=1):
            st.write(f"Leg {i}: {leg['side']} {leg['kind']} @ {leg['strike']:.2f}, Qty {leg['quantity']}, Premium ${leg['premium']:.2f}, DTE {leg['dte']}")

    spot_range = np.linspace(max(0.1, spot * 0.45), max(spot * 1.8, strike * 2.0), 500)
    expiry_curve = strategy_expiry_curve(spot_range, legs)
    current_curve = strategy_current_curve(spot_range, legs, strategy_dte, sigma, rate)
    payoff_fig = make_payoff_figure(
        spot_range,
        expiry_curve,
        current_curve,
        title="Strategy P/L at Expiration vs. T+0",
        underlying_price=spot,
        strike=strike,
    )
    st.plotly_chart(payoff_fig, use_container_width=True)

    metrics = strategy_summary_metrics(spot_range, expiry_curve)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Max Profit", f"${metrics['max_profit']:.2f}")
    with m2:
        st.metric("Max Loss", f"${metrics['max_loss']:.2f}")
    with m3:
        st.metric("Break-even", f"${metrics['breakeven']:.2f}" if metrics['breakeven'] is not None else "N/A")

    strategy_df = pd.DataFrame(
        [{
            "Leg": i + 1,
            "Side": leg["side"],
            "Type": leg["kind"],
            "Strike": leg["strike"],
            "Qty": leg["quantity"],
            "Premium": leg["premium"],
            "DTE": leg.get("dte", strategy_dte),
        } for i, leg in enumerate(legs)]
    )
    st.dataframe(strategy_df, use_container_width=True)

    export_csv = pd.DataFrame({"Spot": spot_range, "Expiration_PnL": expiry_curve, "Current_T0_PnL": current_curve})
    csv_bytes = export_csv.to_csv(index=False).encode("utf-8")
    st.download_button("Download payoff CSV", data=csv_bytes, file_name="strategy_payoff.csv", mime="text/csv")

    png_buffer = io.BytesIO()
    payoff_fig.write_image(png_buffer, format="png", width=1400, height=800)
    st.download_button("Download payoff PNG", data=png_buffer.getvalue(), file_name="strategy_payoff.png", mime="image/png")

with heatmap_tab:
    style = st.radio("Heatmap Option", ["Call", "Put"], horizontal=True)
    spot_grid = np.linspace(max(10.0, spot * 0.5), max(200.0, spot * 2.2), 80)
    vol_shocks = np.linspace(-50, 50, 80)
    vol_grid = sigma * (1 + vol_shocks / 100.0)
    premium_grid = np.array([
        [bs_price(s, strike, dte, vol, rate, option_type=style.lower()) for vol in vol_grid]
        for s in spot_grid
    ])
    heatmap_fig = make_heatmap_figure(
        spot_grid,
        vol_shocks,
        premium_grid,
        z_label=f"{style} Premium",
        title=f"{style} Premium Sensitivity: Spot vs. Volatility Shock",
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

st.markdown("---")
st.subheader("Model Summary")
summary = f"""
Black-Scholes framework for spot={spot:.2f}, strike={strike:.2f}, DTE={dte}, IV={iv:.2f}%, r={rate * 100:.2f}%.

- Call price: ${call_metrics['price']:.2f}
- Put price: ${put_metrics['price']:.2f}
- Call delta: {call_metrics['delta']:.4f}
- Put delta: {put_metrics['delta']:.4f}
- Gamma: {call_metrics['gamma']:.5f}
- Theta/day: ${call_metrics['theta_daily']:.4f} (call) / ${put_metrics['theta_daily']:.4f} (put)
- Vega: ${call_metrics['vega']:.4f}
"""
st.code(summary, language="text")

st.markdown("<div style='text-align:center; color:#94a3b8; margin-top:14px;'>Built with Python, Streamlit, NumPy, SciPy, and Plotly.</div>", unsafe_allow_html=True)
