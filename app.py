import numpy as np
import streamlit as st
import pandas as pd

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
    .centered { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


def template_leg_definitions(template_name: str, spot: float, strike: float):
    base = {
        "side": "Buy",
        "kind": "Call",
        "strike": strike,
        "premium": 3.0,
        "quantity": 1,
        "dte": 30,
    }

    if template_name == "Long Call":
        return [{"side": "Buy", "kind": "Call", "strike": strike, "premium": 3.0, "quantity": 1, "dte": 30}]
    if template_name == "Short Call":
        return [{"side": "Sell", "kind": "Call", "strike": strike, "premium": 3.0, "quantity": 1, "dte": 30}]
    if template_name == "Long Put":
        return [{"side": "Buy", "kind": "Put", "strike": strike, "premium": 2.8, "quantity": 1, "dte": 30}]
    if template_name == "Short Put":
        return [{"side": "Sell", "kind": "Put", "strike": strike, "premium": 2.8, "quantity": 1, "dte": 30}]
    if template_name == "Bull Call Spread":
        return [
            {"side": "Buy", "kind": "Call", "strike": max(spot * 0.95, strike - 10), "premium": 4.20, "quantity": 1, "dte": 30},
            {"side": "Sell", "kind": "Call", "strike": max(spot * 1.05, strike + 10), "premium": 2.10, "quantity": 1, "dte": 30},
        ]
    if template_name == "Bear Put Spread":
        return [
            {"side": "Buy", "kind": "Put", "strike": max(spot * 1.05, strike + 10), "premium": 4.60, "quantity": 1, "dte": 30},
            {"side": "Sell", "kind": "Put", "strike": max(spot * 0.95, strike - 10), "premium": 2.20, "quantity": 1, "dte": 30},
        ]
    if template_name == "Long Butterfly":
        return [
            {"side": "Buy", "kind": "Call", "strike": max(spot * 0.95, strike - 10), "premium": 5.40, "quantity": 1, "dte": 30},
            {"side": "Sell", "kind": "Call", "strike": strike, "premium": 3.00, "quantity": 2, "dte": 30},
            {"side": "Buy", "kind": "Call", "strike": max(spot * 1.05, strike + 10), "premium": 1.80, "quantity": 1, "dte": 30},
        ]
    if template_name == "Iron Condor":
        return [
            {"side": "Sell", "kind": "Put", "strike": max(spot * 0.90, strike - 15), "premium": 1.80, "quantity": 1, "dte": 30},
            {"side": "Buy", "kind": "Put", "strike": max(spot * 0.85, strike - 25), "premium": 0.60, "quantity": 1, "dte": 30},
            {"side": "Sell", "kind": "Call", "strike": max(spot * 1.10, strike + 15), "premium": 1.60, "quantity": 1, "dte": 30},
            {"side": "Buy", "kind": "Call", "strike": max(spot * 1.15, strike + 25), "premium": 0.50, "quantity": 1, "dte": 30},
        ]
    if template_name == "Calendar Spread":
        return [
            {"side": "Buy", "kind": "Call", "strike": strike, "premium": 5.20, "quantity": 1, "dte": 60},
            {"side": "Sell", "kind": "Call", "strike": strike, "premium": 3.10, "quantity": 1, "dte": 30},
        ]
    return [{"side": "Buy", "kind": "Call", "strike": strike, "premium": 3.0, "quantity": 1, "dte": 30}]


def build_expiration_curve(spot_range, legs):
    curve = np.zeros_like(spot_range, dtype=float)
    for leg in legs:
        quantity = leg.get("quantity", 1)
        premium = float(leg.get("premium", 0.0))
        strike = float(leg.get("strike", 0.0))
        direction = 1.0 if leg.get("side", "Buy") == "Buy" else -1.0
        kind = leg.get("kind", "Call")
        if kind == "Call":
            intrinsic = np.maximum(spot_range - strike, 0.0)
        else:
            intrinsic = np.maximum(strike - spot_range, 0.0)
        curve += direction * quantity * (intrinsic - premium)
    return curve


def build_current_curve(spot_range, legs, strategy_dte, sigma, rate):
    curve = np.zeros_like(spot_range, dtype=float)
    for leg in legs:
        quantity = leg.get("quantity", 1)
        premium = float(leg.get("premium", 0.0))
        strike = float(leg.get("strike", 0.0))
        direction = 1.0 if leg.get("side", "Buy") == "Buy" else -1.0
        kind = leg.get("kind", "Call")
        dte = float(leg.get("dte", strategy_dte))
        prices = np.array([bs_price(s, strike, dte, sigma, rate, option_type=kind.lower()) for s in spot_range])
        curve += direction * quantity * (prices - premium)
    return curve


st.title("Greek Lens")
st.caption("Options Greeks • advanced strategy builder • theta decay explorer")

with st.sidebar:
    st.header("Live Greeks Calculator")
    spot = st.number_input("Spot Price", 100.0, min_value=10.0, max_value=2000.0, step=5.0)
    strike = st.number_input("Strike Price", 100.0, min_value=10.0, max_value=2000.0, step=5.0)
    dte = st.number_input("Days to Expiry (DTE)", 30, min_value=1, max_value=3650, step=1)
    iv = st.number_input("Implied Volatility (%)", 20.0, min_value=0.5, max_value=300.0, step=0.5)
    rate = st.number_input("Risk-Free Rate (%)", 4.5, min_value=-10.0, max_value=30.0, step=0.1) / 100.0

    sigma = iv / 100.0
    call_metrics = bs_greeks(spot, strike, dte, sigma, rate, option_type="call")
    put_metrics = bs_greeks(spot, strike, dte, sigma, rate, option_type="put")

    st.markdown("---")
    st.header("Strategy Controls")
    strategy_template = st.selectbox(
        "Strategy Template",
        [
            "Custom",
            "Long Call",
            "Short Call",
            "Long Put",
            "Short Put",
            "Bull Call Spread",
            "Bear Put Spread",
            "Long Butterfly",
            "Iron Condor",
            "Calendar Spread",
        ],
    )
    strategy_dte = st.slider("Strategy Curve DTE", min_value=1, max_value=365, value=30, step=1)

    st.markdown("### Current Option Snapshot")

    st.metric("Call Price", f"${call_metrics['price']:.2f}")
    st.metric("Put Price", f"${put_metrics['price']:.2f}")
    st.metric("Call Delta", f"{call_metrics['delta']:.4f}")
    st.metric("Put Delta", f"{put_metrics['delta']:.4f}")


st.subheader("Greeks Explorer")
tab_price, tab_theta, tab_strategy, tab_heatmap = st.tabs(["Live Greeks", "Theta Decay", "Strategy Builder", "Sensitivity Heatmap"])

with tab_price:
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

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Call")
        st.metric("Price", f"${call_metrics['price']:.2f}")
        st.metric("Delta", f"{call_metrics['delta']:.4f}")
        st.metric("Gamma", f"{call_metrics['gamma']:.5f}")
        st.metric("Theta/day", f"${call_metrics['theta_daily']:.4f}")
        st.metric("Vega", f"${call_metrics['vega']:.4f}")
    with col2:
        st.markdown("### Put")
        st.metric("Price", f"${put_metrics['price']:.2f}")
        st.metric("Delta", f"{put_metrics['delta']:.4f}")
        st.metric("Gamma", f"{put_metrics['gamma']:.5f}")
        st.metric("Theta/day", f"${put_metrics['theta_daily']:.4f}")
        st.metric("Vega", f"${put_metrics['vega']:.4f}")

with tab_theta:
    sim_dte = st.slider("Simulated DTE", min_value=1, max_value=365, value=30, step=1)
    sim_option_type = st.radio("Option Type for Theta Simulation", ["Call", "Put"], horizontal=True)
    sim_days = np.linspace(1, sim_dte, 80)
    sim_values = []
    sim_deltas = []
    for day in sim_days:
        metrics = bs_greeks(spot, strike, day, sigma, rate, option_type=sim_option_type.lower())
        sim_values.append(metrics["price"])
        sim_deltas.append(metrics["delta"])

    sim_fig = make_theta_decay_figure(sim_days, sim_values, sim_deltas, sim_option_type)
    st.plotly_chart(sim_fig, use_container_width=True)
    st.caption("Option value decays as expiry approaches and the Delta profile compresses toward the ATM zone.")

with tab_strategy:
    if strategy_template == "Custom":
        leg_count = st.slider("Number of active legs", min_value=1, max_value=4, value=2, step=1)
        legs = []
        for idx in range(leg_count):
            st.markdown(f"### Leg {idx + 1}")
            col_side, col_kind, col_strike, col_premium, col_quantity, col_dte = st.columns(6)
            with col_side:
                side = st.selectbox("Side", ["Buy", "Sell"], key=f"side_{idx}")
            with col_kind:
                kind = st.selectbox("Type", ["Call", "Put"], key=f"kind_{idx}")
            with col_strike:
                leg_strike = st.number_input("Strike", value=strike + (idx - 1) * 5, min_value=10.0, max_value=2000.0, step=5.0, key=f"strike_{idx}")
            with col_premium:
                premium = st.number_input("Premium", value=2.5 + idx * 0.75, min_value=0.0, max_value=200.0, step=0.25, key=f"premium_{idx}")
            with col_quantity:
                quantity = st.number_input("Qty", value=1, min_value=1, max_value=500, step=1, key=f"qty_{idx}")
            with col_dte:
                dte_leg = st.number_input("DTE", value=strategy_dte, min_value=1, max_value=3650, step=1, key=f"dte_{idx}")
            legs.append({
                "side": side,
                "kind": kind,
                "strike": leg_strike,
                "premium": premium,
                "quantity": quantity,
                "dte": dte_leg,
            })
    else:
        legs = template_leg_definitions(strategy_template, spot, strike)
        st.info(f"Loaded preset: {strategy_template}")
        for i, leg in enumerate(legs, start=1):
            st.markdown(f"- Leg {i}: {leg['side']} {leg['kind']} @ {leg['strike']:.2f}, Qty {leg['quantity']}, Premium ${leg['premium']:.2f}, DTE {leg['dte']}")

    if legs:
        spot_range = np.linspace(max(0.1, spot * 0.45), max(spot * 1.8, strike * 2.0), 500)
        expiry_curve = build_expiration_curve(spot_range, legs)
        current_curve = build_current_curve(spot_range, legs, strategy_dte, sigma, rate)
        payoff_fig = make_payoff_figure(
            spot_range,
            expiry_curve,
            current_curve,
            title="Strategy Profit / Loss: Expiration vs. Current",
            underlying_price=spot,
            strike=strike,
        )
        st.plotly_chart(payoff_fig, use_container_width=True)

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

with tab_heatmap:
    heatmap_style = st.radio("Heatmap Option», ["Call", "Put"], horizontal=True)
    spot_grid = np.linspace(max(10.0, spot * 0.5), max(spot * 2.2, 300.0), 80)
    vol_shocks = np.linspace(-50, 50, 80)
    vol_grid = sigma * (1.0 + vol_shocks / 100.0)
    premium_grid = np.array([
        [bs_price(s, strike, dte, vol, rate, option_type=heatmap_style.lower()) for vol in vol_grid]
        for s in spot_grid
    ])
    heatmap_fig = make_heatmap_figure(
        spot_grid,
        vol_shocks,
        premium_grid,
        z_label=f"{heatmap_style} Premium",
        title=f"{heatmap_style} Premium Sensitivity: Spot vs. Volatility Shock",
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

st.markdown(
    """
    <div class="centered">
        <small>Built with Python, Streamlit, NumPy, SciPy, and Plotly.</small>
    </div>
    """,
    unsafe_allow_html=True,
)
































































































































"""