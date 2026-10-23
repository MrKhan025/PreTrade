import io

import numpy as np
import pandas as pd
import streamlit as st

from black_scholes import (
    bs_greeks_dividend,
    bs_price_dividend,
)
from charts import make_heatmap_figure, make_payoff_figure, make_theta_decay_figure

st.set_page_config(page_title="Greek Lens", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .main { background: #07111f; }
    .stApp { background: linear-gradient(180deg, #091524 0%, #111827 100%); }
    .block-container { padding-top: 1.6rem; }
    div[data-testid="stMetricValue"] { font-size: 1.65rem; }
    .hero {
        background: linear-gradient(135deg, rgba(15,118,110,0.14), rgba(59,130,246,0.12));
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1.2rem;
    }
    .pills {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.8rem;
    }
    .pill {
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148,163,184,0.25);
        color: #e2e8f0;
        font-size: 0.78rem;
    }
    .score-card {
        background: linear-gradient(180deg, rgba(15,23,42,0.88), rgba(17,24,39,0.92));
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 14px;
        padding: 1rem 0.9rem;
        min-height: 118px;
        box-shadow: 0 10px 25px rgba(15,23,42,0.22);
    }
    .summary-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.85rem;
        margin: 0.5rem 0 1rem;
    }
    .summary-box {
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 12px;
        padding: 0.8rem;
    }
    .preset-btn button {
        border-radius: 12px;
        min-height: 54px;
    }
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


def strategy_current_curve(spot_range: np.ndarray, legs: list[dict], dte: float, sigma: float, rate: float, dividend_yield: float = 0.0) -> np.ndarray:
    curve = np.zeros_like(spot_range, dtype=float)
    for leg in legs:
        quantity = leg.get("quantity", 1)
        premium = float(leg.get("premium", 0.0))
        strike = float(leg.get("strike", 0.0))
        direction = 1.0 if leg.get("side", "Buy") == "Buy" else -1.0
        kind = leg.get("kind", "Call")
        leg_dte = float(leg.get("dte", dte))
        prices = np.array(
            [
                bs_price_dividend(
                    s,
                    strike,
                    leg_dte,
                    sigma,
                    rate,
                    dividend_yield=dividend_yield,
                    option_type=kind.lower(),
                    american_adjustment=True,
                )
                for s in spot_range
            ]
        )
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


def strategy_score(metrics: dict) -> float:
    max_profit = max(metrics["max_profit"], 0.0)
    max_loss = abs(min(metrics["max_loss"], 0.0))
    breakeven = metrics["breakeven"] if metrics["breakeven"] is not None else 0.0
    score = 50 + 0.6 * max_profit - 0.35 * max_loss + 0.2 * breakeven
    return float(np.clip(score, 0, 100))


def trade_idea_summary(strategy_name: str, metrics: dict, spot: float, strike: float) -> str:
    max_profit = metrics["max_profit"]
    max_loss = metrics["max_loss"]
    breakeven = metrics["breakeven"]
    if strategy_name in ["Bull Call Spread", "Bear Put Spread", "Long Butterfly", "Iron Condor"]:
        risk_note = "defined-risk setup suitable for a directional or range-bound thesis."
    else:
        risk_note = "directional exposure with broader risk if the market moves against the trade."
    breakeven_note = "No clear break-even point detected in this spot window." if breakeven is None else f"Break-even is around ${breakeven:.2f} per share."
    return (
        f"{strategy_name} is a {risk_note} "
        f"It offers a max profit of ${max_profit:.2f}, max loss of ${abs(max_loss):.2f}, and {breakeven_note} "
        f"The structure is designed around spot={spot:.2f} and strike={strike:.2f}."
    )


st.title("Greek Lens")
st.caption("Options Greeks • multi-leg strategy analysis • portfolio risk • trade ideas")

with st.sidebar:
    st.header("Market Inputs")
    spot = st.number_input("Spot Price", value=100.0, min_value=10.0, max_value=2000.0, step=5.0)
    strike = st.number_input("Strike Price", value=100.0, min_value=10.0, max_value=2000.0, step=5.0)
    dte = st.number_input("Days to Expiry", value=30, min_value=1, max_value=3650, step=1)
    iv = st.number_input("Implied Volatility (%)", value=20.0, min_value=0.5, max_value=300.0, step=0.5)
    rate = st.number_input("Risk-Free Rate (%)", value=4.5, min_value=-10.0, max_value=30.0, step=0.1) / 100.0
    dividend_yield = st.number_input("Dividend Yield (%)", value=0.0, min_value=0.0, max_value=15.0, step=0.1) / 100.0
    sigma = iv / 100.0
    early_adjust = st.checkbox("Apply early-exercise bias adjustment", value=True)

    st.markdown("---")
    st.header("Option Snapshot")
    call_metrics = bs_greeks_dividend(spot, strike, dte, sigma, rate, dividend_yield=dividend_yield, option_type="call", american_adjustment=early_adjust)
    put_metrics = bs_greeks_dividend(spot, strike, dte, sigma, rate, dividend_yield=dividend_yield, option_type="put", american_adjustment=early_adjust)
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

st.markdown(
    """
    <div class="hero">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <div style="font-size:0.8rem; letter-spacing:0.12em; color:#a5f3fc; text-transform:uppercase; font-weight:700;">Trade intelligence</div>
                <h2 style="margin:0.25rem 0 0.35rem; font-size:2rem;">Options workstation</h2>
            </div>
            <div class="pills">
                <span class="pill">Black-Scholes</span>
                <span class="pill">Greeks</span>
                <span class="pill">Payoff curves</span>
                <span class="pill">Portfolio risk</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

summary_row = st.container()
with summary_row:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            "<div class='score-card'><div style='color:#a5f3fc; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.10em;'>Call</div><h3 style='margin:0.35rem 0 0;'>${:.2f}</h3></div>".format(call_metrics["price"]),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='score-card'><div style='color:#a5f3fc; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.10em;'>Put</div><h3 style='margin:0.35rem 0 0;'>${:.2f}</h3></div>".format(put_metrics["price"]),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='score-card'><div style='color:#a5f3fc; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.10em;'>Gamma</div><h3 style='margin:0.35rem 0 0;'>{:.5f}</h3></div>".format(call_metrics["gamma"]),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            "<div class='score-card'><div style='color:#a5f3fc; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.10em;'>Vega</div><h3 style='margin:0.35rem 0 0;'>${:.4f}</h3></div>".format(call_metrics["vega"]),
            unsafe_allow_html=True,
        )

price_tab, theta_tab, strategy_tab, portfolio_tab, idea_tab, heatmap_tab = st.tabs(
    ["Live Greeks", "Theta Decay", "Strategy Builder", "Portfolio Dashboard", "Trade Idea", "Sensitivity Heatmap"]
)

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
        metrics = bs_greeks_dividend(spot, strike, day, sigma, rate, dividend_yield=dividend_yield, option_type=sim_option_type.lower(), american_adjustment=early_adjust)
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
    current_curve = strategy_current_curve(spot_range, legs, strategy_dte, sigma, rate, dividend_yield)
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
    score = strategy_score(metrics)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Setup Score", f"{score:.1f}")
    with s2:
        st.metric("Max Profit", f"${metrics['max_profit']:.2f}")
    with s3:
        st.metric("Max Loss", f"${abs(metrics['max_loss']):.2f}")
    with s4:
        breakeven_label = "N/A" if metrics['breakeven'] is None else f"${metrics['breakeven']:.2f}"
        st.metric("Break-even", breakeven_label)

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
    st.download_button("Download payoff CSV", data=export_csv.to_csv(index=False).encode("utf-8"), file_name="strategy_payoff.csv", mime="text/csv")
    png_buffer = io.BytesIO()
    payoff_fig.write_image(png_buffer, format="png", width=1400, height=800)
    st.download_button("Download payoff PNG", data=png_buffer.getvalue(), file_name="strategy_payoff.png", mime="image/png")

with portfolio_tab:
    default_portfolio = pd.DataFrame(
        [
            {"symbol": "AAPL", "qty": 100, "delta": 0.58, "gamma": 0.02, "theta": -12.40, "vega": 18.0, "market_value": 15600.0},
            {"symbol": "MSFT", "qty": 80, "delta": 0.44, "gamma": 0.015, "theta": -9.10, "vega": 13.2, "market_value": 14880.0},
            {"symbol": "SPY", "qty": 60, "delta": 0.52, "gamma": 0.01, "theta": -7.30, "vega": 9.5, "market_value": 13200.0},
        ]
    )
    portfolio_df = st.data_editor(default_portfolio, num_rows="dynamic", use_container_width=True)
    exposures = {
        "Net Delta": float(portfolio_df["qty"].mul(portfolio_df["delta"]).sum()),
        "Net Gamma": float(portfolio_df["qty"].mul(portfolio_df["gamma"]).sum()),
        "Net Theta": float(portfolio_df["qty"].mul(portfolio_df["theta"]).sum()),
        "Net Vega": float(portfolio_df["qty"].mul(portfolio_df["vega"]).sum()),
        "Notional": float(portfolio_df["market_value"].sum()),
    }
    p1, p2, p3, p4 = st.columns(4)
    for col, label in zip([p1, p2, p3, p4], ["Net Delta", "Net Gamma", "Net Theta", "Net Vega"]):
        with col:
            val = exposures[label]
            st.metric(label, f"{val:.2f}")
    st.metric("Portfolio Notional", f"${exposures['Notional']:.2f}")
    exposure_df = pd.DataFrame({"Exposure": ["Delta", "Gamma", "Theta", "Vega"], "Value": [exposures["Net Delta"], exposures["Net Gamma"], exposures["Net Theta"], exposures["Net Vega"]]})
    st.bar_chart(exposure_df.set_index("Exposure"), use_container_width=True)

with idea_tab:
    strategy_name = st.session_state.get("strategy_preset", "Bull Call Spread")
    idea_metrics = strategy_summary_metrics(np.linspace(50, 200, 300), strategy_expiry_curve(np.linspace(50, 200, 300), strategy_preset(strategy_name, spot, strike)))
    summary_text = trade_idea_summary(strategy_name, idea_metrics, spot, strike)
    st.markdown("### Trade Idea Summary")
    st.write(summary_text)
    idea_df = pd.DataFrame(
        {
            "Field": ["Strategy", "Bias", "Max Profit", "Max Loss", "Break-even", "Setup Score"],
            "Value": [
                strategy_name,
                "Directional / event-driven",
                f"${idea_metrics['max_profit']:.2f}",
                f"${abs(idea_metrics['max_loss']):.2f}",
                f"${idea_metrics['breakeven']:.2f}" if idea_metrics['breakeven'] is not None else "N/A",
                f"{strategy_score(idea_metrics):.1f}",
            ],
        }
    )
    st.dataframe(idea_df, use_container_width=True)

with heatmap_tab:
    style = st.radio("Heatmap Option", ["Call", "Put"], horizontal=True)
    spot_grid = np.linspace(max(10.0, spot * 0.5), max(200.0, spot * 2.2), 80)
    vol_shocks = np.linspace(-50, 50, 80)
    vol_grid = sigma * (1 + vol_shocks / 100.0)
    premium_grid = np.array(
        [[bs_price_dividend(s, strike, dte, vol, rate, dividend_yield=dividend_yield, option_type=style.lower(), american_adjustment=early_adjust) for vol in vol_grid] for s in spot_grid]
    )
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
Black-Scholes framework for spot={spot:.2f}, strike={strike:.2f}, DTE={dte}, IV={iv:.2f}%, r={rate * 100:.2f}%, dividend yield={dividend_yield * 100:.2f}%.

- Call price: ${call_metrics['price']:.2f}
- Put price: ${put_metrics['price']:.2f}
- Call delta: {call_metrics['delta']:.4f}
- Put delta: {put_metrics['delta']:.4f}
- Gamma: {call_metrics['gamma']:.5f}
- Theta/day: ${call_metrics['theta_daily']:.4f} (call) / ${put_metrics['theta_daily']:.4f} (put)
- Vega: ${call_metrics['vega']:.4f}
- Early-exercise adjustment: {'On' if early_adjust else 'Off'}
"""
st.code(summary, language="text")

st.markdown("<div style='text-align:center; color:#94a3b8; margin-top:16px;'>Built with Python, Streamlit, NumPy, SciPy, and Plotly.</div>", unsafe_allow_html=True)












































































































