from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from black_scholes import (
    bs_greeks_dividend,
    bs_price_dividend,
    classify_strategy_bias,
    implied_volatility,
    put_call_parity_check,
    simulate_theta_decay,
    strategy_current_curve,
    strategy_expiry_curve,
    strategy_greeks,
    strategy_preset,
    strategy_score,
    strategy_summary_metrics,
    trade_idea_summary,
)
from charts import (
    make_greeks_profile_figure,
    make_heatmap_figure,
    make_payoff_figure,
    make_portfolio_exposure_figure,
    make_surface_3d_figure,
    make_theta_decay_figure,
)

st.set_page_config(page_title="Greek Lens", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .main { background: #07111f; }
    .stApp { background: linear-gradient(180deg, #091524 0%, #111827 100%); }
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.65rem; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(15,23,42,0.93));
        border-right: 1px solid rgba(148,163,184,0.14);
    }
    [data-testid="stSidebar"] > div {
        padding-top: 0.75rem;
    }

    .premium-hero {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(59,130,246,0.10));
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1.15rem;
        box-shadow: 0 18px 40px rgba(2, 6, 23, 0.18);
        animation: fadeIn 0.7s ease;
    }
    .precision-banner {
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        color: #a5f3fc;
        text-transform: uppercase;
        font-weight: 700;
    }
    .premium-pill {
        display: inline-block;
        padding: 0.38rem 0.7rem;
        margin-right: 0.4rem;
        margin-top: 0.45rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.96);
        border: 1px solid rgba(148,163,184,0.2);
        color: #e2e8f0;
        font-size: 0.76rem;
    }
    .score-card {
        background: linear-gradient(180deg, rgba(15,23,42,0.90), rgba(17,24,39,0.94));
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 14px;
        padding: 0.9rem 0.85rem;
        min-height: 110px;
        box-shadow: 0 12px 28px rgba(15,23,42,0.22);
        animation: riseIn 0.6s ease;
    }
    .trade-setup {
        background: linear-gradient(180deg, rgba(15,23,42,0.9), rgba(2,6,23,0.96));
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 16px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 1rem;
        box-shadow: 0 16px 34px rgba(2,6,23,0.18);
    }
    .trade-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
        gap: 0.75rem;
        margin-top: 0.55rem;
    }
    .trade-box {
        background: rgba(15, 23, 42, 0.68);
        border: 1px solid rgba(148,163,184,0.08);
        border-radius: 12px;
        padding: 0.7rem 0.6rem;
    }
    .panel-shell {
        background: rgba(8, 15, 28, 0.68);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 18px;
        padding: 0.9rem;
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.16);
        animation: fadeIn 0.8s ease;
    }
    .tab-container [data-testid="stHorizontalBlock"] > div {
        border-radius: 12px;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes riseIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_breakevens(metrics: dict) -> str:
    breakevens = metrics.get("breakevens") or (
        [metrics["breakeven"]] if metrics.get("breakeven") is not None else []
    )
    if not breakevens:
        return "N/A"
    return " / ".join(f"${b:.2f}" for b in breakevens)


st.title("Greek Lens")
st.caption("Options Greeks • multi-leg strategy analysis • implied volatility solver • portfolio risk")

with st.sidebar:
    st.header("Market Inputs")
    spot = st.number_input("Spot Price ($)", value=100.0, min_value=1.0, max_value=5000.0, step=5.0)
    strike = st.number_input("Strike Price ($)", value=100.0, min_value=1.0, max_value=5000.0, step=5.0)
    dte = st.number_input("Days to Expiry (DTE)", value=30, min_value=1, max_value=3650, step=1)
    iv = st.number_input("Implied Volatility (%)", value=20.0, min_value=0.5, max_value=500.0, step=0.5)
    rate = st.number_input("Risk-Free Rate (%)", value=4.5, min_value=-10.0, max_value=30.0, step=0.1) / 100.0
    dividend_yield = st.number_input("Dividend Yield (%)", value=0.0, min_value=0.0, max_value=25.0, step=0.1) / 100.0
    sigma = iv / 100.0
    early_adjust = st.checkbox("Apply American early-exercise approximation", value=False)

    st.markdown("---")
    st.header("Option Snapshot")
    call_metrics = bs_greeks_dividend(
        spot,
        strike,
        dte,
        sigma,
        rate,
        dividend_yield=dividend_yield,
        option_type="call",
        american_adjustment=early_adjust,
    )
    put_metrics = bs_greeks_dividend(
        spot,
        strike,
        dte,
        sigma,
        rate,
        dividend_yield=dividend_yield,
        option_type="put",
        american_adjustment=early_adjust,
    )
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
        "Iron Butterfly",
        "Calendar Spread",
        "Long Straddle",
        "Long Strangle",
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
    <div class="premium-hero">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <div class="precision-banner">Trade intelligence</div>
                <h2 style="margin:0.25rem 0 0.35rem; font-size:2rem;">Options workstation</h2>
            </div>
            <div>
                <span class="premium-pill">Black-Scholes-Merton</span>
                <span class="premium-pill">1st & 2nd Order Greeks</span>
                <span class="premium-pill">IV Solver</span>
                <span class="premium-pill">Multi-Leg Payoffs</span>
                <span class="premium-pill">Portfolio Risk</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='score-card'><div style='color:#a5f3fc; font-size:0.76rem; letter-spacing:0.12em; text-transform:uppercase;'>Call Price</div><h3 style='margin:0.35rem 0 0;'>${call_metrics['price']:.2f}</h3><div style='color:#94a3b8; font-size:0.78rem; margin-top:0.25rem;'>ITM Prob: {call_metrics['prob_itm']*100:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='score-card'><div style='color:#a5f3fc; font-size:0.76rem; letter-spacing:0.12em; text-transform:uppercase;'>Put Price</div><h3 style='margin:0.35rem 0 0;'>${put_metrics['price']:.2f}</h3><div style='color:#94a3b8; font-size:0.78rem; margin-top:0.25rem;'>ITM Prob: {put_metrics['prob_itm']*100:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='score-card'><div style='color:#a5f3fc; font-size:0.76rem; letter-spacing:0.12em; text-transform:uppercase;'>Gamma</div><h3 style='margin:0.35rem 0 0;'>{call_metrics['gamma']:.5f}</h3><div style='color:#94a3b8; font-size:0.78rem; margin-top:0.25rem;'>Vanna: {call_metrics['vanna']:.4f}</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='score-card'><div style='color:#a5f3fc; font-size:0.76rem; letter-spacing:0.12em; text-transform:uppercase;'>Vega (per 1% IV)</div><h3 style='margin:0.35rem 0 0;'>${call_metrics['vega_pct']:.4f}</h3><div style='color:#94a3b8; font-size:0.78rem; margin-top:0.25rem;'>Raw Vega: ${call_metrics['vega']:.2f}</div></div>",
            unsafe_allow_html=True,
        )

moneyness_pct = (spot / strike - 1.0) * 100.0
trade_setup = st.container()
with trade_setup:
    st.markdown(
        """
        <div class="trade-setup">
            <div style="font-size:0.74rem; letter-spacing:0.12em; text-transform:uppercase; color:#a5f3fc; margin-bottom:0.5rem;">Market & Contract Configuration</div>
            <div class="trade-row">
                <div class="trade-box"><div style="color:#94a3b8; font-size:0.72rem;">Spot</div><div style="font-size:1.05rem; font-weight:700;">$%.2f</div></div>
                <div class="trade-box"><div style="color:#94a3b8; font-size:0.72rem;">Strike</div><div style="font-size:1.05rem; font-weight:700;">$%.2f</div></div>
                <div class="trade-box"><div style="color:#94a3b8; font-size:0.72rem;">Moneyness</div><div style="font-size:1.05rem; font-weight:700;">%+.2f%%</div></div>
                <div class="trade-box"><div style="color:#94a3b8; font-size:0.72rem;">DTE</div><div style="font-size:1.05rem; font-weight:700;">%d d</div></div>
                <div class="trade-box"><div style="color:#94a3b8; font-size:0.72rem;">IV</div><div style="font-size:1.05rem; font-weight:700;">%.2f%%</div></div>
                <div class="trade-box"><div style="color:#94a3b8; font-size:0.72rem;">Carry (r / q)</div><div style="font-size:1.05rem; font-weight:700;">%.2f%% / %.2f%%</div></div>
            </div>
        </div>
        """
        % (spot, strike, moneyness_pct, dte, iv, rate * 100.0, dividend_yield * 100.0),
        unsafe_allow_html=True,
    )

price_tab, theta_tab, strategy_tab, portfolio_tab, idea_tab, heatmap_tab = st.tabs(
    [
        "Live Greeks & IV Solver",
        "Theta Decay",
        "Strategy Builder",
        "Portfolio Dashboard",
        "Trade Idea",
        "Sensitivity Heatmap",
    ]
)

with price_tab:
    st.subheader("First- & Second-Order Greeks")
    metrics_df = pd.DataFrame(
        {
            "Price": [call_metrics["price"], put_metrics["price"]],
            "Intrinsic": [call_metrics["intrinsic"], put_metrics["intrinsic"]],
            "Extrinsic": [call_metrics["extrinsic"], put_metrics["extrinsic"]],
            "Delta": [call_metrics["delta"], put_metrics["delta"]],
            "Gamma": [call_metrics["gamma"], put_metrics["gamma"]],
            "Theta / day": [call_metrics["theta_daily"], put_metrics["theta_daily"]],
            "Vega (1% IV)": [call_metrics["vega_pct"], put_metrics["vega_pct"]],
            "Rho (1% r)": [call_metrics["rho_pct"], put_metrics["rho_pct"]],
            "Vanna": [call_metrics["vanna"], put_metrics["vanna"]],
            "Volga": [call_metrics["volga"], put_metrics["volga"]],
            "Charm / day": [call_metrics["charm_daily"], put_metrics["charm_daily"]],
            "Prob. ITM": [call_metrics["prob_itm"] * 100.0, put_metrics["prob_itm"] * 100.0],
        },
        index=["Call", "Put"],
    )
    st.dataframe(
        metrics_df.style.format(
            {
                "Price": "${:.2f}",
                "Intrinsic": "${:.2f}",
                "Extrinsic": "${:.2f}",
                "Delta": "{:.4f}",
                "Gamma": "{:.5f}",
                "Theta / day": "${:.4f}",
                "Vega (1% IV)": "${:.4f}",
                "Rho (1% r)": "${:.4f}",
                "Vanna": "{:.4f}",
                "Volga": "{:.4f}",
                "Charm / day": "{:.5f}",
                "Prob. ITM": "{:.1f}%",
            }
        ),
        use_container_width=True,
    )

    parity = put_call_parity_check(
        call_metrics["price"], put_metrics["price"], spot, strike, dte, rate, dividend_yield=dividend_yield
    )
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        st.metric("Call - Put Spread", f"${parity['call_minus_put']:.4f}")
    with pcol2:
        st.metric("Discounted Forward Spread", f"${parity['forward_discount_spread']:.4f}")
    with pcol3:
        st.metric("Put-Call Parity Residual", f"${parity['discrepancy']:+.6f}")

    st.markdown("---")
    gcol1, gcol2 = st.columns([1.35, 1.0])
    with gcol1:
        selected_greek = st.selectbox(
            "Greek Profile Across Spot",
            ["Delta", "Gamma", "Theta / day", "Vega (1% IV)", "Vanna", "Volga"],
        )
        greek_key_map = {
            "Delta": "delta",
            "Gamma": "gamma",
            "Theta / day": "theta_daily",
            "Vega (1% IV)": "vega_pct",
            "Vanna": "vanna",
            "Volga": "volga",
        }
        g_key = greek_key_map[selected_greek]
        spot_profile_grid = np.linspace(max(1.0, spot * 0.5), spot * 1.5, 120)
        call_curve = [
            bs_greeks_dividend(s, strike, dte, sigma, rate, dividend_yield, "call", early_adjust)[g_key]
            for s in spot_profile_grid
        ]
        put_curve = [
            bs_greeks_dividend(s, strike, dte, sigma, rate, dividend_yield, "put", early_adjust)[g_key]
            for s in spot_profile_grid
        ]
        st.plotly_chart(
            make_greeks_profile_figure(spot_profile_grid, call_curve, put_curve, selected_greek, spot, strike),
            use_container_width=True,
        )

    with gcol2:
        st.markdown("### Implied Volatility Solver")
        st.caption("Invert Black-Scholes-Merton to find implied volatility from an observed option price.")
        iv_opt_type = st.radio("Contract Type", ["Call", "Put"], horizontal=True, key="iv_solver_type")
        default_mkt_price = round(
            call_metrics["price"] if iv_opt_type == "Call" else put_metrics["price"], 2
        )
        observed_price = st.number_input(
            "Observed Market Option Price ($)",
            value=max(default_mkt_price, 0.05),
            min_value=0.01,
            max_value=float(max(spot * 2.0, 1000.0)),
            step=0.10,
        )
        try:
            solved_iv = implied_volatility(
                market_price=observed_price,
                S=spot,
                K=strike,
                T=dte,
                r=rate,
                dividend_yield=dividend_yield,
                option_type=iv_opt_type.lower(),
                american_adjustment=early_adjust,
            )
            solved_greeks = bs_greeks_dividend(
                spot,
                strike,
                dte,
                solved_iv,
                rate,
                dividend_yield=dividend_yield,
                option_type=iv_opt_type.lower(),
                american_adjustment=early_adjust,
            )
            st.success(f"Solved Implied Volatility: **{solved_iv * 100.0:.2f}%** (vs. input IV {iv:.2f}%)")
            iv_c1, iv_c2 = st.columns(2)
            with iv_c1:
                st.metric("Solved IV", f"{solved_iv * 100.0:.2f}%", f"{(solved_iv * 100.0 - iv):+.2f}%")
                st.metric("Implied Delta", f"{solved_greeks['delta']:.4f}")
            with iv_c2:
                st.metric("Implied Vega (1%)", f"${solved_greeks['vega_pct']:.4f}")
                st.metric("Implied Theta/day", f"${solved_greeks['theta_daily']:.4f}")
        except ValueError as exc:
            st.warning(str(exc))

    with st.expander("Explore 3D Option Surface (Spot × DTE)", expanded=False):
        surf_opt = st.radio("3D Surface Contract", ["Call", "Put"], horizontal=True, key="surf_opt")
        surf_metric_label = st.selectbox(
            "3D Surface Metric",
            ["Price", "Delta", "Gamma", "Theta / day", "Vega (1% IV)"],
            key="surf_metric",
        )
        surf_key = {
            "Price": "price",
            "Delta": "delta",
            "Gamma": "gamma",
            "Theta / day": "theta_daily",
            "Vega (1% IV)": "vega_pct",
        }[surf_metric_label]
        s_grid_3d = np.linspace(max(1.0, spot * 0.65), spot * 1.35, 35)
        dte_grid_3d = np.linspace(1, max(dte * 2, 60), 35)
        z_mat_3d = np.array(
            [
                [
                    bs_greeks_dividend(
                        s_val,
                        strike,
                        d_val,
                        sigma,
                        rate,
                        dividend_yield=dividend_yield,
                        option_type=surf_opt.lower(),
                        american_adjustment=early_adjust,
                    )[surf_key]
                    for d_val in dte_grid_3d
                ]
                for s_val in s_grid_3d
            ]
        )
        st.plotly_chart(
            make_surface_3d_figure(
                s_grid_3d,
                dte_grid_3d,
                z_mat_3d,
                z_label=surf_metric_label,
                title=f"3D {surf_opt} {surf_metric_label} Surface (Spot × DTE)",
            ),
            use_container_width=True,
        )

with theta_tab:
    sim_dte = st.slider("Simulated Starting DTE", min_value=5, max_value=365, value=max(int(dte), 30), step=1)
    sim_option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
    sim_records = simulate_theta_decay(
        S=spot,
        K=strike,
        sigma=sigma,
        r=rate,
        option_type=sim_option_type.lower(),
        dte_min=1,
        dte_max=sim_dte,
        steps=80,
        dividend_yield=dividend_yield,
        american_adjustment=early_adjust,
    )
    sim_days = [r["dte"] for r in sim_records]
    sim_values = [r["price"] for r in sim_records]
    sim_deltas = [r["delta"] for r in sim_records]
    sim_extrinsic = [r["extrinsic"] for r in sim_records]

    st.plotly_chart(
        make_theta_decay_figure(sim_days, sim_values, sim_deltas, sim_option_type, extrinsic=sim_extrinsic),
        use_container_width=True,
    )

    milestone_dtes = [d for d in [90, 60, 45, 30, 21, 14, 7, 3, 1] if d <= sim_dte]
    milestone_rows = []
    for m_dte in milestone_dtes:
        m = bs_greeks_dividend(
            spot,
            strike,
            m_dte,
            sigma,
            rate,
            dividend_yield=dividend_yield,
            option_type=sim_option_type.lower(),
            american_adjustment=early_adjust,
        )
        milestone_rows.append(
            {
                "DTE": m_dte,
                "Option Price": f"${m['price']:.2f}",
                "Extrinsic Value": f"${m['extrinsic']:.2f}",
                "Theta / Day": f"${m['theta_daily']:.4f}",
                "Delta": f"{m['delta']:.4f}",
                "Gamma": f"{m['gamma']:.5f}",
            }
        )
    if milestone_rows:
        st.markdown("#### Decay Acceleration Milestones")
        st.dataframe(pd.DataFrame(milestone_rows), use_container_width=True, hide_index=True)

with strategy_tab:
    strategy_dte = st.slider("Evaluation DTE (Current T+0 Horizon)", min_value=1, max_value=365, value=int(dte), step=1)
    selected_template = st.session_state.get("strategy_preset", "Bull Call Spread")
    st.info(f"Selected preset: **{selected_template}** — {classify_strategy_bias(selected_template)}")

    custom_mode = st.checkbox("Customize strategy legs manually", value=False)
    default_legs = strategy_preset(
        selected_template,
        spot,
        strike,
        dte=strategy_dte,
        sigma=sigma,
        rate=rate,
        dividend_yield=dividend_yield,
        american_adjustment=early_adjust,
    )

    if custom_mode:
        leg_count = st.slider("Number of legs", min_value=1, max_value=4, value=len(default_legs))
        legs = []
        for idx in range(leg_count):
            seed_leg = default_legs[idx] if idx < len(default_legs) else {
                "side": "Buy",
                "kind": "Call",
                "strike": strike,
                "premium": round(call_metrics["price"], 2),
                "quantity": 1,
                "dte": strategy_dte,
            }
            st.markdown(f"#### Leg {idx + 1}")
            cols = st.columns(6)
            with cols[0]:
                side = st.selectbox(
                    "Side",
                    ["Buy", "Sell"],
                    index=0 if seed_leg["side"] == "Buy" else 1,
                    key=f"side_{idx}",
                )
            with cols[1]:
                kind = st.selectbox(
                    "Type",
                    ["Call", "Put"],
                    index=0 if seed_leg["kind"] == "Call" else 1,
                    key=f"type_{idx}",
                )
            with cols[2]:
                leg_strike = st.number_input(
                    "Strike ($)",
                    value=float(seed_leg["strike"]),
                    min_value=1.0,
                    max_value=5000.0,
                    step=2.5,
                    key=f"strike_{idx}",
                )
            with cols[3]:
                premium = st.number_input(
                    "Entry Premium ($)",
                    value=float(seed_leg["premium"]),
                    min_value=0.0,
                    max_value=1000.0,
                    step=0.10,
                    key=f"premium_{idx}",
                )
            with cols[4]:
                quantity = st.number_input(
                    "Qty",
                    value=int(seed_leg["quantity"]),
                    min_value=1,
                    max_value=500,
                    step=1,
                    key=f"qty_{idx}",
                )
            with cols[5]:
                leg_dte = st.number_input(
                    "Leg DTE",
                    value=int(seed_leg["dte"]),
                    min_value=1,
                    max_value=3650,
                    step=1,
                    key=f"dte_{idx}",
                )
            legs.append(
                {
                    "side": side,
                    "kind": kind,
                    "strike": leg_strike,
                    "premium": premium,
                    "quantity": quantity,
                    "dte": leg_dte,
                }
            )
    else:
        legs = default_legs

    spot_range = np.linspace(max(0.5, spot * 0.45), max(spot * 1.65, strike * 1.65), 500)
    expiry_curve = strategy_expiry_curve(
        spot_range,
        legs,
        sigma=sigma,
        rate=rate,
        dividend_yield=dividend_yield,
        american_adjustment=early_adjust,
    )
    current_curve = strategy_current_curve(
        spot_range,
        legs,
        strategy_dte,
        sigma,
        rate,
        dividend_yield=dividend_yield,
        american_adjustment=early_adjust,
    )
    mid_dte = max(1.0, strategy_dte * 0.5)
    mid_curve = strategy_current_curve(
        spot_range,
        legs,
        mid_dte,
        sigma,
        rate,
        dividend_yield=dividend_yield,
        american_adjustment=early_adjust,
    )

    metrics = strategy_summary_metrics(
        spot_range,
        expiry_curve,
        spot=spot,
        sigma=sigma,
        dte=strategy_dte,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    score = strategy_score(metrics)
    net_greeks = strategy_greeks(
        spot,
        legs,
        strategy_dte,
        sigma,
        rate,
        dividend_yield=dividend_yield,
        american_adjustment=early_adjust,
    )

    payoff_fig = make_payoff_figure(
        spot_range,
        expiry_curve,
        current_curve,
        title=f"{selected_template if not custom_mode else 'Custom Strategy'}: Expiration vs. T+0 P/L",
        underlying_price=spot,
        strike=strike,
        breakevens=metrics.get("breakevens"),
        mid_curve=mid_curve,
        mid_label=f"Mid-Horizon ({int(round(mid_dte))}d DTE)",
    )
    st.plotly_chart(payoff_fig, use_container_width=True)

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Setup Score", f"{score:.1f} / 100")
    with s2:
        st.metric("Max Profit / sh", f"${metrics['max_profit']:.2f}")
    with s3:
        st.metric("Max Loss / sh", f"${abs(metrics['max_loss']):.2f}")
    with s4:
        st.metric("Break-Even(s)", _format_breakevens(metrics))
    with s5:
        st.metric("Prob. of Profit", f"{metrics['prob_profit']:.1f}%")

    st.markdown("#### Net Strategy Greeks (Per 1x Spread Unit)")
    g1, g2, g3, g4, g5 = st.columns(5)
    cost_label = "Net Debit" if net_greeks["net_premium"] >= 0 else "Net Credit"
    with g1:
        st.metric(cost_label, f"${abs(net_greeks['net_premium']):.2f}")
    with g2:
        st.metric("Net Delta", f"{net_greeks['delta']:+.4f}")
    with g3:
        st.metric("Net Gamma", f"{net_greeks['gamma']:+.5f}")
    with g4:
        st.metric("Net Theta / day", f"${net_greeks['theta_daily']:+.4f}")
    with g5:
        st.metric("Net Vega (1% IV)", f"${net_greeks['vega_pct']:+.4f}")

    strategy_df = pd.DataFrame(
        [
            {
                "Leg": i + 1,
                "Side": leg["side"],
                "Type": leg["kind"],
                "Strike ($)": leg["strike"],
                "Qty": leg["quantity"],
                "Entry Premium ($)": leg["premium"],
                "DTE": leg.get("dte", strategy_dte),
            }
            for i, leg in enumerate(legs)
        ]
    )
    st.dataframe(strategy_df, use_container_width=True, hide_index=True)

    ecol1, ecol2, ecol3 = st.columns(3)
    export_csv = pd.DataFrame(
        {
            "Spot": spot_range,
            "Expiration_PnL": expiry_curve,
            "Mid_Horizon_PnL": mid_curve,
            "Current_T0_PnL": current_curve,
        }
    )
    with ecol1:
        st.download_button(
            "Download Payoff CSV",
            data=export_csv.to_csv(index=False).encode("utf-8"),
            file_name="strategy_payoff.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ecol2:
        html_bytes = payoff_fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            "Download Interactive HTML Chart",
            data=html_bytes,
            file_name="strategy_payoff.html",
            mime="text/html",
            use_container_width=True,
        )
    with ecol3:
        try:
            png_buffer = io.BytesIO()
            payoff_fig.write_image(png_buffer, format="png", width=1400, height=800)
            st.download_button(
                "Download Payoff PNG",
                data=png_buffer.getvalue(),
                file_name="strategy_payoff.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception:
            st.caption("PNG export requires Kaleido + headless Chromium; use the Interactive HTML or Plotly toolbar camera icon.")

with portfolio_tab:
    st.subheader("Multi-Asset Options Portfolio Exposure")
    default_portfolio = pd.DataFrame(
        [
            {"symbol": "AAPL", "qty": 100, "delta": 0.58, "gamma": 0.020, "theta": -0.124, "vega": 0.180, "market_value": 15600.0},
            {"symbol": "MSFT", "qty": 80, "delta": 0.44, "gamma": 0.015, "theta": -0.091, "vega": 0.132, "market_value": 14880.0},
            {"symbol": "SPY", "qty": 60, "delta": 0.52, "gamma": 0.010, "theta": -0.073, "vega": 0.095, "market_value": 13200.0},
        ]
    )
    portfolio_df = st.data_editor(default_portfolio, num_rows="dynamic", use_container_width=True)

    clean_df = portfolio_df.copy()
    for numeric_col in ["qty", "delta", "gamma", "theta", "vega", "market_value"]:
        if numeric_col in clean_df.columns:
            clean_df[numeric_col] = pd.to_numeric(clean_df[numeric_col], errors="coerce").fillna(0.0)
        else:
            clean_df[numeric_col] = 0.0

    exposures = {
        "Net Delta": float(clean_df["qty"].mul(clean_df["delta"]).sum()),
        "Net Gamma": float(clean_df["qty"].mul(clean_df["gamma"]).sum()),
        "Net Theta": float(clean_df["qty"].mul(clean_df["theta"]).sum()),
        "Net Vega": float(clean_df["qty"].mul(clean_df["vega"]).sum()),
        "Notional": float(clean_df["market_value"].sum()),
    }
    p1, p2, p3, p4, p5 = st.columns(5)
    for col, label in zip([p1, p2, p3, p4], ["Net Delta", "Net Gamma", "Net Theta", "Net Vega"]):
        with col:
            st.metric(label, f"{exposures[label]:+,.2f}")
    with p5:
        st.metric("Portfolio Notional", f"${exposures['Notional']:,.2f}")

    st.plotly_chart(make_portfolio_exposure_figure(exposures), use_container_width=True)

    st.markdown("#### Portfolio Instant Shock Stress Matrix (2nd-Order Taylor Approximation)")
    spot_shocks_pct = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]
    vol_shocks_pts = [-5.0, 0.0, 5.0]
    stress_rows = []
    for ds_pct in spot_shocks_pct:
        dS = spot * (ds_pct / 100.0)
        row_data: dict[str, str] = {"Spot Move": f"{ds_pct:+.0f}% (${spot + dS:.1f})"}
        for dv_pts in vol_shocks_pts:
            pnl_est = (
                exposures["Net Delta"] * dS
                + 0.5 * exposures["Net Gamma"] * (dS**2)
                + exposures["Net Vega"] * dv_pts
            )
            row_data[f"IV {dv_pts:+.0f} pts"] = f"${pnl_est:+,.2f}"
        stress_rows.append(row_data)
    st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)

with idea_tab:
    strategy_name = st.session_state.get("strategy_preset", "Bull Call Spread")
    idea_spot_grid = np.linspace(max(0.5, spot * 0.45), max(spot * 1.65, strike * 1.65), 400)
    idea_legs = strategy_preset(
        strategy_name,
        spot,
        strike,
        dte=dte,
        sigma=sigma,
        rate=rate,
        dividend_yield=dividend_yield,
        american_adjustment=early_adjust,
    )
    idea_expiry = strategy_expiry_curve(
        idea_spot_grid,
        idea_legs,
        sigma=sigma,
        rate=rate,
        dividend_yield=dividend_yield,
        american_adjustment=early_adjust,
    )
    idea_metrics = strategy_summary_metrics(
        idea_spot_grid,
        idea_expiry,
        spot=spot,
        sigma=sigma,
        dte=dte,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    idea_greeks = strategy_greeks(
        spot,
        idea_legs,
        dte,
        sigma,
        rate,
        dividend_yield=dividend_yield,
        american_adjustment=early_adjust,
    )
    summary_text = trade_idea_summary(strategy_name, idea_metrics, spot, strike)
    st.markdown("### Trade Idea Summary")
    st.write(summary_text)
    idea_df = pd.DataFrame(
        {
            "Field": [
                "Strategy",
                "Market Bias",
                "Net Entry Debit / (Credit)",
                "Max Profit (per share)",
                "Max Loss (per share)",
                "Break-Even Price(s)",
                "Probability of Profit (PoP)",
                "Net Strategy Delta",
                "Net Strategy Theta / day",
                "Net Strategy Vega (1% IV)",
                "Setup Score",
            ],
            "Value": [
                strategy_name,
                classify_strategy_bias(strategy_name),
                f"${idea_greeks['net_premium']:+.2f}",
                f"${idea_metrics['max_profit']:.2f}",
                f"${abs(idea_metrics['max_loss']):.2f}",
                _format_breakevens(idea_metrics),
                f"{idea_metrics['prob_profit']:.1f}%",
                f"{idea_greeks['delta']:+.4f}",
                f"${idea_greeks['theta_daily']:+.4f}",
                f"${idea_greeks['vega_pct']:+.4f}",
                f"{strategy_score(idea_metrics):.1f} / 100",
            ],
        }
    )
    st.dataframe(idea_df, use_container_width=True, hide_index=True)

with heatmap_tab:
    hcol1, hcol2 = st.columns(2)
    with hcol1:
        style = st.radio("Heatmap Option", ["Call", "Put"], horizontal=True)
    with hcol2:
        heatmap_metric = st.selectbox(
            "Sensitivity Metric",
            ["Premium ($)", "Net P/L vs Current Price ($)", "Delta", "Gamma", "Theta / day ($)", "Vega per 1% IV ($)", "Vanna"],
        )

    spot_grid = np.linspace(max(1.0, spot * 0.55), spot * 1.45, 60)
    vol_shocks = np.linspace(-50.0, 50.0, 60)
    vol_grid = np.maximum(sigma * (1.0 + vol_shocks / 100.0), 0.005)
    entry_ref_price = call_metrics["price"] if style == "Call" else put_metrics["price"]

    metric_key_lookup = {
        "Premium ($)": ("price", True, "Plasma"),
        "Net P/L vs Current Price ($)": ("pnl", True, "RdYlGn"),
        "Delta": ("delta", False, "Viridis"),
        "Gamma": ("gamma", False, "Magma"),
        "Theta / day ($)": ("theta_daily", True, "Cividis"),
        "Vega per 1% IV ($)": ("vega_pct", True, "Plasma"),
        "Vanna": ("vanna", False, "RdBu"),
    }
    m_key, is_curr, cscale = metric_key_lookup[heatmap_metric]

    if m_key == "price":
        matrix = np.array(
            [
                [
                    bs_price_dividend(
                        s,
                        strike,
                        dte,
                        vol,
                        rate,
                        dividend_yield=dividend_yield,
                        option_type=style.lower(),
                        american_adjustment=early_adjust,
                    )
                    for vol in vol_grid
                ]
                for s in spot_grid
            ]
        )
    elif m_key == "pnl":
        matrix = np.array(
            [
                [
                    bs_price_dividend(
                        s,
                        strike,
                        dte,
                        vol,
                        rate,
                        dividend_yield=dividend_yield,
                        option_type=style.lower(),
                        american_adjustment=early_adjust,
                    )
                    - entry_ref_price
                    for vol in vol_grid
                ]
                for s in spot_grid
            ]
        )
    else:
        matrix = np.array(
            [
                [
                    bs_greeks_dividend(
                        s,
                        strike,
                        dte,
                        vol,
                        rate,
                        dividend_yield=dividend_yield,
                        option_type=style.lower(),
                        american_adjustment=early_adjust,
                    )[m_key]
                    for vol in vol_grid
                ]
                for s in spot_grid
            ]
        )

    heatmap_fig = make_heatmap_figure(
        spot_grid,
        vol_shocks,
        matrix,
        z_label=f"{style} {heatmap_metric}",
        title=f"{style} {heatmap_metric}: Spot Price vs. Volatility Shock (%)",
        colorscale=cscale,
        is_currency=is_curr,
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

st.markdown("---")
st.subheader("Model Summary")
summary = f"""Black-Scholes-Merton framework for spot=${spot:.2f}, strike=${strike:.2f}, DTE={dte}, IV={iv:.2f}%, r={rate * 100:.2f}%, dividend yield={dividend_yield * 100:.2f}%.

- Call price: ${call_metrics['price']:.2f} (Intrinsic: ${call_metrics['intrinsic']:.2f}, Extrinsic: ${call_metrics['extrinsic']:.2f}, ITM Prob: {call_metrics['prob_itm']*100:.1f}%)
- Put price:  ${put_metrics['price']:.2f} (Intrinsic: ${put_metrics['intrinsic']:.2f}, Extrinsic: ${put_metrics['extrinsic']:.2f}, ITM Prob: {put_metrics['prob_itm']*100:.1f}%)
- Delta:      {call_metrics['delta']:+.4f} (call) / {put_metrics['delta']:+.4f} (put)
- Gamma:      {call_metrics['gamma']:.5f} | Vanna: {call_metrics['vanna']:+.4f} | Volga: {call_metrics['volga']:+.4f}
- Theta/day:  ${call_metrics['theta_daily']:+.4f} (call) / ${put_metrics['theta_daily']:+.4f} (put)
- Vega (1%):  ${call_metrics['vega_pct']:.4f} (raw Vega: ${call_metrics['vega']:.4f})
- Rho (1%):   ${call_metrics['rho_pct']:+.4f} (call) / ${put_metrics['rho_pct']:+.4f} (put)
- American early-exercise approximation: {'On' if early_adjust else 'Off'}
"""
st.code(summary, language="text")

st.markdown(
    "<div style='text-align:center; color:#94a3b8; margin-top:16px;'>Built with Python, Streamlit, NumPy, SciPy, Pandas, and Plotly.</div>",
    unsafe_allow_html=True,
)