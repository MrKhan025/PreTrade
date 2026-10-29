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
    make_price_history_figure,
    make_iv_smile_figure,
    make_interactive_greeks_matrix,
    make_comparison_payoff_figure,
    make_pnl_waterfall_figure,
    make_expected_move_cone_figure,
)
from binomial import crr_price, crr_greeks
from monte_carlo import mc_price
from risk import expected_move, strategy_risk_metrics, portfolio_var_delta_normal, position_sizing_recommendation
from streaming import start_stream, stop_stream, stream_snapshot, apply_stream_price_to_quote, get_manager
from config import get_finnhub_api_key, finnhub_key_source, mask_key
from market_data import (
    load_market_defaults,
    fetch_quote,
    fetch_option_chain_summary,
    fetch_price_history,
    fetch_quotes_batch,
    fetch_intraday_bars,
)
from pdf_export import build_trade_ticket_pdf

st.set_page_config(page_title="PreTrade", page_icon="📈", layout="wide")

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "filename": "pretrade_chart", "scale": 2},
    "responsive": True,
}

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


st.title("PreTrade")
st.caption("See the trade before you take it • strategy payoffs • Greeks • IV • live risk")

with st.sidebar:
    st.header("Real-Time Market Data")
    default_sym = st.session_state.get("ticker_symbol", "AAPL")
    ticker_input = st.text_input(
        "Primary ticker",
        value=default_sym,
        help="Equity / ETF ticker used for the pricing model (e.g. AAPL, SPY, TSLA)",
        key="primary_ticker_input",
    )

    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["AAPL", "MSFT", "SPY"]
    wl_raw = st.text_input(
        "Watchlist (comma-separated)",
        value=", ".join(st.session_state.watchlist),
        help="Symbols polled on each refresh",
        key="watchlist_input",
    )
    st.session_state.watchlist = [s.strip().upper() for s in wl_raw.split(",") if s.strip()]

    st.markdown("##### WebSocket stream")
    ws_provider = st.selectbox(
        "Provider",
        ["demo", "finnhub"],
        format_func=lambda x: "Demo simulator (no key)" if x == "demo" else "Finnhub WebSocket (API key)",
        key="ws_provider",
        help="Demo = local random-walk around last quote. Finnhub = real trade ticks over WSS.",
    )
    finnhub_key = ""
    if ws_provider == "finnhub":
        resolved = get_finnhub_api_key(st.session_state.get("finnhub_key") or None)
        src_label = finnhub_key_source(st.session_state.get("finnhub_key") or None)
        st.caption(
            f"Key status: **{mask_key(resolved)}** · source: `{src_label}`  |  "
            "Priority: sidebar → Streamlit secrets → env → `.env` → `config.local.json`"
        )
        finnhub_key = st.text_input(
            "Finnhub API key (optional if already configured)",
            value="",
            type="password",
            help="Get a free key at https://finnhub.io/register — leave blank to use secrets/.env/env.",
            key="finnhub_key_input",
            placeholder="Paste key here or configure via .env / secrets",
        )
        # Persist typed key into session for resolution
        if finnhub_key.strip():
            st.session_state["finnhub_key"] = finnhub_key.strip()
        finnhub_key = st.session_state.get("finnhub_key") or finnhub_key
        with st.expander("How to add your Finnhub API key", expanded=not bool(resolved)):
            st.markdown(
                """
**Option A — Sidebar** (quick): paste the key in the field above.

**Option B — `.env` file** (recommended for local dev):
```bash
cp .env.example .env
# edit .env and set:
FINNHUB_API_KEY=your_key_here
```

**Option C — Environment variable:**
```bash
export FINNHUB_API_KEY=your_key_here
streamlit run app.py
```

**Option D — Streamlit secrets** (`.streamlit/secrets.toml`):
```toml
FINNHUB_API_KEY = "your_key_here"
# or nested:
[finnhub]
api_key = "your_key_here"
```

**Option E — `config.local.json`** (gitignored):
```json
{"FINNHUB_API_KEY": "your_key_here"}
```

Free keys: [finnhub.io/register](https://finnhub.io/register)
                """
            )
        if not get_finnhub_api_key(finnhub_key or None):
            st.warning("No Finnhub API key found yet. Configure one of the options above before starting the stream.")
    stream_symbols = list(dict.fromkeys(
        [ticker_input.strip().upper()] + list(st.session_state.watchlist)
        if ticker_input.strip() else list(st.session_state.watchlist)
    ))
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        start_ws = st.button("▶ Start stream", use_container_width=True, key="btn_ws_start")
    with sc2:
        stop_ws = st.button("■ Stop stream", use_container_width=True, key="btn_ws_stop")
    with sc3:
        use_stream_px = st.checkbox("Drive spot", value=True, key="use_stream_px", help="Push live stream price into Spot input")

    if start_ws:
        try:
            snap = start_stream(ws_provider, stream_symbols, api_key=get_finnhub_api_key(finnhub_key or None))
            st.session_state["ws_active"] = True
            if snap.get("status") == "error":
                st.error(snap.get("error") or "Stream error")
            else:
                st.toast(f"Stream started ({ws_provider}) · {', '.join(stream_symbols[:4])}")
        except Exception as exc:
            st.error(str(exc))
            st.session_state["ws_active"] = False
    if stop_ws:
        stop_stream()
        st.session_state["ws_active"] = False
        st.toast("Stream stopped")

    snap = stream_snapshot()
    st.session_state["ws_snapshot"] = snap
    status = snap.get("status", "idle")
    color = {"streaming": "#34d399", "connecting": "#fbbf24", "reconnecting": "#fbbf24", "error": "#f87171", "stopped": "#94a3b8", "idle": "#64748b"}.get(status, "#94a3b8")
    st.markdown(
        f'<div style="font-size:0.78rem;color:{color};margin:0.2rem 0 0.5rem;">● {status.upper()}'
        + (f' · {snap.get("provider", "")}' if snap.get("provider") else "")
        + (f' · {snap.get("error")}' if snap.get("error") else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    latest_ticks = snap.get("latest") or {}
    if latest_ticks:
        import pandas as _pd
        tick_rows = []
        for sym, t in latest_ticks.items():
            tick_rows.append({
                "Symbol": sym,
                "Stream $": t.get("price"),
                "Chg %": t.get("change_pct"),
                "Src": t.get("source"),
                "Time": (t.get("iso") or "")[-8:],
            })
        st.dataframe(_pd.DataFrame(tick_rows), use_container_width=True, hide_index=True, height=min(160, 40 + 28 * len(tick_rows)))

    # Faster UI refresh while streaming so ticks appear live
    c_ref1, c_ref2 = st.columns(2)
    with c_ref1:
        auto_refresh = st.checkbox(
            "Auto-refresh",
            value=bool(st.session_state.get("ws_active")) or False,
            key="auto_refresh",
            help="Keep on while streaming so the UI picks up new ticks.",
        )
    with c_ref2:
        default_idx = 0 if st.session_state.get("ws_active") else 1
        refresh_sec = st.selectbox(
            "Interval (s)",
            [5, 10, 15, 30, 60, 120],
            index=default_idx,
            key="refresh_sec",
        )

    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            count = st_autorefresh(interval=int(refresh_sec) * 1000, key="gl_autorefresh")
            st.caption(f"⏱ Live polling every {refresh_sec}s (tick #{count})")
        except ImportError:
            st.caption(
                f"⏱ Enable seamless auto-refresh: `pip install streamlit-autorefresh` — "
                f"or click **Refresh now** every {refresh_sec}s."
            )

    fetch_clicked = st.button("Refresh now", use_container_width=True, type="primary", key="btn_refresh_live")

    if "live_defaults" not in st.session_state:
        st.session_state.live_defaults = None
        st.session_state.ticker_symbol = default_sym
        st.session_state.watchlist_quotes = []
        st.session_state.last_refresh_utc = None

    should_fetch = bool(fetch_clicked) or bool(auto_refresh and ticker_input.strip())

    if should_fetch and ticker_input.strip():
        with st.spinner(f"Updating {ticker_input.upper()} + watchlist…"):
            try:
                st.session_state.live_defaults = load_market_defaults(ticker_input.strip())
                st.session_state.ticker_symbol = ticker_input.strip().upper()
                syms = list(dict.fromkeys([st.session_state.ticker_symbol] + list(st.session_state.watchlist)))
                st.session_state.watchlist_quotes = fetch_quotes_batch(syms)
                from datetime import datetime, timezone as _tz
                st.session_state.last_refresh_utc = datetime.now(_tz.utc).strftime("%H:%M:%S UTC")
            except Exception as exc:
                st.error(f"Market data error: {exc}")

    live = st.session_state.live_defaults
    if live:
        chg = live.get("change_pct")
        chg_str = f" ({chg:+.2f}%)" if chg is not None else ""
        ts = st.session_state.get("last_refresh_utc") or live.get("as_of", "—")
        st.success(
            f"**{live.get('name', live.get('symbol'))}**  ·  "
            f"${live['spot']:.2f}{chg_str}  ·  IV≈{live['iv']:.1f}%  ·  {ts}"
        )
        if live.get("chain", {}).get("error"):
            st.warning(live["chain"]["error"])
        elif live.get("chain", {}).get("nearest_expiry"):
            st.caption(f"Nearest expiry: {live['chain']['nearest_expiry']} ({live['dte']} DTE)")

    wq = st.session_state.get("watchlist_quotes") or []
    if wq:
        import pandas as _pd
        rows = []
        for q in wq:
            rows.append({
                "Symbol": q.get("symbol"),
                "Last": q.get("price"),
                "Chg %": q.get("change_pct"),
                "Div %": (float(q["dividend_yield"]) * 100.0) if q.get("dividend_yield") is not None else None,
            })
        wdf = _pd.DataFrame(rows)
        st.dataframe(
            wdf,
            use_container_width=True,
            hide_index=True,
            height=min(200, 48 + 28 * max(len(rows), 1)),
        )

    apply_live = st.checkbox(
        "Sync model inputs from live quote",
        value=True,
        help="Seed Spot / IV / DTE / Rate / Dividend from the last refresh (still editable).",
        key="apply_live",
    )

    st.markdown("---")
    st.header("Market Inputs")

    _spot0 = 100.0
    _strike0 = 100.0
    _dte0 = 30
    _iv0 = 20.0
    _rate0 = 4.5
    _div0 = 0.0
    if live and apply_live:
        _spot0 = float(live["spot"])
        _iv0 = float(live["iv"])
        _dte0 = int(live["dte"])
        _rate0 = float(live["rate"]) * 100.0
        _div0 = float(live["dividend_yield"]) * 100.0
        _strike0 = float(round(_spot0 / 5) * 5) if _spot0 >= 20 else float(round(_spot0))
        if should_fetch:
            st.session_state["inp_spot"] = _spot0
            st.session_state["inp_iv"] = _iv0
            st.session_state["inp_dte"] = _dte0
            st.session_state["inp_rate"] = _rate0
            st.session_state["inp_div"] = _div0
            st.session_state["inp_strike"] = _strike0

    # Overlay WebSocket stream price onto spot when enabled
    if st.session_state.get("use_stream_px"):
        _sym = (st.session_state.get("ticker_symbol") or ticker_input or "").strip().upper()
        _tick = (st.session_state.get("ws_snapshot") or {}).get("latest", {}).get(_sym)
        if _tick and _tick.get("price"):
            st.session_state["inp_spot"] = float(_tick["price"])
            _spot0 = float(_tick["price"])

    if "inp_spot" not in st.session_state:
        st.session_state["inp_spot"] = _spot0
    if "inp_strike" not in st.session_state:
        st.session_state["inp_strike"] = _strike0
    if "inp_dte" not in st.session_state:
        st.session_state["inp_dte"] = _dte0
    if "inp_iv" not in st.session_state:
        st.session_state["inp_iv"] = _iv0
    if "inp_rate" not in st.session_state:
        st.session_state["inp_rate"] = _rate0
    if "inp_div" not in st.session_state:
        st.session_state["inp_div"] = _div0

    spot = st.number_input("Spot Price ($)", min_value=1.0, max_value=50000.0, step=0.5, key="inp_spot")
    strike = st.number_input("Strike Price ($)", min_value=1.0, max_value=50000.0, step=0.5, key="inp_strike")
    dte = st.number_input("Days to Expiry (DTE)", min_value=1, max_value=3650, step=1, key="inp_dte")
    iv = st.number_input("Implied Volatility (%)", min_value=0.5, max_value=500.0, step=0.5, key="inp_iv")
    rate = st.number_input("Risk-Free Rate (%)", min_value=-10.0, max_value=30.0, step=0.1, key="inp_rate") / 100.0
    dividend_yield = st.number_input("Dividend Yield (%)", min_value=0.0, max_value=25.0, step=0.1, key="inp_div") / 100.0
    sigma = iv / 100.0
    early_adjust = st.checkbox("Apply American early-exercise approximation", value=False)

    if live and live.get("chain", {}).get("chain_preview") is not None:
        preview = live["chain"]["chain_preview"]
        if isinstance(preview, type(__import__("pandas").DataFrame())) and not preview.empty:
            with st.expander("Nearest option chain (ATM)", expanded=False):
                st.dataframe(preview, use_container_width=True, hide_index=True)

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

# Live feed status strip
if st.session_state.get("live_defaults") and st.session_state.get("last_refresh_utc"):
    _ld = st.session_state.live_defaults
    _chg = _ld.get("change_pct")
    _chg_html = f'<span style="color:{"#34d399" if (_chg or 0) >= 0 else "#f87171"};">{(_chg or 0):+.2f}%</span>' if _chg is not None else ""
    st.markdown(
        f"""<div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:center;margin:0.4rem 0 0.8rem;
        padding:0.55rem 0.9rem;border-radius:12px;background:rgba(15,23,42,0.85);
        border:1px solid rgba(34,211,238,0.25);font-size:0.85rem;">
        <span style="color:#22d3ee;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;font-size:0.7rem;">Live</span>
        <span style="color:#e2e8f0;font-weight:600;">{_ld.get("symbol")}</span>
        <span style="color:#f8fafc;font-size:1.05rem;font-weight:700;">${_ld.get("spot", 0):.2f}</span>
        {_chg_html}
        <span style="color:#94a3b8;">IV {_ld.get("iv", 0):.1f}%</span>
        <span style="color:#64748b;">Updated {_ld and st.session_state.get("last_refresh_utc")}</span>
        </div>""",
        unsafe_allow_html=True,
    )


# Stream-driven spot badge
_ws_snap = st.session_state.get("ws_snapshot") or {}
if _ws_snap.get("status") == "streaming":
    _sym = (st.session_state.get("ticker_symbol") or "").upper()
    _t = (_ws_snap.get("latest") or {}).get(_sym) or (next(iter((_ws_snap.get("latest") or {}).values()), None))
    if _t:
        st.markdown(
            f"""<div style="display:flex;gap:0.75rem;flex-wrap:wrap;align-items:center;margin:0 0 0.75rem;
            padding:0.45rem 0.85rem;border-radius:12px;background:rgba(16,185,129,0.08);
            border:1px solid rgba(52,211,153,0.35);font-size:0.82rem;">
            <span style="color:#34d399;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;font-size:0.68rem;">WS Stream</span>
            <span style="color:#e2e8f0;font-weight:600;">{_t.get("symbol")}</span>
            <span style="color:#f8fafc;font-size:1.05rem;font-weight:700;">${float(_t.get("price", 0)):.2f}</span>
            <span style="color:#94a3b8;">{_t.get("source")}</span>
            <span style="color:#64748b;">{(_t.get("iso") or "")[-8:]}</span>
            </div>""",
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

price_tab, theta_tab, strategy_tab, portfolio_tab, idea_tab, heatmap_tab, charts_tab = st.tabs(
    [
        "Live Greeks & IV Solver",
        "Theta Decay",
        "Strategy Builder",
        "Portfolio Dashboard",
        "Trade Idea",
        "Sensitivity Heatmap",
        "Charts Studio",
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


    with st.expander("Advanced Pricing Engines (Binomial Tree & Monte Carlo)", expanded=False):
        st.caption("Cross-validate the closed-form BSM price with a CRR binomial tree (American-capable) and a variance-reduced Monte Carlo engine.")
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Cox-Ross-Rubinstein Binomial Tree**")
            crr_style = st.radio("Tree style", ["european", "american"], horizontal=True, key="crr_style")
            crr_steps = st.slider("Tree steps", 50, 500, 200, 50, key="crr_steps")
            try:
                crr_call = crr_price(spot, strike, dte, sigma, rate, dividend_yield, "call", crr_style, crr_steps)
                crr_put = crr_price(spot, strike, dte, sigma, rate, dividend_yield, "put", crr_style, crr_steps)
                st.metric("CRR Call", f"${crr_call:.4f}", f"{crr_call - call_metrics['price']:+.4f} vs BSM")
                st.metric("CRR Put", f"${crr_put:.4f}", f"{crr_put - put_metrics['price']:+.4f} vs BSM")
                if crr_style == "american":
                    cg = crr_greeks(spot, strike, dte, sigma, rate, dividend_yield, "call", "american", min(crr_steps, 150))
                    st.caption(f"American Call Δ≈{cg['delta']:.4f}  Γ≈{cg['gamma']:.5f}  ν(1%)≈${cg['vega_pct']:.4f}")
            except Exception as exc:
                st.warning(str(exc))
        with a2:
            st.markdown("**Monte Carlo (Antithetic + Control Variate)**")
            n_paths = st.select_slider("Paths", options=[5_000, 10_000, 25_000, 50_000, 100_000], value=25_000, key="mc_paths")
            try:
                mc_c = mc_price(spot, strike, dte, sigma, rate, dividend_yield, "call", n_paths=n_paths, seed=42)
                mc_p = mc_price(spot, strike, dte, sigma, rate, dividend_yield, "put", n_paths=n_paths, seed=42)
                st.metric("MC Call", f"${mc_c['price']:.4f}", f"±{1.96*mc_c['stderr']:.4f} (95% CI)")
                st.metric("MC Put", f"${mc_p['price']:.4f}", f"±{1.96*mc_p['stderr']:.4f} (95% CI)")
                st.caption(f"BSM benchmarks — Call ${mc_c['bsm_benchmark']:.4f} | Put ${mc_p['bsm_benchmark']:.4f}")
            except Exception as exc:
                st.warning(str(exc))

        em = expected_move(spot, sigma, dte, confidence=0.68)
        st.info(
            f"**1σ Expected Move ({dte} DTE):** ±${em['expected_move']:.2f}  "
            f"→ range **${em['lower']:.2f} – ${em['upper']:.2f}** ({em['pct_of_spot']:.1f}% of spot)"
        )

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
    st.plotly_chart(payoff_fig, use_container_width=True, config=PLOTLY_CONFIG)

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
    # Advanced risk-neutral metrics
    try:
        rm = strategy_risk_metrics(spot, legs, sigma, rate, float(dte), dividend_yield)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("RN Expected P/L", f"${rm['expected_pnl']:+.3f}")
        with r2:
            st.metric("Reward / Risk", f"{rm['reward_risk']:.2f}x")
        with r3:
            st.metric("Max Profit / Max Loss", f"${rm['max_profit']:.2f} / ${rm['max_loss']:.2f}")
    except Exception:
        pass


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

    st.plotly_chart(make_portfolio_exposure_figure(exposures), use_container_width=True, config=PLOTLY_CONFIG)

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

    st.markdown("#### 1-Day 95% VaR (Delta-Gamma-Vega Approximation)")
    try:
        var_res = portfolio_var_delta_normal(
            net_delta=exposures["Net Delta"],
            net_gamma=exposures["Net Gamma"],
            net_vega=exposures["Net Vega"],
            spot=spot,
            sigma=sigma,
            horizon_days=1.0,
            confidence=0.95,
        )
        v1, v2, v3 = st.columns(3)
        with v1:
            st.metric("95% 1-Day VaR", f"${var_res['var']:,.2f}")
        with v2:
            st.metric("Expected P/L (convexity)", f"${var_res['expected_pnl']:+,.2f}")
        with v3:
            st.metric("P/L Std Dev", f"${var_res['pnl_std']:,.2f}")
    except Exception as exc:
        st.caption(f"VaR unavailable: {exc}")


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

    st.markdown("#### Position Sizing & Risk-Budget Calculator (100x Contract Multiplier)")
    ps1, ps2 = st.columns(2)
    with ps1:
        acct_size = st.number_input("Account Equity ($)", min_value=500.0, max_value=10_000_000.0, value=25_000.0, step=1_000.0, key="ps_acct")
    with ps2:
        risk_budget_pct = st.slider("Max Risk per Trade (% of Account)", min_value=0.25, max_value=10.0, value=2.0, step=0.25, key="ps_risk_pct")
    sizing = position_sizing_recommendation(
        account_size=acct_size,
        risk_pct=risk_budget_pct,
        max_loss_per_share=abs(idea_metrics["max_loss"]),
        max_profit_per_share=idea_metrics["max_profit"],
        prob_profit=idea_metrics["prob_profit"],
    )
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Recommended Size", f"{sizing['recommended_contracts']} contracts")
    with k2:
        st.metric("Risk Budget ($)", f"${sizing['risk_budget_dollars']:,.2f}")
    with k3:
        st.metric("Capital at Risk", f"${sizing['total_dollar_risk']:,.2f}", f"${sizing['loss_per_contract']:,.0f} / contract")
    with k4:
        st.metric("Half-Kelly Allocation", f"{sizing['half_kelly_pct']:.2f}%")

    st.markdown("---")
    st.subheader("Export Trade Ticket (PDF)")
    st.caption("Download a one-page professional PDF with market snapshot, full Greeks table, and strategy summary.")
    notes = st.text_area("Optional notes for the ticket", value="", max_chars=500, height=70, key="pdf_notes")
    try:
        pdf_bytes = build_trade_ticket_pdf(
            spot=spot,
            strike=strike,
            dte=int(dte),
            iv=iv,
            rate=rate,
            dividend_yield=dividend_yield,
            call_metrics=call_metrics,
            put_metrics=put_metrics,
            strategy_name=strategy_name,
            strategy_metrics=idea_metrics,
            legs=idea_legs,
            symbol=st.session_state.get("ticker_symbol"),
            notes=notes or summary_text,
        )
        fname = f"pretrade_ticket_{st.session_state.get('ticker_symbol', 'manual')}_{strategy_name.replace(' ', '_').lower()}.pdf"
        st.download_button(
            label="Download PDF Trade Ticket",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )
    except Exception as exc:
        st.warning(f"PDF generation failed: {exc}")

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
    st.plotly_chart(heatmap_fig, use_container_width=True, config=PLOTLY_CONFIG)

with charts_tab:
    st.subheader("Interactive Charts Studio")
    st.caption(
        "Candlestick history, IV smile, multi-Greek profiles with toggle buttons, "
        "and side-by-side strategy payoff comparison — all fully interactive (zoom, pan, spikes, download)."
    )
    studio_mode = st.radio(
        "Chart mode",
        ["Price History", "IV Smile", "Greeks Matrix", "Strategy Comparison", "P/L Waterfall", "Volatility Cone"],
        horizontal=True,
        key="studio_mode",
    )

    if studio_mode == "Price History":
        hist_sym = st.session_state.get("ticker_symbol") or "AAPL"
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            hist_ticker = st.text_input("History ticker", value=hist_sym, key="hist_ticker")
        with c2:
            rt_mode = st.selectbox("Resolution", ["Daily", "Intraday 1m", "Intraday 5m"], index=0, key="hist_res")
        with c3:
            period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"], index=4, key="hist_period")
        with c4:
            show_ma = st.checkbox("MAs (20/50)", value=True, key="hist_ma")
        try:
            if rt_mode.startswith("Intraday"):
                interval = "1m" if "1m" in rt_mode else "5m"
                # yfinance intraday limited to short periods
                iperiod = period if period in ("1d", "5d") else "5d"
                hist = fetch_intraday_bars(hist_ticker, interval=interval, period=iperiod)
            else:
                hist = fetch_price_history(hist_ticker, period=period if period not in ("1d", "5d") else "1mo", interval="1d")
            if hist.empty:
                st.warning("No price history returned for this ticker/period.")
            else:
                overlays = None
                if show_ma and "Close" in hist.columns:
                    overlays = {
                        "MA 20": hist["Close"].rolling(20).mean(),
                        "MA 50": hist["Close"].rolling(50).mean(),
                    }
                fig = make_price_history_figure(
                    dates=hist["Date"] if "Date" in hist.columns else hist.index,
                    opens=hist["Open"],
                    highs=hist["High"],
                    lows=hist["Low"],
                    closes=hist["Close"],
                    volumes=hist["Volume"] if "Volume" in hist.columns else None,
                    symbol=hist_ticker.upper(),
                    overlays=overlays,
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                # Export
                html = fig.to_html(include_plotlyjs="cdn")
                st.download_button(
                    "Download interactive HTML",
                    data=html.encode("utf-8"),
                    file_name=f"{hist_ticker.upper()}_history.html",
                    mime="text/html",
                )
        except Exception as exc:
            st.error(f"Could not load history: {exc}")

    elif studio_mode == "IV Smile":
        st.caption("Uses the nearest live option chain when a ticker has been fetched; otherwise builds a synthetic smile from the BSM inputs.")
        live = st.session_state.get("live_defaults")
        if live and live.get("chain", {}).get("chain_preview") is not None:
            preview = live["chain"]["chain_preview"]
            try:
                import pandas as _pd
                if isinstance(preview, _pd.DataFrame) and not preview.empty and "impliedVolatility" in preview.columns:
                    calls = preview[preview.get("type", "Call") == "Call"] if "type" in preview.columns else preview
                    puts = preview[preview["type"] == "Put"] if "type" in preview.columns else None
                    strikes = calls["strike"].astype(float).tolist()
                    call_ivs = calls["impliedVolatility"].astype(float).tolist()
                    put_ivs = puts["impliedVolatility"].astype(float).tolist() if puts is not None and not puts.empty else None
                    # Align put strikes if needed — simple same-order use
                    fig = make_iv_smile_figure(strikes, call_ivs, put_ivs, spot=spot, title=f"IV Smile — {live.get('symbol', '')} nearest expiry")
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                else:
                    st.info("Chain preview has no IV column; showing synthetic smile.")
                    raise ValueError("no iv")
            except Exception:
                # synthetic
                k_grid = np.linspace(spot * 0.7, spot * 1.3, 25)
                # simple sticky-delta-ish skew: lower strikes higher IV
                base = sigma
                synth = [base * (1.0 + 0.35 * (spot - k) / spot) for k in k_grid]
                fig = make_iv_smile_figure(k_grid, synth, None, spot=spot, title="Synthetic IV Smile (skew illustration)")
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            k_grid = np.linspace(spot * 0.7, spot * 1.3, 25)
            base = sigma
            synth = [base * (1.0 + 0.35 * (spot - k) / spot) for k in k_grid]
            fig = make_iv_smile_figure(k_grid, synth, None, spot=spot, title="Synthetic IV Smile (fetch a ticker for live chain)")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            st.info("Fetch a live ticker in the sidebar to plot the real option-chain IV smile.")

    elif studio_mode == "Greeks Matrix":
        st.caption("Toggle individual Greeks with the button bar; drag the rangeslider to zoom into moneyness regions.")
        opt_side = st.radio("Contract", ["Call", "Put"], horizontal=True, key="studio_greek_side")
        s_grid = np.linspace(max(1.0, spot * 0.55), spot * 1.45, 160)
        curves = {}
        key_map = {
            "Price": "price",
            "Delta": "delta",
            "Gamma": "gamma",
            "Theta / day": "theta_daily",
            "Vega (1% IV)": "vega_pct",
            "Vanna": "vanna",
            "Volga": "volga",
        }
        for label, key in key_map.items():
            curves[label] = [
                bs_greeks_dividend(
                    float(s), strike, dte, sigma, rate,
                    dividend_yield=dividend_yield,
                    option_type=opt_side.lower(),
                    american_adjustment=early_adjust,
                )[key]
                for s in s_grid
            ]
        fig = make_interactive_greeks_matrix(s_grid, curves, spot, strike, title=f"{opt_side} Greeks vs Spot")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        html = fig.to_html(include_plotlyjs="cdn")
        st.download_button(
            "Download interactive HTML",
            data=html.encode("utf-8"),
            file_name=f"greeks_matrix_{opt_side.lower()}.html",
            mime="text/html",
            key="dl_greeks_html",
        )

    elif studio_mode == "Strategy Comparison":
        st.caption("Overlay expiry P/L for multiple presets under the current market inputs.")
        compare_names = st.multiselect(
            "Strategies to compare",
            [
                "Bull Call Spread", "Bear Put Spread", "Long Butterfly", "Iron Condor",
                "Iron Butterfly", "Long Straddle", "Long Strangle", "Long Call", "Long Put",
            ],
            default=["Iron Condor", "Long Straddle", "Bull Call Spread"],
            key="compare_strats",
        )
        if not compare_names:
            st.warning("Select at least one strategy.")
        else:
            grid = np.linspace(max(1.0, spot * 0.5), spot * 1.5, 300)
            curves = {}
            for name in compare_names:
                legs = strategy_preset(
                    name, spot, strike, dte=dte, sigma=sigma, rate=rate,
                    dividend_yield=dividend_yield, american_adjustment=early_adjust,
                )
                curves[name] = strategy_expiry_curve(
                    grid, legs, sigma=sigma, rate=rate,
                    dividend_yield=dividend_yield, american_adjustment=early_adjust,
                )
            fig = make_comparison_payoff_figure(grid, curves, spot, title="Strategy Payoff Comparison @ Expiry")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            html = fig.to_html(include_plotlyjs="cdn")
            st.download_button(
                "Download interactive HTML",
                data=html.encode("utf-8"),
                file_name="strategy_comparison.html",
                mime="text/html",
                key="dl_compare_html",
            )

    elif studio_mode == "P/L Waterfall":
        st.caption("Decompose estimated option P/L into Delta, Gamma, Theta, Vega, and Vanna contributions.")
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            wf_side = st.selectbox("Option Type", ["Call", "Put"], key="wf_side")
        with w2:
            wf_ds_pct = st.slider("Spot Move (%)", -20.0, 20.0, 5.0, 0.5, key="wf_ds")
        with w3:
            wf_div_pts = st.slider("IV Shift (vol pts)", -15.0, 15.0, 2.0, 0.5, key="wf_div")
        with w4:
            wf_days = st.slider("Days Elapsed", 0, max(int(dte) - 1, 1), min(5, max(int(dte) - 1, 1)), 1, key="wf_days")
        wf_metrics = call_metrics if wf_side == "Call" else put_metrics
        d_spot = spot * (wf_ds_pct / 100.0)
        wf_components = {
            "Delta P/L": wf_metrics["delta"] * d_spot,
            "Gamma P/L": 0.5 * wf_metrics["gamma"] * (d_spot ** 2),
            "Theta P/L": wf_metrics["theta_daily"] * wf_days,
            "Vega P/L": wf_metrics["vega_pct"] * wf_div_pts,
            "Vanna P/L": wf_metrics["vanna"] * d_spot * (wf_div_pts / 100.0),
        }
        wf_fig = make_pnl_waterfall_figure(
            wf_components,
            title=f"{wf_side} P/L Attribution Waterfall (Spot {wf_ds_pct:+.1f}%, IV {wf_div_pts:+.1f} pts, {wf_days}d)",
        )
        st.plotly_chart(wf_fig, use_container_width=True, config=PLOTLY_CONFIG)

    elif studio_mode == "Volatility Cone":
        st.caption("Project 1σ (68%) and 2σ (95%) forward underlying price bands from today to expiration.")
        cone_fig = make_expected_move_cone_figure(
            spot=spot,
            sigma=sigma,
            dte=int(dte),
            rate=rate,
            dividend_yield=dividend_yield,
            strike=strike,
            title=f"Forward Volatility Cone ({dte} DTE, IV={iv:.1f}%)",
        )
        st.plotly_chart(cone_fig, use_container_width=True, config=PLOTLY_CONFIG)
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


def main() -> None:
    """CLI entrypoint to launch the Streamlit workstation via `pretrade` command."""
    import subprocess
    import sys
    from pathlib import Path

    script_path = str(Path(__file__).resolve())
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", script_path]))
