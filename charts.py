from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def dark_layout(title: str) -> dict:
    """Return a consistent dark-mode Plotly layout dictionary."""
    return {
        "title": {"text": title, "font": {"size": 17, "color": "#f8fafc"}},
        "template": "plotly_dark",
        "paper_bgcolor": "#0b1120",
        "plot_bgcolor": "#0b1120",
        "font": {"color": "#e2e8f0", "family": "Inter, system-ui, sans-serif"},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(15, 23, 42, 0.65)",
        },
        "margin": {"l": 48, "r": 48, "t": 68, "b": 44},
    }


def make_theta_decay_figure(
    days: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    deltas: Sequence[float] | np.ndarray,
    option_type: str,
    extrinsic: Sequence[float] | np.ndarray | None = None,
) -> go.Figure:
    """Create a dual-axis Theta decay & Delta evolution figure across DTE."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=days,
            y=values,
            mode="lines+markers",
            name=f"{option_type.title()} Price ($)",
            line={"color": "#2dd4bf", "width": 3},
            marker={"size": 5},
            hovertemplate="DTE: %{x:.1f}d<br>Price: $%{y:.2f}<extra></extra>",
        )
    )
    if extrinsic is not None:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=extrinsic,
                mode="lines",
                name="Extrinsic (Time) Value ($)",
                line={"color": "#f59e0b", "width": 2, "dash": "dot"},
                hovertemplate="DTE: %{x:.1f}d<br>Extrinsic: $%{y:.2f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=deltas,
            mode="lines+markers",
            name=f"{option_type.title()} Delta",
            line={"color": "#60a5fa", "width": 2.5},
            marker={"size": 5},
            yaxis="y2",
            hovertemplate="DTE: %{x:.1f}d<br>Delta: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **dark_layout(f"Theta Decay & Delta Evolution ({option_type.title()})"),
        xaxis={"title": "Days to Expiry (DTE)", "autorange": "reversed", "gridcolor": "rgba(148,163,184,0.12)"},
        yaxis={"title": "Option Value ($)", "gridcolor": "rgba(148,163,184,0.12)"},
        yaxis2={
            "title": "Delta",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        hovermode="x unified",
    )
    return fig


def make_payoff_figure(
    spot_range: Sequence[float] | np.ndarray,
    expiry_curve: Sequence[float] | np.ndarray,
    current_curve: Sequence[float] | np.ndarray,
    title: str,
    underlying_price: float,
    strike: float,
    breakevens: Sequence[float] | None = None,
    mid_curve: Sequence[float] | np.ndarray | None = None,
    mid_label: str = "Mid-Term T+n",
) -> go.Figure:
    """Create an interactive multi-leg strategy payoff diagram with profit/loss zones and break-even markers."""
    spot_arr = np.asarray(spot_range, dtype=float)
    exp_arr = np.asarray(expiry_curve, dtype=float)
    curr_arr = np.asarray(current_curve, dtype=float)

    profit_fill = np.where(exp_arr >= 0.0, exp_arr, 0.0)
    loss_fill = np.where(exp_arr < 0.0, exp_arr, 0.0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=spot_arr,
            y=profit_fill,
            mode="lines",
            line={"width": 0},
            fill="tozeroy",
            fillcolor="rgba(34, 197, 94, 0.14)",
            name="Expiration Profit Zone",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spot_arr,
            y=loss_fill,
            mode="lines",
            line={"width": 0},
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.14)",
            name="Expiration Loss Zone",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=spot_arr,
            y=exp_arr,
            mode="lines",
            name="Expiration P/L",
            line={"color": "#f59e0b", "width": 3},
            hovertemplate="Spot: $%{x:.2f}<br>Expiry P/L: $%{y:.2f}<extra></extra>",
        )
    )
    if mid_curve is not None:
        fig.add_trace(
            go.Scatter(
                x=spot_arr,
                y=mid_curve,
                mode="lines",
                name=mid_label,
                line={"color": "#38bdf8", "width": 2.2, "dash": "dash"},
                hovertemplate=f"Spot: $%{{x:.2f}}<br>{mid_label}: $%{{y:.2f}}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=spot_arr,
            y=curr_arr,
            mode="lines",
            name="Current T+0",
            line={"color": "#22c55e", "width": 3},
            hovertemplate="Spot: $%{x:.2f}<br>Current T+0: $%{y:.2f}<extra></extra>",
        )
    )

    fig.add_hline(y=0.0, line_dash="solid", line_color="rgba(226, 232, 240, 0.35)", line_width=1.2)
    fig.add_vline(
        x=underlying_price,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text=f"Spot ${underlying_price:.1f}",
        annotation_position="top left",
    )
    if abs(strike - underlying_price) > 1e-3:
        fig.add_vline(
            x=strike,
            line_dash="dot",
            line_color="#e2e8f0",
            annotation_text=f"Ref Strike ${strike:.1f}",
            annotation_position="bottom right",
        )

    if breakevens:
        for idx, be in enumerate(breakevens, start=1):
            fig.add_vline(
                x=float(be),
                line_dash="dashdot",
                line_color="#a855f7",
                annotation_text=f"BE{idx}: ${float(be):.2f}",
                annotation_position="top right",
            )

    fig.update_layout(
        **dark_layout(title),
        xaxis={"title": "Underlying Spot Price ($)", "gridcolor": "rgba(148,163,184,0.12)"},
        yaxis={"title": "Profit / Loss per Share ($)", "gridcolor": "rgba(148,163,184,0.12)"},
        hovermode="x unified",
    )
    return fig


def make_heatmap_figure(
    spot_grid: Sequence[float] | np.ndarray,
    volatility_shocks: Sequence[float] | np.ndarray,
    matrix: Sequence[Sequence[float]] | np.ndarray,
    z_label: str,
    title: str,
    x_label: str = "Volatility Shock (%)",
    colorscale: str = "Plasma",
    is_currency: bool = True,
) -> go.Figure:
    """Create a 2D sensitivity heatmap across Spot Price and Volatility Shock (or DTE)."""
    value_fmt = "$%{z:.2f}" if is_currency else "%{z:.4f}"
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=volatility_shocks,
            y=spot_grid,
            colorscale=colorscale,
            colorbar={"title": z_label},
            hovertemplate=f"Spot: $%{{y:.2f}}<br>{x_label}: %{{x:.1f}}<br>{z_label}: {value_fmt}<extra></extra>",
        )
    )
    fig.update_layout(
        **dark_layout(title),
        xaxis={"title": x_label},
        yaxis={"title": "Spot Price ($)"},
    )
    return fig


def make_greeks_profile_figure(
    spot_range: Sequence[float] | np.ndarray,
    call_values: Sequence[float] | np.ndarray,
    put_values: Sequence[float] | np.ndarray,
    greek_name: str,
    underlying_price: float,
    strike: float,
) -> go.Figure:
    """Plot Call and Put Greek profiles across underlying Spot Price."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=call_values,
            mode="lines",
            name=f"Call {greek_name}",
            line={"color": "#2dd4bf", "width": 3},
            hovertemplate=f"Spot: $%{{x:.2f}}<br>Call {greek_name}: %{{y:.5f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=put_values,
            mode="lines",
            name=f"Put {greek_name}",
            line={"color": "#f43f5e", "width": 3},
            hovertemplate=f"Spot: $%{{x:.2f}}<br>Put {greek_name}: %{{y:.5f}}<extra></extra>",
        )
    )
    fig.add_vline(x=underlying_price, line_dash="dash", line_color="#94a3b8", annotation_text="Spot")
    fig.add_vline(x=strike, line_dash="dot", line_color="#e2e8f0", annotation_text="Strike")
    fig.update_layout(
        **dark_layout(f"{greek_name} Profile Across Spot Price"),
        xaxis={"title": "Spot Price ($)", "gridcolor": "rgba(148,163,184,0.12)"},
        yaxis={"title": greek_name, "gridcolor": "rgba(148,163,184,0.12)"},
        hovermode="x unified",
    )
    return fig


def make_surface_3d_figure(
    spot_grid: Sequence[float] | np.ndarray,
    dte_grid: Sequence[float] | np.ndarray,
    z_matrix: Sequence[Sequence[float]] | np.ndarray,
    z_label: str,
    title: str,
) -> go.Figure:
    """Create an interactive 3D surface plot across Spot Price and Days to Expiry (DTE)."""
    fig = go.Figure(
        data=[
            go.Surface(
                x=dte_grid,
                y=spot_grid,
                z=z_matrix,
                colorscale="Viridis",
                colorbar={"title": z_label},
                hovertemplate=f"DTE: %{{x:.0f}}d<br>Spot: $%{{y:.2f}}<br>{z_label}: %{{z:.4f}}<extra></extra>",
            )
        ]
    )
    layout = dark_layout(title)
    layout["margin"] = {"l": 10, "r": 10, "t": 56, "b": 10}
    fig.update_layout(
        **layout,
        scene={
            "xaxis": {"title": "DTE (Days)", "backgroundcolor": "#0b1120", "gridcolor": "#1e293b"},
            "yaxis": {"title": "Spot ($)", "backgroundcolor": "#0b1120", "gridcolor": "#1e293b"},
            "zaxis": {"title": z_label, "backgroundcolor": "#0b1120", "gridcolor": "#1e293b"},
        },
        height=520,
    )
    return fig


def make_portfolio_exposure_figure(exposures: dict[str, float]) -> go.Figure:
    """Create a dark-mode bar chart for net portfolio Greek exposures."""
    labels = ["Net Delta", "Net Gamma", "Net Theta", "Net Vega"]
    values = [float(exposures.get(k, 0.0)) for k in labels]
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in values]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"{v:+,.2f}" for v in values],
                textposition="outside",
                hovertemplate="%{x}: %{y:+,.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        **dark_layout("Portfolio Greek Exposures"),
        xaxis={"title": "Greek Dimension"},
        yaxis={"title": "Net Dollar / Share Exposure", "gridcolor": "rgba(148,163,184,0.12)"},
    )
    return fig


def make_price_history_figure(
    dates: Sequence[object] | np.ndarray,
    opens: Sequence[float] | np.ndarray,
    highs: Sequence[float] | np.ndarray,
    lows: Sequence[float] | np.ndarray,
    closes: Sequence[float] | np.ndarray,
    volumes: Sequence[float] | np.ndarray | None = None,
    symbol: str = "",
    overlays: Mapping[str, Sequence[float] | np.ndarray] | None = None,
) -> go.Figure:
    """Create an interactive dark-mode Candlestick + Volume chart with optional MA overlays."""
    has_vol = volumes is not None and len(volumes) == len(closes)
    if has_vol:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.74, 0.26],
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Candlestick(
            x=list(dates),
            open=list(opens),
            high=list(highs),
            low=list(lows),
            close=list(closes),
            name=f"{symbol or 'Price'} OHLC",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    palette = ["#38bdf8", "#f59e0b", "#a855f7", "#2dd4bf"]
    if overlays:
        for idx, (label, series) in enumerate(overlays.items()):
            fig.add_trace(
                go.Scatter(
                    x=list(dates),
                    y=list(series),
                    mode="lines",
                    name=label,
                    line={"color": palette[idx % len(palette)], "width": 1.8},
                ),
                row=1,
                col=1,
            )

    if has_vol and volumes is not None:
        opens_arr = np.asarray(opens, dtype=float)
        closes_arr = np.asarray(closes, dtype=float)
        vol_colors = [
            "rgba(34, 197, 94, 0.55)" if c >= o else "rgba(239, 68, 68, 0.55)"
            for o, c in zip(opens_arr, closes_arr)
        ]
        fig.add_trace(
            go.Bar(
                x=list(dates),
                y=list(volumes),
                name="Volume",
                marker_color=vol_colors,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Volume", gridcolor="rgba(148,163,184,0.12)", row=2, col=1)

    title_str = f"{symbol} Price Action & Volume" if symbol else "Underlying Price History"
    fig.update_layout(
        **dark_layout(title_str),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=540,
    )
    fig.update_yaxes(title_text="Price ($)", gridcolor="rgba(148,163,184,0.12)", row=1, col=1)
    return fig


def make_iv_smile_figure(
    strikes: Sequence[float] | np.ndarray,
    call_ivs: Sequence[float] | np.ndarray,
    put_ivs: Sequence[float] | np.ndarray | None = None,
    spot: float | None = None,
    title: str = "Implied Volatility Smile",
) -> go.Figure:
    """Create an interactive Implied Volatility Smile plot across option strikes."""
    strikes_arr = np.asarray(strikes, dtype=float)
    call_arr = np.asarray(call_ivs, dtype=float)
    if call_arr.size > 0 and np.nanmax(call_arr) <= 3.0:
        call_arr = call_arr * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strikes_arr,
            y=call_arr,
            mode="lines+markers",
            name="Call IV (%)",
            line={"color": "#2dd4bf", "width": 3},
            marker={"size": 7},
            hovertemplate="Strike: $%{x:.2f}<br>Call IV: %{y:.2f}%<extra></extra>",
        )
    )

    if put_ivs is not None and len(put_ivs) > 0:
        put_arr = np.asarray(put_ivs, dtype=float)
        if put_arr.size > 0 and np.nanmax(put_arr) <= 3.0:
            put_arr = put_arr * 100.0
        put_x = strikes_arr[: len(put_arr)]
        fig.add_trace(
            go.Scatter(
                x=put_x,
                y=put_arr,
                mode="lines+markers",
                name="Put IV (%)",
                line={"color": "#f43f5e", "width": 2.5, "dash": "dash"},
                marker={"size": 6},
                hovertemplate="Strike: $%{x:.2f}<br>Put IV: %{y:.2f}%<extra></extra>",
            )
        )

    if spot is not None and spot > 0:
        fig.add_vline(
            x=float(spot),
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"Spot ${float(spot):.2f}",
            annotation_position="top left",
        )

    fig.update_layout(
        **dark_layout(title),
        xaxis={"title": "Strike Price ($)", "gridcolor": "rgba(148,163,184,0.12)"},
        yaxis={"title": "Implied Volatility (%)", "gridcolor": "rgba(148,163,184,0.12)"},
        hovermode="x unified",
    )
    return fig


def make_interactive_greeks_matrix(
    spot_range: Sequence[float] | np.ndarray,
    curves: Mapping[str, Sequence[float] | np.ndarray],
    underlying_price: float,
    strike: float,
    title: str = "Greeks vs Spot",
) -> go.Figure:
    """Create an interactive multi-Greek chart with toggle buttons and range slider."""
    spot_arr = np.asarray(spot_range, dtype=float)
    palette = ["#2dd4bf", "#38bdf8", "#f59e0b", "#a855f7", "#f43f5e", "#22c55e", "#eab308"]

    fig = go.Figure()
    labels = list(curves.keys())
    for idx, (label, values) in enumerate(curves.items()):
        fig.add_trace(
            go.Scatter(
                x=spot_arr,
                y=np.asarray(values, dtype=float),
                mode="lines",
                name=label,
                line={"color": palette[idx % len(palette)], "width": 2.8},
                visible=True,
                hovertemplate=f"Spot: $%{{x:.2f}}<br>{label}: %{{y:.5f}}<extra></extra>",
            )
        )

    buttons = [
        {
            "label": "All",
            "method": "update",
            "args": [{"visible": [True] * len(labels)}, {"title": f"{title} — All Metrics"}],
        }
    ]
    for idx, label in enumerate(labels):
        vis = [i == idx for i in range(len(labels))]
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [{"visible": vis}, {"title": f"{title} — {label}"}],
            }
        )

    fig.add_vline(x=underlying_price, line_dash="dash", line_color="#94a3b8", annotation_text="Spot")
    if abs(strike - underlying_price) > 1e-3:
        fig.add_vline(x=strike, line_dash="dot", line_color="#e2e8f0", annotation_text="Strike")

    layout = dark_layout(title)
    layout["margin"] = {"l": 48, "r": 48, "t": 105, "b": 48}
    fig.update_layout(
        **layout,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.0,
                "y": 1.18,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "#1e293b",
                "font": {"color": "#f8fafc", "size": 11},
                "buttons": buttons,
            }
        ],
        xaxis={
            "title": "Spot Price ($)",
            "gridcolor": "rgba(148,163,184,0.12)",
            "rangeslider": {"visible": True, "thickness": 0.06},
        },
        yaxis={"title": "Sensitivity / Value", "gridcolor": "rgba(148,163,184,0.12)"},
        hovermode="x unified",
        height=540,
    )
    return fig


def make_comparison_payoff_figure(
    spot_range: Sequence[float] | np.ndarray,
    curves: Mapping[str, Sequence[float] | np.ndarray],
    underlying_price: float,
    title: str = "Strategy Payoff Comparison @ Expiry",
) -> go.Figure:
    """Overlay expiration P/L curves for multiple option strategies."""
    spot_arr = np.asarray(spot_range, dtype=float)
    palette = ["#22c55e", "#38bdf8", "#f59e0b", "#a855f7", "#f43f5e", "#2dd4bf", "#eab308", "#ec4899"]

    fig = go.Figure()
    for idx, (name, curve) in enumerate(curves.items()):
        fig.add_trace(
            go.Scatter(
                x=spot_arr,
                y=np.asarray(curve, dtype=float),
                mode="lines",
                name=name,
                line={"color": palette[idx % len(palette)], "width": 2.8},
                hovertemplate=f"Spot: $%{{x:.2f}}<br>{name}: $%{{y:.2f}}<extra></extra>",
            )
        )

    fig.add_hline(y=0.0, line_dash="solid", line_color="rgba(226, 232, 240, 0.35)", line_width=1.2)
    fig.add_vline(
        x=underlying_price,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text=f"Spot ${underlying_price:.1f}",
        annotation_position="top left",
    )

    fig.update_layout(
        **dark_layout(title),
        xaxis={"title": "Underlying Spot Price ($)", "gridcolor": "rgba(148,163,184,0.12)"},
        yaxis={"title": "Expiration Profit / Loss per Share ($)", "gridcolor": "rgba(148,163,184,0.12)"},
        hovermode="x unified",
    )
    return fig


def make_pnl_waterfall_figure(
    components: Mapping[str, float],
    title: str = "Option P/L Attribution Waterfall (Taylor Expansion)",
) -> go.Figure:
    """Create a dark-mode P/L attribution waterfall chart across Delta, Gamma, Theta, Vega, and Rho."""
    labels = list(components.keys())
    values = [float(components[k]) for k in labels]
    total = float(sum(values))

    fig = go.Figure(
        go.Waterfall(
            name="P/L Attribution",
            orientation="v",
            measure=["relative"] * len(labels) + ["total"],
            x=labels + ["Net Est. P/L"],
            y=values + [total],
            text=[f"${v:+,.2f}" for v in values] + [f"${total:+,.2f}"],
            textposition="outside",
            increasing={"marker": {"color": "#22c55e"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#38bdf8"}},
            connector={"line": {"color": "rgba(148, 163, 184, 0.35)"}},
        )
    )
    fig.update_layout(
        **dark_layout(title),
        xaxis={"title": "Risk Factor Component"},
        yaxis={"title": "P/L Contribution ($)", "gridcolor": "rgba(148,163,184,0.12)"},
    )
    return fig


def make_expected_move_cone_figure(
    spot: float,
    sigma: float,
    dte: int,
    rate: float = 0.045,
    dividend_yield: float = 0.0,
    strike: float | None = None,
    title: str = "Forward Volatility Cone (1σ & 2σ Expected Move)",
) -> go.Figure:
    """Create a forward 1-sigma (68%) and 2-sigma (95%) probability cone across DTE."""
    horizon = max(int(dte), 5)
    days = np.linspace(0.0, float(horizon), 100)
    years = days / 365.0
    drift = (float(rate) - float(dividend_yield)) * years
    std_t = float(sigma) * np.sqrt(years)

    forward = float(spot) * np.exp(drift)
    upper_1s = forward * np.exp(std_t)
    lower_1s = forward * np.exp(-std_t)
    upper_2s = forward * np.exp(2.0 * std_t)
    lower_2s = forward * np.exp(-2.0 * std_t)

    fig = go.Figure()
    # 2-sigma band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([upper_2s, lower_2s[::-1]]),
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.10)",
            line={"color": "rgba(56, 189, 248, 0.30)", "width": 1, "dash": "dot"},
            name="2σ Cone (95% CI)",
            hoverinfo="skip",
        )
    )
    # 1-sigma band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([upper_1s, lower_1s[::-1]]),
            fill="toself",
            fillcolor="rgba(45, 212, 191, 0.18)",
            line={"color": "#2dd4bf", "width": 2},
            name="1σ Cone (68% CI)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=forward,
            mode="lines",
            name="Forward Drift Path",
            line={"color": "#f8fafc", "width": 2.2, "dash": "dash"},
            hovertemplate="Day +%{x:.1f}<br>Forward: $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=upper_1s,
            mode="lines",
            name="+1σ Upper Bound",
            line={"color": "#2dd4bf", "width": 1.8},
            hovertemplate="Day +%{x:.1f}<br>+1σ: $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=lower_1s,
            mode="lines",
            name="-1σ Lower Bound",
            line={"color": "#f43f5e", "width": 1.8},
            hovertemplate="Day +%{x:.1f}<br>-1σ: $%{y:.2f}<extra></extra>",
        )
    )
    if strike is not None and strike > 0:
        fig.add_hline(
            y=float(strike),
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text=f"Strike ${float(strike):.2f}",
            annotation_position="top right",
        )

    fig.update_layout(
        **dark_layout(title),
        xaxis={"title": "Days Forward from Today", "gridcolor": "rgba(148,163,184,0.12)"},
        yaxis={"title": "Projected Spot Range ($)", "gridcolor": "rgba(148,163,184,0.12)"},
        hovermode="x unified",
    )
    return fig
