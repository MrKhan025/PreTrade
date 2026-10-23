from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def dark_layout(title: str) -> dict:
    return {
        "title": title,
        "template": "plotly_dark",
        "paper_bgcolor": "#0b1120",
        "plot_bgcolor": "#0b1120",
        "font": {"color": "#e2e8f0"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.12, "xanchor": "left", "x": 0.0},
        "margin": {"l": 24, "r": 24, "t": 50, "b": 30},
    }


def make_theta_decay_figure(days, values, deltas, option_type: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=days,
            y=values,
            mode="lines+markers",
            name=f"{option_type.title()} Price",
            line={"color": "#2dd4bf", "width": 3},
            marker={"size": 6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=deltas,
            mode="lines+markers",
            name=f"{option_type.title()} Delta",
            line={"color": "#60a5fa", "width": 3},
            marker={"size": 6},
            yaxis="y2",
        )
    )
    fig.update_layout(
        **dark_layout("Theta Decay Simulator"),
        xaxis_title="Days to Expiry",
        yaxis_title="Option Price",
        yaxis2={"title": "Delta", "overlaying": "y", "side": "right"},
    )
    return fig


def make_payoff_figure(spot_range, expiry_curve, current_curve, title: str, underlying_price: float, strike: float):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=spot_range, y=expiry_curve, mode="lines", name="Expiration P/L", line={"color": "#f59e0b", "width": 3})
    )
    fig.add_trace(
        go.Scatter(x=spot_range, y=current_curve, mode="lines", name="Current T+0", line={"color": "#22c55e", "width": 3})
    )
    fig.add_vline(x=underlying_price, line_dash="dash", line_color="#94a3b8", annotation_text="Spot")
    fig.add_vline(x=strike, line_dash="dot", line_color="#e2e8f0", annotation_text="Strike")
    fig.update_layout(
        **dark_layout(title),
        xaxis_title="Spot Price",
        yaxis_title="Profit / Loss",
        hovermode="x unified",
    )
    return fig


def make_heatmap_figure(spot_grid, volatility_shocks, matrix, z_label: str, title: str):
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=volatility_shocks,
            y=spot_grid,
            colorscale="Plasma",
            hovertemplate="Spot: %{y:.2f}<br>Vol Shock: %{x:.0f}%<br>Premium: $%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        **dark_layout(title),
        xaxis_title="Volatility Shock (%)",
        yaxis_title="Spot Price",
        coloraxis_colorbar={"title": z_label},
    )
    return fig
