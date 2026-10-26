from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go


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

    # Shaded Profit / Loss zones at expiration
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

    # Core curves
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