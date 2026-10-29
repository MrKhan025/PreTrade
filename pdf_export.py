"""
Generate professional single-page (or multi-page) PDF trade tickets and
Greeks summaries using ReportLab.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


# Dark-finance palette
NAVY = colors.HexColor("#0f172a")
SLATE = colors.HexColor("#1e293b")
CYAN = colors.HexColor("#22d3ee")
EMERALD = colors.HexColor("#34d399")
MUTED = colors.HexColor("#94a3b8")
WHITE = colors.white
LIGHT = colors.HexColor("#e2e8f0")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "GLTitle",
            parent=base["Title"],
            fontSize=18,
            textColor=NAVY,
            spaceAfter=4,
            alignment=TA_LEFT,
            fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "GLSub",
            parent=base["Normal"],
            fontSize=9,
            textColor=MUTED,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "GLH2",
            parent=base["Heading2"],
            fontSize=11,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "GLBody",
            parent=base["Normal"],
            fontSize=9,
            textColor=SLATE,
            leading=12,
        ),
        "small": ParagraphStyle(
            "GLSmall",
            parent=base["Normal"],
            fontSize=8,
            textColor=MUTED,
            leading=10,
        ),
        "metric": ParagraphStyle(
            "GLMetric",
            parent=base["Normal"],
            fontSize=10,
            textColor=NAVY,
            fontName="Helvetica-Bold",
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "GLMetricLabel",
            parent=base["Normal"],
            fontSize=7,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=2,
        ),
    }


def _kv_table(rows: list[tuple[str, str]], col_widths: list[float] | None = None) -> Table:
    data = [[Paragraph(k, _styles()["small"]), Paragraph(v, _styles()["body"])] for k, v in rows]
    t = Table(data, colWidths=col_widths or [2.2 * inch, 4.5 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                ("TEXTCOLOR", (0, 0), (-1, -1), SLATE),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    return t


def _metrics_row(items: list[tuple[str, str]]) -> Table:
    """Horizontal strip of label/value pairs."""
    cells = []
    for label, value in items:
        inner = [
            [Paragraph(value, _styles()["metric"])],
            [Paragraph(label, _styles()["metric_label"])],
        ]
        cells.append(Table(inner, colWidths=[1.35 * inch]))
    t = Table([cells], colWidths=[1.4 * inch] * len(items))
    t.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def build_trade_ticket_pdf(
    *,
    spot: float,
    strike: float,
    dte: int,
    iv: float,
    rate: float,
    dividend_yield: float,
    call_metrics: dict[str, float],
    put_metrics: dict[str, float],
    strategy_name: str = "Long Call",
    strategy_metrics: dict[str, Any] | None = None,
    legs: list[dict[str, Any]] | None = None,
    symbol: str | None = None,
    notes: str = "",
) -> bytes:
    """
    Build a professional PDF trade ticket / Greeks summary and return raw bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="PreTrade Trade Ticket",
        author="PreTrade",
    )
    styles = _styles()
    story: list[Any] = []

    title_sym = f" — {symbol}" if symbol else ""
    story.append(Paragraph(f"PreTrade Trade Ticket{title_sym}", styles["title"]))
    story.append(
        Paragraph(
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ·  "
            f"Black-Scholes-Merton  ·  Confidential",
            styles["subtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.2, color=CYAN, spaceAfter=8))

    # Market snapshot
    story.append(Paragraph("Market & Contract Snapshot", styles["h2"]))
    story.append(
        _kv_table(
            [
                ("Underlying / Spot", f"${spot:,.2f}"),
                ("Strike", f"${strike:,.2f}"),
                ("Days to Expiry", f"{dte} d"),
                ("Implied Volatility", f"{iv:.2f}%"),
                ("Risk-Free Rate", f"{rate * 100:.2f}%"),
                ("Dividend Yield", f"{dividend_yield * 100:.2f}%"),
                ("Moneyness", f"{(spot / strike - 1) * 100:+.2f}%"),
            ]
        )
    )

    # Call / Put metrics strip
    story.append(Paragraph("Option Valuation & Greeks", styles["h2"]))
    story.append(
        _metrics_row(
            [
                ("Call Price", f"${call_metrics.get('price', 0):.2f}"),
                ("Put Price", f"${put_metrics.get('price', 0):.2f}"),
                ("Call Δ", f"{call_metrics.get('delta', 0):+.3f}"),
                ("Gamma", f"{call_metrics.get('gamma', 0):.4f}"),
                ("Θ / day", f"${call_metrics.get('theta_daily', 0):+.3f}"),
                ("Vega (1%)", f"${call_metrics.get('vega_pct', 0):.3f}"),
            ]
        )
    )
    story.append(Spacer(1, 6))

    greeks_rows = [
        ["", "Call", "Put"],
        ["Price", f"${call_metrics.get('price', 0):.4f}", f"${put_metrics.get('price', 0):.4f}"],
        ["Intrinsic", f"${call_metrics.get('intrinsic', 0):.4f}", f"${put_metrics.get('intrinsic', 0):.4f}"],
        ["Extrinsic", f"${call_metrics.get('extrinsic', 0):.4f}", f"${put_metrics.get('extrinsic', 0):.4f}"],
        ["Delta", f"{call_metrics.get('delta', 0):+.4f}", f"{put_metrics.get('delta', 0):+.4f}"],
        ["Gamma", f"{call_metrics.get('gamma', 0):.5f}", f"{put_metrics.get('gamma', 0):.5f}"],
        ["Theta / day", f"${call_metrics.get('theta_daily', 0):+.4f}", f"${put_metrics.get('theta_daily', 0):+.4f}"],
        ["Vega (1% IV)", f"${call_metrics.get('vega_pct', 0):.4f}", f"${put_metrics.get('vega_pct', 0):.4f}"],
        ["Rho (1% r)", f"${call_metrics.get('rho_pct', 0):+.4f}", f"${put_metrics.get('rho_pct', 0):+.4f}"],
        ["Vanna", f"{call_metrics.get('vanna', 0):+.4f}", f"{put_metrics.get('vanna', 0):+.4f}"],
        ["Volga", f"{call_metrics.get('volga', 0):+.4f}", f"{put_metrics.get('volga', 0):+.4f}"],
        ["Prob. ITM", f"{call_metrics.get('prob_itm', 0) * 100:.1f}%", f"{put_metrics.get('prob_itm', 0) * 100:.1f}%"],
    ]
    gt = Table(greeks_rows, colWidths=[1.6 * inch, 2.4 * inch, 2.4 * inch])
    gt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (1, 1), (-1, -1), [WHITE, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(gt)

    # Strategy section
    if strategy_metrics is not None:
        story.append(Paragraph(f"Strategy: {strategy_name}", styles["h2"]))
        be = strategy_metrics.get("breakevens") or []
        be_str = " / ".join(f"${b:.2f}" for b in be) if be else "N/A"
        story.append(
            _metrics_row(
                [
                    ("Max Profit", f"${strategy_metrics.get('max_profit', 0):.2f}"),
                    ("Max Loss", f"${strategy_metrics.get('max_loss', 0):.2f}"),
                    ("PoP", f"{strategy_metrics.get('prob_profit', 0):.1f}%"),
                    ("Break-evens", be_str),
                ]
            )
        )
        if legs:
            story.append(Spacer(1, 4))
            leg_rows = [["Side", "Type", "Strike", "Premium", "Qty", "DTE"]]
            for leg in legs:
                leg_rows.append(
                    [
                        str(leg.get("side", "")),
                        str(leg.get("kind", "")),
                        f"${float(leg.get('strike', 0)):.2f}",
                        f"${float(leg.get('premium', 0)):.2f}",
                        str(leg.get("quantity", 1)),
                        str(leg.get("dte", dte)),
                    ]
                )
            lt = Table(leg_rows, colWidths=[0.9 * inch] * 6)
            lt.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), SLATE),
                        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#f8fafc")]),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(lt)

    if notes:
        story.append(Paragraph("Notes", styles["h2"]))
        story.append(Paragraph(notes.replace("\n", "<br/>"), styles["body"]))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.6, color=MUTED, spaceAfter=4))
    story.append(
        Paragraph(
            "PreTrade · Educational / research use only · Not investment advice · "
            "Model prices assume Black-Scholes-Merton dynamics and may differ from live markets.",
            styles["small"],
        )
    )

    doc.build(story)
    return buffer.getvalue()
