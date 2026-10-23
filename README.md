# greek-lens

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Dark%20Charts-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Open%20Source-00C853)

A premium options analytics platform for pricing, multi-leg strategies, Greeks, and portfolio risk in one interactive dashboard.

</div>

## Hero

Greek Lens helps traders, quants, and students quickly understand the risk profile of an option or multi-leg trade before placing a position.

With live Black-Scholes pricing, interactive payoff curves, strategy scoring, and portfolio exposure views, Greek Lens turns complex derivatives analysis into a clean, decision-friendly workflow.

<div align="center">

<a href="https://github.com/MrKhan025/greek-lens"><img src="https://img.shields.io/badge/TRY%20IT%20NOW-Open%20App-00C853?style=for-the-badge" alt="Try it now" /></a>

</div>

## Product Highlights

### Pricing & Greeks Engine
- Live option pricing for Calls and Puts
- Exact Delta, Gamma, Theta, and Vega calculations
- Dividend-yield and early-exercise-aware approximation layer

### Strategy Builder
- Multi-leg strategy construction up to 4 legs
- Preset templates for spreads and advanced structures
- Expiration P/L and T+0 payoff comparison
- Strategy scoring, break-even detection, and max profit / max loss metrics

### Portfolio & Trade Idea Views
- Portfolio exposure dashboard with delta/gamma/theta/vega totals
- Trade idea summary panel with directional bias and setup score
- Simplified decision support for trade review and education

### Charting & Export
- Dark-mode Plotly charts for Greeks, decay, and sensitivity analysis
- CSV export for payoff data
- PNG export for screenshots or research notes

## Feature Grid

| Capability | Details |
| --- | --- |
| Greek Calculator | Spot, strike, DTE, IV, rate, dividend-adjusted pricing |
| Strategy Builder | Up to 4 option legs with Buy/Sell, Call/Put support |
| Payoff Curves | Expiration vs. current T+0 visualization |
| Theta Simulator | Dynamic DTE decay and Delta shift analysis |
| Heatmap Analysis | Spot vs. volatility premium sensitivity |
| Portfolio Dashboard | Net delta/gamma/theta/vega with notional view |
| Strategy Scoring | Max profit/loss, break-even, score overview |
| Export Tools | CSV + PNG snapshots |

## Screenshots

### Hero Dashboard

```text
[Dashboard Screenshot Placeholder]
```

### Strategy Builder

```text
[Strategy Payoff Screenshot Placeholder]
```

### Theta Decay

```text
[Theta Decay Screenshot Placeholder]
```

### Heatmap

```text
[Heatmap Screenshot Placeholder]
```

## Customer Testimonials

> “This is the fastest way I’ve seen to walk through Greeks, payoff structures, and risk without needing a full derivatives notebook.”
>
> — Quant Research Intern

> “The strategy builder makes it easy to compare a trade idea before I put a position on.”
>
> — Options Trader

> “The dark UI and payoff visuals make the educational value really clear.”
>
> — Finance Student

## Pricing

| Plan | Price | Includes |
| --- | --- | --- |
| Open Source | Free | Full local dashboard, all features, community support |
| Pro (optional future extension) | $29/mo | Team exports, saved strategies, premium templates |

## Project Structure

```text
greek-lens/
├── app.py
├── black_scholes.py
├── charts.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Quick Start

```bash
git clone https://github.com/MrKhan025/greek-lens.git
cd greek-lens
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Math Summary

Black-Scholes price for a European option under constant volatility and short rates:

- Call: C = S N(d1) - K e^{-rT} N(d2)
- Put: P = K e^{-rT} N(-d2) - S N(-d1)

where:

- d1 = [ln(S/K) + (r + 0.5 sigma^2)T] / (sigma sqrt(T))
- d2 = d1 - sigma sqrt(T)

Core Greeks:

- Delta = ∂V/∂S
- Gamma = ∂²V/∂S²
- Theta = ∂V/∂T
- Vega = ∂V/∂σ

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome. Please open an issue or pull request for improvements to the pricing model, strategy analytics, or product UX.

## Live Demo

Run locally:

```bash
streamlit run app.py
```

---

Built for the modern options desk.
