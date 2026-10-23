# greek-lens

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Dark%20Charts-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Open%20Source-00C853)

A premium options analytics dashboard for pricing, Greeks, payoff visualization, and advanced spread analysis.

</div>

## Hero

Greek Lens is an open-source interactive web app for options traders, quants, and students who want a transparent way to analyze:

- Black-Scholes option prices
- Greeks: Delta, Gamma, Theta, and Vega
- multi-leg payoff structures
- advanced spread definitions
- theta decay near expiration
- premium sensitivity under spot/volatility shocks

## Why It Stands Out

- Fast, browser-based experience with Streamlit
- Dark mode charts optimized for trading workflows
- Strategy presets for spreads and butterflies
- Real-time computations with exact Black-Scholes Greeks
- CSV and PNG export for research and presentation

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

## Features

- Live Greeks Calculator for spot, strike, DTE, IV, and risk-free rate
- Call/Put pricing with exact Delta, Gamma, Theta, and Vega
- Multi-leg strategy builder with up to four legs
- Strategy payoff curve at expiration vs. current T+0
- Strategy performance metrics: max profit, max loss, break-even
- Preset strategy cards for long/short calls, spreads, butterfly, iron condor, and calendar spreads
- Theta decay simulator over DTE
- 2D premium sensitivity heatmap across spot and volatility shocks
- CSV and PNG export for strategy analysis

## Math Summary

Black-Scholes price for a European call/put under constant volatility and short rates:

- Call: C = S N(d1) - K e^{-rT} N(d2)
- Put: P = K e^{-rT} N(-d2) - S N(-d1)

where:

- d1 = (ln(S/K) + (r + 0.5 sigma^2) T) / (sigma sqrt(T))
- d2 = d1 - sigma sqrt(T)

Core Greeks:

- Delta = ∂V/∂S
- Gamma = ∂²V/∂S²
- Theta = ∂V/∂T
- Vega = ∂V/∂σ

## Quick Start

```bash
git clone https://github.com/MrKhan025/greek-lens.git
cd greek-lens
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Screenshot Placeholders

```text
[Hero Dashboard Screenshot Placeholder]
[Strategy Builder Screenshot Placeholder]
[Theta Decay Screenshot Placeholder]
[Heatmap Screenshot Placeholder]
```

## Use Cases

- option education and Greeks walkthroughs
- strategy evaluation before execution
- portfolio risk review and trade planning
- teaching derivative mechanics and payoff structures

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome. Please open an issue or pull request for improvements to the pricing model, UI enhancements, or new strategy presets.

## Live Demo

Run locally:

```bash
streamlit run app.py
```

---

Built for the modern options desk.
