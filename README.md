# 📈 PreTrade — See the Trade Before You Take It

[![CI Tests](https://github.com/MrKhan025/PreTrade/actions/workflows/ci.yml/badge.svg)](https://github.com/MrKhan025/PreTrade/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Tests: 30 Passing](https://img.shields.io/badge/tests-30%20passing-22c55e?logo=pytest&logoColor=white)](https://github.com/MrKhan025/PreTrade/tree/main/tests)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docker Ready](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](https://github.com/MrKhan025/PreTrade/blob/main/Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-10b981.svg)](https://github.com/MrKhan025/PreTrade/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-38bdf8.svg)](https://github.com/MrKhan025/PreTrade/blob/main/CONTRIBUTING.md)

> **Institutional-grade pre-trade options workstation: Black-Scholes-Merton pricing, CRR Binomial trees, variance-reduced Monte Carlo, 1st- & 2nd-order Greeks, implied volatility solver, 12 multi-leg strategies, position sizing, live market data, WebSocket streaming, and PDF trade tickets in one dark-mode dashboard.**

**Tech Stack:** Python 3.10+ • Streamlit • NumPy • SciPy • Pandas • Plotly • ReportLab • yfinance  
**License:** MIT • **Test Suite:** 30/30 Unit Tests Passing (`pytest` & `unittest`)

## 🌟 Why PreTrade?

Options trading involves multiple moving parts—spot price, time decay, implied volatility, interest rates, and dividend yields all interact simultaneously. **PreTrade** turns derivatives math into a clear, visual decision workflow:

- **For Traders:** Test 12 multi-leg option strategies (Iron Condors, Butterflies, Straddles, Calendar Spreads, Verticals), inspect exact break-evens and Probability of Profit (PoP), size positions using your account risk budget and Half-Kelly criterion, and export 1-page PDF trade tickets before risking capital.
- **For Quants & Developers:** Import a dependency-light Python pricing library (`black_scholes.py`, `binomial.py`, `monte_carlo.py`, `risk.py`) with exact analytical 1st- and 2nd-order Greeks, an implied volatility root-finder, CRR American early-exercise trees, and Cornish-Fisher Delta-Gamma-Vega VaR.
- **For Students & Educators:** Visually explore how option premiums erode over time (Theta decay), how Delta and Gamma curve across strikes, and how volatility shifts impact option prices across 2D heatmaps, P/L attribution waterfalls, and 3D surfaces.

## ⚡ Quantitative & Real-Time Engines (v1.6)

| Engine | Module | Capability |
| :--- | :--- | :--- |
| **Black-Scholes-Merton** | `black_scholes.py` | Continuous dividend yield BSM pricing, 1st & 2nd order Greeks, Brentq IV solver, Put-Call parity, 12 strategy presets |
| **CRR Binomial Tree** | `binomial.py` | European & American pricing; exact tree-layer Delta, Gamma & Theta plus finite-difference Vega & Rho |
| **Monte Carlo** | `monte_carlo.py` | Antithetic variates + terminal-spot control variate; 95% confidence intervals; pathwise Delta & Vega estimators |
| **Risk & Position Sizing** | `risk.py` | 1σ expected move, lognormal risk-neutral E[PnL] & PoP, Cornish-Fisher Delta-Gamma-Vega VaR/CVaR, contract position sizing & Half-Kelly |
| **Live Market Data** | `market_data.py` | `yfinance` live quote, batch watchlist quotes, 13-week T-bill rate (`^IRX`), nearest ATM option chain & IV, daily/intraday OHLCV + offline fallback |
| **WebSocket Stream** | `streaming.py` | Zero-key Demo tick simulator or live Finnhub WSS trade ticks (`wss://ws.finnhub.io`) driving Spot in real time |
| **API Key Resolver** | `config.py` | 5-tier key resolution: Sidebar → `.streamlit/secrets.toml` → `FINNHUB_API_KEY` env → `.env` → `config.local.json` |
| **PDF Trade Ticket** | `pdf_export.py` | One-page executive PDF report with market snapshot, Call/Put Greeks table, strategy legs, break-evens & notes |
| **Interactive Charts** | `charts.py` | 12 dark-mode Plotly builders: Candlesticks + volume, IV smile, Greeks matrix with toggle buttons, P/L attribution waterfall, 3D surfaces |

## 🚀 Quick Start (60 Seconds)

### Option A — Local Python Environment

```bash
git clone https://github.com/MrKhan025/PreTrade.git
cd PreTrade
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Option B — Docker Container

```bash
docker build -t pretrade:latest .
docker run --rm -p 8501:8501 pretrade:latest
```

### Run the Automated Test Suite (30 Unit Tests)

```bash
pytest -v
# Or using standard library unittest:
python -m unittest discover -s tests -v
```

## 🖥️ Workstation Tour (7 Interactive Tabs)

Use the **left sidebar** to fetch a live equity/ETF ticker (e.g., `AAPL`, `SPY`, `NVDA`, `TSLA`), start the real-time tick stream, or adjust market inputs (**Spot Price**, **Strike Price**, **Days to Expiry**, **Implied Volatility**, **Risk-Free Rate**, and **Dividend Yield**):

| Tab | What It Does | Key Insights Provided |
| :--- | :--- | :--- |
| **1. Live Greeks & IV Solver** | Full Call/Put valuation, 1st & 2nd order Greeks, Put-Call parity check, 2D Greek profiles, CRR Binomial & Monte Carlo cross-validation, 3D surfaces, and IV solver | Intrinsic vs. extrinsic value, ITM probability, solved IV from market prices, 1σ expected move, and 3D Spot × DTE surfaces |
| **2. Theta Decay** | Simulates how an option's total value, extrinsic time value, and Delta evolve day-by-day from entry until expiration | Visualizes the non-linear "theta cliff" inside 30 DTE with a milestone table at 90, 60, 45, 30, 21, 14, 7, 3, and 1 DTE |
| **3. Strategy Builder** | Builds 12 preset strategies or custom 1-to-4-leg positions with shaded profit/loss zones | Compares **Expiration P/L**, **Mid-Horizon P/L**, and **Current T+0 P/L**, marking all break-even prices, PoP, Setup Score, and Net Greeks |
| **4. Portfolio Dashboard** | Editable multi-asset table (`st.data_editor`) tracking net Greek exposures across tickers | Computes Net Delta, Gamma, Theta, Vega, Notional value, **2nd-Order Taylor Stress Matrix**, and **1-Day 95% Delta-Gamma-Vega VaR** |
| **5. Trade Idea & Sizing** | Generates an executive trade ticket, **Position Sizing & Half-Kelly Calculator**, and 1-click PDF export | Computes exact recommended contracts from account equity & risk %, dollar capital at risk, and downloadable PDF trade ticket |
| **6. Sensitivity Heatmap** | Maps 60×60 scenario grids across Spot Price (±45%) and Volatility Shocks (±50%) | Visualizes how Premium, Net P/L, Delta, Gamma, Theta/day, Vega, or Vanna behave under simultaneous price and volatility moves |
| **7. Charts Studio** | Interactive Candlesticks + MA overlays, IV Smile, Greeks Matrix with toggle buttons, Strategy Comparison, and **P/L Attribution Waterfall** | Decomposes option P/L into Delta, Gamma, Theta, Vega & Vanna bars; exports standalone interactive HTML charts |

## 🧭 Plain-English Guide to the Options Greeks

### First-Order Greeks

| Greek | Symbol | Plain-English Meaning | Trader Question Answered | Example Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Delta** | `Δ` | Change in option price per **$1.00** move in the underlying stock (also approximates hedge ratio) | *"How much will my option gain or lose if the stock moves up $1?"* | `Δ = +0.54` means the call gains ~$0.54 if the stock rises $1.00 |
| **Gamma** | `Γ` | Change in **Delta** per **$1.00** move in the underlying stock (convexity / acceleration) | *"How fast does my directional exposure accelerate as the stock moves?"* | `Γ = 0.065` means Delta increases from `0.54` to `0.605` on a +$1 move |
| **Theta / Day** | `Θ` | Change in option price per **1 calendar day** passing (time decay) | *"How much value does this option lose overnight if the stock stays flat?"* | `Θ = -$0.072/day` means the option loses ~7.2 cents per share each day |
| **Vega (1% IV)** | `ν` | Change in option price per **+1% point** shift in Implied Volatility | *"How much does my option gain if IV rises from 20% to 21%?"* | `ν = +$0.114` means a +1% IV increase adds ~11.4 cents to the option |
| **Rho (1% r)** | `ρ` | Change in option price per **+1% point** shift in interest rates | *"How sensitive is this contract to central bank interest rate changes?"* | `ρ = +$0.042` means a +1% rate hike adds ~4.2 cents to the call |

### Second-Order Greeks & Probability Metrics

| Metric | Symbol / Key | What It Measures | Why Advanced Traders Care |
| :--- | :--- | :--- | :--- |
| **Vanna** | `∂Δ / ∂σ` (`vanna`) | Sensitivity of **Delta** to changes in **Implied Volatility** (or Vega to Spot) | Crucial for managing delta-hedged books when implied volatility spikes or crushes |
| **Volga (Vomma)** | `∂ν / ∂σ` (`volga`) | Sensitivity of **Vega** to changes in **Implied Volatility** (volatility convexity) | Shows which options benefit most from large volatility-of-volatility regimes |
| **Charm / Day** | `-∂Δ / ∂t` (`charm_daily`) | Daily drift in **Delta** simply from the passage of time (Delta bleed) | Tells you how much your Delta changes over a weekend even if Spot doesn't move |
| **Speed** | `∂Γ / ∂S` (`speed`) | Rate of change in **Gamma** per $1.00 move in the underlying Spot price | Identifies where Gamma pin risk accelerates fastest near expiration |
| **Prob. ITM** | `N(d₂)` / `N(-d₂)` (`prob_itm`) | Risk-neutral probability that the option expires In-The-Money | Gives a theoretically grounded probability of finishing in-the-money at expiry |

## 💻 Using PreTrade as a Python Library

All pricing, Greek calculations, binomial trees, Monte Carlo simulations, and risk analytics are decoupled from Streamlit so you can import them directly into research notebooks or backtesters:

```python
import numpy as np
from black_scholes import bs_greeks_dividend, implied_volatility, strategy_preset
from binomial import crr_price, crr_greeks
from monte_carlo import mc_price
from risk import expected_move, strategy_risk_metrics, position_sizing_recommendation

# 1. Analytical BSM 1st- and 2nd-order Greeks
metrics = bs_greeks_dividend(S=105.0, K=100.0, T=30, sigma=0.22, r=0.045, dividend_yield=0.015, option_type="call")
print(f"BSM Call: ${metrics['price']:.2f} | Delta: {metrics['delta']:.4f} | Vanna: {metrics['vanna']:.4f}")

# 2. American Put valuation & exact tree-node Greeks via CRR Binomial Tree
amer_put = crr_greeks(S=95.0, K=100.0, T=60, sigma=0.25, r=0.045, option_type="put", style="american", steps=200)
print(f"American Put: ${amer_put['price']:.2f} | Delta: {amer_put['delta']:.4f} | Gamma: {amer_put['gamma']:.5f}")

# 3. Variance-reduced Monte Carlo pricing with 95% confidence interval
mc = mc_price(S=105.0, K=100.0, T=30, sigma=0.22, r=0.045, dividend_yield=0.015, option_type="call", n_paths=50_000)
print(f"MC Call: ${mc['price']:.4f} ± {1.96 * mc['stderr']:.4f} (BSM benchmark: ${mc['bsm_benchmark']:.4f})")

# 4. Strategy risk & contract position sizing for a $25,000 account risking 2%
legs = strategy_preset("Iron Condor", spot=105.0, strike=105.0, dte=30, sigma=0.22, rate=0.045)
rm = strategy_risk_metrics(spot=105.0, legs=legs, sigma=0.22, rate=0.045, dte=30)
sizing = position_sizing_recommendation(25_000, 2.0, rm["max_loss"], rm["max_profit"], rm["prob_profit"])
print(f"Recommended Contracts: {sizing['recommended_contracts']} | Capital at Risk: ${sizing['total_dollar_risk']:.2f}")
```

## 📂 Project Structure

```text
PreTrade/
├── app.py                              # Streamlit interactive 7-tab options workstation
├── black_scholes.py                    # Core BSM pricing, 1st/2nd-order Greeks, IV solver, strategy engine
├── binomial.py                         # Cox-Ross-Rubinstein binomial tree (European & American + Greeks)
├── monte_carlo.py                      # Variance-reduced Monte Carlo pricing & pathwise Greeks
├── risk.py                             # Expected move, strategy PoP/E[PnL], VaR/CVaR & position sizing
├── market_data.py                      # Live quotes, option chain, history & ATM IV via yfinance
├── streaming.py                        # Demo simulator & Finnhub WebSocket real-time tick streamer
├── config.py                           # Multi-source API key resolution (.env, secrets, env, JSON)
├── pdf_export.py                       # Professional PDF trade-ticket generator (ReportLab)
├── charts.py                           # 12 dark-mode Plotly 2D/3D & Studio visualizations
├── .env.example                        # Environment variable template for API keys
├── .streamlit/
│   ├── config.toml                     # Native dark-mode theme & server security config
│   └── secrets.toml.example            # Template for Streamlit Community Cloud secrets
├── Dockerfile                          # Production container image with healthcheck
├── .dockerignore                       # Container build exclusions
├── Makefile                            # Developer task runner (install, run, test, lint, docker)
├── requirements.txt                    # Runtime & test dependencies
├── pyproject.toml                      # PEP 621 package metadata, pytest & ruff configuration
├── tests/
│   ├── test_black_scholes.py           # BSM benchmarks, parity, finite differences, IV & strategy tests
│   ├── test_advanced.py                # Binomial, Monte Carlo, Risk & Position Sizing tests
│   └── test_market_pdf.py              # Market data, config, streaming & PDF export tests
├── .github/
│   ├── workflows/ci.yml                # GitHub Actions CI across Python 3.10–3.12
│   ├── dependabot.yml                  # Automated weekly dependency security updates
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md               # Structured bug report template
│   │   └── feature_request.md          # Structured feature request template
│   └── PULL_REQUEST_TEMPLATE.md        # Pull request checklist template
├── CHANGELOG.md                        # Release version history
├── CONTRIBUTING.md                     # Contributor setup & testing guidelines
├── SECURITY.md                         # Secret hygiene & vulnerability reporting policy
├── LICENSE                             # MIT License
└── README.md                           # Project documentation & mathematical reference
```

## 🔑 API Keys & Configuration

Standard quotes, option chains, historical bars, and the **Demo** WebSocket streamer work **out of the box with no API key required**. To stream live market trade ticks via Finnhub WebSocket (`wss://ws.finnhub.io`), PreTrade resolves `FINNHUB_API_KEY` in this priority order:

1. Sidebar password field  
2. Streamlit secrets (`.streamlit/secrets.toml`)  
3. Environment variable `FINNHUB_API_KEY`  
4. Project `.env` file (`cp .env.example .env`)  
5. `config.local.json`

Get a free key at [finnhub.io/register](https://finnhub.io/register) — never commit real keys (`.env`, `.streamlit/secrets.toml`, and `config.local.json` are gitignored).

## 🤝 Contributing & License

Contributions, bug reports, and feature requests are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) for details.  
Released under the **MIT License**—see [LICENSE](LICENSE) for full details.
