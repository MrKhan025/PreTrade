# 📈 Greek Lens — Interactive Options Greeks & Strategy Workstation

> **Understand option pricing, first- and second-order Greeks, implied volatility, multi-leg strategies, and portfolio stress testing in one interactive dark-mode dashboard.**

**Tech Stack:** Python 3.10+ • Streamlit • NumPy • SciPy • Pandas • Plotly  
**License:** MIT • **Test Suite:** 9/9 Unit Tests Passing (`unittest` & `pytest`)

## 🌟 Why Greek Lens?

Options trading involves multiple moving parts—spot price, time decay, implied volatility, interest rates, and dividend yields all interact simultaneously. **Greek Lens** turns complex Black-Scholes-Merton derivatives math into a clear, visual decision workflow:

- **For Traders:** Test 12 multi-leg option strategies (like Iron Condors, Butterflies, Straddles, and Calendar Spreads), see exact break-even points and Probability of Profit (PoP), and stress-test your portfolio against spot and volatility shocks before risking capital.
- **For Quants & Developers:** Import a clean, dependency-light Python pricing engine (`black_scholes.py`) with exact analytical 1st- and 2nd-order Greeks, an implied volatility root-finder, and Put-Call parity diagnostics verified against finite differences.
- **For Students & Educators:** Visually explore how option premiums erode over time (Theta decay), how Delta and Gamma curve across strikes, and how volatility shifts impact option prices across 2D heatmaps and 3D surfaces.

## 🚀 Quick Start (60 Seconds)

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/MrKhan025/greek-lens.git
cd greek-lens
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard

```bash
streamlit run app.py
```

### 3. Run the Automated Test Suite

```bash
python -m unittest discover -s tests -v
# Or using pytest:
pytest -v
```

## 🖥️ Workstation Tour (6 Interactive Tabs)

Once you launch `streamlit run app.py`, use the **left sidebar** to set your market inputs (**Spot Price**, **Strike Price**, **Days to Expiry**, **Implied Volatility**, **Risk-Free Rate**, and **Dividend Yield**) and explore six specialized tabs:

| Tab | What It Does | Key Insights Provided |
| :--- | :--- | :--- |
| **1. Live Greeks & IV Solver** | Displays full Call/Put valuation, 1st & 2nd order Greeks, Put-Call parity check, 2D Greek profiles, 3D surfaces, and an Implied Volatility solver | Intrinsic vs. extrinsic value, ITM probability, solved IV from market prices, and interactive 3D Spot × DTE surfaces |
| **2. Theta Decay** | Simulates how an option's total value, extrinsic time value, and Delta evolve day-by-day from entry until expiration | Visualizes the non-linear "theta cliff" inside 30 DTE with a milestone table at 90, 60, 45, 30, 21, 14, 7, 3, and 1 DTE |
| **3. Strategy Builder** | Builds 12 preset strategies or custom 1-to-4-leg positions with shaded profit/loss zones | Compares **Expiration P/L**, **Mid-Horizon P/L**, and **Current T+0 P/L**, marking all break-even prices, PoP, Setup Score, and Net Greeks |
| **4. Portfolio Dashboard** | Editable multi-asset table (`st.data_editor`) tracking net Greek exposures across tickers | Computes Net Delta, Gamma, Theta, Vega, Notional value, and a **2nd-Order Taylor Stress Matrix** across ±10% Spot and ±5 pt IV shocks |
| **5. Trade Idea** | Generates an executive trade ticket and plain-English risk summary for the selected strategy | Summarizes market bias, net debit/credit, max profit/loss per share, break-even points, PoP, and risk-adjusted setup score |
| **6. Sensitivity Heatmap** | Maps 60×60 scenario grids across Spot Price (±45%) and Volatility Shocks (±50%) | Visualizes how Premium, Net P/L, Delta, Gamma, Theta/day, Vega, or Vanna behave under simultaneous price and volatility moves |

## 🧭 Plain-English Guide to the Options Greeks

Greek Lens computes both **First-Order Greeks** (direct sensitivity to market inputs) and **Second-Order Greeks** (how the Greeks themselves change as the market moves).

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

## 🧩 Built-In Multi-Leg Strategy Presets

All 12 presets automatically price each leg at its fair Black-Scholes-Merton value based on your current sidebar inputs, or you can toggle **Customize strategy legs manually** to build any 1-to-4-leg structure.

| Strategy Preset | Market Bias | Structure | Risk / Reward Profile | Ideal Market Environment |
| :--- | :--- | :--- | :--- | :--- |
| **Bull Call Spread** | Moderately Bullish | Buy lower-strike Call + Sell higher-strike Call | Defined Risk / Capped Reward | Steady upward trend with elevated IV where you want to reduce call cost |
| **Bear Put Spread** | Moderately Bearish | Buy higher-strike Put + Sell lower-strike Put | Defined Risk / Capped Reward | Controlled pullback toward a target support level |
| **Long Butterfly** | Neutral (Target Pin) | Buy 1 ITM Call + Sell 2 ATM Calls + Buy 1 OTM Call | Low Cost / High Reward near center strike | Low-volatility pinning around a specific price target at expiration |
| **Iron Condor** | Neutral (Range-Bound) | Bull Put Spread + Bear Call Spread (4 legs) | Defined Risk / Net Credit Income | Sideways market trading within a well-defined support/resistance channel |
| **Iron Butterfly** | Neutral (Tight Range) | Short ATM Straddle + Long OTM Wings (4 legs) | Defined Risk / High Net Credit | Expecting implied volatility crush and minimal underlying movement |
| **Calendar Spread** | Neutral / Term-Structure | Sell near-term ATM Call + Buy longer-term ATM Call | Defined Debit / Positive Theta Carry | Capturing faster front-month Theta decay while staying long back-month Vega |
| **Long Straddle** | Bi-Directional Breakout | Buy ATM Call + Buy ATM Put | Defined Debit / Uncapped Move Potential | Major catalyst (earnings, CPI, FOMC) expected to move the stock beyond priced IV |
| **Long Strangle** | Bi-Directional Tail Move | Buy OTM Put + Buy OTM Call | Lower Debit / Wider Break-Evens | Expecting an explosive directional move in either direction |
| **Long Call / Put** | Strong Directional | Single long Call or Put leg | Defined Premium Risk / Convex Upside | High-conviction directional breakout with rising implied volatility |
| **Short Call / Put** | Income / Premium Decay | Single short Call or Put leg | Capped Credit / Uncapped (or Strike) Risk | Collecting time decay when implied volatility is overstated |

## 💻 Using Greek Lens as a Python Library

All pricing, Greek calculations, implied volatility root-finding, and strategy analytics live in `black_scholes.py` with zero UI dependencies—so you can import them directly into notebooks, backtesters, or trading scripts:

```python
import numpy as np
from black_scholes import (
    bs_greeks_dividend,
    implied_volatility,
    put_call_parity_check,
    strategy_preset,
    strategy_expiry_curve,
    strategy_summary_metrics,
)

# 1. Compute 1st- and 2nd-order Greeks for an option with a 1.5% dividend yield
metrics = bs_greeks_dividend(
    S=105.0, K=100.0, T=30, sigma=0.22, r=0.045, dividend_yield=0.015, option_type="call"
)
print(f"Call Price: ${metrics['price']:.2f} | Delta: {metrics['delta']:.4f} | Gamma: {metrics['gamma']:.5f}")
print(f"Theta/day:  ${metrics['theta_daily']:.4f} | Vega (1%): ${metrics['vega_pct']:.4f} | Vanna: {metrics['vanna']:.4f}")

# 2. Solve Implied Volatility from an observed market option price
iv_solved = implied_volatility(
    market_price=6.85, S=105.0, K=100.0, T=30, r=0.045, dividend_yield=0.015, option_type="call"
)
print(f"Solved Implied Volatility: {iv_solved * 100:.2f}%")

# 3. Analyze a 4-leg Iron Condor across a Spot grid
spot_grid = np.linspace(75, 135, 500)
legs = strategy_preset("Iron Condor", spot=105.0, strike=105.0, dte=30, sigma=0.22, rate=0.045)
expiry_pnl = strategy_expiry_curve(spot_grid, legs, sigma=0.22, rate=0.045)
summary = strategy_summary_metrics(spot_grid, expiry_pnl, spot=105.0, sigma=0.22, dte=30, rate=0.045)
print(f"Break-Evens: {summary['breakevens']} | PoP: {summary['prob_profit']:.1f}%")
```

## 📐 Mathematical Reference (Black-Scholes-Merton)

Under constant volatility `σ`, continuous risk-free rate `r`, continuous dividend yield `q`, and time to expiration `T = DTE / 365`:

- **Call Price:** `C = S · e⁻qᵀ · N(d₁) - K · e⁻ʳᵀ · N(d₂)`
- **Put Price:** `P = K · e⁻ʳᵀ · N(-d₂) - S · e⁻qᵀ · N(-d₁)`
- **Put-Call Parity:** `C - P = S · e⁻qᵀ - K · e⁻ʳᵀ`

where `N(·)` is the standard normal cumulative distribution function, `φ(·)` is the standard normal probability density function, and:

- `d₁ = [ ln(S / K) + (r - q + 0.5 · σ²) · T ] / (σ · √T)`
- `d₂ = d₁ - σ · √T`

| Analytical Formula | Call Option | Put Option |
| :--- | :--- | :--- |
| **Delta (`∂V/∂S`)** | `e⁻qᵀ · N(d₁)` | `-e⁻qᵀ · N(-d₁)` |
| **Gamma (`∂²V/∂S²`)** | `[ e⁻qᵀ · φ(d₁) ] / (S · σ · √T)` | `[ e⁻qᵀ · φ(d₁) ] / (S · σ · √T)` |
| **Vega (`∂V/∂σ`)** | `S · e⁻qᵀ · φ(d₁) · √T` | `S · e⁻qᵀ · φ(d₁) · √T` |
| **Theta (`∂V/∂t`)** | `-[S · e⁻qᵀ · φ(d₁) · σ] / (2√T) - r · K · e⁻ʳᵀ · N(d₂) + q · S · e⁻qᵀ · N(d₁)` | `-[S · e⁻qᵀ · φ(d₁) · σ] / (2√T) + r · K · e⁻ʳᵀ · N(-d₂) - q · S · e⁻qᵀ · N(-d₁)` |
| **Rho (`∂V/∂r`)** | `K · T · e⁻ʳᵀ · N(d₂)` | `-K · T · e⁻ʳᵀ · N(-d₂)` |
| **Vanna (`∂Δ/∂σ`)** | `-e⁻qᵀ · φ(d₁) · (d₂ / σ)` | `-e⁻qᵀ · φ(d₁) · (d₂ / σ)` |
| **Volga (`∂ν/∂σ`)** | `Vega · (d₁ · d₂ / σ)` | `Vega · (d₁ · d₂ / σ)` |

## 📂 Project Structure

```text
greek-lens/
├── app.py                      # Streamlit interactive 6-tab options workstation
├── black_scholes.py            # Core BSM pricing, 1st/2nd-order Greeks, IV solver, strategy engine
├── charts.py                   # Dark-mode Plotly 2D/3D visualizations (payoffs, decay, heatmaps, surfaces)
├── requirements.txt            # Python package dependencies
├── tests/
│   └── test_black_scholes.py   # Automated unit test suite (benchmarks, parity, finite differences, IV)
├── assets/                     # High-resolution dark-mode chart previews
├── README.md                   # Project documentation & quick-reference guide
├── LICENSE                     # MIT License
└── .gitignore                  # Standard Python & environment ignore rules
```

## 🤝 Contributing & License

Contributions, bug reports, and feature requests are welcome! Feel free to open an issue or submit a pull request.  
Released under the **MIT License**—see `LICENSE` for full details.
