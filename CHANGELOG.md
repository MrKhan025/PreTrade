# Changelog

All notable changes to **PreTrade** are documented in this file.

## [1.6.0] - 2026-10-29

### Added
- **Position Sizing & Fractional Kelly Calculator** (`risk.py` & `app.py` Trade Idea tab): Computes recommended contract count, dollar capital at risk, max dollar reward, and Kelly fraction based on account size and per-trade risk budget.
- **P/L Attribution Waterfall** (`charts.py` & `app.py` Charts Studio tab): Interactive Taylor-series P/L waterfall decomposing option P/L into Delta, Gamma, Theta, Vega, and Vanna bars.
- **Complete Advanced Pricing & Market Modules**: Added `binomial.py` (CRR European/American tree + exact tree-layer Greeks), `monte_carlo.py` (antithetic + control variate MC + pathwise Greeks), `risk.py` (1σ expected move, lognormal E[PnL], Cornish-Fisher Delta-Gamma-Vega VaR & CVaR), `market_data.py` (live `yfinance` + offline deterministic fallback), `streaming.py` (Demo + Finnhub WebSocket streamer), and `config.py` (5-tier API key resolver).
- **Production & Community Health Files**: Added `.streamlit/config.toml`, `.streamlit/secrets.toml.example`, `.env.example`, `Dockerfile`, `.dockerignore`, `Makefile`, `SECURITY.md`, `.github/dependabot.yml`, GitHub Issue/PR templates, and GitHub Actions CI workflow (`.github/workflows/ci.yml`).
- **Expanded Test Suite**: 30 automated unit tests across `tests/test_black_scholes.py`, `tests/test_advanced.py`, and `tests/test_market_pdf.py`.

### Fixed
- Renamed `Black Scholes.py` → `black_scholes.py`, `Charts.py` → `charts.py`, and `Test Black Scholes.py` → `tests/test_black_scholes.py` to fix `ModuleNotFoundError` across all platforms.
- Removed duplicate legacy v1.0 files (`App.py`, `Readme.md`, `Requirements.txt`) that caused case-insensitive filesystem collisions on Windows and macOS.
- Added NumPy 1.x / 2.x compatibility (`np.trapezoid` / `np.trapz`) and CLI `main()` entrypoint.

## [1.5.1] - 2026-10-01

### Added
- PDF Trade Ticket export (`pdf_export.py`) using ReportLab.
- 7-tab Streamlit workstation layout (`app.py`).

## [1.0.0] - 2026-09-01

### Added
- Initial Black-Scholes-Merton pricing engine, 1st- & 2nd-order Greeks, IV solver, and 12 multi-leg strategy presets.
