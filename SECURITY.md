# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.6.x   | :white_check_mark: |
| 1.5.x   | :white_check_mark: |
| < 1.5   | :x:                |

## API Keys & Secret Hygiene

- **Never commit real API keys** to the repository.
- PreTrade is pre-configured to ignore `.env`, `config.local.json`, and `.streamlit/secrets.toml` via `.gitignore`.
- If you accidentally commit a Finnhub or market data API key, revoke and rotate it immediately in your provider dashboard.

## Reporting a Vulnerability

If you discover a security issue in PreTrade:
1. **Do not** open a public GitHub issue containing sensitive exploit details.
2. Report the vulnerability privately via [GitHub Security Advisories](https://github.com/MrKhan025/PreTrade/security/advisories/new) or contact `@MrKhan025`.
3. Include steps to reproduce and affected files; we aim to respond within 48 hours.
