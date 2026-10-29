# Contributing to PreTrade

Thank you for your interest in improving PreTrade!

## Development setup

```bash
git clone https://github.com/MrKhan025/PreTrade.git
cd PreTrade
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -v
```

## Code style

- Prefer type hints and clear docstrings on public functions.
- Keep pure pricing logic free of Streamlit imports so it remains usable as a library.
- Add unit tests for any new pricing or risk function.

## Pull requests

1. Fork the repository and create a feature branch.
2. Ensure all tests pass.
3. Open a PR describing the change and any new math assumptions.

## License

By contributing you agree that your contributions will be licensed under the MIT License.
