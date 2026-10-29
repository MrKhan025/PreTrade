from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from black_scholes import bs_greeks_dividend, strategy_expiry_curve, strategy_preset, strategy_summary_metrics
from config import finnhub_key_source, get_finnhub_api_key, mask_key, resolve_api_key
from market_data import (
    fetch_intraday_bars,
    fetch_option_chain_summary,
    fetch_price_history,
    fetch_quote,
    fetch_quotes_batch,
    load_market_defaults,
)
from pdf_export import build_trade_ticket_pdf
from streaming import apply_stream_price_to_quote, start_stream, stop_stream


class TestMarketDataConfigStreamingAndPDF(unittest.TestCase):
    def test_config_key_resolution_priority_and_masking(self):
        self.assertEqual(mask_key(""), "Not configured")
        self.assertEqual(mask_key("abcd"), "***cd")
        self.assertEqual(mask_key("fh_1234567890xyz"), "fh_…xyz")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            env_file = tmp_path / ".env"
            json_file = tmp_path / "config.local.json"
            env_file.write_text("FINNHUB_API_KEY=dotenv_secret_123\n", encoding="utf-8")
            json_file.write_text('{"FINNHUB_API_KEY": "json_secret_456"}', encoding="utf-8")

            k1, s1 = resolve_api_key("sidebar_key_999", dotenv_path=env_file, json_path=json_file)
            self.assertEqual(k1, "sidebar_key_999")
            self.assertEqual(s1, "sidebar")

            old_env = os.environ.pop("FINNHUB_API_KEY", None)
            try:
                k2, s2 = resolve_api_key(None, dotenv_path=env_file, json_path=json_file)
                self.assertEqual(k2, "dotenv_secret_123")
                self.assertEqual(s2, ".env")

                env_file.unlink()
                k3, s3 = resolve_api_key(None, dotenv_path=env_file, json_path=json_file)
                self.assertEqual(k3, "json_secret_456")
                self.assertEqual(s3, "config.local.json")
            finally:
                if old_env is not None:
                    os.environ["FINNHUB_API_KEY"] = old_env

    def test_market_data_quotes_and_defaults(self):
        q = fetch_quote("AAPL")
        self.assertEqual(q["symbol"], "AAPL")
        self.assertGreater(q["price"], 0.0)
        self.assertIn("dividend_yield", q)

        batch = fetch_quotes_batch(["AAPL", "MSFT", "SPY"])
        self.assertEqual(len(batch), 3)

        defaults = load_market_defaults("SPY")
        self.assertEqual(defaults["symbol"], "SPY")
        self.assertGreater(defaults["spot"], 0.0)
        self.assertGreater(defaults["iv"], 0.0)
        self.assertGreaterEqual(defaults["dte"], 1)
        self.assertIn("chain", defaults)

    def test_option_chain_and_price_history(self):
        chain = fetch_option_chain_summary("AAPL", spot=225.0)
        self.assertIn("chain_preview", chain)
        self.assertFalse(chain["chain_preview"].empty)

        hist = fetch_price_history("AAPL", period="3mo", interval="1d")
        self.assertFalse(hist.empty)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            self.assertIn(col, hist.columns)

        intra = fetch_intraday_bars("AAPL", interval="5m", period="5d")
        self.assertFalse(intra.empty)

    def test_demo_streaming_lifecycle(self):
        snap = start_stream("demo", ["AAPL", "MSFT"], seed_prices={"AAPL": 225.0, "MSFT": 420.0})
        self.assertIn(snap["status"], ("streaming", "connecting"))
        self.assertIn("AAPL", snap["latest"])

        quote = {"symbol": "AAPL", "price": 220.0, "prev_close": 220.0}
        updated = apply_stream_price_to_quote(quote, snap)
        self.assertGreater(updated["price"], 0.0)

        stopped = stop_stream()
        self.assertEqual(stopped["status"], "stopped")

    def test_finnhub_stream_without_key_reports_error(self):
        snap = start_stream("finnhub", ["AAPL"], api_key="")
        self.assertEqual(snap["status"], "error")
        self.assertIn("API key", snap["error"] or "")
        stop_stream()

    def test_pdf_trade_ticket_export(self):
        spot, strike, dte, iv, rate, div = 100.0, 100.0, 30, 20.0, 0.045, 0.01
        call_m = bs_greeks_dividend(spot, strike, dte, iv / 100.0, rate, dividend_yield=div, option_type="call")
        put_m = bs_greeks_dividend(spot, strike, dte, iv / 100.0, rate, dividend_yield=div, option_type="put")
        legs = strategy_preset("Bull Call Spread", spot, strike, dte=dte, sigma=iv / 100.0, rate=rate)
        grid = np.linspace(60.0, 140.0, 300)
        exp_curve = strategy_expiry_curve(grid, legs, sigma=iv / 100.0, rate=rate)
        strat_m = strategy_summary_metrics(grid, exp_curve, spot=spot, sigma=iv / 100.0, dte=dte, rate=rate)

        pdf_bytes = build_trade_ticket_pdf(
            spot=spot,
            strike=strike,
            dte=dte,
            iv=iv,
            rate=rate,
            dividend_yield=div,
            call_metrics=call_m,
            put_metrics=put_m,
            strategy_name="Bull Call Spread",
            strategy_metrics=strat_m,
            legs=legs,
            symbol="AAPL",
            notes="Unit test ticket validation",
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)


if __name__ == "__main__":
    unittest.main()
