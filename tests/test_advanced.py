from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binomial import crr_greeks, crr_price
from black_scholes import bs_greeks_dividend, bs_price_dividend, strategy_preset
from monte_carlo import mc_price
from risk import expected_move, portfolio_var_delta_normal, strategy_risk_metrics


class TestAdvancedEngines(unittest.TestCase):
    def test_crr_european_converges_to_bsm(self):
        S, K, T, sigma, r, q = 100.0, 105.0, 60.0, 0.25, 0.045, 0.015
        for opt in ("call", "put"):
            bsm = bs_price_dividend(S, K, T, sigma, r, dividend_yield=q, option_type=opt)
            crr = crr_price(S, K, T, sigma, r, dividend_yield=q, option_type=opt, style="european", steps=300)
            self.assertAlmostEqual(crr, bsm, delta=0.02)

    def test_crr_american_put_early_exercise_premium(self):
        S, K, T, sigma, r, q = 90.0, 105.0, 180.0, 0.22, 0.06, 0.0
        euro_put = crr_price(S, K, T, sigma, r, dividend_yield=q, option_type="put", style="european", steps=200)
        amer_put = crr_price(S, K, T, sigma, r, dividend_yield=q, option_type="put", style="american", steps=200)
        self.assertGreaterEqual(amer_put, euro_put)
        self.assertGreaterEqual(amer_put, K - S)

    def test_crr_american_call_no_dividend_matches_european(self):
        S, K, T, sigma, r = 105.0, 100.0, 90.0, 0.20, 0.05
        euro_call = crr_price(S, K, T, sigma, r, dividend_yield=0.0, option_type="call", style="european", steps=180)
        amer_call = crr_price(S, K, T, sigma, r, dividend_yield=0.0, option_type="call", style="american", steps=180)
        self.assertAlmostEqual(amer_call, euro_call, places=5)

    def test_crr_greeks_consistency(self):
        S, K, T, sigma, r, q = 100.0, 100.0, 45.0, 0.20, 0.045, 0.01
        bsm_g = bs_greeks_dividend(S, K, T, sigma, r, dividend_yield=q, option_type="call")
        crr_g = crr_greeks(S, K, T, sigma, r, dividend_yield=q, option_type="call", style="european", steps=200)
        self.assertAlmostEqual(crr_g["delta"], bsm_g["delta"], delta=0.01)
        self.assertAlmostEqual(crr_g["gamma"], bsm_g["gamma"], delta=0.005)
        self.assertAlmostEqual(crr_g["vega_pct"], bsm_g["vega_pct"], delta=0.01)

    def test_monte_carlo_pricing_and_confidence_interval(self):
        S, K, T, sigma, r, q = 100.0, 100.0, 60.0, 0.22, 0.045, 0.01
        for opt in ("call", "put"):
            res = mc_price(S, K, T, sigma, r, dividend_yield=q, option_type=opt, n_paths=40_000, seed=123)
            self.assertAlmostEqual(res["price"], res["bsm_benchmark"], delta=0.05)
            self.assertLess(res["stderr"], 0.03)
            self.assertLessEqual(res["ci_lower"], res["bsm_benchmark"] + 0.05)
            self.assertGreaterEqual(res["ci_upper"], res["bsm_benchmark"] - 0.05)

    def test_monte_carlo_pathwise_greeks(self):
        S, K, T, sigma, r, q = 105.0, 100.0, 45.0, 0.20, 0.04, 0.01
        bsm_g = bs_greeks_dividend(S, K, T, sigma, r, dividend_yield=q, option_type="call")
        mc_res = mc_price(S, K, T, sigma, r, dividend_yield=q, option_type="call", n_paths=50_000, seed=7)
        self.assertAlmostEqual(mc_res["delta"], bsm_g["delta"], delta=0.015)
        self.assertAlmostEqual(mc_res["vega_pct"], bsm_g["vega_pct"], delta=0.015)

    def test_expected_move_scaling(self):
        em_68 = expected_move(100.0, 0.20, 30.0, confidence=0.68)
        em_95 = expected_move(100.0, 0.20, 30.0, confidence=0.95)
        self.assertGreater(em_95["expected_move"], em_68["expected_move"])
        self.assertAlmostEqual(em_68["upper"] - em_68["lower"], 2.0 * em_68["expected_move"], places=6)

    def test_strategy_risk_metrics_and_portfolio_var(self):
        legs = strategy_preset("Iron Condor", 100.0, 100.0, dte=30.0, sigma=0.20, rate=0.045)
        rm = strategy_risk_metrics(100.0, legs, 0.20, 0.045, 30.0, 0.0)
        self.assertGreater(rm["max_profit"], 0.0)
        self.assertGreater(rm["max_loss"], 0.0)
        self.assertTrue(0.0 <= rm["prob_profit"] <= 100.0)
        self.assertTrue(np.isfinite(rm["expected_pnl"]))

        var_res = portfolio_var_delta_normal(
            net_delta=120.0,
            net_gamma=3.5,
            net_vega=25.0,
            spot=100.0,
            sigma=0.22,
            horizon_days=1.0,
            confidence=0.95,
        )
        self.assertGreater(var_res["var"], 0.0)
        self.assertGreaterEqual(var_res["cvar"], var_res["var"])
        self.assertGreater(var_res["pnl_std"], 0.0)


    def test_crr_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            crr_price(100.0, 100.0, 30.0, 0.20, 0.05, style="bermudan")
        with self.assertRaises(ValueError):
            crr_price(100.0, 100.0, 30.0, 0.20, 0.05, steps=1)

    def test_monte_carlo_put_call_parity(self):
        S, K, T, sigma, r, q = 102.0, 100.0, 45.0, 0.24, 0.045, 0.01
        call_mc = mc_price(S, K, T, sigma, r, dividend_yield=q, option_type="call", n_paths=30_000, seed=99)
        put_mc = mc_price(S, K, T, sigma, r, dividend_yield=q, option_type="put", n_paths=30_000, seed=99)
        time = T / 365.0
        parity_rhs = S * np.exp(-q * time) - K * np.exp(-r * time)
        self.assertAlmostEqual(call_mc["price"] - put_mc["price"], parity_rhs, delta=0.05)

    def test_all_twelve_strategy_presets_risk_metrics(self):
        presets = [
            "Bull Call Spread", "Bear Put Spread", "Long Butterfly", "Iron Condor",
            "Iron Butterfly", "Calendar Spread", "Long Straddle", "Long Strangle",
            "Long Call", "Short Call", "Long Put", "Short Put",
        ]
        for name in presets:
            legs = strategy_preset(name, 100.0, 100.0, dte=30.0, sigma=0.22, rate=0.045)
            self.assertGreaterEqual(len(legs), 1)
            rm = strategy_risk_metrics(100.0, legs, 0.22, 0.045, 30.0, 0.0)
            self.assertTrue(0.0 <= rm["prob_profit"] <= 100.0)


if __name__ == "__main__":
    unittest.main()
