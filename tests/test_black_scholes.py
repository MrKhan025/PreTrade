from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from black_scholes import (
    bs_greeks,
    bs_greeks_dividend,
    bs_price,
    bs_price_dividend,
    classify_strategy_bias,
    implied_volatility,
    normalize_dte,
    put_call_parity_check,
    simulate_theta_decay,
    strategy_current_curve,
    strategy_expiry_curve,
    strategy_greeks,
    strategy_preset,
    strategy_score,
    strategy_summary_metrics,
    trade_idea_summary,
)


class TestBlackScholesEngine(unittest.TestCase):
    def test_normalize_dte(self):
        self.assertTrue(np.isclose(normalize_dte(365.0), 1.0))
        self.assertEqual(normalize_dte(0.0), 1e-9)
        self.assertEqual(normalize_dte(-10.0), 1e-9)
        with self.assertRaises(ValueError):
            normalize_dte(float("nan"))

    def test_bs_price_standard_benchmark(self):
        # Hull textbook benchmark: S=100, K=100, T=365 (1 yr), sigma=0.20, r=0.05
        call = bs_price(100.0, 100.0, 365.0, 0.20, 0.05, "call")
        put = bs_price(100.0, 100.0, 365.0, 0.20, 0.05, "put")
        self.assertAlmostEqual(call, 10.45058, places=4)
        self.assertAlmostEqual(put, 5.57353, places=4)

    def test_put_call_parity_with_dividends(self):
        S, K, T, sigma, r, q = 120.0, 110.0, 90.0, 0.28, 0.045, 0.025
        call = bs_price_dividend(S, K, T, sigma, r, dividend_yield=q, option_type="call", american_adjustment=False)
        put = bs_price_dividend(S, K, T, sigma, r, dividend_yield=q, option_type="put", american_adjustment=False)
        parity = put_call_parity_check(call, put, S, K, T, r, dividend_yield=q)
        self.assertTrue(parity["parity_holds"])
        self.assertLess(parity["abs_discrepancy"], 1e-10)
        self.assertAlmostEqual(parity["synthetic_call"], call, places=8)
        self.assertAlmostEqual(parity["synthetic_put"], put, places=8)

    def test_analytical_greeks_match_finite_differences(self):
        S, K, T, sigma, r, q = 105.0, 100.0, 60.0, 0.25, 0.04, 0.02
        eps_s = 1e-4
        eps_vol = 1e-5
        eps_r = 1e-5
        eps_t = 1e-3  # in days

        for opt in ("call", "put"):
            g = bs_greeks_dividend(S, K, T, sigma, r, dividend_yield=q, option_type=opt, american_adjustment=False)

            # Delta & Gamma via central difference wrt S
            p_up = bs_price_dividend(S + eps_s, K, T, sigma, r, q, opt, False)
            p_mid = bs_price_dividend(S, K, T, sigma, r, q, opt, False)
            p_dn = bs_price_dividend(S - eps_s, K, T, sigma, r, q, opt, False)
            fd_delta = (p_up - p_dn) / (2 * eps_s)
            fd_gamma = (p_up - 2 * p_mid + p_dn) / (eps_s**2)

            self.assertAlmostEqual(g["delta"], fd_delta, places=6)
            self.assertAlmostEqual(g["gamma"], fd_gamma, places=5)

            # Vega & Volga via central difference wrt sigma
            v_up = bs_price_dividend(S, K, T, sigma + eps_vol, r, q, opt, False)
            v_dn = bs_price_dividend(S, K, T, sigma - eps_vol, r, q, opt, False)
            fd_vega = (v_up - v_dn) / (2 * eps_vol)
            fd_volga = (v_up - 2 * p_mid + v_dn) / (eps_vol**2)

            self.assertAlmostEqual(g["vega"], fd_vega, places=4)
            self.assertAlmostEqual(g["vega_pct"], fd_vega / 100.0, places=6)
            self.assertAlmostEqual(g["volga"], fd_volga, places=3)

            # Vanna via central difference of Delta wrt sigma
            d_vol_up = bs_greeks_dividend(S, K, T, sigma + eps_vol, r, q, opt, False)["delta"]
            d_vol_dn = bs_greeks_dividend(S, K, T, sigma - eps_vol, r, q, opt, False)["delta"]
            fd_vanna = (d_vol_up - d_vol_dn) / (2 * eps_vol)
            self.assertAlmostEqual(g["vanna"], fd_vanna, places=5)

            # Rho via central difference wrt r
            r_up = bs_price_dividend(S, K, T, sigma, r + eps_r, q, opt, False)
            r_dn = bs_price_dividend(S, K, T, sigma, r - eps_r, q, opt, False)
            fd_rho = (r_up - r_dn) / (2 * eps_r)
            self.assertAlmostEqual(g["rho"], fd_rho, places=4)

            # Daily Theta via difference wrt DTE (-dV/d(DTE))
            t_up = bs_price_dividend(S, K, T + eps_t, sigma, r, q, opt, False)
            t_dn = bs_price_dividend(S, K, T - eps_t, sigma, r, q, opt, False)
            fd_theta_daily = -(t_up - t_dn) / (2 * eps_t)
            self.assertAlmostEqual(g["theta_daily"], fd_theta_daily, places=5)

    def test_implied_volatility_round_trip(self):
        S, K, T, r, q = 100.0, 105.0, 45.0, 0.045, 0.015
        for target_vol in (0.12, 0.25, 0.65, 1.25):
            for opt in ("call", "put"):
                mkt_price = bs_price_dividend(S, K, T, target_vol, r, dividend_yield=q, option_type=opt)
                solved = implied_volatility(mkt_price, S, K, T, r, dividend_yield=q, option_type=opt)
                self.assertAlmostEqual(solved, target_vol, places=6)

    def test_implied_volatility_arbitrage_bounds(self):
        with self.assertRaises(ValueError):
            implied_volatility(-1.0, 100.0, 100.0, 30.0, 0.05)
        with self.assertRaises(ValueError):
            implied_volatility(105.0, 100.0, 100.0, 30.0, 0.05, option_type="call")

    def test_multi_leg_strategies_and_multiple_breakevens(self):
        spot, strike, dte, sigma, rate = 100.0, 100.0, 30.0, 0.20, 0.045
        spot_range = np.linspace(60.0, 140.0, 801)

        ic_legs = strategy_preset("Iron Condor", spot, strike, dte=dte, sigma=sigma, rate=rate)
        self.assertEqual(len(ic_legs), 4)
        ic_expiry = strategy_expiry_curve(spot_range, ic_legs, sigma=sigma, rate=rate)
        ic_metrics = strategy_summary_metrics(spot_range, ic_expiry, spot=spot, sigma=sigma, dte=dte, rate=rate)
        self.assertEqual(len(ic_metrics["breakevens"]), 2)
        self.assertGreater(ic_metrics["max_profit"], 0.0)
        self.assertLess(ic_metrics["max_loss"], 0.0)
        self.assertTrue(0.0 <= strategy_score(ic_metrics) <= 100.0)

        cal_legs = strategy_preset("Calendar Spread", spot, strike, dte=dte, sigma=sigma, rate=rate)
        cal_expiry = strategy_expiry_curve(spot_range, cal_legs, sigma=sigma, rate=rate)
        self.assertGreater(float(np.max(cal_expiry)), 0.10)
        self.assertFalse(np.allclose(cal_expiry, 0.0))

        net_g = strategy_greeks(spot, ic_legs, dte, sigma, rate)
        self.assertIn("delta", net_g)
        self.assertLess(abs(net_g["delta"]), 0.15)
        self.assertGreater(net_g["theta_daily"], 0.0)

    def test_simulate_theta_decay_and_trade_summary(self):
        records = simulate_theta_decay(100.0, 100.0, 0.20, 0.05, option_type="call", dte_min=1, dte_max=30, steps=10)
        self.assertEqual(len(records), 10)
        self.assertGreater(records[-1]["price"], records[0]["price"])

        spot_range = np.linspace(70.0, 130.0, 300)
        legs = strategy_preset("Long Butterfly", 100.0, 100.0)
        expiry = strategy_expiry_curve(spot_range, legs)
        metrics = strategy_summary_metrics(spot_range, expiry, spot=100.0)
        summary = trade_idea_summary("Long Butterfly", metrics, 100.0, 100.0)
        self.assertIn("Long Butterfly", summary)
        self.assertIn(classify_strategy_bias("Long Butterfly"), summary)

    def test_syntax_all_python_files(self):
        repo_root = Path(__file__).resolve().parents[1]
        for py_file in repo_root.glob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            ast.parse(source, filename=str(py_file))


if __name__ == "__main__":
    unittest.main()