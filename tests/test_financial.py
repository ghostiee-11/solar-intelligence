"""Tests for financial module."""

from __future__ import annotations

import pytest

from solar_intelligence.financial import FinancialAnalyzer


class TestFinancialAnalyzer:
    """Tests for FinancialAnalyzer class."""

    def test_net_system_cost(self):
        fa = FinancialAnalyzer(system_cost=10000, incentive_percent=0.30)
        assert fa.net_system_cost == 7000

    def test_annual_savings_year_0(self):
        fa = FinancialAnalyzer(electricity_rate=0.12)
        savings = fa.annual_savings(5000, year=0)
        assert abs(savings - 600) < 0.01  # 5000 × 0.12

    def test_annual_savings_escalation(self):
        fa = FinancialAnalyzer(electricity_rate=0.10, rate_increase=0.05)
        s0 = fa.annual_savings(5000, year=0)
        s1 = fa.annual_savings(5000, year=1)
        assert s1 > s0  # Rate escalation should increase savings

    def test_annual_savings_degradation(self):
        fa = FinancialAnalyzer(panel_degradation=0.01)
        s0 = fa.annual_savings(5000, year=0)
        s10 = fa.annual_savings(5000, year=10)
        assert s10 < s0 * 1.5  # Degradation should limit growth despite rate increase

    def test_payback_period_positive(self):
        fa = FinancialAnalyzer(
            system_cost=10000, electricity_rate=0.12, incentive_percent=0.30,
            maintenance_cost=200,
        )
        payback = fa.payback_period(5000)
        assert 0 < payback < 25, f"Payback {payback} years seems off"

    def test_payback_never_for_tiny_system(self):
        fa = FinancialAnalyzer(system_cost=100000, electricity_rate=0.05)
        payback = fa.payback_period(100)  # Tiny system, huge cost
        assert payback == float("inf")

    def test_npv_positive_for_good_system(self):
        fa = FinancialAnalyzer(
            system_cost=15000, electricity_rate=0.15, incentive_percent=0.30,
            maintenance_cost=200,
        )
        npv = fa.net_present_value(8000)
        assert npv > 0, f"NPV should be positive for a well-sized system, got {npv}"

    def test_roi_positive(self):
        fa = FinancialAnalyzer(
            system_cost=15000, electricity_rate=0.12, incentive_percent=0.30,
            maintenance_cost=200,
        )
        roi = fa.return_on_investment(5000)
        assert roi > 0

    def test_lifetime_savings_shape(self):
        fa = FinancialAnalyzer(system_lifetime=25)
        df = fa.lifetime_savings(5000)
        assert len(df) == 25
        assert "year" in df.columns
        assert "cumulative_net_savings" in df.columns
        assert "carbon_offset_kg" in df.columns

    def test_lifetime_cumulative_increases(self):
        fa = FinancialAnalyzer()
        df = fa.lifetime_savings(5000)
        # Cumulative savings should generally increase after payback
        last_5 = df.tail(5)["cumulative_net_savings"].values
        assert all(last_5[i] <= last_5[i + 1] for i in range(len(last_5) - 1))

    def test_carbon_offset(self):
        fa = FinancialAnalyzer(carbon_factor=0.42)
        offset = fa.carbon_offset(5000)
        assert abs(offset - 2100) < 0.01  # 5000 × 0.42

    def test_lifetime_carbon(self):
        fa = FinancialAnalyzer(system_lifetime=25, panel_degradation=0)
        total = fa.lifetime_carbon_offset(5000)
        # With no degradation: 25 × 5000 × 0.42 / 1000 = 52.5 tonnes
        assert abs(total - 52.5) < 0.1

    def test_equivalent_trees(self):
        fa = FinancialAnalyzer()
        trees = fa.equivalent_trees(5000)
        assert trees > 0
        # 5000 × 0.42 / 22 ≈ 95 trees
        assert 80 < trees < 120

    def test_equivalent_car_miles(self):
        fa = FinancialAnalyzer()
        miles = fa.equivalent_car_miles(5000)
        assert miles > 0

    def test_financial_summary(self):
        fa = FinancialAnalyzer()
        summary = fa.financial_summary(5000)

        assert "investment" in summary
        assert "returns" in summary
        assert "environmental" in summary
        assert summary["investment"]["system_cost"] == fa.system_cost
        assert summary["returns"]["roi_pct"] > 0
