"""Financial analysis module for solar investments.

Computes ROI, payback period, NPV, carbon offsets,
and lifetime savings projections.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import param

from solar_intelligence.config import (
    CARBON_FACTOR_KG_PER_KWH,
    CAR_KG_CO2_PER_MILE,
    DEFAULT_ELECTRICITY_RATE,
    DEFAULT_INCENTIVE_PERCENT,
    DEFAULT_MAINTENANCE_COST,
    DEFAULT_PANEL_DEGRADATION,
    DEFAULT_RATE_INCREASE,
    DEFAULT_SYSTEM_COST,
    DEFAULT_SYSTEM_LIFETIME,
    TREES_KG_CO2_PER_YEAR,
)

logger = logging.getLogger(__name__)


class FinancialAnalyzer(param.Parameterized):
    """Analyze financial viability of solar PV investments.

    Computes payback period, NPV, ROI, lifetime savings,
    and carbon offset metrics.

    Parameters
    ----------
    system_cost : float
        Total system cost in USD (before incentives).
    electricity_rate : float
        Current electricity rate in $/kWh.
    rate_increase : float
        Annual electricity rate increase (fraction, e.g. 0.03 = 3%).
    incentive_percent : float
        Tax credit / incentive as fraction of system cost.
    maintenance_cost : float
        Annual maintenance cost in USD.
    panel_degradation : float
        Annual panel degradation rate (fraction, e.g. 0.005 = 0.5%).
    carbon_factor : float
        Grid carbon intensity in kg CO2/kWh.
    system_lifetime : int
        System lifetime in years.
    """

    system_cost = param.Number(default=DEFAULT_SYSTEM_COST, bounds=(0, 50_000_000))
    electricity_rate = param.Number(default=DEFAULT_ELECTRICITY_RATE, bounds=(0, 100))
    rate_increase = param.Number(default=DEFAULT_RATE_INCREASE, bounds=(0, 0.2))
    incentive_percent = param.Number(default=DEFAULT_INCENTIVE_PERCENT, bounds=(0, 1))
    maintenance_cost = param.Number(default=DEFAULT_MAINTENANCE_COST, bounds=(0, 500_000))
    panel_degradation = param.Number(default=DEFAULT_PANEL_DEGRADATION, bounds=(0, 0.05))
    carbon_factor = param.Number(default=CARBON_FACTOR_KG_PER_KWH, bounds=(0, 2))
    system_lifetime = param.Integer(default=DEFAULT_SYSTEM_LIFETIME, bounds=(1, 50))

    @property
    def net_system_cost(self) -> float:
        """System cost after incentives."""
        return self.system_cost * (1 - self.incentive_percent)

    # -------------------------------------------------------------------
    # Savings & Payback
    # -------------------------------------------------------------------

    def annual_savings(self, annual_energy_kwh: float, year: int = 0) -> float:
        """Compute annual electricity savings.

        Parameters
        ----------
        annual_energy_kwh : float
            Annual energy production in kWh.
        year : int
            Year number (0 = first year) for rate escalation.

        Returns
        -------
        float
            Annual savings in USD.
        """
        rate = self.electricity_rate * (1 + self.rate_increase) ** year
        degraded_energy = annual_energy_kwh * (1 - self.panel_degradation) ** year
        return degraded_energy * rate

    def payback_period(self, annual_energy_kwh: float) -> float:
        """Compute simple payback period.

        Parameters
        ----------
        annual_energy_kwh : float
            First-year annual energy production in kWh.

        Returns
        -------
        float
            Payback period in years. Returns inf if never pays back.
        """
        cumulative = 0.0
        net_cost = self.net_system_cost

        for year in range(1, self.system_lifetime + 1):
            savings = self.annual_savings(annual_energy_kwh, year - 1)
            net_annual = savings - self.maintenance_cost
            cumulative += net_annual

            if cumulative >= net_cost:
                # Interpolate fractional year
                prev_cumulative = cumulative - net_annual
                remaining = net_cost - prev_cumulative
                fraction = remaining / net_annual if net_annual > 0 else 0
                return year - 1 + fraction

        return float("inf")

    def net_present_value(
        self,
        annual_energy_kwh: float,
        discount_rate: float = 0.05,
    ) -> float:
        """Compute Net Present Value of solar investment.

        Parameters
        ----------
        annual_energy_kwh : float
            First-year annual energy production.
        discount_rate : float
            Discount rate for NPV calculation.

        Returns
        -------
        float
            NPV in USD.
        """
        npv = -self.net_system_cost
        for year in range(1, self.system_lifetime + 1):
            savings = self.annual_savings(annual_energy_kwh, year - 1)
            net_cash = savings - self.maintenance_cost
            npv += net_cash / (1 + discount_rate) ** year
        return npv

    def return_on_investment(self, annual_energy_kwh: float) -> float:
        """Compute total ROI over system lifetime.

        Returns
        -------
        float
            ROI as percentage.
        """
        total_savings = sum(
            self.annual_savings(annual_energy_kwh, y) - self.maintenance_cost
            for y in range(self.system_lifetime)
        )
        net_cost = self.net_system_cost
        if net_cost <= 0:
            return float("inf")
        return ((total_savings - net_cost) / net_cost) * 100

    def lifetime_savings(self, annual_energy_kwh: float) -> pd.DataFrame:
        """Compute year-by-year savings and cumulative returns.

        Returns
        -------
        pd.DataFrame
            Year-by-year breakdown with columns: year, energy_kwh,
            electricity_rate, savings, maintenance, net_savings,
            cumulative_net_savings, carbon_offset_kg.
        """
        records = []
        cumulative = -self.net_system_cost

        for year in range(1, self.system_lifetime + 1):
            degraded = annual_energy_kwh * (1 - self.panel_degradation) ** (year - 1)
            rate = self.electricity_rate * (1 + self.rate_increase) ** (year - 1)
            savings = degraded * rate
            net = savings - self.maintenance_cost
            cumulative += net
            carbon = self.carbon_offset(degraded)

            records.append({
                "year": year,
                "energy_kwh": round(degraded, 1),
                "electricity_rate": round(rate, 4),
                "savings_usd": round(savings, 2),
                "maintenance_usd": self.maintenance_cost,
                "net_savings_usd": round(net, 2),
                "cumulative_net_savings": round(cumulative, 2),
                "carbon_offset_kg": round(carbon, 1),
            })

        return pd.DataFrame(records)

    # -------------------------------------------------------------------
    # Carbon & Environmental
    # -------------------------------------------------------------------

    def carbon_offset(self, annual_energy_kwh: float) -> float:
        """Annual carbon offset in kg CO2."""
        return annual_energy_kwh * self.carbon_factor

    def lifetime_carbon_offset(self, annual_energy_kwh: float) -> float:
        """Lifetime carbon offset in tonnes CO2."""
        total = sum(
            annual_energy_kwh * (1 - self.panel_degradation) ** y * self.carbon_factor
            for y in range(self.system_lifetime)
        )
        return total / 1000  # kg → tonnes

    def equivalent_trees(self, annual_energy_kwh: float) -> int:
        """Number of trees equivalent to annual carbon offset."""
        offset = self.carbon_offset(annual_energy_kwh)
        return int(offset / TREES_KG_CO2_PER_YEAR)

    def equivalent_car_miles(self, annual_energy_kwh: float) -> float:
        """Car miles equivalent to annual carbon offset."""
        offset = self.carbon_offset(annual_energy_kwh)
        return offset / CAR_KG_CO2_PER_MILE

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    def financial_summary(self, annual_energy_kwh: float) -> dict[str, Any]:
        """Comprehensive financial analysis summary.

        Returns
        -------
        dict
            Complete financial metrics.
        """
        payback = self.payback_period(annual_energy_kwh)
        npv = self.net_present_value(annual_energy_kwh)
        roi = self.return_on_investment(annual_energy_kwh)
        carbon = self.carbon_offset(annual_energy_kwh)
        lifetime_co2 = self.lifetime_carbon_offset(annual_energy_kwh)

        return {
            "investment": {
                "system_cost": self.system_cost,
                "incentive": round(self.system_cost * self.incentive_percent, 2),
                "net_cost": round(self.net_system_cost, 2),
            },
            "returns": {
                "first_year_savings": round(self.annual_savings(annual_energy_kwh, 0), 2),
                "payback_years": round(payback, 1) if payback != float("inf") else "N/A",
                "npv_25yr": round(npv, 2),
                "roi_pct": round(roi, 1),
            },
            "environmental": {
                "annual_co2_offset_kg": round(carbon, 1),
                "lifetime_co2_offset_tonnes": round(lifetime_co2, 1),
                "equivalent_trees": self.equivalent_trees(annual_energy_kwh),
                "equivalent_car_miles_avoided": round(
                    self.equivalent_car_miles(annual_energy_kwh), 0
                ),
            },
        }
