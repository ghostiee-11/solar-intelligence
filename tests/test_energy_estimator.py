"""Tests for energy_estimator module."""

from __future__ import annotations

import numpy as np
import pytest

from solar_intelligence.energy_estimator import EnergyEstimator


class TestEnergyEstimator:
    """Tests for EnergyEstimator class."""

    def test_default_params(self):
        est = EnergyEstimator()
        assert est.panel_efficiency == 0.20
        assert est.num_panels == 10
        assert est.system_losses == 0.14

    def test_total_area(self):
        est = EnergyEstimator(panel_area=1.7, num_panels=10)
        assert est.total_area == 17.0

    def test_system_capacity(self):
        est = EnergyEstimator(panel_area=1.7, num_panels=10, panel_efficiency=0.20)
        # 17.0 m² × 0.20 = 3.4 kW
        assert abs(est.system_capacity_kw - 3.4) < 0.01

    # -------------------------------------------------------------------
    # Cell Temperature
    # -------------------------------------------------------------------

    def test_cell_temperature_increases_with_irradiance(self):
        est = EnergyEstimator()
        t_low = est.cell_temperature(25.0, 2.0)
        t_high = est.cell_temperature(25.0, 7.0)
        assert t_high > t_low

    def test_cell_temperature_increases_with_ambient(self):
        est = EnergyEstimator()
        t_cold = est.cell_temperature(10.0, 5.0)
        t_hot = est.cell_temperature(40.0, 5.0)
        assert t_hot > t_cold

    def test_temp_factor_below_stc(self):
        """Below 25°C cell temp, factor should be > 1 (negative coeff → better performance)."""
        est = EnergyEstimator()
        # Low ambient, low irradiance → cell temp < 25°C
        factor = est.temperature_factor(10.0, 1.0)
        assert factor > 1.0

    def test_temp_factor_above_stc(self):
        """Above 25°C cell temp, factor should be < 1 (performance loss)."""
        est = EnergyEstimator()
        factor = est.temperature_factor(40.0, 7.0)
        assert factor < 1.0

    # -------------------------------------------------------------------
    # Daily Energy
    # -------------------------------------------------------------------

    def test_zero_irradiance(self):
        est = EnergyEstimator()
        assert est.estimate_daily_energy(0.0) == 0.0

    def test_negative_irradiance(self):
        est = EnergyEstimator()
        assert est.estimate_daily_energy(-1.0) == 0.0

    def test_daily_energy_positive(self):
        est = EnergyEstimator()
        energy = est.estimate_daily_energy(5.0)
        assert energy > 0

    def test_daily_energy_formula(self):
        """Verify manual formula calculation."""
        est = EnergyEstimator(
            panel_efficiency=0.20,
            panel_area=1.7,
            num_panels=10,
            system_losses=0.14,
            inverter_efficiency=0.96,
        )
        ghi = 5.0  # kWh/m²/day
        # Without temperature: E = 5.0 × 0.20 × 17.0 × 0.86 × 0.96
        expected = 5.0 * 0.20 * 17.0 * (1 - 0.14) * 0.96
        actual = est.estimate_daily_energy(ghi, temperature=None)
        assert abs(actual - expected) < 0.01

    def test_daily_energy_with_temp_derating(self):
        est = EnergyEstimator()
        # Hot day should produce less than no-temp estimate
        energy_no_temp = est.estimate_daily_energy(6.0, temperature=None)
        energy_hot = est.estimate_daily_energy(6.0, temperature=45.0)
        assert energy_hot < energy_no_temp

    def test_more_panels_more_energy(self):
        est5 = EnergyEstimator(num_panels=5)
        est20 = EnergyEstimator(num_panels=20)
        assert est20.estimate_daily_energy(5.0) > est5.estimate_daily_energy(5.0)

    def test_higher_efficiency_more_energy(self):
        est_low = EnergyEstimator(panel_efficiency=0.15)
        est_high = EnergyEstimator(panel_efficiency=0.22)
        assert est_high.estimate_daily_energy(5.0) > est_low.estimate_daily_energy(5.0)

    # -------------------------------------------------------------------
    # Dataset-level Estimation
    # -------------------------------------------------------------------

    def test_estimate_from_dataset(self, sample_dataset):
        est = EnergyEstimator()
        daily = est.estimate_from_dataset(sample_dataset)

        assert "time" in daily.columns
        assert "energy_kwh" in daily.columns
        assert "ghi_kwh_m2" in daily.columns
        assert len(daily) == len(sample_dataset.time)
        assert all(daily["energy_kwh"] >= 0)

    def test_monthly_energy(self, sample_dataset):
        est = EnergyEstimator()
        monthly = est.estimate_monthly_energy(sample_dataset)

        assert len(monthly) == 12
        assert "avg_monthly_energy" in monthly.columns
        assert "month_name" in monthly.columns
        assert all(monthly["avg_monthly_energy"] > 0)

    def test_annual_energy(self, sample_dataset):
        est = EnergyEstimator()
        annual = est.estimate_annual_energy(sample_dataset)

        assert annual > 0
        # 10 panels × 1.7m² × 0.20 eff = 3.4 kW system
        # Typical yield: 1200-2000 kWh/kWp → 4080-6800 kWh/year
        assert 2000 < annual < 15000, f"Annual energy {annual} kWh seems implausible"

    def test_capacity_factor(self):
        est = EnergyEstimator()
        # 3.4 kW system, 5000 kWh/year
        cf = est.capacity_factor(5000)
        # CF = 5000 / (3.4 × 8760) = 16.8%
        assert 10 < cf < 30, f"Capacity factor {cf}% seems off"

    def test_system_summary(self, sample_dataset):
        est = EnergyEstimator()
        summary = est.system_summary(sample_dataset)

        assert "system" in summary
        assert "production" in summary
        assert "performance" in summary
        assert summary["system"]["capacity_kw"] > 0
        assert summary["production"]["annual_energy_kwh"] > 0
        assert 0 < summary["performance"]["capacity_factor_pct"] < 50
