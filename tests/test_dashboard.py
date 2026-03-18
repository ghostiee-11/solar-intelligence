"""Tests for dashboard components and Lumen integration.

Tests UI component construction, dashboard initialization, and
Lumen Source/Transform functionality without requiring a running server.
"""

from __future__ import annotations

import pandas as pd
import pytest

from solar_intelligence.ai_engine import SolarAIEngine
from solar_intelligence.data_loader import generate_synthetic_solar_data
from solar_intelligence.ui.components import (
    FinancialConfigurator,
    KPICard,
    LocationPicker,
    PanelConfigurator,
)


# ---------------------------------------------------------------------------
# Component Tests
# ---------------------------------------------------------------------------

class TestLocationPicker:

    def test_default_values(self):
        lp = LocationPicker()
        assert lp.latitude == 28.6139
        assert lp.longitude == 77.2090

    def test_custom_coords(self):
        lp = LocationPicker(latitude=40.7, longitude=-74.0, location_name="NYC")
        assert lp.latitude == 40.7
        assert lp.longitude == -74.0
        assert lp.location_name == "NYC"

    def test_panel_property_returns_column(self):
        lp = LocationPicker()
        panel = lp.panel
        assert panel is not None
        # Should be a pn.Column
        assert hasattr(panel, 'objects')


class TestPanelConfigurator:

    def test_default_params(self):
        pc = PanelConfigurator()
        assert pc.panel_efficiency == 0.20
        assert pc.num_panels == 10
        assert pc.direction == "South"

    def test_custom_params(self):
        pc = PanelConfigurator(panel_efficiency=0.22, num_panels=20, direction="East")
        assert pc.panel_efficiency == 0.22
        assert pc.num_panels == 20
        assert pc.direction == "East"

    def test_panel_property(self):
        pc = PanelConfigurator()
        panel = pc.panel
        assert panel is not None


class TestFinancialConfigurator:

    def test_defaults(self):
        fc = FinancialConfigurator()
        # Default currency is INR
        assert fc.currency == "INR"
        assert fc.system_cost == 500000  # ₹5 lakh
        assert fc.electricity_rate == 8.0  # ₹8/kWh
        assert fc.currency_symbol == "₹"

    def test_panel_property(self):
        fc = FinancialConfigurator()
        panel = fc.panel
        assert panel is not None


class TestKPICard:

    def test_create_returns_column(self):
        card = KPICard.create("Test Metric", "42", "units", "#FF0000")
        assert card is not None
        assert hasattr(card, 'objects')

    def test_card_with_empty_subtitle(self):
        card = KPICard.create("Title", "99", "")
        assert card is not None


# ---------------------------------------------------------------------------
# AI Engine Tests
# ---------------------------------------------------------------------------

class TestAIEngine:

    def test_template_mode_default(self):
        ai = SolarAIEngine()
        assert ai.mode == "template"

    def test_generate_report(self):
        ai = SolarAIEngine()
        solar = {
            "location": {"latitude": 28.6, "longitude": 77.2},
            "average_daily_ghi": 5.5,
            "average_daily_dni": 4.0,
            "average_daily_dhi": 2.0,
            "annual_solar_energy_kwh_m2": 2000,
            "best_month": "Jun",
            "best_month_ghi": 8.0,
            "worst_month": "Dec",
            "worst_month_ghi": 3.0,
            "seasonal_ratio": 2.7,
        }
        energy = {
            "system": {"capacity_kw": 6.8, "num_panels": 20, "total_area_m2": 34,
                       "panel_efficiency": 0.2, "system_losses": 0.14},
            "production": {"annual_energy_kwh": 9000, "avg_daily_energy_kwh": 24.7,
                           "best_month": "Jun", "best_month_energy_kwh": 1000,
                           "worst_month": "Dec", "worst_month_energy_kwh": 500},
            "performance": {"capacity_factor_pct": 15.1, "specific_yield_kwh_kwp": 1324},
        }
        financial = {
            "investment": {"system_cost": 20000, "incentive": 6000, "net_cost": 14000},
            "returns": {"first_year_savings": 1080, "payback_years": 12.5,
                        "npv_25yr": 5000, "roi_pct": 130},
            "environmental": {"annual_co2_offset_kg": 3780, "lifetime_co2_offset_tonnes": 90,
                              "equivalent_trees": 172, "equivalent_car_miles_avoided": 9356},
        }
        orientation = {
            "best_direction": "South", "best_tilt": 30, "best_azimuth": 180,
            "annual_energy_kwh": 10000,
            "energy_gain_vs_horizontal_pct": 5.0,
            "energy_gain_vs_worst_pct": 45.0,
            "worst_direction": "North", "worst_tilt": 90,
        }

        report = ai.generate_report(solar, energy, financial, orientation)
        assert isinstance(report, str)
        assert len(report) > 500
        assert "Solar Resource Assessment" in report
        assert "System Performance" in report
        assert "Orientation Recommendation" in report
        assert "Financial Analysis" in report
        assert "Environmental Impact" in report
        assert "South" in report
        assert "5.50" in report or "5.5" in report

    def test_generate_report_without_orientation(self):
        ai = SolarAIEngine()
        solar = {
            "location": {"latitude": 28.6, "longitude": 77.2},
            "average_daily_ghi": 5.5, "average_daily_dni": 4.0,
            "average_daily_dhi": 2.0, "annual_solar_energy_kwh_m2": 2000,
            "best_month": "Jun", "best_month_ghi": 8.0,
            "worst_month": "Dec", "worst_month_ghi": 3.0,
            "seasonal_ratio": 2.7,
        }
        energy = {
            "system": {"capacity_kw": 3.4, "num_panels": 10, "total_area_m2": 17,
                       "panel_efficiency": 0.2, "system_losses": 0.14},
            "production": {"annual_energy_kwh": 5000, "avg_daily_energy_kwh": 13.7,
                           "best_month": "Jun", "best_month_energy_kwh": 600,
                           "worst_month": "Dec", "worst_month_energy_kwh": 300},
            "performance": {"capacity_factor_pct": 16.8, "specific_yield_kwh_kwp": 1471},
        }
        financial = {
            "investment": {"system_cost": 10000, "incentive": 3000, "net_cost": 7000},
            "returns": {"first_year_savings": 600, "payback_years": 11,
                        "npv_25yr": 2000, "roi_pct": 100},
            "environmental": {"annual_co2_offset_kg": 2100, "lifetime_co2_offset_tonnes": 50,
                              "equivalent_trees": 95, "equivalent_car_miles_avoided": 5198},
        }

        report = ai.generate_report(solar, energy, financial)
        assert "Orientation" not in report

    def test_quick_insight_ghi(self):
        ai = SolarAIEngine()
        insight = ai.quick_insight("ghi", 5.5)
        assert "5.5" in insight
        assert "good" in insight

    def test_quick_insight_payback(self):
        ai = SolarAIEngine()
        insight = ai.quick_insight("payback", 7.0)
        assert "7.0" in insight

    def test_quick_insight_unknown_metric(self):
        ai = SolarAIEngine()
        insight = ai.quick_insight("unknown_metric", 42)
        assert "42" in insight

    def test_classify_irradiance_levels(self):
        ai = SolarAIEngine()
        assert ai._classify_irradiance(7.0) == "excellent"
        assert ai._classify_irradiance(5.0) == "good"
        assert ai._classify_irradiance(3.5) == "moderate"
        assert ai._classify_irradiance(1.0) == "low"


# ---------------------------------------------------------------------------
# Lumen Source/Transform Tests
# ---------------------------------------------------------------------------

class TestLumenIntegration:

    def test_solar_data_source_tables(self):
        from solar_intelligence.ui.lumen_app import SolarDataSource
        source = SolarDataSource(latitude=28.6, longitude=77.2)
        tables = source.get_tables()
        assert "daily_solar" in tables
        assert "monthly_solar" in tables
        assert "metadata" in tables

    def test_solar_data_source_schema(self):
        from solar_intelligence.ui.lumen_app import SolarDataSource
        source = SolarDataSource()
        schema = source.get_schema("daily_solar")
        assert "ALLSKY_SFC_SW_DWN" in schema
        assert "T2M" in schema

    def test_solar_data_source_get_daily(self):
        from solar_intelligence.ui.lumen_app import SolarDataSource
        source = SolarDataSource(use_synthetic=True, start_year=2023, end_year=2023)
        df = source.get("daily_solar")
        assert isinstance(df, pd.DataFrame)
        assert "ALLSKY_SFC_SW_DWN" in df.columns
        assert len(df) == 365

    def test_solar_data_source_get_monthly(self):
        from solar_intelligence.ui.lumen_app import SolarDataSource
        source = SolarDataSource(use_synthetic=True, start_year=2023, end_year=2023)
        df = source.get("monthly_solar")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 12
        assert "GHI" in df.columns

    def test_solar_data_source_get_metadata(self):
        from solar_intelligence.ui.lumen_app import SolarDataSource
        source = SolarDataSource(use_synthetic=True)
        df = source.get("metadata")
        assert "latitude" in df["key"].values

    def test_solar_data_source_unknown_table(self):
        from solar_intelligence.ui.lumen_app import SolarDataSource
        source = SolarDataSource(use_synthetic=True)
        with pytest.raises(ValueError, match="Unknown table"):
            source.get("nonexistent")

    def test_solar_energy_transform(self):
        from solar_intelligence.ui.lumen_app import SolarDataSource, SolarEnergyTransform
        source = SolarDataSource(use_synthetic=True, start_year=2023, end_year=2023)
        df = source.get("daily_solar")
        transform = SolarEnergyTransform(panel_efficiency=0.20, total_area=17.0)
        result = transform.apply(df)
        assert "energy_kwh" in result.columns
        assert all(result["energy_kwh"] >= 0)

    def test_monthly_aggregate_transform(self):
        from solar_intelligence.ui.lumen_app import SolarDataSource, MonthlyAggregateTransform
        source = SolarDataSource(use_synthetic=True, start_year=2023, end_year=2023)
        df = source.get("daily_solar")
        transform = MonthlyAggregateTransform(
            value_columns=["ALLSKY_SFC_SW_DWN"],
        )
        result = transform.apply(df)
        assert "month" in result.columns
        assert "month_name" in result.columns
        assert len(result) == 12

    def test_anomaly_transform(self):
        from solar_intelligence.ui.lumen_app import SolarDataSource, AnomalyTransform
        source = SolarDataSource(use_synthetic=True, start_year=2023, end_year=2023)
        df = source.get("daily_solar")
        transform = AnomalyTransform()
        result = transform.apply(df)
        assert "anomaly" in result.columns
        assert "climatology" in result.columns

    def test_pipeline_creation(self):
        from solar_intelligence.ui.lumen_app import create_solar_pipeline
        pipeline = create_solar_pipeline(latitude=28.6, longitude=77.2)
        assert pipeline is not None
        assert pipeline.source is not None
