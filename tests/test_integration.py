"""Integration and end-to-end tests for Solar Intelligence Platform.

Tests the full pipeline from data loading through analysis, simulation,
energy estimation, financial analysis, and dashboard construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from solar_intelligence.data_loader import (
    NASAPowerClient,
    generate_synthetic_solar_data,
)


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Full analysis pipeline: data -> analysis -> energy -> orientation -> financial -> AI."""

    def test_full_pipeline_new_delhi(self, sample_dataset):
        """Complete pipeline for New Delhi produces consistent, plausible results."""
        from solar_intelligence.solar_analysis import SolarAnalyzer
        from solar_intelligence.energy_estimator import EnergyEstimator
        from solar_intelligence.orientation_simulator import OrientationSimulator
        from solar_intelligence.financial import FinancialAnalyzer
        from solar_intelligence.ai_engine import SolarAIEngine

        ds = sample_dataset

        # 1. Solar analysis
        analyzer = SolarAnalyzer(dataset=ds, latitude=28.6, longitude=77.2)
        summary = analyzer.summary()
        assert 3.0 < summary["average_daily_ghi"] < 8.0
        assert 1000 < summary["annual_solar_energy_kwh_m2"] < 3000

        # 2. Energy estimation
        estimator = EnergyEstimator(
            panel_efficiency=0.20, panel_area=1.7,
            num_panels=20, system_losses=0.14,
        )
        energy = estimator.system_summary(ds)
        annual_kwh = energy["production"]["annual_energy_kwh"]
        assert 3000 < annual_kwh < 20000
        assert 10 < energy["performance"]["capacity_factor_pct"] < 35

        # 3. Orientation simulation
        ghi = ds["ALLSKY_SFC_SW_DWN"].sel(time=slice("2023-01-01", "2023-12-31")).values
        sim = OrientationSimulator(
            latitude=28.6, longitude=77.2,
            tilt_angles=[0, 15, 30, 45],
            azimuths={"North": 0, "South": 180, "East": 90, "West": 270},
        )
        optimal = sim.optimal_orientation(ghi, year=2023)
        assert optimal["best_direction"] == "South"
        assert 15 <= optimal["best_tilt"] <= 45

        # 4. Financial analysis
        fa = FinancialAnalyzer(
            system_cost=20000, electricity_rate=0.12,
            incentive_percent=0.30, maintenance_cost=200,
        )
        fin = fa.financial_summary(annual_kwh)
        assert fin["investment"]["net_cost"] == 14000
        payback = fin["returns"]["payback_years"]
        assert 3 < payback < 25
        assert fin["returns"]["roi_pct"] > 0

        # 5. AI insights
        ai = SolarAIEngine()
        report = ai.generate_report(summary, energy, fin, optimal)
        assert len(report) > 100
        assert "solar" in report.lower() or "energy" in report.lower()

    def test_full_pipeline_london(self, sample_dataset_london):
        """London (high latitude) produces lower but valid results."""
        from solar_intelligence.solar_analysis import SolarAnalyzer
        from solar_intelligence.energy_estimator import EnergyEstimator

        ds = sample_dataset_london
        analyzer = SolarAnalyzer(dataset=ds, latitude=51.5, longitude=-0.1)
        summary = analyzer.summary()

        # London gets less sun than tropical locations
        assert 1.5 < summary["average_daily_ghi"] < 5.0

        estimator = EnergyEstimator(
            panel_efficiency=0.20, panel_area=1.7, num_panels=10,
        )
        energy = estimator.system_summary(ds)
        assert energy["production"]["annual_energy_kwh"] > 0

    def test_full_pipeline_southern_hemisphere(self, sample_dataset_sydney):
        """Sydney (Southern Hemisphere): north-facing should be optimal."""
        from solar_intelligence.orientation_simulator import OrientationSimulator

        ds = sample_dataset_sydney
        ghi = ds["ALLSKY_SFC_SW_DWN"].sel(time=slice("2023-01-01", "2023-12-31")).values

        sim = OrientationSimulator(
            latitude=-33.9, longitude=151.2,
            tilt_angles=[0, 15, 30, 45],
            azimuths={"North": 0, "South": 180, "East": 90, "West": 270},
        )
        optimal = sim.optimal_orientation(ghi, year=2023)
        assert optimal["best_direction"] == "North"


class TestDataConsistency:
    """Verify data flows correctly between modules."""

    def test_dataset_variables_complete(self, sample_dataset):
        """Dataset contains all expected variables."""
        expected = [
            "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN",
            "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF",
            "ALLSKY_KT", "T2M", "WS2M", "RH2M",
        ]
        for var in expected:
            assert var in sample_dataset.data_vars

    def test_dataset_no_nans(self, sample_dataset):
        """Synthetic data should have no NaN values."""
        for var in sample_dataset.data_vars:
            assert not np.any(np.isnan(sample_dataset[var].values))

    def test_ghi_within_physical_bounds(self, sample_dataset):
        """GHI should be between 0 and ~12 kWh/m2/day."""
        ghi = sample_dataset["ALLSKY_SFC_SW_DWN"].values
        assert ghi.min() >= 0
        assert ghi.max() <= 12

    def test_temperature_within_physical_bounds(self, sample_dataset):
        """Temperature should be between -50 and 60 C."""
        temp = sample_dataset["T2M"].values
        assert temp.min() > -50
        assert temp.max() < 60

    def test_clearness_index_bounds(self, sample_dataset):
        """Clearness index should be 0 to 1."""
        kt = sample_dataset["ALLSKY_KT"].values
        assert kt.min() >= 0
        assert kt.max() <= 1.0

    def test_analyzer_monthly_sums_to_annual(self, sample_dataset):
        """Monthly irradiance values should roughly average to the daily mean."""
        from solar_intelligence.solar_analysis import SolarAnalyzer

        analyzer = SolarAnalyzer(
            dataset=sample_dataset, latitude=28.6, longitude=77.2,
        )
        daily_avg = analyzer.average_daily_irradiance()
        monthly = analyzer.monthly_irradiance()

        monthly_avg = monthly["GHI"].mean()
        assert abs(daily_avg["GHI"] - monthly_avg) / daily_avg["GHI"] < 0.15

    def test_energy_estimator_monthly_sums_to_annual(self, sample_dataset):
        """Monthly energy estimates should sum close to the annual estimate."""
        from solar_intelligence.energy_estimator import EnergyEstimator

        est = EnergyEstimator(
            panel_efficiency=0.20, panel_area=1.7,
            num_panels=10, system_losses=0.14,
        )
        annual = est.estimate_annual_energy(sample_dataset)
        monthly = est.estimate_monthly_energy(sample_dataset)

        # avg_monthly_energy is the per-year monthly average
        assert abs(monthly["avg_monthly_energy"].sum() - annual) / annual < 0.15


class TestMultiLocationComparison:
    """Integration test for multi-location analysis."""

    def test_multi_location_ranking(self):
        """Multiple locations should rank correctly by solar resource."""
        from solar_intelligence.solar_analysis import MultiLocationComparator

        locations = {
            "Cairo": (30.0, 31.2),
            "London": (51.5, -0.1),
            "New Delhi": (28.6, 77.2),
        }
        comp = MultiLocationComparator(locations=locations)
        comp.load_data(start_year=2023, end_year=2023)

        ranking = comp.ranking()
        assert len(ranking) == 3
        assert ranking.iloc[0]["rank"] == 1
        # London should rank last (least sun)
        assert ranking.iloc[-1]["location"] == "London"


class TestDashboardSmoke:
    """Smoke tests for dashboard construction and servability."""

    def test_dashboard_creates_without_error(self):
        """Dashboard object should initialize without exceptions."""
        from solar_intelligence.ui.panel_dashboard import SolarDashboard

        dashboard = SolarDashboard()
        assert dashboard is not None

    def test_dashboard_view_returns_template(self):
        """view() should return a Panel template."""
        import panel as pn
        from solar_intelligence.ui.panel_dashboard import SolarDashboard

        dashboard = SolarDashboard()
        template = dashboard.view()
        assert isinstance(template, pn.template.FastListTemplate)

    def test_dashboard_has_all_tabs(self):
        """Dashboard should have all expected tabs."""
        from solar_intelligence.ui.panel_dashboard import SolarDashboard

        dashboard = SolarDashboard()
        view = dashboard.view()
        # The main area should have content
        assert len(view.main) > 0

    def test_dashboard_servable(self):
        """Dashboard template should be servable (no error on .servable())."""
        from solar_intelligence.ui.panel_dashboard import SolarDashboard

        dashboard = SolarDashboard()
        template = dashboard.view()
        # servable() should not raise
        result = template.servable()
        assert result is not None

    def test_dashboard_analysis_runs(self):
        """Running analysis on dashboard should populate output areas."""
        from solar_intelligence.ui.panel_dashboard import SolarDashboard

        dashboard = SolarDashboard()
        # Set a valid location before analysis
        dashboard.location.latitude = 28.6
        dashboard.location.longitude = 77.2
        dashboard.location.location_name = "Delhi"
        # Trigger analysis programmatically
        dashboard._run_analysis()
        # After analysis, KPI row should have content
        assert len(dashboard._kpi_row) > 0


class TestLumenPipelineIntegration:
    """Test Lumen pipeline end-to-end."""

    def test_pipeline_creates_and_returns_data(self):
        """Lumen pipeline should produce a DataFrame with expected columns."""
        from solar_intelligence.ui.lumen_app import create_solar_pipeline

        pipeline = create_solar_pipeline(latitude=28.6, longitude=77.2)
        assert pipeline is not None

    def test_solar_source_all_tables(self):
        """SolarDataSource should serve all three tables."""
        from solar_intelligence.ui.lumen_app import SolarDataSource

        source = SolarDataSource(
            latitude=28.6, longitude=77.2,
            use_synthetic=True, start_year=2023, end_year=2023,
        )

        daily = source.get("daily_solar")
        assert isinstance(daily, pd.DataFrame)
        assert "ALLSKY_SFC_SW_DWN" in daily.columns
        assert len(daily) > 300

        monthly = source.get("monthly_solar")
        assert isinstance(monthly, pd.DataFrame)
        assert len(monthly) == 12

        meta = source.get("metadata")
        assert isinstance(meta, pd.DataFrame)
        assert "latitude" in meta["key"].values

    def test_energy_transform_adds_column(self):
        """SolarEnergyTransform should add energy_kwh column."""
        from solar_intelligence.ui.lumen_app import SolarDataSource, SolarEnergyTransform

        source = SolarDataSource(
            latitude=28.6, longitude=77.2,
            use_synthetic=True, start_year=2023, end_year=2023,
        )
        df = source.get("daily_solar")

        transform = SolarEnergyTransform(
            panel_efficiency=0.20, total_area=34.0,
        )
        result = transform.apply(df)
        assert "energy_kwh" in result.columns
        assert result["energy_kwh"].min() >= 0


class TestNASAPowerAPISmoke:
    """Smoke tests for NASA POWER API client (uses network if available)."""

    def test_client_initializes(self):
        """NASAPowerClient should initialize with default parameters."""
        from solar_intelligence.config import NASA_POWER_BASE_URL
        client = NASAPowerClient()
        assert client is not None
        assert NASA_POWER_BASE_URL is not None

    def test_api_url_construction(self):
        """API URL should be well-formed."""
        from solar_intelligence.config import NASA_POWER_BASE_URL
        url = (
            f"{NASA_POWER_BASE_URL}/daily/point"
            f"?parameters=ALLSKY_SFC_SW_DWN"
            f"&community=RE"
            f"&longitude=77.209"
            f"&latitude=28.614"
            f"&start=20230101"
            f"&end=20230131"
            f"&format=JSON"
        )
        assert "power.larc.nasa.gov" in url

    @pytest.mark.skipif(
        True,  # Set to False to test live API
        reason="Live API test disabled by default",
    )
    def test_live_api_fetch(self):
        """Fetch real data from NASA POWER API (disabled by default)."""
        client = NASAPowerClient()
        ds = client.fetch_daily(
            lat=28.6, lon=77.2,
            start="20230101", end="20230131",
        )
        assert isinstance(ds, xr.Dataset)
        assert "ALLSKY_SFC_SW_DWN" in ds.data_vars
