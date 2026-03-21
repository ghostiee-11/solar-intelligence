"""Tests for visualization module.

Tests that all chart generators produce valid HoloViews/hvPlot objects
without requiring a running Panel server.
"""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd
import pytest

from solar_intelligence.data_loader import generate_synthetic_solar_data
from solar_intelligence.energy_estimator import EnergyEstimator
from solar_intelligence.financial import FinancialAnalyzer
from solar_intelligence.orientation_simulator import OrientationSimulator
from solar_intelligence.solar_analysis import SolarAnalyzer
from solar_intelligence.visualization import SolarVisualizer

hv.extension("bokeh")


@pytest.fixture
def visualizer():
    return SolarVisualizer(width=600, height=350)


@pytest.fixture
def dataset():
    return generate_synthetic_solar_data(lat=28.6, lon=77.2, start_year=2023, end_year=2023)


@pytest.fixture
def analyzer(dataset):
    return SolarAnalyzer(dataset=dataset, latitude=28.6, longitude=77.2)


@pytest.fixture
def estimator():
    return EnergyEstimator(num_panels=10)


@pytest.fixture
def sim_data(dataset):
    sim = OrientationSimulator(
        latitude=28.6, longitude=77.2,
        tilt_angles=[0, 30],
        azimuths={"North": 0, "South": 180, "East": 90, "West": 270},
    )
    ghi = dataset["ALLSKY_SFC_SW_DWN"].values
    return sim.simulate_all_orientations(ghi, year=2023)


@pytest.fixture
def sensitivity_data(dataset):
    sim = OrientationSimulator(latitude=28.6, longitude=77.2)
    ghi = dataset["ALLSKY_SFC_SW_DWN"].values
    return sim.tilt_sensitivity_analysis(ghi, tilt_range=[0, 15, 30, 45])


@pytest.fixture
def profile_data(dataset):
    sim = OrientationSimulator(
        latitude=28.6, longitude=77.2,
        azimuths={"South": 180, "East": 90},
    )
    ghi = dataset["ALLSKY_SFC_SW_DWN"].values
    return sim.daily_profile_by_orientation(ghi, date="2023-06-21", directions=["South", "East"])


@pytest.fixture
def seasonal_data(dataset):
    sim = OrientationSimulator(
        latitude=28.6, longitude=77.2,
        tilt_angles=[0, 30],
        azimuths={"South": 180, "North": 0, "East": 90, "West": 270},
    )
    ghi = dataset["ALLSKY_SFC_SW_DWN"].values
    return sim.seasonal_comparison(ghi, directions=["South", "North"])


# ---------------------------------------------------------------------------
# Irradiance Charts
# ---------------------------------------------------------------------------

class TestIrradianceCharts:

    def test_monthly_irradiance_bar(self, visualizer, analyzer):
        monthly = analyzer.monthly_irradiance()
        chart = visualizer.monthly_irradiance_bar(monthly)
        assert chart is not None
        # hvplot returns an object with a plot method or is an HoloViews element
        assert hasattr(chart, 'opts') or hasattr(chart, 'data')

    def test_daily_irradiance_timeseries(self, visualizer, analyzer):
        rolling = analyzer.rolling_average()
        chart = visualizer.daily_irradiance_timeseries(rolling)
        assert chart is not None

    def test_seasonal_heatmap(self, visualizer, dataset):
        chart = visualizer.seasonal_heatmap(dataset)
        assert chart is not None

    def test_clearsky_vs_actual(self, visualizer, dataset):
        chart = visualizer.clearsky_vs_actual(dataset)
        assert chart is not None

    def test_irradiance_distribution(self, visualizer, dataset):
        chart = visualizer.irradiance_distribution(dataset)
        assert chart is not None


# ---------------------------------------------------------------------------
# Orientation Charts
# ---------------------------------------------------------------------------

class TestOrientationCharts:

    def test_orientation_comparison_bar(self, visualizer, sim_data):
        chart = visualizer.orientation_comparison_bar(sim_data, tilt=30)
        assert chart is not None

    def test_tilt_energy_curve(self, visualizer, sensitivity_data):
        chart = visualizer.tilt_energy_curve(sensitivity_data)
        assert chart is not None

    def test_orientation_heatmap(self, visualizer, sim_data):
        chart = visualizer.orientation_heatmap(sim_data)
        assert chart is not None

    def test_daily_profile_overlay(self, visualizer, profile_data):
        chart = visualizer.daily_profile_overlay(profile_data)
        assert chart is not None

    def test_seasonal_orientation_comparison(self, visualizer, seasonal_data):
        chart = visualizer.seasonal_orientation_comparison(seasonal_data)
        assert chart is not None


# ---------------------------------------------------------------------------
# Energy Charts
# ---------------------------------------------------------------------------

class TestEnergyCharts:

    def test_energy_projection_area(self, visualizer, dataset, estimator):
        monthly = estimator.estimate_monthly_energy(dataset)
        chart = visualizer.energy_projection_area(monthly)
        assert chart is not None

    def test_annual_energy_summary_table(self, visualizer, dataset, estimator):
        summary = estimator.system_summary(dataset)
        table = visualizer.annual_energy_summary_table(summary)
        assert isinstance(table, hv.Table)


# ---------------------------------------------------------------------------
# Map Visualizations
# ---------------------------------------------------------------------------

class TestMapVisualization:

    def test_global_solar_map(self, visualizer):
        lats = np.linspace(-60, 60, 120)
        lons = np.linspace(-180, 180, 360)
        ghi_grid = 7 - 0.08 * np.abs(np.meshgrid(lons, lats)[1])
        chart = visualizer.global_solar_map(lats, lons, ghi_grid)
        assert isinstance(chart, (hv.Image, hv.Overlay))

    def test_location_marker(self, visualizer):
        marker = visualizer.location_marker(28.6, 77.2, "Delhi")
        assert isinstance(marker, hv.Points)

    def test_map_with_marker_overlay(self, visualizer):
        lats = np.linspace(10, 40, 30)
        lons = np.linspace(60, 100, 40)
        ghi = np.random.default_rng(42).uniform(3, 7, (30, 40))
        solar_map = visualizer.global_solar_map(lats, lons, ghi)
        marker = visualizer.location_marker(28.6, 77.2)
        combined = solar_map * marker
        assert combined is not None


# ---------------------------------------------------------------------------
# Financial Charts
# ---------------------------------------------------------------------------

class TestFinancialCharts:

    def test_payback_timeline(self, visualizer):
        fa = FinancialAnalyzer()
        savings = fa.lifetime_savings(5000)
        chart = visualizer.payback_timeline(savings)
        assert chart is not None

    def test_carbon_savings_bar(self, visualizer):
        fa = FinancialAnalyzer()
        savings = fa.lifetime_savings(5000)
        chart = visualizer.carbon_savings_bar(savings)
        assert chart is not None


# ---------------------------------------------------------------------------
# Composite Layouts
# ---------------------------------------------------------------------------

class TestCompositeLayouts:

    def test_overview_layout(self, visualizer, analyzer, dataset):
        monthly = analyzer.monthly_irradiance()
        rolling = analyzer.rolling_average()
        layout = visualizer.create_overview_layout(monthly, rolling, dataset)
        assert isinstance(layout, hv.Layout)

    def test_orientation_layout(self, visualizer, sim_data, sensitivity_data,
                                 profile_data, seasonal_data):
        layout = visualizer.create_orientation_layout(
            sim_data, sensitivity_data, profile_data, seasonal_data,
        )
        assert isinstance(layout, hv.Layout)


# ---------------------------------------------------------------------------
# Datashader Integration
# ---------------------------------------------------------------------------

class TestDatashaderIntegration:

    def test_large_grid_renders(self, visualizer):
        """Test that a 1000x2000 grid (2M points) renders without error."""
        lats = np.linspace(-90, 90, 1000)
        lons = np.linspace(-180, 180, 2000)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        ghi = 7 - 0.08 * np.abs(lat_grid) + np.random.default_rng(42).normal(0, 0.2, lat_grid.shape)
        ghi = np.clip(ghi, 0.5, 9)

        chart = visualizer.global_solar_map(lats, lons, ghi)
        assert isinstance(chart, (hv.Image, hv.Overlay))

    def test_datashader_rasterize_points(self):
        """Test Datashader rasterization of point cloud data."""
        import datashader as ds

        n = 500_000
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "lon": rng.uniform(-180, 180, n),
            "lat": rng.uniform(-90, 90, n),
            "ghi": rng.uniform(1, 9, n),
        })

        canvas = ds.Canvas(plot_width=400, plot_height=200)
        agg = canvas.points(df, "lon", "lat", agg=ds.mean("ghi"))
        assert agg.shape == (200, 400)
        assert float(agg.mean()) > 0
