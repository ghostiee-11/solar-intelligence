"""Tests for new features: global grid, ERA5/SARAH-3, multi-location, Datashader.

Covers:
- generate_global_solar_grid() for Datashader maps
- ClimateDatasetLoader for ERA5 and SARAH-3 file loading
- generate_multi_location_data() utility
- MultiLocationComparator for cross-city analysis
- SolarVisualizer Datashader and multi-location chart methods
"""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from solar_intelligence.data_loader import (
    ClimateDatasetLoader,
    generate_global_solar_grid,
    generate_multi_location_data,
    generate_synthetic_solar_data,
)
from solar_intelligence.solar_analysis import MultiLocationComparator
from solar_intelligence.visualization import SolarVisualizer

hv.extension("bokeh")


# ---------------------------------------------------------------------------
# Global Solar Grid Tests
# ---------------------------------------------------------------------------

class TestGlobalSolarGrid:

    def test_default_grid_shape(self):
        ds = generate_global_solar_grid(resolution=2.0)
        assert "GHI" in ds
        assert "lat" in ds.dims
        assert "lon" in ds.dims
        assert ds["GHI"].shape[0] > 50  # lat points
        assert ds["GHI"].shape[1] > 150  # lon points

    def test_high_resolution_grid(self):
        ds = generate_global_solar_grid(
            resolution=0.5,
            lat_range=(-30, 30),
            lon_range=(-30, 30),
        )
        n_lat = len(ds.lat)
        n_lon = len(ds.lon)
        total_points = n_lat * n_lon
        assert total_points > 10000

    def test_ghi_range_is_physical(self):
        ds = generate_global_solar_grid(resolution=2.0)
        ghi = ds["GHI"].values
        assert ghi.min() >= 0.5
        assert ghi.max() <= 9.0

    def test_equator_higher_than_poles(self):
        ds = generate_global_solar_grid(resolution=2.0)
        equator_ghi = float(ds["GHI"].sel(lat=0, method="nearest").mean())
        pole_ghi = float(ds["GHI"].sel(lat=58, method="nearest").mean())
        assert equator_ghi > pole_ghi

    def test_desert_boost_applied(self):
        ds = generate_global_solar_grid(resolution=2.0)
        # Sahara region should be higher than same-latitude ocean
        sahara_ghi = float(ds["GHI"].sel(lat=25, lon=30, method="nearest"))
        ocean_ghi = float(ds["GHI"].sel(lat=25, lon=-60, method="nearest"))
        assert sahara_ghi > ocean_ghi

    def test_xarray_dataset_attrs(self):
        ds = generate_global_solar_grid(resolution=5.0)
        assert "source" in ds.attrs
        assert "description" in ds.attrs


# ---------------------------------------------------------------------------
# ERA5 / SARAH-3 Loader Tests
# ---------------------------------------------------------------------------

class TestClimateDatasetLoader:

    @pytest.fixture
    def era5_file(self, tmp_path):
        """Create a mock ERA5 NetCDF file."""
        times = pd.date_range("2023-01-01", periods=365, freq="D")
        ds = xr.Dataset(
            {
                "ssrd": ("time", np.random.default_rng(42).uniform(5e6, 25e6, 365)),
                "t2m": ("time", np.random.default_rng(42).uniform(270, 310, 365)),
            },
            coords={"time": times},
        )
        path = tmp_path / "era5_test.nc"
        ds.to_netcdf(path)
        return path

    @pytest.fixture
    def sarah_file(self, tmp_path):
        """Create a mock SARAH-3 NetCDF file."""
        times = pd.date_range("2023-01-01", periods=365, freq="D")
        ds = xr.Dataset(
            {
                "SIS": ("time", np.random.default_rng(42).uniform(50, 300, 365)),
                "SID": ("time", np.random.default_rng(42).uniform(30, 250, 365)),
            },
            coords={"time": times},
        )
        path = tmp_path / "sarah3_test.nc"
        ds.to_netcdf(path)
        return path

    def test_load_era5_basic(self, era5_file):
        loader = ClimateDatasetLoader()
        ds = loader.load_era5(era5_file)
        assert "ALLSKY_SFC_SW_DWN" in ds
        assert "T2M" in ds

    def test_era5_ssrd_conversion(self, era5_file):
        loader = ClimateDatasetLoader()
        ds = loader.load_era5(era5_file)
        ghi = ds["ALLSKY_SFC_SW_DWN"].values
        # J/m2 -> kWh/m2: original range 5e6-25e6 J/m2 -> ~1.4-6.9 kWh/m2
        assert all(ghi >= 0)
        assert all(ghi < 10)

    def test_era5_temperature_conversion(self, era5_file):
        loader = ClimateDatasetLoader()
        ds = loader.load_era5(era5_file)
        temp = ds["T2M"].values
        # Kelvin -> Celsius: 270-310K -> -3 to 37C
        assert all(temp < 50)
        assert all(temp > -10)

    def test_era5_attrs(self, era5_file):
        loader = ClimateDatasetLoader()
        ds = loader.load_era5(era5_file)
        assert "ERA5" in ds.attrs["source"]

    def test_era5_file_not_found(self):
        loader = ClimateDatasetLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_era5("/nonexistent/file.nc")

    def test_load_sarah3_basic(self, sarah_file):
        loader = ClimateDatasetLoader()
        ds = loader.load_sarah3(sarah_file)
        assert "ALLSKY_SFC_SW_DWN" in ds
        assert "ALLSKY_SFC_SW_DNI" in ds

    def test_sarah3_conversion(self, sarah_file):
        loader = ClimateDatasetLoader()
        ds = loader.load_sarah3(sarah_file)
        ghi = ds["ALLSKY_SFC_SW_DWN"].values
        # W/m2 * 24 / 1000: original 50-300 W/m2 -> 1.2-7.2 kWh/m2/day
        assert all(ghi >= 0)
        assert all(ghi < 10)

    def test_sarah3_attrs(self, sarah_file):
        loader = ClimateDatasetLoader()
        ds = loader.load_sarah3(sarah_file)
        assert "SARAH" in ds.attrs["source"]

    def test_sarah3_file_not_found(self):
        loader = ClimateDatasetLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_sarah3("/nonexistent/file.nc")


# ---------------------------------------------------------------------------
# Multi-Location Data Generation Tests
# ---------------------------------------------------------------------------

class TestMultiLocationData:

    def test_generate_multiple(self):
        locations = {
            "Delhi": (28.6, 77.2),
            "London": (51.5, -0.1),
            "Sydney": (-33.9, 151.2),
        }
        datasets = generate_multi_location_data(locations, 2023, 2023)
        assert len(datasets) == 3
        assert all(isinstance(ds, xr.Dataset) for ds in datasets.values())
        assert "Delhi" in datasets
        assert "London" in datasets
        assert "Sydney" in datasets

    def test_datasets_have_correct_coords(self):
        locations = {"Tokyo": (35.7, 139.7)}
        datasets = generate_multi_location_data(locations, 2023, 2023)
        ds = datasets["Tokyo"]
        assert float(ds.latitude) == 35.7
        assert float(ds.longitude) == 139.7


# ---------------------------------------------------------------------------
# Multi-Location Comparator Tests
# ---------------------------------------------------------------------------

class TestMultiLocationComparator:

    @pytest.fixture
    def comparator(self):
        locations = {
            "Delhi": (28.6, 77.2),
            "London": (51.5, -0.1),
            "Cairo": (30.0, 31.2),
        }
        comp = MultiLocationComparator(locations=locations)
        comp.load_data(start_year=2023, end_year=2023)
        return comp

    def test_compare_ghi(self, comparator):
        df = comparator.compare_ghi()
        assert len(df) == 3
        assert "location" in df.columns
        assert "GHI" in df.columns
        assert "annual_kwh_m2" in df.columns
        assert all(df["GHI"] > 0)

    def test_compare_monthly(self, comparator):
        df = comparator.compare_monthly()
        assert "location" in df.columns
        assert "month" in df.columns
        assert "GHI" in df.columns
        # 3 locations × 12 months = 36 rows
        assert len(df) == 36

    def test_compare_seasonal(self, comparator):
        df = comparator.compare_seasonal()
        assert "location" in df.columns
        assert "GHI_mean" in df.columns
        # 3 locations × 4 seasons = 12 rows
        assert len(df) == 12

    def test_ranking(self, comparator):
        df = comparator.ranking()
        assert "rank" in df.columns
        assert df["rank"].tolist() == [1, 2, 3]
        # Check sorted descending by annual_kwh_m2
        assert df["annual_kwh_m2"].is_monotonic_decreasing

    def test_summary(self, comparator):
        summaries = comparator.summary()
        assert len(summaries) == 3
        for name, summary in summaries.items():
            assert "average_daily_ghi" in summary
            assert "annual_solar_energy_kwh_m2" in summary

    def test_delhi_better_than_london(self, comparator):
        df = comparator.compare_ghi()
        delhi_ghi = df[df["location"] == "Delhi"]["GHI"].iloc[0]
        london_ghi = df[df["location"] == "London"]["GHI"].iloc[0]
        assert delhi_ghi > london_ghi


# ---------------------------------------------------------------------------
# Datashader Visualization Tests
# ---------------------------------------------------------------------------

class TestDatashaderVisualization:

    @pytest.fixture
    def visualizer(self):
        return SolarVisualizer(width=600, height=350)

    def test_datashader_global_map(self, visualizer):
        ds = generate_global_solar_grid(resolution=2.0)
        chart = visualizer.datashader_global_map(ds)
        assert chart is not None
        assert isinstance(chart, (hv.Image, hv.Overlay))

    def test_datashader_global_map_custom_range(self, visualizer):
        ds = generate_global_solar_grid(
            resolution=1.0,
            lat_range=(10, 40),
            lon_range=(60, 100),
        )
        chart = visualizer.datashader_global_map(ds)
        assert chart is not None

    def test_datashader_point_density(self, visualizer):
        rng = np.random.default_rng(42)
        n = 100_000
        df = pd.DataFrame({
            "lon": rng.uniform(-180, 180, n),
            "lat": rng.uniform(-60, 60, n),
            "ghi": rng.uniform(1, 9, n),
        })
        chart = visualizer.datashader_point_density(df)
        assert chart is not None
        assert isinstance(chart, (hv.Image, hv.Overlay))

    def test_datashader_million_points(self, visualizer):
        """Test that 1M+ point rendering works without error."""
        ds = generate_global_solar_grid(resolution=0.25, lat_range=(-60, 60), lon_range=(-180, 180))
        total = ds["GHI"].shape[0] * ds["GHI"].shape[1]
        assert total > 500_000  # Should be ~700K+ points
        chart = visualizer.datashader_global_map(ds)
        assert isinstance(chart, (hv.Image, hv.Overlay))


# ---------------------------------------------------------------------------
# Multi-Location Visualization Tests
# ---------------------------------------------------------------------------

class TestMultiLocationVisualization:

    @pytest.fixture
    def visualizer(self):
        return SolarVisualizer(width=600, height=350)

    @pytest.fixture
    def comparator(self):
        locations = {
            "Delhi": (28.6, 77.2),
            "London": (51.5, -0.1),
            "Sydney": (-33.9, 151.2),
        }
        comp = MultiLocationComparator(locations=locations)
        comp.load_data(start_year=2023, end_year=2023)
        return comp

    def test_multi_location_bar(self, visualizer, comparator):
        df = comparator.compare_ghi()
        chart = visualizer.multi_location_bar(df)
        assert chart is not None

    def test_multi_location_monthly(self, visualizer, comparator):
        df = comparator.compare_monthly()
        chart = visualizer.multi_location_monthly(df)
        assert chart is not None

    def test_multi_location_ranking_table(self, visualizer, comparator):
        df = comparator.ranking()
        table = visualizer.multi_location_radar_table(df)
        assert isinstance(table, hv.Table)
