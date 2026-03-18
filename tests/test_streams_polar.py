"""Tests for Phase 1: HoloViews Streams, Polar Plot, Dynamic Datashader.

Covers:
- orientation_polar_plot() — radial energy visualization
- interactive_map_with_tap() — Tap stream on solar map
- interactive_timeseries_with_range() — RangeX stream
- interactive_orientation_selector() — Selection1D stream
- dynamic_rasterized_map() — Datashader rasterize() integration
"""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from holoviews import streams

from solar_intelligence.data_loader import (
    generate_global_solar_grid,
    generate_synthetic_solar_data,
)
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
def sim_data(dataset):
    sim = OrientationSimulator(
        latitude=28.6, longitude=77.2,
        tilt_angles=[0, 30],
        azimuths={"North": 0, "South": 180, "East": 90, "West": 270},
    )
    ghi = dataset["ALLSKY_SFC_SW_DWN"].values
    return sim.simulate_all_orientations(ghi, year=2023)


# ---------------------------------------------------------------------------
# Polar Plot Tests
# ---------------------------------------------------------------------------

class TestPolarPlot:

    def test_polar_plot_returns_overlay_or_points(self, visualizer, sim_data):
        chart = visualizer.orientation_polar_plot(sim_data, tilt=30)
        assert chart is not None
        # Should be either hv.Overlay (points + labels) or hv.Points
        assert isinstance(chart, (hv.Overlay, hv.Points))

    def test_polar_plot_different_tilt(self, visualizer, sim_data):
        chart = visualizer.orientation_polar_plot(sim_data, tilt=0)
        assert chart is not None

    def test_polar_plot_has_directions(self, visualizer, sim_data):
        chart = visualizer.orientation_polar_plot(sim_data, tilt=30)
        # Extract points from the overlay
        if isinstance(chart, hv.Overlay):
            points = [el for el in chart if isinstance(el, hv.Points)]
            assert len(points) > 0
            pts = points[0]
        else:
            pts = chart
        # Should have 4 directions (N, S, E, W)
        assert len(pts) == 4


# ---------------------------------------------------------------------------
# HoloViews Streams Tests
# ---------------------------------------------------------------------------

class TestTapStream:

    def test_interactive_map_returns_tuple(self, visualizer):
        lats = np.linspace(-30, 30, 60)
        lons = np.linspace(-60, 60, 120)
        ghi = 7 - 0.08 * np.abs(np.meshgrid(lons, lats)[1])
        result = visualizer.interactive_map_with_tap(lats, lons, ghi)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_tap_stream_instance(self, visualizer):
        lats = np.linspace(-30, 30, 30)
        lons = np.linspace(-60, 60, 60)
        ghi = np.random.default_rng(42).uniform(3, 7, (30, 60))
        _, tap_stream = visualizer.interactive_map_with_tap(lats, lons, ghi)
        assert isinstance(tap_stream, streams.Tap)
        assert hasattr(tap_stream, 'x')
        assert hasattr(tap_stream, 'y')

    def test_tap_stream_default_coords(self, visualizer):
        lats = np.linspace(-10, 10, 20)
        lons = np.linspace(-20, 20, 40)
        ghi = np.ones((20, 40)) * 5.0
        _, tap_stream = visualizer.interactive_map_with_tap(lats, lons, ghi)
        assert tap_stream.x == 0
        assert tap_stream.y == 0


class TestRangeXStream:

    def test_timeseries_with_range_returns_tuple(self, visualizer, dataset):
        analyzer = SolarAnalyzer(dataset=dataset, latitude=28.6, longitude=77.2)
        rolling = analyzer.rolling_average()
        result = visualizer.interactive_timeseries_with_range(rolling)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_range_stream_instance(self, visualizer, dataset):
        analyzer = SolarAnalyzer(dataset=dataset, latitude=28.6, longitude=77.2)
        rolling = analyzer.rolling_average()
        _, range_stream = visualizer.interactive_timeseries_with_range(rolling)
        assert isinstance(range_stream, streams.RangeX)


class TestSelection1DStream:

    def test_orientation_selector_returns_tuple(self, visualizer, sim_data):
        result = visualizer.interactive_orientation_selector(sim_data, tilt=30)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_selection_stream_instance(self, visualizer, sim_data):
        _, sel_stream = visualizer.interactive_orientation_selector(sim_data, tilt=30)
        assert isinstance(sel_stream, streams.Selection1D)

    def test_bars_element(self, visualizer, sim_data):
        bars, _ = visualizer.interactive_orientation_selector(sim_data, tilt=30)
        assert isinstance(bars, hv.Bars)


# ---------------------------------------------------------------------------
# Dynamic Datashader Rasterization Tests
# ---------------------------------------------------------------------------

class TestDynamicRasterization:

    def test_dynamic_rasterized_map_returns_dynamicmap(self, visualizer):
        ds = generate_global_solar_grid(resolution=2.0)
        result = visualizer.dynamic_rasterized_map(ds)
        # rasterize() returns a DynamicMap or HoloMap-like object
        assert result is not None

    def test_dynamic_rasterized_large_grid(self, visualizer):
        ds = generate_global_solar_grid(
            resolution=0.5,
            lat_range=(-60, 60),
            lon_range=(-180, 180),
        )
        total = ds["GHI"].shape[0] * ds["GHI"].shape[1]
        assert total > 100_000
        result = visualizer.dynamic_rasterized_map(ds)
        assert result is not None

    def test_dynamic_rasterized_custom_var(self, visualizer):
        ds = generate_global_solar_grid(resolution=3.0)
        ds = ds.rename({"GHI": "irradiance"})
        result = visualizer.dynamic_rasterized_map(ds, ghi_var="irradiance")
        assert result is not None
