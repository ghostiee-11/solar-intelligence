"""Tests for ERA5 client, DualSourceLoader, DualSourceAnalyzer, and dual-source visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from solar_intelligence.data_loader import (
    DualSourceLoader,
    ERA5Client,
    generate_synthetic_solar_data,
)
from solar_intelligence.solar_analysis import DualSourceAnalyzer
from solar_intelligence.visualization import SolarVisualizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nasa_dataset():
    """Synthetic NASA POWER-style dataset."""
    return generate_synthetic_solar_data(lat=28.6, lon=77.2, start_year=2023, end_year=2023)


@pytest.fixture
def era5_dataset():
    """Synthetic ERA5-style dataset with slight offset to simulate source differences."""
    ds = generate_synthetic_solar_data(lat=28.6, lon=77.2, start_year=2023, end_year=2023)
    # Add small bias to simulate ERA5 being slightly different
    rng = np.random.default_rng(seed=99)
    for var in ds.data_vars:
        noise = rng.normal(0, 0.1, ds[var].shape)
        ds[var].values = ds[var].values + noise.astype(ds[var].dtype)
    ds.attrs["source"] = "ERA5 (Copernicus Climate Data Store)"
    return ds


@pytest.fixture
def dual_datasets(nasa_dataset, era5_dataset):
    """Both sources as a dict."""
    return {"NASA POWER": nasa_dataset, "ERA5": era5_dataset}


@pytest.fixture
def visualizer():
    return SolarVisualizer(width=500, height=300)


# ---------------------------------------------------------------------------
# ERA5Client Tests
# ---------------------------------------------------------------------------

class TestERA5Client:
    """Test ERA5 client initialization and configuration."""

    def test_client_initializes(self):
        """ERA5Client should initialize with defaults."""
        client = ERA5Client()
        assert client.dataset == "reanalysis-era5-single-levels"
        assert len(client.variables) > 0

    def test_cache_key_deterministic(self):
        """Same inputs produce same cache key."""
        client = ERA5Client()
        key1 = client._era5_cache_key(28.6, 77.2, "20230101", "20231231")
        key2 = client._era5_cache_key(28.6, 77.2, "20230101", "20231231")
        assert key1 == key2

    def test_cache_key_varies_by_location(self):
        """Different locations produce different cache keys."""
        client = ERA5Client()
        key1 = client._era5_cache_key(28.6, 77.2, "20230101", "20231231")
        key2 = client._era5_cache_key(51.5, -0.1, "20230101", "20231231")
        assert key1 != key2

    def test_cache_key_format(self):
        """Cache key should be .nc file."""
        client = ERA5Client()
        key = client._era5_cache_key(28.6, 77.2, "20230101", "20231231")
        assert key.startswith("era5_")
        assert key.endswith(".nc")

    def test_check_cdsapi_raises_without_package(self):
        """_check_cdsapi should raise ImportError with helpful message if cdsapi missing."""
        client = ERA5Client()
        try:
            client._check_cdsapi()
        except ImportError as e:
            assert "cdsapi" in str(e)
            assert "pip install" in str(e)


# ---------------------------------------------------------------------------
# DualSourceLoader Tests
# ---------------------------------------------------------------------------

class TestDualSourceLoader:
    """Test dual-source loading and alignment."""

    def test_align_datasets(self, dual_datasets):
        """align_datasets should produce a DataFrame with source columns."""
        aligned = DualSourceLoader.align_datasets(dual_datasets)
        assert isinstance(aligned, pd.DataFrame)
        assert "NASA POWER" in aligned.columns
        assert "ERA5" in aligned.columns
        assert len(aligned) > 300

    def test_align_datasets_single_source(self, nasa_dataset):
        """Single source should still produce valid DataFrame."""
        aligned = DualSourceLoader.align_datasets({"NASA": nasa_dataset})
        assert isinstance(aligned, pd.DataFrame)
        assert "NASA" in aligned.columns

    def test_align_datasets_empty(self):
        """Empty dict should return empty DataFrame."""
        aligned = DualSourceLoader.align_datasets({})
        assert aligned.empty

    def test_comparison_stats(self, dual_datasets):
        """comparison_stats should compute correlation, RMSE, bias."""
        stats = DualSourceLoader.comparison_stats(dual_datasets)
        assert "sources" in stats
        assert "comparison" in stats
        comp = stats["comparison"]
        assert "correlation" in comp
        assert "rmse" in comp
        assert "bias" in comp
        assert "mae" in comp
        # Synthetic data with small noise should correlate highly
        assert comp["correlation"] > 0.95

    def test_comparison_stats_single_source(self, nasa_dataset):
        """Single source should return error."""
        stats = DualSourceLoader.comparison_stats({"NASA": nasa_dataset})
        assert "error" in stats

    def test_comparison_stats_rmse_reasonable(self, dual_datasets):
        """RMSE should be small for datasets with minor noise."""
        stats = DualSourceLoader.comparison_stats(dual_datasets)
        rmse = stats["comparison"]["rmse"]
        # Small noise added -> RMSE should be < 1
        assert rmse < 1.0

    def test_loader_initializes(self):
        """DualSourceLoader should initialize with clients."""
        loader = DualSourceLoader(use_era5=False, use_nasa=True)
        assert loader._nasa_client is not None
        assert loader._era5_client is None


# ---------------------------------------------------------------------------
# DualSourceAnalyzer Tests
# ---------------------------------------------------------------------------

class TestDualSourceAnalyzer:
    """Test dual-source solar analysis."""

    def test_source_summaries(self, dual_datasets):
        """source_summaries should return dict per source."""
        analyzer = DualSourceAnalyzer(
            datasets=dual_datasets, latitude=28.6, longitude=77.2,
        )
        summaries = analyzer.source_summaries()
        assert "NASA POWER" in summaries
        assert "ERA5" in summaries
        for name, summary in summaries.items():
            assert "average_daily_ghi" in summary
            assert summary["average_daily_ghi"] > 0

    def test_compare_daily_ghi(self, dual_datasets):
        """compare_daily_ghi should return aligned DataFrame."""
        analyzer = DualSourceAnalyzer(
            datasets=dual_datasets, latitude=28.6, longitude=77.2,
        )
        aligned = analyzer.compare_daily_ghi()
        assert isinstance(aligned, pd.DataFrame)
        assert len(aligned.columns) == 2

    def test_compare_monthly(self, dual_datasets):
        """compare_monthly should return 12-row DataFrame with source columns."""
        analyzer = DualSourceAnalyzer(
            datasets=dual_datasets, latitude=28.6, longitude=77.2,
        )
        monthly = analyzer.compare_monthly()
        assert len(monthly) == 12
        assert "month_name" in monthly.columns
        assert "NASA POWER" in monthly.columns
        assert "ERA5" in monthly.columns

    def test_cross_validation(self, dual_datasets):
        """cross_validation should return stats with correlation."""
        analyzer = DualSourceAnalyzer(
            datasets=dual_datasets, latitude=28.6, longitude=77.2,
        )
        stats = analyzer.cross_validation()
        assert "comparison" in stats
        assert stats["comparison"]["correlation"] > 0.9

    def test_agreement_report(self, dual_datasets):
        """agreement_report should return human-readable text."""
        analyzer = DualSourceAnalyzer(
            datasets=dual_datasets, latitude=28.6, longitude=77.2,
        )
        report = analyzer.agreement_report()
        assert "Cross-Validation" in report
        assert "Correlation" in report
        assert "RMSE" in report
        assert len(report) > 100

    def test_agreement_report_quality_rating(self, dual_datasets):
        """Report should contain a quality rating."""
        analyzer = DualSourceAnalyzer(
            datasets=dual_datasets, latitude=28.6, longitude=77.2,
        )
        report = analyzer.agreement_report()
        assert any(
            word in report
            for word in ["Excellent", "Good", "Moderate", "Poor"]
        )


# ---------------------------------------------------------------------------
# Dual-Source Visualization Tests
# ---------------------------------------------------------------------------

class TestDualSourceVisualization:
    """Test dual-source comparison charts."""

    def test_dual_source_timeseries(self, dual_datasets, visualizer):
        """Timeseries overlay should render for two sources."""
        import holoviews as hv
        aligned = DualSourceLoader.align_datasets(dual_datasets)
        result = visualizer.dual_source_timeseries(aligned)
        assert isinstance(result, hv.Overlay)

    def test_dual_source_monthly_bar(self, dual_datasets, visualizer):
        """Monthly bar should render grouped bars."""
        import holoviews as hv
        analyzer = DualSourceAnalyzer(
            datasets=dual_datasets, latitude=28.6, longitude=77.2,
        )
        monthly = analyzer.compare_monthly()
        result = visualizer.dual_source_monthly_bar(monthly)
        assert isinstance(result, hv.Bars)

    def test_dual_source_scatter(self, dual_datasets, visualizer):
        """Scatter plot should render with 1:1 line."""
        import holoviews as hv
        aligned = DualSourceLoader.align_datasets(dual_datasets)
        result = visualizer.dual_source_scatter(aligned)
        assert isinstance(result, hv.Overlay)

    def test_dual_source_difference_heatmap(self, dual_datasets, visualizer):
        """Difference heatmap should render."""
        import holoviews as hv
        aligned = DualSourceLoader.align_datasets(dual_datasets)
        result = visualizer.dual_source_difference_heatmap(aligned)
        assert isinstance(result, hv.HeatMap)

    def test_source_location_map(self, visualizer):
        """Location map with source annotations should render."""
        import holoviews as hv
        ghi_values = {"NASA POWER": 5.5, "ERA5": 5.3}
        result = visualizer.source_location_map(28.6, 77.2, ghi_values)
        assert isinstance(result, (hv.Overlay, hv.NdOverlay))


# ---------------------------------------------------------------------------
# Dashboard Integration Tests
# ---------------------------------------------------------------------------

class TestDashboardDualSource:
    """Test dashboard dual-source integration."""

    def test_dashboard_has_era5_toggle(self):
        """Dashboard should have ERA5 toggle widget."""
        from solar_intelligence.ui.panel_dashboard import SolarDashboard
        dashboard = SolarDashboard()
        assert hasattr(dashboard, "_use_era5")
        assert dashboard._use_era5.value is False

    def test_dashboard_has_dual_source_area(self):
        """Dashboard should have dual source output area."""
        from solar_intelligence.ui.panel_dashboard import SolarDashboard
        dashboard = SolarDashboard()
        assert hasattr(dashboard, "_dual_source_area")

    def test_dashboard_view_has_data_sources_tab(self):
        """Dashboard view should include Data Sources tab."""
        from solar_intelligence.ui.panel_dashboard import SolarDashboard
        dashboard = SolarDashboard()
        view = dashboard.view()
        # Check tabs contain Data Sources
        tabs = view.main[0]
        assert len(tabs) == 7  # 6 original + Data Sources
        assert dashboard._dual_source_area in tabs.objects

    def test_dual_source_update_single_source(self):
        """_update_dual_source with single source should show instructions."""
        from solar_intelligence.ui.panel_dashboard import SolarDashboard
        ds = generate_synthetic_solar_data(lat=28.6, lon=77.2, start_year=2023, end_year=2023)
        dashboard = SolarDashboard()
        dashboard._update_dual_source({"NASA POWER": ds}, 28.6, 77.2)
        content = str(dashboard._dual_source_area[0].object)
        assert "ERA5" in content

    def test_dual_source_update_two_sources(self, dual_datasets):
        """_update_dual_source with two sources should show comparison charts."""
        from solar_intelligence.ui.panel_dashboard import SolarDashboard
        dashboard = SolarDashboard()
        dashboard._update_dual_source(dual_datasets, 28.6, 77.2)
        # Should have report + 2 rows of charts
        assert len(dashboard._dual_source_area) >= 3
