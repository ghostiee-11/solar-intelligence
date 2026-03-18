"""Tests for solar_analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_intelligence.solar_analysis import SolarAnalyzer


class TestSolarAnalyzer:
    """Tests for SolarAnalyzer class."""

    def _make_analyzer(self, dataset, lat=28.6139, lon=77.2090):
        return SolarAnalyzer(dataset=dataset, latitude=lat, longitude=lon)

    def test_no_dataset_raises(self):
        analyzer = SolarAnalyzer()
        with pytest.raises(ValueError, match="No dataset loaded"):
            analyzer.average_daily_irradiance()

    def test_average_daily_irradiance(self, sample_dataset):
        analyzer = self._make_analyzer(sample_dataset)
        avg = analyzer.average_daily_irradiance()

        assert "GHI" in avg
        assert "DNI" in avg
        assert "DHI" in avg
        # New Delhi should have ~4-7 kWh/m²/day GHI
        assert 2.0 < avg["GHI"] < 9.0, f"GHI {avg['GHI']} outside plausible range"

    def test_monthly_irradiance_shape(self, sample_dataset):
        analyzer = self._make_analyzer(sample_dataset)
        monthly = analyzer.monthly_irradiance()

        assert len(monthly) == 12
        assert "GHI" in monthly.columns
        assert "month_name" in monthly.columns
        assert monthly.index.name == "month"

    def test_monthly_irradiance_summer_vs_winter(self, sample_dataset):
        """Northern hemisphere: summer months should have higher GHI."""
        analyzer = self._make_analyzer(sample_dataset, lat=28.6)
        monthly = analyzer.monthly_irradiance()

        summer_avg = monthly.loc[[5, 6, 7], "GHI"].mean()
        winter_avg = monthly.loc[[11, 12, 1], "GHI"].mean()
        assert summer_avg > winter_avg, "Summer GHI should exceed winter GHI"

    def test_seasonal_patterns(self, sample_dataset):
        analyzer = self._make_analyzer(sample_dataset)
        seasonal = analyzer.seasonal_patterns()

        assert len(seasonal) == 4
        assert all(s in seasonal.index for s in ["DJF", "MAM", "JJA", "SON"])
        assert "GHI_mean" in seasonal.columns
        assert "GHI_std" in seasonal.columns

    def test_annual_solar_energy(self, sample_dataset):
        analyzer = self._make_analyzer(sample_dataset)
        annual = analyzer.annual_solar_energy()

        # Should be daily_avg × 365.25
        avg = analyzer.average_daily_irradiance()["GHI"]
        expected = avg * 365.25
        assert abs(annual - expected) < 0.1

    def test_annual_energy_plausible(self, sample_dataset):
        analyzer = self._make_analyzer(sample_dataset, lat=28.6)
        annual = analyzer.annual_solar_energy()
        # New Delhi: ~1500-2500 kWh/m²/year
        assert 1000 < annual < 3000

    def test_clearsky_index(self, sample_dataset):
        analyzer = self._make_analyzer(sample_dataset)
        kt = analyzer.clearsky_index()

        assert "clearsky_index" in kt.columns
        assert len(kt) == 12  # 12 months
        # Clearsky index should be between 0 and 1
        valid = kt["clearsky_index"].dropna()
        assert all(0 < v < 1.5 for v in valid), "Clearsky index out of range"

    def test_peak_sun_hours(self, sample_dataset):
        analyzer = self._make_analyzer(sample_dataset)
        psh = analyzer.peak_sun_hours()

        assert "peak_sun_hours" in psh.columns
        assert "month_name" in psh.columns
        assert len(psh) == 12

    def test_anomaly_detection(self, sample_dataset):
        analyzer = self._make_analyzer(sample_dataset)
        anomalies = analyzer.anomaly_detection()

        assert "anomaly" in anomalies.columns
        assert "GHI" in anomalies.columns
        # Anomalies should sum to approximately 0
        assert abs(anomalies["anomaly"].mean()) < 0.5

    def test_rolling_average(self, sample_dataset):
        analyzer = self._make_analyzer(sample_dataset)
        rolled = analyzer.rolling_average(window=30)

        assert "GHI" in rolled.columns
        assert "GHI_rolling_30d" in rolled.columns
        # Rolling average should be smoother (lower std)
        raw_std = rolled["GHI"].std()
        smooth_std = rolled["GHI_rolling_30d"].dropna().std()
        assert smooth_std < raw_std

    def test_variability_index(self, sample_dataset):
        analyzer = self._make_analyzer(sample_dataset)
        var_idx = analyzer.variability_index()

        assert "variability_index" in var_idx.columns
        assert len(var_idx) == 12

    def test_summary(self, sample_dataset):
        analyzer = self._make_analyzer(sample_dataset)
        summary = analyzer.summary()

        assert "average_daily_ghi" in summary
        assert "annual_solar_energy_kwh_m2" in summary
        assert "best_month" in summary
        assert "worst_month" in summary
        assert "seasonal_ratio" in summary
        assert summary["seasonal_ratio"] > 1.0

    def test_higher_latitude_lower_ghi(self, sample_dataset, sample_dataset_london):
        """Higher latitude locations should have lower average GHI."""
        delhi = self._make_analyzer(sample_dataset, lat=28.6)
        london = self._make_analyzer(sample_dataset_london, lat=51.5)

        delhi_ghi = delhi.average_daily_irradiance()["GHI"]
        london_ghi = london.average_daily_irradiance()["GHI"]

        assert delhi_ghi > london_ghi
