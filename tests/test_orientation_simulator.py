"""Tests for orientation_simulator module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_intelligence.data_loader import generate_synthetic_solar_data
from solar_intelligence.orientation_simulator import OrientationSimulator


@pytest.fixture
def simulator_delhi():
    """Orientation simulator for New Delhi."""
    return OrientationSimulator(
        latitude=28.6139, longitude=77.2090,
        tilt_angles=[0, 15, 30, 45],
        azimuths={"North": 0, "East": 90, "South": 180, "West": 270},
    )


@pytest.fixture
def simulator_sydney():
    """Orientation simulator for Sydney (Southern Hemisphere)."""
    return OrientationSimulator(
        latitude=-33.8688, longitude=151.2093,
        tilt_angles=[0, 15, 30, 45],
        azimuths={"North": 0, "East": 90, "South": 180, "West": 270},
    )


@pytest.fixture
def ghi_daily_delhi():
    """Daily GHI for one year — New Delhi."""
    ds = generate_synthetic_solar_data(lat=28.6139, lon=77.2090, start_year=2023, end_year=2023)
    return ds["ALLSKY_SFC_SW_DWN"].values


@pytest.fixture
def ghi_daily_sydney():
    """Daily GHI for one year — Sydney."""
    ds = generate_synthetic_solar_data(lat=-33.8688, lon=151.2093, start_year=2023, end_year=2023)
    return ds["ALLSKY_SFC_SW_DWN"].values


class TestSolarPosition:
    def test_solar_position_returns_dataframe(self, simulator_delhi):
        solpos = simulator_delhi.solar_position_timeseries(year=2023)
        assert isinstance(solpos, pd.DataFrame)
        assert "apparent_zenith" in solpos.columns
        assert "azimuth" in solpos.columns
        assert len(solpos) == 8760  # 365 × 24


class TestOrientationSimulation:
    def test_simulate_returns_dataframe(self, simulator_delhi, ghi_daily_delhi):
        sim = simulator_delhi.simulate_all_orientations(ghi_daily_delhi, year=2023)
        assert isinstance(sim, pd.DataFrame)
        assert "direction" in sim.columns
        assert "tilt_deg" in sim.columns
        assert "monthly_energy_kwh" in sim.columns
        assert "annual_energy_kwh" in sim.columns

    def test_all_orientations_present(self, simulator_delhi, ghi_daily_delhi):
        sim = simulator_delhi.simulate_all_orientations(ghi_daily_delhi)
        directions = set(sim["direction"])
        assert directions == {"North", "East", "South", "West"}

    def test_all_tilts_present(self, simulator_delhi, ghi_daily_delhi):
        sim = simulator_delhi.simulate_all_orientations(ghi_daily_delhi)
        tilts = set(sim["tilt_deg"])
        assert tilts == {0, 15, 30, 45}

    def test_energy_positive(self, simulator_delhi, ghi_daily_delhi):
        sim = simulator_delhi.simulate_all_orientations(ghi_daily_delhi)
        assert all(sim["annual_energy_kwh"] > 0)

    def test_south_beats_north_in_northern_hemisphere(self, simulator_delhi, ghi_daily_delhi):
        """In Northern Hemisphere, south-facing should outperform north-facing."""
        sim = simulator_delhi.simulate_all_orientations(ghi_daily_delhi)
        annual = sim.drop_duplicates(subset=["direction", "tilt_deg"])

        south_30 = annual[
            (annual["direction"] == "South") & (annual["tilt_deg"] == 30)
        ]["annual_energy_kwh"].values[0]

        north_30 = annual[
            (annual["direction"] == "North") & (annual["tilt_deg"] == 30)
        ]["annual_energy_kwh"].values[0]

        assert south_30 > north_30, (
            f"South ({south_30}) should beat North ({north_30}) in NH"
        )

    def test_north_beats_south_in_southern_hemisphere(self, simulator_sydney, ghi_daily_sydney):
        """In Southern Hemisphere, north-facing should outperform south-facing."""
        sim = simulator_sydney.simulate_all_orientations(ghi_daily_sydney)
        annual = sim.drop_duplicates(subset=["direction", "tilt_deg"])

        north_30 = annual[
            (annual["direction"] == "North") & (annual["tilt_deg"] == 30)
        ]["annual_energy_kwh"].values[0]

        south_30 = annual[
            (annual["direction"] == "South") & (annual["tilt_deg"] == 30)
        ]["annual_energy_kwh"].values[0]

        assert north_30 > south_30, (
            f"North ({north_30}) should beat South ({south_30}) in SH"
        )


class TestOptimalOrientation:
    def test_optimal_returns_dict(self, simulator_delhi, ghi_daily_delhi):
        optimal = simulator_delhi.optimal_orientation(ghi_daily_delhi)
        assert "best_direction" in optimal
        assert "best_tilt" in optimal
        assert "annual_energy_kwh" in optimal
        assert "energy_gain_vs_horizontal_pct" in optimal

    def test_optimal_direction_is_south_for_nh(self, simulator_delhi, ghi_daily_delhi):
        """For Northern Hemisphere, optimal direction should be South."""
        optimal = simulator_delhi.optimal_orientation(ghi_daily_delhi)
        assert optimal["best_direction"] == "South"

    def test_optimal_tilt_near_latitude(self, simulator_delhi, ghi_daily_delhi):
        """Optimal tilt should approximate latitude for annual optimization."""
        optimal = simulator_delhi.optimal_orientation(ghi_daily_delhi)
        lat = simulator_delhi.latitude
        # Should be within ±20° of latitude
        assert abs(optimal["best_tilt"] - lat) < 20, (
            f"Optimal tilt {optimal['best_tilt']}° far from latitude {lat}°"
        )

    def test_gain_vs_horizontal_positive(self, simulator_delhi, ghi_daily_delhi):
        """Tilted panels should outperform horizontal."""
        optimal = simulator_delhi.optimal_orientation(ghi_daily_delhi)
        assert optimal["energy_gain_vs_horizontal_pct"] > 0


class TestDailyProfile:
    def test_daily_profile_shape(self, simulator_delhi, ghi_daily_delhi):
        profile = simulator_delhi.daily_profile_by_orientation(
            ghi_daily_delhi, date="2023-06-21",
        )
        assert isinstance(profile, pd.DataFrame)
        assert "hour" in profile.columns
        assert "direction" in profile.columns
        assert "energy_kwh" in profile.columns

    def test_nighttime_low_energy(self, simulator_delhi, ghi_daily_delhi):
        """Energy in deep night hours (UTC 20-22, local 1:30-3:30 AM IST) should be ~0."""
        profile = simulator_delhi.daily_profile_by_orientation(
            ghi_daily_delhi, date="2023-06-21",
        )
        # UTC hour 21 = IST 2:30 AM — definitely nighttime
        night = profile[profile["hour"] == 21]
        assert all(night["energy_kwh"] < 0.01)


class TestTiltSensitivity:
    def test_tilt_sensitivity_shape(self, simulator_delhi, ghi_daily_delhi):
        sensitivity = simulator_delhi.tilt_sensitivity_analysis(
            ghi_daily_delhi, tilt_range=[0, 15, 30, 45, 60, 90],
        )
        assert len(sensitivity) == 6
        assert "tilt_deg" in sensitivity.columns
        assert "annual_energy_kwh" in sensitivity.columns

    def test_extreme_tilt_less_energy(self, simulator_delhi, ghi_daily_delhi):
        """90° tilt (vertical) should produce less than 30° tilt."""
        sensitivity = simulator_delhi.tilt_sensitivity_analysis(
            ghi_daily_delhi, tilt_range=[30, 90],
        )
        e_30 = sensitivity[sensitivity["tilt_deg"] == 30]["annual_energy_kwh"].values[0]
        e_90 = sensitivity[sensitivity["tilt_deg"] == 90]["annual_energy_kwh"].values[0]
        assert e_30 > e_90


class TestSeasonalComparison:
    def test_seasonal_comparison_shape(self, simulator_delhi, ghi_daily_delhi):
        seasonal = simulator_delhi.seasonal_comparison(ghi_daily_delhi)
        assert isinstance(seasonal, pd.DataFrame)
        assert "season" in seasonal.columns
        assert "direction" in seasonal.columns
        assert "seasonal_energy_kwh" in seasonal.columns
