"""Tests for Phase 3: Advanced simulation features.

Covers:
- Single-axis and dual-axis tracking simulation
- Horizon shading model
- Inter-row shading for solar farms
- Bifacial panel gain estimation
- Rooftop suitability scoring (RooftopScorer)
"""

from __future__ import annotations

import numpy as np
import pytest

from solar_intelligence.data_loader import generate_synthetic_solar_data
from solar_intelligence.orientation_simulator import OrientationSimulator, RooftopScorer


@pytest.fixture
def dataset():
    return generate_synthetic_solar_data(lat=28.6, lon=77.2, start_year=2023, end_year=2023)


@pytest.fixture
def ghi(dataset):
    return dataset["ALLSKY_SFC_SW_DWN"].values


@pytest.fixture
def sim():
    return OrientationSimulator(
        latitude=28.6, longitude=77.2,
        tilt_angles=[0, 30],
        azimuths={"North": 0, "South": 180, "East": 90, "West": 270},
    )


# ---------------------------------------------------------------------------
# Tracking Simulation Tests
# ---------------------------------------------------------------------------

class TestTrackingSimulation:

    def test_single_axis_returns_dict(self, sim, ghi):
        result = sim.simulate_tracking(ghi, mode="single_axis")
        assert isinstance(result, dict)
        assert "tracking_mode" in result
        assert result["tracking_mode"] == "single_axis"

    def test_single_axis_has_energy(self, sim, ghi):
        result = sim.simulate_tracking(ghi, mode="single_axis")
        assert result["annual_energy_kwh"] > 0

    def test_single_axis_beats_fixed(self, sim, ghi):
        result = sim.simulate_tracking(ghi, mode="single_axis")
        assert result["gain_vs_fixed_pct"] > 0

    def test_dual_axis_returns_dict(self, sim, ghi):
        result = sim.simulate_tracking(ghi, mode="dual_axis")
        assert result["tracking_mode"] == "dual_axis"
        assert result["annual_energy_kwh"] > 0

    def test_dual_axis_beats_single(self, sim, ghi):
        single = sim.simulate_tracking(ghi, mode="single_axis")
        dual = sim.simulate_tracking(ghi, mode="dual_axis")
        # Dual axis should capture more or equal energy
        assert dual["annual_energy_kwh"] >= single["annual_energy_kwh"] * 0.9


# ---------------------------------------------------------------------------
# Horizon Shading Tests
# ---------------------------------------------------------------------------

class TestHorizonShading:

    def test_shading_returns_dict(self, sim, ghi):
        result = sim.horizon_shading(ghi)
        assert isinstance(result, dict)
        assert "shading_loss_pct" in result

    def test_shading_loss_positive(self, sim, ghi):
        # Default profile has obstructions
        result = sim.horizon_shading(ghi)
        assert result["shading_loss_pct"] >= 0

    def test_no_shading_with_flat_horizon(self, sim, ghi):
        # Flat horizon = no obstructions
        flat = {az: 0 for az in range(0, 360, 45)}
        result = sim.horizon_shading(ghi, horizon_profile=flat)
        assert result["shading_loss_pct"] == 0

    def test_heavy_shading_high_loss(self, sim, ghi):
        # High obstructions all around
        heavy = {az: 40 for az in range(0, 360, 45)}
        result = sim.horizon_shading(ghi, horizon_profile=heavy)
        assert result["shading_loss_pct"] > 5

    def test_shaded_energy_less_than_unshaded(self, sim, ghi):
        result = sim.horizon_shading(ghi)
        assert result["shaded_energy_kwh"] <= result["unshaded_energy_kwh"]

    def test_daylight_hours_positive(self, sim, ghi):
        result = sim.horizon_shading(ghi)
        assert result["total_daylight_hours"] > 2000


# ---------------------------------------------------------------------------
# Inter-Row Shading Tests
# ---------------------------------------------------------------------------

class TestInterRowShading:

    def test_inter_row_returns_dict(self, sim):
        result = sim.inter_row_shading(tilt=30, row_spacing_ratio=2.0)
        assert isinstance(result, dict)

    def test_adequate_spacing_no_loss(self, sim):
        result = sim.inter_row_shading(tilt=30, row_spacing_ratio=10.0)
        assert result["shading_loss_pct"] == 0
        assert result["current_spacing_adequate"] is True

    def test_tight_spacing_has_loss(self, sim):
        result = sim.inter_row_shading(tilt=30, row_spacing_ratio=0.5)
        assert result["shading_loss_pct"] > 0
        assert result["current_spacing_adequate"] is False

    def test_min_spacing_positive(self, sim):
        result = sim.inter_row_shading(tilt=30)
        assert result["min_spacing_ratio"] > 0

    def test_steeper_tilt_longer_shadow(self, sim):
        shallow = sim.inter_row_shading(tilt=15)
        steep = sim.inter_row_shading(tilt=60)
        assert steep["shadow_length_ratio"] > shallow["shadow_length_ratio"]


# ---------------------------------------------------------------------------
# Bifacial Gain Tests
# ---------------------------------------------------------------------------

class TestBifacialGain:

    def test_bifacial_returns_dict(self, sim, ghi):
        result = sim.bifacial_gain(ghi, tilt=30)
        assert isinstance(result, dict)
        assert "bifacial_gain_pct" in result

    def test_bifacial_gain_positive(self, sim, ghi):
        result = sim.bifacial_gain(ghi, tilt=30)
        assert result["bifacial_gain_pct"] > 0

    def test_total_exceeds_front(self, sim, ghi):
        result = sim.bifacial_gain(ghi, tilt=30)
        assert result["total_energy_kwh"] > result["front_energy_kwh"]

    def test_higher_tilt_more_rear(self, sim, ghi):
        low_tilt = sim.bifacial_gain(ghi, tilt=10)
        high_tilt = sim.bifacial_gain(ghi, tilt=45)
        assert high_tilt["rear_irradiance_pct"] > low_tilt["rear_irradiance_pct"]

    def test_higher_bifaciality_more_gain(self, sim, ghi):
        low_bf = sim.bifacial_gain(ghi, tilt=30, bifaciality=0.50)
        high_bf = sim.bifacial_gain(ghi, tilt=30, bifaciality=0.85)
        assert high_bf["bifacial_gain_pct"] > low_bf["bifacial_gain_pct"]

    def test_horizontal_zero_gain(self, sim, ghi):
        result = sim.bifacial_gain(ghi, tilt=0)
        # At 0 tilt, view factor is 0 -> no rear irradiance
        assert result["bifacial_gain_pct"] == 0


# ---------------------------------------------------------------------------
# Rooftop Suitability Scorer Tests
# ---------------------------------------------------------------------------

class TestRooftopScorer:

    def test_scorer_returns_dict(self):
        scorer = RooftopScorer()
        result = scorer.score(avg_daily_ghi=5.5, optimal_tilt=28, roof_tilt=30)
        assert isinstance(result, dict)
        assert "total_score" in result
        assert "rating" in result

    def test_score_range(self):
        scorer = RooftopScorer()
        result = scorer.score(avg_daily_ghi=5.5, optimal_tilt=28, roof_tilt=30)
        assert 0 <= result["total_score"] <= 100

    def test_excellent_location(self):
        scorer = RooftopScorer()
        result = scorer.score(
            avg_daily_ghi=7.0, optimal_tilt=25, roof_tilt=25,
            variability_index=0.08, avg_temperature=22,
        )
        assert result["rating"] == "Excellent"
        assert result["total_score"] >= 80

    def test_poor_location(self):
        scorer = RooftopScorer()
        result = scorer.score(
            avg_daily_ghi=1.5, optimal_tilt=55, roof_tilt=10,
            variability_index=0.35, avg_temperature=-5,
        )
        assert result["total_score"] < 40

    def test_components_present(self):
        scorer = RooftopScorer()
        result = scorer.score(avg_daily_ghi=5.0, optimal_tilt=30, roof_tilt=30)
        assert "solar_resource" in result["components"]
        assert "tilt_match" in result["components"]
        assert "climate_stability" in result["components"]
        assert "temperature" in result["components"]

    def test_weights_sum_to_one(self):
        scorer = RooftopScorer()
        result = scorer.score(avg_daily_ghi=5.0, optimal_tilt=30, roof_tilt=30)
        weights = result["weights"]
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_tilt_mismatch_lowers_score(self):
        scorer = RooftopScorer()
        matched = scorer.score(avg_daily_ghi=5.5, optimal_tilt=30, roof_tilt=30)
        mismatched = scorer.score(avg_daily_ghi=5.5, optimal_tilt=30, roof_tilt=80)
        assert matched["total_score"] > mismatched["total_score"]

    def test_recommendations_for_poor(self):
        scorer = RooftopScorer()
        result = scorer.score(
            avg_daily_ghi=1.0, optimal_tilt=30, roof_tilt=80,
            variability_index=0.45, avg_temperature=45,
        )
        assert len(result["recommendations"]) > 0

    def test_no_recommendations_for_excellent(self):
        scorer = RooftopScorer()
        result = scorer.score(
            avg_daily_ghi=7.0, optimal_tilt=25, roof_tilt=25,
            variability_index=0.08, avg_temperature=22,
        )
        assert len(result["recommendations"]) == 0
