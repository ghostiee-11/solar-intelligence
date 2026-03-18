"""Quick start example for Solar Intelligence Platform.

Demonstrates the core workflow:
1. Load solar data (synthetic)
2. Analyze irradiance patterns
3. Estimate energy production
4. Find optimal panel orientation
5. Run financial analysis
6. Generate AI insights
"""

import logging

from solar_intelligence.ai_engine import SolarAIEngine
from solar_intelligence.data_loader import generate_synthetic_solar_data
from solar_intelligence.energy_estimator import EnergyEstimator
from solar_intelligence.financial import FinancialAnalyzer
from solar_intelligence.orientation_simulator import OrientationSimulator
from solar_intelligence.solar_analysis import SolarAnalyzer

logger = logging.getLogger(__name__)


def main():
    """Run the complete Solar Intelligence analysis pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Configuration ---
    CITY = "New Delhi"
    LAT, LON = 28.6139, 77.2090

    logger.info("Solar Potential Intelligence Platform")
    logger.info("Location: %s (%.4f N, %.4f E)", CITY, LAT, LON)

    # --- 1. Load Data ---
    logger.info("1. Loading solar radiation data...")
    ds = generate_synthetic_solar_data(lat=LAT, lon=LON, start_year=2020, end_year=2023)
    logger.info("   Dataset: %d days, %s...", len(ds.time), list(ds.data_vars)[:5])

    # --- 2. Solar Analysis ---
    logger.info("2. Analyzing solar irradiance...")
    analyzer = SolarAnalyzer(dataset=ds, latitude=LAT, longitude=LON)
    summary = analyzer.summary()

    logger.info("   Average daily GHI: %.2f kWh/m2/day", summary["average_daily_ghi"])
    logger.info("   Annual solar energy: %.0f kWh/m2/year", summary["annual_solar_energy_kwh_m2"])
    logger.info("   Best month: %s (%.2f)", summary["best_month"], summary["best_month_ghi"])
    logger.info("   Worst month: %s (%.2f)", summary["worst_month"], summary["worst_month_ghi"])

    # --- 3. Energy Estimation ---
    logger.info("3. Estimating energy production...")
    estimator = EnergyEstimator(
        panel_efficiency=0.20,
        panel_area=1.7,
        num_panels=20,
        system_losses=0.14,
    )
    energy_summary = estimator.system_summary(ds)

    logger.info("   System capacity: %.1f kW", energy_summary["system"]["capacity_kw"])
    logger.info("   Annual energy: %,.0f kWh", energy_summary["production"]["annual_energy_kwh"])
    logger.info("   Capacity factor: %.1f%%", energy_summary["performance"]["capacity_factor_pct"])

    # --- 4. Orientation Simulation ---
    logger.info("4. Simulating panel orientations...")
    simulator = OrientationSimulator(
        latitude=LAT, longitude=LON,
        panel_efficiency=0.20,
        panel_area=estimator.total_area,
        system_losses=0.14,
        tilt_angles=[0, 15, 30, 45],
        azimuths={"North": 0, "East": 90, "South": 180, "West": 270},
    )

    ghi_year = ds["ALLSKY_SFC_SW_DWN"].sel(time=slice("2023-01-01", "2023-12-31")).values
    optimal = simulator.optimal_orientation(ghi_year, year=2023)

    logger.info("   Optimal: %s at %d tilt", optimal["best_direction"], optimal["best_tilt"])
    logger.info("   Gain vs horizontal: %.1f%%", optimal["energy_gain_vs_horizontal_pct"])
    logger.info("   Gain vs worst: %.1f%%", optimal["energy_gain_vs_worst_pct"])

    # --- 5. Financial Analysis ---
    logger.info("5. Financial analysis...")
    annual_energy = energy_summary["production"]["annual_energy_kwh"]
    financial = FinancialAnalyzer(
        system_cost=20000,
        electricity_rate=0.12,
        incentive_percent=0.30,
    )
    fin = financial.financial_summary(annual_energy)

    logger.info("   Net cost: $%,.0f", fin["investment"]["net_cost"])
    logger.info("   Payback: %s years", fin["returns"]["payback_years"])
    logger.info("   25-year NPV: $%,.0f", fin["returns"]["npv_25yr"])
    logger.info("   ROI: %.0f%%", fin["returns"]["roi_pct"])
    logger.info("   CO2 offset: %,.0f kg/year", fin["environmental"]["annual_co2_offset_kg"])
    logger.info("   Equivalent trees: %d", fin["environmental"]["equivalent_trees"])

    # --- 6. AI Insights ---
    logger.info("6. Generating AI insights...")
    ai = SolarAIEngine()
    report = ai.generate_report(summary, energy_summary, fin, optimal)
    logger.info("AI Report:\n%s", report)

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
