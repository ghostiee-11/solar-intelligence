"""Solar irradiance analysis module.

Computes solar radiation statistics, seasonal patterns, anomalies,
and clearsky indices using xarray operations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import param
import xarray as xr

from solar_intelligence.config import DEFAULT_END_YEAR, DEFAULT_START_YEAR

logger = logging.getLogger(__name__)

# Season mapping for groupby
SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF",
              3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA",
              9: "SON", 10: "SON", 11: "SON"}

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


class SolarAnalyzer(param.Parameterized):
    """Analyze solar irradiance data from xarray Datasets.

    Provides statistical analysis of solar radiation including monthly/seasonal
    patterns, anomaly detection, clearsky indices, and peak sun hours.

    All xarray operations use .mean(), .groupby(), .resample(), .rolling()
    as required for scientific data processing.

    Parameters
    ----------
    dataset : xr.Dataset
        Solar radiation dataset with time dimension.
    latitude : float
        Location latitude.
    longitude : float
        Location longitude.
    ghi_var : str
        Variable name for Global Horizontal Irradiance.
    dni_var : str
        Variable name for Direct Normal Irradiance.
    dhi_var : str
        Variable name for Diffuse Horizontal Irradiance.
    clearsky_var : str
        Variable name for Clear Sky GHI.
    temp_var : str
        Variable name for temperature.
    """

    dataset = param.Parameter(doc="xarray Dataset with solar radiation data")
    latitude = param.Number(default=0.0, bounds=(-90, 90))
    longitude = param.Number(default=0.0, bounds=(-180, 180))
    ghi_var = param.String(default="ALLSKY_SFC_SW_DWN")
    dni_var = param.String(default="ALLSKY_SFC_SW_DNI")
    dhi_var = param.String(default="ALLSKY_SFC_SW_DIFF")
    clearsky_var = param.String(default="CLRSKY_SFC_SW_DWN")
    temp_var = param.String(default="T2M")

    def _get_var(self, var_name: str) -> xr.DataArray:
        """Safely retrieve a variable from the dataset."""
        if self.dataset is None:
            raise ValueError("No dataset loaded. Set the 'dataset' parameter first.")
        if var_name not in self.dataset:
            raise KeyError(f"Variable '{var_name}' not found in dataset. "
                           f"Available: {list(self.dataset.data_vars)}")
        return self.dataset[var_name]

    # -------------------------------------------------------------------
    # Core Statistics
    # -------------------------------------------------------------------

    def average_daily_irradiance(self) -> dict[str, float]:
        """Compute average daily irradiance for all components.

        Returns
        -------
        dict[str, float]
            Average daily GHI, DNI, DHI in kWh/m²/day.
        """
        result = {}
        for label, var in [("GHI", self.ghi_var), ("DNI", self.dni_var),
                           ("DHI", self.dhi_var)]:
            try:
                result[label] = float(self._get_var(var).mean())
            except KeyError:
                logger.warning("Variable %s not available for %s", var, label)
        return result

    def monthly_irradiance(self) -> pd.DataFrame:
        """Compute monthly-averaged irradiance for all components.

        Uses xarray .groupby('time.month').mean() for temporal aggregation.

        Returns
        -------
        pd.DataFrame
            DataFrame with month (1-12) as index and GHI/DNI/DHI columns.
        """
        records = {}
        for label, var in [("GHI", self.ghi_var), ("DNI", self.dni_var),
                           ("DHI", self.dhi_var)]:
            try:
                monthly = self._get_var(var).groupby("time.month").mean()
                records[label] = monthly.to_pandas()
            except KeyError:
                logger.debug("Variable %s not available, skipping", var)

        df = pd.DataFrame(records)
        df.index.name = "month"

        # Add month names
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        df["month_name"] = [month_names[i - 1] for i in df.index]

        return df

    def seasonal_patterns(self) -> pd.DataFrame:
        """Compute seasonal irradiance statistics.

        Uses xarray .groupby('time.season') for seasonal aggregation.

        Returns
        -------
        pd.DataFrame
            Season × (GHI_mean, GHI_std, DNI_mean, DHI_mean, temperature).
        """
        ghi = self._get_var(self.ghi_var)

        # Use month-based seasons for consistent ordering
        months = self.dataset["time"].dt.month
        seasons = xr.DataArray(
            [SEASON_MAP[int(m)] for m in months.values],
            dims="time",
            coords={"time": self.dataset.time},
        )

        records = []
        for season in SEASON_ORDER:
            mask = seasons == season
            ghi_season = ghi.where(mask, drop=True)
            row = {
                "season": season,
                "GHI_mean": float(ghi_season.mean()),
                "GHI_std": float(ghi_season.std()),
            }
            try:
                dni = self._get_var(self.dni_var).where(mask, drop=True)
                row["DNI_mean"] = float(dni.mean())
            except KeyError:
                logger.debug("Variable %s not available, skipping", self.dni_var)
            try:
                dhi = self._get_var(self.dhi_var).where(mask, drop=True)
                row["DHI_mean"] = float(dhi.mean())
            except KeyError:
                logger.debug("Variable %s not available, skipping", self.dhi_var)
            try:
                temp = self._get_var(self.temp_var).where(mask, drop=True)
                row["temperature"] = float(temp.mean())
            except KeyError:
                logger.debug("Variable %s not available, skipping", self.temp_var)
            records.append(row)

        return pd.DataFrame(records).set_index("season")

    def annual_solar_energy(self) -> float:
        """Compute total annual solar energy potential.

        Returns
        -------
        float
            Annual solar energy in kWh/m²/year (sum of daily GHI).
        """
        ghi = self._get_var(self.ghi_var)
        # Average daily GHI × 365.25 days
        return float(ghi.mean()) * 365.25

    # -------------------------------------------------------------------
    # Advanced Analysis
    # -------------------------------------------------------------------

    def clearsky_index(self) -> pd.DataFrame:
        """Compute clearsky index (actual GHI / clear-sky GHI).

        The clearsky index (Kt) measures atmospheric transparency.
        Values close to 1.0 indicate clear skies; lower values indicate clouds.

        Returns
        -------
        pd.DataFrame
            Monthly clearsky index values.
        """
        ghi = self._get_var(self.ghi_var)
        clearsky = self._get_var(self.clearsky_var)

        # Avoid division by zero
        ratio = ghi / clearsky.where(clearsky > 0.1)

        monthly_kt = ratio.groupby("time.month").mean()

        df = monthly_kt.to_dataframe(name="clearsky_index").reset_index()
        return df

    def peak_sun_hours(self) -> pd.DataFrame:
        """Compute Peak Sun Hours (PSH) by month.

        PSH = daily GHI in kWh/m² (since 1 PSH = 1 kWh/m² at 1000 W/m²).
        This directly equals the daily GHI value from NASA POWER.

        Returns
        -------
        pd.DataFrame
            Monthly average Peak Sun Hours.
        """
        ghi = self._get_var(self.ghi_var)
        monthly_psh = ghi.groupby("time.month").mean()

        df = monthly_psh.to_dataframe(name="peak_sun_hours").reset_index()

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        df["month_name"] = df["month"].apply(lambda m: month_names[m - 1])

        return df

    def anomaly_detection(self) -> pd.DataFrame:
        """Detect irradiance anomalies by comparing to climatological mean.

        Uses xarray .groupby() to compute monthly climatology, then
        subtracts it to find deviations.

        Returns
        -------
        pd.DataFrame
            Time series with GHI, climatology, and anomaly columns.
        """
        ghi = self._get_var(self.ghi_var)

        # Monthly climatology
        climatology = ghi.groupby("time.month").mean()

        # Anomaly = actual - climatology
        anomaly = ghi.groupby("time.month") - climatology

        # Resample to monthly for cleaner output
        monthly_ghi = ghi.resample(time="ME").mean()
        monthly_anomaly = anomaly.resample(time="ME").mean()

        df = pd.DataFrame({
            "time": monthly_ghi.time.values,
            "GHI": monthly_ghi.values,
            "anomaly": monthly_anomaly.values,
        })

        return df

    def rolling_average(self, window: int = 30) -> pd.DataFrame:
        """Compute rolling average of GHI.

        Uses xarray .rolling() for smoothed time series.

        Parameters
        ----------
        window : int
            Rolling window size in days.

        Returns
        -------
        pd.DataFrame
            Time series with raw GHI and rolling average.
        """
        ghi = self._get_var(self.ghi_var)
        rolled = ghi.rolling(time=window, center=True).mean()

        df = pd.DataFrame({
            "time": ghi.time.values,
            "GHI": ghi.values,
            f"GHI_rolling_{window}d": rolled.values,
        })

        return df

    def variability_index(self) -> pd.DataFrame:
        """Compute solar variability index by month.

        Higher variability = less predictable solar resource.
        Variability = coefficient of variation (std/mean).

        Returns
        -------
        pd.DataFrame
            Monthly variability index.
        """
        ghi = self._get_var(self.ghi_var)

        monthly_mean = ghi.groupby("time.month").mean()
        monthly_std = ghi.groupby("time.month").std()
        variability = monthly_std / monthly_mean

        df = variability.to_dataframe(name="variability_index").reset_index()
        return df

    def hourly_profile_estimate(self) -> pd.DataFrame:
        """Estimate hourly solar profile from daily data.

        Uses a cosine model to distribute daily GHI across daylight hours.
        This is an approximation for daily-resolution data.

        Returns
        -------
        pd.DataFrame
            Estimated hourly GHI profile (hour 0-23) averaged across dataset.
        """
        avg_daily_ghi = float(self._get_var(self.ghi_var).mean())

        # Approximate day length from latitude
        lat_rad = np.radians(self.latitude)
        # Use average declination (equinox)
        hours = np.arange(24)

        # Cosine-shaped solar profile centered at solar noon (12:00)
        sunrise = 6 - abs(self.latitude) / 30  # rough approximation
        sunset = 18 + abs(self.latitude) / 30
        sunrise = max(4, min(8, sunrise))
        sunset = max(16, min(20, sunset))

        day_length = sunset - sunrise
        solar_noon = (sunrise + sunset) / 2

        profile = np.zeros(24)
        for h in range(24):
            if sunrise <= h <= sunset:
                x = np.pi * (h - sunrise) / day_length
                profile[h] = np.sin(x)

        # Normalize to match daily total
        if profile.sum() > 0:
            profile = profile * avg_daily_ghi / profile.sum()

        return pd.DataFrame({
            "hour": hours,
            "estimated_ghi_kwh": profile,
        })

    def summary(self) -> dict[str, Any]:
        """Generate a comprehensive summary of solar analysis.

        Returns
        -------
        dict
            Summary statistics including averages, seasonal patterns, and quality metrics.
        """
        avg = self.average_daily_irradiance()
        annual = self.annual_solar_energy()
        monthly = self.monthly_irradiance()

        best_month_idx = monthly["GHI"].idxmax()
        worst_month_idx = monthly["GHI"].idxmin()

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        summary = {
            "location": {"latitude": self.latitude, "longitude": self.longitude},
            "average_daily_ghi": avg.get("GHI", 0),
            "average_daily_dni": avg.get("DNI", 0),
            "average_daily_dhi": avg.get("DHI", 0),
            "annual_solar_energy_kwh_m2": annual,
            "best_month": month_names[best_month_idx - 1],
            "best_month_ghi": float(monthly.loc[best_month_idx, "GHI"]),
            "worst_month": month_names[worst_month_idx - 1],
            "worst_month_ghi": float(monthly.loc[worst_month_idx, "GHI"]),
            "seasonal_ratio": float(monthly["GHI"].max() / monthly["GHI"].min()),
            "data_years": len(set(self.dataset.time.dt.year.values)),
        }

        return summary


# ---------------------------------------------------------------------------
# Multi-Location Comparison
# ---------------------------------------------------------------------------

class MultiLocationComparator(param.Parameterized):
    """Compare solar potential across multiple locations.

    Produces side-by-side analysis of GHI, energy potential, seasonal
    patterns, and optimal orientations for 2-5 cities.

    Parameters
    ----------
    locations : dict
        Mapping of location name -> (lat, lon).
    datasets : dict
        Mapping of location name -> xr.Dataset (populated by analyze()).
    """

    locations = param.Dict(default={}, doc="Location name -> (lat, lon)")

    def __init__(self, locations: dict[str, tuple[float, float]], **params):
        super().__init__(locations=locations, **params)
        self._analyzers: dict[str, SolarAnalyzer] = {}
        self._datasets: dict[str, xr.Dataset] = {}

    def load_data(
        self,
        datasets: dict[str, xr.Dataset] | None = None,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
    ) -> None:
        """Load or generate data for all locations.

        Parameters
        ----------
        datasets : dict, optional
            Pre-loaded datasets. If None, generates synthetic data.
        start_year, end_year : int
            Date range for synthetic data generation.
        """
        from solar_intelligence.data_loader import generate_synthetic_solar_data

        if datasets is not None:
            self._datasets = datasets
        else:
            for name, (lat, lon) in self.locations.items():
                self._datasets[name] = generate_synthetic_solar_data(
                    lat, lon, start_year, end_year,
                )

        for name, (lat, lon) in self.locations.items():
            self._analyzers[name] = SolarAnalyzer(
                dataset=self._datasets[name], latitude=lat, longitude=lon,
            )

    def compare_ghi(self) -> pd.DataFrame:
        """Compare average daily GHI across locations.

        Returns
        -------
        pd.DataFrame
            Columns: location, GHI, DNI, DHI, annual_kwh_m2.
        """
        rows = []
        for name, analyzer in self._analyzers.items():
            avg = analyzer.average_daily_irradiance()
            annual = analyzer.annual_solar_energy()
            rows.append({
                "location": name,
                "GHI": avg.get("GHI", 0),
                "DNI": avg.get("DNI", 0),
                "DHI": avg.get("DHI", 0),
                "annual_kwh_m2": annual,
            })
        return pd.DataFrame(rows)

    def compare_monthly(self) -> pd.DataFrame:
        """Compare monthly GHI patterns across locations.

        Returns
        -------
        pd.DataFrame
            Columns: month, month_name, location, GHI.
        """
        frames = []
        for name, analyzer in self._analyzers.items():
            monthly = analyzer.monthly_irradiance().reset_index()
            monthly["location"] = name
            frames.append(monthly[["month", "month_name", "location", "GHI"]])
        return pd.concat(frames, ignore_index=True)

    def compare_seasonal(self) -> pd.DataFrame:
        """Compare seasonal patterns across locations.

        Returns
        -------
        pd.DataFrame
            Columns: season, location, GHI_mean, GHI_std.
        """
        frames = []
        for name, analyzer in self._analyzers.items():
            seasonal = analyzer.seasonal_patterns().reset_index()
            seasonal["location"] = name
            frames.append(seasonal)
        return pd.concat(frames, ignore_index=True)

    def ranking(self) -> pd.DataFrame:
        """Rank locations by solar potential.

        Returns
        -------
        pd.DataFrame
            Sorted by annual GHI descending, with rank column.
        """
        df = self.compare_ghi().sort_values("annual_kwh_m2", ascending=False)
        df["rank"] = range(1, len(df) + 1)
        return df.reset_index(drop=True)

    def summary(self) -> dict[str, dict]:
        """Generate per-location summaries.

        Returns
        -------
        dict[str, dict]
            Mapping of location name -> summary dict.
        """
        return {
            name: analyzer.summary()
            for name, analyzer in self._analyzers.items()
        }


# ---------------------------------------------------------------------------
# Dual-Source Analysis (NASA POWER + ERA5)
# ---------------------------------------------------------------------------

class DualSourceAnalyzer(param.Parameterized):
    """Compare solar analysis results from multiple data sources.

    Runs SolarAnalyzer on each source dataset independently,
    then computes cross-source comparison metrics.

    Parameters
    ----------
    datasets : dict
        Source name -> xr.Dataset mapping (e.g., {"nasa_power": ds1, "era5": ds2}).
    latitude : float
        Location latitude.
    longitude : float
        Location longitude.
    """

    datasets = param.Dict(default={}, doc="Source name -> xr.Dataset mapping")
    latitude = param.Number(default=0.0, bounds=(-90, 90))
    longitude = param.Number(default=0.0, bounds=(-180, 180))

    _analyzers = param.Dict(default={}, precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self._analyzers = {}
        for name, ds in self.datasets.items():
            self._analyzers[name] = SolarAnalyzer(
                dataset=ds, latitude=self.latitude, longitude=self.longitude,
            )

    def source_summaries(self) -> dict[str, dict]:
        """Get analysis summary from each source.

        Returns
        -------
        dict[str, dict]
            Source name -> summary dict.
        """
        return {
            name: analyzer.summary()
            for name, analyzer in self._analyzers.items()
        }

    def compare_daily_ghi(self) -> pd.DataFrame:
        """Align daily GHI from all sources into a single DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns are source names, index is time.
        """
        from solar_intelligence.data_loader import DualSourceLoader
        return DualSourceLoader.align_datasets(self.datasets, "ALLSKY_SFC_SW_DWN")

    def compare_monthly(self) -> pd.DataFrame:
        """Compare monthly GHI averages across sources.

        Returns
        -------
        pd.DataFrame
            Columns: month, month_name, plus one GHI column per source.
        """
        monthly_data = {}
        for name, analyzer in self._analyzers.items():
            monthly = analyzer.monthly_irradiance()
            monthly_data[name] = monthly["GHI"].values

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        df = pd.DataFrame({
            "month": range(1, 13),
            "month_name": month_names,
        })
        for name, values in monthly_data.items():
            df[name] = values

        return df

    def cross_validation(self) -> dict[str, Any]:
        """Compute cross-validation statistics between sources.

        Returns
        -------
        dict
            Per-source stats, correlation, RMSE, bias between sources.
        """
        from solar_intelligence.data_loader import DualSourceLoader
        return DualSourceLoader.comparison_stats(self.datasets, "ALLSKY_SFC_SW_DWN")

    def agreement_report(self) -> str:
        """Generate a human-readable agreement report between sources.

        Returns
        -------
        str
            Multi-line report describing how well sources agree.
        """
        stats = self.cross_validation()
        summaries = self.source_summaries()

        lines = ["## Dual-Source Cross-Validation Report\n"]

        # Per-source summary
        for name, summary in summaries.items():
            ghi = summary.get("average_daily_ghi", 0)
            annual = summary.get("annual_solar_energy_kwh_m2", 0)
            lines.append(
                f"**{name}**: GHI = {ghi:.2f} kWh/m2/day, "
                f"Annual = {annual:.0f} kWh/m2/year"
            )

        # Cross-comparison
        comp = stats.get("comparison", {})
        if comp:
            lines.append(f"\n### Agreement Metrics")
            lines.append(f"- Correlation: {comp.get('correlation', 0):.4f}")
            lines.append(f"- RMSE: {comp.get('rmse', 0):.3f} kWh/m2/day")
            lines.append(f"- MAE: {comp.get('mae', 0):.3f} kWh/m2/day")
            lines.append(f"- Bias ({comp['source_a']} - {comp['source_b']}): "
                         f"{comp.get('bias', 0):.3f} kWh/m2/day "
                         f"({comp.get('bias_pct', 0):.1f}%)")
            lines.append(f"- Common days compared: {comp.get('common_days', 0)}")

            corr = comp.get("correlation", 0)
            if corr > 0.95:
                quality = "Excellent agreement"
            elif corr > 0.85:
                quality = "Good agreement"
            elif corr > 0.70:
                quality = "Moderate agreement"
            else:
                quality = "Poor agreement -- investigate data quality"
            lines.append(f"\n**Overall: {quality}** (r = {corr:.4f})")

        return "\n".join(lines)
