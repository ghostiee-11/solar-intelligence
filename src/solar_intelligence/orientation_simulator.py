"""Solar panel orientation and tilt simulation engine.

Simulates energy generation for different panel orientations and tilt angles
using pvlib for physics-accurate solar position and irradiance transposition.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import param
import pvlib
from pvlib.irradiance import get_total_irradiance
from pvlib.location import Location

from solar_intelligence.config import DEFAULT_ALBEDO, DEFAULT_TILT_ANGLES, ORIENTATIONS

logger = logging.getLogger(__name__)


class OrientationSimulator(param.Parameterized):
    """Simulate solar energy generation across panel orientations and tilts.

    Uses pvlib for physics-accurate:
    - Solar position calculation (zenith, azimuth per hour)
    - Irradiance transposition (GHI → plane-of-array irradiance)
    - GHI → DNI/DHI decomposition (Erbs model)

    Parameters
    ----------
    latitude : float
        Location latitude (-90 to 90).
    longitude : float
        Location longitude (-180 to 180).
    altitude : float
        Location altitude in meters.
    tilt_angles : list[int]
        List of tilt angles to simulate (degrees from horizontal).
    azimuths : dict[str, int]
        Mapping of direction names to azimuth angles.
    surface_albedo : float
        Ground surface reflectance (0-1).
    panel_efficiency : float
        Panel conversion efficiency.
    panel_area : float
        Total panel area in m².
    system_losses : float
        Combined system losses fraction.
    """

    latitude = param.Number(default=0.0, bounds=(-90, 90))
    longitude = param.Number(default=0.0, bounds=(-180, 180))
    altitude = param.Number(default=0, bounds=(0, 9000))
    tilt_angles = param.List(default=DEFAULT_TILT_ANGLES, item_type=(int, float))
    azimuths = param.Dict(default=ORIENTATIONS)
    surface_albedo = param.Number(default=DEFAULT_ALBEDO, bounds=(0, 1))
    panel_efficiency = param.Number(default=0.20, bounds=(0.05, 0.40))
    panel_area = param.Number(default=17.0, bounds=(0.1, 10000))  # total m²
    system_losses = param.Number(default=0.14, bounds=(0, 0.5))

    def _get_location(self) -> Location:
        """Create pvlib Location object."""
        return Location(
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
        )

    @staticmethod
    def smart_tilt_range(latitude: float) -> list[int]:
        """Generate tilt angles centered around optimal for given latitude."""
        optimal = int(abs(latitude))
        tilt_min = max(0, optimal - 20)
        tilt_max = min(90, optimal + 25)
        tilts = sorted(set([0] + list(range(tilt_min, tilt_max + 1, 5)) + [90]))
        return tilts

    # -------------------------------------------------------------------
    # Solar Position
    # -------------------------------------------------------------------

    def solar_position_timeseries(
        self,
        year: int = 2023,
        freq: str = "h",
    ) -> pd.DataFrame:
        """Compute solar position for every hour of the year.

        Uses pvlib.solarposition for accurate zenith/azimuth calculation.

        Parameters
        ----------
        year : int
            Year to simulate.
        freq : str
            Time frequency ('h' for hourly).

        Returns
        -------
        pd.DataFrame
            Columns: apparent_zenith, zenith, apparent_elevation, elevation,
                     azimuth, equation_of_time
        """
        loc = self._get_location()
        times = pd.date_range(
            f"{year}-01-01", f"{year}-12-31 23:00", freq=freq, tz="UTC",
        )
        solpos = loc.get_solarposition(times)
        return solpos

    # -------------------------------------------------------------------
    # Irradiance Decomposition & Transposition
    # -------------------------------------------------------------------

    def _decompose_ghi(
        self,
        ghi_daily: np.ndarray,
        times: pd.DatetimeIndex,
        solpos: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompose daily GHI into hourly DNI and DHI using Erbs model.

        Parameters
        ----------
        ghi_daily : array
            Daily GHI values in kWh/m²/day.
        times : DatetimeIndex
            Hourly timestamps.
        solpos : DataFrame
            Solar position data.

        Returns
        -------
        tuple[array, array]
            Hourly DNI and DHI in W/m².
        """
        # Create hourly GHI profile from daily values using cosine model
        zenith = solpos["apparent_zenith"].values
        cos_zenith = np.cos(np.radians(zenith))
        cos_zenith = np.clip(cos_zenith, 0, 1)

        # Map daily GHI to each hour's date
        dates = times.date
        unique_dates = np.unique(dates)

        # Build daily GHI lookup (handle mismatched lengths)
        daily_lookup = {}
        for i, d in enumerate(unique_dates):
            if i < len(ghi_daily):
                daily_lookup[d] = ghi_daily[i]

        # Distribute daily GHI across hours proportional to cos(zenith)
        hourly_ghi_w = np.zeros(len(times))
        for d in unique_dates:
            mask = dates == d
            cz = cos_zenith[mask]
            daily_total = daily_lookup.get(d, 0)
            cz_sum = cz.sum()
            if cz_sum > 0 and daily_total > 0:
                # Convert kWh/m²/day to W/m² distributed across hours
                # daily kWh/m² → hourly W/m²
                hourly_ghi_w[mask] = cz * (daily_total * 1000 / cz_sum)

        # Use Erbs model for decomposition
        # First compute extraterrestrial radiation for clearness index
        dni_extra = pvlib.irradiance.get_extra_radiation(times)
        cos_z_safe = np.clip(cos_zenith, 0.05, 1.0)

        # Clearness index
        kt = np.zeros_like(hourly_ghi_w)
        hor_extra = dni_extra.values * cos_z_safe
        valid = hor_extra > 0
        kt[valid] = hourly_ghi_w[valid] / hor_extra[valid]
        kt = np.clip(kt, 0, 1.0)

        # Erbs model for diffuse fraction
        kd = np.where(
            kt <= 0.22,
            1.0 - 0.09 * kt,
            np.where(
                kt <= 0.80,
                0.9511 - 0.1604 * kt + 4.388 * kt**2
                - 16.638 * kt**3 + 12.336 * kt**4,
                0.165,
            ),
        )

        dhi = hourly_ghi_w * kd
        # DNI from GHI and DHI: GHI = DNI × cos(z) + DHI
        dni = np.zeros_like(hourly_ghi_w)
        valid_z = cos_z_safe > 0.05
        dni[valid_z] = (hourly_ghi_w[valid_z] - dhi[valid_z]) / cos_z_safe[valid_z]
        dni = np.clip(dni, 0, 1400)

        return dni, dhi

    def irradiance_on_tilted_surface(
        self,
        tilt: float,
        azimuth: float,
        ghi_daily: np.ndarray,
        times: pd.DatetimeIndex,
        solpos: pd.DataFrame,
        dni_hourly: np.ndarray | None = None,
        dhi_hourly: np.ndarray | None = None,
    ) -> pd.Series:
        """Compute plane-of-array irradiance for a tilted surface.

        Uses pvlib.irradiance.get_total_irradiance() with the isotropic model.

        Parameters
        ----------
        tilt : float
            Panel tilt angle (0=horizontal, 90=vertical).
        azimuth : float
            Panel azimuth (180=south in Northern Hemisphere).
        ghi_daily : array
            Daily GHI in kWh/m²/day.
        times : DatetimeIndex
            Hourly timestamps.
        solpos : DataFrame
            Solar position data.
        dni_hourly, dhi_hourly : array, optional
            Pre-computed hourly DNI/DHI. If None, computed via Erbs model.

        Returns
        -------
        pd.Series
            Hourly plane-of-array total irradiance in W/m².
        """
        if dni_hourly is None or dhi_hourly is None:
            dni_hourly, dhi_hourly = self._decompose_ghi(ghi_daily, times, solpos)

        # Reconstruct hourly GHI
        cos_zenith = np.cos(np.radians(solpos["apparent_zenith"].values))
        cos_zenith = np.clip(cos_zenith, 0, 1)
        ghi_hourly = dni_hourly * cos_zenith + dhi_hourly

        # pvlib transposition
        poa = get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solpos["apparent_zenith"],
            solar_azimuth=solpos["azimuth"],
            dni=pd.Series(dni_hourly, index=times),
            ghi=pd.Series(ghi_hourly, index=times),
            dhi=pd.Series(dhi_hourly, index=times),
            albedo=self.surface_albedo,
            model="isotropic",
        )

        return poa["poa_global"]

    # -------------------------------------------------------------------
    # Full Simulation
    # -------------------------------------------------------------------

    def simulate_all_orientations(
        self,
        ghi_daily: np.ndarray,
        year: int = 2023,
    ) -> pd.DataFrame:
        """Simulate energy production for all orientation × tilt combinations.

        Parameters
        ----------
        ghi_daily : array
            Daily GHI values in kWh/m²/day (365 or 366 values).
        year : int
            Year to simulate.

        Returns
        -------
        pd.DataFrame
            Columns: direction, azimuth_deg, tilt_deg, month,
                     monthly_energy_kwh, annual_energy_kwh
        """
        logger.info("Simulating %d orientations × %d tilts",
                     len(self.azimuths), len(self.tilt_angles))

        times = pd.date_range(
            f"{year}-01-01", f"{year}-12-31 23:00", freq="h", tz="UTC",
        )
        solpos = self._get_location().get_solarposition(times)

        # Decompose GHI once
        dni_hourly, dhi_hourly = self._decompose_ghi(ghi_daily, times, solpos)

        records = []
        for direction, az in self.azimuths.items():
            for tilt in self.tilt_angles:
                poa = self.irradiance_on_tilted_surface(
                    tilt=tilt, azimuth=az,
                    ghi_daily=ghi_daily, times=times, solpos=solpos,
                    dni_hourly=dni_hourly, dhi_hourly=dhi_hourly,
                )

                # Convert W/m² hourly → kWh/m² daily → energy
                poa_kwh = poa.clip(lower=0) / 1000  # W → kW per m²

                # Monthly energy
                monthly_poa = poa_kwh.resample("ME").sum()  # kWh/m² per month

                for month_end, poa_month in monthly_poa.items():
                    energy = (
                        float(poa_month)
                        * self.panel_efficiency
                        * self.panel_area
                        * (1 - self.system_losses)
                    )
                    records.append({
                        "direction": direction,
                        "azimuth_deg": az,
                        "tilt_deg": tilt,
                        "month": month_end.month,
                        "monthly_energy_kwh": round(energy, 2),
                    })

        df = pd.DataFrame(records)

        # Add annual totals
        annual = df.groupby(["direction", "azimuth_deg", "tilt_deg"])[
            "monthly_energy_kwh"
        ].sum().reset_index()
        annual = annual.rename(columns={"monthly_energy_kwh": "annual_energy_kwh"})

        df = df.merge(annual, on=["direction", "azimuth_deg", "tilt_deg"])

        return df

    def optimal_orientation(
        self,
        ghi_daily: np.ndarray,
        year: int = 2023,
    ) -> dict[str, Any]:
        """Find the optimal panel orientation for maximum annual energy.

        Parameters
        ----------
        ghi_daily : array
            Daily GHI values (365/366 values).
        year : int
            Simulation year.

        Returns
        -------
        dict
            best_direction, best_tilt, best_azimuth, annual_energy_kwh,
            energy_gain_vs_horizontal_pct, energy_gain_vs_worst_pct
        """
        sim = self.simulate_all_orientations(ghi_daily, year)

        annual = sim.drop_duplicates(subset=["direction", "tilt_deg"])[
            ["direction", "azimuth_deg", "tilt_deg", "annual_energy_kwh"]
        ]

        best_row = annual.loc[annual["annual_energy_kwh"].idxmax()]
        worst_row = annual.loc[annual["annual_energy_kwh"].idxmin()]
        horizontal = annual[annual["tilt_deg"] == 0].iloc[0] if 0 in self.tilt_angles else best_row

        best_energy = float(best_row["annual_energy_kwh"])
        horiz_energy = float(horizontal["annual_energy_kwh"])
        worst_energy = float(worst_row["annual_energy_kwh"])

        return {
            "best_direction": best_row["direction"],
            "best_tilt": int(best_row["tilt_deg"]),
            "best_azimuth": int(best_row["azimuth_deg"]),
            "annual_energy_kwh": round(best_energy, 1),
            "energy_gain_vs_horizontal_pct": round(
                (best_energy - horiz_energy) / max(horiz_energy, 1) * 100, 1
            ),
            "energy_gain_vs_worst_pct": round(
                (best_energy - worst_energy) / max(worst_energy, 1) * 100, 1
            ),
            "worst_direction": worst_row["direction"],
            "worst_tilt": int(worst_row["tilt_deg"]),
        }

    def daily_profile_by_orientation(
        self,
        ghi_daily: np.ndarray,
        date: str = "2023-06-21",
        directions: list[str] | None = None,
        tilt: float = 30,
    ) -> pd.DataFrame:
        """Compute hourly energy profile for a specific date across orientations.

        Parameters
        ----------
        ghi_daily : array
            Full year daily GHI.
        date : str
            Target date (YYYY-MM-DD).
        directions : list[str], optional
            Directions to compare. Default: South, East, West, North.
        tilt : float
            Panel tilt angle.

        Returns
        -------
        pd.DataFrame
            Hourly energy by direction for the given date.
        """
        if directions is None:
            directions = ["South", "East", "West", "North"]

        year = int(date[:4])
        times = pd.date_range(
            f"{year}-01-01", f"{year}-12-31 23:00", freq="h", tz="UTC",
        )
        solpos = self._get_location().get_solarposition(times)
        dni_hourly, dhi_hourly = self._decompose_ghi(ghi_daily, times, solpos)

        # Filter to target date
        target = pd.Timestamp(date, tz="UTC")
        day_mask = times.date == target.date()

        records = []
        for direction in directions:
            az = self.azimuths.get(direction, 180)
            poa = self.irradiance_on_tilted_surface(
                tilt=tilt, azimuth=az,
                ghi_daily=ghi_daily, times=times, solpos=solpos,
                dni_hourly=dni_hourly, dhi_hourly=dhi_hourly,
            )

            poa_day = poa[day_mask]
            for hour_time, irr in poa_day.items():
                energy = (
                    max(float(irr), 0) / 1000
                    * self.panel_efficiency
                    * self.panel_area
                    * (1 - self.system_losses)
                )
                records.append({
                    "hour": hour_time.hour,
                    "direction": direction,
                    "irradiance_w_m2": max(float(irr), 0),
                    "energy_kwh": round(energy, 4),
                })

        return pd.DataFrame(records)

    def tilt_sensitivity_analysis(
        self,
        ghi_daily: np.ndarray,
        azimuth: float = 180,
        year: int = 2023,
        tilt_range: list[float] | None = None,
    ) -> pd.DataFrame:
        """Analyze energy sensitivity to tilt angle for a fixed azimuth.

        Parameters
        ----------
        ghi_daily : array
            Daily GHI values.
        azimuth : float
            Fixed azimuth angle (default 180 = south).
        year : int
            Simulation year.
        tilt_range : list[float], optional
            Custom tilt angles to test.

        Returns
        -------
        pd.DataFrame
            Tilt angle vs annual energy.
        """
        tilts = tilt_range or list(range(0, 91, 5))

        times = pd.date_range(
            f"{year}-01-01", f"{year}-12-31 23:00", freq="h", tz="UTC",
        )
        solpos = self._get_location().get_solarposition(times)
        dni_hourly, dhi_hourly = self._decompose_ghi(ghi_daily, times, solpos)

        records = []
        for tilt in tilts:
            poa = self.irradiance_on_tilted_surface(
                tilt=tilt, azimuth=azimuth,
                ghi_daily=ghi_daily, times=times, solpos=solpos,
                dni_hourly=dni_hourly, dhi_hourly=dhi_hourly,
            )
            annual_kwh_m2 = float(poa.clip(lower=0).sum()) / 1000
            annual_energy = (
                annual_kwh_m2
                * self.panel_efficiency
                * self.panel_area
                * (1 - self.system_losses)
            )
            records.append({
                "tilt_deg": tilt,
                "annual_energy_kwh": round(annual_energy, 1),
                "annual_kwh_m2": round(annual_kwh_m2, 1),
            })

        return pd.DataFrame(records)

    def seasonal_comparison(
        self,
        ghi_daily: np.ndarray,
        directions: list[str] | None = None,
        tilt: float = 30,
        year: int = 2023,
    ) -> pd.DataFrame:
        """Compare seasonal energy production across orientations.

        Parameters
        ----------
        ghi_daily : array
            Daily GHI values.
        directions : list[str], optional
            Directions to compare.
        tilt : float
            Panel tilt angle.
        year : int
            Simulation year.

        Returns
        -------
        pd.DataFrame
            Season × direction energy matrix.
        """
        if directions is None:
            directions = ["South", "East", "West", "North"]

        sim = self.simulate_all_orientations(ghi_daily, year)
        filtered = sim[
            (sim["direction"].isin(directions)) & (sim["tilt_deg"] == tilt)
        ].copy()

        # Map months to seasons
        season_map = {12: "DJF", 1: "DJF", 2: "DJF",
                      3: "MAM", 4: "MAM", 5: "MAM",
                      6: "JJA", 7: "JJA", 8: "JJA",
                      9: "SON", 10: "SON", 11: "SON"}
        filtered["season"] = filtered["month"].map(season_map)

        seasonal = filtered.groupby(["direction", "season"])[
            "monthly_energy_kwh"
        ].sum().reset_index()
        seasonal = seasonal.rename(columns={"monthly_energy_kwh": "seasonal_energy_kwh"})

        return seasonal

    # -------------------------------------------------------------------
    # Tracking Simulation
    # -------------------------------------------------------------------

    def simulate_tracking(
        self,
        ghi_daily: np.ndarray,
        year: int = 2023,
        mode: str = "single_axis",
    ) -> dict[str, Any]:
        """Simulate single-axis or dual-axis solar tracker performance.

        Single-axis: rotates east-west to follow the sun's daily arc.
        Dual-axis: tracks both azimuth and elevation for maximum irradiance.

        Parameters
        ----------
        ghi_daily : array
            Daily GHI values (365/366 values).
        year : int
            Simulation year.
        mode : str
            'single_axis' or 'dual_axis'.

        Returns
        -------
        dict
            tracking_mode, annual_energy_kwh, gain_vs_fixed_pct,
            best_fixed_energy_kwh.
        """
        times = pd.date_range(
            f"{year}-01-01", f"{year}-12-31 23:00", freq="h", tz="UTC",
        )
        solpos = self._get_location().get_solarposition(times)
        dni_hourly, dhi_hourly = self._decompose_ghi(ghi_daily, times, solpos)

        cos_zenith = np.cos(np.radians(solpos["apparent_zenith"].values))
        cos_zenith = np.clip(cos_zenith, 0, 1)
        ghi_hourly = dni_hourly * cos_zenith + dhi_hourly

        if mode == "dual_axis":
            # Dual-axis: tilt = zenith, azimuth = solar azimuth (always face sun)
            # POA = DNI + DHI (maximum possible capture)
            poa = np.clip(dni_hourly + dhi_hourly, 0, None)
        else:
            # Single-axis (N-S axis, tracks E-W):
            # Effective tilt follows solar elevation, azimuth = solar azimuth
            solar_elev = 90 - solpos["apparent_zenith"].values
            solar_elev = np.clip(solar_elev, 0, 90)
            # Tracking tilt = 90 - elevation (face the sun vertically)
            tracking_tilt = 90 - solar_elev

            # Simplified: single-axis captures ~85-90% of dual-axis gain
            poa = np.zeros(len(times))
            for i in range(len(times)):
                if solar_elev[i] > 0:
                    result = get_total_irradiance(
                        surface_tilt=float(tracking_tilt[i]),
                        surface_azimuth=float(solpos["azimuth"].values[i]),
                        solar_zenith=float(solpos["apparent_zenith"].values[i]),
                        solar_azimuth=float(solpos["azimuth"].values[i]),
                        dni=float(dni_hourly[i]),
                        ghi=float(ghi_hourly[i]),
                        dhi=float(dhi_hourly[i]),
                        albedo=self.surface_albedo,
                        model="isotropic",
                    )
                    val = result["poa_global"]
                    poa[i] = max(float(val.iloc[0] if hasattr(val, 'iloc') else val), 0)

        annual_kwh_m2 = float(np.sum(np.clip(poa, 0, None))) / 1000
        tracking_energy = (
            annual_kwh_m2
            * self.panel_efficiency
            * self.panel_area
            * (1 - self.system_losses)
        )

        # Get best fixed orientation for comparison
        optimal = self.optimal_orientation(ghi_daily, year)
        fixed_energy = optimal["annual_energy_kwh"]

        gain = (tracking_energy - fixed_energy) / max(fixed_energy, 1) * 100

        return {
            "tracking_mode": mode,
            "annual_energy_kwh": round(tracking_energy, 1),
            "best_fixed_energy_kwh": fixed_energy,
            "gain_vs_fixed_pct": round(gain, 1),
        }

    # -------------------------------------------------------------------
    # Shading Model
    # -------------------------------------------------------------------

    def horizon_shading(
        self,
        ghi_daily: np.ndarray,
        horizon_profile: dict[float, float] | None = None,
        year: int = 2023,
    ) -> dict[str, Any]:
        """Estimate energy loss from horizon obstructions.

        Models the effect of surrounding buildings/terrain that block
        low-angle sunlight. The horizon profile defines the minimum
        solar elevation visible at each azimuth.

        Parameters
        ----------
        ghi_daily : array
            Daily GHI values.
        horizon_profile : dict[float, float], optional
            Azimuth (degrees) -> minimum visible elevation (degrees).
            Default: flat horizon (no shading).
        year : int
            Simulation year.

        Returns
        -------
        dict
            shading_loss_pct, unshaded_energy_kwh, shaded_energy_kwh,
            worst_azimuth, worst_elevation.
        """
        if horizon_profile is None:
            # Default: some obstruction to the north/east
            horizon_profile = {
                0: 15, 45: 10, 90: 5, 135: 2, 180: 0,
                225: 2, 270: 5, 315: 10,
            }

        times = pd.date_range(
            f"{year}-01-01", f"{year}-12-31 23:00", freq="h", tz="UTC",
        )
        solpos = self._get_location().get_solarposition(times)

        solar_elev = 90 - solpos["apparent_zenith"].values
        solar_az = solpos["azimuth"].values

        # Interpolate horizon profile
        hz_azimuths = sorted(horizon_profile.keys())
        hz_elevations = [horizon_profile[a] for a in hz_azimuths]

        # Add wrap-around
        hz_azimuths_ext = [a - 360 for a in hz_azimuths] + hz_azimuths + [a + 360 for a in hz_azimuths]
        hz_elevations_ext = hz_elevations * 3

        min_elev = np.interp(solar_az % 360, hz_azimuths, hz_elevations)

        # Compute shading mask
        shaded = solar_elev < min_elev
        daylight = solar_elev > 0
        shaded_daylight = shaded & daylight

        # Compute energy with and without shading (using GHI as proxy)
        dni_hourly, dhi_hourly = self._decompose_ghi(ghi_daily, times, solpos)
        cos_z = np.clip(np.cos(np.radians(solpos["apparent_zenith"].values)), 0, 1)
        ghi_hourly = dni_hourly * cos_z + dhi_hourly

        total_ghi = float(np.sum(ghi_hourly[daylight]))
        shaded_ghi = float(np.sum(ghi_hourly[shaded_daylight]))

        loss_pct = (shaded_ghi / max(total_ghi, 1)) * 100

        unshaded_energy = (
            (total_ghi / 1000)
            * self.panel_efficiency
            * self.panel_area
            * (1 - self.system_losses)
        )
        shaded_energy = unshaded_energy * (1 - loss_pct / 100)

        return {
            "shading_loss_pct": round(loss_pct, 1),
            "unshaded_energy_kwh": round(unshaded_energy, 1),
            "shaded_energy_kwh": round(shaded_energy, 1),
            "hours_shaded": int(shaded_daylight.sum()),
            "total_daylight_hours": int(daylight.sum()),
        }

    def inter_row_shading(
        self,
        tilt: float = 30,
        row_spacing_ratio: float = 2.0,
    ) -> dict[str, float]:
        """Estimate inter-row shading loss for ground-mount solar farms.

        Uses geometric analysis of shadow length at winter solstice
        to determine minimum row spacing and shading losses.

        Parameters
        ----------
        tilt : float
            Panel tilt angle (degrees).
        row_spacing_ratio : float
            Ratio of row spacing to panel height (distance / height).

        Returns
        -------
        dict
            shadow_length_ratio, shading_loss_pct,
            min_spacing_ratio, current_spacing_adequate.
        """
        tilt_rad = np.radians(tilt)
        panel_height = np.sin(tilt_rad)  # Projected height

        # Winter solstice noon solar elevation
        lat_rad = np.radians(abs(self.latitude))
        winter_elev = 90 - abs(self.latitude) - 23.45  # Approximate
        winter_elev = max(winter_elev, 5)
        winter_elev_rad = np.radians(winter_elev)

        # Shadow length = panel_height / tan(solar_elevation)
        shadow_length = panel_height / max(np.tan(winter_elev_rad), 0.05)

        # Minimum spacing to avoid shading
        min_spacing = shadow_length + np.cos(tilt_rad)  # Panel ground projection

        # Actual spacing
        actual_spacing = row_spacing_ratio

        # Shading loss estimate
        if actual_spacing >= min_spacing:
            loss = 0.0
        else:
            # Linear model: loss proportional to overlap fraction
            overlap = (min_spacing - actual_spacing) / min_spacing
            loss = min(overlap * 25, 30)  # Cap at 30% loss

        return {
            "shadow_length_ratio": round(shadow_length, 2),
            "min_spacing_ratio": round(min_spacing, 2),
            "shading_loss_pct": round(loss, 1),
            "current_spacing_adequate": bool(actual_spacing >= min_spacing),
        }

    # -------------------------------------------------------------------
    # Bifacial Gain
    # -------------------------------------------------------------------

    def bifacial_gain(
        self,
        ghi_daily: np.ndarray,
        tilt: float = 30,
        bifaciality: float = 0.70,
        height: float = 1.0,
        year: int = 2023,
    ) -> dict[str, float]:
        """Estimate energy gain from bifacial (double-sided) solar panels.

        Bifacial panels collect reflected light on the rear side. The gain
        depends on ground albedo, panel height, and tilt angle.

        Parameters
        ----------
        ghi_daily : array
            Daily GHI values.
        tilt : float
            Panel tilt angle (degrees).
        bifaciality : float
            Ratio of rear-to-front efficiency (typically 0.65-0.80).
        height : float
            Panel mounting height above ground (meters).
        year : int
            Simulation year.

        Returns
        -------
        dict
            bifacial_gain_pct, rear_irradiance_pct,
            front_energy_kwh, total_energy_kwh.
        """
        # Rear-side irradiance model (simplified)
        # Ground-reflected irradiance reaching rear = GHI * albedo * view_factor
        tilt_rad = np.radians(tilt)

        # View factor depends on tilt and height
        # Higher panels and steeper tilts see more ground reflection
        view_factor = 0.5 * (1 - np.cos(tilt_rad))  # Isotropic sky model
        height_factor = min(height / 1.5, 1.0)  # Normalized to 1.5m reference

        rear_fraction = self.surface_albedo * view_factor * height_factor * bifaciality

        # Compute front energy
        avg_daily_ghi = float(np.mean(ghi_daily))
        annual_ghi = avg_daily_ghi * len(ghi_daily)
        front_energy = (
            annual_ghi
            * self.panel_efficiency
            * self.panel_area
            * (1 - self.system_losses)
        )

        rear_energy = front_energy * rear_fraction
        total_energy = front_energy + rear_energy
        gain_pct = (rear_energy / max(front_energy, 1)) * 100

        return {
            "bifacial_gain_pct": round(gain_pct, 1),
            "rear_irradiance_pct": round(rear_fraction * 100, 1),
            "front_energy_kwh": round(front_energy, 1),
            "total_energy_kwh": round(total_energy, 1),
        }


# ---------------------------------------------------------------------------
# Rooftop Suitability Scoring
# ---------------------------------------------------------------------------

class RooftopScorer(param.Parameterized):
    """Score rooftop solar suitability on a 0-100 scale.

    Combines multiple factors into a weighted composite score:
    - Solar resource quality (GHI level)
    - Optimal tilt match (how close roof pitch is to ideal)
    - Climate stability (irradiance variability)
    - Temperature factor (extreme heat reduces efficiency)

    Parameters
    ----------
    latitude : float
        Location latitude.
    longitude : float
        Location longitude.
    """

    latitude = param.Number(default=28.6, bounds=(-90, 90))
    longitude = param.Number(default=77.2, bounds=(-180, 180))

    # Scoring weights
    weight_solar = param.Number(default=0.40, doc="Weight for solar resource quality")
    weight_tilt = param.Number(default=0.20, doc="Weight for tilt match")
    weight_stability = param.Number(default=0.20, doc="Weight for climate stability")
    weight_temperature = param.Number(default=0.20, doc="Weight for temperature factor")

    def score(
        self,
        avg_daily_ghi: float,
        optimal_tilt: float,
        roof_tilt: float = 30,
        variability_index: float = 0.15,
        avg_temperature: float = 25,
    ) -> dict[str, Any]:
        """Compute rooftop suitability score.

        Parameters
        ----------
        avg_daily_ghi : float
            Average daily GHI in kWh/m2/day.
        optimal_tilt : float
            Optimal panel tilt angle for the location.
        roof_tilt : float
            Actual roof pitch in degrees.
        variability_index : float
            Solar variability (std/mean of daily GHI). Lower = more stable.
        avg_temperature : float
            Average annual temperature in Celsius.

        Returns
        -------
        dict
            total_score (0-100), component scores, rating, recommendations.
        """
        # 1. Solar resource score (0-100)
        # 7+ kWh/m2/day = 100, 1 kWh/m2/day = 0
        solar_score = min(100, max(0, (avg_daily_ghi - 1) / 6 * 100))

        # 2. Tilt match score (0-100)
        # Perfect match = 100, 45+ degree difference = 0
        tilt_diff = abs(roof_tilt - optimal_tilt)
        tilt_score = max(0, 100 - tilt_diff * (100 / 45))

        # 3. Stability score (0-100)
        # Variability < 0.10 = 100, > 0.40 = 0
        stability_score = max(0, min(100, (0.40 - variability_index) / 0.30 * 100))

        # 4. Temperature score (0-100)
        # Moderate temps (15-25C) are ideal for solar
        # Very hot (>40C) or very cold (<-10C) reduce score
        if 15 <= avg_temperature <= 25:
            temp_score = 100
        elif avg_temperature > 25:
            temp_score = max(0, 100 - (avg_temperature - 25) * 4)
        else:
            temp_score = max(0, 100 - (15 - avg_temperature) * 3)

        # Weighted total
        total = (
            solar_score * self.weight_solar
            + tilt_score * self.weight_tilt
            + stability_score * self.weight_stability
            + temp_score * self.weight_temperature
        )

        # Rating
        if total >= 80:
            rating = "Excellent"
        elif total >= 60:
            rating = "Good"
        elif total >= 40:
            rating = "Moderate"
        elif total >= 20:
            rating = "Poor"
        else:
            rating = "Not Recommended"

        # Recommendations
        recommendations = []
        if solar_score < 50:
            recommendations.append("Low solar resource - consider alternative energy")
        if tilt_score < 50:
            recommendations.append(
                f"Roof pitch ({roof_tilt}) differs from optimal ({optimal_tilt:.0f}) - "
                "consider adjustable mounting"
            )
        if stability_score < 50:
            recommendations.append("High variability - consider battery storage")
        if temp_score < 50:
            recommendations.append("Extreme temperatures reduce panel efficiency")

        return {
            "total_score": round(total, 1),
            "rating": rating,
            "components": {
                "solar_resource": round(solar_score, 1),
                "tilt_match": round(tilt_score, 1),
                "climate_stability": round(stability_score, 1),
                "temperature": round(temp_score, 1),
            },
            "weights": {
                "solar_resource": self.weight_solar,
                "tilt_match": self.weight_tilt,
                "climate_stability": self.weight_stability,
                "temperature": self.weight_temperature,
            },
            "recommendations": recommendations,
        }
