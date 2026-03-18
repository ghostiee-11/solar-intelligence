"""Solar energy production estimation module.

Calculates expected energy output from solar PV systems based on
irradiance data, panel specifications, and environmental conditions.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import param
import xarray as xr

from solar_intelligence.config import (
    DEFAULT_ALBEDO,
    DEFAULT_INVERTER_EFFICIENCY,
    DEFAULT_NOCT,
    DEFAULT_NUM_PANELS,
    DEFAULT_PANEL_AREA,
    DEFAULT_PANEL_EFFICIENCY,
    DEFAULT_STC_TEMP,
    DEFAULT_SYSTEM_LOSSES,
    DEFAULT_TEMP_COEFFICIENT,
)

logger = logging.getLogger(__name__)


class EnergyEstimator(param.Parameterized):
    """Estimate solar PV energy production.

    Uses the standard PV performance model:
        E = GHI × η_panel × A × N × (1 - losses) × η_inverter × f_temp

    Where f_temp accounts for cell temperature effects:
        T_cell = T_ambient + (NOCT - 20) × GHI_W / 800
        f_temp = 1 + γ × (T_cell - T_STC)

    Parameters
    ----------
    panel_efficiency : float
        Solar panel conversion efficiency (0.05 to 0.40).
    panel_area : float
        Single panel area in m².
    num_panels : int
        Number of panels in the system.
    system_losses : float
        Combined system losses fraction (wiring, soiling, mismatch).
    inverter_efficiency : float
        DC-to-AC inverter efficiency.
    temp_coefficient : float
        Power temperature coefficient (%/°C), typically negative.
    noct : float
        Nominal Operating Cell Temperature in °C.
    """

    panel_efficiency = param.Number(
        default=DEFAULT_PANEL_EFFICIENCY, bounds=(0.05, 0.40),
        doc="Panel conversion efficiency",
    )
    panel_area = param.Number(
        default=DEFAULT_PANEL_AREA, bounds=(0.1, 100.0),
        doc="Single panel area (m²)",
    )
    num_panels = param.Integer(
        default=DEFAULT_NUM_PANELS, bounds=(1, 10000),
        doc="Number of panels",
    )
    system_losses = param.Number(
        default=DEFAULT_SYSTEM_LOSSES, bounds=(0.0, 0.5),
        doc="System losses fraction",
    )
    inverter_efficiency = param.Number(
        default=DEFAULT_INVERTER_EFFICIENCY, bounds=(0.8, 1.0),
        doc="Inverter efficiency",
    )
    temp_coefficient = param.Number(
        default=DEFAULT_TEMP_COEFFICIENT, bounds=(-0.01, 0.0),
        doc="Power temp coefficient (%/°C)",
    )
    noct = param.Number(
        default=DEFAULT_NOCT, bounds=(35, 55),
        doc="Nominal Operating Cell Temperature (°C)",
    )

    @property
    def total_area(self) -> float:
        """Total panel area in m²."""
        return self.panel_area * self.num_panels

    @property
    def system_capacity_kw(self) -> float:
        """Nominal system capacity in kW (at STC: 1000 W/m²)."""
        return self.total_area * self.panel_efficiency

    # -------------------------------------------------------------------
    # Cell Temperature
    # -------------------------------------------------------------------

    def cell_temperature(
        self,
        ambient_temp: float | np.ndarray,
        ghi_kwh: float | np.ndarray,
    ) -> float | np.ndarray:
        """Estimate solar cell temperature.

        Uses the NOCT model:
            T_cell = T_ambient + (NOCT - 20) × G / 800

        where G is irradiance in W/m² (converted from kWh/m²/day).

        Parameters
        ----------
        ambient_temp : float or array
            Ambient temperature in °C.
        ghi_kwh : float or array
            Daily GHI in kWh/m²/day.

        Returns
        -------
        float or array
            Estimated cell temperature in °C.
        """
        # Convert daily kWh/m² to approximate peak W/m²
        # Assume ~5 peak sun hours → daily kWh/m² ≈ peak W/m² / 5
        # More precisely: GHI_W ≈ GHI_daily * 1000 / day_hours
        # For NOCT purposes, use effective irradiance
        ghi_w = np.asarray(ghi_kwh) * 1000 / 5.0  # approximate peak W/m²
        ghi_w = np.clip(ghi_w, 0, 1400)

        t_cell = np.asarray(ambient_temp) + (self.noct - 20) * ghi_w / 800
        return t_cell

    def temperature_factor(
        self,
        ambient_temp: float | np.ndarray,
        ghi_kwh: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute temperature derating factor.

        Parameters
        ----------
        ambient_temp : float or array
            Ambient temperature in °C.
        ghi_kwh : float or array
            Daily GHI in kWh/m²/day.

        Returns
        -------
        float or array
            Temperature factor (1.0 = no derating).
        """
        t_cell = self.cell_temperature(ambient_temp, ghi_kwh)
        factor = 1.0 + self.temp_coefficient * (t_cell - DEFAULT_STC_TEMP)
        return np.clip(factor, 0.5, 1.2)

    # -------------------------------------------------------------------
    # Energy Estimation
    # -------------------------------------------------------------------

    def estimate_daily_energy(
        self,
        ghi: float,
        temperature: float | None = None,
    ) -> float:
        """Estimate daily energy production.

        Parameters
        ----------
        ghi : float
            Daily Global Horizontal Irradiance in kWh/m²/day.
        temperature : float, optional
            Ambient temperature in °C. If None, no temperature derating.

        Returns
        -------
        float
            Estimated daily energy production in kWh.
        """
        if ghi <= 0:
            return 0.0

        f_temp = 1.0
        if temperature is not None:
            f_temp = float(self.temperature_factor(temperature, ghi))

        energy = (
            ghi
            * self.panel_efficiency
            * self.total_area
            * (1 - self.system_losses)
            * self.inverter_efficiency
            * f_temp
        )

        return max(0.0, energy)

    def estimate_from_dataset(
        self,
        dataset: xr.Dataset,
        ghi_var: str = "ALLSKY_SFC_SW_DWN",
        temp_var: str = "T2M",
    ) -> pd.DataFrame:
        """Estimate energy production from an xarray Dataset.

        Parameters
        ----------
        dataset : xr.Dataset
            Solar radiation dataset with time dimension.
        ghi_var : str
            GHI variable name.
        temp_var : str
            Temperature variable name.

        Returns
        -------
        pd.DataFrame
            Daily energy estimates with time, GHI, temperature, and energy columns.
        """
        ghi = dataset[ghi_var].values
        temp = dataset[temp_var].values if temp_var in dataset else None

        # Vectorized computation
        f_temp = np.ones_like(ghi)
        if temp is not None:
            f_temp = self.temperature_factor(temp, ghi)

        daily_energy = (
            ghi
            * self.panel_efficiency
            * self.total_area
            * (1 - self.system_losses)
            * self.inverter_efficiency
            * f_temp
        )
        daily_energy = np.maximum(daily_energy, 0)

        df = pd.DataFrame({
            "time": dataset.time.values,
            "ghi_kwh_m2": ghi,
            "energy_kwh": daily_energy,
        })

        if temp is not None:
            df["temperature_c"] = temp
            df["temp_factor"] = f_temp

        return df

    def estimate_monthly_energy(
        self,
        dataset: xr.Dataset,
        ghi_var: str = "ALLSKY_SFC_SW_DWN",
        temp_var: str = "T2M",
    ) -> pd.DataFrame:
        """Estimate monthly energy production.

        Parameters
        ----------
        dataset : xr.Dataset
            Solar radiation dataset.

        Returns
        -------
        pd.DataFrame
            Monthly energy totals with month name, days, and energy.
        """
        daily = self.estimate_from_dataset(dataset, ghi_var, temp_var)
        daily["time"] = pd.to_datetime(daily["time"])
        daily["month"] = daily["time"].dt.month
        daily["year"] = daily["time"].dt.year

        # Monthly totals
        monthly = daily.groupby("month").agg(
            avg_daily_ghi=("ghi_kwh_m2", "mean"),
            avg_daily_energy=("energy_kwh", "mean"),
            total_energy=("energy_kwh", "sum"),
            days_count=("energy_kwh", "count"),
        ).reset_index()

        # Average per month (across years)
        n_years = daily["year"].nunique()
        monthly["avg_monthly_energy"] = monthly["total_energy"] / n_years

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m - 1])

        return monthly

    def estimate_annual_energy(
        self,
        dataset: xr.Dataset,
        ghi_var: str = "ALLSKY_SFC_SW_DWN",
        temp_var: str = "T2M",
    ) -> float:
        """Estimate average annual energy production in kWh.

        Parameters
        ----------
        dataset : xr.Dataset
            Solar radiation dataset.

        Returns
        -------
        float
            Average annual energy production in kWh.
        """
        daily = self.estimate_from_dataset(dataset, ghi_var, temp_var)
        avg_daily = daily["energy_kwh"].mean()
        return float(avg_daily * 365.25)

    def capacity_factor(
        self,
        annual_energy_kwh: float,
    ) -> float:
        """Compute capacity factor.

        Capacity factor = actual energy / maximum theoretical energy
        Maximum = system_capacity_kW × 8760 hours/year

        Parameters
        ----------
        annual_energy_kwh : float
            Actual annual energy in kWh.

        Returns
        -------
        float
            Capacity factor as percentage (0-100).
        """
        max_energy = self.system_capacity_kw * 8760
        if max_energy <= 0:
            return 0.0
        return (annual_energy_kwh / max_energy) * 100

    def system_summary(
        self,
        dataset: xr.Dataset,
        ghi_var: str = "ALLSKY_SFC_SW_DWN",
        temp_var: str = "T2M",
    ) -> dict[str, Any]:
        """Generate comprehensive system performance summary.

        Returns
        -------
        dict
            System specs, energy output, and performance metrics.
        """
        annual = self.estimate_annual_energy(dataset, ghi_var, temp_var)
        monthly = self.estimate_monthly_energy(dataset, ghi_var, temp_var)
        cf = self.capacity_factor(annual)

        return {
            "system": {
                "capacity_kw": round(self.system_capacity_kw, 2),
                "total_area_m2": round(self.total_area, 1),
                "num_panels": self.num_panels,
                "panel_efficiency": self.panel_efficiency,
                "system_losses": self.system_losses,
            },
            "production": {
                "annual_energy_kwh": round(annual, 1),
                "avg_daily_energy_kwh": round(annual / 365.25, 2),
                "best_month": monthly.loc[monthly["avg_monthly_energy"].idxmax(), "month_name"],
                "best_month_energy_kwh": round(monthly["avg_monthly_energy"].max(), 1),
                "worst_month": monthly.loc[monthly["avg_monthly_energy"].idxmin(), "month_name"],
                "worst_month_energy_kwh": round(monthly["avg_monthly_energy"].min(), 1),
            },
            "performance": {
                "capacity_factor_pct": round(cf, 1),
                "specific_yield_kwh_kwp": round(annual / max(self.system_capacity_kw, 0.001), 0),
            },
        }
