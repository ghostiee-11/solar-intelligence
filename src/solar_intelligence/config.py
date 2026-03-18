"""Configuration constants and default parameters for Solar Intelligence."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# NASA POWER API
# ---------------------------------------------------------------------------
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal"

NASA_POWER_SOLAR_PARAMS = [
    "ALLSKY_SFC_SW_DWN",   # GHI — All Sky Surface Shortwave Downward Irradiance (kWh/m²/day)
    "CLRSKY_SFC_SW_DWN",   # Clear Sky GHI
    "ALLSKY_SFC_SW_DNI",   # DNI — Direct Normal Irradiance
    "ALLSKY_SFC_SW_DIFF",  # DHI — Diffuse Horizontal Irradiance
    "ALLSKY_KT",           # Clearness Index
    "T2M",                 # Temperature at 2m (°C)
    "T2M_MAX",             # Max daily temperature
    "T2M_MIN",             # Min daily temperature
    "WS2M",                # Wind speed at 2m (m/s)
    "RH2M",                # Relative humidity at 2m (%)
]

NASA_POWER_COMMUNITIES = {
    "renewable_energy": "RE",
    "sustainable_buildings": "SB",
    "agroclimatology": "AG",
}

# Cache TTL in days
CACHE_TTL_DAYS = 30

# ---------------------------------------------------------------------------
# ERA5 (Copernicus Climate Data Store) API
# ---------------------------------------------------------------------------
ERA5_CDS_URL = "https://cds.climate.copernicus.eu/api"

ERA5_SOLAR_VARIABLES = [
    "surface_solar_radiation_downwards",        # ssrd — GHI (J/m2 cumulative)
    "total_sky_direct_solar_radiation_at_surface",  # fdir — DNI (J/m2 cumulative)
    "2m_temperature",                            # t2m (K)
    "10m_u_component_of_wind",                   # u10 (m/s)
    "10m_v_component_of_wind",                   # v10 (m/s)
    "2m_dewpoint_temperature",                   # d2m (K)
    "total_cloud_cover",                         # tcc (0-1)
    "surface_pressure",                          # sp (Pa)
]

ERA5_DATASET_NAME = "reanalysis-era5-single-levels"

# ERA5 variable name -> standard name mapping
ERA5_VAR_MAP = {
    "ssrd": "ALLSKY_SFC_SW_DWN",
    "fdir": "ALLSKY_SFC_SW_DNI",
    "t2m": "T2M",
    "u10": "WS2M",       # will combine u10+v10 into wind speed
    "v10": "WS2M_v",     # helper for wind speed calc
    "d2m": "DEWPOINT",
    "tcc": "CLOUD_COVER",
    "sp": "PRESSURE",
}

# ---------------------------------------------------------------------------
# Default Solar Panel Specifications
# ---------------------------------------------------------------------------
DEFAULT_PANEL_EFFICIENCY = 0.20       # 20% — typical monocrystalline
DEFAULT_PANEL_AREA = 1.7              # m² — standard 60-cell panel
DEFAULT_NUM_PANELS = 10
DEFAULT_SYSTEM_LOSSES = 0.14          # 14% — wiring, soiling, mismatch
DEFAULT_INVERTER_EFFICIENCY = 0.96    # 96%
DEFAULT_TEMP_COEFFICIENT = -0.004     # -0.4%/°C (power temperature coefficient)
DEFAULT_NOCT = 45                     # °C — Nominal Operating Cell Temperature
DEFAULT_STC_TEMP = 25                 # °C — Standard Test Conditions temperature
DEFAULT_ALBEDO = 0.25                 # Ground surface reflectance (grass)

# ---------------------------------------------------------------------------
# Solar Panel Orientations
# ---------------------------------------------------------------------------
ORIENTATIONS = {
    "North": 0,
    "North-East": 45,
    "East": 90,
    "South-East": 135,
    "South": 180,
    "South-West": 225,
    "West": 270,
    "North-West": 315,
}

DEFAULT_TILT_ANGLES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# ---------------------------------------------------------------------------
# Currency Settings
# ---------------------------------------------------------------------------
CURRENCIES = {
    "INR": {"symbol": "₹", "name": "Indian Rupee", "locale": "en_IN"},
    "USD": {"symbol": "$", "name": "US Dollar", "locale": "en_US"},
    "EUR": {"symbol": "€", "name": "Euro", "locale": "en_EU"},
    "GBP": {"symbol": "£", "name": "British Pound", "locale": "en_GB"},
}

# Defaults per currency
CURRENCY_DEFAULTS = {
    "INR": {
        "system_cost": 500000,          # ₹5 lakh for 3kW system
        "electricity_rate": 8.0,        # ₹8/kWh
        "maintenance_cost": 5000,       # ₹5000/year
        "incentive_percent": 0.40,      # 40% MNRE subsidy
        "rate_increase": 0.05,          # 5% annual increase
    },
    "USD": {
        "system_cost": 15000,
        "electricity_rate": 0.12,
        "maintenance_cost": 200,
        "incentive_percent": 0.30,      # 30% ITC
        "rate_increase": 0.03,
    },
    "EUR": {
        "system_cost": 12000,
        "electricity_rate": 0.30,
        "maintenance_cost": 150,
        "incentive_percent": 0.20,
        "rate_increase": 0.03,
    },
    "GBP": {
        "system_cost": 10000,
        "electricity_rate": 0.28,
        "maintenance_cost": 120,
        "incentive_percent": 0.00,      # No direct subsidy (SEG payments instead)
        "rate_increase": 0.04,
    },
}

DEFAULT_CURRENCY = "INR"

# ---------------------------------------------------------------------------
# Financial Defaults (use INR as default)
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_COST = CURRENCY_DEFAULTS[DEFAULT_CURRENCY]["system_cost"]
DEFAULT_ELECTRICITY_RATE = CURRENCY_DEFAULTS[DEFAULT_CURRENCY]["electricity_rate"]
DEFAULT_RATE_INCREASE = CURRENCY_DEFAULTS[DEFAULT_CURRENCY]["rate_increase"]
DEFAULT_INCENTIVE_PERCENT = CURRENCY_DEFAULTS[DEFAULT_CURRENCY]["incentive_percent"]
DEFAULT_MAINTENANCE_COST = CURRENCY_DEFAULTS[DEFAULT_CURRENCY]["maintenance_cost"]
DEFAULT_PANEL_DEGRADATION = 0.005     # 0.5%/year
DEFAULT_SYSTEM_LIFETIME = 25          # years

# ---------------------------------------------------------------------------
# Carbon & Environmental
# ---------------------------------------------------------------------------
CARBON_FACTOR_KG_PER_KWH = 0.42      # kg CO2 per kWh (US grid average)
TREES_KG_CO2_PER_YEAR = 22           # kg CO2 absorbed per tree per year
CAR_KG_CO2_PER_MILE = 0.404          # kg CO2 per mile driven

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
SOLAR_CMAP = "YlOrRd"
IRRADIANCE_CMAP = "inferno"
ENERGY_CMAP = "viridis"
COMPARISON_CMAP = "Category10"
ANOMALY_CMAP = "RdBu_r"

# Global irradiance reference ranges (kWh/m²/day)
IRRADIANCE_EXCELLENT = 6.0    # Sahara, Middle East, Australia
IRRADIANCE_GOOD = 4.5         # Southern US, Mediterranean
IRRADIANCE_MODERATE = 3.0     # Northern US, Central Europe
IRRADIANCE_LOW = 1.5          # Scandinavia, UK winter
