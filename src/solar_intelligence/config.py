"""Configuration constants and default parameters for Solar Intelligence."""

from __future__ import annotations

import logging
from datetime import datetime
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
    "CNY": {"symbol": "¥", "name": "Chinese Yuan", "locale": "zh_CN"},
    "JPY": {"symbol": "¥", "name": "Japanese Yen", "locale": "ja_JP"},
    "AUD": {"symbol": "A$", "name": "Australian Dollar", "locale": "en_AU"},
    "BRL": {"symbol": "R$", "name": "Brazilian Real", "locale": "pt_BR"},
    "ZAR": {"symbol": "R", "name": "South African Rand", "locale": "en_ZA"},
    "CAD": {"symbol": "C$", "name": "Canadian Dollar", "locale": "en_CA"},
    "KRW": {"symbol": "₩", "name": "South Korean Won", "locale": "ko_KR"},
    "AED": {"symbol": "د.إ", "name": "UAE Dirham", "locale": "ar_AE"},
    "MXN": {"symbol": "$", "name": "Mexican Peso", "locale": "es_MX"},
    "SGD": {"symbol": "S$", "name": "Singapore Dollar", "locale": "en_SG"},
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
    "CNY": {
        "system_cost": 35000,           # ~¥35,000 for 3kW system
        "electricity_rate": 0.55,
        "maintenance_cost": 800,
        "incentive_percent": 0.15,
        "rate_increase": 0.03,
    },
    "JPY": {
        "system_cost": 1050000,         # ~¥1,050,000 for 3kW system
        "electricity_rate": 30.0,
        "maintenance_cost": 25000,
        "incentive_percent": 0.10,
        "rate_increase": 0.02,
    },
    "AUD": {
        "system_cost": 8000,
        "electricity_rate": 0.30,
        "maintenance_cost": 200,
        "incentive_percent": 0.30,      # SRES rebate
        "rate_increase": 0.04,
    },
    "BRL": {
        "system_cost": 25000,           # R$25,000 for 3kW system
        "electricity_rate": 0.80,
        "maintenance_cost": 500,
        "incentive_percent": 0.00,
        "rate_increase": 0.06,
    },
    "ZAR": {
        "system_cost": 90000,           # R90,000 for 3kW system
        "electricity_rate": 2.50,
        "maintenance_cost": 2000,
        "incentive_percent": 0.25,
        "rate_increase": 0.08,
    },
    "CAD": {
        "system_cost": 12000,
        "electricity_rate": 0.13,
        "maintenance_cost": 200,
        "incentive_percent": 0.25,
        "rate_increase": 0.03,
    },
    "KRW": {
        "system_cost": 5000000,         # ₩5,000,000 for 3kW system
        "electricity_rate": 120.0,
        "maintenance_cost": 100000,
        "incentive_percent": 0.30,
        "rate_increase": 0.03,
    },
    "AED": {
        "system_cost": 20000,
        "electricity_rate": 0.29,
        "maintenance_cost": 500,
        "incentive_percent": 0.00,
        "rate_increase": 0.02,
    },
    "MXN": {
        "system_cost": 120000,          # MX$120,000 for 3kW system
        "electricity_rate": 1.50,
        "maintenance_cost": 3000,
        "incentive_percent": 0.00,
        "rate_increase": 0.05,
    },
    "SGD": {
        "system_cost": 10000,
        "electricity_rate": 0.25,
        "maintenance_cost": 200,
        "incentive_percent": 0.00,
        "rate_increase": 0.02,
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

# ---------------------------------------------------------------------------
# Country-Specific Carbon Emission Factors (kg CO2 per kWh)
# Source: IEA Emissions Factors (2023 edition)
# ---------------------------------------------------------------------------
COUNTRY_CARBON_FACTORS: dict[str, float] = {
    "IN": 0.82,   # India
    "US": 0.42,   # United States
    "DE": 0.35,   # Germany
    "FR": 0.06,   # France (nuclear-heavy)
    "GB": 0.23,   # United Kingdom
    "CN": 0.58,   # China
    "JP": 0.47,   # Japan
    "BR": 0.07,   # Brazil (hydro-heavy)
    "AU": 0.66,   # Australia
    "ZA": 0.93,   # South Africa (coal-heavy)
    "CA": 0.12,   # Canada (hydro-heavy)
    "IT": 0.33,   # Italy
    "ES": 0.22,   # Spain
    "KR": 0.46,   # South Korea
    "MX": 0.43,   # Mexico
    "SA": 0.62,   # Saudi Arabia
    "AE": 0.42,   # UAE
    "EG": 0.47,   # Egypt
    "NG": 0.43,   # Nigeria
    "KE": 0.03,   # Kenya (geothermal-heavy)
    "TR": 0.41,   # Turkey
    "TH": 0.49,   # Thailand
    "ID": 0.72,   # Indonesia
    "PK": 0.49,   # Pakistan
    "BD": 0.60,   # Bangladesh
    "VN": 0.52,   # Vietnam
    "PH": 0.61,   # Philippines
    "RU": 0.33,   # Russia
    "NL": 0.33,   # Netherlands
    "SE": 0.01,   # Sweden (nuclear + hydro)
    "NO": 0.01,   # Norway (hydro-heavy)
    "PL": 0.66,   # Poland (coal-heavy)
    "AR": 0.31,   # Argentina
    "CL": 0.35,   # Chile
    "CO": 0.16,   # Colombia (hydro-heavy)
}

_DEFAULT_CARBON_FACTOR = 0.42  # World average fallback


def get_carbon_factor(country_code: str) -> float:
    """Return carbon emission factor for a country, with fallback to world average."""
    return COUNTRY_CARBON_FACTORS.get(country_code.upper(), _DEFAULT_CARBON_FACTOR)


# ---------------------------------------------------------------------------
# Named Constants (replace magic numbers throughout codebase)
# ---------------------------------------------------------------------------
PEAK_SUN_HOURS_APPROX = 5.0          # Average peak sun hours per day
MAX_IRRADIANCE_W_M2 = 1400           # Maximum solar irradiance at Earth surface (W/m2)
MIN_TEMP_FACTOR = 0.5                # Minimum temperature correction factor
MAX_TEMP_FACTOR = 1.2                # Maximum temperature correction factor
NOCT_REFERENCE_IRRADIANCE = 800      # W/m2 — NOCT reference irradiance
NOCT_REFERENCE_TEMP = 20             # deg C — NOCT reference ambient temperature

# ---------------------------------------------------------------------------
# Dynamic Date Range (auto-update each year)
# ---------------------------------------------------------------------------


def default_end_year() -> int:
    """Return the last complete calendar year for data queries."""
    return datetime.now().year - 1


def default_start_year() -> int:
    """Return 3 years before the end year for a 4-year analysis window."""
    return default_end_year() - 3


DEFAULT_START_YEAR = default_start_year()
DEFAULT_END_YEAR = default_end_year()

# ---------------------------------------------------------------------------
# Colorblind-Friendly Palette (Okabe-Ito)
# ---------------------------------------------------------------------------
CB_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]
CB_SOLAR_CMAP = "cividis"
CB_IRRADIANCE_CMAP = "viridis"

# ---------------------------------------------------------------------------
# Default Appliance Wattages (kW)
# ---------------------------------------------------------------------------
DEFAULT_APPLIANCE_WATTAGES: dict[str, float] = {
    "air_conditioner": 1.5,
    "central_ac": 3.5,
    "refrigerator": 0.15,
    "washing_machine": 0.5,
    "water_heater": 2.0,
    "led_light": 0.01,
    "ceiling_fan": 0.075,
    "ev_charger_l2": 7.4,
    "microwave": 1.2,
    "laptop": 0.065,
    "television": 0.1,
    "iron": 1.0,
    "hair_dryer": 1.5,
}

# ---------------------------------------------------------------------------
# Payback Period Thresholds (years) by Region
# ---------------------------------------------------------------------------
PAYBACK_THRESHOLDS: dict[str, dict[str, int]] = {
    "default": {"excellent": 5, "very_good": 8, "good": 12, "moderate": 20},
    "IN": {"excellent": 4, "very_good": 6, "good": 8, "moderate": 12},
    "DE": {"excellent": 7, "very_good": 10, "good": 14, "moderate": 20},
}

# ---------------------------------------------------------------------------
# Country Profiles (currency, carbon, electricity, subsidies)
# ---------------------------------------------------------------------------
COUNTRY_PROFILES: dict[str, dict] = {
    "IN": {
        "default_currency": "INR",
        "carbon_factor": 0.82,
        "electricity_rate": 8.0,
        "subsidy_percent": 0.40,
        "subsidy_type": "Central Financial Assistance (MNRE)",
    },
    "US": {
        "default_currency": "USD",
        "carbon_factor": 0.42,
        "electricity_rate": 0.12,
        "subsidy_percent": 0.30,
        "subsidy_type": "Investment Tax Credit (ITC)",
    },
    "DE": {
        "default_currency": "EUR",
        "carbon_factor": 0.35,
        "electricity_rate": 0.30,
        "subsidy_percent": 0.20,
        "subsidy_type": "Feed-in Tariff (EEG)",
    },
    "FR": {
        "default_currency": "EUR",
        "carbon_factor": 0.06,
        "electricity_rate": 0.22,
        "subsidy_percent": 0.25,
        "subsidy_type": "Feed-in Premium + Tax Credit",
    },
    "GB": {
        "default_currency": "GBP",
        "carbon_factor": 0.23,
        "electricity_rate": 0.28,
        "subsidy_percent": 0.00,
        "subsidy_type": "Smart Export Guarantee (SEG)",
    },
    "CN": {
        "default_currency": "CNY",
        "carbon_factor": 0.58,
        "electricity_rate": 0.55,
        "subsidy_percent": 0.15,
        "subsidy_type": "Provincial Feed-in Tariff",
    },
    "JP": {
        "default_currency": "JPY",
        "carbon_factor": 0.47,
        "electricity_rate": 30.0,
        "subsidy_percent": 0.10,
        "subsidy_type": "Feed-in Tariff (FIT)",
    },
    "BR": {
        "default_currency": "BRL",
        "carbon_factor": 0.07,
        "electricity_rate": 0.80,
        "subsidy_percent": 0.00,
        "subsidy_type": "Net Metering",
    },
    "AU": {
        "default_currency": "AUD",
        "carbon_factor": 0.66,
        "electricity_rate": 0.30,
        "subsidy_percent": 0.30,
        "subsidy_type": "Small-scale Renewable Energy Scheme (SRES)",
    },
    "ZA": {
        "default_currency": "ZAR",
        "carbon_factor": 0.93,
        "electricity_rate": 2.50,
        "subsidy_percent": 0.25,
        "subsidy_type": "Section 12B Tax Deduction",
    },
    "CA": {
        "default_currency": "CAD",
        "carbon_factor": 0.12,
        "electricity_rate": 0.13,
        "subsidy_percent": 0.25,
        "subsidy_type": "Canada Greener Homes Grant",
    },
    "KR": {
        "default_currency": "KRW",
        "carbon_factor": 0.46,
        "electricity_rate": 120.0,
        "subsidy_percent": 0.30,
        "subsidy_type": "Renewable Portfolio Standard (RPS)",
    },
    "MX": {
        "default_currency": "MXN",
        "carbon_factor": 0.43,
        "electricity_rate": 1.50,
        "subsidy_percent": 0.00,
        "subsidy_type": "Net Metering",
    },
    "AE": {
        "default_currency": "AED",
        "carbon_factor": 0.42,
        "electricity_rate": 0.29,
        "subsidy_percent": 0.00,
        "subsidy_type": "Shams Dubai Net Metering",
    },
    "ES": {
        "default_currency": "EUR",
        "carbon_factor": 0.22,
        "electricity_rate": 0.25,
        "subsidy_percent": 0.40,
        "subsidy_type": "EU NextGen Funds + Self-Consumption Bonus",
    },
}
