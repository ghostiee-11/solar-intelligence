"""Data ingestion module for solar radiation datasets.

Supports NASA POWER API, local NetCDF/HDF5/Zarr files, and geocoding.
All data is returned as xarray Datasets for downstream analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import param
import requests
import xarray as xr

from solar_intelligence.config import (
    CACHE_DIR,
    CACHE_TTL_DAYS,
    DEFAULT_END_YEAR,
    DEFAULT_START_YEAR,
    ERA5_CDS_URL,
    ERA5_DATASET_NAME,
    ERA5_SOLAR_VARIABLES,
    ERA5_VAR_MAP,
    NASA_POWER_BASE_URL,
    NASA_POWER_SOLAR_PARAMS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def geocode_location(city_name: str) -> tuple[float, float]:
    """Convert a city name to (latitude, longitude) using geopy.

    Parameters
    ----------
    city_name : str
        City name, e.g. "New Delhi", "San Francisco, CA".

    Returns
    -------
    tuple[float, float]
        (latitude, longitude) rounded to 4 decimal places.

    Raises
    ------
    ValueError
        If the city cannot be geocoded.
    """
    if not isinstance(city_name, str) or not city_name.strip():
        raise ValueError("city_name must be a non-empty string")

    from geopy.exc import GeocoderServiceError
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="solar-intelligence-platform")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(city_name, timeout=10)
            break
        except (GeocoderServiceError, Exception) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    "Nominatim retry %d/%d after %ds: %s",
                    attempt + 1, max_retries, wait, e,
                )
                time.sleep(wait)
            else:
                raise ConnectionError(
                    f"Geocoding service unreachable after {max_retries} attempts: {e}"
                ) from e
    if location is None:
        raise ValueError(f"Could not geocode location: '{city_name}'")
    return round(location.latitude, 4), round(location.longitude, 4)


# ---------------------------------------------------------------------------
# Cache Helpers
# ---------------------------------------------------------------------------

def _cache_key(lat: float, lon: float, start: str, end: str, temporal: str) -> str:
    """Generate a deterministic cache filename."""
    raw = f"{lat:.4f}_{lon:.4f}_{start}_{end}_{temporal}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"nasa_power_{temporal}_{digest}.nc"


def _cache_is_valid(cache_path: Path, ttl_days: int = CACHE_TTL_DAYS) -> bool:
    """Check if a cached file exists and is within TTL."""
    if not cache_path.exists():
        return False
    age_days = (time.time() - cache_path.stat().st_mtime) / 86400
    return age_days < ttl_days


# ---------------------------------------------------------------------------
# NASA POWER API Client
# ---------------------------------------------------------------------------

class NASAPowerClient(param.Parameterized):
    """Client for fetching solar radiation data from NASA POWER API.

    NASA POWER provides global solar and meteorological data derived from
    satellite observations and reanalysis, at 1°×1° spatial resolution,
    from 1981 to near real-time.

    Parameters
    ----------
    cache_dir : Path
        Directory for caching API responses as NetCDF files.
    parameters : list[str]
        NASA POWER parameter codes to fetch.
    community : str
        POWER community identifier (RE, SB, AG).
    """

    cache_dir = param.Path(default=CACHE_DIR, doc="Cache directory for API responses")
    parameters = param.List(
        default=NASA_POWER_SOLAR_PARAMS,
        item_type=str,
        doc="NASA POWER parameter codes",
    )
    community = param.String(default="RE", doc="POWER community (RE/SB/AG)")

    def fetch_daily(
        self,
        lat: float,
        lon: float,
        start: str | None = None,
        end: str | None = None,
    ) -> xr.Dataset:
        """Fetch daily solar data from NASA POWER.

        Parameters
        ----------
        lat, lon : float
            Location coordinates.
        start, end : str
            Date range in YYYYMMDD format.

        Returns
        -------
        xr.Dataset
            Dataset with time dimension and solar radiation variables.
        """
        if start is None:
            start = f"{DEFAULT_START_YEAR}0101"
        if end is None:
            end = f"{DEFAULT_END_YEAR}1231"
        return self._fetch("daily", lat, lon, start, end)

    def fetch_monthly(
        self,
        lat: float,
        lon: float,
        start: str | None = None,
        end: str | None = None,
    ) -> xr.Dataset:
        """Fetch monthly-averaged solar data from NASA POWER."""
        if start is None:
            start = f"{DEFAULT_START_YEAR}0101"
        if end is None:
            end = f"{DEFAULT_END_YEAR}1231"
        return self._fetch("monthly", lat, lon, start, end)

    def fetch_hourly(
        self,
        lat: float,
        lon: float,
        start: str | None = None,
        end: str | None = None,
    ) -> xr.Dataset:
        """Fetch hourly solar data from NASA POWER.

        Note: Hourly data is limited to shorter date ranges (~1 year max).
        """
        if start is None:
            start = f"{DEFAULT_END_YEAR}0101"
        if end is None:
            end = f"{DEFAULT_END_YEAR}0107"
        return self._fetch("hourly", lat, lon, start, end)

    def _fetch(
        self,
        temporal: str,
        lat: float,
        lon: float,
        start: str,
        end: str,
    ) -> xr.Dataset:
        """Internal method to fetch and cache NASA POWER data."""
        cache_file = Path(self.cache_dir) / _cache_key(lat, lon, start, end, temporal)

        if _cache_is_valid(cache_file):
            logger.info("Loading cached data: %s", cache_file.name)
            return xr.open_dataset(cache_file)

        logger.info(
            "Fetching NASA POWER %s data: lat=%.4f, lon=%.4f, %s to %s",
            temporal, lat, lon, start, end,
        )

        url = f"{NASA_POWER_BASE_URL}/{temporal}/point"
        params_str = ",".join(self.parameters)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url,
                    params={
                        "parameters": params_str,
                        "community": self.community,
                        "longitude": lon,
                        "latitude": lat,
                        "start": start,
                        "end": end,
                        "format": "JSON",
                    },
                    timeout=60,
                )
                response.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "NASA POWER retry %d/%d after %ds: %s",
                        attempt + 1, max_retries, wait, e,
                    )
                    time.sleep(wait)
                else:
                    raise ConnectionError(
                        f"NASA POWER API unreachable after {max_retries} attempts: {e}"
                    ) from e

        data = response.json()
        ds = self._parse_response(data, temporal, lat, lon)

        # Cache as NetCDF
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(cache_file)
        logger.info("Cached to: %s", cache_file.name)

        return ds

    def _parse_response(
        self,
        data: dict[str, Any],
        temporal: str,
        lat: float,
        lon: float,
    ) -> xr.Dataset:
        """Parse NASA POWER JSON response into xr.Dataset."""
        properties = data.get("properties", {})
        parameter_data = properties.get("parameter", {})

        if not parameter_data:
            raise ValueError(
                f"No data returned from NASA POWER API. "
                f"Response keys: {list(data.keys())}"
            )

        # Build DataFrame from parameter records
        records: dict[str, dict[str, float]] = {}
        for param_name, values in parameter_data.items():
            for date_key, value in values.items():
                if date_key not in records:
                    records[date_key] = {}
                # NASA POWER uses -999.0 for missing data
                records[date_key][param_name] = value if value != -999.0 else np.nan

        df = pd.DataFrame.from_dict(records, orient="index")
        df.index.name = "date"

        # Parse dates based on temporal resolution
        if temporal == "daily":
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
        elif temporal == "monthly":
            # Monthly keys are YYYYMM or YEAR13 (annual average)
            valid_idx = ~df.index.str.endswith("13")
            df = df[valid_idx]
            df.index = pd.to_datetime(df.index, format="%Y%m")
        elif temporal == "hourly":
            df.index = pd.to_datetime(df.index, format="%Y%m%d%H")

        df = df.sort_index()
        df = df.rename_axis("time")

        # Convert to xarray Dataset
        ds = df.to_xarray()
        ds = ds.assign_coords(latitude=lat, longitude=lon)

        # Add metadata attributes
        ds.attrs["source"] = "NASA POWER API"
        ds.attrs["temporal_resolution"] = temporal
        ds.attrs["latitude"] = lat
        ds.attrs["longitude"] = lon
        ds.attrs["fetched_at"] = datetime.now().isoformat()

        # Variable-level attributes
        var_attrs = {
            "ALLSKY_SFC_SW_DWN": {"long_name": "GHI (All Sky)", "units": "kWh/m²/day"},
            "CLRSKY_SFC_SW_DWN": {"long_name": "GHI (Clear Sky)", "units": "kWh/m²/day"},
            "ALLSKY_SFC_SW_DNI": {"long_name": "DNI (All Sky)", "units": "kWh/m²/day"},
            "ALLSKY_SFC_SW_DIFF": {"long_name": "DHI (All Sky)", "units": "kWh/m²/day"},
            "ALLSKY_KT": {"long_name": "Clearness Index", "units": "dimensionless"},
            "T2M": {"long_name": "Temperature at 2m", "units": "°C"},
            "T2M_MAX": {"long_name": "Max Temperature at 2m", "units": "°C"},
            "T2M_MIN": {"long_name": "Min Temperature at 2m", "units": "°C"},
            "WS2M": {"long_name": "Wind Speed at 2m", "units": "m/s"},
            "RH2M": {"long_name": "Relative Humidity at 2m", "units": "%"},
        }
        for var_name, attrs in var_attrs.items():
            if var_name in ds:
                ds[var_name].attrs.update(attrs)

        return ds


# ---------------------------------------------------------------------------
# Unified Data Loader
# ---------------------------------------------------------------------------

class DataLoader(param.Parameterized):
    """Unified loader for solar radiation datasets.

    Supports NASA POWER API, local NetCDF/HDF5/Zarr files,
    and provides spatial/temporal slicing utilities.
    """

    api_client = param.ClassSelector(
        class_=NASAPowerClient,
        default=None,
        allow_None=True,
        doc="NASA POWER API client instance",
    )

    def __init__(self, **params):
        super().__init__(**params)
        if self.api_client is None:
            self.api_client = NASAPowerClient()

    def load_from_api(
        self,
        lat: float,
        lon: float,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
        temporal: str = "daily",
    ) -> xr.Dataset:
        """Load solar data from NASA POWER API.

        Parameters
        ----------
        lat, lon : float
            Location coordinates.
        start_year, end_year : int
            Year range for data retrieval.
        temporal : str
            Resolution: "daily", "monthly", or "hourly".

        Returns
        -------
        xr.Dataset
        """
        start = f"{start_year}0101"
        end = f"{end_year}1231"

        if temporal == "daily":
            return self.api_client.fetch_daily(lat, lon, start, end)
        elif temporal == "monthly":
            return self.api_client.fetch_monthly(lat, lon, start, end)
        elif temporal == "hourly":
            return self.api_client.fetch_hourly(lat, lon, start, end)
        else:
            raise ValueError(f"Unknown temporal resolution: {temporal}")

    def load_for_location(
        self,
        city: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
    ) -> xr.Dataset:
        """Load solar data for a city name or coordinates.

        Provide either `city` or both `lat`/`lon`.
        """
        if city is not None:
            lat, lon = geocode_location(city)
            logger.info("Geocoded '%s' → (%.4f, %.4f)", city, lat, lon)
        elif lat is None or lon is None:
            raise ValueError("Provide either 'city' or both 'lat' and 'lon'.")

        ds = self.load_from_api(lat, lon, start_year, end_year)
        return ds

    @staticmethod
    def load_netcdf(path: str | Path) -> xr.Dataset:
        """Load a local NetCDF file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"NetCDF file not found: {path}")
        return xr.open_dataset(path)

    @staticmethod
    def load_zarr(path: str | Path) -> xr.Dataset:
        """Load a local Zarr store."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Zarr store not found: {path}")
        return xr.open_zarr(path)

    @staticmethod
    def load_hdf5(path: str | Path) -> xr.Dataset:
        """Load a local HDF5 file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        return xr.open_dataset(path, engine="h5netcdf")

    @staticmethod
    def slice_time(
        ds: xr.Dataset,
        start: str | None = None,
        end: str | None = None,
    ) -> xr.Dataset:
        """Slice dataset along time dimension.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset with 'time' dimension.
        start, end : str, optional
            ISO date strings for slicing.
        """
        if "time" not in ds.dims:
            raise ValueError("Dataset has no 'time' dimension.")
        return ds.sel(time=slice(start, end))

    @staticmethod
    def slice_location(
        ds: xr.Dataset,
        lat: float,
        lon: float,
        method: str = "nearest",
    ) -> xr.Dataset:
        """Select nearest grid point to given coordinates.

        Works with gridded datasets that have lat/lon dimensions.
        """
        sel_kwargs: dict[str, Any] = {}
        for lat_name in ("lat", "latitude"):
            if lat_name in ds.dims or lat_name in ds.coords:
                sel_kwargs[lat_name] = lat
                break
        for lon_name in ("lon", "longitude"):
            if lon_name in ds.dims or lon_name in ds.coords:
                sel_kwargs[lon_name] = lon
                break

        if not sel_kwargs:
            logger.warning("No lat/lon dimensions found — returning dataset as-is.")
            return ds

        return ds.sel(**sel_kwargs, method=method)


# ---------------------------------------------------------------------------
# Synthetic Data Generator (for testing & demos)
# ---------------------------------------------------------------------------

def generate_synthetic_solar_data(
    lat: float = 0.0,
    lon: float = 0.0,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> xr.Dataset:
    """Generate realistic synthetic solar data for testing.

    Uses latitude-dependent irradiance model with seasonal variation,
    cloud effects, and temperature correlation.

    Parameters
    ----------
    lat, lon : float
        Location coordinates.
    start_year, end_year : int
        Date range.

    Returns
    -------
    xr.Dataset
        Synthetic dataset matching NASA POWER variable structure.
    """
    rng = np.random.default_rng(seed=abs(int(lat * 100 + lon * 10)))

    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    n_days = len(dates)

    # Day of year for seasonal cycle
    doy = dates.dayofyear.values
    lat_rad = np.radians(lat)

    # Solar declination angle (approximate)
    declination = 23.45 * np.sin(np.radians((360 / 365) * (doy - 81)))
    decl_rad = np.radians(declination)

    # Day length factor
    cos_hour_angle = -np.tan(lat_rad) * np.tan(decl_rad)
    cos_hour_angle = np.clip(cos_hour_angle, -1, 1)
    day_length_hours = (2 / 15) * np.degrees(np.arccos(cos_hour_angle))

    # Extra-terrestrial irradiance on horizontal surface
    solar_constant = 1361  # W/m²
    cos_zenith_noon = np.sin(lat_rad) * np.sin(decl_rad) + \
        np.cos(lat_rad) * np.cos(decl_rad)
    cos_zenith_noon = np.clip(cos_zenith_noon, 0, 1)

    # Clear sky GHI (kWh/m²/day) — simplified model
    clearsky_ghi = (solar_constant * cos_zenith_noon * day_length_hours *
                    0.75 / 1000)  # atmospheric transmittance ~0.75
    clearsky_ghi = np.clip(clearsky_ghi, 0, 12)

    # Cloud / clearness factor with seasonal and random variation
    cloud_base = 0.55 + 0.15 * np.cos(np.radians((360 / 365) * (doy - 172)))
    cloud_noise = rng.normal(0, 0.08, n_days)
    clearness = np.clip(cloud_base + cloud_noise, 0.15, 0.95)

    # Actual GHI
    ghi = clearsky_ghi * clearness

    # DNI / DHI decomposition (simplified Erbs model)
    kt = clearness
    kd = np.where(
        kt <= 0.22, 1.0 - 0.09 * kt,
        np.where(
            kt <= 0.80,
            0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
            0.165,
        ),
    )
    dhi = ghi * kd
    dni = (ghi - dhi) / np.clip(cos_zenith_noon, 0.05, 1.0)
    dni = np.clip(dni, 0, 15)

    # Temperature model (seasonal + diurnal mean + noise)
    temp_annual_mean = 15 + 20 * np.cos(np.radians(lat))  # latitude-dependent
    temp_seasonal = 10 * np.sin(np.radians((360 / 365) * (doy - 81)))
    if lat < 0:
        temp_seasonal = -temp_seasonal
    temp_noise = rng.normal(0, 2, n_days)
    temperature = temp_annual_mean + temp_seasonal + temp_noise

    # Wind speed (random with slight seasonal variation)
    wind = 3.0 + 1.5 * np.sin(np.radians((360 / 365) * (doy - 45))) + \
        rng.normal(0, 0.8, n_days)
    wind = np.clip(wind, 0.5, 15)

    # Relative humidity
    humidity = 55 + 15 * np.cos(np.radians((360 / 365) * (doy - 200))) + \
        rng.normal(0, 5, n_days)
    humidity = np.clip(humidity, 15, 95)

    ds = xr.Dataset(
        {
            "ALLSKY_SFC_SW_DWN": ("time", ghi.astype(np.float32)),
            "CLRSKY_SFC_SW_DWN": ("time", clearsky_ghi.astype(np.float32)),
            "ALLSKY_SFC_SW_DNI": ("time", dni.astype(np.float32)),
            "ALLSKY_SFC_SW_DIFF": ("time", dhi.astype(np.float32)),
            "ALLSKY_KT": ("time", kt.astype(np.float32)),
            "T2M": ("time", temperature.astype(np.float32)),
            "T2M_MAX": ("time", (temperature + rng.uniform(3, 7, n_days)).astype(np.float32)),
            "T2M_MIN": ("time", (temperature - rng.uniform(3, 7, n_days)).astype(np.float32)),
            "WS2M": ("time", wind.astype(np.float32)),
            "RH2M": ("time", humidity.astype(np.float32)),
        },
        coords={
            "time": dates,
            "latitude": lat,
            "longitude": lon,
        },
        attrs={
            "source": "Synthetic (solar-intelligence)",
            "temporal_resolution": "daily",
            "latitude": lat,
            "longitude": lon,
            "description": "Realistic synthetic solar data for testing and demos",
        },
    )

    # Add variable attributes
    var_attrs = {
        "ALLSKY_SFC_SW_DWN": {"long_name": "GHI (All Sky)", "units": "kWh/m²/day"},
        "CLRSKY_SFC_SW_DWN": {"long_name": "GHI (Clear Sky)", "units": "kWh/m²/day"},
        "ALLSKY_SFC_SW_DNI": {"long_name": "DNI (All Sky)", "units": "kWh/m²/day"},
        "ALLSKY_SFC_SW_DIFF": {"long_name": "DHI (All Sky)", "units": "kWh/m²/day"},
        "ALLSKY_KT": {"long_name": "Clearness Index", "units": "dimensionless"},
        "T2M": {"long_name": "Temperature at 2m", "units": "°C"},
        "T2M_MAX": {"long_name": "Max Temperature at 2m", "units": "°C"},
        "T2M_MIN": {"long_name": "Min Temperature at 2m", "units": "°C"},
        "WS2M": {"long_name": "Wind Speed at 2m", "units": "m/s"},
        "RH2M": {"long_name": "Relative Humidity at 2m", "units": "%"},
    }
    for var_name, attrs in var_attrs.items():
        if var_name in ds:
            ds[var_name].attrs.update(attrs)

    return ds


# ---------------------------------------------------------------------------
# Global Solar Grid Generator (for Datashader maps)
# ---------------------------------------------------------------------------

def generate_global_solar_grid(
    resolution: float = 1.0,
    lat_range: tuple[float, float] = (-60, 60),
    lon_range: tuple[float, float] = (-180, 180),
) -> xr.Dataset:
    """Generate a global solar irradiance grid for large-scale Datashader maps.

    Uses a physics-based latitude model to produce a realistic global GHI
    distribution with millions of grid points for Datashader rendering.

    Parameters
    ----------
    resolution : float
        Grid resolution in degrees (0.25 -> ~1M points, 0.1 -> ~6.5M points).
    lat_range, lon_range : tuple
        Bounding box for the grid.

    Returns
    -------
    xr.Dataset
        Gridded dataset with lat/lon dimensions and GHI values.
    """
    rng = np.random.default_rng(seed=42)

    lats = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lons = np.arange(lon_range[0], lon_range[1] + resolution, resolution)

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # Physics-based GHI model: peaks at equator, drops at poles
    lat_rad = np.radians(lat_grid)
    base_ghi = 7.0 * np.cos(lat_rad) ** 0.8

    # Longitude-dependent "desert boost" (Sahara, Arabian, Australian deserts)
    desert_boost = np.zeros_like(lon_grid)
    # Sahara/Arabian (lat 15-30N, lon -15 to 60)
    mask_sahara = (lat_grid > 10) & (lat_grid < 35) & (lon_grid > -15) & (lon_grid < 60)
    desert_boost[mask_sahara] = 0.8
    # Australian (lat -30 to -15, lon 115 to 150)
    mask_australia = (lat_grid > -35) & (lat_grid < -15) & (lon_grid > 115) & (lon_grid < 150)
    desert_boost[mask_australia] = 0.6
    # Southwest US (lat 25-40N, lon -120 to -100)
    mask_sw_us = (lat_grid > 25) & (lat_grid < 40) & (lon_grid > -120) & (lon_grid < -100)
    desert_boost[mask_sw_us] = 0.5

    ghi = base_ghi + desert_boost

    # Add spatial noise
    ghi += rng.normal(0, 0.15, ghi.shape)
    ghi = np.clip(ghi, 0.5, 9.0).astype(np.float32)

    ds = xr.Dataset(
        {"GHI": (["lat", "lon"], ghi)},
        coords={"lat": lats, "lon": lons},
        attrs={
            "source": "Synthetic global grid (solar-intelligence)",
            "units": "kWh/m^2/day",
            "description": "Global solar irradiance grid for Datashader visualization",
        },
    )
    return ds


# ---------------------------------------------------------------------------
# ERA5 CDS API Client
# ---------------------------------------------------------------------------

class ERA5Client(param.Parameterized):
    """Client for fetching solar radiation data from Copernicus ERA5 via CDS API.

    ERA5 provides global hourly reanalysis data at 0.25 degree resolution
    from 1940 to near real-time. Requires a free CDS account and API key.

    Setup:
        1. Register at https://cds.climate.copernicus.eu
        2. Create ~/.cdsapirc with your UID:API-KEY
        3. pip install cdsapi

    Parameters
    ----------
    cache_dir : Path
        Directory for caching downloaded ERA5 NetCDF files.
    variables : list[str]
        ERA5 variable names to fetch.
    dataset : str
        CDS dataset identifier.
    """

    cache_dir = param.Path(default=CACHE_DIR, doc="Cache directory for ERA5 data")
    variables = param.List(
        default=ERA5_SOLAR_VARIABLES,
        item_type=str,
        doc="ERA5 variable names to fetch",
    )
    dataset = param.String(default=ERA5_DATASET_NAME)

    def _era5_cache_key(
        self, lat: float, lon: float, start: str, end: str,
    ) -> str:
        """Generate a cache filename for ERA5 data."""
        raw = f"era5_{lat:.4f}_{lon:.4f}_{start}_{end}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
        return f"era5_{digest}.nc"

    def _check_cdsapi(self):
        """Check that cdsapi is installed and configured."""
        try:
            import cdsapi  # noqa: F401
        except ImportError:
            raise ImportError(
                "cdsapi package not installed. Run: pip install cdsapi\n"
                "Then create ~/.cdsapirc with your CDS API key.\n"
                "Register free at https://cds.climate.copernicus.eu"
            )

    def fetch_daily(
        self,
        lat: float,
        lon: float,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
        area_margin: float = 0.5,
    ) -> xr.Dataset:
        """Fetch daily-aggregated ERA5 solar data from CDS API.

        Downloads hourly data and aggregates to daily resolution
        to match NASA POWER format.

        Parameters
        ----------
        lat, lon : float
            Location coordinates.
        start_year, end_year : int
            Year range.
        area_margin : float
            Margin in degrees around the point for area selection.

        Returns
        -------
        xr.Dataset
            Daily dataset with standardized variable names matching
            NASA POWER format (GHI in kWh/m2/day, T in Celsius).
        """
        self._check_cdsapi()
        import cdsapi

        start = f"{start_year}0101"
        end = f"{end_year}1231"
        cache_file = Path(self.cache_dir) / self._era5_cache_key(lat, lon, start, end)

        if _cache_is_valid(cache_file):
            logger.info("Loading cached ERA5 data: %s", cache_file.name)
            return xr.open_dataset(cache_file)

        logger.info(
            "Fetching ERA5 data: lat=%.4f, lon=%.4f, %d-%d",
            lat, lon, start_year, end_year,
        )

        # Area bounding box: [North, West, South, East]
        area = [
            lat + area_margin,
            lon - area_margin,
            lat - area_margin,
            lon + area_margin,
        ]

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        client = cdsapi.Client()

        # Fetch in monthly chunks to avoid CDS cost limits
        monthly_files = []
        days = [str(d).zfill(2) for d in range(1, 32)]
        # Sample 4 times per day (enough for daily aggregation)
        hours = ["00:00", "06:00", "12:00", "18:00"]

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                month_str = str(month).zfill(2)
                raw_file = Path(self.cache_dir) / f"era5_raw_{year}{month_str}_{hashlib.sha256(f'{lat}{lon}'.encode()).hexdigest()[:8]}.nc"

                if raw_file.exists():
                    monthly_files.append(raw_file)
                    continue

                logger.info("Fetching ERA5: %d-%s...", year, month_str)
                try:
                    client.retrieve(
                        self.dataset,
                        {
                            "product_type": ["reanalysis"],
                            "variable": self.variables,
                            "year": [str(year)],
                            "month": [month_str],
                            "day": days,
                            "time": hours,
                            "area": area,
                            "data_format": "netcdf",
                        },
                        str(raw_file),
                    )
                    monthly_files.append(raw_file)
                except Exception as e:
                    logger.warning("Failed to fetch %d-%s: %s", year, month_str, e)

        if not monthly_files:
            raise RuntimeError("No ERA5 data could be fetched")

        # CDS may return zip files -- extract NetCDF from them
        import zipfile

        nc_files = []
        for f in monthly_files:
            if zipfile.is_zipfile(f):
                extract_dir = f.parent / f"{f.stem}_extracted"
                extract_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(f, "r") as zf:
                    zf.extractall(extract_dir)
                # Find the .nc file inside
                nc_found = list(extract_dir.glob("*.nc"))
                if nc_found:
                    nc_files.append(nc_found[0])
                else:
                    logger.warning("No .nc file found in zip: %s", f)
            else:
                nc_files.append(f)

        if not nc_files:
            raise RuntimeError("No valid NetCDF files extracted from ERA5 downloads")

        # Merge all monthly files
        if len(nc_files) == 1:
            merged_ds = xr.open_dataset(nc_files[0])
        else:
            merged_ds = xr.open_mfdataset(nc_files, combine="by_coords")

        # Save merged raw to temp
        raw_merged = Path(self.cache_dir) / f"era5_merged_{hashlib.sha256(start.encode()).hexdigest()[:8]}.nc"
        merged_ds.to_netcdf(raw_merged)
        merged_ds.close()

        # Parse and convert to standard format
        ds = self._parse_era5(raw_merged, lat, lon)

        # Cache the processed result
        ds.to_netcdf(cache_file)
        logger.info("ERA5 data cached to: %s", cache_file.name)

        # Clean up raw files and extracted directories
        import shutil

        for f in monthly_files:
            if f.exists():
                f.unlink()
            extract_dir = f.parent / f"{f.stem}_extracted"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
        if raw_merged.exists():
            raw_merged.unlink()

        return ds

    def _parse_era5(
        self, path: Path, lat: float, lon: float,
    ) -> xr.Dataset:
        """Parse raw ERA5 NetCDF and convert to standard daily format.

        Handles:
        - Selecting nearest grid point to requested lat/lon
        - Aggregating hourly -> daily
        - Unit conversions: J/m2 -> kWh/m2/day, K -> C
        - Wind speed from u/v components
        - Variable renaming to NASA POWER standard names
        """
        ds = xr.open_dataset(path)

        # Select nearest grid point
        for lat_name in ("latitude", "lat"):
            if lat_name in ds.dims:
                ds = ds.sel(**{lat_name: lat}, method="nearest")
                break
        for lon_name in ("longitude", "lon"):
            if lon_name in ds.dims:
                ds = ds.sel(**{lon_name: lon}, method="nearest")
                break

        # Rename time if needed
        for time_name in ("valid_time", "time"):
            if time_name in ds.dims:
                if time_name != "time":
                    ds = ds.rename({time_name: "time"})
                break

        result_vars = {}

        # --- Solar radiation: J/m2 cumulative per hour -> kWh/m2/day ---
        # Determine sampling rate to scale sub-sampled data
        if "time" in ds.dims and len(ds.time) > 48:
            # Count unique hours in a day to determine sampling rate
            hours_per_day = len(set(ds.time.dt.hour.values))
            hours_per_day = min(hours_per_day, 24)
            scale_factor = 24 / max(hours_per_day, 1)
        else:
            scale_factor = 1.0

        if "ssrd" in ds:
            # ssrd is accumulated J/m2 per hour; sum over day then convert
            daily_ssrd = ds["ssrd"].resample(time="1D").sum()
            # Scale up if sub-sampled (e.g., 4 samples/day -> scale by 6)
            ghi = daily_ssrd * scale_factor / 3_600_000  # J/m2 -> kWh/m2
            ghi = ghi.clip(min=0)
            result_vars["ALLSKY_SFC_SW_DWN"] = ghi

        if "fdir" in ds:
            daily_fdir = ds["fdir"].resample(time="1D").sum()
            dni = daily_fdir * scale_factor / 3_600_000
            dni = dni.clip(min=0)
            result_vars["ALLSKY_SFC_SW_DNI"] = dni

        # Compute DHI = GHI - DNI * cos(zenith) ≈ GHI - DNI (simplified)
        if "ALLSKY_SFC_SW_DWN" in result_vars and "ALLSKY_SFC_SW_DNI" in result_vars:
            dhi = result_vars["ALLSKY_SFC_SW_DWN"] - result_vars["ALLSKY_SFC_SW_DNI"] * 0.6
            result_vars["ALLSKY_SFC_SW_DIFF"] = dhi.clip(min=0)

        # --- Temperature: K -> C ---
        if "t2m" in ds:
            t2m_daily = ds["t2m"].resample(time="1D").mean()
            if float(t2m_daily.mean()) > 200:
                t2m_daily = t2m_daily - 273.15
            result_vars["T2M"] = t2m_daily

            t2m_max = ds["t2m"].resample(time="1D").max()
            if float(t2m_max.mean()) > 200:
                t2m_max = t2m_max - 273.15
            result_vars["T2M_MAX"] = t2m_max

            t2m_min = ds["t2m"].resample(time="1D").min()
            if float(t2m_min.mean()) > 200:
                t2m_min = t2m_min - 273.15
            result_vars["T2M_MIN"] = t2m_min

        # --- Wind speed: combine u10 + v10 ---
        if "u10" in ds and "v10" in ds:
            u10_daily = ds["u10"].resample(time="1D").mean()
            v10_daily = ds["v10"].resample(time="1D").mean()
            ws = np.sqrt(u10_daily**2 + v10_daily**2)
            result_vars["WS2M"] = ws

        # --- Dewpoint -> Relative Humidity (approximation) ---
        if "d2m" in ds and "t2m" in ds:
            d2m_daily = ds["d2m"].resample(time="1D").mean()
            t2m_daily_rh = ds["t2m"].resample(time="1D").mean()
            if float(d2m_daily.mean()) > 200:
                d2m_daily = d2m_daily - 273.15
                t2m_daily_rh = t2m_daily_rh - 273.15
            # Magnus formula approximation for RH
            rh = 100 * np.exp((17.625 * d2m_daily) / (243.04 + d2m_daily)) / \
                np.exp((17.625 * t2m_daily_rh) / (243.04 + t2m_daily_rh))
            result_vars["RH2M"] = rh.clip(min=0, max=100)

        # --- Cloud cover ---
        if "tcc" in ds:
            tcc_daily = ds["tcc"].resample(time="1D").mean()
            result_vars["CLOUD_COVER"] = tcc_daily

        # --- Clearness index (GHI / theoretical clearsky) ---
        if "ALLSKY_SFC_SW_DWN" in result_vars:
            ghi_vals = result_vars["ALLSKY_SFC_SW_DWN"]
            # Simple clearsky estimate for KT
            doy = ghi_vals.time.dt.dayofyear
            lat_rad = np.radians(lat)
            decl = 23.45 * np.sin(np.radians((360 / 365) * (doy - 81)))
            cos_z = np.sin(lat_rad) * np.sin(np.radians(decl)) + \
                np.cos(lat_rad) * np.cos(np.radians(decl))
            cos_z = cos_z.clip(min=0.05)
            cos_hour = (-np.tan(lat_rad) * np.tan(np.radians(decl))).clip(-1, 1)
            day_length = (2 / 15) * np.degrees(np.arccos(cos_hour))
            clearsky = (1361 * cos_z * day_length * 0.75 / 1000).clip(min=0.1)
            kt = (ghi_vals / clearsky).clip(min=0, max=1)
            result_vars["ALLSKY_KT"] = kt
            result_vars["CLRSKY_SFC_SW_DWN"] = clearsky

        ds_out = xr.Dataset(result_vars)
        ds_out = ds_out.assign_coords(latitude=lat, longitude=lon)
        ds_out.attrs["source"] = "ERA5 (Copernicus Climate Data Store)"
        ds_out.attrs["temporal_resolution"] = "daily"
        ds_out.attrs["latitude"] = lat
        ds_out.attrs["longitude"] = lon
        ds_out.attrs["fetched_at"] = datetime.now().isoformat()

        # Variable attributes
        var_attrs = {
            "ALLSKY_SFC_SW_DWN": {"long_name": "GHI (ERA5)", "units": "kWh/m²/day"},
            "CLRSKY_SFC_SW_DWN": {"long_name": "Clear Sky GHI (estimated)", "units": "kWh/m²/day"},
            "ALLSKY_SFC_SW_DNI": {"long_name": "DNI (ERA5)", "units": "kWh/m²/day"},
            "ALLSKY_SFC_SW_DIFF": {"long_name": "DHI (ERA5, derived)", "units": "kWh/m²/day"},
            "ALLSKY_KT": {"long_name": "Clearness Index", "units": "dimensionless"},
            "T2M": {"long_name": "Temperature at 2m", "units": "°C"},
            "T2M_MAX": {"long_name": "Max Temperature at 2m", "units": "°C"},
            "T2M_MIN": {"long_name": "Min Temperature at 2m", "units": "°C"},
            "WS2M": {"long_name": "Wind Speed at 10m", "units": "m/s"},
            "RH2M": {"long_name": "Relative Humidity at 2m", "units": "%"},
            "CLOUD_COVER": {"long_name": "Total Cloud Cover", "units": "fraction"},
        }
        for var_name, attrs in var_attrs.items():
            if var_name in ds_out:
                ds_out[var_name].attrs.update(attrs)

        return ds_out

    def fetch_monthly(
        self,
        lat: float,
        lon: float,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
    ) -> xr.Dataset:
        """Fetch ERA5 data and aggregate to monthly resolution.

        Parameters
        ----------
        lat, lon : float
            Location coordinates.
        start_year, end_year : int
            Year range.

        Returns
        -------
        xr.Dataset
            Monthly-averaged dataset.
        """
        daily = self.fetch_daily(lat, lon, start_year, end_year)
        monthly_vars = {}
        for var in daily.data_vars:
            monthly_vars[var] = daily[var].resample(time="1ME").mean()

        ds_monthly = xr.Dataset(monthly_vars)
        ds_monthly.attrs = daily.attrs.copy()
        ds_monthly.attrs["temporal_resolution"] = "monthly"
        return ds_monthly


# ---------------------------------------------------------------------------
# Dual-Source Data Loader (NASA POWER + ERA5)
# ---------------------------------------------------------------------------

class DualSourceLoader(param.Parameterized):
    """Fetch solar data from both NASA POWER and ERA5 for cross-validation.

    Provides a unified interface to load data from both sources,
    align them on a common time axis, and compute comparison metrics.

    Parameters
    ----------
    use_era5 : bool
        Whether to fetch ERA5 data (requires CDS API setup).
    use_nasa : bool
        Whether to fetch NASA POWER data.
    """

    use_era5 = param.Boolean(default=True, doc="Fetch ERA5 data")
    use_nasa = param.Boolean(default=True, doc="Fetch NASA POWER data")

    _nasa_client = param.Parameter(default=None, precedence=-1)
    _era5_client = param.Parameter(default=None, precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        if self.use_nasa:
            self._nasa_client = NASAPowerClient()
        if self.use_era5:
            self._era5_client = ERA5Client()

    def fetch(
        self,
        lat: float,
        lon: float,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
    ) -> dict[str, xr.Dataset]:
        """Fetch data from all enabled sources.

        Parameters
        ----------
        lat, lon : float
            Location coordinates.
        start_year, end_year : int
            Year range.

        Returns
        -------
        dict[str, xr.Dataset]
            Keys are source names ("nasa_power", "era5"), values are datasets.
        """
        results = {}

        if self.use_nasa and self._nasa_client:
            try:
                start = f"{start_year}0101"
                end = f"{end_year}1231"
                results["nasa_power"] = self._nasa_client.fetch_daily(
                    lat, lon, start, end,
                )
                logger.info("NASA POWER data loaded: %d days", len(results["nasa_power"].time))
            except Exception as e:
                logger.error("NASA POWER fetch failed: %s", e)

        if self.use_era5 and self._era5_client:
            try:
                results["era5"] = self._era5_client.fetch_daily(
                    lat, lon, start_year, end_year,
                )
                logger.info("ERA5 data loaded: %d days", len(results["era5"].time))
            except Exception as e:
                logger.error("ERA5 fetch failed: %s", e)

        return results

    @staticmethod
    def align_datasets(
        datasets: dict[str, xr.Dataset],
        variable: str = "ALLSKY_SFC_SW_DWN",
    ) -> pd.DataFrame:
        """Align multiple source datasets on common time axis.

        Parameters
        ----------
        datasets : dict[str, xr.Dataset]
            Source name -> dataset mapping.
        variable : str
            Variable to extract and align.

        Returns
        -------
        pd.DataFrame
            DataFrame with time index and one column per source.
        """
        series = {}
        for name, ds in datasets.items():
            if variable in ds.data_vars:
                s = ds[variable].to_series()
                s.index = pd.to_datetime(s.index)
                series[name] = s

        if not series:
            return pd.DataFrame()

        df = pd.DataFrame(series)
        df.index.name = "time"
        return df

    @staticmethod
    def comparison_stats(
        datasets: dict[str, xr.Dataset],
        variable: str = "ALLSKY_SFC_SW_DWN",
    ) -> dict[str, Any]:
        """Compute comparison statistics between sources.

        Parameters
        ----------
        datasets : dict[str, xr.Dataset]
            Source name -> dataset mapping.
        variable : str
            Variable to compare.

        Returns
        -------
        dict
            Statistics including per-source means, correlation, RMSE, bias.
        """
        aligned = DualSourceLoader.align_datasets(datasets, variable)
        if aligned.empty or len(aligned.columns) < 2:
            return {"error": "Need at least 2 sources for comparison"}

        stats = {"variable": variable, "sources": {}}

        for col in aligned.columns:
            stats["sources"][col] = {
                "mean": float(aligned[col].mean()),
                "std": float(aligned[col].std()),
                "min": float(aligned[col].min()),
                "max": float(aligned[col].max()),
                "count": int(aligned[col].count()),
            }

        # Pairwise comparison between first two sources
        cols = list(aligned.columns)
        common = aligned[cols].dropna()
        if len(common) > 10:
            a, b = common[cols[0]], common[cols[1]]
            stats["comparison"] = {
                "source_a": cols[0],
                "source_b": cols[1],
                "correlation": float(a.corr(b)),
                "rmse": float(np.sqrt(((a - b) ** 2).mean())),
                "mae": float((a - b).abs().mean()),
                "bias": float((a - b).mean()),
                "bias_pct": float((a - b).mean() / a.mean() * 100),
                "common_days": len(common),
            }

        return stats


# ---------------------------------------------------------------------------
# ERA5 / SARAH-3 Dataset Loader (local files)
# ---------------------------------------------------------------------------

class ClimateDatasetLoader(param.Parameterized):
    """Load solar radiation data from ERA5 or SARAH-3 climate datasets.

    ERA5 (Copernicus Climate Data Store):
        - Global reanalysis dataset, 0.25 deg resolution, hourly
        - Variable: 'ssrd' (surface solar radiation downwards, J/m2)
        - Requires CDS API key and cdsapi package

    SARAH-3 (CM SAF):
        - Satellite-derived surface radiation, Europe/Africa, 0.05 deg resolution
        - Variable: 'SIS' (Surface Incoming Shortwave radiation, W/m2)
        - NetCDF format from CM SAF web interface

    Parameters
    ----------
    era5_var_map : dict
        Mapping of ERA5 variable names to standard names.
    sarah_var_map : dict
        Mapping of SARAH-3 variable names to standard names.
    """

    era5_var_map = param.Dict(
        default={
            "ssrd": "ALLSKY_SFC_SW_DWN",
            "fdir": "ALLSKY_SFC_SW_DNI",
            "t2m": "T2M",
            "u10": "WS2M",
        },
        doc="ERA5 variable name mapping",
    )

    sarah_var_map = param.Dict(
        default={
            "SIS": "ALLSKY_SFC_SW_DWN",
            "SID": "ALLSKY_SFC_SW_DNI",
            "SDU": "ALLSKY_SFC_SW_DIFF",
        },
        doc="SARAH-3 variable name mapping",
    )

    def load_era5(
        self,
        path: str | Path,
        lat: float | None = None,
        lon: float | None = None,
    ) -> xr.Dataset:
        """Load an ERA5 NetCDF file and convert to standard format.

        Handles unit conversions:
        - ssrd: J/m2 cumulative -> kWh/m2/day
        - t2m: Kelvin -> Celsius

        Parameters
        ----------
        path : str or Path
            Path to ERA5 NetCDF file.
        lat, lon : float, optional
            If provided, select nearest grid point.

        Returns
        -------
        xr.Dataset
            Standardized dataset matching NASA POWER variable structure.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ERA5 file not found: {path}")

        ds = xr.open_dataset(path)

        # Select location if specified
        if lat is not None and lon is not None:
            for lat_name in ("latitude", "lat"):
                if lat_name in ds.dims:
                    ds = ds.sel(**{lat_name: lat}, method="nearest")
                    break
            for lon_name in ("longitude", "lon"):
                if lon_name in ds.dims:
                    ds = ds.sel(**{lon_name: lon}, method="nearest")
                    break

        # Rename time dimension if needed
        for time_name in ("valid_time", "time"):
            if time_name in ds.dims:
                if time_name != "time":
                    ds = ds.rename({time_name: "time"})
                break

        # Convert variables
        result_vars = {}
        for era5_name, std_name in self.era5_var_map.items():
            if era5_name not in ds:
                continue
            data = ds[era5_name]

            if era5_name in ("ssrd", "fdir"):
                # J/m2 -> kWh/m2
                data = data / 3_600_000
                data = data.clip(min=0)

            elif era5_name == "t2m":
                # Kelvin -> Celsius
                if float(data.mean()) > 200:
                    data = data - 273.15

            result_vars[std_name] = data

        ds_out = xr.Dataset(result_vars)
        ds_out.attrs["source"] = "ERA5 (Copernicus Climate Data Store)"
        ds_out.attrs["original_file"] = str(path)

        return ds_out

    def load_sarah3(
        self,
        path: str | Path,
        lat: float | None = None,
        lon: float | None = None,
    ) -> xr.Dataset:
        """Load a SARAH-3 NetCDF file and convert to standard format.

        Handles unit conversions:
        - SIS/SID/SDU: W/m2 (daily mean) -> kWh/m2/day

        Parameters
        ----------
        path : str or Path
            Path to SARAH-3 NetCDF file.
        lat, lon : float, optional
            If provided, select nearest grid point.

        Returns
        -------
        xr.Dataset
            Standardized dataset matching NASA POWER variable structure.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SARAH-3 file not found: {path}")

        ds = xr.open_dataset(path)

        if lat is not None and lon is not None:
            for lat_name in ("lat", "latitude"):
                if lat_name in ds.dims:
                    ds = ds.sel(**{lat_name: lat}, method="nearest")
                    break
            for lon_name in ("lon", "longitude"):
                if lon_name in ds.dims:
                    ds = ds.sel(**{lon_name: lon}, method="nearest")
                    break

        result_vars = {}
        for sarah_name, std_name in self.sarah_var_map.items():
            if sarah_name not in ds:
                continue
            data = ds[sarah_name]

            # W/m2 daily mean -> kWh/m2/day = W/m2 * 24 / 1000
            data = data * 24 / 1000
            data = data.clip(min=0)

            result_vars[std_name] = data

        ds_out = xr.Dataset(result_vars)
        ds_out.attrs["source"] = "SARAH-3 (CM SAF)"
        ds_out.attrs["original_file"] = str(path)

        return ds_out


# ---------------------------------------------------------------------------
# Multi-Location Comparison Utility
# ---------------------------------------------------------------------------

def generate_multi_location_data(
    locations: dict[str, tuple[float, float]],
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> dict[str, xr.Dataset]:
    """Generate synthetic solar data for multiple locations.

    Parameters
    ----------
    locations : dict
        Mapping of location name -> (lat, lon).
    start_year, end_year : int
        Date range.

    Returns
    -------
    dict[str, xr.Dataset]
        Mapping of location name -> dataset.
    """
    datasets = {}
    for name, (lat, lon) in locations.items():
        datasets[name] = generate_synthetic_solar_data(lat, lon, start_year, end_year)
    return datasets
