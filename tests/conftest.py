"""Shared test fixtures for Solar Intelligence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from solar_intelligence.data_loader import generate_synthetic_solar_data


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    """Synthetic solar dataset for New Delhi (4 years)."""
    return generate_synthetic_solar_data(
        lat=28.6139, lon=77.2090, start_year=2020, end_year=2023,
    )


@pytest.fixture
def sample_dataset_london() -> xr.Dataset:
    """Synthetic solar dataset for London."""
    return generate_synthetic_solar_data(
        lat=51.5074, lon=-0.1278, start_year=2022, end_year=2023,
    )


@pytest.fixture
def sample_dataset_sydney() -> xr.Dataset:
    """Synthetic solar dataset for Sydney (Southern Hemisphere)."""
    return generate_synthetic_solar_data(
        lat=-33.8688, lon=151.2093, start_year=2022, end_year=2023,
    )


@pytest.fixture
def mock_nasa_power_response() -> dict:
    """Mock NASA POWER API JSON response."""
    dates = pd.date_range("2023-01-01", "2023-01-05", freq="D")
    parameter_data = {}

    for param_name in [
        "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI",
        "ALLSKY_SFC_SW_DIFF", "ALLSKY_KT", "T2M", "T2M_MAX", "T2M_MIN",
        "WS2M", "RH2M",
    ]:
        values = {}
        for d in dates:
            key = d.strftime("%Y%m%d")
            if "SW" in param_name or "KT" in param_name:
                values[key] = float(np.random.uniform(2, 7))
            elif "T2M" in param_name:
                values[key] = float(np.random.uniform(10, 35))
            elif "WS" in param_name:
                values[key] = float(np.random.uniform(1, 8))
            else:
                values[key] = float(np.random.uniform(30, 80))
        parameter_data[param_name] = values

    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [77.209, 28.6139, 216.0]},
        "properties": {
            "parameter": parameter_data,
        },
        "header": {
            "title": "NASA/POWER CERES/MERRA2",
        },
    }


@pytest.fixture
def tmp_netcdf(tmp_path, sample_dataset) -> str:
    """Write sample dataset to a temporary NetCDF file and return its path."""
    path = tmp_path / "test_solar.nc"
    sample_dataset.to_netcdf(path)
    return str(path)
