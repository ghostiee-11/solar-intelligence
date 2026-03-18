"""Tests for data_loader module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from solar_intelligence.data_loader import (
    DataLoader,
    NASAPowerClient,
    _cache_is_valid,
    _cache_key,
    generate_synthetic_solar_data,
    geocode_location,
)


# ---------------------------------------------------------------------------
# Geocoding Tests
# ---------------------------------------------------------------------------

class TestGeocode:
    """Tests for geocode_location function."""

    @patch("geopy.geocoders.Nominatim")
    def test_geocode_valid_city(self, mock_nominatim_cls):
        mock_geolocator = MagicMock()
        mock_nominatim_cls.return_value = mock_geolocator
        mock_location = MagicMock()
        mock_location.latitude = 28.613889
        mock_location.longitude = 77.209
        mock_geolocator.geocode.return_value = mock_location

        lat, lon = geocode_location("New Delhi")
        assert abs(lat - 28.6139) < 0.001
        assert abs(lon - 77.209) < 0.001

    @patch("geopy.geocoders.Nominatim")
    def test_geocode_invalid_city_raises(self, mock_nominatim_cls):
        mock_geolocator = MagicMock()
        mock_nominatim_cls.return_value = mock_geolocator
        mock_geolocator.geocode.return_value = None

        with pytest.raises(ValueError, match="Could not geocode"):
            geocode_location("Nonexistent City 12345")


# ---------------------------------------------------------------------------
# Cache Tests
# ---------------------------------------------------------------------------

class TestCache:
    """Tests for caching utilities."""

    def test_cache_key_deterministic(self):
        k1 = _cache_key(28.6, 77.2, "20230101", "20231231", "daily")
        k2 = _cache_key(28.6, 77.2, "20230101", "20231231", "daily")
        assert k1 == k2

    def test_cache_key_different_for_different_inputs(self):
        k1 = _cache_key(28.6, 77.2, "20230101", "20231231", "daily")
        k2 = _cache_key(40.7, -74.0, "20230101", "20231231", "daily")
        assert k1 != k2

    def test_cache_key_includes_temporal(self):
        k1 = _cache_key(28.6, 77.2, "20230101", "20231231", "daily")
        k2 = _cache_key(28.6, 77.2, "20230101", "20231231", "monthly")
        assert k1 != k2

    def test_cache_validity_nonexistent_file(self, tmp_path):
        assert not _cache_is_valid(tmp_path / "nonexistent.nc")

    def test_cache_validity_fresh_file(self, tmp_path):
        f = tmp_path / "test.nc"
        f.write_text("data")
        assert _cache_is_valid(f, ttl_days=30)


# ---------------------------------------------------------------------------
# NASA POWER Client Tests
# ---------------------------------------------------------------------------

class TestNASAPowerClient:
    """Tests for NASAPowerClient."""

    @patch("solar_intelligence.data_loader.requests.get")
    def test_fetch_daily_parses_response(self, mock_get, mock_nasa_power_response, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_nasa_power_response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        client = NASAPowerClient(cache_dir=tmp_path)
        ds = client.fetch_daily(28.6139, 77.209, "20230101", "20230105")

        assert isinstance(ds, xr.Dataset)
        assert "time" in ds.dims
        assert "ALLSKY_SFC_SW_DWN" in ds
        assert len(ds.time) == 5

    @patch("solar_intelligence.data_loader.requests.get")
    def test_fetch_daily_caches_result(self, mock_get, mock_nasa_power_response, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_nasa_power_response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        client = NASAPowerClient(cache_dir=tmp_path)

        # First call — hits API
        client.fetch_daily(28.6139, 77.209, "20230101", "20230105")
        assert mock_get.call_count == 1

        # Second call — uses cache
        client.fetch_daily(28.6139, 77.209, "20230101", "20230105")
        assert mock_get.call_count == 1  # No additional API call

    @patch("solar_intelligence.data_loader.requests.get")
    def test_fetch_handles_missing_values(self, mock_get, tmp_path):
        """NASA POWER uses -999.0 for missing data — should become NaN."""
        response = {
            "properties": {
                "parameter": {
                    "ALLSKY_SFC_SW_DWN": {
                        "20230101": 5.0,
                        "20230102": -999.0,
                        "20230103": 4.5,
                    }
                }
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        client = NASAPowerClient(
            cache_dir=tmp_path,
            parameters=["ALLSKY_SFC_SW_DWN"],
        )
        ds = client.fetch_daily(28.6, 77.2, "20230101", "20230103")

        assert np.isnan(ds["ALLSKY_SFC_SW_DWN"].values[1])
        assert ds["ALLSKY_SFC_SW_DWN"].values[0] == 5.0


# ---------------------------------------------------------------------------
# DataLoader Tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    """Tests for the unified DataLoader."""

    def test_load_netcdf(self, tmp_netcdf):
        loader = DataLoader()
        ds = loader.load_netcdf(tmp_netcdf)
        assert isinstance(ds, xr.Dataset)
        assert "ALLSKY_SFC_SW_DWN" in ds

    def test_load_netcdf_file_not_found(self):
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_netcdf("/nonexistent/path.nc")

    def test_slice_time(self, sample_dataset):
        sliced = DataLoader.slice_time(sample_dataset, "2021-01-01", "2021-12-31")
        assert sliced.time.values[0] >= np.datetime64("2021-01-01")
        assert sliced.time.values[-1] <= np.datetime64("2021-12-31")

    def test_slice_time_no_time_dim_raises(self):
        ds = xr.Dataset({"x": ("a", [1, 2, 3])})
        with pytest.raises(ValueError, match="no 'time' dimension"):
            DataLoader.slice_time(ds, "2021-01-01")

    @patch("solar_intelligence.data_loader.geocode_location")
    @patch.object(NASAPowerClient, "fetch_daily")
    def test_load_for_location_by_city(self, mock_fetch, mock_geocode, sample_dataset):
        mock_geocode.return_value = (28.6139, 77.2090)
        mock_fetch.return_value = sample_dataset

        loader = DataLoader()
        ds = loader.load_for_location(city="New Delhi")

        mock_geocode.assert_called_once_with("New Delhi")
        assert isinstance(ds, xr.Dataset)

    @patch.object(NASAPowerClient, "fetch_daily")
    def test_load_for_location_by_coords(self, mock_fetch, sample_dataset):
        mock_fetch.return_value = sample_dataset

        loader = DataLoader()
        ds = loader.load_for_location(lat=28.6139, lon=77.2090)

        assert isinstance(ds, xr.Dataset)

    def test_load_for_location_no_input_raises(self):
        loader = DataLoader()
        with pytest.raises(ValueError, match="city.*lat.*lon"):
            loader.load_for_location()


# ---------------------------------------------------------------------------
# Synthetic Data Tests
# ---------------------------------------------------------------------------

class TestSyntheticData:
    """Tests for synthetic data generator."""

    def test_output_structure(self, sample_dataset):
        assert isinstance(sample_dataset, xr.Dataset)
        assert "time" in sample_dataset.dims
        expected_vars = [
            "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI",
            "ALLSKY_SFC_SW_DIFF", "ALLSKY_KT", "T2M", "WS2M", "RH2M",
        ]
        for var in expected_vars:
            assert var in sample_dataset, f"Missing variable: {var}"

    def test_ghi_range(self, sample_dataset):
        ghi = sample_dataset["ALLSKY_SFC_SW_DWN"].values
        assert np.all(ghi >= 0), "GHI should be non-negative"
        assert np.all(ghi <= 12), "GHI should not exceed 12 kWh/m²/day"

    def test_clearness_index_range(self, sample_dataset):
        kt = sample_dataset["ALLSKY_KT"].values
        assert np.all(kt >= 0.1), "Clearness index should be >= 0.1"
        assert np.all(kt <= 1.0), "Clearness index should be <= 1.0"

    def test_temperature_reasonable(self, sample_dataset):
        t = sample_dataset["T2M"].values
        assert np.all(t > -50), "Temperature should be > -50°C"
        assert np.all(t < 60), "Temperature should be < 60°C"

    def test_different_locations_differ(self):
        ds1 = generate_synthetic_solar_data(lat=28.6, lon=77.2)
        ds2 = generate_synthetic_solar_data(lat=51.5, lon=-0.1)
        # Higher latitude should have lower average GHI
        ghi1 = float(ds1["ALLSKY_SFC_SW_DWN"].mean())
        ghi2 = float(ds2["ALLSKY_SFC_SW_DWN"].mean())
        assert ghi1 > ghi2, "Lower latitude should have higher GHI"

    def test_southern_hemisphere(self, sample_dataset_sydney):
        """Southern hemisphere should have valid data."""
        ghi = sample_dataset_sydney["ALLSKY_SFC_SW_DWN"].values
        assert np.all(ghi >= 0)
        assert float(sample_dataset_sydney["ALLSKY_SFC_SW_DWN"].mean()) > 0

    def test_has_metadata_attributes(self, sample_dataset):
        assert "source" in sample_dataset.attrs
        assert "latitude" in sample_dataset.attrs
        assert sample_dataset["ALLSKY_SFC_SW_DWN"].attrs["units"] == "kWh/m²/day"

    def test_deterministic_with_same_coords(self):
        ds1 = generate_synthetic_solar_data(lat=28.6, lon=77.2)
        ds2 = generate_synthetic_solar_data(lat=28.6, lon=77.2)
        np.testing.assert_array_equal(
            ds1["ALLSKY_SFC_SW_DWN"].values,
            ds2["ALLSKY_SFC_SW_DWN"].values,
        )
