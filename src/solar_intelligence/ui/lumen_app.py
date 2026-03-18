"""Lumen pipeline application for Solar Intelligence.

Provides a Lumen-based data exploration interface with custom
Source and Transform implementations for solar datasets.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import param

from lumen.pipeline import Pipeline
from lumen.sources.base import Source
from lumen.transforms.base import Transform

from solar_intelligence.data_loader import generate_synthetic_solar_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Lumen Source
# ---------------------------------------------------------------------------

class SolarDataSource(Source):
    """Lumen Source that wraps NASA POWER API and synthetic solar data.

    Exposes solar radiation datasets as tabular data for Lumen pipelines.

    Parameters
    ----------
    latitude : float
        Location latitude.
    longitude : float
        Location longitude.
    use_synthetic : bool
        Use synthetic data instead of API.
    start_year : int
        Start year for data range.
    end_year : int
        End year for data range.
    """

    source_type = "solar_data"

    latitude = param.Number(default=28.6139, bounds=(-90, 90))
    longitude = param.Number(default=77.2090, bounds=(-180, 180))
    use_synthetic = param.Boolean(default=True)
    start_year = param.Integer(default=2020, bounds=(1981, 2025))
    end_year = param.Integer(default=2023, bounds=(1981, 2025))

    _dataset = param.Parameter(default=None, precedence=-1)

    def _load_data(self):
        """Load solar dataset."""
        if self._dataset is not None:
            return

        if self.use_synthetic:
            ds = generate_synthetic_solar_data(
                lat=self.latitude, lon=self.longitude,
                start_year=self.start_year, end_year=self.end_year,
            )
        else:
            from solar_intelligence.data_loader import DataLoader
            loader = DataLoader()
            ds = loader.load_from_api(
                self.latitude, self.longitude,
                self.start_year, self.end_year,
            )
        self._dataset = ds

    def get_tables(self) -> list[str]:
        """Return available table names."""
        return ["daily_solar", "monthly_solar", "metadata"]

    def get_schema(self, table: str | None = None) -> dict[str, Any]:
        """Return schema for a table."""
        schemas = {
            "daily_solar": {
                "time": {"type": "datetime64[ns]"},
                "ALLSKY_SFC_SW_DWN": {"type": "float64"},
                "CLRSKY_SFC_SW_DWN": {"type": "float64"},
                "ALLSKY_SFC_SW_DNI": {"type": "float64"},
                "ALLSKY_SFC_SW_DIFF": {"type": "float64"},
                "ALLSKY_KT": {"type": "float64"},
                "T2M": {"type": "float64"},
                "WS2M": {"type": "float64"},
                "RH2M": {"type": "float64"},
            },
            "monthly_solar": {
                "month": {"type": "int64"},
                "month_name": {"type": "string"},
                "GHI": {"type": "float64"},
                "DNI": {"type": "float64"},
                "DHI": {"type": "float64"},
            },
            "metadata": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
        }
        if table:
            return schemas.get(table, {})
        return schemas

    def get(self, table: str, **query) -> pd.DataFrame:
        """Get data for a specific table.

        Parameters
        ----------
        table : str
            Table name: "daily_solar", "monthly_solar", or "metadata".

        Returns
        -------
        pd.DataFrame
        """
        self._load_data()
        ds = self._dataset

        if table == "daily_solar":
            df = ds.to_dataframe().reset_index()
            if "time" not in df.columns and df.index.name == "time":
                df = df.reset_index()
            return df

        elif table == "monthly_solar":
            from solar_intelligence.solar_analysis import SolarAnalyzer
            analyzer = SolarAnalyzer(
                dataset=ds, latitude=self.latitude, longitude=self.longitude,
            )
            return analyzer.monthly_irradiance().reset_index()

        elif table == "metadata":
            return pd.DataFrame({
                "key": ["latitude", "longitude", "source", "start", "end"],
                "value": [
                    str(self.latitude), str(self.longitude),
                    ds.attrs.get("source", "unknown"),
                    str(self.start_year), str(self.end_year),
                ],
            })

        raise ValueError(f"Unknown table: {table}")


# ---------------------------------------------------------------------------
# Custom Lumen Transforms
# ---------------------------------------------------------------------------

class SolarEnergyTransform(Transform):
    """Lumen Transform that adds energy estimation columns to solar data.

    Parameters
    ----------
    panel_efficiency : float
        Panel conversion efficiency.
    total_area : float
        Total panel area in m².
    system_losses : float
        System losses fraction.
    ghi_column : str
        Name of GHI column in input data.
    """

    transform_type = "solar_energy"

    panel_efficiency = param.Number(default=0.20, bounds=(0.05, 0.40))
    total_area = param.Number(default=17.0, bounds=(0.1, 10000))
    system_losses = param.Number(default=0.14, bounds=(0, 0.5))
    ghi_column = param.String(default="ALLSKY_SFC_SW_DWN")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add energy estimation columns."""
        df = df.copy()
        ghi = df[self.ghi_column].values

        df["energy_kwh"] = (
            ghi
            * self.panel_efficiency
            * self.total_area
            * (1 - self.system_losses)
        )
        df["energy_kwh"] = df["energy_kwh"].clip(lower=0)

        return df


class MonthlyAggregateTransform(Transform):
    """Lumen Transform that aggregates daily solar data to monthly.

    Parameters
    ----------
    time_column : str
        Name of the time column.
    value_columns : list[str]
        Columns to aggregate.
    method : str
        Aggregation method (mean, sum, max, min).
    """

    transform_type = "monthly_aggregate"

    time_column = param.String(default="time")
    value_columns = param.List(default=["ALLSKY_SFC_SW_DWN"])
    method = param.Selector(default="mean", objects=["mean", "sum", "max", "min"])

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate to monthly resolution."""
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df["month"] = df[self.time_column].dt.month

        agg_funcs = {col: self.method for col in self.value_columns if col in df.columns}
        if not agg_funcs:
            return df

        monthly = df.groupby("month").agg(agg_funcs).reset_index()

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m - 1])

        return monthly


class AnomalyTransform(Transform):
    """Lumen Transform that computes irradiance anomalies.

    Parameters
    ----------
    time_column : str
        Name of the time column.
    value_column : str
        Column to compute anomaly for.
    """

    transform_type = "solar_anomaly"

    time_column = param.String(default="time")
    value_column = param.String(default="ALLSKY_SFC_SW_DWN")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute anomaly from monthly climatology."""
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df["month"] = df[self.time_column].dt.month

        climatology = df.groupby("month")[self.value_column].mean()
        df["climatology"] = df["month"].map(climatology)
        df["anomaly"] = df[self.value_column] - df["climatology"]

        return df


def create_solar_pipeline(
    latitude: float = 28.6139,
    longitude: float = 77.2090,
) -> Pipeline:
    """Create a Lumen pipeline for solar data analysis.

    Parameters
    ----------
    latitude, longitude : float
        Location coordinates.

    Returns
    -------
    Pipeline
        Configured Lumen pipeline.
    """
    source = SolarDataSource(latitude=latitude, longitude=longitude)

    pipeline = Pipeline(
        source=source,
        table="daily_solar",
        transforms=[
            SolarEnergyTransform(),
            AnomalyTransform(),
        ],
    )

    return pipeline


# ---------------------------------------------------------------------------
# Lumen YAML Configuration
# ---------------------------------------------------------------------------

LUMEN_YAML_CONFIG = """
# Solar Intelligence - Lumen Declarative Pipeline Configuration
# Usage: lumen serve lumen_config.yaml

sources:
  solar_data:
    type: solar_intelligence.ui.lumen_app.SolarDataSource
    latitude: 28.6139
    longitude: 77.2090
    use_synthetic: true
    start_year: 2020
    end_year: 2023

pipelines:
  daily_analysis:
    source: solar_data
    table: daily_solar
    transforms:
      - type: solar_intelligence.ui.lumen_app.SolarEnergyTransform
        panel_efficiency: 0.20
        total_area: 17.0
        system_losses: 0.14
      - type: solar_intelligence.ui.lumen_app.AnomalyTransform
        value_column: ALLSKY_SFC_SW_DWN

  monthly_summary:
    source: solar_data
    table: daily_solar
    transforms:
      - type: solar_intelligence.ui.lumen_app.MonthlyAggregateTransform
        value_columns:
          - ALLSKY_SFC_SW_DWN
          - ALLSKY_SFC_SW_DNI

  metadata:
    source: solar_data
    table: metadata
"""


def get_lumen_yaml_config(
    latitude: float = 28.6139,
    longitude: float = 77.2090,
    panel_efficiency: float = 0.20,
    num_panels: int = 10,
    panel_area: float = 1.7,
) -> str:
    """Generate Lumen YAML configuration for a specific location.

    Parameters
    ----------
    latitude, longitude : float
        Location coordinates.
    panel_efficiency : float
        Solar panel efficiency.
    num_panels : int
        Number of panels.
    panel_area : float
        Area per panel in m2.

    Returns
    -------
    str
        YAML configuration string for Lumen.
    """
    total_area = num_panels * panel_area
    return f"""# Solar Intelligence - Lumen Pipeline Config
# Generated for ({latitude}, {longitude})

sources:
  solar_data:
    type: solar_intelligence.ui.lumen_app.SolarDataSource
    latitude: {latitude}
    longitude: {longitude}
    use_synthetic: true
    start_year: 2020
    end_year: 2023

pipelines:
  daily_analysis:
    source: solar_data
    table: daily_solar
    transforms:
      - type: solar_intelligence.ui.lumen_app.SolarEnergyTransform
        panel_efficiency: {panel_efficiency}
        total_area: {total_area}
        system_losses: 0.14
      - type: solar_intelligence.ui.lumen_app.AnomalyTransform
        value_column: ALLSKY_SFC_SW_DWN

  monthly_summary:
    source: solar_data
    table: daily_solar
    transforms:
      - type: solar_intelligence.ui.lumen_app.MonthlyAggregateTransform
        value_columns:
          - ALLSKY_SFC_SW_DWN
          - ALLSKY_SFC_SW_DNI

  metadata:
    source: solar_data
    table: metadata
"""


def write_lumen_config(
    output_path: str = "lumen_config.yaml",
    **kwargs,
) -> str:
    """Write Lumen YAML config to a file.

    Parameters
    ----------
    output_path : str
        Path to write the config file.
    **kwargs
        Passed to get_lumen_yaml_config().

    Returns
    -------
    str
        Path to the written file.
    """
    from pathlib import Path

    config = get_lumen_yaml_config(**kwargs)
    path = Path(output_path)
    path.write_text(config)
    logger.info("Lumen config written to: %s", path)
    return str(path)


# ---------------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------------

class CacheManager(param.Parameterized):
    """Manage the solar data cache directory.

    Provides inspection, cleanup, and TTL management for cached
    NASA POWER API responses stored as NetCDF files.

    Parameters
    ----------
    cache_dir : str
        Path to the cache directory.
    """

    cache_dir = param.String(
        default=str(Source._get_type("solar_data") if False else ""),
        doc="Cache directory path",
    )

    def __init__(self, cache_dir: str | None = None, **params):
        from solar_intelligence.config import CACHE_DIR
        super().__init__(**params)
        self.cache_dir = cache_dir or str(CACHE_DIR)

    def list_cached_files(self) -> list[dict[str, Any]]:
        """List all cached files with metadata.

        Returns
        -------
        list[dict]
            Each dict has: filename, size_kb, age_days, path.
        """
        import time
        from pathlib import Path

        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            return []

        files = []
        for f in sorted(cache_path.glob("*.nc")):
            stat = f.stat()
            age_days = (time.time() - stat.st_mtime) / 86400
            files.append({
                "filename": f.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "age_days": round(age_days, 1),
                "path": str(f),
            })
        return files

    def cache_size_mb(self) -> float:
        """Total cache size in megabytes."""
        files = self.list_cached_files()
        total_kb = sum(f["size_kb"] for f in files)
        return round(total_kb / 1024, 2)

    def cache_count(self) -> int:
        """Number of cached files."""
        return len(self.list_cached_files())

    def clear_expired(self, ttl_days: int = 30) -> int:
        """Remove cached files older than TTL.

        Parameters
        ----------
        ttl_days : int
            Maximum age in days.

        Returns
        -------
        int
            Number of files removed.
        """
        from pathlib import Path

        removed = 0
        for entry in self.list_cached_files():
            if entry["age_days"] > ttl_days:
                Path(entry["path"]).unlink()
                removed += 1
                logger.info("Removed expired cache: %s", entry["filename"])
        return removed

    def clear_all(self) -> int:
        """Remove all cached files.

        Returns
        -------
        int
            Number of files removed.
        """
        from pathlib import Path

        files = self.list_cached_files()
        for entry in files:
            Path(entry["path"]).unlink()
        logger.info("Cleared %d cached files", len(files))
        return len(files)

    def summary(self) -> dict[str, Any]:
        """Cache summary statistics.

        Returns
        -------
        dict
            count, total_size_mb, oldest_days, newest_days.
        """
        files = self.list_cached_files()
        if not files:
            return {
                "count": 0,
                "total_size_mb": 0,
                "oldest_days": 0,
                "newest_days": 0,
            }

        ages = [f["age_days"] for f in files]
        return {
            "count": len(files),
            "total_size_mb": self.cache_size_mb(),
            "oldest_days": max(ages),
            "newest_days": min(ages),
        }
