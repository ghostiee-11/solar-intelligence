"""Tests for Phase 4: Lumen YAML config + Cache management.

Covers:
- LUMEN_YAML_CONFIG constant
- get_lumen_yaml_config() dynamic generation
- write_lumen_config() file output
- CacheManager listing, sizing, cleanup
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from solar_intelligence.ui.lumen_app import (
    LUMEN_YAML_CONFIG,
    CacheManager,
    get_lumen_yaml_config,
    write_lumen_config,
)


# ---------------------------------------------------------------------------
# Lumen YAML Config Tests
# ---------------------------------------------------------------------------

class TestLumenYAMLConfig:

    def test_yaml_config_is_string(self):
        assert isinstance(LUMEN_YAML_CONFIG, str)

    def test_yaml_config_has_sources(self):
        assert "sources:" in LUMEN_YAML_CONFIG

    def test_yaml_config_has_pipelines(self):
        assert "pipelines:" in LUMEN_YAML_CONFIG

    def test_yaml_config_has_solar_data_source(self):
        assert "SolarDataSource" in LUMEN_YAML_CONFIG

    def test_yaml_config_has_transforms(self):
        assert "SolarEnergyTransform" in LUMEN_YAML_CONFIG
        assert "AnomalyTransform" in LUMEN_YAML_CONFIG
        assert "MonthlyAggregateTransform" in LUMEN_YAML_CONFIG


class TestDynamicYAMLConfig:

    def test_generates_yaml_string(self):
        config = get_lumen_yaml_config()
        assert isinstance(config, str)
        assert "sources:" in config

    def test_custom_latitude(self):
        config = get_lumen_yaml_config(latitude=35.7, longitude=139.7)
        assert "35.7" in config
        assert "139.7" in config

    def test_custom_panel_params(self):
        config = get_lumen_yaml_config(
            panel_efficiency=0.22, num_panels=20, panel_area=2.0,
        )
        assert "0.22" in config
        assert "40.0" in config  # 20 * 2.0 = 40.0

    def test_write_config_to_file(self, tmp_path):
        output = tmp_path / "test_config.yaml"
        result = write_lumen_config(str(output), latitude=40.0, longitude=-74.0)
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "40.0" in content
        assert "-74.0" in content


# ---------------------------------------------------------------------------
# Cache Manager Tests
# ---------------------------------------------------------------------------

class TestCacheManager:

    @pytest.fixture
    def cache_dir(self, tmp_path):
        cache = tmp_path / "test_cache"
        cache.mkdir()
        return str(cache)

    @pytest.fixture
    def populated_cache(self, cache_dir):
        # Create some fake .nc files
        for i in range(3):
            f = Path(cache_dir) / f"test_file_{i}.nc"
            f.write_bytes(b"x" * (1024 * (i + 1)))  # 1KB, 2KB, 3KB
        return cache_dir

    def test_empty_cache(self, cache_dir):
        cm = CacheManager(cache_dir=cache_dir)
        assert cm.cache_count() == 0
        assert cm.cache_size_mb() == 0

    def test_list_cached_files(self, populated_cache):
        cm = CacheManager(cache_dir=populated_cache)
        files = cm.list_cached_files()
        assert len(files) == 3
        assert all("filename" in f for f in files)
        assert all("size_kb" in f for f in files)

    def test_cache_count(self, populated_cache):
        cm = CacheManager(cache_dir=populated_cache)
        assert cm.cache_count() == 3

    def test_cache_size(self, populated_cache):
        cm = CacheManager(cache_dir=populated_cache)
        # 1 + 2 + 3 = 6 KB = ~0.006 MB
        size = cm.cache_size_mb()
        assert size > 0

    def test_clear_all(self, populated_cache):
        cm = CacheManager(cache_dir=populated_cache)
        removed = cm.clear_all()
        assert removed == 3
        assert cm.cache_count() == 0

    def test_summary(self, populated_cache):
        cm = CacheManager(cache_dir=populated_cache)
        s = cm.summary()
        assert s["count"] == 3
        assert s["total_size_mb"] > 0
        assert s["newest_days"] >= 0

    def test_summary_empty(self, cache_dir):
        cm = CacheManager(cache_dir=cache_dir)
        s = cm.summary()
        assert s["count"] == 0

    def test_clear_expired_none_old(self, populated_cache):
        cm = CacheManager(cache_dir=populated_cache)
        removed = cm.clear_expired(ttl_days=30)
        assert removed == 0  # All files are fresh
        assert cm.cache_count() == 3

    def test_nonexistent_cache_dir(self, tmp_path):
        cm = CacheManager(cache_dir=str(tmp_path / "nonexistent"))
        assert cm.cache_count() == 0
        assert cm.list_cached_files() == []
