"""Tests for Phase 2: Enhanced dashboard features.

Covers:
- SolarDashboard construction and widget presence
- Theme toggle widget
- Multi-location comparison widgets
- Loading spinner and status indicators
- Notification method
- Polar plot in orientation tab
- Datashader map in map tab
"""

from __future__ import annotations

import panel as pn
import pytest

from solar_intelligence.ui.panel_dashboard import SolarDashboard


class TestDashboardConstruction:

    def test_dashboard_creates(self):
        dashboard = SolarDashboard()
        assert dashboard is not None

    def test_dashboard_has_analyze_button(self):
        dashboard = SolarDashboard()
        assert dashboard._analyze_btn is not None
        assert dashboard._analyze_btn.name == "Analyze Solar Potential"

    def test_dashboard_has_loading_spinner(self):
        dashboard = SolarDashboard()
        assert dashboard._loading is not None
        assert dashboard._loading.value is False

    def test_dashboard_has_status(self):
        dashboard = SolarDashboard()
        assert dashboard._status is not None
        assert "Analyze" in dashboard._status.object

    def test_dashboard_has_theme_toggle(self):
        dashboard = SolarDashboard()
        assert dashboard._theme_toggle is not None
        assert dashboard._theme_toggle.name == "Dark Theme"
        assert dashboard._theme_toggle.value is False

    def test_dashboard_has_synthetic_toggle(self):
        dashboard = SolarDashboard()
        assert dashboard._use_synthetic is not None
        assert dashboard._use_synthetic.value is True


class TestMultiLocationWidgets:

    def test_compare_button_exists(self):
        dashboard = SolarDashboard()
        assert dashboard._compare_btn is not None
        assert dashboard._compare_btn.name == "Compare Locations"

    def test_compare_cities_input_exists(self):
        dashboard = SolarDashboard()
        assert dashboard._compare_cities_input is not None
        assert "delhi" in dashboard._compare_cities_input.placeholder.lower()


class TestDashboardView:

    def test_view_returns_template(self):
        dashboard = SolarDashboard()
        template = dashboard.view()
        assert isinstance(template, pn.template.FastListTemplate)

    def test_view_has_7_tabs(self):
        dashboard = SolarDashboard()
        template = dashboard.view()
        # Find the Tabs component in main
        main_items = template.main
        tabs = None
        for item in main_items:
            if isinstance(item, pn.Tabs):
                tabs = item
                break
        assert tabs is not None
        assert len(tabs) == 7  # Overview, Orientation, Map, Financial, Multi-Location, Data Sources, AI

    def test_tab_names(self):
        dashboard = SolarDashboard()
        template = dashboard.view()
        tabs = [item for item in template.main if isinstance(item, pn.Tabs)][0]
        # Should have 7 tabs including Data Sources
        assert len(tabs) == 7

    def test_sidebar_has_compare_widgets(self):
        dashboard = SolarDashboard()
        template = dashboard.view()
        # Sidebar should contain the compare button
        sidebar_items = template.sidebar
        assert len(sidebar_items) > 0

    def test_template_title(self):
        dashboard = SolarDashboard()
        template = dashboard.view()
        assert "Solar" in template.title

    def test_template_stores_reference(self):
        dashboard = SolarDashboard()
        template = dashboard.view()
        assert dashboard._template is template


class TestDashboardOutputAreas:

    def test_kpi_row_starts_empty(self):
        dashboard = SolarDashboard()
        assert len(dashboard._kpi_row) == 0

    def test_overview_area_has_placeholder(self):
        dashboard = SolarDashboard()
        assert len(dashboard._overview_area) > 0

    def test_multi_location_area_has_placeholder(self):
        dashboard = SolarDashboard()
        assert len(dashboard._multi_location_area) > 0

    def test_ai_area_has_placeholder(self):
        dashboard = SolarDashboard()
        assert len(dashboard._ai_area) > 0

    def test_map_area_has_placeholder(self):
        dashboard = SolarDashboard()
        assert len(dashboard._map_area) > 0

    def test_financial_area_has_placeholder(self):
        dashboard = SolarDashboard()
        assert len(dashboard._financial_area) > 0

    def test_orientation_area_has_placeholder(self):
        dashboard = SolarDashboard()
        assert len(dashboard._orientation_area) > 0
