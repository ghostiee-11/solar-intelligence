"""Reusable UI components for Solar Intelligence dashboard."""

from __future__ import annotations

import logging

import panel as pn
import param

logger = logging.getLogger(__name__)


class LocationPicker(param.Parameterized):
    """Combined location input widget with geocoding support."""

    city = param.String(default="", doc="City name for geocoding")
    latitude = param.Number(default=0.0, bounds=(-90, 90))
    longitude = param.Number(default=0.0, bounds=(-180, 180))
    location_name = param.String(default="")

    def __init__(self, **params):
        super().__init__(**params)
        self._city_input = pn.widgets.TextInput(
            name="City Name", value=self.city, placeholder="e.g. New York, London, Tokyo",
        )
        self._lat_input = pn.widgets.FloatInput(
            name="Latitude", value=self.latitude, step=0.1, start=-90, end=90,
        )
        self._lon_input = pn.widgets.FloatInput(
            name="Longitude", value=self.longitude, step=0.1, start=-180, end=180,
        )
        self._search_btn = pn.widgets.Button(
            name="Search Location", button_type="primary",
        )
        self._status = pn.pane.Markdown("")

        self._search_btn.on_click(self._on_search)
        self._lat_input.param.watch(self._on_coord_change, "value")
        self._lon_input.param.watch(self._on_coord_change, "value")

    def _on_search(self, event):
        city = self._city_input.value.strip()
        if not city:
            self._status.object = "*Enter a city name*"
            return
        try:
            from solar_intelligence.data_loader import geocode_location
            lat, lon = geocode_location(city)
            self._lat_input.value = lat
            self._lon_input.value = lon
            self.latitude = lat
            self.longitude = lon
            self.location_name = city
            self._status.object = f"Found: {city} ({lat:.4f}, {lon:.4f})"
        except Exception as e:
            self._status.object = f"*Error: {e}*"

    def _on_coord_change(self, event):
        self.latitude = self._lat_input.value
        self.longitude = self._lon_input.value

    @property
    def panel(self) -> pn.Column:
        """Return the location picker widget column."""
        return pn.Column(
            "### Location",
            self._city_input,
            self._search_btn,
            self._lat_input,
            self._lon_input,
            self._status,
            width=260,
        )


class PanelConfigurator(param.Parameterized):
    """Solar panel specification input widgets."""

    panel_efficiency = param.Number(default=0.20, bounds=(0.05, 0.40))
    panel_area = param.Number(default=1.7, bounds=(0.1, 10.0))
    num_panels = param.Integer(default=10, bounds=(1, 1000))
    system_losses = param.Number(default=0.14, bounds=(0.0, 0.5))
    tilt_angle = param.Number(default=30, bounds=(0, 90))
    direction = param.Selector(
        default="South",
        objects=["North", "North-East", "East", "South-East",
                 "South", "South-West", "West", "North-West"],
    )

    @property
    def panel(self) -> pn.Column:
        """Return the panel configuration widget column."""
        return pn.Column(
            "### Panel Configuration",
            pn.widgets.FloatSlider.from_param(
                self.param.panel_efficiency, name="Efficiency", format="0.0%",
            ),
            pn.widgets.FloatInput.from_param(
                self.param.panel_area, name="Panel Area (m²)",
            ),
            pn.widgets.IntSlider.from_param(
                self.param.num_panels, name="Number of Panels",
            ),
            pn.widgets.FloatSlider.from_param(
                self.param.system_losses, name="System Losses", format="0.0%",
            ),
            pn.widgets.FloatSlider.from_param(
                self.param.tilt_angle, name="Tilt Angle (°)",
            ),
            pn.widgets.Select.from_param(
                self.param.direction, name="Panel Direction",
            ),
            width=260,
        )


class FinancialConfigurator(param.Parameterized):
    """Financial parameter input widgets with multi-currency support."""

    currency = param.Selector(
        default="INR",
        objects=["INR", "USD", "EUR", "GBP", "CNY", "JPY", "AUD", "BRL",
                 "ZAR", "CAD", "KRW", "AED", "MXN", "SGD"],
        doc="Currency for financial calculations",
    )
    system_cost = param.Number(default=500000, bounds=(1000, 50_000_000))
    electricity_rate = param.Number(default=8.0, bounds=(0.01, 100.0))
    incentive_percent = param.Number(default=0.40, bounds=(0, 1.0))

    def __init__(self, **params):
        super().__init__(**params)
        from solar_intelligence.config import CURRENCIES, CURRENCY_DEFAULTS
        self._currencies = CURRENCIES
        self._currency_defaults = CURRENCY_DEFAULTS
        # Apply defaults for initial currency
        self._apply_currency_defaults(self.currency)

    def _apply_currency_defaults(self, currency_code: str):
        """Apply default financial values for selected currency."""
        defaults = self._currency_defaults.get(currency_code, {})
        if defaults:
            self.system_cost = defaults["system_cost"]
            self.electricity_rate = defaults["electricity_rate"]
            self.incentive_percent = defaults["incentive_percent"]

    @property
    def currency_symbol(self) -> str:
        """Get the symbol for current currency."""
        return self._currencies.get(self.currency, {}).get("symbol", "$")

    @property
    def panel(self) -> pn.Column:
        """Return the financial parameters widget column."""
        sym = self.currency_symbol

        currency_select = pn.widgets.Select.from_param(
            self.param.currency, name="Currency",
        )

        cost_input = pn.widgets.FloatInput.from_param(
            self.param.system_cost, name=f"System Cost ({sym})",
        )
        rate_input = pn.widgets.FloatInput.from_param(
            self.param.electricity_rate, name=f"Electricity Rate ({sym}/kWh)",
        )
        incentive_input = pn.widgets.FloatSlider.from_param(
            self.param.incentive_percent, name="Subsidy/Incentive", format="0.0%",
        )

        # Watch currency changes to update defaults
        def on_currency_change(event):
            self._apply_currency_defaults(event.new)
            sym_new = self._currencies.get(event.new, {}).get("symbol", "$")
            cost_input.name = f"System Cost ({sym_new})"
            rate_input.name = f"Electricity Rate ({sym_new}/kWh)"

        currency_select.param.watch(on_currency_change, "value")

        return pn.Column(
            "### Financial Parameters",
            currency_select,
            cost_input,
            rate_input,
            incentive_input,
            width=260,
        )


class KPICard:
    """Styled metric display card."""

    @staticmethod
    def create(title: str, value: str, subtitle: str = "", color: str = "#FFB900") -> pn.Column:
        """Create a styled KPI metric card."""
        return pn.Column(
            pn.pane.HTML(f"""
            <div style="background: linear-gradient(135deg, {color}22, {color}11);
                        border-left: 4px solid {color}; border-radius: 8px;
                        padding: 16px; min-width: 180px;">
                <div style="font-size: 12px; color: #666; text-transform: uppercase;
                            letter-spacing: 1px;">{title}</div>
                <div style="font-size: 28px; font-weight: 700; color: #333;
                            margin: 4px 0;">{value}</div>
                <div style="font-size: 11px; color: #888;">{subtitle}</div>
            </div>
            """),
            width=220,
        )
