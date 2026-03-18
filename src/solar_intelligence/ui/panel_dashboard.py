"""Main Panel dashboard for Solar Intelligence Platform.

Provides a complete interactive interface for solar analysis,
simulation, and financial planning using the HoloViz ecosystem.

Features:
- 6-tab layout: Overview, Orientation, Map, Financial, Multi-Location, AI Insights
- Loading spinners and status notifications during analysis
- Error toasts for graceful failure handling
- Dark/light theme toggle via FastListTemplate
- Geocoding integration for city search
"""

from __future__ import annotations

import logging

import holoviews as hv
import numpy as np
import panel as pn
import param

from solar_intelligence.ai_engine import SolarAIEngine
from solar_intelligence.data_loader import (
    DataLoader,
    DualSourceLoader,
    ERA5Client,
    generate_global_solar_grid,
    generate_synthetic_solar_data,
)
from solar_intelligence.energy_estimator import EnergyEstimator
from solar_intelligence.financial import FinancialAnalyzer
from solar_intelligence.orientation_simulator import OrientationSimulator
from solar_intelligence.solar_analysis import (
    DualSourceAnalyzer,
    MultiLocationComparator,
    SolarAnalyzer,
)
from solar_intelligence.ui.components import (
    FinancialConfigurator,
    KPICard,
    LocationPicker,
    PanelConfigurator,
)
from solar_intelligence.visualization import SolarVisualizer

hv.extension("bokeh")
pn.extension(sizing_mode="stretch_width", notifications=True)

logger = logging.getLogger(__name__)


class SolarDashboard(param.Parameterized):
    """Main Solar Intelligence Dashboard.

    Integrates all analysis modules into an interactive Panel application
    with 6 tabs, loading indicators, error handling, and theme control.

    Parameters
    ----------
    location : LocationPicker
        Location input widget with geocoding.
    panel_config : PanelConfigurator
        Solar panel specification inputs.
    financial_config : FinancialConfigurator
        Financial parameter inputs.
    """

    # Sub-components
    location = param.ClassSelector(class_=LocationPicker, default=LocationPicker())
    panel_config = param.ClassSelector(class_=PanelConfigurator, default=PanelConfigurator())
    financial_config = param.ClassSelector(
        class_=FinancialConfigurator, default=FinancialConfigurator(),
    )

    # State
    _dataset = param.Parameter(default=None)
    _analysis_done = param.Boolean(default=False)

    def __init__(self, **params):
        super().__init__(**params)
        self._loader = DataLoader()
        self._visualizer = SolarVisualizer()
        self._ai = SolarAIEngine()

        # --- Control widgets ---
        self._analyze_btn = pn.widgets.Button(
            name="Analyze Solar Potential", button_type="success", width=250,
        )
        self._analyze_btn.on_click(self._run_analysis)

        self._use_synthetic = pn.widgets.Toggle(
            name="Use Synthetic Data (Offline)", value=True, width=250,
        )
        self._use_era5 = pn.widgets.Toggle(
            name="Enable ERA5 (CDS API)", value=False, width=250,
        )

        self._theme_toggle = pn.widgets.Toggle(
            name="Dark Theme", value=False, width=250,
        )
        self._theme_toggle.param.watch(self._on_theme_change, "value")

        # --- Status indicators ---
        self._status = pn.pane.Markdown("*Click 'Analyze' to start*")
        self._loading = pn.indicators.LoadingSpinner(value=False, size=30)

        # --- Output areas ---
        self._kpi_row = pn.Row()
        self._overview_area = pn.Column(
            pn.pane.Markdown("*Run analysis to see results*"),
        )
        self._orientation_area = pn.Column(
            pn.pane.Markdown("*Run analysis to see orientation results*"),
        )
        self._map_area = pn.Column(
            pn.pane.Markdown("*Run analysis to see solar map*"),
        )
        self._financial_area = pn.Column(
            pn.pane.Markdown("*Run analysis to see financial projections*"),
        )
        self._multi_location_area = pn.Column(
            pn.pane.Markdown("*Run analysis to enable multi-location comparison*"),
        )
        self._ai_area = pn.Column(
            pn.pane.Markdown("*Run analysis to generate AI insights*"),
        )
        self._dual_source_area = pn.Column(
            pn.pane.Markdown(
                "*Enable ERA5 data source and run analysis to see "
                "cross-validation results*"
            ),
        )

        # Multi-location comparison widgets
        self._compare_btn = pn.widgets.Button(
            name="Compare Locations", button_type="primary", width=250,
        )
        self._compare_btn.on_click(self._run_comparison)
        self._compare_cities_input = pn.widgets.TextInput(
            name="Cities (comma-separated)",
            placeholder="Delhi, London, Cairo, Tokyo, Sydney",
            width=250,
        )

        # --- Interactive simulation widgets ---
        self._sim_tilt = pn.widgets.IntSlider(
            name="Panel Tilt", value=30, start=0, end=90, step=5, width=220,
        )
        self._sim_azimuth = pn.widgets.Select(
            name="Facing Direction",
            options={"South": 180, "South-East": 135, "South-West": 225,
                     "East": 90, "West": 270, "North": 0},
            value=180, width=220,
        )
        self._sim_panels = pn.widgets.IntSlider(
            name="Number of Panels", value=10, start=1, end=50, step=1, width=220,
        )
        self._sim_run_btn = pn.widgets.Button(
            name="Simulate at Location", button_type="primary", width=220,
        )
        self._sim_run_btn.on_click(self._run_map_simulation)
        self._sim_result_area = pn.Column(
            pn.pane.Markdown("*Click on the map or adjust parameters*"),
        )
        self._clicked_lat = None
        self._clicked_lon = None
        self._chat_solar = None
        self._chat_energy = None
        self._chat_financial = None

        # Template reference (set in view())
        self._template = None

    def _notify(self, message: str, notification_type: str = "info"):
        """Send a notification toast to the UI."""
        if pn.state.notifications:
            if notification_type == "error":
                pn.state.notifications.error(message, duration=5000)
            elif notification_type == "success":
                pn.state.notifications.success(message, duration=3000)
            elif notification_type == "warning":
                pn.state.notifications.warning(message, duration=4000)
            else:
                pn.state.notifications.info(message, duration=3000)

    def _on_theme_change(self, event):
        """Toggle between dark and light theme."""
        if self._template is not None:
            if event.new:
                self._template.theme = pn.template.DarkTheme
            else:
                self._template.theme = pn.template.DefaultTheme

    def _run_analysis(self, event=None):
        """Execute full solar analysis pipeline with status updates."""
        self._loading.value = True
        self._analyze_btn.disabled = True
        self._status.object = "**Loading data...**"

        try:
            lat = self.location.latitude
            lon = self.location.longitude
            name = self.location.location_name or f"({lat:.2f}, {lon:.2f})"

            # 1. Load data
            if self._use_synthetic.value:
                ds = generate_synthetic_solar_data(lat=lat, lon=lon)
                self._status.object = f"Using synthetic data for **{name}**"
            else:
                try:
                    ds = self._loader.load_from_api(lat, lon)
                    self._status.object = f"Loaded NASA POWER data for **{name}**"
                except Exception as e:
                    logger.warning("API failed, falling back to synthetic: %s", e)
                    ds = generate_synthetic_solar_data(lat=lat, lon=lon)
                    self._status.object = f"API unavailable - using synthetic for **{name}**"
                    self._notify(f"NASA POWER API unavailable: {e}", "warning")

            self._dataset = ds

            # 2. Solar analysis
            self._status.object = "**Analyzing solar radiation...**"
            analyzer = SolarAnalyzer(dataset=ds, latitude=lat, longitude=lon)
            solar_summary = analyzer.summary()
            monthly = analyzer.monthly_irradiance()
            rolling = analyzer.rolling_average()

            # 3. Energy estimation
            self._status.object = "**Estimating energy production...**"
            estimator = EnergyEstimator(
                panel_efficiency=self.panel_config.panel_efficiency,
                panel_area=self.panel_config.panel_area,
                num_panels=self.panel_config.num_panels,
                system_losses=self.panel_config.system_losses,
            )
            annual_energy = estimator.estimate_annual_energy(ds)
            monthly_energy = estimator.estimate_monthly_energy(ds)
            energy_summary = estimator.system_summary(ds)

            # 4. Orientation simulation
            self._status.object = "**Simulating panel orientations...**"
            simulator = OrientationSimulator(
                latitude=lat, longitude=lon,
                panel_efficiency=self.panel_config.panel_efficiency,
                panel_area=estimator.total_area,
                system_losses=self.panel_config.system_losses,
                tilt_angles=[0, 15, 30, 45, 60],
                azimuths={
                    "North": 0, "East": 90, "South": 180, "West": 270,
                    "South-East": 135, "South-West": 225,
                },
            )

            ghi_year = ds["ALLSKY_SFC_SW_DWN"].sel(
                time=slice("2023-01-01", "2023-12-31"),
            ).values
            if len(ghi_year) < 365:
                ghi_year = ds["ALLSKY_SFC_SW_DWN"].values[:365]

            sim_data = simulator.simulate_all_orientations(ghi_year, year=2023)
            optimal = simulator.optimal_orientation(ghi_year, year=2023)
            sensitivity = simulator.tilt_sensitivity_analysis(ghi_year, year=2023)
            daily_profile = simulator.daily_profile_by_orientation(
                ghi_year, date="2023-06-21",
            )
            seasonal = simulator.seasonal_comparison(ghi_year, year=2023)

            # 5. Financial analysis
            self._status.object = "**Computing financial projections...**"
            financial = FinancialAnalyzer(
                system_cost=self.financial_config.system_cost,
                electricity_rate=self.financial_config.electricity_rate,
                incentive_percent=self.financial_config.incentive_percent,
            )
            fin_summary = financial.financial_summary(annual_energy)
            lifetime = financial.lifetime_savings(annual_energy)

            # 6. Dual-source comparison (ERA5)
            dual_datasets = {"NASA POWER": ds}
            if self._use_era5.value:
                self._status.object = "**Fetching ERA5 data...**"
                try:
                    era5_client = ERA5Client()
                    era5_ds = era5_client.fetch_daily(lat, lon, 2023, 2023)
                    dual_datasets["ERA5"] = era5_ds
                    self._notify("ERA5 data loaded", "success")
                except ImportError:
                    self._notify(
                        "cdsapi not installed. Run: pip install cdsapi", "warning",
                    )
                except Exception as e:
                    logger.warning("ERA5 fetch failed: %s", e)
                    self._notify(f"ERA5 unavailable: {e}", "warning")

            # 7. Update all UI sections
            self._update_kpis(solar_summary, energy_summary, fin_summary)
            self._update_overview(monthly, rolling, ds, monthly_energy)
            self._update_orientation(
                sim_data, sensitivity, daily_profile, seasonal, optimal,
            )
            self._update_map(lat, lon, ds, dual_datasets)
            self._update_financial(lifetime, fin_summary)
            self._update_ai(solar_summary, energy_summary, fin_summary, optimal)
            self._update_dual_source(dual_datasets, lat, lon)

            self._analysis_done = True
            self._status.object = f"Analysis complete for **{name}**"
            self._notify(f"Analysis complete for {name}", "success")

        except Exception as e:
            logger.exception("Analysis failed")
            self._status.object = f"**Error:** {e}"
            self._notify(f"Analysis failed: {e}", "error")
        finally:
            self._loading.value = False
            self._analyze_btn.disabled = False

    def _run_comparison(self, event=None):
        """Run multi-location comparison from user input."""
        cities_text = self._compare_cities_input.value.strip()
        if not cities_text:
            self._notify("Enter comma-separated city names", "warning")
            return

        self._loading.value = True
        self._status.object = "**Comparing locations...**"

        try:
            from solar_intelligence.data_loader import geocode_location

            city_names = [c.strip() for c in cities_text.split(",") if c.strip()]
            if len(city_names) < 2:
                self._notify("Enter at least 2 cities to compare", "warning")
                return
            if len(city_names) > 5:
                city_names = city_names[:5]
                self._notify("Limited to 5 cities", "info")

            # Geocode all cities
            locations = {}
            for city in city_names:
                try:
                    lat, lon = geocode_location(city)
                    locations[city] = (lat, lon)
                except ValueError:
                    self._notify(f"Could not geocode '{city}', skipping", "warning")

            if len(locations) < 2:
                self._notify("Need at least 2 valid locations", "error")
                return

            # Run comparison
            self._status.object = f"**Comparing {len(locations)} locations...**"
            comparator = MultiLocationComparator(locations=locations)
            comparator.load_data(start_year=2023, end_year=2023)

            ghi_df = comparator.compare_ghi()
            monthly_df = comparator.compare_monthly()
            ranking_df = comparator.ranking()

            # Update UI
            self._multi_location_area.clear()
            self._multi_location_area.extend([
                pn.pane.Markdown("### Location Ranking"),
                pn.pane.HoloViews(
                    self._visualizer.multi_location_radar_table(ranking_df),
                    sizing_mode="stretch_width",
                ),
                pn.Row(
                    pn.pane.HoloViews(
                        self._visualizer.multi_location_bar(ghi_df),
                        sizing_mode="stretch_width",
                    ),
                    pn.pane.HoloViews(
                        self._visualizer.multi_location_monthly(monthly_df),
                        sizing_mode="stretch_width",
                    ),
                ),
            ])

            self._status.object = f"**Compared {len(locations)} locations**"
            self._notify(f"Compared {len(locations)} locations", "success")

        except Exception as e:
            logger.exception("Comparison failed")
            self._status.object = f"**Comparison error:** {e}"
            self._notify(f"Comparison failed: {e}", "error")
        finally:
            self._loading.value = False

    def _update_kpis(self, solar, energy, financial):
        """Update KPI cards row."""
        self._kpi_row.clear()
        self._kpi_row.extend([
            KPICard.create(
                "Daily GHI",
                f"{solar['average_daily_ghi']:.2f}",
                "kWh/m\u00b2/day",
                "#FFB900",
            ),
            KPICard.create(
                "Annual Energy",
                f"{energy['production']['annual_energy_kwh']:,.0f}",
                "kWh/year",
                "#4CAF50",
            ),
            KPICard.create(
                "Capacity Factor",
                f"{energy['performance']['capacity_factor_pct']:.1f}%",
                "System efficiency",
                "#2196F3",
            ),
            KPICard.create(
                "Payback",
                f"{financial['returns']['payback_years']}",
                "years",
                "#FF6B00",
            ),
            KPICard.create(
                "CO\u2082 Offset",
                f"{financial['environmental']['annual_co2_offset_kg']:,.0f}",
                "kg/year",
                "#2E7D32",
            ),
        ])

    def _update_overview(self, monthly, rolling, ds, monthly_energy):
        """Update overview tab charts."""
        self._overview_area.clear()
        self._overview_area.extend([
            pn.Row(
                pn.pane.HoloViews(
                    self._visualizer.monthly_irradiance_bar(monthly),
                    sizing_mode="stretch_width",
                ),
                pn.pane.HoloViews(
                    self._visualizer.energy_projection_area(monthly_energy),
                    sizing_mode="stretch_width",
                ),
            ),
            pn.Row(
                pn.pane.HoloViews(
                    self._visualizer.daily_irradiance_timeseries(rolling),
                    sizing_mode="stretch_width",
                ),
            ),
            pn.Row(
                pn.pane.HoloViews(
                    self._visualizer.irradiance_distribution(ds),
                    sizing_mode="stretch_width",
                ),
                pn.pane.HoloViews(
                    self._visualizer.seasonal_heatmap(ds),
                    sizing_mode="stretch_width",
                ),
            ),
        ])

    def _update_orientation(self, sim_data, sensitivity, profile, seasonal, optimal):
        """Update orientation analysis tab."""
        self._orientation_area.clear()

        opt_html = f"""
        <div style="background: #E8F5E9; border-radius: 8px; padding: 16px;
                    border-left: 4px solid #4CAF50; margin-bottom: 16px;">
            <h3 style="margin: 0 0 8px 0; color: #2E7D32;">Optimal Configuration</h3>
            <p style="margin: 4px 0; font-size: 16px;">
                <strong>{optimal['best_direction']}</strong> facing at
                <strong>{optimal['best_tilt']}\u00b0 tilt</strong>
            </p>
            <p style="margin: 4px 0; color: #666;">
                {optimal['energy_gain_vs_horizontal_pct']:.1f}% better than horizontal |
                {optimal['energy_gain_vs_worst_pct']:.1f}% better than worst orientation
            </p>
        </div>
        """

        self._orientation_area.extend([
            pn.pane.HTML(opt_html),
            pn.Row(
                pn.pane.HoloViews(
                    self._visualizer.orientation_comparison_bar(sim_data),
                    sizing_mode="stretch_width",
                ),
                pn.pane.HoloViews(
                    self._visualizer.tilt_energy_curve(sensitivity),
                    sizing_mode="stretch_width",
                ),
            ),
            pn.Row(
                pn.pane.HoloViews(
                    self._visualizer.daily_profile_overlay(profile),
                    sizing_mode="stretch_width",
                ),
                pn.pane.HoloViews(
                    self._visualizer.orientation_heatmap(sim_data),
                    sizing_mode="stretch_width",
                ),
            ),
            pn.Row(
                pn.pane.HoloViews(
                    self._visualizer.seasonal_orientation_comparison(seasonal),
                    sizing_mode="stretch_width",
                ),
                pn.pane.HoloViews(
                    self._visualizer.orientation_polar_plot(sim_data),
                    sizing_mode="stretch_width",
                ),
            ),
        ])

    def _update_map(self, lat, lon, ds, dual_datasets=None):
        """Update solar map tab with interactive Datashader map and simulation panel."""
        self._map_area.clear()
        self._clicked_lat = lat
        self._clicked_lon = lon

        # Generate global grid for Datashader rendering
        global_ds = generate_global_solar_grid(
            resolution=1.0,
            lat_range=(lat - 30, lat + 30),
            lon_range=(lon - 40, lon + 40),
        )

        # Interactive map with tap stream
        ghi_grid = global_ds["GHI"].values
        lats = global_ds.coords["lat"].values
        lons = global_ds.coords["lon"].values
        interactive_map, tap_stream = self._visualizer.interactive_map_with_tap(
            lats, lons, ghi_grid,
        )

        # React to map clicks -- update clicked location
        def on_tap(x, y):
            if x != 0 or y != 0:
                self._clicked_lat = y
                self._clicked_lon = x
                self._sim_result_area.clear()
                self._sim_result_area.append(
                    pn.pane.Markdown(
                        f"**Selected: ({y:.2f}, {x:.2f})** -- "
                        f"Click 'Simulate at Location' to run analysis"
                    )
                )

        tap_stream.add_subscriber(on_tap)

        # Build GHI annotation from available sources
        ghi_values = {}
        if dual_datasets:
            for name, src_ds in dual_datasets.items():
                if "ALLSKY_SFC_SW_DWN" in src_ds.data_vars:
                    ghi_values[name] = float(src_ds["ALLSKY_SFC_SW_DWN"].mean())

        # Source annotation on map
        if ghi_values:
            label_parts = [f"{src}: {ghi:.2f}" for src, ghi in ghi_values.items()]
            label = " | ".join(label_parts)
            source_info = pn.pane.HTML(
                f'<div style="background:#1a1a2e; color:white; padding:8px 16px; '
                f'border-radius:8px; font-size:14px;">'
                f'GHI at ({lat:.2f}, {lon:.2f}): {label}</div>'
            )
        else:
            source_info = pn.pane.HTML("")

        # Simulation controls sidebar
        sim_controls = pn.Column(
            pn.pane.Markdown("### Interactive Simulation"),
            pn.pane.Markdown("*Click on the map to pick a location, then simulate:*"),
            self._sim_tilt,
            self._sim_azimuth,
            self._sim_panels,
            self._sim_run_btn,
            pn.layout.Divider(),
            self._sim_result_area,
            width=300,
        )

        self._map_area.extend([
            source_info,
            pn.Row(
                pn.pane.HoloViews(interactive_map, sizing_mode="stretch_width"),
                sim_controls,
            ),
        ])

    def _update_financial(self, lifetime_df, fin_summary):
        """Update financial tab."""
        self._financial_area.clear()

        sym = self.financial_config.currency_symbol
        inv = fin_summary["investment"]
        ret = fin_summary["returns"]
        env = fin_summary["environmental"]

        summary_html = f"""
        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 16px;">
            <div style="background: #FFF3E0; padding: 16px; border-radius: 8px;
                        flex: 1; min-width: 200px;">
                <h4>Investment</h4>
                <p>System: {sym}{inv['system_cost']:,.0f}<br>
                   Subsidy: -{sym}{inv['incentive']:,.0f}<br>
                   <strong>Net: {sym}{inv['net_cost']:,.0f}</strong></p>
            </div>
            <div style="background: #E8F5E9; padding: 16px; border-radius: 8px;
                        flex: 1; min-width: 200px;">
                <h4>Returns</h4>
                <p>First Year: {sym}{ret['first_year_savings']:,.0f}<br>
                   Payback: {ret['payback_years']} years<br>
                   <strong>ROI: {ret['roi_pct']}%</strong></p>
            </div>
            <div style="background: #E3F2FD; padding: 16px; border-radius: 8px;
                        flex: 1; min-width: 200px;">
                <h4>Environmental</h4>
                <p>CO\u2082/year: {env['annual_co2_offset_kg']:,.0f} kg<br>
                   Lifetime: {env['lifetime_co2_offset_tonnes']:.1f} tonnes<br>
                   <strong>Trees: {env['equivalent_trees']}</strong></p>
            </div>
        </div>
        """

        self._financial_area.extend([
            pn.pane.HTML(summary_html),
            pn.Row(
                pn.pane.HoloViews(
                    self._visualizer.payback_timeline(lifetime_df, currency_symbol=sym),
                    sizing_mode="stretch_width",
                ),
                pn.pane.HoloViews(
                    self._visualizer.carbon_savings_bar(lifetime_df),
                    sizing_mode="stretch_width",
                ),
            ),
        ])

    def _run_map_simulation(self, event=None):
        """Run quick simulation at clicked map location with current slider values."""
        lat = self._clicked_lat or self.location.latitude
        lon = self._clicked_lon or self.location.longitude
        tilt = self._sim_tilt.value
        azimuth = self._sim_azimuth.value
        n_panels = self._sim_panels.value
        panel_area = self.panel_config.panel_area
        efficiency = self.panel_config.panel_efficiency

        self._sim_result_area.clear()
        self._sim_result_area.append(
            pn.pane.Markdown(f"**Simulating at ({lat:.2f}, {lon:.2f})...**")
        )

        try:
            # Generate data for location
            ds = generate_synthetic_solar_data(lat=lat, lon=lon, start_year=2023, end_year=2023)
            ghi_year = ds["ALLSKY_SFC_SW_DWN"].values[:365]
            avg_temp = float(ds["T2M"].mean())

            # Quick orientation simulation
            simulator = OrientationSimulator(
                latitude=lat, longitude=lon,
                panel_efficiency=efficiency,
                panel_area=n_panels * panel_area,
                system_losses=self.panel_config.system_losses,
                tilt_angles=[tilt],
                azimuths={"Selected": azimuth},
            )

            sim_data = simulator.simulate_all_orientations(ghi_year, year=2023)
            if not sim_data.empty:
                energy_kwh = float(sim_data["annual_energy_kwh"].iloc[0])
            else:
                energy_kwh = 0

            # Quick financial calc
            daily_ghi = float(np.mean(ghi_year))
            annual_gen = daily_ghi * 365 * efficiency * n_panels * panel_area * (1 - self.panel_config.system_losses)
            sym = self.financial_config.currency_symbol
            monthly_savings = annual_gen * self.financial_config.electricity_rate / 12
            co2_offset = annual_gen * 0.42  # kg

            # Direction name
            dir_names = {0: "North", 45: "NE", 90: "East", 135: "SE",
                         180: "South", 225: "SW", 270: "West", 315: "NW"}
            dir_name = dir_names.get(azimuth, f"{azimuth}deg")

            result_html = f"""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border-radius: 12px; padding: 20px; color: white;">
                <h3 style="margin:0 0 12px 0; color: #FFB900;">
                    Simulation: ({lat:.2f}, {lon:.2f})
                </h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 12px;">
                        <div style="font-size: 12px; opacity: 0.7;">Avg Daily GHI</div>
                        <div style="font-size: 24px; font-weight: bold; color: #FFB900;">
                            {daily_ghi:.2f} <span style="font-size:12px">kWh/m2/day</span>
                        </div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 12px;">
                        <div style="font-size: 12px; opacity: 0.7;">Annual Generation</div>
                        <div style="font-size: 24px; font-weight: bold; color: #4CAF50;">
                            {annual_gen:,.0f} <span style="font-size:12px">kWh/yr</span>
                        </div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 12px;">
                        <div style="font-size: 12px; opacity: 0.7;">Monthly Savings</div>
                        <div style="font-size: 24px; font-weight: bold; color: #2196F3;">
                            {sym}{monthly_savings:,.0f} <span style="font-size:12px">/month</span>
                        </div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 12px;">
                        <div style="font-size: 12px; opacity: 0.7;">CO2 Offset</div>
                        <div style="font-size: 24px; font-weight: bold; color: #66BB6A;">
                            {co2_offset:,.0f} <span style="font-size:12px">kg/yr</span>
                        </div>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 13px; opacity: 0.8;">
                    Config: {n_panels} panels x {tilt}deg tilt x {dir_name} |
                    Temp: {avg_temp:.1f}C
                </div>
            </div>
            """

            # Monthly energy profile chart
            estimator = EnergyEstimator(
                panel_efficiency=efficiency,
                panel_area=panel_area,
                num_panels=n_panels,
                system_losses=self.panel_config.system_losses,
            )
            monthly_energy = estimator.estimate_monthly_energy(ds)
            energy_chart = self._visualizer.energy_projection_area(monthly_energy)

            self._sim_result_area.clear()
            self._sim_result_area.extend([
                pn.pane.HTML(result_html),
                pn.pane.HoloViews(energy_chart, sizing_mode="stretch_width", height=250),
            ])
            self._notify(f"Simulation complete for ({lat:.2f}, {lon:.2f})", "success")

        except Exception as e:
            logger.exception("Map simulation failed")
            self._sim_result_area.clear()
            self._sim_result_area.append(
                pn.pane.Markdown(f"**Error:** {e}")
            )
            self._notify(f"Simulation failed: {e}", "error")

    def _update_dual_source(self, dual_datasets, lat, lon):
        """Update dual-source cross-validation tab."""
        self._dual_source_area.clear()

        if len(dual_datasets) < 2:
            self._dual_source_area.append(
                pn.pane.Markdown(
                    "### Single Source Mode\n\n"
                    "Enable **ERA5 (CDS API)** in the sidebar and re-run analysis "
                    "to compare NASA POWER and ERA5 data side by side.\n\n"
                    "ERA5 requires a free Copernicus CDS account. "
                    "[Register here](https://cds.climate.copernicus.eu)"
                )
            )
            return

        analyzer = DualSourceAnalyzer(
            datasets=dual_datasets, latitude=lat, longitude=lon,
        )

        # Cross-validation stats
        aligned = analyzer.compare_daily_ghi()
        monthly = analyzer.compare_monthly()
        report = analyzer.agreement_report()

        self._dual_source_area.extend([
            pn.pane.Markdown(report),
            pn.Row(
                pn.pane.HoloViews(
                    self._visualizer.dual_source_timeseries(aligned),
                    sizing_mode="stretch_width",
                ),
                pn.pane.HoloViews(
                    self._visualizer.dual_source_scatter(aligned),
                    sizing_mode="stretch_width",
                ),
            ),
            pn.Row(
                pn.pane.HoloViews(
                    self._visualizer.dual_source_monthly_bar(monthly),
                    sizing_mode="stretch_width",
                ),
                pn.pane.HoloViews(
                    self._visualizer.dual_source_difference_heatmap(aligned),
                    sizing_mode="stretch_width",
                ),
            ),
        ])

    def _update_ai(self, solar, energy, financial, orientation):
        """Update AI insights tab with report and chat interface."""
        self._ai_area.clear()

        sym = self.financial_config.currency_symbol
        code = self.financial_config.currency

        report = self._ai.generate_report(
            solar, energy, financial, orientation,
            currency_symbol=sym, currency_code=code,
        )

        # Store analysis context for chat
        self._chat_solar = solar
        self._chat_energy = energy
        self._chat_financial = financial

        # Chat input widgets
        chat_input = pn.widgets.TextAreaInput(
            name="Ask about your solar analysis",
            placeholder="e.g., How many ACs can I run? What if I add 5 more panels? Is rooftop solar worth it in my area?",
            height=80, width=600,
        )
        chat_btn = pn.widgets.Button(
            name="Ask AI", button_type="success", width=120,
        )
        chat_output = pn.Column(
            pn.pane.Markdown("*Ask a question about your solar data...*"),
        )

        def on_chat(event):
            question = chat_input.value.strip()
            if not question:
                return
            chat_output.clear()
            chat_output.append(pn.pane.Markdown("**Thinking...**"))
            try:
                answer = self._ai.chat_query(
                    question,
                    solar_summary=self._chat_solar,
                    energy_summary=self._chat_energy,
                    financial_summary=self._chat_financial,
                    currency_symbol=sym,
                    currency_code=code,
                )
                chat_output.clear()
                chat_output.extend([
                    pn.pane.Markdown(f"**Q:** {question}"),
                    pn.pane.Markdown(answer),
                    pn.layout.Divider(),
                ])
            except Exception as e:
                chat_output.clear()
                chat_output.append(pn.pane.Markdown(f"**Error:** {e}"))

        chat_btn.on_click(on_chat)

        self._ai_area.extend([
            pn.pane.Markdown(report),
            pn.layout.Divider(),
            pn.pane.Markdown("## Ask AI About Your Solar Setup"),
            pn.Row(chat_input, chat_btn),
            chat_output,
        ])

    def view(self) -> pn.template.FastListTemplate:
        """Build and return the complete dashboard template.

        Returns
        -------
        pn.template.FastListTemplate
            The fully configured dashboard ready for serving.
        """
        sidebar = pn.Column(
            self.location.panel,
            pn.layout.Divider(),
            self.panel_config.panel,
            pn.layout.Divider(),
            self.financial_config.panel,
            pn.layout.Divider(),
            self._use_synthetic,
            self._use_era5,
            self._analyze_btn,
            pn.layout.Divider(),
            "### Multi-Location",
            self._compare_cities_input,
            self._compare_btn,
            pn.layout.Divider(),
            self._theme_toggle,
            self._loading,
            self._status,
            width=280,
        )

        tabs = pn.Tabs(
            ("Overview", pn.Column(self._kpi_row, self._overview_area)),
            ("Orientation Analysis", self._orientation_area),
            ("Solar Map", self._map_area),
            ("Financial", self._financial_area),
            ("Multi-Location", self._multi_location_area),
            ("Data Sources", self._dual_source_area),
            ("AI Insights", self._ai_area),
            dynamic=True,
        )

        self._template = pn.template.FastListTemplate(
            title="Solar Potential Intelligence Platform",
            sidebar=[sidebar],
            main=[tabs],
            accent_base_color="#FFB900",
            header_background="#1a1a2e",
        )

        return self._template


def main():
    """Entry point for the Solar Intelligence dashboard."""
    pn.extension(sizing_mode="stretch_width", notifications=True)
    dashboard = SolarDashboard()
    template = dashboard.view()
    template.servable()
    return template


# Allow `panel serve panel_dashboard.py`
if __name__ == "__main__" or __name__.startswith("bokeh"):
    main()
