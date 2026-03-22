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
from solar_intelligence.config import CB_PALETTE, CARBON_FACTOR_KG_PER_KWH
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

        # --- AI Settings widgets ---
        from solar_intelligence.ai_engine import LLM_PROVIDERS
        self._ai_provider = pn.widgets.Select(
            name="AI Provider",
            options={"OpenAI": "openai", "Groq (Free)": "groq", "Gemini (Free)": "gemini"},
            value="groq", width=250,
        )
        self._ai_model = pn.widgets.Select(
            name="Model",
            options=LLM_PROVIDERS["groq"]["models"],
            value=LLM_PROVIDERS["groq"]["default"],
            width=250,
        )
        self._ai_api_key = pn.widgets.PasswordInput(
            name="API Key",
            placeholder="Paste your API key here",
            width=250,
        )
        self._ai_provider.param.watch(self._on_provider_change, "value")

        # Template reference (set in view())
        self._template = None

    def _on_provider_change(self, event):
        """Update model list when AI provider changes."""
        from solar_intelligence.ai_engine import LLM_PROVIDERS
        provider = event.new
        info = LLM_PROVIDERS.get(provider, LLM_PROVIDERS["openai"])
        self._ai_model.options = info["models"]
        self._ai_model.value = info["default"]

    def _apply_ai_settings(self):
        """Apply current AI settings to the engine."""
        provider = self._ai_provider.value
        model = self._ai_model.value
        api_key = self._ai_api_key.value.strip()
        self._ai.provider = provider
        self._ai.llm_model = model
        if api_key:
            self._ai.api_key = api_key
            self._ai.mode = "llm"
        else:
            self._ai.mode = "template"

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

            if lat == 0.0 and lon == 0.0 and not self.location.location_name:
                self._status.object = "**Please enter a city name or coordinates first.**"
                self._notify("Enter a location before analyzing", "warning")
                return

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
            # Compute latitude-aware tilt range
            optimal_tilt = int(abs(lat))
            tilt_min = max(0, optimal_tilt - 20)
            tilt_max = min(90, optimal_tilt + 25)
            tilt_angles = sorted(set([0] + list(range(tilt_min, tilt_max + 1, 15)) + [90]))

            simulator = OrientationSimulator(
                latitude=lat, longitude=lon,
                panel_efficiency=self.panel_config.panel_efficiency,
                panel_area=estimator.total_area,
                system_losses=self.panel_config.system_losses,
                tilt_angles=tilt_angles,
                azimuths={
                    "North": 0, "East": 90, "South": 180, "West": 270,
                    "South-East": 135, "South-West": 225,
                },
            )

            # Use the last full year available in the dataset
            years = ds.time.dt.year.values
            last_full_year = int(max(years))

            ghi_year = ds["ALLSKY_SFC_SW_DWN"].sel(
                time=slice(f"{last_full_year}-01-01", f"{last_full_year}-12-31"),
            ).values
            if len(ghi_year) < 365:
                ghi_year = ds["ALLSKY_SFC_SW_DWN"].values[:365]

            sim_data = simulator.simulate_all_orientations(ghi_year, year=last_full_year)
            optimal = simulator.optimal_orientation(ghi_year, year=last_full_year)
            sensitivity = simulator.tilt_sensitivity_analysis(ghi_year, year=last_full_year)
            daily_profile = simulator.daily_profile_by_orientation(
                ghi_year, date=f"{last_full_year}-06-21",
            )
            seasonal = simulator.seasonal_comparison(ghi_year, year=last_full_year)

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
                    era5_ds = era5_client.fetch_daily(lat, lon, last_full_year, last_full_year)
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

        except ConnectionError as e:
            self._status.object = "**Network error.** Check your internet connection."
            self._notify("Network error - check internet connection", "error")
        except ValueError as e:
            logger.exception("Analysis failed with ValueError")
            self._status.object = f"**Invalid input:** {e}"
            self._notify(f"Invalid input: {e}", "error")
        except Exception as e:
            logger.exception("Analysis failed")
            self._status.object = "**Unexpected error.** Check logs for details."
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
                CB_PALETTE[1],
            ),
            KPICard.create(
                "Annual Energy",
                f"{energy['production']['annual_energy_kwh']:,.0f}",
                "kWh/year",
                CB_PALETTE[2],
            ),
            KPICard.create(
                "Capacity Factor",
                f"{energy['performance']['capacity_factor_pct']:.1f}%",
                "System efficiency",
                CB_PALETTE[5],
            ),
            KPICard.create(
                "Payback",
                f"{financial['returns']['payback_years']}",
                "years",
                CB_PALETTE[4],
            ),
            KPICard.create(
                "CO\u2082 Offset",
                f"{financial['environmental']['annual_co2_offset_kg']:,.0f}",
                "kg/year",
                CB_PALETTE[2],
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

    def _build_bokeh_map(self, ghi_grid, lats, lons):
        """Build a Bokeh figure with heatmap + coastlines rendered directly."""
        from bokeh.plotting import figure
        from bokeh.models import LinearColorMapper, ColorBar, BasicTicker
        from bokeh.palettes import Inferno256

        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())

        mapper = LinearColorMapper(
            palette=Inferno256,
            low=float(ghi_grid.min()),
            high=float(ghi_grid.max()),
        )

        fig = figure(
            title="Click to Select Location",
            x_axis_label="Longitude", y_axis_label="Latitude",
            x_range=(lon_min, lon_max), y_range=(lat_min, lat_max),
            width=900, height=500,
            tools=["pan", "wheel_zoom", "box_zoom", "reset", "tap", "hover"],
            active_scroll="wheel_zoom",
        )

        fig.image(
            image=[ghi_grid], x=lon_min, y=lat_min,
            dw=lon_max - lon_min, dh=lat_max - lat_min,
            color_mapper=mapper,
            level="image",
        )

        color_bar = ColorBar(
            color_mapper=mapper, ticker=BasicTicker(),
            label_standoff=12, border_line_color=None, location=(0, 0),
            title="GHI (kWh/m²/day)",
        )
        fig.add_layout(color_bar, "right")

        # Add coastlines directly using Bokeh plotting API
        try:
            import cartopy.feature as cfeature_local

            def _extract(feature):
                xs, ys = [], []
                for geom in feature.geometries():
                    geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
                    for g in geoms:
                        c = np.array(g.coords)
                        if len(c) > 1:
                            xs.append(c[:, 0].tolist())
                            ys.append(c[:, 1].tolist())
                return xs, ys

            cxs, cys = _extract(cfeature_local.COASTLINE)
            fig.multi_line(
                cxs, cys,
                line_color="white", line_width=1.0, line_alpha=0.8,
                level="overlay",
            )
            bxs, bys = _extract(cfeature_local.BORDERS)
            fig.multi_line(
                bxs, bys,
                line_color="gray", line_width=0.4, line_alpha=0.5,
                line_dash="dotted", level="overlay",
            )
        except ImportError:
            pass

        # Marker for clicked location (initially hidden)
        from bokeh.models import ColumnDataSource as CDS
        marker_src = CDS(data={"x": [], "y": []})
        fig.triangle("x", "y", source=marker_src, size=18, color="red",
                     line_color="black", line_width=1)

        return fig, marker_src

    def _update_map(self, lat, lon, ds, dual_datasets=None):
        """Update solar map tab with interactive Bokeh map and simulation panel."""
        self._map_area.clear()
        self._clicked_lat = lat
        self._clicked_lon = lon

        # Generate global grid
        global_ds = generate_global_solar_grid(
            resolution=1.0,
            lat_range=(-90, 90),
            lon_range=(-180, 180),
        )

        ghi_grid = global_ds["GHI"].values
        lats = global_ds.coords["lat"].values
        lons = global_ds.coords["lon"].values

        # Build Bokeh figure with coastlines baked in
        bokeh_fig, marker_src = self._build_bokeh_map(ghi_grid, lats, lons)
        map_pane = pn.pane.Bokeh(bokeh_fig, sizing_mode="stretch_width")

        # Handle tap events via Bokeh callback
        from bokeh.models import TapTool, CustomJS

        tap_callback = CustomJS(args=dict(source=marker_src), code="""
            const {x, y} = cb_obj;
            source.data = {x: [x], y: [y]};
            source.change.emit();
        """)
        bokeh_fig.js_on_event("tap", tap_callback)

        # Python-side tap handling via Panel event
        def on_map_tap(event):
            if hasattr(event, "x") and hasattr(event, "y"):
                self._clicked_lat = event.y
                self._clicked_lon = event.x
                self._sim_result_area.clear()
                self._sim_result_area.append(
                    pn.pane.Markdown(
                        f"**Selected: ({event.y:.2f}, {event.x:.2f})** -- "
                        f"Click 'Simulate at Location' to run analysis"
                    )
                )

        bokeh_fig.on_event("tap", on_map_tap)

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
                map_pane,
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
            from solar_intelligence.config import default_end_year
            sim_year = default_end_year()
            ds = generate_synthetic_solar_data(lat=lat, lon=lon, start_year=sim_year, end_year=sim_year)
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

            sim_data = simulator.simulate_all_orientations(ghi_year, year=sim_year)
            if not sim_data.empty:
                energy_kwh = float(sim_data["annual_energy_kwh"].iloc[0])
            else:
                energy_kwh = 0

            # Quick financial calc
            daily_ghi = float(np.mean(ghi_year))
            annual_gen = daily_ghi * 365 * efficiency * n_panels * panel_area * (1 - self.panel_config.system_losses)
            sym = self.financial_config.currency_symbol
            monthly_savings = annual_gen * self.financial_config.electricity_rate / 12
            co2_offset = annual_gen * CARBON_FACTOR_KG_PER_KWH

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
        self._apply_ai_settings()

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
            self._apply_ai_settings()
            if self._ai.mode != "llm":
                chat_output.clear()
                chat_output.append(pn.pane.Markdown(
                    "**Add an API key in the sidebar under AI Settings to enable chat.**\n\n"
                    "Free options: [Groq](https://console.groq.com/keys) or "
                    "[Gemini](https://aistudio.google.com/apikey)"
                ))
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
            "### AI Settings",
            self._ai_provider,
            self._ai_model,
            self._ai_api_key,
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
