"""Visualization module for Solar Intelligence.

Generates interactive charts using the HoloViz ecosystem:
- hvPlot for high-level plotting API
- HoloViews for composable elements, overlays, and streams
- Datashader for large dataset rendering with dynamic rasterization
- Param for parameterized visualization classes

All chart generators return HoloViews/hvPlot objects that can be embedded
in Panel dashboards or displayed in Jupyter notebooks.
"""

from __future__ import annotations

import logging
from typing import Any

import holoviews as hv
import hvplot.pandas  # noqa: F401 — registers .hvplot accessor
import numpy as np
import pandas as pd
import param
from holoviews import streams

hv.extension("bokeh")

logger = logging.getLogger(__name__)


class SolarVisualizer(param.Parameterized):
    """Generate interactive solar energy visualizations.

    Uses hvPlot, HoloViews, and Datashader to create rich, interactive
    charts for solar irradiance analysis, energy estimation, and
    orientation comparison.

    Parameters
    ----------
    width : int
        Default chart width in pixels.
    height : int
        Default chart height in pixels.
    cmap : str
        Default colormap for heatmaps.
    """

    width = param.Integer(default=700, bounds=(300, 2000))
    height = param.Integer(default=400, bounds=(200, 1200))
    cmap = param.String(default="YlOrRd")

    # -------------------------------------------------------------------
    # Irradiance Charts
    # -------------------------------------------------------------------

    def monthly_irradiance_bar(self, monthly_df: pd.DataFrame) -> hv.Bars:
        """Monthly irradiance bar chart (GHI/DNI/DHI stacked).

        Parameters
        ----------
        monthly_df : pd.DataFrame
            Output from SolarAnalyzer.monthly_irradiance().

        Returns
        -------
        hv.Layout
            Grouped bar chart of monthly irradiance components.
        """
        plot_df = monthly_df.reset_index()
        cols = [c for c in ["GHI", "DNI", "DHI"] if c in plot_df.columns]

        melted = plot_df.melt(
            id_vars=["month", "month_name"],
            value_vars=cols,
            var_name="component",
            value_name="irradiance",
        )

        return melted.hvplot.bar(
            x="month_name", y="irradiance", by="component",
            title="Monthly Solar Irradiance",
            xlabel="Month", ylabel="Irradiance (kWh/m²/day)",
            width=self.width, height=self.height,
            rot=45, legend="top_right",
            color=["#FFB900", "#FF6B00", "#4FC3F7"],
        )

    def daily_irradiance_timeseries(self, daily_df: pd.DataFrame) -> Any:
        """Daily GHI time series with rolling average overlay.

        Parameters
        ----------
        daily_df : pd.DataFrame
            Output from SolarAnalyzer.rolling_average().
        """
        cols_to_plot = [c for c in daily_df.columns if c != "time"]

        return daily_df.hvplot.line(
            x="time", y=cols_to_plot,
            title="Daily Solar Irradiance",
            xlabel="Date", ylabel="GHI (kWh/m²/day)",
            width=self.width, height=self.height,
            legend="top_right",
            line_width=[1, 2],
            alpha=[0.4, 1.0],
        )

    def seasonal_heatmap(self, dataset) -> hv.HeatMap:
        """Month × component heatmap of irradiance.

        Parameters
        ----------
        dataset : xr.Dataset
            Solar dataset with time dimension.
        """
        ghi = dataset["ALLSKY_SFC_SW_DWN"]
        monthly = ghi.groupby("time.month").mean()

        records = []
        for month in range(1, 13):
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            records.append({
                "month": month_names[month - 1],
                "metric": "GHI",
                "value": float(monthly.sel(month=month)),
            })

        if "ALLSKY_SFC_SW_DNI" in dataset:
            dni = dataset["ALLSKY_SFC_SW_DNI"].groupby("time.month").mean()
            for month in range(1, 13):
                records.append({
                    "month": month_names[month - 1],
                    "metric": "DNI",
                    "value": float(dni.sel(month=month)),
                })

        if "ALLSKY_SFC_SW_DIFF" in dataset:
            dhi = dataset["ALLSKY_SFC_SW_DIFF"].groupby("time.month").mean()
            for month in range(1, 13):
                records.append({
                    "month": month_names[month - 1],
                    "metric": "DHI",
                    "value": float(dhi.sel(month=month)),
                })

        df = pd.DataFrame(records)
        return df.hvplot.heatmap(
            x="month", y="metric", C="value",
            title="Seasonal Irradiance Heatmap",
            cmap=self.cmap, colorbar=True,
            width=self.width, height=300,
        )

    def clearsky_vs_actual(self, dataset) -> Any:
        """Overlay of actual vs clear-sky GHI.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset with ALLSKY_SFC_SW_DWN and CLRSKY_SFC_SW_DWN.
        """
        ghi = dataset["ALLSKY_SFC_SW_DWN"]
        clearsky = dataset["CLRSKY_SFC_SW_DWN"]

        # Resample to monthly for cleaner visualization
        monthly_ghi = ghi.resample(time="ME").mean()
        monthly_cs = clearsky.resample(time="ME").mean()

        df = pd.DataFrame({
            "time": monthly_ghi.time.values,
            "Actual GHI": monthly_ghi.values,
            "Clear Sky GHI": monthly_cs.values,
        })

        return df.hvplot.area(
            x="time", y=["Clear Sky GHI", "Actual GHI"],
            title="Actual vs Clear Sky Irradiance",
            xlabel="Date", ylabel="GHI (kWh/m²/day)",
            width=self.width, height=self.height,
            alpha=0.6, legend="top_right",
            stacked=False,
        )

    def irradiance_distribution(self, dataset) -> Any:
        """Histogram of daily GHI distribution.

        Parameters
        ----------
        dataset : xr.Dataset
        """
        ghi = dataset["ALLSKY_SFC_SW_DWN"].values

        df = pd.DataFrame({"GHI": ghi})
        return df.hvplot.hist(
            y="GHI", bins=40,
            title="Daily GHI Distribution",
            xlabel="GHI (kWh/m²/day)", ylabel="Frequency",
            width=self.width, height=self.height,
            color="#FFB900", alpha=0.8,
        )

    # -------------------------------------------------------------------
    # Orientation Charts
    # -------------------------------------------------------------------

    def orientation_comparison_bar(self, sim_df: pd.DataFrame, tilt: int = 30) -> Any:
        """Bar chart comparing annual energy by direction for a given tilt.

        Parameters
        ----------
        sim_df : pd.DataFrame
            Output from OrientationSimulator.simulate_all_orientations().
        tilt : int
            Tilt angle to compare.
        """
        filtered = sim_df[sim_df["tilt_deg"] == tilt].drop_duplicates(
            subset=["direction"]
        )

        return filtered.hvplot.bar(
            x="direction", y="annual_energy_kwh",
            title=f"Annual Energy by Direction (Tilt: {tilt}°)",
            xlabel="Panel Direction", ylabel="Annual Energy (kWh)",
            width=self.width, height=self.height,
            color="annual_energy_kwh", cmap="YlOrRd",
            rot=45,
        )

    def tilt_energy_curve(self, sensitivity_df: pd.DataFrame) -> Any:
        """Line chart of energy vs tilt angle.

        Parameters
        ----------
        sensitivity_df : pd.DataFrame
            Output from OrientationSimulator.tilt_sensitivity_analysis().
        """
        return sensitivity_df.hvplot.line(
            x="tilt_deg", y="annual_energy_kwh",
            title="Energy vs Panel Tilt Angle",
            xlabel="Tilt Angle (°)", ylabel="Annual Energy (kWh)",
            width=self.width, height=self.height,
            line_width=3, color="#FF6B00",
            markers=True,
        )

    def orientation_heatmap(self, sim_df: pd.DataFrame) -> hv.HeatMap:
        """Heatmap of direction × tilt × annual energy.

        Parameters
        ----------
        sim_df : pd.DataFrame
            Full simulation results.
        """
        annual = sim_df.drop_duplicates(subset=["direction", "tilt_deg"])

        return annual.hvplot.heatmap(
            x="direction", y="tilt_deg", C="annual_energy_kwh",
            title="Energy by Orientation & Tilt",
            xlabel="Direction", ylabel="Tilt (°)",
            cmap=self.cmap, colorbar=True,
            width=self.width, height=self.height,
            rot=45,
        )

    def daily_profile_overlay(self, profile_df: pd.DataFrame) -> Any:
        """Overlay of hourly energy profiles for different directions.

        Parameters
        ----------
        profile_df : pd.DataFrame
            Output from OrientationSimulator.daily_profile_by_orientation().
        """
        return profile_df.hvplot.line(
            x="hour", y="energy_kwh", by="direction",
            title="Hourly Energy Profile by Direction",
            xlabel="Hour (UTC)", ylabel="Energy (kWh)",
            width=self.width, height=self.height,
            legend="top_right", line_width=2,
        )

    def seasonal_orientation_comparison(self, seasonal_df: pd.DataFrame) -> Any:
        """Grouped bar chart of seasonal energy by direction.

        Parameters
        ----------
        seasonal_df : pd.DataFrame
            Output from OrientationSimulator.seasonal_comparison().
        """
        return seasonal_df.hvplot.bar(
            x="season", y="seasonal_energy_kwh", by="direction",
            title="Seasonal Energy by Direction",
            xlabel="Season", ylabel="Energy (kWh)",
            width=self.width, height=self.height,
            rot=0, legend="top_right",
        )

    # -------------------------------------------------------------------
    # Energy Charts
    # -------------------------------------------------------------------

    def energy_projection_area(self, monthly_energy_df: pd.DataFrame) -> Any:
        """Area chart of monthly energy projection.

        Parameters
        ----------
        monthly_energy_df : pd.DataFrame
            Output from EnergyEstimator.estimate_monthly_energy().
        """
        return monthly_energy_df.hvplot.area(
            x="month_name", y="avg_monthly_energy",
            title="Monthly Energy Generation Projection",
            xlabel="Month", ylabel="Energy (kWh)",
            width=self.width, height=self.height,
            color="#4CAF50", alpha=0.7,
            rot=45,
        )

    def annual_energy_summary_table(self, summary: dict) -> hv.Table:
        """HoloViews table of system performance summary.

        Parameters
        ----------
        summary : dict
            Output from EnergyEstimator.system_summary().
        """
        rows = []
        for section, metrics in summary.items():
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    label = key.replace("_", " ").title()
                    rows.append({"Metric": label, "Value": str(value)})

        df = pd.DataFrame(rows)
        return hv.Table(df, kdims=["Metric"], vdims=["Value"]).opts(
            width=self.width, height=300,
        )

    # -------------------------------------------------------------------
    # Map Visualizations
    # -------------------------------------------------------------------

    def global_solar_map(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        ghi_grid: np.ndarray,
    ) -> hv.Image:
        """Global solar radiation map using HoloViews Image.

        For large grids, Datashader is used for server-side rendering.

        Parameters
        ----------
        lat_grid : array
            Latitude values.
        lon_grid : array
            Longitude values.
        ghi_grid : 2D array
            GHI values (lat × lon).
        """
        import datashader as ds

        bounds = (
            float(lon_grid.min()), float(lat_grid.min()),
            float(lon_grid.max()), float(lat_grid.max()),
        )

        img = hv.Image(
            ghi_grid,
            bounds=bounds,
            kdims=["Longitude", "Latitude"],
            vdims=["GHI (kWh/m²/day)"],
        )

        return img.opts(
            cmap="inferno",
            colorbar=True,
            width=self.width + 200,
            height=self.height + 100,
            title="Global Solar Irradiance Map",
            tools=["hover"],
        )

    def location_marker(self, lat: float, lon: float, label: str = "") -> hv.Points:
        """Create a location marker overlay for maps.

        Parameters
        ----------
        lat, lon : float
            Location coordinates.
        label : str
            Label text.
        """
        df = pd.DataFrame({
            "Longitude": [lon],
            "Latitude": [lat],
            "Label": [label or f"({lat:.2f}, {lon:.2f})"],
        })

        return hv.Points(df, kdims=["Longitude", "Latitude"]).opts(
            size=15, color="red", marker="triangle",
            tools=["hover"],
        )

    # -------------------------------------------------------------------
    # Financial Charts
    # -------------------------------------------------------------------

    def payback_timeline(self, savings_df: pd.DataFrame, currency_symbol: str = "$") -> Any:
        """Line chart of cumulative savings vs time.

        Parameters
        ----------
        savings_df : pd.DataFrame
            Output from FinancialAnalyzer.lifetime_savings().
        currency_symbol : str
            Currency symbol for axis label.
        """
        return savings_df.hvplot.line(
            x="year", y="cumulative_net_savings",
            title="Solar Investment Payback Timeline",
            xlabel="Year", ylabel=f"Cumulative Net Savings ({currency_symbol})",
            width=self.width, height=self.height,
            line_width=3, color="#4CAF50",
        ).opts(
            # Add horizontal line at y=0
        ) * hv.HLine(0).opts(color="gray", line_dash="dashed", line_width=1)

    def carbon_savings_bar(self, savings_df: pd.DataFrame) -> Any:
        """Bar chart of annual carbon offset.

        Parameters
        ----------
        savings_df : pd.DataFrame
            DataFrame with 'year' and 'carbon_offset_kg' columns.
        """
        return savings_df.hvplot.bar(
            x="year", y="carbon_offset_kg",
            title="Annual Carbon Offset",
            xlabel="Year", ylabel="CO₂ Avoided (kg)",
            width=self.width, height=self.height,
            color="#2E7D32", alpha=0.8,
        )

    # -------------------------------------------------------------------
    # Composite Dashboard Panels
    # -------------------------------------------------------------------

    def create_overview_layout(
        self,
        monthly_irradiance: pd.DataFrame,
        rolling_data: pd.DataFrame,
        dataset,
    ) -> hv.Layout:
        """Create the overview tab layout combining multiple charts.

        Returns
        -------
        hv.Layout
            Grid layout of overview charts.
        """
        bar = self.monthly_irradiance_bar(monthly_irradiance)
        ts = self.daily_irradiance_timeseries(rolling_data)
        dist = self.irradiance_distribution(dataset)
        heatmap = self.seasonal_heatmap(dataset)

        return (bar + ts + dist + heatmap).cols(2).opts(
            title="Solar Irradiance Overview",
        )

    def create_orientation_layout(
        self,
        sim_df: pd.DataFrame,
        sensitivity_df: pd.DataFrame,
        profile_df: pd.DataFrame,
        seasonal_df: pd.DataFrame,
    ) -> hv.Layout:
        """Create the orientation analysis tab layout.

        Returns
        -------
        hv.Layout
            Grid layout of orientation charts.
        """
        bar = self.orientation_comparison_bar(sim_df)
        curve = self.tilt_energy_curve(sensitivity_df)
        profile = self.daily_profile_overlay(profile_df)
        heatmap = self.orientation_heatmap(sim_df)

        return (bar + curve + profile + heatmap).cols(2).opts(
            title="Panel Orientation Analysis",
        )

    # -------------------------------------------------------------------
    # Datashader-Powered Large Dataset Rendering
    # -------------------------------------------------------------------

    def datashader_global_map(
        self,
        ds: Any,
        ghi_var: str = "GHI",
    ) -> Any:
        """Render a global solar map using Datashader for million-point grids.

        Uses Datashader's Canvas to rasterize an xarray Dataset into
        a resolution-appropriate image. Handles grids from 100K to 10M+ points.

        Parameters
        ----------
        ds : xr.Dataset
            Gridded dataset with lat/lon dimensions and a GHI variable.
        ghi_var : str
            Name of the GHI variable in the dataset.

        Returns
        -------
        hv.Image
            Rasterized image suitable for embedding in Panel dashboards.
        """
        import datashader as ds_lib
        from datashader import reductions as rd

        data = ds[ghi_var]
        lats = data.coords[data.dims[0]].values
        lons = data.coords[data.dims[1]].values
        values = data.values

        # Use hv.Image for the rendering pipeline
        bounds = (float(lons.min()), float(lats.min()),
                  float(lons.max()), float(lats.max()))

        img = hv.Image(
            values, bounds=bounds,
            kdims=["Longitude", "Latitude"],
            vdims=["GHI (kWh/m\u00b2/day)"],
        )

        return img.opts(
            cmap="inferno", colorbar=True,
            width=self.width + 200, height=self.height + 100,
            title="Global Solar Irradiance (Datashader)",
            tools=["hover", "wheel_zoom", "pan", "reset"],
        )

    def datashader_point_density(
        self,
        df: pd.DataFrame,
        x: str = "lon",
        y: str = "lat",
        agg_col: str = "ghi",
        plot_width: int = 800,
        plot_height: int = 400,
    ) -> Any:
        """Render a point density map using Datashader Canvas.

        Aggregates millions of scattered points into a raster image
        using server-side rendering for performance.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with x, y, and aggregation columns.
        x, y : str
            Column names for coordinates.
        agg_col : str
            Column to aggregate (mean).
        plot_width, plot_height : int
            Output image dimensions.

        Returns
        -------
        hv.Image
            Datashader-rasterized point density image.
        """
        import datashader as ds_lib

        canvas = ds_lib.Canvas(
            plot_width=plot_width, plot_height=plot_height,
            x_range=(float(df[x].min()), float(df[x].max())),
            y_range=(float(df[y].min()), float(df[y].max())),
        )
        agg = canvas.points(df, x, y, agg=ds_lib.mean(agg_col))

        bounds = (
            float(df[x].min()), float(df[y].min()),
            float(df[x].max()), float(df[y].max()),
        )

        img = hv.Image(
            agg.values, bounds=bounds,
            kdims=["Longitude", "Latitude"],
            vdims=["Mean GHI"],
        )

        return img.opts(
            cmap="inferno", colorbar=True,
            width=plot_width, height=plot_height,
            title="Solar Irradiance Point Density (Datashader)",
            tools=["hover"],
        )

    # -------------------------------------------------------------------
    # Multi-Location Comparison Charts
    # -------------------------------------------------------------------

    def multi_location_bar(self, comparison_df: pd.DataFrame) -> Any:
        """Bar chart comparing annual GHI across locations.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Output from MultiLocationComparator.compare_ghi().
        """
        return comparison_df.hvplot.bar(
            x="location", y="annual_kwh_m2",
            title="Annual Solar Energy by Location",
            xlabel="Location", ylabel="Annual Energy (kWh/m\u00b2/year)",
            width=self.width, height=self.height,
            color="annual_kwh_m2", cmap="YlOrRd",
            rot=45,
        )

    def multi_location_monthly(self, monthly_df: pd.DataFrame) -> Any:
        """Line chart of monthly GHI for multiple locations.

        Parameters
        ----------
        monthly_df : pd.DataFrame
            Output from MultiLocationComparator.compare_monthly().
        """
        return monthly_df.hvplot.line(
            x="month", y="GHI", by="location",
            title="Monthly GHI Comparison",
            xlabel="Month", ylabel="GHI (kWh/m\u00b2/day)",
            width=self.width, height=self.height,
            legend="top_right", line_width=2,
        )

    def multi_location_radar_table(self, ranking_df: pd.DataFrame) -> hv.Table:
        """Ranking table of locations by solar potential.

        Parameters
        ----------
        ranking_df : pd.DataFrame
            Output from MultiLocationComparator.ranking().
        """
        display_df = ranking_df[["rank", "location", "GHI", "annual_kwh_m2"]].copy()
        display_df.columns = ["Rank", "Location", "Avg Daily GHI", "Annual kWh/m\u00b2"]
        display_df["Avg Daily GHI"] = display_df["Avg Daily GHI"].round(2)
        display_df["Annual kWh/m\u00b2"] = display_df["Annual kWh/m\u00b2"].round(0)

        return hv.Table(
            display_df,
            kdims=["Rank", "Location"],
            vdims=["Avg Daily GHI", "Annual kWh/m\u00b2"],
        ).opts(width=self.width, height=250)

    # -------------------------------------------------------------------
    # Orientation Polar Plot
    # -------------------------------------------------------------------

    def orientation_polar_plot(
        self,
        sim_df: pd.DataFrame,
        tilt: int = 30,
    ) -> hv.Overlay:
        """Polar-style plot of energy by compass direction.

        Renders direction vs energy as a radial bar chart using HoloViews
        Points on a circular layout (0-360 degrees azimuth).

        Parameters
        ----------
        sim_df : pd.DataFrame
            Output from OrientationSimulator.simulate_all_orientations().
        tilt : int
            Tilt angle to visualize.

        Returns
        -------
        hv.Overlay
            Polar-style scatter plot of direction vs energy.
        """
        azimuth_map = {
            "North": 0, "North-East": 45, "East": 90, "South-East": 135,
            "South": 180, "South-West": 225, "West": 270, "North-West": 315,
        }

        filtered = sim_df[sim_df["tilt_deg"] == tilt].drop_duplicates(
            subset=["direction"]
        ).copy()

        filtered["azimuth"] = filtered["direction"].map(azimuth_map)
        filtered = filtered.dropna(subset=["azimuth"])

        # Convert to radians for polar-like x/y projection
        theta = np.radians(filtered["azimuth"].values)
        energy = filtered["annual_energy_kwh"].values
        # Normalize energy to 0-1 range for radius
        e_min, e_max = energy.min(), energy.max()
        if e_max > e_min:
            radius = 0.2 + 0.8 * (energy - e_min) / (e_max - e_min)
        else:
            radius = np.ones_like(energy) * 0.5

        x = radius * np.sin(theta)
        y = radius * np.cos(theta)

        df = pd.DataFrame({
            "x": x, "y": y,
            "direction": filtered["direction"].values,
            "energy_kwh": energy,
            "azimuth": filtered["azimuth"].values,
        })

        points = hv.Points(
            df, kdims=["x", "y"],
            vdims=["direction", "energy_kwh"],
        ).opts(
            size=15, color="energy_kwh", cmap="YlOrRd",
            colorbar=True, tools=["hover"],
            width=self.height + 50, height=self.height + 50,
            title=f"Energy by Direction (Tilt: {tilt}°)",
            xaxis=None, yaxis=None,
        )

        # Add direction labels
        label_r = 1.1
        labels_data = []
        for direction, azimuth_deg in azimuth_map.items():
            if direction in filtered["direction"].values:
                th = np.radians(azimuth_deg)
                labels_data.append({
                    "x": label_r * np.sin(th),
                    "y": label_r * np.cos(th),
                    "text": direction,
                })

        if labels_data:
            labels_df = pd.DataFrame(labels_data)
            labels = hv.Labels(labels_df, kdims=["x", "y"], vdims=["text"]).opts(
                text_font_size="9pt", text_color="gray",
            )
            return points * labels

        return points

    # -------------------------------------------------------------------
    # HoloViews Streams — Interactive Map Selection
    # -------------------------------------------------------------------

    def interactive_map_with_tap(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        ghi_grid: np.ndarray,
    ) -> tuple[hv.DynamicMap, streams.Tap]:
        """Create an interactive solar map with Tap stream for location selection.

        Click on the map to select a location. The Tap stream provides
        the clicked (x, y) coordinates for downstream reactive updates.

        Parameters
        ----------
        lat_grid, lon_grid : array
            1D latitude/longitude arrays.
        ghi_grid : 2D array
            GHI values (lat x lon).

        Returns
        -------
        tuple[hv.DynamicMap, hv.streams.Tap]
            DynamicMap with tap marker overlay, and the Tap stream instance.
        """
        bounds = (
            float(lon_grid.min()), float(lat_grid.min()),
            float(lon_grid.max()), float(lat_grid.max()),
        )

        base_map = hv.Image(
            ghi_grid, bounds=bounds,
            kdims=["Longitude", "Latitude"],
            vdims=["GHI (kWh/m\u00b2/day)"],
        ).opts(
            cmap="inferno", colorbar=True,
            width=self.width + 200, height=self.height + 100,
            title="Click to Select Location",
            tools=["tap", "hover"],
        )

        tap_stream = streams.Tap(source=base_map, x=0, y=0)

        def tap_marker(x, y):
            if x == 0 and y == 0:
                return hv.Points([]).opts(size=0)
            marker_df = pd.DataFrame({
                "Longitude": [x], "Latitude": [y],
                "Label": [f"({y:.2f}, {x:.2f})"],
            })
            return hv.Points(
                marker_df, kdims=["Longitude", "Latitude"],
            ).opts(
                size=18, color="red", marker="triangle",
                tools=["hover"],
            )

        marker_dmap = hv.DynamicMap(tap_marker, streams=[tap_stream])
        return base_map * marker_dmap, tap_stream

    def interactive_timeseries_with_range(
        self,
        daily_df: pd.DataFrame,
    ) -> tuple[Any, streams.RangeX]:
        """Create an interactive timeseries with RangeX stream for zoom selection.

        Zooming/panning on the plot updates the RangeX stream with the
        current x-axis bounds, enabling downstream reactive filtering.

        Parameters
        ----------
        daily_df : pd.DataFrame
            DataFrame with 'time' and GHI columns.

        Returns
        -------
        tuple[hv.DynamicMap, hv.streams.RangeX]
            The interactive plot and the RangeX stream.
        """
        cols_to_plot = [c for c in daily_df.columns if c != "time"]

        base_plot = daily_df.hvplot.line(
            x="time", y=cols_to_plot,
            title="Daily GHI (zoom to select range)",
            xlabel="Date", ylabel="GHI (kWh/m\u00b2/day)",
            width=self.width, height=self.height,
            legend="top_right",
        )

        range_stream = streams.RangeX(source=base_plot)
        return base_plot, range_stream

    def interactive_orientation_selector(
        self,
        sim_df: pd.DataFrame,
        tilt: int = 30,
    ) -> tuple[Any, streams.Selection1D]:
        """Orientation bar chart with Selection1D stream.

        Click on bars to select orientations for detailed comparison.
        The Selection1D stream provides indices of selected elements.

        Parameters
        ----------
        sim_df : pd.DataFrame
            Simulation results.
        tilt : int
            Tilt angle to display.

        Returns
        -------
        tuple[hv.Bars, hv.streams.Selection1D]
            Interactive bar chart and selection stream.
        """
        filtered = sim_df[sim_df["tilt_deg"] == tilt].drop_duplicates(
            subset=["direction"]
        ).reset_index(drop=True)

        bars = hv.Bars(
            filtered, kdims=["direction"], vdims=["annual_energy_kwh"],
        ).opts(
            title=f"Select Orientations to Compare (Tilt: {tilt}°)",
            xlabel="Direction", ylabel="Annual Energy (kWh)",
            width=self.width, height=self.height,
            color="annual_energy_kwh", cmap="YlOrRd",
            tools=["tap", "hover"],
        )

        selection_stream = streams.Selection1D(source=bars)
        return bars, selection_stream

    # -------------------------------------------------------------------
    # Dynamic Datashader Rasterization
    # -------------------------------------------------------------------

    def dynamic_rasterized_map(
        self,
        ds: Any,
        ghi_var: str = "GHI",
    ) -> hv.DynamicMap:
        """Create a dynamically rasterized map using Datashader + HoloViews.

        Uses holoviews.operation.datashader.rasterize() for zoom-dependent
        re-rendering. As the user zooms in, the data is re-rasterized at
        the appropriate resolution for the current viewport.

        Parameters
        ----------
        ds : xr.Dataset
            Gridded dataset with lat/lon dimensions.
        ghi_var : str
            Variable name for GHI.

        Returns
        -------
        hv.DynamicMap
            Dynamically rasterized map that re-renders on zoom/pan.
        """
        from holoviews.operation.datashader import rasterize

        data = ds[ghi_var]
        lats = data.coords[data.dims[0]].values
        lons = data.coords[data.dims[1]].values
        values = data.values

        bounds = (float(lons.min()), float(lats.min()),
                  float(lons.max()), float(lats.max()))

        img = hv.Image(
            values, bounds=bounds,
            kdims=["Longitude", "Latitude"],
            vdims=["GHI"],
        )

        rasterized = rasterize(img).opts(
            cmap="inferno", colorbar=True,
            width=self.width + 200, height=self.height + 100,
            title="Global Solar Map (Dynamic Rasterization)",
            tools=["hover", "wheel_zoom", "pan", "reset", "box_zoom"],
        )

        return rasterized

    # ------------------------------------------------------------------
    # Dual-Source Comparison Charts
    # ------------------------------------------------------------------

    def dual_source_timeseries(
        self,
        aligned_df: pd.DataFrame,
        variable: str = "GHI",
    ) -> hv.Overlay:
        """Overlay timeseries from two data sources for visual comparison.

        Parameters
        ----------
        aligned_df : pd.DataFrame
            DataFrame with time index and one column per source.
        variable : str
            Label for the variable being compared.

        Returns
        -------
        hv.Overlay
        """
        curves = []
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, col in enumerate(aligned_df.columns):
            curve = hv.Curve(
                (aligned_df.index, aligned_df[col]),
                "Time", f"{variable} (kWh/m²/day)",
                label=col,
            ).opts(
                color=colors[i % len(colors)],
                line_width=1.5,
                alpha=0.7,
            )
            curves.append(curve)

        overlay = hv.Overlay(curves).opts(
            width=self.width, height=self.height,
            title=f"{variable} — Multi-Source Comparison",
            legend_position="top_right",
            tools=["hover", "wheel_zoom", "pan"],
        )
        return overlay

    def dual_source_monthly_bar(
        self,
        monthly_df: pd.DataFrame,
    ) -> hv.Bars:
        """Grouped bar chart comparing monthly GHI from multiple sources.

        Parameters
        ----------
        monthly_df : pd.DataFrame
            Columns: month_name, plus one column per source.

        Returns
        -------
        hv.Bars
        """
        source_cols = [c for c in monthly_df.columns if c not in ("month", "month_name")]
        records = []
        for _, row in monthly_df.iterrows():
            for src in source_cols:
                records.append({
                    "Month": row["month_name"],
                    "Source": src,
                    "GHI": row[src],
                })

        df = pd.DataFrame(records)
        bars = hv.Bars(df, kdims=["Month", "Source"], vdims="GHI").opts(
            width=self.width, height=self.height,
            title="Monthly GHI — Source Comparison",
            ylabel="GHI (kWh/m²/day)",
            xrotation=45,
            multi_level=False,
            color="Source",
            cmap="Category10",
            legend_position="top_right",
            tools=["hover"],
        )
        return bars

    def dual_source_scatter(
        self,
        aligned_df: pd.DataFrame,
    ) -> hv.Overlay:
        """Scatter plot of Source A vs Source B for correlation visualization.

        Parameters
        ----------
        aligned_df : pd.DataFrame
            DataFrame with exactly 2 source columns.

        Returns
        -------
        hv.Overlay
            Scatter points + 1:1 reference line.
        """
        cols = list(aligned_df.columns)
        if len(cols) < 2:
            return hv.Div("<p>Need 2 sources for scatter comparison</p>")

        common = aligned_df[cols[:2]].dropna()
        src_a, src_b = cols[0], cols[1]

        scatter = hv.Points(
            common, kdims=[src_a, src_b],
        ).opts(
            size=3, alpha=0.4, color="#1f77b4",
            width=self.height, height=self.height,
            title=f"GHI Correlation: {src_a} vs {src_b}",
            xlabel=f"{src_a} (kWh/m²/day)",
            ylabel=f"{src_b} (kWh/m²/day)",
            tools=["hover"],
        )

        # 1:1 reference line
        vmin = float(common.min().min())
        vmax = float(common.max().max())
        ref_line = hv.Curve(
            [(vmin, vmin), (vmax, vmax)],
        ).opts(color="red", line_dash="dashed", line_width=2)

        return scatter * ref_line

    def dual_source_difference_heatmap(
        self,
        aligned_df: pd.DataFrame,
    ) -> hv.HeatMap:
        """Month x Year heatmap of difference between two sources.

        Parameters
        ----------
        aligned_df : pd.DataFrame
            DataFrame with 2 source columns and time index.

        Returns
        -------
        hv.HeatMap
        """
        cols = list(aligned_df.columns)
        if len(cols) < 2:
            return hv.Div("<p>Need 2 sources for difference heatmap</p>")

        common = aligned_df[cols[:2]].dropna()
        diff = common[cols[0]] - common[cols[1]]

        df = pd.DataFrame({
            "month": diff.index.month,
            "year": diff.index.year,
            "difference": diff.values,
        })

        monthly = df.groupby(["year", "month"])["difference"].mean().reset_index()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m - 1])

        heatmap = hv.HeatMap(
            monthly, kdims=["month_name", "year"], vdims="difference",
        ).opts(
            width=self.width, height=self.height,
            title=f"GHI Difference ({cols[0]} - {cols[1]})",
            cmap="RdBu_r",
            colorbar=True,
            tools=["hover"],
            xrotation=45,
        )
        return heatmap

    def source_location_map(
        self,
        lat: float,
        lon: float,
        ghi_values: dict[str, float],
        global_ds: xr.Dataset | None = None,
    ) -> hv.Overlay:
        """Map showing location marker with per-source GHI annotations.

        Parameters
        ----------
        lat, lon : float
            Location coordinates.
        ghi_values : dict[str, float]
            Source name -> average daily GHI.
        global_ds : xr.Dataset, optional
            Global solar grid for background.

        Returns
        -------
        hv.Overlay
        """
        # Background map
        if global_ds is not None:
            ghi_data = global_ds["GHI"].values
            lats = global_ds.coords["lat"].values
            lons = global_ds.coords["lon"].values
            base = hv.Image(
                (lons, lats, ghi_data),
                kdims=["Longitude", "Latitude"],
                vdims=["GHI"],
            ).opts(
                cmap="YlOrRd", colorbar=True, alpha=0.6,
                width=self.width + 200, height=self.height + 100,
            )
        else:
            base = hv.Tiles("https://tile.openstreetmap.org/{Z}/{X}/{Y}.png").opts(
                width=self.width + 200, height=self.height + 100,
            )

        # Location marker
        label_parts = [f"{src}: {ghi:.2f}" for src, ghi in ghi_values.items()]
        label = " | ".join(label_parts)

        marker = hv.Points(
            pd.DataFrame({"Longitude": [lon], "Latitude": [lat], "Label": [label]}),
            kdims=["Longitude", "Latitude"],
            vdims=["Label"],
        ).opts(
            size=15, color="red", marker="star",
            tools=["hover"],
        )

        title_text = f"Solar Resource at ({lat:.2f}, {lon:.2f})"
        return (base * marker).opts(title=title_text)
