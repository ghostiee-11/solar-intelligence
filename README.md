# Solar Potential Intelligence Platform

A production-grade, AI-powered solar intelligence system that analyzes solar radiation from real scientific datasets (NASA POWER, ERA5), estimates photovoltaic energy production, simulates panel orientations, and delivers interactive analysis through the **HoloViz ecosystem**.

Built with **xarray** for multidimensional climate data processing, **pvlib** for physics-accurate solar calculations, and **Panel/Lumen/hvPlot/HoloViews/Datashader/Param** for interactive visualization.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-300%20passed-brightgreen.svg)](#testing)
[![Panel](https://img.shields.io/badge/Panel-dashboard-F7931E?logo=python&logoColor=white)](https://panel.holoviz.org)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

### Built With

<p>
  <a href="https://panel.holoviz.org/"><img src="https://panel.holoviz.org/_static/logo_horizontal_light_theme.png" alt="Panel" height="40"/></a>&nbsp;&nbsp;
  <a href="https://lumen.holoviz.org/"><img src="https://raw.githubusercontent.com/holoviz/lumen/main/docs/assets/logo.png" alt="Lumen" height="40"/></a>&nbsp;&nbsp;
  <a href="https://hvplot.holoviz.org/"><img src="https://raw.githubusercontent.com/holoviz/hvplot/main/doc/_static/logo_horizontal.png" alt="hvPlot" height="40"/></a>&nbsp;&nbsp;
  <a href="https://holoviews.org/"><img src="https://raw.githubusercontent.com/holoviz/holoviews/main/doc/_static/logo_horizontal.png" alt="HoloViews" height="40"/></a>&nbsp;&nbsp;
  <a href="https://datashader.org/"><img src="https://raw.githubusercontent.com/holoviz/datashader/main/doc/_static/logo_horizontal.png" alt="Datashader" height="40"/></a>&nbsp;&nbsp;
  <a href="https://param.holoviz.org/"><img src="https://raw.githubusercontent.com/holoviz/param/main/doc/_static/logo_horizontal.png" alt="Param" height="40"/></a>
</p>
<p>
  <a href="https://xarray.dev/"><img src="https://docs.xarray.dev/en/stable/_static/Xarray_Logo_RGB_Final.svg" alt="xarray" height="35"/></a>&nbsp;&nbsp;&nbsp;
  <a href="https://numpy.org/"><img src="https://numpy.org/images/logo.svg" alt="NumPy" height="35"/></a>&nbsp;&nbsp;&nbsp;
  <a href="https://pandas.pydata.org/"><img src="https://pandas.pydata.org/static/img/pandas_mark.svg" alt="Pandas" height="35"/></a>&nbsp;&nbsp;&nbsp;
  <a href="https://www.dask.org/"><img src="https://docs.dask.org/en/stable/_images/dask_horizontal.svg" alt="Dask" height="35"/></a>&nbsp;&nbsp;&nbsp;
  <a href="https://pvlib-python.readthedocs.io/"><img src="https://pvlib-python.readthedocs.io/en/stable/_images/pvlib_logo_horiz.png" alt="pvlib" height="35"/></a>&nbsp;&nbsp;&nbsp;
  <a href="https://www.python.org/"><img src="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/files/python-logo-only.svg" alt="Python" height="35"/></a>&nbsp;&nbsp;&nbsp;
  <a href="https://openai.com/"><img src="https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_Logo.svg" alt="OpenAI" height="30"/></a>&nbsp;&nbsp;&nbsp;
  <a href="https://power.larc.nasa.gov/"><img src="https://cdn.simpleicons.org/nasa/0033A0" alt="NASA POWER" height="30"/></a>
</p>

---

## Dashboard Preview

### Overview - KPI Cards + Monthly Charts

![Overview Dashboard](screenshots/final_01_overview.png)

> Real-time KPI cards showing Daily GHI, Annual Energy, Capacity Factor, Payback Period, and CO2 Offset. Monthly Solar Irradiance bar chart (GHI/DNI/DHI breakdown) and Energy Generation Projection area chart.

### Overview - Daily Timeseries + Distribution + Seasonal Heatmap

![Overview Scrolled](screenshots/overview_scrolled.png)

> Daily solar irradiance timeseries with 30-day rolling average, GHI frequency distribution histogram, and seasonal irradiance heatmap (month x metric).

---

### Orientation Analysis - Find the Optimal Panel Setup

![Orientation Analysis](screenshots/final_02_orientation.png)

> Optimal configuration banner (South-facing at 15 degrees tilt), Annual Energy by Direction comparison, Energy vs Tilt Angle curve, Hourly Energy Profile by Direction, and Energy by Orientation & Tilt heatmap.

---

### Interactive Solar Map - Click to Simulate Anywhere

![Solar Map](screenshots/final_03_solar_map.png)

> Global solar radiation heatmap rendered with Datashader. Click any point on the map, adjust tilt/direction/panels, and run instant simulations.

### Map Simulation Results

![Map Simulation](screenshots/map_simulation_clicked.png)

> After clicking a location on the map, the simulation panel shows: Average Daily GHI, Annual Generation (kWh/year), Monthly Savings, CO2 Offset, and system configuration details.

---

### Financial Analysis - Investment Returns in Your Currency

![Financial Analysis](screenshots/final_04_financial.png)

> Multi-currency support (INR, USD, EUR, GBP) with automatic defaults. Investment breakdown, Returns summary (first-year savings, payback period, ROI), Environmental impact (CO2 offset, equivalent trees). Solar Investment Payback Timeline and Annual Carbon Offset charts.

---

### Multi-Location Comparison - Rank Cities by Solar Potential

![Multi-Location](screenshots/final_05_multi_location.png)

> Compare up to 10 cities side-by-side. Ranking table sorted by solar resource quality. Annual Solar Energy bar chart and Monthly GHI Comparison timeseries overlay.

---

### Dual-Source Cross-Validation - NASA POWER vs ERA5

![Dual Source](screenshots/final_10_dual_source.png)

> When ERA5 (Copernicus CDS) is enabled, the Data Sources tab shows a full cross-validation report: correlation, RMSE, MAE, bias metrics. GHI timeseries overlay and NASA POWER vs ERA5 scatter plot with 1:1 reference line.

---

### AI-Powered Insights - Natural Language Analysis

![AI Insights](screenshots/final_07_ai_insights.png)

> Template-based analysis report (works without API key) covering Solar Resource Assessment, System Performance, Panel Orientation Recommendation, Financial Analysis, and Environmental Impact. All values use your actual analysis data.

### AI Chat - Ask Anything About Your Solar Setup

| Question | Screenshot |
|----------|-----------|
| "What's the best time to use heavy appliances?" | ![Appliances](screenshots/ai_chat_appliances.png) |
| "What if electricity rates increase to 12 rupees/kWh?" | ![Rate Increase](screenshots/ai_chat_rate_increase.png) |
| "Should I add a battery storage system?" | ![Battery](screenshots/ai_chat_battery.png) |
| "How does my carbon offset compare to driving an EV?" | ![Carbon EV](screenshots/ai_chat_carbon_ev.png) |

> GPT-4o-mini powered chat that uses your actual system data (production, costs, savings) to give specific, actionable answers.

---

### Dark Theme

![Dark Theme](screenshots/final_09_dark_theme.png)

> Full dark theme support with one click. All charts, KPI cards, and UI elements adapt automatically.

---

## Architecture

```mermaid
graph TD
    subgraph UI["Panel Dashboard"]
        direction LR
        W1["Location Input"]
        W2["Simulation Controls"]
        W3["Charts & Maps"]
        W4["AI Chat"]
    end

    subgraph PIPELINE["Lumen Pipeline Layer"]
        direction LR
        SDS["SolarDataSource"] --> TF["Transforms"] --> VW["Views"]
    end

    subgraph CORE["Core Engine"]
        direction LR
        DL["Data Loader\n(xarray)"]
        SA["Solar Analysis\n(pvlib)"]
        EE["Energy Estimator\n(formulas)"]
        OS["Orientation Sim\n(pvlib + numpy)"]
        FA["Financial Analyzer\n(NPV / ROI)"]
        AI["AI Engine\n(GPT-4o-mini)"]
    end

    subgraph DATA["Scientific Data Sources"]
        direction LR
        NASA["NASA POWER API"]
        ERA5["ERA5 / Copernicus CDS"]
        NC["NetCDF / Zarr"]
        CS["pvlib Clearsky"]
    end

    UI --> PIPELINE
    PIPELINE --> CORE
    CORE --> DATA

    style UI fill:#F7931E,stroke:#e07800,color:#fff,font-weight:bold
    style PIPELINE fill:#4B8BBE,stroke:#3a6f99,color:#fff,font-weight:bold
    style CORE fill:#306998,stroke:#1e4f6e,color:#fff,font-weight:bold
    style DATA fill:#0033A0,stroke:#002070,color:#fff,font-weight:bold
```

## Features

### Core Scientific Analysis
- **Solar radiation analysis** using xarray `.groupby()`, `.resample()`, `.rolling()` operations
- **Energy estimation** with temperature derating (NOCT model), inverter losses, panel degradation
- **Orientation simulation** with full direction x tilt matrix using pvlib transposition models
- **Financial modeling** with payback period, NPV, ROI, carbon offset, lifetime projections
- **Dual-source validation** comparing NASA POWER vs ERA5 reanalysis data

### Interactive Dashboard (7 Tabs)
- **Overview** - KPI cards, monthly irradiance, energy projection, daily timeseries, distribution, heatmap
- **Orientation Analysis** - Direction comparison, tilt curve, hourly profiles, orientation heatmap
- **Solar Map** - Datashader-rendered global radiation map with click-to-simulate
- **Financial** - Investment/returns/environmental summary, payback timeline, carbon offset chart
- **Multi-Location** - Compare up to 10 cities with ranking table and comparison charts
- **Data Sources** - NASA POWER vs ERA5 cross-validation with correlation analysis
- **AI Insights** - Template-based report + GPT-4o-mini powered Q&A chat

### Multi-Currency Support
- **INR** (Indian Rupee) - Default: 5,00,000 system cost, 8 rupees/kWh, 40% subsidy
- **USD** (US Dollar) - Default: $15,000 system cost, $0.12/kWh, 30% ITC
- **EUR** (Euro) - Default: 12,000 euros, 0.25 euros/kWh, 25% subsidy
- **GBP** (British Pound) - Default: 8,000 pounds, 0.28 pounds/kWh, 0% subsidy

### HoloViz Ecosystem Usage
| Library | Usage |
|---------|-------|
| **Panel** | Dashboard framework, widgets, layout, FastListTemplate, reactive callbacks |
| **Lumen** | Custom `SolarDataSource`, `SolarEnergyTransform`, pipeline integration |
| **hvPlot** | All chart generation (`.hvplot.bar()`, `.hvplot.line()`, `.hvplot.area()`, `.hvplot.hist()`) |
| **HoloViews** | `hv.HeatMap`, `hv.Image`, `hv.Table`, `hv.Points`, overlays, streams |
| **Datashader** | Server-side rendering for global solar radiation maps |
| **Param** | All classes use `param.Parameterized` for typed configuration |

---

## Quick Start

### Installation

```bash
git clone https://github.com/ghostiee-11/solar-intelligence.git
cd solar-intelligence
pip install -e .
```

### Optional Dependencies

```bash
# AI chat (GPT-4o-mini)
pip install -e ".[ai]"
export OPENAI_API_KEY="your-key-here"

# ERA5 data (Copernicus CDS)
pip install -e ".[era5]"
# Create ~/.cdsapirc with your CDS credentials

# All extras
pip install -e ".[all]"
```

### Run the Dashboard

```bash
panel serve src/solar_intelligence/ui/panel_dashboard.py --show
```

Then:
1. Enter a city name or coordinates
2. Adjust panel configuration (efficiency, area, number of panels)
3. Click **"Analyze Solar Potential"**
4. Explore all 7 tabs

### Python API

```python
from solar_intelligence.data_loader import generate_synthetic_solar_data
from solar_intelligence.solar_analysis import SolarAnalyzer
from solar_intelligence.energy_estimator import EnergyEstimator
from solar_intelligence.financial import FinancialAnalyzer

# Load data for New Delhi
ds = generate_synthetic_solar_data(lat=28.6, lon=77.2)

# Solar analysis
analyzer = SolarAnalyzer(dataset=ds, latitude=28.6, longitude=77.2)
summary = analyzer.summary()
print(f"Average GHI: {summary['average_daily_ghi']:.2f} kWh/m2/day")
print(f"Best month: {summary['best_month']} ({summary['best_month_ghi']:.1f} kWh/m2/day)")

# Energy estimation (20 panels, 1.7m2 each, 20% efficiency)
estimator = EnergyEstimator(panel_efficiency=0.20, panel_area=1.7, num_panels=20)
energy = estimator.system_summary(ds)
print(f"Annual production: {energy['production']['annual_energy_kwh']:,.0f} kWh")
print(f"Capacity factor: {energy['performance']['capacity_factor_pct']:.1f}%")

# Financial analysis (INR)
fa = FinancialAnalyzer(system_cost=500000, electricity_rate=8, incentive_percent=0.40)
fin = fa.financial_summary(energy['production']['annual_energy_kwh'])
print(f"Payback: {fin['returns']['payback_years']:.1f} years")
print(f"ROI: {fin['returns']['roi_pct']:.0f}%")
```

### Orientation Simulation

```python
from solar_intelligence.orientation_simulator import OrientationSimulator

sim = OrientationSimulator(latitude=28.6, longitude=77.2)
optimal = sim.optimal_orientation(ghi_daily_array, year=2023)
print(f"Best: {optimal['best_direction']} at {optimal['best_tilt']} degrees tilt")
print(f"Gain vs horizontal: {optimal['energy_gain_vs_horizontal_pct']:.1f}%")
```

---

## Project Structure

```
solar-intelligence/
|-- src/solar_intelligence/
|   |-- __init__.py
|   |-- config.py                 # Constants, API config, currency defaults
|   |-- data_loader.py            # NASA POWER + ERA5 clients, geocoding, caching
|   |-- solar_analysis.py         # Irradiance stats, seasonal patterns, multi-location
|   |-- energy_estimator.py       # PV energy output with temperature derating
|   |-- orientation_simulator.py  # Tilt/azimuth simulation (pvlib transposition)
|   |-- visualization.py          # hvPlot/HoloViews chart generators (20+ charts)
|   |-- financial.py              # ROI, NPV, payback, carbon offset, lifetime savings
|   |-- ai_engine.py              # Template + LLM-powered insights and chat
|   +-- ui/
|       |-- panel_dashboard.py    # Main Panel dashboard (7 tabs, 900+ lines)
|       |-- lumen_app.py          # Lumen pipeline interface
|       +-- components.py         # Reusable UI widgets
|-- tests/
|   |-- conftest.py               # Shared fixtures (3 location datasets)
|   |-- test_data_loader.py       # Data loading, caching, geocoding
|   |-- test_solar_analysis.py    # Irradiance stats validation
|   |-- test_energy_estimator.py  # Energy formula correctness
|   |-- test_orientation_simulator.py  # Physics constraints (S > N in NH)
|   |-- test_financial.py         # Payback, NPV, carbon calculations
|   |-- test_visualization.py     # Chart rendering validation
|   |-- test_integration.py       # End-to-end pipeline tests
|   |-- test_dual_source.py       # NASA POWER vs ERA5 cross-validation
|   +-- ...                       # 300+ tests total
|-- examples/quickstart.py
|-- notebooks/                    # Jupyter exploration notebooks
|-- screenshots/                  # Dashboard screenshots
|-- pyproject.toml                # Modern Python packaging
|-- requirements.txt
+-- LICENSE                       # MIT
```

## Data Sources

| Source | Coverage | Resolution | Access |
|--------|----------|------------|--------|
| [NASA POWER](https://power.larc.nasa.gov/) | Global, 1981-present | 1 degree x 1 degree, daily | Free API, no key needed |
| [ERA5 (Copernicus)](https://cds.climate.copernicus.eu/) | Global, 1940-present | 0.25 degree x 0.25 degree, hourly | Free account + API key |
| Synthetic Generator | Any location | Daily | Built-in, offline |

## Technology Stack

| Category | Libraries |
|----------|-----------|
| **Scientific Computing** | xarray, numpy, pandas, pvlib, dask |
| **Visualization** | Panel, hvPlot, HoloViews, Datashader, Param, Lumen |
| **Climate Data** | NetCDF4, h5netcdf, zarr, cdsapi |
| **Geocoding** | geopy |
| **AI/LLM** | OpenAI (GPT-4o-mini) |
| **Testing** | pytest (300+ tests) |

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=solar_intelligence

# Quick run
pytest tests/ -x -q
```

**Test results:** 300 passed, 1 skipped (live API test) in ~50 seconds.

Tests cover:
- Unit tests for all 8 modules
- Integration tests (full pipeline: data -> analysis -> energy -> orientation -> financial -> AI)
- Multi-location comparison (Delhi, London, Cairo, Sydney)
- Southern hemisphere validation (Sydney: North-facing optimal)
- Dashboard smoke tests
- Dual-source cross-validation tests
- Lumen pipeline integration tests

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=solar_intelligence

# Lint
ruff check src/

# Run dashboard with auto-reload
panel serve src/solar_intelligence/ui/panel_dashboard.py --show --autoreload
```

## License

MIT License. See [LICENSE](LICENSE) for details.

---
