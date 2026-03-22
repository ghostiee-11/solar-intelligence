"""AI explanation engine for Solar Intelligence.

Generates natural language insights about solar analysis results.
Supports template-based explanations (no API key needed) and
optional LLM-powered analysis.
"""

from __future__ import annotations

import logging
from typing import Any

import param

from solar_intelligence.config import (
    DEFAULT_SYSTEM_LIFETIME,
    IRRADIANCE_EXCELLENT,
    IRRADIANCE_GOOD,
    IRRADIANCE_LOW,
    IRRADIANCE_MODERATE,
)

logger = logging.getLogger(__name__)

DEFAULT_APPLIANCE_WATTAGES = {
    "air_conditioner": 1.5,
    "central_ac": 3.5,
    "refrigerator": 0.15,
    "washing_machine": 0.5,
    "water_heater": 2.0,
    "led_light": 0.01,
    "ceiling_fan": 0.075,
    "ev_charger_l2": 7.4,
    "microwave": 1.2,
    "laptop": 0.065,
    "television": 0.1,
}

# Region-specific payback thresholds (years): (excellent, very_good, good, moderate)
# Default thresholds are used when no country-specific entry exists.
COUNTRY_PAYBACK_THRESHOLDS: dict[str, tuple[float, float, float, float]] = {
    "IN": (4, 7, 10, 18),   # India — strong subsidies
    "US": (5, 8, 12, 20),   # USA — federal ITC
    "DE": (6, 9, 13, 22),   # Germany — high electricity cost offsets longer payback
    "AU": (4, 7, 10, 18),   # Australia — high irradiance
}


LLM_PROVIDERS = {
    "openai": {
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-nano", "gpt-4.1-mini"],
        "default": "gpt-4o-mini",
        "free": False,
    },
    "groq": {
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"],
        "default": "llama-3.3-70b-versatile",
        "free": True,
    },
    "gemini": {
        "models": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"],
        "default": "gemini-2.0-flash",
        "free": True,
    },
}


def _create_llm_client(provider: str, api_key: str):
    """Create an OpenAI-compatible client for the given provider."""
    import openai

    if provider == "openai":
        return openai.OpenAI(api_key=api_key)
    elif provider == "groq":
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    elif provider == "gemini":
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


class SolarAIEngine(param.Parameterized):
    """Generate natural language explanations of solar analysis results.

    Operates in two modes:
    1. Template-based (default): Rule-based insights, no API key needed.
    2. LLM-powered (optional): Uses OpenAI, Groq, or Gemini for richer explanations.

    Parameters
    ----------
    mode : str
        "template" or "llm".
    provider : str
        LLM provider: "openai", "groq", or "gemini".
    api_key : str
        API key for the selected provider.
    """

    mode = param.Selector(default="template", objects=["template", "llm"])
    provider = param.Selector(default="openai", objects=["openai", "groq", "gemini"])
    llm_model = param.String(default="gpt-4o-mini", doc="LLM model identifier")
    api_key = param.String(default="", doc="API key for LLM provider")
    country_code = param.String(default="", doc="ISO country code for region-specific thresholds")

    def _get_client(self):
        """Get or create the LLM client for current provider/key."""
        import openai
        key = self.api_key or None
        return _create_llm_client(self.provider, key)

    def _classify_irradiance(self, ghi: float) -> str:
        """Classify solar resource quality."""
        if ghi >= IRRADIANCE_EXCELLENT:
            return "excellent"
        elif ghi >= IRRADIANCE_GOOD:
            return "good"
        elif ghi >= IRRADIANCE_MODERATE:
            return "moderate"
        else:
            return "low"

    def _classify_payback(self, years: float) -> str:
        """Classify payback period quality.

        Uses country-specific thresholds when ``self.country_code`` matches
        an entry in ``COUNTRY_PAYBACK_THRESHOLDS``; otherwise falls back to
        sensible defaults.
        """
        code = self.country_code.upper()
        excellent, very_good, good, moderate = COUNTRY_PAYBACK_THRESHOLDS.get(
            code, (5, 8, 12, 20),
        )
        if years <= excellent:
            return "excellent"
        elif years <= very_good:
            return "very good"
        elif years <= good:
            return "good"
        elif years <= moderate:
            return "moderate"
        else:
            return "poor"

    def generate_report(
        self,
        solar_summary: dict[str, Any],
        energy_summary: dict[str, Any],
        financial_summary: dict[str, Any],
        orientation_result: dict[str, Any] | None = None,
        currency_symbol: str = "$",
        currency_code: str = "USD",
    ) -> str:
        """Generate a comprehensive natural language analysis report.

        Parameters
        ----------
        solar_summary : dict
            From SolarAnalyzer.summary().
        energy_summary : dict
            From EnergyEstimator.system_summary().
        financial_summary : dict
            From FinancialAnalyzer.financial_summary().
        orientation_result : dict, optional
            From OrientationSimulator.optimal_orientation().

        Returns
        -------
        str
            Multi-paragraph analysis report in plain English.
        """
        if self.mode == "template":
            return self._template_report(
                solar_summary, energy_summary, financial_summary, orientation_result,
                currency_symbol,
            )
        else:
            return self._llm_report(
                solar_summary, energy_summary, financial_summary, orientation_result,
                currency_symbol, currency_code,
            )

    def _template_report(
        self,
        solar: dict,
        energy: dict,
        financial: dict,
        orientation: dict | None,
        sym: str = "$",
    ) -> str:
        """Generate template-based analysis report."""
        sections = []

        # --- Solar Resource Assessment ---
        ghi = solar.get("average_daily_ghi", 0)
        quality = self._classify_irradiance(ghi)
        lat = solar.get("location", {}).get("latitude", 0)
        lon = solar.get("location", {}).get("longitude", 0)

        sections.append(
            f"## Solar Resource Assessment\n\n"
            f"Your location ({lat:.2f}°N, {lon:.2f}°E) receives an average of "
            f"**{ghi:.2f} kWh/m²/day** of solar radiation, which is classified as "
            f"**{quality}** solar potential. "
            f"The annual solar energy available is approximately "
            f"**{solar.get('annual_solar_energy_kwh_m2', 0):.0f} kWh/m²/year**.\n\n"
            f"The best month for solar generation is **{solar.get('best_month', 'N/A')}** "
            f"({solar.get('best_month_ghi', 0):.2f} kWh/m²/day), while the lowest "
            f"production month is **{solar.get('worst_month', 'N/A')}** "
            f"({solar.get('worst_month_ghi', 0):.2f} kWh/m²/day). "
            f"The seasonal ratio is {solar.get('seasonal_ratio', 0):.1f}x."
        )

        # --- System Performance ---
        sys_info = energy.get("system", {})
        prod = energy.get("production", {})
        perf = energy.get("performance", {})

        sections.append(
            f"## System Performance\n\n"
            f"Your **{sys_info.get('capacity_kw', 0):.1f} kW** solar system "
            f"({sys_info.get('num_panels', 0)} panels, "
            f"{sys_info.get('total_area_m2', 0):.0f} m²) is estimated to produce "
            f"**{prod.get('annual_energy_kwh', 0):,.0f} kWh/year** "
            f"(~{prod.get('avg_daily_energy_kwh', 0):.1f} kWh/day).\n\n"
            f"Peak production occurs in **{prod.get('best_month', 'N/A')}** "
            f"({prod.get('best_month_energy_kwh', 0):,.0f} kWh), "
            f"with lowest output in **{prod.get('worst_month', 'N/A')}** "
            f"({prod.get('worst_month_energy_kwh', 0):,.0f} kWh).\n\n"
            f"The system capacity factor is **{perf.get('capacity_factor_pct', 0):.1f}%** "
            f"with a specific yield of "
            f"**{perf.get('specific_yield_kwh_kwp', 0):,.0f} kWh/kWp**."
        )

        # --- Orientation Recommendation ---
        if orientation:
            best_dir = orientation.get("best_direction", "South")
            best_tilt = orientation.get("best_tilt", 30)
            gain_h = orientation.get("energy_gain_vs_horizontal_pct", 0)
            gain_w = orientation.get("energy_gain_vs_worst_pct", 0)
            worst_dir = orientation.get("worst_direction", "North")

            sections.append(
                f"## Panel Orientation Recommendation\n\n"
                f"For maximum annual energy production, install panels facing "
                f"**{best_dir}** at a **{best_tilt}° tilt angle**. "
                f"This configuration produces **{gain_h:.1f}% more energy** than "
                f"horizontal (flat) panels and **{gain_w:.1f}% more** than the worst "
                f"orientation ({worst_dir}-facing).\n\n"
                f"The optimal annual energy with this configuration is "
                f"**{orientation.get('annual_energy_kwh', 0):,.0f} kWh**."
            )

        # --- Financial Analysis ---
        inv = financial.get("investment", {})
        ret = financial.get("returns", {})
        env = financial.get("environmental", {})
        payback = ret.get("payback_years", "N/A")
        payback_quality = (
            self._classify_payback(payback) if isinstance(payback, (int, float)) else "N/A"
        )

        sections.append(
            f"## Financial Analysis\n\n"
            f"**Investment:** {sym}{inv.get('system_cost', 0):,.0f} system cost "
            f"- {sym}{inv.get('incentive', 0):,.0f} subsidy/incentive = "
            f"**{sym}{inv.get('net_cost', 0):,.0f} net cost**.\n\n"
            f"**Returns:** First-year savings of "
            f"**{sym}{ret.get('first_year_savings', 0):,.0f}**. "
            f"The investment pays back in **{payback} years** ({payback_quality}). "
            f"Over the system's {DEFAULT_SYSTEM_LIFETIME}-year lifetime, "
            f"the NPV is **{sym}{ret.get('npv_25yr', 0):,.0f}** "
            f"with **{ret.get('roi_pct', 0):.0f}% ROI**."
        )

        sections.append(
            f"## Environmental Impact\n\n"
            f"Your solar system offsets **{env.get('annual_co2_offset_kg', 0):,.0f} kg** "
            f"of CO₂ annually — equivalent to planting "
            f"**{env.get('equivalent_trees', 0)} trees** or avoiding "
            f"**{env.get('equivalent_car_miles_avoided', 0):,.0f} car miles**.\n\n"
            f"Over the system lifetime, you will avoid "
            f"**{env.get('lifetime_co2_offset_tonnes', 0):.1f} tonnes** of CO₂ emissions."
        )

        return "\n\n".join(sections)

    def _llm_report(
        self,
        solar: dict,
        energy: dict,
        financial: dict,
        orientation: dict | None,
        currency_symbol: str = "$",
        currency_code: str = "USD",
    ) -> str:
        """Generate LLM-powered analysis report (requires API key)."""
        try:
            import openai
        except ImportError:
            logger.warning("openai package not installed. Falling back to template mode.")
            return self._template_report(solar, energy, financial, orientation, currency_symbol)

        prompt = (
            "You are a solar energy analyst. Based on the following data, "
            "write a clear, professional analysis report.\n"
            f"IMPORTANT: All financial values are in {currency_code} ({currency_symbol}). "
            f"Use the {currency_symbol} symbol for all monetary values.\n\n"
            f"Solar Data: {solar}\n"
            f"Energy System: {energy}\n"
            f"Financial Analysis: {financial}\n"
        )
        if orientation:
            prompt += f"Orientation Analysis: {orientation}\n"

        prompt += (
            "\nWrite 4-5 paragraphs covering: solar resource quality, "
            "system performance, optimal orientation, financial returns, "
            "and environmental impact. Use specific numbers from the data. "
            f"Remember to use {currency_symbol} for all currency values."
        )

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("LLM report generation failed: %s", e)
            return self._template_report(solar, energy, financial, orientation)

    def chat_query(
        self,
        question: str,
        solar_summary: dict[str, Any] | None = None,
        energy_summary: dict[str, Any] | None = None,
        financial_summary: dict[str, Any] | None = None,
        currency_symbol: str = "$",
        currency_code: str = "USD",
    ) -> str:
        """Answer a user question about the solar analysis using LLM.

        Parameters
        ----------
        question : str
            User's natural language question.
        solar_summary, energy_summary, financial_summary : dict, optional
            Analysis context to inform the answer.
        currency_symbol : str
            Currency symbol to use.
        currency_code : str
            Currency code (INR, USD, EUR, GBP).

        Returns
        -------
        str
            LLM-generated answer.
        """
        try:
            import openai
        except ImportError:
            return (
                "LLM not available (openai package not installed). "
                "Run: `pip install openai` and set OPENAI_API_KEY."
            )

        # Build context
        context_parts = [
            "You are an expert solar energy consultant with deep knowledge of "
            "photovoltaic systems, energy economics, and residential solar installations. "
            "Answer the user's question using the specific analysis data below. "
            "Always ground your answers in the actual numbers from their system.",
            f"Currency: {currency_code} ({currency_symbol}). Use {currency_symbol} for all money values.",
            "",
            "IMPORTANT GUIDELINES:",
            "- Use the ACTUAL data values (production, costs, savings) in your answer.",
            "- Apply solar energy domain knowledge to give practical, actionable advice.",
            "- For time-of-day questions: peak solar production is 10am-3pm; recommend "
            "  running heavy loads (AC, washing machine, water heater) during these hours.",
            "- For appliance questions: typical wattages — "
            + ", ".join(
                f"{name.replace('_', ' ')} = {watt}kW"
                for name, watt in DEFAULT_APPLIANCE_WATTAGES.items()
            )
            + ".",
            "- For comparison questions: use the system's actual kWh and financial figures.",
            "- Give specific numbers, not vague generalities.",
            "",
        ]

        if solar_summary:
            context_parts.append(f"Solar Analysis: {solar_summary}")
        if energy_summary:
            context_parts.append(f"Energy System: {energy_summary}")
        if financial_summary:
            context_parts.append(f"Financial Data: {financial_summary}")

        context_parts.append(
            "\nAnswer thoroughly but concisely. Use bullet points and bold numbers. "
            "Always reference the user's specific system data in your answer. "
            "If the data doesn't contain enough info, use your solar domain expertise "
            "to give the best possible answer, noting any assumptions."
        )

        system_msg = "\n".join(context_parts)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": question},
                ],
                max_tokens=800,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("LLM chat failed: %s", e)
            err_str = str(e)
            if "api_key" in err_str.lower() or "authentication" in err_str.lower():
                provider = self.provider.capitalize()
                return (
                    f"**{provider} API key not configured.** "
                    f"Add your API key in the AI Settings section of the sidebar.\n\n"
                    f"Free options: Groq (free tier) or Gemini (free tier)."
                )
            return f"**AI chat error:** {e}"

    def quick_insight(
        self,
        metric: str,
        value: float,
        context: dict | None = None,
    ) -> str:
        """Generate a single quick insight for a metric.

        Parameters
        ----------
        metric : str
            Metric name (e.g., "ghi", "payback", "capacity_factor").
        value : float
            Metric value.
        context : dict, optional
            Additional context.

        Returns
        -------
        str
            One-sentence insight.
        """
        insights = {
            "ghi": lambda v: (
                f"Average daily irradiance of {v:.1f} kWh/m²/day is "
                f"{self._classify_irradiance(v)} for solar generation."
            ),
            "payback": lambda v: (
                f"Payback period of {v:.1f} years is "
                f"{self._classify_payback(v)} for residential solar."
            ),
            "capacity_factor": lambda v: (
                f"Capacity factor of {v:.1f}% "
                f"{'exceeds' if v > 20 else 'is typical for'} "
                f"residential solar installations (typical: 15-25%)."
            ),
            "annual_energy": lambda v: (
                f"Annual production of {v:,.0f} kWh can power approximately "
                f"{v / 10000:.1f} average US households."
            ),
            "carbon": lambda v: (
                f"Annual CO₂ offset of {v:,.0f} kg equals "
                f"{int(v / TREES_KG_CO2_PER_YEAR)} trees planted."
            ),
        }

        generator = insights.get(metric)
        if generator:
            return generator(value)
        return f"{metric}: {value}"
