# Solar Intelligence — Phased TODO

> Based on Andrew (@ahuang11) feedback from 2026-03-18 Discord conversation.
> Goal: Ship improvements fast to demonstrate responsiveness to mentor feedback.

---

## Phase 1: Validation & Scientific Credibility (Priority: CRITICAL)
*Andrew's biggest concern — "you're not misleading a bunch of people with hallucinated formulas"*

- [ ] **1.1** Cross-validate against PVWatts for Delhi (28.6°N, 77.2°E)
  - Run PVWatts manually for same location, system size, tilt
  - Compare: annual kWh, monthly profile, capacity factor
  - Document % deviation and explain any differences
- [ ] **1.2** Cross-validate against PVGIS for a European city (e.g., Madrid)
  - Same approach — compare GHI, optimal tilt, annual yield
- [ ] **1.3** Cross-validate for Southern Hemisphere (e.g., Sydney)
  - Verify north-facing optimality, seasonal inversion
- [ ] **1.4** Create `notebooks/validation_pvwatts.ipynb`
  - Side-by-side comparison tables and charts
  - Show formula derivations with references (IEC 61215, Erbs 1982, PVWatts methodology)
  - Clearly state assumptions and where your model simplifies
- [ ] **1.5** Add validation badge/section to README
  - "Validated against NREL PVWatts within ±X% for tested locations"

**Deliverable:** Validation notebook committed to repo, README updated.

---

## Phase 2: GeoViews Coastline Integration (Priority: HIGH)
*Andrew: "might want to incorporate coastlines with geoviews"*

- [ ] **2.1** Add `geoviews` and `cartopy` to core dependencies (move from optional `[geo]`)
- [ ] **2.2** Update `visualization.py` — overlay coastlines on Datashader global map
  - Use `gv.feature.coastline` overlay on the existing HoloViews Image
  - Keep Datashader rendering for the solar data layer
  - Ensure the map shows full global extent (-90 to 90 lat, -180 to 180 lon) by default
- [ ] **2.3** Add country borders as optional overlay (`gv.feature.borders`)
- [ ] **2.4** Fix default map view to show full globe (Andrew noticed it was zoomed in)
- [ ] **2.5** Update screenshots in README with new map

**Deliverable:** Map with coastlines, full global default view.

---

## Phase 3: UI/UX Simplification (Priority: HIGH)
*Andrew: "think about the UI/UX from a layman's perspective"*

- [ ] **3.1** Audit the dashboard from a non-technical user's perspective
  - What does a homeowner in India need vs. what's currently shown?
  - Identify jargon that needs plain-English labels (GHI, DNI, DHI, Kt, NOCT)
- [ ] **3.2** Create a guided entry flow
  - Step 1: Enter city name (prominent search bar)
  - Step 2: See summary results (big numbers: savings, payback, optimal setup)
  - Step 3: "Explore Details" expands into the full 7-tab dashboard
- [ ] **3.3** Add tooltips / info icons for technical terms
  - e.g., hover over "GHI" → "Global Horizontal Irradiance — total solar energy hitting a flat surface"
- [ ] **3.4** Simplify the Overview tab
  - Lead with actionable info: "Your optimal setup: South-facing, 15° tilt"
  - Show payback period and monthly savings prominently
  - Move technical charts (heatmaps, distributions) to an "Advanced" section
- [ ] **3.5** Add a "Quick Result" card at the top
  - Plain English summary: "Installing 10 panels on your south-facing roof in Delhi could save you ₹1.2L/year with a 4-year payback"
- [ ] **3.6** Improve mobile responsiveness (if Panel supports it)
- [ ] **3.7** Test with 2-3 non-technical people and iterate

**Deliverable:** Cleaner dashboard that a homeowner can understand without solar expertise.

---

## Phase 4: HuggingFace Deployment (Priority: HIGH)
*Andrew: "perhaps you can deploy it on huggingface"*

- [ ] **4.1** Read Panel HF deployment docs: https://panel.holoviz.org/how_to/deployment/huggingface.html
- [ ] **4.2** Create `Dockerfile` for HuggingFace Spaces
  - Base image with Python 3.10+
  - Install core + geo dependencies
  - `CMD panel serve` with appropriate flags (`--address 0.0.0.0 --port 7860 --allow-websocket-origin`)
- [ ] **4.3** Create HuggingFace Space (Panel SDK type)
  - Add `README.md` with HF metadata (title, emoji, sdk, etc.)
- [ ] **4.4** Handle API rate limits gracefully in deployed environment
  - NASA POWER has no key requirement but may rate-limit
  - Add request throttling / retry with backoff
  - Pre-cache popular cities (Delhi, Mumbai, Bangalore, NYC, London, Sydney)
- [ ] **4.5** Add loading states and error messages for slow API calls
- [ ] **4.6** Test the deployed version end-to-end
- [ ] **4.7** Add HuggingFace Space link to README and repo description

**Deliverable:** Live demo at huggingface.co/spaces/ghostiee-11/solar-intelligence.

---

## Phase 5: Blog Post & Social Media (Priority: MEDIUM)
*Andrew: "announce it on social media too and/or write a blog post; perhaps mention how you built this too"*

- [ ] **5.1** Write blog post draft covering:
  - **The problem:** Solar calculators in India are vendor-biased, no transparency
  - **The approach:** Real NASA data + pvlib physics + HoloViz stack
  - **Technical deep-dive:** How the HoloViz tools work together
    - Panel for the dashboard framework
    - Lumen for data pipeline
    - Datashader + GeoViews for the global map
    - hvPlot/HoloViews for interactive charts
    - xarray for climate data processing
  - **Validation:** Show PVWatts comparison results
  - **What I learned:** Building a real scientific app with HoloViz
- [ ] **5.2** Include screenshots and GIFs of the dashboard
- [ ] **5.3** Publish on:
  - [ ] Dev.to or Medium
  - [ ] HoloViz Discourse (discourse.holoviz.org)
  - [ ] Twitter/X — tag @HoloViz_org
  - [ ] LinkedIn
- [ ] **5.4** Share HuggingFace demo link in all posts
- [ ] **5.5** Post in OWASP/GSoC Discord channels too (shows breadth)

**Deliverable:** Published blog post + social media announcements with live demo link.

---

## Phase 6: Polish & Advanced Features (Priority: LOW)
*Nice-to-haves after the core feedback loop is closed*

- [ ] **6.1** Add city presets dropdown (Delhi, Mumbai, Bangalore, Chennai, Kolkata, NYC, London, Tokyo, Sydney)
- [ ] **6.2** Indian-specific electricity rate database (state-wise tariffs)
- [ ] **6.3** Subsidy calculator for Indian states (PM Surya Ghar Muft Bijli Yojana rates)
- [ ] **6.4** PDF report export — generate downloadable summary for homeowners
- [ ] **6.5** Comparison mode: "Solar vs. Grid — 25-year cost projection"
- [ ] **6.6** Add unit tests for GeoViews integration
- [ ] **6.7** CI/CD pipeline (GitHub Actions) — run tests on push
- [ ] **6.8** Performance profiling — optimize slow API calls and computations

---

## Timeline (suggested)

| Phase | Target | Days |
|-------|--------|------|
| Phase 1: Validation | Mar 20-22 | 2-3 days |
| Phase 2: GeoViews | Mar 22-23 | 1-2 days |
| Phase 3: UI/UX | Mar 23-25 | 2-3 days |
| Phase 4: HF Deploy | Mar 25-26 | 1-2 days |
| Phase 5: Blog | Mar 26-27 | 1-2 days |
| Phase 6: Polish | Mar 28+ | Ongoing |

> **Key principle:** Close the feedback loop with Andrew FAST. Each completed phase is a reason to share an update in Discord and build rapport before GSoC proposal deadline (Mar 28).
