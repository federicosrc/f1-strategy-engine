# F1 Strategy Engine — V1.3

Current-season Formula 1 pre-race strategy dashboard built with Streamlit.

## V1.3 changes

- Race setup moved from the left sidebar to a horizontal control bar.
- Driver-specific recalculation fixed: after the first **Run / refresh strategy**, changing GP or driver automatically rebuilds the selected driver's model.
- OpenF1 session payloads remain cached, so switching drivers normally reuses the same downloaded FP laps/stints.
- Tyre-set overrides are scoped by GP + driver and no longer leak from one driver to another.
- Team-choice probabilities now react to grid position / track-position value as well as driver-specific degradation and tyre availability.
- Formula1.com circuit parsing is more robust against client-rendered/embedded JSON.
- Circuit length and race distance use dedicated dashboard cards, preventing Streamlit metric truncation.
- If Formula1.com temporarily omits one distance field, V1.3 shows a clearly-marked derived fallback instead of a blank card.
- The current-season fallback calendar was aligned to the official 24-round 2026 Formula 1 calendar. No previous-season circuit database is bundled.

## Data hierarchy

1. **Formula 1 official** — current-season calendar, circuit page, laps, circuit length, race distance, track image.
2. **Pirelli official** — current-season dry compound nominations where published.
3. **Open-Meteo** — race-time weather forecast and asphalt-temperature estimate.
4. **OpenF1** — current-weekend grid, practice laps, stints, race control and driver-specific analytics.
5. **Model prior / override** — only where no sufficiently reliable automatic source exists.

## How driver switching works

The app deliberately avoids heavy OpenF1 calls before you ask for them.

1. Select GP and driver.
2. Press **Run / refresh strategy** once.
3. Current-weekend data are loaded and cached.
4. From then on, changing driver or GP triggers the relevant recalculation automatically.

The header shows **Driver-specific** when current-weekend driver data are active and **Baseline** when the dashboard is still using strategic priors.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Replace the same files in your GitHub repository and commit them. Streamlit Community Cloud should rebuild the application automatically.

Files to replace/add:

- `app.py`
- `data_sources.py`
- `official_sources.py`
- `strategy_engine.py`
- `requirements.txt`
- `README.md`
- `.streamlit/config.toml`

## Known limitation

The authoritative list of tyre sets remaining for each driver is still treated as low confidence because it is not currently supplied by the OpenF1 endpoints used by the app. It remains manually overrideable.
