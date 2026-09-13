# Strategy Engine — V2.7.1

Target Outcome hotfix.

## Fixed

- Target Outcome no longer collapses every driver to the same generic P10 / pace 0.0 fallback.
- If FastF1 weekend evidence is unavailable, the app builds a driver-specific current-season strength prior.
- For the 2026 Spanish GP, the known current-season starting-grid baseline is used for the top 10 and merged with the season prior for the rest of the field.
- Target results are namespaced with a new model revision so stale 69% results from an older Streamlit session are invalidated.
- If no driver-specific context can be built, Target Outcome now stops instead of showing a misleading generic probability.

## GitHub update

Replace:
- `app.py`
- `official_sources.py`
- `README.md`

Keep `strategy_engine_v21.py` and `data_sources.py` unchanged from V2.7.
