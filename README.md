# F1 Strategy Engine — V1.5

V1.5 removes OpenF1 from pre-race mode.

## Sources
- Formula1.com official: current-season calendar, circuit information and track map
- Pirelli official: current-season compound nominations
- Open-Meteo: race forecast
- FastF1: completed FP1/FP2/FP3 timing, driver laps, compounds, TyreLife, stint evidence and qualifying result
- Strategy Engine: FIA legality filter, degradation, race pace, pit-window optimisation and Monte Carlo

## What FastF1 feeds into the model
- driver race-pace delta
- Soft / Medium / Hard degradation
- qualifying position when available
- current-weekend practice quality
- evidence of which compounds were used

Telemetry and FastF1 weather are disabled to keep the app lighter.

## Graceful fallback
If FastF1 cannot load a session, the simulation still runs with conservative fallback values. No large source-error banner is shown; the Data Quality panel marks those inputs LOW confidence.

## Important limitation
FastF1 does not provide the authoritative remaining tyre-set inventory. Tyre availability remains LOW confidence and overrideable.

## GitHub update
Replace:
- app.py
- data_sources.py
- requirements.txt
- README.md

Keep:
- official_sources.py
- strategy_engine.py
- .streamlit/config.toml
