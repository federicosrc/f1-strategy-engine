# F1 Strategy Engine — V1.6

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


## V1.5.1 UI changes
- Fixes the accidental Streamlit source-code/debug rendering below the strategy panel.
- Makes the final finishing position the largest visual result.
- Replaces the ambiguous strategy-cost histogram with a direct Your plan vs Best plan benchmark.
- Explains the meaning of strategy cost.
- Keeps the Monte Carlo finish-position distribution and labels its purpose explicitly.


## V1.6 — full-grid race outcome engine

The estimated finishing position is no longer calculated by converting a time penalty into an arbitrary number of positions.

FastF1 now builds a current-weekend model of the field:
- qualifying/grid position
- compound-normalised long-run pace delta
- team/driver identity
- data-confidence flags

For every Monte Carlo run:
1. the selected driver's chosen strategy produces a simulated race-time score;
2. every rival samples a competitive legal strategy;
3. each rival's long-run pace is accumulated over the full race distance;
4. grid position, traffic/execution variance, SC/rain effects and DNF risk are applied;
5. simulated race times are ordered to create the finishing classification.

The app then reports the actual Monte Carlo finishing-position distribution.

The estimated final position is now displayed immediately below the strategy controls.
