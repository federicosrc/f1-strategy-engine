# F1 Strategy Engine — V1.4

V1.4 changes the application from a passive strategy predictor into a user-driven race strategy simulator.

## Workflow

1. Choose the current-season Grand Prix.
2. Choose the driver.
3. Choose the starting tyre.
4. Choose 1 or 2 pit stops.
5. Choose the following tyre compounds.
6. Press **Simulate my strategy**.

The dashboard returns:
- expected finish
- most likely finish
- win / podium / top-5 probabilities
- automatically optimised pit windows
- strategy cost versus the model optimum
- finish-position distribution
- scenario probabilities

## Regulation filter

For a planned dry race the app enforces:
- at least two different dry tyre specifications;
- at least one mandatory Race tyre specification;
- enough tyre sets for the chosen sequence.

If Intermediate or Wet tyres are used during the actual race, the dry two-specification requirement no longer applies.

## Data sources

- Formula 1 official: current-season calendar and circuit data
- Pirelli: current-season compound nominations
- Open-Meteo: race weather forecast
- OpenF1: current-weekend driver grid, practice laps, stints and degradation inputs

## Important limitation

Remaining tyre sets are still the weakest automatic variable in V1.4. The app keeps this value overrideable.

## Update the existing GitHub repository

Replace:
- `app.py`
- `strategy_engine.py`
- `README.md`

Keep:
- `data_sources.py`
- `official_sources.py`
- `requirements.txt`
- `.streamlit/config.toml`

Streamlit Community Cloud should redeploy automatically after the commit.
