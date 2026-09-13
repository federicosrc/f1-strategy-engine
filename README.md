# F1 Strategy Engine — V1.7.1

V1.7 adds explicit race-scenario and pit-lap controls and simplifies the dashboard.

## New controls
The top row now contains:
- Grand Prix
- Driver
- starting tyre
- number of stops
- Stint 2 compound
- Stint 3 compound
- neutralisation: No SC/VSC, Safety Car, or Virtual Safety Car
- Pit lap 1
- Pit lap 2

The selected pit laps are fixed inputs to the Monte Carlo model.

## Safety Car / VSC logic
Choosing Safety Car or VSC guarantees that neutralisation type in the simulated race scenario, but does not make every stop cheaper.

The neutralisation lap is sampled within each Monte Carlo race. A pit stop receives the reduced neutralisation pit-loss only when it falls within the modelled event window. VSC receives a smaller pit-loss benefit than a full Safety Car.

## Dashboard changes
- The old Tyre Compounds panel has been removed.
- `Your Strategy` now occupies that area.
- `Weather & Track` is directly below the circuit.
- `Data Quality & Confidence` has been removed from the visible dashboard.
- A complete `Estimated Final Classification` has been added.
- The selected driver remains highlighted in the projected classification.

## Estimated classification
The full-grid engine simulates the selected driver plus the rest of the field in every Monte Carlo race. The classification table is ordered by average simulated finishing position and shows:
- projected position
- driver
- team
- grid position
- average simulated finish
- podium probability

## Data sources
- Formula1.com official: current-season calendar, circuit information and track map
- Pirelli: current-season tyre nominations
- Open-Meteo: race weather forecast
- FastF1: current-weekend practice, long-run pace, tyre degradation and qualifying/grid evidence

## GitHub update
Replace:
- `app.py`
- `strategy_engine.py`
- `README.md`

Keep the existing:
- `data_sources.py`
- `official_sources.py`
- `requirements.txt`
- `.streamlit/config.toml`


## V1.7.1 cache-safe module fix
The strategy engine is now imported from `strategy_engine_v17.py` so Streamlit cannot reuse an older cached `strategy_engine.py` module. Replace `app.py` and `data_sources.py`, and add `strategy_engine_v17.py`. The old `strategy_engine.py` can be left in the repository; it is no longer used by V1.7.1.
