# F1 Strategy Engine — V1.8

V1.8 turns the app further toward a realistic race-scenario simulator.

## Main changes

### One canonical finishing position
The large Estimated Final Position and the Estimated Final Classification now use the same ranking method.

The classification is ordered by average Monte Carlo finishing position. The selected driver's row in that table becomes the canonical projected finishing position displayed at the top.

The modal finishing position is retained internally for distribution analysis, but it is no longer shown as a conflicting headline result.

### Weather is now a scenario input
The top setup row includes:
- Forecast
- Dry
- Hot & dry
- Cool & dry
- Changeable
- Rain likely

The selection dynamically changes:
- air temperature
- estimated track temperature
- rain probability
- humidity / wind assumptions
- tyre thermal degradation through track temperature
- Monte Carlo rain occurrence

Forecast uses Open-Meteo. Other options are deliberate stress-test scenarios around the forecast baseline.

### Shared race scenario
Rain occurrence and race-control assumptions are shared across the field within each Monte Carlo race. The selected driver and all rivals therefore experience the same weather realization.

### Layout
- A clear `Race Scenario Setup` title appears above the input menus.
- `Weather & Track` remains under the circuit and reacts to the weather input.
- `Finish-position distribution` is directly below `Your Strategy`.
- The estimated final classification is full width below the strategy analysis.

## GitHub update

Replace:
- `app.py`
- `data_sources.py`
- `README.md`

Add:
- `strategy_engine_v18.py`

You may leave `strategy_engine_v17.py` in the repository; V1.8 no longer imports it.
