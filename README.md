# Strategy Engine — V2.1

V2.1 is a visual redesign of the V2.0 race-scenario simulator. The simulation engine is unchanged.

## Visual direction

The interface is rebuilt as a compact dark pit-wall workstation:

- original `STRATEGY ENGINE` text header — no Formula 1 logo
- Grand Prix / round information in the top bar
- compact Race Scenario Setup
- large Estimated Race Result card
- Strategy Comparison card
- full-width race timeline
- circuit information panel
- Weather & Track Evolution chart
- Tyre Performance table
- Key Insights panel
- full Estimated Final Classification timing screen

## Tyre colours

- Soft: red
- Medium: yellow
- Hard: white
- Intermediate: green
- Wet: blue

## Functionality retained

V2.1 keeps the V2.0 engine unchanged:
- 30,000 Monte Carlo simulations
- full-grid race outcome
- static / 2-phase / 3-phase weather
- Intermediate / Wet logic
- exact or random Safety Car / VSC timing
- fixed pit laps
- strategy benchmarking
- FastF1 current-weekend inputs

## GitHub update

Replace only:
- `app.py`
- `README.md`

Keep:
- `strategy_engine_v20.py`
- `data_sources.py`
- `official_sources.py`
- `requirements.txt`
- `.streamlit/config.toml`

No F1 logo or F1 brand mark is included in the app.
