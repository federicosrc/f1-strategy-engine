# Strategy Engine — V2.4

V2.4 adds an inverse **Target Outcome** optimiser while preserving the existing race simulator.

## Two modes

- **Simulate Race** — configure a race scenario and estimate the result.
- **Target Outcome** — choose a driver and a target (Win, Podium, Top 5 or Points) and let the engine search for the conditions that maximise the chance of reaching it.

## Target Outcome engine

The target optimiser keeps the driver's weekend pace/grid evidence fixed and searches plausible combinations of weather evolution, Safety Car / VSC timing, tyre strategy and pit-stop timing.

It uses a coarse-to-fine search: hundreds of candidate plans are screened, then the strongest candidates are confirmed with 30,000 Monte Carlo race simulations.

Outputs include maximum achievable probability, best path, most robust path, minimum requirements, sensitivity analysis and top scenario paths.

## GitHub update

Replace:
- `app.py`
- `data_sources.py`
- `README.md`

Add:
- `strategy_engine_v21.py`

Keep the remaining files unchanged.
