# F1 Strategy Engine — V2.0

V2.0 turns the project into a race-scenario simulator rather than a static pre-race calculator.

## New: race scenario timeline

The dashboard now visualises the complete scenario on one horizontal race timeline:

- tyre stints
- pit-stop laps
- weather phases
- exact Safety Car / VSC timing when selected

Soft, Medium, Hard, Intermediate and Wet use distinct tyre colours.

## New: multi-phase weather

Choose:

- Static
- 2 phases
- 3 phases

Each phase can be:

- Expected conditions
- Dry
- Hot & dry
- Cool & dry
- Changeable
- Rain
- Heavy rain

For multi-phase weather you choose the lap at which the next phase starts.

Example:

- L1–L21 Dry
- L22–L39 Rain
- L40–L57 Heavy rain

The strategy engine evaluates each stint against the weather actually present during those laps.

## New: Safety Car / VSC timing

Choose:

- No SC / VSC
- Safety Car
- Virtual Safety Car

For SC/VSC choose either:

- Random timing
- a specific race lap

A pit stop only receives the neutralisation pit-loss benefit when it occurs close to the event.

## Engine changes

Pit timing optimisation is now weather-timeline aware.

For example, if rain starts on lap 23, an Intermediate switch around that transition can become preferable to a dry-only pit window.

The full-grid Monte Carlo model uses the same race scenario for every driver in each simulation.

## Interface

The interface has been reworked toward a pit-wall dashboard:

1. Race & Strategy
2. Scenario Evolution
3. Outcome
4. Race Scenario Timeline
5. Circuit / Weather Evolution / Your Strategy
6. Strategy Benchmark
7. Estimated Final Classification

## GitHub update

Replace:
- `app.py`
- `data_sources.py`
- `README.md`

Add:
- `strategy_engine_v20.py`

Older engine versions may remain in the repository. V2.0 imports only `strategy_engine_v20.py`.
