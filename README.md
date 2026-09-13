# Strategy Engine — V2.6.1

Target Outcome correctness update.

## Fixed

- selected-driver race pace no longer leaks into the simulated rival field
- changing driver now changes Target Outcome when driver-specific pace/grid evidence differs
- when the selected GP has no completed weekend data, Target Outcome uses a current-season-only prior from recent 2026 qualifying sessions instead of giving every driver the same P10 / zero-pace fallback
- target probability labels are explicit: Win = P1, Podium = P1–P3, Top 5 = P1–P5, Points = P1–P10
- the best-path panel now shows all four probabilities side by side so a 69% Top-5 chance cannot be confused with an expected P3 finish
- driver baseline and pace are shown in Target Outcome for transparency

## GitHub update

Replace:
- `app.py`
- `data_sources.py`
- `strategy_engine_v21.py`
- `README.md`

All other files can remain unchanged.
