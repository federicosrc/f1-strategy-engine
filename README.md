# F1 Strategy Engine — V1.9

## Weather labels
`Forecast` has been replaced by **Expected conditions**.

Available scenarios:
- Expected conditions
- Dry
- Hot & dry
- Cool & dry
- Changeable
- Rain
- Heavy rain

## Wet-weather tyres
When wet weather is plausible, the tyre menus also show:
- INTERMEDIATE
- WET

They are enabled for Changeable, Rain and Heavy rain scenarios, and also for Expected conditions when the race-time rain probability is at least 20%.

## Regulation logic
If a strategy uses Intermediate or Wet tyres, the dry-race requirement to use two different slick specifications is not enforced.

If the strategy uses slicks only, the normal dry tyre rule remains active.

## Performance logic
Intermediate and Wet are not cosmetic menu options.

The model applies condition-dependent pace penalties:
- Intermediate is strongly penalised on a dry track and favoured in normal rain.
- Wet is heavily penalised on a dry track and favoured in heavy rain.
- Slicks receive large penalties when a simulated race is wet.

This lets the engine compare plans such as:
- M → I
- I → W
- I → I
- W → I
- M → I → M

## GitHub update
Replace:
- `app.py`
- `data_sources.py`
- `README.md`

Add:
- `strategy_engine_v19.py`

V1.9 imports `strategy_engine_v19.py`, so the older v18/v17 engine files can remain in the repository without being used.
