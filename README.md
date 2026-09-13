# F1 Strategy Engine — V1

Deployable Streamlit prototype for Formula 1 pre-race strategy prediction.

## Automatically collected
- meetings, sessions, drivers and grid: OpenF1
- practice laps, stints, compound and tyre age: OpenF1
- weather forecast: Open-Meteo
- track-temperature estimate: air temperature + solar-radiation proxy
- tyre degradation: within-stint regression on practice long runs
- pit loss: measured from pit-lane data when available, otherwise a circuit-type prior
- SC/VSC and overtaking: circuit-type priors in V1
- strategy probabilities: Monte Carlo

## Low-confidence input
Remaining tyre sets are not authoritative in OpenF1, so V1 estimates them and always exposes an override.

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload this folder to GitHub.
2. Create a Streamlit Community Cloud app.
3. Entrypoint: `app.py`.
4. Deploy.

## V2
- historical SC/VSC probability by circuit and lap
- automatic Pirelli/FIA remaining-tyre ingestion
- fuel-corrected long-run degradation
- competitor undercut/overcut threat engine
- Bayesian lap-by-lap updates
- Ferrari/Red Bull team-choice behavioural calibration

Sources:
- https://openf1.org/docs/
- https://open-meteo.com/en/docs
