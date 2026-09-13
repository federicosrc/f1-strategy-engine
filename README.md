# Strategy Engine V1.2 — Current Season Dashboard

Formula 1 pre-race strategy dashboard built in Streamlit.

## What changed from V1

- **Current season only**: no season selector and no previous-season circuit database.
- **Formula 1 official website is the primary circuit source**:
  - current championship calendar
  - circuit length
  - number of laps
  - race distance
  - circuit map when the official page exposes a usable image URL
- **Pirelli official current-season nominations** for Hard / Medium / Soft.
- **Open-Meteo** for race-time weather forecast.
- **OpenF1 is only called after pressing `Run strategy simulation`**.
- **Anti-429 design**:
  - 2.05 second minimum spacing between uncached OpenF1 requests
  - Retry-After / exponential retry for HTTP 429 and transient 5xx errors
  - in-process TTL cache
  - one current practice session loaded first; a second is queried only if the latest session has no usable long run
- New dark **race-control dashboard UI** inspired by motorsport engineering dashboards, without copying Formula 1/Pirelli artwork.
- Every important input shows a **source and confidence level**.

## Data hierarchy

| Variable | Primary source | Confidence target |
|---|---|---|
| Calendar / circuit | Formula 1 official | High |
| Circuit map | Formula 1 official | High |
| Pirelli compounds | Pirelli official | High |
| Weather | Open-Meteo | High |
| Grid | OpenF1 current weekend | High |
| Degradation | OpenF1 current-weekend practice long runs | Medium/High |
| Race pace | OpenF1 current-weekend practice | Medium |
| Tyre sets remaining | Current-weekend inference + manual override | Low |
| SC/VSC probability | Circuit prior + current-weekend race control | Medium |
| Pit-lane loss | Circuit-type prior until measured current-season data are available | Low/Medium |

## Files to upload to GitHub

Replace or add these files in the repository root:

- `app.py`
- `data_sources.py`
- `official_sources.py` **(new)**
- `strategy_engine.py`
- `requirements.txt`
- `README.md`

Also replace:

- `.streamlit/config.toml`

Streamlit Community Cloud will redeploy automatically after the GitHub commit.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Important limitation

OpenF1 does not expose an authoritative machine-readable list of the remaining tyre sets for each driver. V1.2 therefore marks tyre inventory as **LOW confidence** and keeps only those cells manually overrideable.

The next data-engine priority is automatic ingestion of the official remaining-tyre publication when a sufficiently stable source format is available.
