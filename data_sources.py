from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OPENF1_BASE = "https://api.openf1.org/v1"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
GEOCODE_BASE = "https://geocoding-api.open-meteo.com/v1/search"


class APIError(RuntimeError):
    pass


class OpenF1Client:
    """OpenF1 client designed to be polite to the free endpoint.

    V1 made several calls on every Streamlit rerun. V1.2 does the opposite:
    - nothing heavy is requested before the user presses ANALYSE;
    - calls are throttled;
    - 429 respects Retry-After / exponential backoff;
    - completed session payloads are cached in memory for the app process.
    """

    def __init__(self, timeout: int = 20, min_interval: float = 2.05):
        self.timeout = timeout
        self.min_interval = min_interval
        self.session = requests.Session()
        retry = Retry(
            total=4,
            connect=4,
            read=4,
            backoff_factor=1.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
        self.session.headers.update({"User-Agent": "F1StrategyEngine/1.3"})
        self._lock = threading.Lock()
        self._last_network_call = 0.0
        self._cache: dict[tuple, tuple[float, Any]] = {}

    def _cache_get(self, key, ttl):
        item = self._cache.get(key)
        if not item:
            return None
        ts, value = item
        if time.time() - ts <= ttl:
            return value
        self._cache.pop(key, None)
        return None

    def _cache_put(self, key, value):
        self._cache[key] = (time.time(), value)

    def _throttle(self):
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_network_call)
            if wait > 0:
                time.sleep(wait)
            self._last_network_call = time.monotonic()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None, ttl: int = 900):
        params = params or {}
        key = (endpoint, tuple(sorted((str(k), str(v)) for k, v in params.items())))
        cached = self._cache_get(key, ttl)
        if cached is not None:
            return cached

        self._throttle()
        r = self.session.get(f"{OPENF1_BASE}/{endpoint}", params=params, timeout=self.timeout)
        if r.status_code == 429:
            raise APIError("OpenF1 rate limit reached. Cached data remain available; retry in about a minute.")
        if not r.ok:
            raise APIError(f"OpenF1 {endpoint}: HTTP {r.status_code}")
        data = r.json()
        self._cache_put(key, data)
        return data

    def meetings(self, year: int):
        return self._get("meetings", {"year": year}, ttl=3600)

    def sessions(self, meeting_key: int):
        return self._get("sessions", {"meeting_key": meeting_key}, ttl=3600)

    def drivers(self, session_key: int):
        return self._get("drivers", {"session_key": session_key}, ttl=3600)

    def starting_grid(self, session_key: int):
        return self._get("starting_grid", {"session_key": session_key}, ttl=1800)

    def stints(self, session_key: int):
        return self._get("stints", {"session_key": session_key}, ttl=1800)

    def laps(self, session_key: int):
        return self._get("laps", {"session_key": session_key}, ttl=1800)

    def race_control(self, session_key: int):
        return self._get("race_control", {"session_key": session_key}, ttl=1800)

    def pit(self, session_key: int):
        return self._get("pit", {"session_key": session_key}, ttl=900)

    def weather(self, session_key: int):
        return self._get("weather", {"session_key": session_key}, ttl=300)

    def intervals(self, session_key: int, driver_number: int):
        return self._get("intervals", {"session_key": session_key, "driver_number": driver_number}, ttl=8)

    def positions(self, session_key: int, driver_number: int):
        return self._get("position", {"session_key": session_key, "driver_number": driver_number}, ttl=8)


class OpenMeteoClient:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()

    def geocode(self, location: str, country: str | None = None):
        q = location if not country else f"{location}, {country}"
        r = self.session.get(
            GEOCODE_BASE,
            params={"name": q, "count": 5, "language": "en", "format": "json"},
            timeout=self.timeout,
        )
        if not r.ok:
            raise APIError(f"Open-Meteo geocoding: HTTP {r.status_code}")
        results = r.json().get("results") or []
        if not results:
            raise APIError(f"Could not geocode {q}")
        return results[0]

    def forecast_at(self, lat: float, lon: float, race_time_utc: datetime | None = None):
        hourly = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation_probability",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "shortwave_radiation",
            "cloud_cover",
        ]
        r = self.session.get(
            OPEN_METEO_BASE,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(hourly),
                "current": ",".join(["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]),
                "timezone": "UTC",
                "forecast_days": 16,
            },
            timeout=self.timeout,
        )
        if not r.ok:
            raise APIError(f"Open-Meteo forecast: HTTP {r.status_code}")
        payload = r.json()
        data = payload["hourly"]
        times = pd.to_datetime(data["time"], utc=True)
        if race_time_utc is None:
            target = pd.Timestamp.now(tz="UTC")
        else:
            target = pd.Timestamp(race_time_utc)
        idx = int(np.argmin(np.abs(times - target)))
        out = {k: data[k][idx] for k in data if k != "time"}
        out["time"] = times[idx].isoformat()
        out["current"] = payload.get("current", {})

        # Asphalt proxy: deliberately labelled as an estimate in the UI.
        air = float(out.get("temperature_2m") or 20.0)
        solar = float(out.get("shortwave_radiation") or 0.0)
        cloud = float(out.get("cloud_cover") or 0.0)
        wind = float(out.get("wind_speed_10m") or 0.0)
        radiation_effect = min(25.0, max(0.0, solar / 34.0))
        cloud_discount = max(0.50, 1.0 - cloud / 165.0)
        wind_discount = max(0.65, 1.0 - wind / 85.0)
        out["track_temperature_estimate"] = air + radiation_effect * cloud_discount * wind_discount
        return out


def resolve_meeting(openf1: OpenF1Client, event: dict[str, Any], year: int):
    meetings = openf1.meetings(year)
    if not meetings:
        raise APIError("No OpenF1 meetings found for current season")

    key = event.get("key", "").lower()
    name = event.get("name", "").lower()
    loc = event.get("location", "").lower()
    country = event.get("country", "").lower()

    aliases = {
        "spain": ["spain", "madrid"],
        "barcelona-catalunya": ["barcelona", "catalunya"],
        "bahrain": ["bahrain", "malaysia", "kuala lumpur", "sepang"],
        "united-states": ["united states", "austin"],
        "great-britain": ["great britain", "silverstone"],
    }
    terms = set([key.replace("-", " "), name, loc, country] + aliases.get(key, []))

    def score(m):
        hay = " ".join(str(m.get(k, "")) for k in ["meeting_name", "meeting_official_name", "location", "country_name", "circuit_short_name"]).lower()
        return sum(3 if t and t in hay else 0 for t in terms)

    ranked = sorted(meetings, key=score, reverse=True)
    if score(ranked[0]) <= 0:
        raise APIError(f"Could not match {event.get('name')} to OpenF1")
    return ranked[0]


def _session_completed(session: dict[str, Any]) -> bool:
    end = session.get("date_end") or session.get("date_start")
    if not end:
        return True
    try:
        return pd.to_datetime(end, utc=True) <= pd.Timestamp.now(tz="UTC")
    except Exception:
        return True


def select_practice_sessions(sessions: list[dict[str, Any]]):
    practices = [s for s in sessions if s.get("session_name") in {"Practice 1", "Practice 2", "Practice 3"} and _session_completed(s)]
    order = {"Practice 3": 3, "Practice 2": 2, "Practice 1": 1}
    return sorted(practices, key=lambda s: order.get(s.get("session_name"), 0), reverse=True)


def _driver_number(drivers: list[dict[str, Any]], selected_name: str):
    target = selected_name.lower().strip()
    best = None
    for d in drivers:
        full = str(d.get("full_name", "")).lower().strip()
        broadcast = str(d.get("broadcast_name", "")).lower().strip()
        if target == full or target == broadcast or target in full or full in target:
            return int(d["driver_number"]), d
        if target.split()[-1] in full:
            best = d
    if best:
        return int(best["driver_number"]), best
    raise APIError(f"Driver {selected_name} not found in OpenF1 session")


def practice_dataset(openf1: OpenF1Client, session: dict, driver_number: int):
    laps = pd.DataFrame(openf1.laps(session["session_key"]))
    stints = pd.DataFrame(openf1.stints(session["session_key"]))
    if laps.empty or stints.empty:
        return pd.DataFrame(), laps, pd.DataFrame()

    dlaps = laps[laps["driver_number"] == driver_number].copy()
    dstints = stints[stints["driver_number"] == driver_number].copy()
    if dlaps.empty or dstints.empty:
        return pd.DataFrame(), laps, dstints

    dlaps = dlaps[dlaps["lap_duration"].notna()].copy()
    if "is_pit_out_lap" in dlaps:
        dlaps = dlaps[~dlaps["is_pit_out_lap"].fillna(False)]

    rows = []
    for _, st in dstints.iterrows():
        if pd.isna(st.get("lap_start")):
            continue
        ls = int(st["lap_start"])
        le = int(st["lap_end"]) if pd.notna(st.get("lap_end")) else int(dlaps["lap_number"].max())
        comp = str(st.get("compound", "")).upper()
        tyre0 = int(st.get("tyre_age_at_start") or 0)
        if comp not in {"SOFT", "MEDIUM", "HARD"}:
            continue
        sl = dlaps[(dlaps["lap_number"] >= ls) & (dlaps["lap_number"] <= le)].copy()
        if len(sl) < 4:
            continue
        sl["tyre_age"] = tyre0 + (sl["lap_number"] - ls)
        sl["compound"] = comp
        sl["stint_id"] = f"{session['session_key']}-{int(st.get('stint_number', 0))}"
        rows.append(sl[["lap_duration", "lap_number", "tyre_age", "compound", "stint_id"]])
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), laps, dstints


def race_pace_delta(all_laps: pd.DataFrame, driver_number: int):
    if all_laps is None or all_laps.empty or "lap_duration" not in all_laps:
        return 0.0
    clean = all_laps[all_laps["lap_duration"].notna()].copy()
    if "is_pit_out_lap" in clean:
        clean = clean[~clean["is_pit_out_lap"].fillna(False)]
    if len(clean) < 20:
        return 0.0
    med = clean.groupby("driver_number")["lap_duration"].median()
    if driver_number not in med.index:
        return 0.0
    return max(0.0, float(med.loc[driver_number] - med.min()))


def weekend_incident_risk(openf1: OpenF1Client, session: dict | None):
    if not session:
        return 0.30, "low"
    try:
        rc = openf1.race_control(session["session_key"])
    except Exception:
        return 0.30, "low"
    messages = " ".join(str(x.get("message", "")) for x in rc).upper()
    red = messages.count("RED FLAG")
    yellow = messages.count("YELLOW")
    sc = messages.count("SAFETY CAR")
    score = min(0.70, 0.24 + 0.07 * red + 0.015 * yellow + 0.05 * sc)
    return score, "medium"


def grid_position(openf1: OpenF1Client, sessions: list[dict[str, Any]], driver_number: int):
    races = [s for s in sessions if s.get("session_name") == "Race"]
    if not races:
        return None
    try:
        grid = openf1.starting_grid(races[0]["session_key"])
    except Exception:
        return None
    for row in grid:
        if int(row.get("driver_number", -1)) == driver_number:
            return int(row.get("position"))
    return None


def infer_tyre_inventory(stints: pd.DataFrame | None):
    # OpenF1 does not expose the authoritative remaining-set list. We use current-weekend
    # evidence only and clearly mark this output LOW confidence.
    used = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
    if stints is not None and not stints.empty:
        for comp in used:
            used[comp] = int((stints.get("compound", pd.Series(dtype=str)).astype(str).str.upper() == comp).sum() > 0)
    return {
        "SOFT": {"new": 1, "used": used["SOFT"]},
        "MEDIUM": {"new": 1, "used": used["MEDIUM"]},
        "HARD": {"new": 1, "used": used["HARD"]},
    }
