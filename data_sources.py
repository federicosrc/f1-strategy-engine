from __future__ import annotations

from datetime import datetime
from typing import Any
import requests
import numpy as np
import pandas as pd

OPENF1_BASE = "https://api.openf1.org/v1"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
GEOCODE_BASE = "https://geocoding-api.open-meteo.com/v1/search"


class APIError(RuntimeError):
    pass


class OpenF1Client:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "F1StrategyEngine/0.1"})

    def _get(self, endpoint: str, params: dict[str, Any] | None = None):
        r = self.session.get(
            f"{OPENF1_BASE}/{endpoint}",
            params=params or {},
            timeout=self.timeout,
        )
        if not r.ok:
            raise APIError(f"OpenF1 {endpoint}: HTTP {r.status_code}")
        return r.json()

    def meetings(self, year: int):
        return sorted(self._get("meetings", {"year": year}), key=lambda x: x.get("date_start", ""))

    def sessions(self, meeting_key: int):
        return self._get("sessions", {"meeting_key": meeting_key})

    def drivers(self, session_key: int):
        return self._get("drivers", {"session_key": session_key})

    def starting_grid(self, session_key: int):
        return self._get("starting_grid", {"session_key": session_key})

    def session_result(self, session_key: int):
        return self._get("session_result", {"session_key": session_key})

    def stints(self, session_key: int, driver_number: int | None = None):
        p = {"session_key": session_key}
        if driver_number is not None:
            p["driver_number"] = driver_number
        return self._get("stints", p)

    def laps(self, session_key: int, driver_number: int | None = None):
        p = {"session_key": session_key}
        if driver_number is not None:
            p["driver_number"] = driver_number
        return self._get("laps", p)

    def weather(self, session_key: int):
        return self._get("weather", {"session_key": session_key})

    def pit(self, session_key: int):
        return self._get("pit", {"session_key": session_key})

    def intervals(self, session_key: int, driver_number: int | None = None):
        p = {"session_key": session_key}
        if driver_number is not None:
            p["driver_number"] = driver_number
        return self._get("intervals", p)

    def positions(self, session_key: int, driver_number: int | None = None):
        p = {"session_key": session_key}
        if driver_number is not None:
            p["driver_number"] = driver_number
        return self._get("position", p)


class OpenMeteoClient:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def geocode(self, location: str, country: str | None = None):
        q = location if not country else f"{location}, {country}"
        r = requests.get(
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

    def forecast_at(self, lat: float, lon: float, race_time_utc: datetime):
        hourly = [
            "temperature_2m",
            "precipitation_probability",
            "precipitation",
            "wind_speed_10m",
            "shortwave_radiation",
            "cloud_cover",
        ]
        r = requests.get(
            OPEN_METEO_BASE,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(hourly),
                "timezone": "UTC",
                "forecast_days": 16,
            },
            timeout=self.timeout,
        )
        if not r.ok:
            raise APIError(f"Open-Meteo forecast: HTTP {r.status_code}")

        data = r.json()["hourly"]
        times = pd.to_datetime(data["time"], utc=True)
        target = pd.Timestamp(race_time_utc)
        idx = int(np.argmin(np.abs(times - target)))
        out = {k: data[k][idx] for k in data if k != "time"}
        out["time"] = times[idx].isoformat()

        air = float(out.get("temperature_2m") or 20.0)
        solar = float(out.get("shortwave_radiation") or 0.0)
        cloud = float(out.get("cloud_cover") or 0.0)
        radiation_effect = min(24.0, max(0.0, solar / 35.0))
        cloud_discount = max(0.55, 1.0 - cloud / 180.0)
        out["track_temperature_estimate"] = air + radiation_effect * cloud_discount
        return out


def _race_datetime(session):
    return pd.to_datetime(session["date_start"], utc=True).to_pydatetime()


def _latest(values, key):
    vals = [v.get(key) for v in values if v.get(key) is not None]
    return vals[-1] if vals else None


def _estimate_pit_loss(openf1, meeting):
    sessions = openf1.sessions(meeting["meeting_key"])
    races = [s for s in sessions if s.get("session_name") == "Race"]
    if races:
        try:
            pits = openf1.pit(races[0]["session_key"])
            lane = [float(p["lane_duration"]) for p in pits if p.get("lane_duration")]
            if len(lane) >= 3:
                return float(np.median(lane)), "high"
        except Exception:
            pass

    ctype = meeting.get("circuit_type", "")
    if "Street" in ctype:
        return 23.5, "medium"
    if "Road" in ctype:
        return 22.0, "medium"
    return 21.5, "medium"


def _estimate_sc_prior(meeting):
    ctype = meeting.get("circuit_type", "")
    if "Street" in ctype:
        return 0.50, "medium"
    if "Road" in ctype:
        return 0.35, "medium"
    return 0.28, "medium"


def _estimate_overtaking(meeting):
    ctype = meeting.get("circuit_type", "")
    if "Street" in ctype:
        return 0.78, "medium"
    if "Road" in ctype:
        return 0.60, "medium"
    return 0.48, "medium"


def _estimate_race_laps(openf1, race_session):
    try:
        res = openf1.session_result(race_session["session_key"])
        laps = [int(x["number_of_laps"]) for x in res if x.get("number_of_laps")]
        if laps:
            return max(laps), "high"
    except Exception:
        pass
    return 57, "low"


def _grid_position(openf1, race_session, driver_number):
    try:
        grid = openf1.starting_grid(race_session["session_key"])
        for x in grid:
            if int(x.get("driver_number", -1)) == int(driver_number):
                return int(x["position"]), "high"
    except Exception:
        pass
    return 10, "low"


def _estimate_tyre_sets_from_stints(openf1, sessions, driver_number):
    observed = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
    for s in sessions:
        if s.get("session_name") not in {"Practice 1", "Practice 2", "Practice 3", "Sprint"}:
            continue
        try:
            stints = openf1.stints(s["session_key"], driver_number)
        except Exception:
            continue
        for st in stints:
            comp = str(st.get("compound", "")).upper()
            if comp in observed:
                observed[comp] += 1

    # Low-confidence race allocation estimate. The UI keeps it overrideable.
    return {
        "SOFT": {"new": 1, "used": 1 if observed["SOFT"] else 0, "confidence": "low"},
        "MEDIUM": {"new": 1, "used": 1 if observed["MEDIUM"] else 0, "confidence": "low"},
        "HARD": {"new": 1, "used": 1 if observed["HARD"] else 0, "confidence": "low"},
    }


def practice_degradation_dataset(openf1, sessions, driver_number):
    rows = []
    for s in sessions:
        if s.get("session_name") not in {"Practice 1", "Practice 2", "Practice 3"}:
            continue
        try:
            laps = pd.DataFrame(openf1.laps(s["session_key"], driver_number))
            stints = pd.DataFrame(openf1.stints(s["session_key"], driver_number))
        except Exception:
            continue

        if laps.empty or stints.empty:
            continue

        laps = laps[laps["lap_duration"].notna()].copy()
        if "is_pit_out_lap" in laps:
            laps = laps[~laps["is_pit_out_lap"].fillna(False)]
        if laps.empty:
            continue

        for _, st in stints.iterrows():
            if pd.isna(st.get("lap_start")):
                continue
            ls = int(st["lap_start"])
            le = int(st["lap_end"]) if pd.notna(st.get("lap_end")) else int(laps["lap_number"].max())
            comp = str(st.get("compound", "")).upper()
            tyre0 = int(st.get("tyre_age_at_start") or 0)
            if comp not in {"SOFT", "MEDIUM", "HARD"}:
                continue

            sl = laps[(laps["lap_number"] >= ls) & (laps["lap_number"] <= le)].copy()
            if len(sl) < 4:
                continue
            sl["tyre_age"] = tyre0 + (sl["lap_number"] - ls)
            sl["compound"] = comp
            sl["stint_id"] = f"{s['session_key']}-{int(st.get('stint_number', 0))}"
            rows.append(sl[["lap_duration", "tyre_age", "compound", "stint_id"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _race_pace_delta(openf1, sessions, driver_number):
    fps = [s for s in sessions if s.get("session_name") in {"Practice 2", "Practice 3"}]
    for s in reversed(fps):
        try:
            laps = pd.DataFrame(openf1.laps(s["session_key"]))
        except Exception:
            continue
        if laps.empty or "lap_duration" not in laps:
            continue
        clean = laps[laps["lap_duration"].notna()].copy()
        if "is_pit_out_lap" in clean:
            clean = clean[~clean["is_pit_out_lap"].fillna(False)]
        if len(clean) < 20:
            continue
        med = clean.groupby("driver_number")["lap_duration"].median()
        if driver_number in med.index:
            return max(0.0, float(med.loc[driver_number] - med.min())), "medium"
    return 0.0, "low"


def build_prerace_snapshot(openf1, meteo, meeting, sessions, race_session, driver):
    from strategy_engine import estimate_degradation_from_practice

    confidence = {}

    grid, conf = _grid_position(openf1, race_session, int(driver["driver_number"]))
    confidence["Grid"] = conf

    pit_loss, conf = _estimate_pit_loss(openf1, meeting)
    confidence["Pit loss"] = conf

    sc_prob, conf = _estimate_sc_prior(meeting)
    confidence["SC/VSC prior"] = conf

    overtaking, conf = _estimate_overtaking(meeting)
    confidence["Overtaking difficulty"] = conf

    race_laps, conf = _estimate_race_laps(openf1, race_session)
    confidence["Race laps"] = conf

    race_pace_delta, conf = _race_pace_delta(openf1, sessions, int(driver["driver_number"]))
    confidence["Race pace"] = conf

    weather = {}
    try:
        geo = meteo.geocode(
            meeting.get("location", meeting.get("country_name", "")),
            meeting.get("country_name"),
        )
        weather = meteo.forecast_at(
            geo["latitude"],
            geo["longitude"],
            _race_datetime(race_session),
        )
        confidence["Weather"] = "high"
        confidence["Track temperature"] = "medium"
    except Exception:
        confidence["Weather"] = "low"
        confidence["Track temperature"] = "low"

    practice_df = practice_degradation_dataset(openf1, sessions, int(driver["driver_number"]))
    degradation = estimate_degradation_from_practice(practice_df)
    confidence["Tyre degradation"] = "medium" if degradation else "low"

    availability = _estimate_tyre_sets_from_stints(openf1, sessions, int(driver["driver_number"]))
    confidence["Tyre sets remaining"] = "low"

    automatic_share = sum(v != "low" for v in confidence.values()) / max(1, len(confidence))

    auto = {
        "grid_position": grid,
        "race_laps": race_laps,
        "air_temperature": weather.get("temperature_2m", 25.0),
        "track_temperature": weather.get("track_temperature_estimate", 40.0),
        "rain_probability": (weather.get("precipitation_probability") or 0) / 100.0,
        "pit_loss_green": pit_loss,
        "pit_loss_sc": pit_loss * 0.55,
        "sc_probability": sc_prob,
        "overtaking_difficulty": overtaking,
        "undercut_power": min(
            0.95,
            0.45 + 0.35 * overtaking + 0.35 * min(0.30, max(degradation.values(), default=0.08)),
        ),
        "team_risk": 0.50,
        "race_pace_delta": race_pace_delta,
    }
    return {
        "auto": auto,
        "confidence": confidence,
        "practice": {"degradation": degradation, "rows": len(practice_df)},
        "tyre_availability": availability,
        "automatic_share": automatic_share,
    }


def build_live_snapshot(openf1, meeting, race_session, driver):
    driver_number = int(driver["driver_number"])
    base = {
        "auto": {},
        "confidence": {"Live timing": "high", "Tyre sets remaining": "low"},
        "practice": {"degradation": {}},
        "tyre_availability": {
            "SOFT": {"new": 1, "used": 1, "confidence": "low"},
            "MEDIUM": {"new": 1, "used": 1, "confidence": "low"},
            "HARD": {"new": 1, "used": 1, "confidence": "low"},
        },
        "automatic_share": 0.75,
    }

    try:
        weather = openf1.weather(race_session["session_key"])
        base["auto"]["air_temperature"] = float(_latest(weather, "air_temperature") or 25)
        base["auto"]["track_temperature"] = float(_latest(weather, "track_temperature") or 40)
        base["auto"]["rain_probability"] = 0.95 if _latest(weather, "rainfall") else 0.02
    except Exception:
        pass

    grid, _ = _grid_position(openf1, race_session, driver_number)
    base["auto"]["grid_position"] = grid

    pit_loss, _ = _estimate_pit_loss(openf1, meeting)
    sc_prior, _ = _estimate_sc_prior(meeting)
    overtaking, _ = _estimate_overtaking(meeting)

    base["auto"].update({
        "pit_loss_green": pit_loss,
        "pit_loss_sc": pit_loss * 0.55,
        "sc_probability": sc_prior,
        "overtaking_difficulty": overtaking,
        "undercut_power": min(0.95, 0.50 + 0.40 * overtaking),
        "race_laps": 57,
        "race_pace_delta": 0.0,
        "team_risk": 0.50,
    })

    live = {}
    try:
        stints = openf1.stints(race_session["session_key"], driver_number)
        if stints:
            last = sorted(stints, key=lambda x: x.get("stint_number", 0))[-1]
            live["compound"] = last.get("compound", "—")
            live["lap"] = last.get("lap_end") or last.get("lap_start")
    except Exception:
        pass

    try:
        pos = openf1.positions(race_session["session_key"], driver_number)
        if pos:
            live["position"] = pos[-1].get("position")
    except Exception:
        pass

    try:
        ints = openf1.intervals(race_session["session_key"], driver_number)
        if ints:
            live["gap_ahead"] = ints[-1].get("interval")
    except Exception:
        pass

    base["live"] = live
    return base
