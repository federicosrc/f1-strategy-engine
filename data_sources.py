from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import logging

import numpy as np
import pandas as pd
import requests

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
GEOCODE_BASE = "https://geocoding-api.open-meteo.com/v1/search"


class APIError(RuntimeError):
    pass


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
                "current": ",".join(
                    ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
                ),
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
        target = pd.Timestamp.now(tz="UTC") if race_time_utc is None else pd.Timestamp(race_time_utc)
        idx = int(np.argmin(np.abs(times - target)))
        out = {k: data[k][idx] for k in data if k != "time"}
        out["time"] = times[idx].isoformat()
        out["current"] = payload.get("current", {})

        air = float(out.get("temperature_2m") or 20.0)
        solar = float(out.get("shortwave_radiation") or 0.0)
        cloud = float(out.get("cloud_cover") or 0.0)
        wind = float(out.get("wind_speed_10m") or 0.0)
        radiation_effect = min(25.0, max(0.0, solar / 34.0))
        cloud_discount = max(0.50, 1.0 - cloud / 165.0)
        wind_discount = max(0.65, 1.0 - wind / 85.0)
        out["track_temperature_estimate"] = air + radiation_effect * cloud_discount * wind_discount
        return out


class FastF1DataClient:
    """Pre-race current-weekend analysis powered by FastF1 only."""

    def __init__(self, cache_dir: str = "/tmp/f1_strategy_fastf1_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fastf1 = None
        self._import_error = None
        try:
            import fastf1
            self._fastf1 = fastf1
            try:
                fastf1.Cache.enable_cache(str(self.cache_dir))
            except Exception:
                pass
            for name in ("fastf1", "fastf1.req", "fastf1.core", "fastf1.events"):
                logging.getLogger(name).setLevel(logging.WARNING)
        except Exception as exc:
            self._import_error = str(exc)

    @property
    def available(self):
        return self._fastf1 is not None

    def _load_session(self, year: int, round_number: int, code: str, load_laps: bool = True):
        if self._fastf1 is None:
            raise APIError(f"FastF1 unavailable: {self._import_error or 'not installed'}")
        session = self._fastf1.get_session(int(year), int(round_number), code)
        session.load(laps=load_laps, telemetry=False, weather=False, messages=False)
        return session

    @staticmethod
    def _match_driver(results: pd.DataFrame, selected_driver: str):
        if results is None or len(results) == 0:
            return None
        target = selected_driver.strip().lower()
        surname = target.split()[-1]
        for _, row in results.iterrows():
            full = str(row.get("FullName", "")).strip()
            last = str(row.get("LastName", "")).strip()
            if target == full.lower() or surname == last.lower() or surname in full.lower():
                return {
                    "abbreviation": str(row.get("Abbreviation", "")).strip(),
                    "driver_number": str(row.get("DriverNumber", "")).strip(),
                    "team_name": str(row.get("TeamName", "")).strip(),
                    "full_name": full or selected_driver,
                }
        return None

    @staticmethod
    def _clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
        if laps is None or laps.empty or "LapTime" not in laps:
            return pd.DataFrame()
        df = laps[laps["LapTime"].notna()].copy()

        for col in ("PitOutTime", "PitInTime"):
            if col in df:
                df = df[df[col].isna()]
        if "Deleted" in df:
            df = df[~df["Deleted"].fillna(False)]
        if "FastF1Generated" in df:
            df = df[~df["FastF1Generated"].fillna(False)]
        if "IsAccurate" in df:
            accurate = df[df["IsAccurate"].fillna(False)]
            if len(accurate) >= 5:
                df = accurate
        if "TrackStatus" in df:
            green = df[df["TrackStatus"].astype(str).eq("1")]
            if len(green) >= 5:
                df = green
        if "Compound" in df:
            df = df[df["Compound"].astype(str).str.upper().isin(["SOFT", "MEDIUM", "HARD"])]

        df["lap_seconds"] = df["LapTime"].dt.total_seconds()
        if len(df) >= 8:
            lo = df["lap_seconds"].quantile(0.03)
            hi = df["lap_seconds"].quantile(0.95)
            df = df[(df["lap_seconds"] >= lo) & (df["lap_seconds"] <= hi)]
        return df

    def _driver_laps(self, session, selected_driver: str):
        match = self._match_driver(session.results, selected_driver)
        if match is None or session.laps is None or session.laps.empty:
            return pd.DataFrame(), match

        laps = session.laps.copy()
        if match["abbreviation"] and "Driver" in laps:
            laps = laps[laps["Driver"].astype(str).eq(match["abbreviation"])]
        elif match["driver_number"] and "DriverNumber" in laps:
            laps = laps[laps["DriverNumber"].astype(str).eq(match["driver_number"])]
        else:
            laps = pd.DataFrame()

        return self._clean_laps(laps), match

    @staticmethod
    def _degradation_dataset(clean_laps: pd.DataFrame, code: str):
        needed = {"Compound", "TyreLife", "Stint", "lap_seconds"}
        if clean_laps is None or clean_laps.empty or not needed.issubset(clean_laps.columns):
            return pd.DataFrame()

        df = clean_laps.dropna(subset=["Compound", "TyreLife", "Stint"]).copy()
        df["compound"] = df["Compound"].astype(str).str.upper()
        df["tyre_age"] = pd.to_numeric(df["TyreLife"], errors="coerce")
        df["lap_duration"] = pd.to_numeric(df["lap_seconds"], errors="coerce")
        df["stint_id"] = code + "-" + df["Stint"].astype(str)
        df = df[df["compound"].isin(["SOFT", "MEDIUM", "HARD"])].dropna(
            subset=["tyre_age", "lap_duration"]
        )

        counts = df.groupby("stint_id").size()
        valid = counts[counts >= 4].index
        return df[df["stint_id"].isin(valid)][
            ["lap_duration", "tyre_age", "compound", "stint_id"]
        ].copy()

    @staticmethod
    def _race_pace_delta(session, match):
        if match is None or session.laps is None or session.laps.empty:
            return 0.0
        clean = FastF1DataClient._clean_laps(session.laps)
        if clean.empty or "Driver" not in clean or "Compound" not in clean:
            return 0.0

        target = match["abbreviation"]
        deltas, weights = [], []
        for compound, cg in clean.groupby(clean["Compound"].astype(str).str.upper()):
            if compound not in {"SOFT", "MEDIUM", "HARD"}:
                continue
            grouped = cg.groupby("Driver")["lap_seconds"].agg(["median", "count"])
            grouped = grouped[grouped["count"] >= 4]
            if target not in grouped.index or grouped.empty:
                continue
            ref = float(grouped["median"].min())
            deltas.append(max(0.0, float(grouped.loc[target, "median"] - ref)))
            weights.append(float(grouped.loc[target, "count"]))

        return float(np.average(deltas, weights=weights)) if deltas else 0.0


    @staticmethod
    def _all_driver_pace(session):
        """Race-pace deltas for the complete field, compound-normalised."""
        if session is None or session.laps is None or session.laps.empty:
            return {}

        clean = FastF1DataClient._clean_laps(session.laps)
        if clean.empty or "Driver" not in clean or "Compound" not in clean:
            return {}

        evidence = {}
        for compound, cg in clean.groupby(clean["Compound"].astype(str).str.upper()):
            if compound not in {"SOFT", "MEDIUM", "HARD"}:
                continue
            grouped = cg.groupby("Driver")["lap_seconds"].agg(["median", "count"])
            grouped = grouped[grouped["count"] >= 4]
            if grouped.empty:
                continue

            ref = float(grouped["median"].min())
            for driver, row in grouped.iterrows():
                evidence.setdefault(str(driver), []).append(
                    (max(0.0, float(row["median"] - ref)), float(row["count"]))
                )

        pace = {}
        for driver, values in evidence.items():
            deltas = [v[0] for v in values]
            weights = [v[1] for v in values]
            pace[driver] = float(np.average(deltas, weights=weights))
        return pace

    def _qualifying_grid_all(self, year, round_number):
        try:
            session = self._load_session(year, round_number, "Q", load_laps=False)
            results = session.results
            grid = {}
            if results is None or len(results) == 0:
                return grid, "low"

            for _, row in results.iterrows():
                abbr = str(row.get("Abbreviation", "")).strip()
                if not abbr:
                    continue
                pos = row.get("Position")
                try:
                    pos = int(float(pos)) if pd.notna(pos) else None
                except Exception:
                    pos = None
                grid[abbr] = {
                    "grid_position": pos,
                    "driver_name": str(row.get("FullName", "")).strip(),
                    "team_name": str(row.get("TeamName", "")).strip(),
                }
            return grid, "medium" if grid else "low"
        except Exception:
            return {}, "low"

    @staticmethod
    def _build_grid_model(session, pace_map, qualifying_grid, grid_confidence):
        if session is None or session.results is None or len(session.results) == 0:
            return []

        known_paces = list(pace_map.values())
        fallback_pace = float(np.median(known_paces) + 0.65) if known_paces else 0.85
        rows = []

        for idx, (_, row) in enumerate(session.results.iterrows(), start=1):
            abbr = str(row.get("Abbreviation", "")).strip()
            if not abbr:
                continue

            q = qualifying_grid.get(abbr, {})
            full_name = q.get("driver_name") or str(row.get("FullName", "")).strip() or abbr
            team_name = q.get("team_name") or str(row.get("TeamName", "")).strip()
            grid_pos = q.get("grid_position")
            pace = pace_map.get(abbr, fallback_pace)

            rows.append({
                "driver_name": full_name,
                "abbreviation": abbr,
                "team_name": team_name,
                "grid_position": grid_pos,
                "race_pace_delta": float(max(0.0, pace)),
                "pace_confidence": "medium" if abbr in pace_map else "low",
                "grid_confidence": grid_confidence if grid_pos is not None else "low",
            })

        return rows

    @staticmethod
    def _incident_prior(session):
        try:
            laps = session.laps
            if laps is None or laps.empty or "TrackStatus" not in laps:
                return 0.30, "low"
            abnormal = (~laps["TrackStatus"].astype(str).eq("1")).mean()
            return float(np.clip(0.24 + abnormal * 1.8, 0.22, 0.65)), "medium"
        except Exception:
            return 0.30, "low"

    @staticmethod
    def _inventory_evidence(driver_laps):
        observed = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
        for df in driver_laps:
            if df is None or df.empty or "Compound" not in df:
                continue
            for comp in observed:
                if (df["Compound"].astype(str).str.upper() == comp).any():
                    observed[comp] = 1

        # Deliberately permissive because this is not the official remaining-set list.
        return {
            "SOFT": {"new": 1, "used": observed["SOFT"], "confidence": "low"},
            "MEDIUM": {"new": 2, "used": observed["MEDIUM"], "confidence": "low"},
            "HARD": {"new": 2, "used": observed["HARD"], "confidence": "low"},
        }

    def _qualifying_position(self, year, round_number, selected_driver):
        try:
            session = self._load_session(year, round_number, "Q", load_laps=False)
            match = self._match_driver(session.results, selected_driver)
            if match is None:
                return None, "low"
            results = session.results
            if "Abbreviation" not in results:
                return None, "low"
            row = results[results["Abbreviation"].astype(str).eq(match["abbreviation"])]
            if row.empty:
                return None, "low"
            pos = row.iloc[0].get("Position")
            if pd.notna(pos):
                return int(float(pos)), "medium"
        except Exception:
            pass
        return None, "low"

    def analyse_target_context(
        self,
        year: int,
        round_number: int,
        selected_driver: str,
        lookback: int = 4,
    ):
        """Driver-specific context for Target Outcome.

        Prefer current-weekend FP/Q evidence. If the selected weekend has not
        started yet, derive a current-season-only prior from the most recent
        completed qualifying sessions. This prevents every driver from collapsing
        to the same generic P10 / zero-pace fallback.
        """
        weekend = self.analyse_weekend(year, round_number, selected_driver)
        if weekend.get("available") and (
            weekend.get("grid_position") is not None
            or weekend.get("grid_model")
            or weekend.get("pace_confidence") in {"medium", "high"}
        ):
            weekend["target_context"] = "current_weekend"
            return weekend

        if not self.available or int(round_number) <= 1:
            weekend["target_context"] = "generic_fallback"
            return weekend

        sessions = []
        start_round = max(1, int(round_number) - int(max(1, lookback)))
        for rnd in range(int(round_number) - 1, start_round - 1, -1):
            try:
                q = self._load_session(year, rnd, "Q", load_laps=False)
                if q.results is None or len(q.results) < 10:
                    continue
                sessions.append(q)
            except Exception:
                continue

        if not sessions:
            weekend["target_context"] = "generic_fallback"
            return weekend

        driver_history = {}
        metadata = {}
        for q in sessions:
            for _, row in q.results.iterrows():
                abbr = str(row.get("Abbreviation", "")).strip()
                full = str(row.get("FullName", "")).strip()
                if not abbr:
                    continue
                pos = row.get("Position")
                try:
                    pos = float(pos) if pd.notna(pos) else None
                except Exception:
                    pos = None
                if pos is None or not np.isfinite(pos):
                    continue
                driver_history.setdefault(abbr, []).append(float(pos))
                metadata[abbr] = {
                    "driver_name": full or abbr,
                    "team_name": str(row.get("TeamName", "")).strip(),
                }

        if not driver_history:
            weekend["target_context"] = "generic_fallback"
            return weekend

        summaries = []
        for abbr, positions in driver_history.items():
            avg_pos = float(np.average(positions, weights=np.linspace(1.0, 1.35, len(positions))))
            summaries.append((abbr, avg_pos))
        summaries.sort(key=lambda x: x[1])
        best_avg = summaries[0][1]

        grid_model = []
        for predicted_grid, (abbr, avg_pos) in enumerate(summaries, start=1):
            meta = metadata.get(abbr, {})
            pace_delta = float(np.clip((avg_pos - best_avg) * 0.060, 0.0, 1.80))
            grid_model.append({
                "driver_name": meta.get("driver_name", abbr),
                "abbreviation": abbr,
                "team_name": meta.get("team_name", ""),
                "grid_position": predicted_grid,
                "race_pace_delta": pace_delta,
                "pace_confidence": "low",
                "grid_confidence": "low",
            })

        selected_match = None
        target_name = selected_driver.strip().lower()
        for row in grid_model:
            name = str(row.get("driver_name", "")).strip().lower()
            if name == target_name or target_name in name or name in target_name:
                selected_match = row
                break

        if selected_match is None:
            # Try FastF1 name matching against the newest qualifying result.
            try:
                matched = self._match_driver(sessions[0].results, selected_driver)
            except Exception:
                matched = None
            if matched:
                abbr = matched.get("abbreviation")
                selected_match = next((r for r in grid_model if r.get("abbreviation") == abbr), None)

        if selected_match is None:
            weekend["target_context"] = "generic_fallback"
            return weekend

        out = dict(weekend)
        out.update({
            "available": True,
            "practice_name": "Current-season prior",
            "team_name": selected_match.get("team_name", ""),
            "grid_position": int(selected_match["grid_position"]),
            "grid_confidence": "low",
            "pace_delta": float(selected_match["race_pace_delta"]),
            "pace_confidence": "low",
            "grid_model": grid_model,
            "grid_model_confidence": "low",
            "target_context": "current_season_prior",
        })
        return out

    def analyse_weekend(self, year: int, round_number: int, selected_driver: str):
        fallback = {
            "available": False,
            "source": "FastF1 fallback",
            "practice_name": "No completed practice data",
            "team_name": "",
            "grid_position": None,
            "grid_confidence": "low",
            "degradation": {},
            "degradation_confidence": "low",
            "pace_delta": 0.0,
            "pace_confidence": "low",
            "incident_risk": 0.30,
            "incident_confidence": "low",
            "inventory": {
                "SOFT": {"new": 1, "used": 1, "confidence": "low"},
                "MEDIUM": {"new": 2, "used": 1, "confidence": "low"},
                "HARD": {"new": 2, "used": 1, "confidence": "low"},
            },
            "practice_rows": 0,
            "sessions_used": [],
            "grid_model": [],
            "grid_model_confidence": "low",
        }
        if not self.available:
            return fallback

        sessions_used, datasets, driver_laps = [], [], []
        best_session = best_match = None
        best_pace = 0.0
        best_count = -1

        for code in ("FP2", "FP3", "FP1"):
            try:
                session = self._load_session(year, round_number, code, load_laps=True)
                clean, match = self._driver_laps(session, selected_driver)
                if match is None or clean.empty:
                    continue

                sessions_used.append(code)
                driver_laps.append(clean)
                ds = self._degradation_dataset(clean, code)
                if not ds.empty:
                    datasets.append(ds)

                pace = self._race_pace_delta(session, match)
                count = len(clean)
                # Prefer FP2 for race pace, otherwise the richest available session.
                score = count + (1000 if code == "FP2" else 0)
                if score > best_count:
                    best_count = score
                    best_session, best_match, best_pace = session, match, pace
            except Exception:
                continue

        if best_session is None:
            return fallback

        from strategy_engine_v21 import estimate_degradation_from_practice
        merged = pd.concat(datasets, ignore_index=True) if datasets else pd.DataFrame()
        degradation = estimate_degradation_from_practice(merged)
        rows = len(merged)

        pace_map = self._all_driver_pace(best_session)
        qualifying_grid, all_grid_conf = self._qualifying_grid_all(year, round_number)
        grid_model = self._build_grid_model(best_session, pace_map, qualifying_grid, all_grid_conf)

        selected_abbr = best_match.get("abbreviation", "")
        best_pace = float(pace_map.get(selected_abbr, best_pace or 0.0))

        selected_grid_row = qualifying_grid.get(selected_abbr, {})
        grid = selected_grid_row.get("grid_position")
        grid_conf = all_grid_conf if grid is not None else "low"

        incident, incident_conf = self._incident_prior(best_session)

        useful_rivals = sum(
            1 for r in grid_model
            if r.get("grid_position") is not None
            and r.get("pace_confidence") in {"medium", "high"}
        )
        grid_model_conf = "high" if useful_rivals >= 16 else "medium" if useful_rivals >= 10 else "low"

        fallback.update({
            "available": True,
            "source": "FastF1 current weekend",
            "practice_name": " + ".join(sessions_used),
            "team_name": best_match.get("team_name", ""),
            "grid_position": grid,
            "grid_confidence": grid_conf,
            "degradation": degradation,
            "degradation_confidence": "high" if rows >= 16 else "medium" if rows >= 8 else "low",
            "pace_delta": float(best_pace),
            "pace_confidence": "medium",
            "incident_risk": incident,
            "incident_confidence": incident_conf,
            "inventory": self._inventory_evidence(driver_laps),
            "practice_rows": int(rows),
            "sessions_used": sessions_used,
            "grid_model": grid_model,
            "grid_model_confidence": grid_model_conf,
        })
        return fallback
