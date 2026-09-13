from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Dict, Any, Iterable
import math
import numpy as np
import pandas as pd


DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")
ALL_COMPOUNDS = DRY_COMPOUNDS + WET_COMPOUNDS


@dataclass
class CircuitProfile:
    race_laps: int
    pit_loss_green: float
    pit_loss_sc: float
    overtaking_difficulty: float
    undercut_power: float
    track_temperature: float


@dataclass
class TyreModel:
    compound: str
    pace_delta: float
    degradation: float


@dataclass
class DriverContext:
    driver_name: str
    team_name: str
    grid_position: int
    race_pace_delta: float
    team_risk: float = 0.50


@dataclass
class SimulationInputs:
    circuit: CircuitProfile
    driver: DriverContext
    tyres: Dict[str, TyreModel]
    tyre_sets: Dict[str, Dict[str, int]]
    sc_probability: float
    rain_probability: float
    simulations: int = 30000
    objective: str = "Best expected finish"
    mandatory_race_compounds: tuple[str, ...] = ("HARD", "MEDIUM")
    rivals: list[dict[str, Any]] | None = None
    neutralisation_mode: str = "NONE"  # NONE | SC | VSC
    allowed_compounds: tuple[str, ...] = DRY_COMPOUNDS
    weather_mode: str = "EXPECTED"
    weather_timeline: list[dict[str, Any]] | None = None
    neutralisation_lap: int | None = None  # None = random timing when SC/VSC is selected


def _robust_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 4 or np.ptp(x) < 2:
        return None
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad > 0:
        keep = np.abs(y - med) <= 3.5 * 1.4826 * mad
        x, y = x[keep], y[keep]
    if len(x) < 4:
        return None
    return float(np.polyfit(x, y, 1)[0])


def estimate_degradation_from_practice(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    out = {}
    for compound, g in df.groupby("compound"):
        centered = []
        for _, stint in g.groupby("stint_id"):
            if len(stint) < 4:
                continue
            s = stint.sort_values("tyre_age").copy()
            s["x"] = s["tyre_age"] - s["tyre_age"].mean()
            s["y"] = s["lap_duration"] - s["lap_duration"].mean()
            centered.append(s[["x", "y"]])
        if not centered:
            continue
        c = pd.concat(centered, ignore_index=True)
        slope = _robust_slope(c["x"], c["y"])
        if slope is not None:
            out[str(compound).upper()] = float(np.clip(slope, 0.01, 0.35))
    return out


def total_sets(sets: dict, compound: str) -> int:
    row = sets.get(compound, {})
    return int(row.get("new", 0)) + int(row.get("used", 0))


def strategy_requirements(compounds: Iterable[str]) -> dict[str, int]:
    req = {c: 0 for c in ALL_COMPOUNDS}
    for c in compounds:
        if c in req:
            req[c] += 1
    return req


def validate_strategy(
    compounds: list[str] | tuple[str, ...],
    inputs: SimulationInputs,
    wet_used: bool | None = None,
) -> tuple[bool, list[str]]:
    """Validate a pre-race strategy against tyre availability and dry/wet rule logic.

    If Intermediate or Wet tyres are used, the dry-race requirement to use two
    different dry specifications is not applied.
    """
    compounds = [str(c).upper() for c in compounds]
    reasons = []
    allowed = tuple(str(c).upper() for c in inputs.allowed_compounds)

    if len(compounds) not in (2, 3):
        reasons.append("Choose exactly 1 or 2 pit stops.")

    invalid = [c for c in compounds if c not in allowed]
    if invalid:
        reasons.append(
            "Tyre choice is not available in the selected weather scenario: "
            + ", ".join(invalid)
        )

    actual_wet_used = any(c in WET_COMPOUNDS for c in compounds)
    if wet_used is None:
        wet_used = actual_wet_used

    # FIA-style dry tyre rule is only enforced when no Intermediate/Wet is used.
    if not wet_used:
        dry_used = [c for c in compounds if c in DRY_COMPOUNDS]
        if len(set(dry_used)) < 2:
            reasons.append(
                "Dry-race rule: at least two different dry tyre specifications must be used."
            )
        if not any(c in set(inputs.mandatory_race_compounds) for c in dry_used):
            reasons.append(
                "Dry-race rule: the strategy must include at least one mandatory Race tyre specification."
            )

    req = strategy_requirements(compounds)
    for comp, needed in req.items():
        if needed <= 0:
            continue
        have = total_sets(inputs.tyre_sets, comp)
        if needed > have:
            reasons.append(
                f"Tyre availability: {needed} {comp} sets required, only {have} available."
            )

    return len(reasons) == 0, reasons

def legal_next_compounds(
    partial: list[str],
    total_stints: int,
    inputs: SimulationInputs,
) -> list[str]:
    """Return tyre choices that still permit at least one legal completion."""
    partial = [c.upper() for c in partial]
    if len(partial) >= total_stints:
        return []

    allowed = tuple(str(c).upper() for c in inputs.allowed_compounds)
    options = []
    remaining_slots = total_stints - len(partial) - 1

    for candidate in allowed:
        trial = partial + [candidate]
        req = strategy_requirements(trial)
        if any(req[c] > total_sets(inputs.tyre_sets, c) for c in ALL_COMPOUNDS):
            continue

        if remaining_slots == 0:
            valid, _ = validate_strategy(trial, inputs)
            if valid:
                options.append(candidate)
            continue

        feasible_future = False
        for suffix in product(allowed, repeat=remaining_slots):
            full = trial + list(suffix)
            valid, _ = validate_strategy(full, inputs)
            if valid:
                feasible_future = True
                break
        if feasible_future:
            options.append(candidate)

    return options

def _effective_degradation(tyre: TyreModel, track_temp: float) -> float:
    heat = max(0.0, track_temp - 35.0)
    return tyre.degradation * (1.0 + 0.012 * heat)


def _normalised_weather_timeline(inputs: SimulationInputs) -> list[dict[str, Any]]:
    race_laps = int(inputs.circuit.race_laps)
    if not inputs.weather_timeline:
        return [{
            "start_lap": 1,
            "end_lap": race_laps,
            "mode": str(inputs.weather_mode or "EXPECTED").upper(),
            "track_temp": float(inputs.circuit.track_temperature),
            "rain_probability": float(inputs.rain_probability),
        }]

    rows = []
    for row in inputs.weather_timeline:
        try:
            start = int(row.get("start_lap", 1))
            end = int(row.get("end_lap", race_laps))
        except Exception:
            continue
        start = max(1, min(race_laps, start))
        end = max(start, min(race_laps, end))
        rows.append({
            "start_lap": start,
            "end_lap": end,
            "mode": str(row.get("mode", "EXPECTED")).upper(),
            "track_temp": float(row.get("track_temp", inputs.circuit.track_temperature)),
            "rain_probability": float(row.get("rain_probability", inputs.rain_probability)),
        })

    if not rows:
        return _normalised_weather_timeline(
            SimulationInputs(
                circuit=inputs.circuit,
                driver=inputs.driver,
                tyres=inputs.tyres,
                tyre_sets=inputs.tyre_sets,
                sc_probability=inputs.sc_probability,
                rain_probability=inputs.rain_probability,
                simulations=inputs.simulations,
                objective=inputs.objective,
                mandatory_race_compounds=inputs.mandatory_race_compounds,
                rivals=inputs.rivals,
                neutralisation_mode=inputs.neutralisation_mode,
                allowed_compounds=inputs.allowed_compounds,
                weather_mode=inputs.weather_mode,
                weather_timeline=None,
                neutralisation_lap=inputs.neutralisation_lap,
            )
        )

    return sorted(rows, key=lambda r: r["start_lap"])


def _weather_adjustment_scalar(compound: str, mode: str, rain_probability: float = 0.0) -> float:
    comp = str(compound).upper()
    mode = str(mode).upper()

    dry = {
        "SOFT": 0.0,
        "MEDIUM": 0.0,
        "HARD": 0.0,
        "INTERMEDIATE": 2.4,
        "WET": 4.2,
    }
    rain = {
        "SOFT": 4.8,
        "MEDIUM": 4.6,
        "HARD": 4.5,
        "INTERMEDIATE": -0.6,
        "WET": 0.5,
    }
    heavy = {
        "SOFT": 7.2,
        "MEDIUM": 7.0,
        "HARD": 6.8,
        "INTERMEDIATE": 0.9,
        "WET": -0.8,
    }

    if mode == "HEAVY_RAIN":
        return float(heavy.get(comp, 0.0))
    if mode == "RAIN":
        return float(rain.get(comp, 0.0))
    if mode == "CHANGEABLE":
        return float(0.55 * dry.get(comp, 0.0) + 0.45 * rain.get(comp, 0.0))
    if mode == "EXPECTED":
        p = float(np.clip(rain_probability, 0.0, 1.0))
        return float((1.0 - p) * dry.get(comp, 0.0) + p * rain.get(comp, 0.0))
    return float(dry.get(comp, 0.0))


def _phase_overlap(start_lap: int, end_lap: int, phase: dict[str, Any]) -> int:
    lo = max(start_lap, int(phase["start_lap"]))
    hi = min(end_lap, int(phase["end_lap"]))
    return max(0, hi - lo + 1)


def _stint_weather_profile(
    compound: str,
    start_lap: int,
    end_lap: int,
    inputs: SimulationInputs,
) -> tuple[float, float]:
    """Return (weather pace cost, weighted-average track temp) for one stint."""
    timeline = _normalised_weather_timeline(inputs)
    total_laps = max(1, end_lap - start_lap + 1)
    weather_cost = 0.0
    weighted_temp = 0.0
    covered = 0

    for phase in timeline:
        overlap = _phase_overlap(start_lap, end_lap, phase)
        if overlap <= 0:
            continue
        weather_cost += overlap * _weather_adjustment_scalar(
            compound,
            phase["mode"],
            phase.get("rain_probability", 0.0),
        )
        weighted_temp += overlap * float(phase.get("track_temp", inputs.circuit.track_temperature))
        covered += overlap

    if covered < total_laps:
        missing = total_laps - covered
        weather_cost += missing * _weather_adjustment_scalar(
            compound,
            inputs.weather_mode,
            inputs.rain_probability,
        )
        weighted_temp += missing * float(inputs.circuit.track_temperature)
        covered += missing

    return float(weather_cost), float(weighted_temp / max(1, covered))


def _deterministic_stint_cost_at(
    compound: str,
    start_lap: int,
    end_lap: int,
    inputs: SimulationInputs,
) -> float:
    laps = max(0, end_lap - start_lap + 1)
    if laps <= 0:
        return 0.0
    tyre = inputs.tyres[compound]
    weather_cost, avg_track_temp = _stint_weather_profile(
        compound, start_lap, end_lap, inputs
    )
    deg = _effective_degradation(tyre, avg_track_temp)
    return tyre.pace_delta * laps + weather_cost + deg * laps * (laps - 1) / 2.0


def _deterministic_stint_cost(compound: str, laps: int, inputs: SimulationInputs) -> float:
    # Backwards-compatible helper for callers without an explicit race-lap range.
    return _deterministic_stint_cost_at(compound, 1, max(1, laps), inputs)


def _timing_leverage(pit_laps: list[int], inputs: SimulationInputs) -> float:
    # Expected track-position benefit from stopping around the useful undercut zone.
    if not pit_laps:
        return 0.0
    race_laps = inputs.circuit.race_laps
    centrality = np.mean([1.0 - min(1.0, abs((p / race_laps) - 0.42) / 0.42) for p in pit_laps])
    return inputs.circuit.undercut_power * centrality * len(pit_laps) * 1.8


def optimise_pit_laps(compounds: list[str], inputs: SimulationInputs) -> tuple[list[int], list[int], float]:
    """Optimise pit timing for a fixed compound sequence, including weather evolution."""
    n_stints = len(compounds)
    laps = int(inputs.circuit.race_laps)
    min_stint = max(5, int(round(laps * 0.09)))
    pit_cost = inputs.circuit.pit_loss_green
    best = None

    if n_stints == 2:
        for p1 in range(min_stint, laps - min_stint + 1):
            lengths = [p1, laps - p1]
            tyre_cost = (
                _deterministic_stint_cost_at(compounds[0], 1, p1, inputs)
                + _deterministic_stint_cost_at(compounds[1], p1 + 1, laps, inputs)
            )
            traffic = inputs.circuit.overtaking_difficulty * 3.0
            leverage = _timing_leverage([p1], inputs)
            total = tyre_cost + pit_cost + traffic - leverage
            if best is None or total < best[0]:
                best = (total, [p1], lengths)

    elif n_stints == 3:
        step = 1 if laps <= 65 else 2
        for p1 in range(min_stint, laps - 2 * min_stint + 1, step):
            for p2 in range(p1 + min_stint, laps - min_stint + 1, step):
                lengths = [p1, p2 - p1, laps - p2]
                tyre_cost = (
                    _deterministic_stint_cost_at(compounds[0], 1, p1, inputs)
                    + _deterministic_stint_cost_at(compounds[1], p1 + 1, p2, inputs)
                    + _deterministic_stint_cost_at(compounds[2], p2 + 1, laps, inputs)
                )
                traffic = inputs.circuit.overtaking_difficulty * 6.2
                leverage = _timing_leverage([p1, p2], inputs)
                total = tyre_cost + 2 * pit_cost + traffic - leverage
                if best is None or total < best[0]:
                    best = (total, [p1, p2], lengths)
    else:
        raise ValueError("Only one-stop and two-stop strategies are supported.")

    assert best is not None
    return best[1], best[2], float(best[0])

def _pit_window(pit_laps: list[int], race_laps: int, width: int = 2) -> str:
    if not pit_laps:
        return "—"
    return " / ".join(
        f"L{max(1, p-width)}–{min(race_laps-1, p+width)}"
        for p in pit_laps
    )



def _stint_lengths_from_pit_laps(
    pit_laps: list[int],
    race_laps: int,
    expected_stops: int,
) -> list[int]:
    pits = [int(x) for x in pit_laps]
    if len(pits) != expected_stops:
        raise ValueError(f"Expected {expected_stops} pit-stop lap(s).")
    if any(p <= 1 or p >= race_laps for p in pits):
        raise ValueError("Pit laps must be inside the race distance.")
    if pits != sorted(pits) or len(set(pits)) != len(pits):
        raise ValueError("Pit laps must be strictly increasing.")
    if any((b - a) < 3 for a, b in zip(pits, pits[1:])):
        raise ValueError("Pit stops must be separated by at least 3 laps.")

    boundaries = [0] + pits + [race_laps]
    lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    if any(l < 2 for l in lengths):
        raise ValueError("Each stint must last at least 2 laps.")
    return lengths


def _fixed_plan_cost(
    compounds: list[str],
    pit_laps: list[int],
    inputs: SimulationInputs,
) -> tuple[list[int], float]:
    lengths = _stint_lengths_from_pit_laps(
        pit_laps,
        inputs.circuit.race_laps,
        len(compounds) - 1,
    )
    bounds = [0] + pit_laps + [inputs.circuit.race_laps]
    tyre_cost = 0.0
    for i, comp in enumerate(compounds):
        tyre_cost += _deterministic_stint_cost_at(
            comp,
            bounds[i] + 1,
            bounds[i + 1],
            inputs,
        )
    stops = len(compounds) - 1
    traffic = inputs.circuit.overtaking_difficulty * (3.0 if stops == 1 else 6.2)
    leverage = _timing_leverage(pit_laps, inputs)
    total = tyre_cost + stops * inputs.circuit.pit_loss_green + traffic - leverage
    return lengths, float(total)


def _sample_neutralisation_laps(
    inputs: SimulationInputs,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    mode = str(inputs.neutralisation_mode or "NONE").upper()
    if mode not in {"SC", "VSC"}:
        return np.full(n, -999, dtype=int)

    race_laps = max(8, int(inputs.circuit.race_laps))
    if inputs.neutralisation_lap is not None:
        lap = int(np.clip(inputs.neutralisation_lap, 2, race_laps - 2))
        return np.full(n, lap, dtype=int)
    # None means random timing across the race.
    return rng.integers(3, race_laps - 2, size=n)


def _neutralisation_pit_loss(inputs: SimulationInputs) -> float:
    mode = str(inputs.neutralisation_mode or "NONE").upper()
    green = float(inputs.circuit.pit_loss_green)
    sc_loss = float(inputs.circuit.pit_loss_sc)
    if mode == "SC":
        return sc_loss
    if mode == "VSC":
        # VSC benefit is modelled as smaller than a full Safety Car.
        return green - 0.62 * (green - sc_loss)
    return green


def _effective_neutralisation_probability(inputs: SimulationInputs) -> float:
    mode = str(inputs.neutralisation_mode or "NONE").upper()
    if mode == "SC":
        return 1.0
    if mode == "VSC":
        return 0.75
    return 0.0



def _weather_compound_adjustment(
    compound: str,
    rain: np.ndarray,
    inputs: SimulationInputs,
) -> np.ndarray:
    """Per-lap pace adjustment for tyre suitability to the weather state.

    Negative is faster. Values are scenario-model parameters rather than official
    Pirelli deltas; they are deliberately large enough to prevent unrealistic
    slick-on-heavy-rain or full-wet-on-dry solutions.
    """
    comp = str(compound).upper()
    mode = str(inputs.weather_mode or "EXPECTED").upper()
    rain = np.asarray(rain, dtype=bool)

    # Dry condition adjustments.
    dry_adjust = {
        "SOFT": 0.0,
        "MEDIUM": 0.0,
        "HARD": 0.0,
        "INTERMEDIATE": 2.4,
        "WET": 4.2,
    }

    # Wet condition adjustments vary by scenario intensity.
    if mode == "HEAVY_RAIN":
        wet_adjust = {
            "SOFT": 7.2,
            "MEDIUM": 7.0,
            "HARD": 6.8,
            "INTERMEDIATE": 0.9,
            "WET": -0.8,
        }
    elif mode == "RAIN":
        wet_adjust = {
            "SOFT": 4.8,
            "MEDIUM": 4.6,
            "HARD": 4.5,
            "INTERMEDIATE": -0.6,
            "WET": 0.5,
        }
    else:
        # Expected / changeable rain is treated as normal wet-track conditions.
        wet_adjust = {
            "SOFT": 3.1,
            "MEDIUM": 3.0,
            "HARD": 2.9,
            "INTERMEDIATE": -0.45,
            "WET": 1.0,
        }

    return np.where(
        rain,
        float(wet_adjust.get(comp, 0.0)),
        float(dry_adjust.get(comp, 0.0)),
    )


def _simulate_cost(
    compounds: list[str],
    stint_lengths: list[int],
    inputs: SimulationInputs,
    rng: np.random.Generator,
    n: int,
    pit_laps: list[int] | None = None,
    neutralisation_laps: np.ndarray | None = None,
    rain_events: np.ndarray | None = None,
):
    costs = np.zeros(n, dtype=float)
    deg_shock = rng.lognormal(mean=0.0, sigma=0.16, size=n)
    traffic_shock = rng.gamma(shape=1.8, scale=0.85, size=n)

    if pit_laps is None:
        pit_laps = list(np.cumsum(stint_lengths)[:-1].astype(int))

    bounds = [0] + [int(x) for x in pit_laps] + [int(inputs.circuit.race_laps)]

    for i, comp in enumerate(compounds):
        start_lap = bounds[i] + 1
        end_lap = bounds[i + 1]
        stint_laps = max(0, end_lap - start_lap + 1)
        tyre = inputs.tyres[comp]

        if inputs.weather_timeline:
            weather_cost, avg_track_temp = _stint_weather_profile(
                comp, start_lap, end_lap, inputs
            )
            costs += weather_cost
        else:
            # Backwards-compatible single-weather stochastic mode.
            if rain_events is None:
                rain = rng.random(n) < inputs.rain_probability
            else:
                rain = np.asarray(rain_events, dtype=bool)
            costs += _weather_compound_adjustment(comp, rain, inputs) * stint_laps
            avg_track_temp = inputs.circuit.track_temperature

        deg = _effective_degradation(tyre, avg_track_temp)
        costs += tyre.pace_delta * stint_laps
        costs += deg * deg_shock * stint_laps * (stint_laps - 1) / 2.0

    stops = len(compounds) - 1

    if neutralisation_laps is None:
        neutralisation_laps = _sample_neutralisation_laps(inputs, rng, n)

    cheap_loss = _neutralisation_pit_loss(inputs)
    green_loss = float(inputs.circuit.pit_loss_green)

    for pit_lap in pit_laps:
        aligned = np.abs(neutralisation_laps - int(pit_lap)) <= 2
        costs += np.where(aligned, cheap_loss, green_loss)

    costs += stops * inputs.circuit.overtaking_difficulty * (1.7 + 2.8 * traffic_shock)
    costs -= stops * inputs.circuit.undercut_power * (0.9 + 1.8 * rng.random(n))

    if compounds[0] == "SOFT":
        costs -= 1.0 + rng.normal(0.0, 0.65, size=n)

    costs += inputs.driver.race_pace_delta * inputs.circuit.race_laps
    return costs

def enumerate_legal_strategies(
    inputs: SimulationInputs,
    start_compound: str | None = None,
    stops: tuple[int, ...] = (1, 2),
) -> list[list[str]]:
    out = []
    for stop_count in stops:
        n = stop_count + 1
        for seq in product(inputs.allowed_compounds, repeat=n):
            if start_compound and seq[0] != start_compound.upper():
                continue
            valid, _ = validate_strategy(seq, inputs)
            if valid:
                out.append(list(seq))
    return out


def _strategy_name(compounds: list[str]) -> str:
    short = {
        "SOFT": "S",
        "MEDIUM": "M",
        "HARD": "H",
        "INTERMEDIATE": "I",
        "WET": "W",
    }
    return "→".join(short.get(c.upper(), c[:1].upper()) for c in compounds)



def _softmax_from_costs(costs: np.ndarray, temperature: float = 5.5) -> np.ndarray:
    costs = np.asarray(costs, dtype=float)
    scores = -(costs - np.nanmin(costs)) / max(0.5, temperature)
    scores = scores - np.nanmax(scores)
    weights = np.exp(scores)
    return weights / weights.sum()


def _synthetic_rivals(inputs: SimulationInputs) -> list[dict[str, Any]]:
    """Fallback grid used only when current-weekend opponent data are unavailable."""
    total = 22
    selected_grid = int(np.clip(inputs.driver.grid_position, 1, total))
    rows = []
    pace_steps = np.linspace(0.0, 1.45, total)
    for pos in range(1, total + 1):
        if pos == selected_grid:
            continue
        # Grid position is used only to produce a conservative fallback ordering.
        pace = float(pace_steps[min(total - 1, pos - 1)])
        rows.append({
            "driver_name": f"Rival {pos}",
            "abbreviation": f"R{pos}",
            "team_name": "",
            "grid_position": pos,
            "race_pace_delta": pace,
            "pace_confidence": "low",
            "grid_confidence": "low",
        })
    return rows


def _generic_rival_strategy_pool(
    inputs: SimulationInputs,
    n: int,
    seed: int = 7711,
    neutralisation_laps: np.ndarray | None = None,
    rain_events: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """Build a reusable stochastic strategy pool for the field.

    Rivals are not assumed to all run the same strategy. Each simulation samples
    from competitive legal strategies, weighted by their expected strategic cost.
    """
    neutral_inputs = SimulationInputs(
        circuit=inputs.circuit,
        driver=DriverContext(
            driver_name="Neutral rival",
            team_name="",
            grid_position=10,
            race_pace_delta=0.0,
            team_risk=0.50,
        ),
        tyres=inputs.tyres,
        tyre_sets={
            "SOFT": {"new": 2, "used": 1},
            "MEDIUM": {"new": 2, "used": 1},
            "HARD": {"new": 2, "used": 1},
            "INTERMEDIATE": {"new": 4, "used": 0},
            "WET": {"new": 3, "used": 0},
        },
        sc_probability=inputs.sc_probability,
        rain_probability=inputs.rain_probability,
        simulations=n,
        mandatory_race_compounds=inputs.mandatory_race_compounds,
        rivals=None,
        neutralisation_mode=inputs.neutralisation_mode,
        allowed_compounds=inputs.allowed_compounds,
        weather_mode=inputs.weather_mode,
        weather_timeline=inputs.weather_timeline,
        neutralisation_lap=inputs.neutralisation_lap,
    )

    candidates = enumerate_legal_strategies(neutral_inputs, start_compound=None, stops=(1, 2))
    arrays = []
    means = []
    kept = []

    for idx, seq in enumerate(candidates):
        pits, lengths, _ = optimise_pit_laps(seq, neutral_inputs)
        rng = np.random.default_rng(seed + idx * 103)
        arr = _simulate_cost(
            seq,
            lengths,
            neutral_inputs,
            rng,
            n,
            pit_laps=pits,
            neutralisation_laps=neutralisation_laps,
            rain_events=rain_events,
        )
        arrays.append(arr)
        means.append(float(np.mean(arr)))
        kept.append(seq)

    if not arrays:
        # This should never happen with the permissive inventory above.
        fallback = np.zeros((n, 1), dtype=float)
        return fallback, np.array([1.0]), [["MEDIUM", "HARD"]]

    matrix = np.vstack(arrays).T
    means_arr = np.asarray(means, dtype=float)

    # Keep only reasonably competitive strategies; a real field will not choose
    # obviously dominated plans at equal tyre availability.
    cutoff = float(np.min(means_arr) + 13.0)
    mask = means_arr <= cutoff
    matrix = matrix[:, mask]
    means_arr = means_arr[mask]
    kept = [seq for seq, keep in zip(kept, mask) if keep]

    probabilities = _softmax_from_costs(means_arr, temperature=5.8)
    return matrix, probabilities, kept


def _full_grid_finish_distribution(
    selected_costs: np.ndarray,
    inputs: SimulationInputs,
    rng: np.random.Generator,
    neutralisation_laps: np.ndarray | None = None,
    rain_events: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Simulate and rank the complete field for every Monte Carlo race."""
    n = len(selected_costs)
    selected_name = inputs.driver.driver_name.strip().lower()

    raw_rivals = inputs.rivals or []
    rivals = []
    for row in raw_rivals:
        name = str(row.get("driver_name", "")).strip()
        if name and name.lower() == selected_name:
            continue
        if row.get("race_pace_delta") is None:
            continue
        rivals.append(dict(row))

    using_synthetic = len(rivals) < 8
    if using_synthetic:
        rivals = _synthetic_rivals(inputs)

    total_cars = len(rivals) + 1
    strategy_matrix, strategy_probs, _ = _generic_rival_strategy_pool(
        inputs,
        n,
        neutralisation_laps=neutralisation_laps,
        rain_events=rain_events,
    )

    mode = str(inputs.neutralisation_mode or "NONE").upper()
    if mode == "SC":
        grid_time_factor = (0.11 + 0.10 * inputs.circuit.overtaking_difficulty) * 0.48
    elif mode == "VSC":
        grid_time_factor = (0.11 + 0.10 * inputs.circuit.overtaking_difficulty) * 0.82
    else:
        grid_time_factor = 0.11 + 0.10 * inputs.circuit.overtaking_difficulty

    selected_grid = int(np.clip(inputs.driver.grid_position, 1, total_cars))
    selected_time = selected_costs.copy()
    selected_time += (selected_grid - 1) * grid_time_factor
    selected_time += rng.normal(0.0, 0.55, size=n)

    neutral_p = _effective_neutralisation_probability(inputs)
    selected_dnf_p = float(np.clip(
        0.030 + 0.025 * neutral_p + 0.070 * inputs.rain_probability,
        0.025,
        0.16,
    ))
    selected_dnf = rng.random(n) < selected_dnf_p

    rival_times = np.empty((n, len(rivals)), dtype=float)

    for j, rival in enumerate(rivals):
        choice = rng.choice(strategy_matrix.shape[1], size=n, p=strategy_probs)
        race_strategy = strategy_matrix[np.arange(n), choice].copy()

        pace_delta = float(max(0.0, rival.get("race_pace_delta", 0.8)))
        grid_position = rival.get("grid_position")
        try:
            grid_position = int(grid_position)
        except Exception:
            grid_position = min(total_cars, j + 1)

        race_strategy += pace_delta * inputs.circuit.race_laps
        race_strategy += (max(1, grid_position) - 1) * grid_time_factor
        race_strategy += rng.normal(
            0.0,
            1.25 + 0.75 * inputs.circuit.overtaking_difficulty,
            size=n,
        )

        rival_dnf_p = float(np.clip(
            0.035 + 0.028 * neutral_p + 0.075 * inputs.rain_probability,
            0.03,
            0.18,
        ))
        rival_dnf = rng.random(n) < rival_dnf_p

        # Give DNFs a very large but unique race-time score to preserve a complete order.
        race_strategy[rival_dnf] = 1_000_000.0 + rng.uniform(0, 1000, rival_dnf.sum())
        rival_times[:, j] = race_strategy

    selected_time[selected_dnf] = 1_000_000.0 + rng.uniform(0, 1000, selected_dnf.sum())

    all_times = np.column_stack([selected_time, rival_times])
    order = np.argsort(all_times, axis=1)
    ranks = np.empty_like(order)
    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(1, total_cars + 1)[None, :]

    finishes = ranks[:, 0].astype(int)

    driver_rows = [{
        "driver_name": inputs.driver.driver_name,
        "team_name": inputs.driver.team_name,
        "grid_position": selected_grid,
        "selected_driver": True,
        "ranks": ranks[:, 0].astype(int),
    }]
    for j, rival in enumerate(rivals, start=1):
        gp = rival.get("grid_position")
        try:
            gp = int(gp)
        except Exception:
            gp = None
        driver_rows.append({
            "driver_name": str(rival.get("driver_name") or rival.get("abbreviation") or f"Rival {j}"),
            "team_name": str(rival.get("team_name") or ""),
            "grid_position": gp,
            "selected_driver": False,
            "ranks": ranks[:, j].astype(int),
        })

    estimated_classification = []
    for row in driver_rows:
        rr = row["ranks"]
        mode_rank = int(pd.Series(rr).mode().iloc[0])
        estimated_classification.append({
            "driver_name": row["driver_name"],
            "team_name": row["team_name"],
            "grid_position": row["grid_position"],
            "selected_driver": row["selected_driver"],
            "expected_finish": float(np.mean(rr)),
            "most_likely_finish": mode_rank,
            "win_probability": float(np.mean(rr == 1)),
            "podium_probability": float(np.mean(rr <= 3)),
            "points_probability": float(np.mean(rr <= 10)),
        })

    estimated_classification.sort(key=lambda x: (x["expected_finish"], x["most_likely_finish"]))
    selected_projected_position = None
    for projected_position, row in enumerate(estimated_classification, start=1):
        row["projected_position"] = projected_position
        if row.get("selected_driver"):
            selected_projected_position = projected_position

    real_pace_count = sum(
        1 for r in rivals
        if str(r.get("pace_confidence", "")).lower() in {"high", "medium"}
    )
    real_grid_count = sum(
        1 for r in rivals
        if str(r.get("grid_confidence", "")).lower() in {"high", "medium"}
        or r.get("grid_position") is not None
    )

    if using_synthetic:
        confidence = "low"
    elif real_pace_count >= max(14, int(0.70 * len(rivals))) and real_grid_count >= max(14, int(0.70 * len(rivals))):
        confidence = "high"
    elif real_pace_count >= 8:
        confidence = "medium"
    else:
        confidence = "low"

    meta = {
        "competitors_modelled": len(rivals),
        "field_size": total_cars,
        "race_model_confidence": confidence,
        "synthetic_grid": using_synthetic,
        "selected_dnf_probability": selected_dnf_p,
        "estimated_classification": estimated_classification,
        "selected_projected_position": selected_projected_position,
    }
    return finishes, meta

def simulate_selected_strategy(
    inputs: SimulationInputs,
    compounds: list[str],
    compare_same_start: bool = True,
    pit_laps_override: list[int] | None = None,
) -> Dict[str, Any]:
    compounds = [c.upper() for c in compounds]
    valid, reasons = validate_strategy(compounds, inputs)
    if not valid:
        raise ValueError(" | ".join(reasons))

    if pit_laps_override is not None:
        pit_laps = [int(x) for x in pit_laps_override]
        stint_lengths, deterministic = _fixed_plan_cost(compounds, pit_laps, inputs)
    else:
        pit_laps, stint_lengths, deterministic = optimise_pit_laps(compounds, inputs)

    n = int(inputs.simulations)
    rng = np.random.default_rng(260913)
    neutralisation_laps = _sample_neutralisation_laps(inputs, rng, n)
    rain_events = rng.random(n) < inputs.rain_probability

    selected_costs = _simulate_cost(
        compounds,
        stint_lengths,
        inputs,
        rng,
        n,
        pit_laps=pit_laps,
        neutralisation_laps=neutralisation_laps,
        rain_events=rain_events,
    )

    alternatives = enumerate_legal_strategies(
        inputs,
        start_compound=compounds[0] if compare_same_start else None,
        stops=(1, 2),
    )
    alt_rows = []
    alt_cost_arrays = []

    for seq in alternatives:
        pits, lengths, det = optimise_pit_laps(seq, inputs)
        seed = 260913 + sum((i + 1) * ord(c[0]) for i, c in enumerate(seq))
        seq_rng = np.random.default_rng(seed)
        arr = _simulate_cost(
            seq,
            lengths,
            inputs,
            seq_rng,
            n,
            pit_laps=pits,
            neutralisation_laps=neutralisation_laps,
            rain_events=rain_events,
        )
        alt_cost_arrays.append(arr)
        alt_rows.append({
            "strategy": _strategy_name(seq),
            "compounds": seq,
            "pit_laps": pits,
            "pit_window": _pit_window(pits, inputs.circuit.race_laps),
            "expected_cost_s": float(np.mean(arr)),
            "deterministic_cost_s": det,
        })

    if not alt_cost_arrays:
        raise ValueError("No legal comparison strategies are available with this tyre inventory.")

    matrix = np.vstack(alt_cost_arrays).T
    optimal_idx = np.argmin(np.mean(matrix, axis=0))
    optimal = alt_rows[int(optimal_idx)]

    selected_idx = next(
        (i for i, row in enumerate(alt_rows) if row["compounds"] == compounds),
        None
    )
    optimal_probability = (
        float(np.mean(np.argmin(matrix, axis=1) == selected_idx))
        if selected_idx is not None
        else 0.0
    )

    finish_rng = np.random.default_rng(99173)
    finishes, race_meta = _full_grid_finish_distribution(
        selected_costs,
        inputs,
        finish_rng,
        neutralisation_laps=neutralisation_laps,
        rain_events=rain_events,
    )

    expected_finish = float(np.mean(finishes))
    finish_mode = int(pd.Series(finishes).mode().iloc[0])
    win = float(np.mean(finishes == 1))
    podium = float(np.mean(finishes <= 3))
    top5 = float(np.mean(finishes <= 5))
    points = float(np.mean(finishes <= 10))
    downside = float(np.mean(finishes >= 11))

    selected_mean = float(np.mean(selected_costs))
    delta_to_opt = selected_mean - float(optimal["expected_cost_s"])

    return {
        "strategy": _strategy_name(compounds),
        "compounds": compounds,
        "pit_laps": pit_laps,
        "pit_window": " / ".join(f"L{p}" for p in pit_laps),
        "stint_lengths": stint_lengths,
        "neutralisation_mode": str(inputs.neutralisation_mode or "NONE").upper(),
        "neutralisation_lap": inputs.neutralisation_lap,
        "weather_timeline": _normalised_weather_timeline(inputs),
        "expected_cost_s": selected_mean,
        "delta_to_optimal_s": delta_to_opt,
        "expected_finish": expected_finish,
        "most_likely_finish": finish_mode,
        "win_probability": win,
        "podium_probability": podium,
        "top5_probability": top5,
        "points_probability": points,
        "downside_probability": downside,
        "optimal_probability": optimal_probability,
        "finish_distribution": {
            int(pos): float(np.mean(finishes == pos))
            for pos in range(1, int(np.max(finishes)) + 1)
            if np.any(finishes == pos)
        },
        "competitors_modelled": race_meta["competitors_modelled"],
        "field_size": race_meta["field_size"],
        "race_model_confidence": race_meta["race_model_confidence"],
        "synthetic_grid": race_meta["synthetic_grid"],
        "selected_dnf_probability": race_meta["selected_dnf_probability"],
        "estimated_classification": race_meta["estimated_classification"],
        "projected_finish_position": int(
            race_meta.get("selected_projected_position")
            or finish_mode
        ),
        "modal_finish_position": finish_mode,
        "optimal": optimal,
        "alternatives": sorted(alt_rows, key=lambda r: r["expected_cost_s"]),
    }


TARGET_POSITION_LIMITS = {
    "WIN": 1,
    "PODIUM": 3,
    "TOP5": 5,
    "POINTS": 10,
}


def _target_metric(finishes: np.ndarray, target: str) -> float:
    target = str(target).upper()
    limit = TARGET_POSITION_LIMITS.get(target, 5)
    return float(np.mean(np.asarray(finishes) <= limit))


def _target_weather_label(timeline: list[dict[str, Any]]) -> str:
    modes = [str(x.get("mode", "EXPECTED")).upper() for x in timeline]
    short = {
        "EXPECTED": "Expected",
        "DRY": "Dry",
        "HOT_DRY": "Hot & dry",
        "COOL_DRY": "Cool & dry",
        "CHANGEABLE": "Changeable",
        "RAIN": "Rain",
        "HEAVY_RAIN": "Heavy rain",
    }
    labels = [short.get(m, m.title()) for m in modes]
    # preserve order while removing adjacent duplicates
    out = []
    for label in labels:
        if not out or out[-1] != label:
            out.append(label)
    return " → ".join(out)


def _target_wet_share(timeline: list[dict[str, Any]], race_laps: int) -> float:
    wet = 0.0
    total = 0.0
    for phase in timeline:
        laps = int(phase["end_lap"]) - int(phase["start_lap"]) + 1
        mode = str(phase.get("mode", "EXPECTED")).upper()
        if mode == "HEAVY_RAIN":
            p = 1.0
        elif mode == "RAIN":
            p = 0.90
        elif mode == "CHANGEABLE":
            p = 0.45
        elif mode == "EXPECTED":
            p = float(phase.get("rain_probability", 0.0))
        else:
            p = 0.0
        wet += laps * p
        total += laps
    return float(np.clip(wet / max(1.0, total), 0.0, 1.0))


def _target_weather_variants(inputs: SimulationInputs) -> list[dict[str, Any]]:
    """Generate a bounded set of realistic race-weather families for inverse search."""
    laps = int(inputs.circuit.race_laps)
    base_track = float(inputs.circuit.track_temperature)
    base = _normalised_weather_timeline(inputs)

    def phase(start, end, mode, temp, rain_p):
        return {
            "start_lap": int(start),
            "end_lap": int(end),
            "mode": str(mode),
            "track_temp": float(temp),
            "rain_probability": float(rain_p),
        }

    p40 = max(6, min(laps - 12, int(round(laps * 0.40))))
    p58 = max(p40 + 5, min(laps - 7, int(round(laps * 0.58))))
    p68 = max(p40 + 6, min(laps - 5, int(round(laps * 0.68))))
    p78 = max(p58 + 5, min(laps - 3, int(round(laps * 0.78))))

    variants = [
        {"weather_key": "BASELINE", "timeline": base},
        {"weather_key": "DRY", "timeline": [phase(1, laps, "DRY", max(31.0, base_track), 0.0)]},
        {"weather_key": "MID_RAIN", "timeline": [
            phase(1, p40 - 1, "DRY", max(34.0, base_track), 0.0),
            phase(p40, laps, "RAIN", min(28.0, base_track - 7.0), 0.90),
        ]},
        {"weather_key": "LATE_RAIN", "timeline": [
            phase(1, p68 - 1, "DRY", max(33.0, base_track), 0.0),
            phase(p68, laps, "RAIN", min(28.0, base_track - 6.0), 0.90),
        ]},
        {"weather_key": "CHANGEABLE", "timeline": [
            phase(1, p40 - 1, "DRY", max(33.0, base_track), 0.0),
            phase(p40, p78 - 1, "RAIN", min(28.0, base_track - 7.0), 0.90),
            phase(p78, laps, "CHANGEABLE", min(30.0, base_track - 4.0), 0.45),
        ]},
        {"weather_key": "LATE_HEAVY", "timeline": [
            phase(1, p58 - 1, "DRY", max(33.0, base_track), 0.0),
            phase(p58, p78 - 1, "RAIN", min(27.0, base_track - 8.0), 0.90),
            phase(p78, laps, "HEAVY_RAIN", min(24.0, base_track - 10.0), 0.98),
        ]},
    ]

    # Deduplicate timelines (baseline can equal one of the synthetic families).
    seen = set()
    out = []
    for row in variants:
        key = tuple(
            (p["start_lap"], p["end_lap"], p["mode"], round(float(p["track_temp"]), 1))
            for p in row["timeline"]
        )
        if key in seen:
            continue
        seen.add(key)
        row["weather_label"] = _target_weather_label(row["timeline"])
        row["wet_share"] = _target_wet_share(row["timeline"], laps)
        out.append(row)
    return out


def _target_neutralisation_variants(inputs: SimulationInputs) -> list[dict[str, Any]]:
    laps = int(inputs.circuit.race_laps)
    points = {
        "EARLY": max(5, int(round(laps * 0.28))),
        "MID": max(8, int(round(laps * 0.48))),
        "LATE": max(10, int(round(laps * 0.68))),
    }
    return [
        {"neutralisation_mode": "NONE", "neutralisation_lap": None, "neutralisation_label": "No SC / VSC"},
        {"neutralisation_mode": "SC", "neutralisation_lap": points["EARLY"], "neutralisation_label": f'Safety Car L{points["EARLY"]}'},
        {"neutralisation_mode": "SC", "neutralisation_lap": points["MID"], "neutralisation_label": f'Safety Car L{points["MID"]}'},
        {"neutralisation_mode": "SC", "neutralisation_lap": points["LATE"], "neutralisation_label": f'Safety Car L{points["LATE"]}'},
        {"neutralisation_mode": "VSC", "neutralisation_lap": points["MID"], "neutralisation_label": f'VSC L{points["MID"]}'},
        {"neutralisation_mode": "VSC", "neutralisation_lap": points["LATE"], "neutralisation_label": f'VSC L{points["LATE"]}'},
    ]


def _target_inputs_for_scenario(
    inputs: SimulationInputs,
    weather_row: dict[str, Any],
    neutral_row: dict[str, Any],
    simulations: int,
) -> SimulationInputs:
    wet = float(weather_row.get("wet_share", 0.0)) > 0.05
    allowed = ALL_COMPOUNDS if wet else DRY_COMPOUNDS
    first_mode = str(weather_row["timeline"][0].get("mode", "EXPECTED")).upper()
    track_temp = float(np.mean([float(p.get("track_temp", inputs.circuit.track_temperature)) for p in weather_row["timeline"]]))
    circuit = replace(inputs.circuit, track_temperature=track_temp)
    return replace(
        inputs,
        circuit=circuit,
        rain_probability=float(weather_row.get("wet_share", inputs.rain_probability)),
        simulations=int(simulations),
        weather_mode=first_mode,
        weather_timeline=weather_row["timeline"],
        allowed_compounds=tuple(allowed),
        neutralisation_mode=str(neutral_row["neutralisation_mode"]),
        neutralisation_lap=neutral_row["neutralisation_lap"],
    )


def _target_grid_factor(inputs: SimulationInputs) -> float:
    mode = str(inputs.neutralisation_mode or "NONE").upper()
    base = 0.11 + 0.10 * inputs.circuit.overtaking_difficulty
    if mode == "SC":
        return base * 0.48
    if mode == "VSC":
        return base * 0.82
    return base



def _target_strategy_templates(inputs: SimulationInputs) -> list[list[str]]:
    """Small, race-realistic strategy library used by the inverse search.

    This deliberately avoids brute-forcing every 5-compound permutation, which
    would make the target optimiser too slow for an interactive Streamlit app.
    """
    allowed = set(str(c).upper() for c in inputs.allowed_compounds)
    dry = [
        ["MEDIUM", "HARD"], ["HARD", "MEDIUM"], ["SOFT", "MEDIUM"],
        ["SOFT", "HARD"], ["MEDIUM", "SOFT"], ["HARD", "SOFT"],
        ["MEDIUM", "HARD", "SOFT"], ["MEDIUM", "SOFT", "SOFT"],
        ["SOFT", "MEDIUM", "HARD"], ["SOFT", "HARD", "MEDIUM"],
        ["HARD", "MEDIUM", "SOFT"], ["MEDIUM", "HARD", "MEDIUM"],
    ]
    wet = [
        ["MEDIUM", "INTERMEDIATE"], ["HARD", "INTERMEDIATE"],
        ["INTERMEDIATE", "INTERMEDIATE"], ["INTERMEDIATE", "WET"],
        ["WET", "INTERMEDIATE"], ["WET", "WET"],
        ["MEDIUM", "INTERMEDIATE", "SOFT"], ["MEDIUM", "INTERMEDIATE", "HARD"],
        ["MEDIUM", "INTERMEDIATE", "WET"], ["HARD", "INTERMEDIATE", "WET"],
        ["INTERMEDIATE", "WET", "INTERMEDIATE"], ["INTERMEDIATE", "INTERMEDIATE", "SOFT"],
        ["WET", "INTERMEDIATE", "SOFT"], ["WET", "INTERMEDIATE", "MEDIUM"],
    ]
    out = []
    for seq in dry + wet:
        if not all(c in allowed for c in seq):
            continue
        valid, _ = validate_strategy(seq, inputs)
        if valid and seq not in out:
            out.append(seq)
    return out


def _target_rival_strategy_pool(
    inputs: SimulationInputs,
    n: int,
    seed: int,
    neutralisation_laps: np.ndarray,
    rain_events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build rival strategy costs without leaking the selected driver's pace into rivals.

    Strategy-only costs must be generated with a neutral driver pace. Rival pace is
    added later, driver by driver, inside `_target_build_environment`. This keeps
    the selected driver's pace truly driver-specific in Target Outcome mode.
    """
    neutral_inputs = SimulationInputs(
        circuit=inputs.circuit,
        driver=DriverContext(
            driver_name="Neutral target rival",
            team_name="",
            grid_position=10,
            race_pace_delta=0.0,
            team_risk=0.50,
        ),
        tyres=inputs.tyres,
        tyre_sets={
            "SOFT": {"new": 2, "used": 1},
            "MEDIUM": {"new": 2, "used": 1},
            "HARD": {"new": 2, "used": 1},
            "INTERMEDIATE": {"new": 4, "used": 0},
            "WET": {"new": 3, "used": 0},
        },
        sc_probability=inputs.sc_probability,
        rain_probability=inputs.rain_probability,
        simulations=n,
        mandatory_race_compounds=inputs.mandatory_race_compounds,
        rivals=None,
        neutralisation_mode=inputs.neutralisation_mode,
        allowed_compounds=inputs.allowed_compounds,
        weather_mode=inputs.weather_mode,
        weather_timeline=inputs.weather_timeline,
        neutralisation_lap=inputs.neutralisation_lap,
    )

    rows = []
    for idx, seq in enumerate(_target_strategy_templates(neutral_inputs)):
        try:
            pits, lengths, _ = optimise_pit_laps(seq, neutral_inputs)
        except Exception:
            continue
        rng = np.random.default_rng(seed + idx * 97)
        arr = _simulate_cost(
            seq, lengths, neutral_inputs, rng, n,
            pit_laps=pits,
            neutralisation_laps=neutralisation_laps,
            rain_events=rain_events,
        )
        rows.append((float(np.mean(arr)), arr))
    if not rows:
        return np.zeros((n, 1), dtype=float), np.array([1.0])
    rows.sort(key=lambda x: x[0])
    rows = rows[:8]
    matrix = np.vstack([r[1] for r in rows]).T
    probs = _softmax_from_costs(np.asarray([r[0] for r in rows]), temperature=5.8)
    return matrix, probs

def _target_build_environment(inputs: SimulationInputs, n: int, seed: int) -> dict[str, Any]:
    """Simulate the rival field once and reuse it across many selected-driver plans."""
    rng = np.random.default_rng(seed)
    neutralisation_laps = _sample_neutralisation_laps(inputs, rng, n)
    rain_events = rng.random(n) < inputs.rain_probability

    selected_name = inputs.driver.driver_name.strip().lower()
    rivals = []
    for row in inputs.rivals or []:
        name = str(row.get("driver_name", "")).strip()
        if name and name.lower() == selected_name:
            continue
        if row.get("race_pace_delta") is None:
            continue
        rivals.append(dict(row))
    if len(rivals) < 8:
        rivals = _synthetic_rivals(inputs)

    strategy_matrix, strategy_probs = _target_rival_strategy_pool(
        inputs,
        n,
        seed=seed + 301,
        neutralisation_laps=neutralisation_laps,
        rain_events=rain_events,
    )

    grid_factor = _target_grid_factor(inputs)
    neutral_p = _effective_neutralisation_probability(inputs)
    rival_times = np.empty((n, len(rivals)), dtype=float)

    for j, rival in enumerate(rivals):
        choice = rng.choice(strategy_matrix.shape[1], size=n, p=strategy_probs)
        times = strategy_matrix[np.arange(n), choice].copy()
        pace_delta = float(max(0.0, rival.get("race_pace_delta", 0.8)))
        try:
            grid_pos = int(rival.get("grid_position"))
        except Exception:
            grid_pos = min(len(rivals) + 1, j + 1)
        times += pace_delta * inputs.circuit.race_laps
        times += (max(1, grid_pos) - 1) * grid_factor
        times += rng.normal(0.0, 1.25 + 0.75 * inputs.circuit.overtaking_difficulty, size=n)
        dnf_p = float(np.clip(
            0.035 + 0.028 * neutral_p + 0.075 * inputs.rain_probability,
            0.03, 0.18,
        ))
        dnf = rng.random(n) < dnf_p
        times[dnf] = 1_000_000.0 + rng.uniform(0, 1000, dnf.sum())
        rival_times[:, j] = times

    return {
        "neutralisation_laps": neutralisation_laps,
        "rain_events": rain_events,
        "rival_times": rival_times,
        "grid_factor": grid_factor,
        "field_size": len(rivals) + 1,
        "competitors_modelled": len(rivals),
    }


def _target_candidate_result(
    inputs: SimulationInputs,
    compounds: list[str],
    pit_laps: list[int],
    env: dict[str, Any],
    target: str,
    seed: int,
) -> dict[str, Any]:
    n = len(env["neutralisation_laps"])
    lengths = _stint_lengths_from_pit_laps(pit_laps, inputs.circuit.race_laps, len(compounds) - 1)
    rng = np.random.default_rng(seed)
    costs = _simulate_cost(
        compounds,
        lengths,
        inputs,
        rng,
        n,
        pit_laps=pit_laps,
        neutralisation_laps=env["neutralisation_laps"],
        rain_events=env["rain_events"],
    )
    selected_time = costs.copy()
    selected_grid = int(np.clip(inputs.driver.grid_position, 1, env["field_size"]))
    selected_time += (selected_grid - 1) * env["grid_factor"]
    selected_time += rng.normal(0.0, 0.55, size=n)

    neutral_p = _effective_neutralisation_probability(inputs)
    dnf_p = float(np.clip(
        0.030 + 0.025 * neutral_p + 0.070 * inputs.rain_probability,
        0.025, 0.16,
    ))
    dnf = rng.random(n) < dnf_p
    selected_time[dnf] = 1_000_000.0 + rng.uniform(0, 1000, dnf.sum())

    finishes = 1 + np.sum(env["rival_times"] < selected_time[:, None], axis=1)
    finishes = finishes.astype(int)
    return {
        "target_probability": _target_metric(finishes, target),
        "expected_finish": float(np.mean(finishes)),
        "win_probability": float(np.mean(finishes == 1)),
        "podium_probability": float(np.mean(finishes <= 3)),
        "top5_probability": float(np.mean(finishes <= 5)),
        "points_probability": float(np.mean(finishes <= 10)),
        "expected_cost_s": float(np.mean(costs)),
        "finishes": finishes,
    }


def _target_key_event_laps(inputs: SimulationInputs) -> list[int]:
    laps = []
    for phase in _normalised_weather_timeline(inputs)[1:]:
        laps.append(int(phase["start_lap"]))
    if inputs.neutralisation_lap is not None:
        laps.append(int(inputs.neutralisation_lap))
    race_laps = int(inputs.circuit.race_laps)
    return sorted(set(max(3, min(race_laps - 3, x)) for x in laps))


def _target_pit_variants(
    compounds: list[str],
    optimal_pits: list[int],
    inputs: SimulationInputs,
) -> list[list[int]]:
    variants = [list(optimal_pits)]
    events = _target_key_event_laps(inputs)
    race_laps = int(inputs.circuit.race_laps)

    if len(compounds) == 2:
        for e in events:
            for shift in (-1, 0, 1):
                p = max(3, min(race_laps - 3, e + shift))
                variants.append([p])
        p = optimal_pits[0]
        for shift in (-4, 4):
            variants.append([max(3, min(race_laps - 3, p + shift))])
    else:
        p1, p2 = optimal_pits
        for e in events:
            if abs(e - p1) <= abs(e - p2):
                variants.append([max(3, min(p2 - 3, e)), p2])
            else:
                variants.append([p1, min(race_laps - 3, max(p1 + 3, e))])
        variants += [
            [max(3, p1 - 3), p2],
            [min(p2 - 3, p1 + 3), p2],
            [p1, max(p1 + 3, p2 - 3)],
            [p1, min(race_laps - 3, p2 + 3)],
        ]

    out = []
    seen = set()
    for pits in variants:
        try:
            _stint_lengths_from_pit_laps(pits, race_laps, len(compounds) - 1)
        except Exception:
            continue
        key = tuple(int(x) for x in pits)
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return out[:6]


def _target_candidate_pool(inputs: SimulationInputs, max_strategies: int = 5) -> list[dict[str, Any]]:
    candidates = []
    for seq in _target_strategy_templates(inputs):
        try:
            pits, _, det = optimise_pit_laps(seq, inputs)
        except Exception:
            continue
        candidates.append({"compounds": seq, "pits": pits, "deterministic_cost": float(det)})
    candidates.sort(key=lambda r: r["deterministic_cost"])

    expanded = []
    for row in candidates[:max_strategies]:
        for pits in _target_pit_variants(row["compounds"], row["pits"], inputs):
            try:
                _, det = _fixed_plan_cost(row["compounds"], pits, inputs)
            except Exception:
                continue
            expanded.append({
                "compounds": row["compounds"],
                "pit_laps": pits,
                "deterministic_cost": float(det),
            })
    expanded.sort(key=lambda r: r["deterministic_cost"])
    return expanded[:14]

def _target_success_floor(target: str, maximum: float) -> float:
    floors = {"WIN": 0.03, "PODIUM": 0.12, "TOP5": 0.30, "POINTS": 0.55}
    return float(max(floors.get(str(target).upper(), 0.30), 0.68 * maximum))


def _target_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    overall = float(np.mean([r["target_probability"] for r in rows]))
    factors = []

    def add_groups(field, pretty):
        vals = {}
        for r in rows:
            vals.setdefault(str(r[field]), []).append(r["target_probability"])
        for key, arr in vals.items():
            if len(arr) < 2:
                continue
            mean = float(np.mean(arr))
            factors.append({
                "factor": f"{pretty}: {key}",
                "probability": mean,
                "impact_pp": (mean - overall) * 100.0,
            })

    add_groups("weather_label", "Weather")
    add_groups("neutralisation_type", "Race control")
    add_groups("stops", "Stops")
    factors.sort(key=lambda x: abs(x["impact_pp"]), reverse=True)
    return factors[:8]


def _target_minimum_requirements(rows: list[dict[str, Any]], target: str) -> list[dict[str, str]]:
    if not rows:
        return []
    maximum = max(r["target_probability"] for r in rows)
    floor = _target_success_floor(target, maximum)
    good = [r for r in rows if r["target_probability"] >= floor]
    if not good:
        good = sorted(rows, key=lambda r: r["target_probability"], reverse=True)[:5]

    requirements = []
    neutral_share = float(np.mean([r["neutralisation_type"] != "NONE" for r in good]))
    sc_share = float(np.mean([r["neutralisation_type"] == "SC" for r in good]))
    if neutral_share >= 0.70:
        text = "Safety Car strongly favours the target" if sc_share >= 0.55 else "A neutralisation is usually needed"
    elif neutral_share <= 0.30:
        text = "Safety Car / VSC is not required"
    else:
        text = "Neutralisation is helpful but not essential"
    requirements.append({"name": "Race control", "value": text})

    weather_counts = {}
    for r in good:
        weather_counts[r["weather_label"]] = weather_counts.get(r["weather_label"], 0) + 1
    weather_name, weather_count = max(weather_counts.items(), key=lambda kv: kv[1])
    if weather_count / len(good) >= 0.55:
        weather_text = f"Best results cluster around {weather_name}"
    else:
        weather_text = "No single weather pattern is essential"
    requirements.append({"name": "Weather", "value": weather_text})

    pit1 = sorted(r["pit_laps"][0] for r in good if r.get("pit_laps"))
    if pit1:
        lo = int(np.percentile(pit1, 25))
        hi = int(np.percentile(pit1, 75))
        requirements.append({"name": "First stop", "value": f"Most successful paths pit around L{lo}–L{hi}"})

    strategy_counts = {}
    for r in good:
        strategy_counts[r["strategy"]] = strategy_counts.get(r["strategy"], 0) + 1
    if strategy_counts:
        strat, count = max(strategy_counts.items(), key=lambda kv: kv[1])
        requirements.append({"name": "Tyre plan", "value": f"{strat} is the most recurrent successful sequence"})
    return requirements[:4]


def _target_robust_path(rows: list[dict[str, Any]], total_scenarios: int) -> dict[str, Any] | None:
    groups = {}
    for r in rows:
        groups.setdefault(r["strategy"], []).append(r)
    scored = []
    for strategy, arr in groups.items():
        # Use best plan per scenario, then penalise strategies that only work in rare conditions.
        by_scenario = {}
        for r in arr:
            key = (r["weather_key"], r["neutralisation_label"])
            if key not in by_scenario or r["target_probability"] > by_scenario[key]["target_probability"]:
                by_scenario[key] = r
        probs = [x["target_probability"] for x in by_scenario.values()]
        if not probs:
            continue
        coverage = min(1.0, len(by_scenario) / max(1, total_scenarios))
        mean = float(np.mean(probs))
        q25 = float(np.percentile(probs, 25))
        score = (0.62 * mean + 0.38 * q25) * (0.55 + 0.45 * coverage)
        best_row = max(by_scenario.values(), key=lambda x: x["target_probability"])
        pit_lists = [x["pit_laps"] for x in by_scenario.values() if len(x["pit_laps"]) == len(best_row["pit_laps"])]
        if pit_lists:
            typical_pits = [int(round(np.median([p[i] for p in pit_lists]))) for i in range(len(pit_lists[0]))]
        else:
            typical_pits = best_row["pit_laps"]
        scored.append({
            "strategy": strategy,
            "robust_probability": mean,
            "downside_probability": q25,
            "coverage": coverage,
            "score": score,
            "typical_pit_laps": typical_pits,
            "best_example": best_row,
        })
    return max(scored, key=lambda x: x["score"]) if scored else None


def _target_status(target: str, probability: float) -> tuple[str, str]:
    target = str(target).upper()
    if target == "WIN":
        if probability >= 0.25: return "STRONG PATH", "A realistic route to victory exists in the searched scenario space."
        if probability >= 0.10: return "POSSIBLE", "Victory is achievable, but it needs favourable race conditions."
        if probability >= 0.04: return "OUTSIDE CHANCE", "A win requires a narrow combination of favourable events."
        return "UNLIKELY", "No realistic searched scenario produces a strong win probability."
    if target == "PODIUM":
        if probability >= 0.55: return "STRONG PATH", "The podium is robustly achievable in favourable scenarios."
        if probability >= 0.30: return "POSSIBLE", "A podium is realistic with the right strategy and race evolution."
        return "DIFFICULT", "The podium remains scenario-dependent and relatively unlikely."
    if target == "TOP5":
        if probability >= 0.70: return "ROBUST", "A Top 5 is achievable across a broad set of favourable scenarios."
        if probability >= 0.45: return "POSSIBLE", "A clear Top-5 path exists, but conditions still matter."
        return "DIFFICULT", "The driver needs a favourable race evolution to reach the Top 5."
    if probability >= 0.80: return "ROBUST", "Points are achievable in most favourable scenarios."
    if probability >= 0.55: return "POSSIBLE", "A points finish is realistic with the right race execution."
    return "DIFFICULT", "Even a points finish needs significant help from the scenario."


def find_target_outcome(
    inputs: SimulationInputs,
    target: str = "TOP5",
    coarse_simulations: int = 300,
    final_simulations: int | None = None,
    top_finalists: int = 3,
) -> dict[str, Any]:
    """Inverse scenario optimiser.

    Searches weather, neutralisation, tyre strategy and pit timing for conditions
    that maximise the selected driver's chance of hitting the requested outcome.
    The best coarse candidates are then re-run at full Monte Carlo resolution.
    """
    target = str(target).upper()
    if target not in TARGET_POSITION_LIMITS:
        raise ValueError(f"Unsupported target: {target}")

    final_n = int(final_simulations or max(10_000, inputs.simulations))
    weather_variants = _target_weather_variants(inputs)
    neutral_variants = _target_neutralisation_variants(inputs)
    coarse_rows = []
    scenario_counter = 0

    for w_idx, weather in enumerate(weather_variants):
        for n_idx, neutral in enumerate(neutral_variants):
            scenario_counter += 1
            scenario_inputs = _target_inputs_for_scenario(
                inputs, weather, neutral, coarse_simulations
            )
            env = _target_build_environment(
                scenario_inputs,
                coarse_simulations,
                seed=840_000 + w_idx * 1000 + n_idx * 71,
            )
            pool = _target_candidate_pool(scenario_inputs, max_strategies=5)
            for c_idx, candidate in enumerate(pool):
                result = _target_candidate_result(
                    scenario_inputs,
                    candidate["compounds"],
                    candidate["pit_laps"],
                    env,
                    target,
                    seed=910_000 + w_idx * 10000 + n_idx * 500 + c_idx * 13,
                )
                coarse_rows.append({
                    "weather_key": weather["weather_key"],
                    "weather_label": weather["weather_label"],
                    "weather_timeline": weather["timeline"],
                    "wet_share": weather["wet_share"],
                    "neutralisation_type": neutral["neutralisation_mode"],
                    "neutralisation_lap": neutral["neutralisation_lap"],
                    "neutralisation_label": neutral["neutralisation_label"],
                    "strategy": _strategy_name(candidate["compounds"]),
                    "compounds": list(candidate["compounds"]),
                    "pit_laps": list(candidate["pit_laps"]),
                    "stops": len(candidate["compounds"]) - 1,
                    **{k: v for k, v in result.items() if k != "finishes"},
                })

    if not coarse_rows:
        raise ValueError("No legal target-outcome scenarios could be generated.")

    coarse_rows.sort(key=lambda r: (r["target_probability"], -r["expected_finish"]), reverse=True)
    finalists = []
    seen = set()
    for row in coarse_rows:
        key = (
            row["weather_key"], row["neutralisation_type"], row["neutralisation_lap"],
            tuple(row["compounds"]), tuple(row["pit_laps"]),
        )
        if key in seen:
            continue
        seen.add(key)
        finalists.append(row)
        if len(finalists) >= max(1, int(top_finalists)):
            break

    confirmed = []
    for idx, row in enumerate(finalists):
        weather = next(w for w in weather_variants if w["weather_key"] == row["weather_key"])
        neutral = {
            "neutralisation_mode": row["neutralisation_type"],
            "neutralisation_lap": row["neutralisation_lap"],
            "neutralisation_label": row["neutralisation_label"],
        }
        final_inputs = _target_inputs_for_scenario(inputs, weather, neutral, final_n)
        env = _target_build_environment(final_inputs, final_n, seed=1_700_000 + idx * 1777)
        res = _target_candidate_result(
            final_inputs,
            row["compounds"],
            row["pit_laps"],
            env,
            target,
            seed=1_900_000 + idx * 977,
        )
        confirmed.append({
            **row,
            **{k: v for k, v in res.items() if k != "finishes"},
            "confirmed_simulations": final_n,
        })

    confirmed.sort(key=lambda r: (r["target_probability"], -r["expected_finish"]), reverse=True)
    best = confirmed[0]
    robust = _target_robust_path(coarse_rows, scenario_counter)
    requirements = _target_minimum_requirements(coarse_rows, target)
    sensitivity = _target_sensitivity(coarse_rows)
    status, status_text = _target_status(target, best["target_probability"])

    # Use the best coarse plan from each distinct scenario as a transparent top list.
    top_scenarios = []
    used_scenarios = set()
    for row in confirmed + coarse_rows:
        skey = (row["weather_key"], row["neutralisation_label"])
        if skey in used_scenarios:
            continue
        used_scenarios.add(skey)
        top_scenarios.append(row)
        if len(top_scenarios) >= 5:
            break

    return {
        "target": target,
        "target_position_limit": TARGET_POSITION_LIMITS[target],
        "target_probability_definition": f"P1-P{TARGET_POSITION_LIMITS[target]}",
        "max_achievable_probability": float(best["target_probability"]),
        "best_case_probabilities": {
            "WIN": float(best["win_probability"]),
            "PODIUM": float(best["podium_probability"]),
            "TOP5": float(best["top5_probability"]),
            "POINTS": float(best["points_probability"]),
        },
        "status": status,
        "status_text": status_text,
        "best_case": best,
        "robust_path": robust,
        "minimum_requirements": requirements,
        "sensitivity": sensitivity,
        "top_scenarios": top_scenarios,
        "coarse_scenarios_evaluated": scenario_counter,
        "coarse_candidates_evaluated": len(coarse_rows),
        "coarse_simulations": int(coarse_simulations),
        "final_simulations": int(final_n),
    }


def scenario_probabilities(inputs: SimulationInputs) -> list[dict[str, Any]]:
    mode = str(inputs.neutralisation_mode or "NONE").upper()
    p_sc = 1.0 if mode in {"SC", "VSC"} else 0.0

    timeline = _normalised_weather_timeline(inputs)
    wet_weight = 0.0
    total = 0.0
    for phase in timeline:
        laps = phase["end_lap"] - phase["start_lap"] + 1
        pmode = str(phase["mode"]).upper()
        if pmode == "HEAVY_RAIN":
            p = 1.0
        elif pmode == "RAIN":
            p = 0.9
        elif pmode == "CHANGEABLE":
            p = 0.45
        elif pmode == "EXPECTED":
            p = float(phase.get("rain_probability", inputs.rain_probability))
        else:
            p = 0.0
        wet_weight += laps * p
        total += laps

    p_rain = float(np.clip(wet_weight / max(1.0, total), 0, 1))
    med_deg = inputs.tyres["MEDIUM"].degradation
    p_high_deg = float(np.clip((med_deg - 0.05) / 0.13, 0.12, 0.75))

    probs = {
        "Dry stable": (1-p_rain)*(1-p_sc)*(1-p_high_deg),
        "Dry high degradation": (1-p_rain)*(1-p_sc)*p_high_deg,
        "Dry + SC/VSC": (1-p_rain)*p_sc,
        "Mixed / Wet": p_rain*(1-p_sc),
        "Wet + SC/VSC": p_rain*p_sc,
    }
    return [{"scenario": k, "probability": float(v)} for k, v in probs.items()]

