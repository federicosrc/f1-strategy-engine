from __future__ import annotations

from dataclasses import dataclass
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


def _deterministic_stint_cost(compound: str, laps: int, inputs: SimulationInputs) -> float:
    tyre = inputs.tyres[compound]
    deg = _effective_degradation(tyre, inputs.circuit.track_temperature)
    return tyre.pace_delta * laps + deg * laps * (laps - 1) / 2.0


def _timing_leverage(pit_laps: list[int], inputs: SimulationInputs) -> float:
    # Expected track-position benefit from stopping around the useful undercut zone.
    if not pit_laps:
        return 0.0
    race_laps = inputs.circuit.race_laps
    centrality = np.mean([1.0 - min(1.0, abs((p / race_laps) - 0.42) / 0.42) for p in pit_laps])
    return inputs.circuit.undercut_power * centrality * len(pit_laps) * 1.8


def optimise_pit_laps(compounds: list[str], inputs: SimulationInputs) -> tuple[list[int], list[int], float]:
    """Optimise stint lengths for a fixed compound sequence.

    Returns (pit_laps, stint_lengths, deterministic_strategy_cost).
    """
    n_stints = len(compounds)
    laps = int(inputs.circuit.race_laps)
    min_stint = max(5, int(round(laps * 0.09)))
    pit_cost = inputs.circuit.pit_loss_green
    best = None

    if n_stints == 2:
        lo = min_stint
        hi = laps - min_stint
        for p1 in range(lo, hi + 1):
            lengths = [p1, laps - p1]
            tyre_cost = sum(_deterministic_stint_cost(c, l, inputs) for c, l in zip(compounds, lengths))
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
                tyre_cost = sum(_deterministic_stint_cost(c, l, inputs) for c, l in zip(compounds, lengths))
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
    tyre_cost = sum(
        _deterministic_stint_cost(comp, laps, inputs)
        for comp, laps in zip(compounds, lengths)
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
    # Avoid the formation-lap/start and the very end. Timing remains stochastic
    # because the user chooses occurrence type, not the exact neutralisation lap.
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
    if rain_events is None:
        rain = rng.random(n) < inputs.rain_probability
    else:
        rain = np.asarray(rain_events, dtype=bool)

    for comp, stint_laps in zip(compounds, stint_lengths):
        tyre = inputs.tyres[comp]
        deg = _effective_degradation(tyre, inputs.circuit.track_temperature)
        costs += tyre.pace_delta * stint_laps
        costs += _weather_compound_adjustment(comp, rain, inputs) * stint_laps
        costs += deg * deg_shock * stint_laps * (stint_laps - 1) / 2.0

    stops = len(compounds) - 1
    if pit_laps is None:
        pit_laps = list(np.cumsum(stint_lengths)[:-1].astype(int))

    if neutralisation_laps is None:
        neutralisation_laps = _sample_neutralisation_laps(inputs, rng, n)

    cheap_loss = _neutralisation_pit_loss(inputs)
    green_loss = float(inputs.circuit.pit_loss_green)

    # Only stops close to the neutralisation event receive the cheap-stop benefit.
    # A ±2 lap window is a practical race-strategy approximation.
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

def scenario_probabilities(inputs: SimulationInputs) -> list[dict[str, Any]]:
    mode = str(inputs.neutralisation_mode or "NONE").upper()
    p_sc = 1.0 if mode in {"SC", "VSC"} else 0.0
    p_rain = float(np.clip(inputs.rain_probability, 0, 1))
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
