from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Any, Iterable
import math
import numpy as np
import pandas as pd


DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


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
    req = {c: 0 for c in DRY_COMPOUNDS}
    for c in compounds:
        if c in req:
            req[c] += 1
    return req


def validate_strategy(
    compounds: list[str] | tuple[str, ...],
    inputs: SimulationInputs,
    wet_used: bool = False,
) -> tuple[bool, list[str]]:
    """Validate a planned dry strategy against current sporting-rule logic and set inventory.

    Current dry-race rule represented here:
    - if no Intermediate/Wet is used, at least two different dry specifications must be used;
    - at least one used dry specification must be a mandatory Race specification;
    - every planned set must exist in the driver's available inventory.

    Wet/intermediate use during the actual race would remove the dry two-specification requirement.
    """
    compounds = [str(c).upper() for c in compounds]
    reasons = []

    if len(compounds) not in (2, 3):
        reasons.append("Choose exactly 1 or 2 pit stops.")
    if any(c not in DRY_COMPOUNDS for c in compounds):
        reasons.append("Only current dry compounds can be selected in pre-race dry strategy mode.")

    if not wet_used:
        if len(set(compounds)) < 2:
            reasons.append("Dry-race rule: at least two different dry tyre specifications must be used.")
        if not any(c in set(inputs.mandatory_race_compounds) for c in compounds):
            reasons.append("Dry-race rule: the strategy must include at least one mandatory Race tyre specification.")

    req = strategy_requirements(compounds)
    for comp, needed in req.items():
        have = total_sets(inputs.tyre_sets, comp)
        if needed > have:
            reasons.append(f"Tyre availability: {needed} {comp} sets required, only {have} available.")

    return len(reasons) == 0, reasons


def legal_next_compounds(
    partial: list[str],
    total_stints: int,
    inputs: SimulationInputs,
) -> list[str]:
    """Return compounds that keep at least one legal completion possible."""
    partial = [c.upper() for c in partial]
    if len(partial) >= total_stints:
        return []
    options = []
    remaining_slots = total_stints - len(partial) - 1
    for candidate in DRY_COMPOUNDS:
        trial = partial + [candidate]
        # Inventory must be valid at the partial stage.
        req = strategy_requirements(trial)
        if any(req[c] > total_sets(inputs.tyre_sets, c) for c in DRY_COMPOUNDS):
            continue

        if remaining_slots == 0:
            valid, _ = validate_strategy(trial, inputs)
            if valid:
                options.append(candidate)
            continue

        # At least one legal future completion must exist.
        feasible_future = False
        for suffix in product(DRY_COMPOUNDS, repeat=remaining_slots):
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


def _simulate_cost(
    compounds: list[str],
    stint_lengths: list[int],
    inputs: SimulationInputs,
    rng: np.random.Generator,
    n: int,
):
    costs = np.zeros(n, dtype=float)
    deg_shock = rng.lognormal(mean=0.0, sigma=0.16, size=n)
    traffic_shock = rng.gamma(shape=1.8, scale=0.85, size=n)
    sc = rng.random(n) < inputs.sc_probability
    rain = rng.random(n) < inputs.rain_probability

    for comp, stint_laps in zip(compounds, stint_lengths):
        tyre = inputs.tyres[comp]
        deg = _effective_degradation(tyre, inputs.circuit.track_temperature)
        costs += tyre.pace_delta * stint_laps
        costs += deg * deg_shock * stint_laps * (stint_laps - 1) / 2.0

    stops = len(compounds) - 1
    sc_alignment = min(0.88, 0.42 + 0.16 * stops)
    effective_pit = inputs.circuit.pit_loss_green - sc.astype(float) * sc_alignment * (
        inputs.circuit.pit_loss_green - inputs.circuit.pit_loss_sc
    )
    costs += stops * effective_pit
    costs += stops * inputs.circuit.overtaking_difficulty * (1.7 + 2.8 * traffic_shock)
    costs -= stops * inputs.circuit.undercut_power * (0.9 + 1.8 * rng.random(n))

    if compounds[0] == "SOFT":
        # Start-performance upside, with some launch variance.
        costs -= 1.0 + rng.normal(0.0, 0.65, size=n)

    # A pre-race slick strategy loses relevance if rain arrives; this is reflected
    # as a broad scenario penalty, not as a detailed wet-crossover model.
    costs += rain.astype(float) * 15.0
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
        for seq in product(DRY_COMPOUNDS, repeat=n):
            if start_compound and seq[0] != start_compound.upper():
                continue
            valid, _ = validate_strategy(seq, inputs)
            if valid:
                out.append(list(seq))
    return out


def _strategy_name(compounds: list[str]) -> str:
    return "→".join(c[0] for c in compounds)


def _expected_finish_distribution(
    selected_costs: np.ndarray,
    best_costs: np.ndarray,
    inputs: SimulationInputs,
    rng: np.random.Generator,
) -> np.ndarray:
    grid = float(np.clip(inputs.driver.grid_position, 1, 22))
    pace_penalty = max(0.0, inputs.driver.race_pace_delta) * 3.8
    strategy_penalty = np.maximum(0.0, selected_costs - best_costs) / 6.8
    race_noise = rng.normal(0.0, 1.35 + 0.55 * inputs.circuit.overtaking_difficulty, size=len(selected_costs))
    finishes = np.rint(grid + pace_penalty + strategy_penalty + race_noise)
    return np.clip(finishes, 1, 22).astype(int)


def simulate_selected_strategy(
    inputs: SimulationInputs,
    compounds: list[str],
    compare_same_start: bool = True,
) -> Dict[str, Any]:
    compounds = [c.upper() for c in compounds]
    valid, reasons = validate_strategy(compounds, inputs)
    if not valid:
        raise ValueError(" | ".join(reasons))

    pit_laps, stint_lengths, deterministic = optimise_pit_laps(compounds, inputs)
    n = int(inputs.simulations)
    rng = np.random.default_rng(260913)

    selected_costs = _simulate_cost(compounds, stint_lengths, inputs, rng, n)

    alternatives = enumerate_legal_strategies(
        inputs,
        start_compound=compounds[0] if compare_same_start else None,
        stops=(1, 2),
    )
    alt_rows = []
    alt_cost_arrays = []
    for seq in alternatives:
        pits, lengths, det = optimise_pit_laps(seq, inputs)
        # Use a deterministic seed per sequence so Streamlit reruns are stable.
        seed = 260913 + sum((i + 1) * ord(c[0]) for i, c in enumerate(seq))
        seq_rng = np.random.default_rng(seed)
        arr = _simulate_cost(seq, lengths, inputs, seq_rng, n)
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
    per_run_best = np.min(matrix, axis=1)
    optimal_idx = np.argmin(np.mean(matrix, axis=0))
    optimal = alt_rows[int(optimal_idx)]

    # Selected optimality: probability it beats every feasible same-start strategy in each run.
    selected_idx = next(
        (i for i, row in enumerate(alt_rows) if row["compounds"] == compounds),
        None
    )
    optimal_probability = float(np.mean(np.argmin(matrix, axis=1) == selected_idx)) if selected_idx is not None else 0.0

    finish_rng = np.random.default_rng(99173)
    finishes = _expected_finish_distribution(selected_costs, per_run_best, inputs, finish_rng)

    expected_finish = float(np.mean(finishes))
    finish_mode = int(pd.Series(finishes).mode().iloc[0])
    win = float(np.mean(finishes == 1))
    podium = float(np.mean(finishes <= 3))
    top5 = float(np.mean(finishes <= 5))
    points = float(np.mean(finishes <= 10))
    downside = float(np.mean(finishes >= 11))

    selected_mean = float(np.mean(selected_costs))
    delta_to_opt = selected_mean - float(optimal["expected_cost_s"])
    result = {
        "strategy": _strategy_name(compounds),
        "compounds": compounds,
        "pit_laps": pit_laps,
        "pit_window": _pit_window(pit_laps, inputs.circuit.race_laps),
        "stint_lengths": stint_lengths,
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
            for pos in range(1, 23)
            if np.any(finishes == pos)
        },
        "optimal": optimal,
        "alternatives": sorted(alt_rows, key=lambda r: r["expected_cost_s"]),
    }
    return result


def scenario_probabilities(inputs: SimulationInputs) -> list[dict[str, Any]]:
    p_sc = float(np.clip(inputs.sc_probability, 0, 1))
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
