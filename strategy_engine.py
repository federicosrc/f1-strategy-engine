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
    rivals: list[dict[str, Any]] | None = None


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
        },
        sc_probability=inputs.sc_probability,
        rain_probability=inputs.rain_probability,
        simulations=n,
        mandatory_race_compounds=inputs.mandatory_race_compounds,
        rivals=None,
    )

    candidates = enumerate_legal_strategies(neutral_inputs, start_compound=None, stops=(1, 2))
    arrays = []
    means = []
    kept = []

    for idx, seq in enumerate(candidates):
        pits, lengths, _ = optimise_pit_laps(seq, neutral_inputs)
        rng = np.random.default_rng(seed + idx * 103)
        arr = _simulate_cost(seq, lengths, neutral_inputs, rng, n)
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
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert simulated race times into finishing positions by racing the full field.

    Position is now obtained by ordering simulated race-time scores for the
    selected driver and every rival. It is no longer inferred from a
    seconds-to-positions conversion.
    """
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

    # Common strategic menu for the rest of the field.
    strategy_matrix, strategy_probs, _ = _generic_rival_strategy_pool(inputs, n)

    # Selected driver race-time score. Strategy simulation already contains the
    # selected driver's race-pace delta across the race.
    selected_grid = int(np.clip(inputs.driver.grid_position, 1, total_cars))
    grid_time_factor = 0.11 + 0.10 * inputs.circuit.overtaking_difficulty
    selected_time = selected_costs.copy()
    selected_time += (selected_grid - 1) * grid_time_factor
    selected_time += rng.normal(0.0, 0.55, size=n)

    # Small DNF model. This is deliberately conservative and shared by all cars.
    selected_dnf_p = float(np.clip(
        0.030 + 0.035 * inputs.sc_probability + 0.070 * inputs.rain_probability,
        0.025,
        0.16,
    ))
    selected_dnf = rng.random(n) < selected_dnf_p

    rival_times = np.empty((n, len(rivals)), dtype=float)

    for j, rival in enumerate(rivals):
        # Each rival independently samples from the competitive strategy pool.
        choice = rng.choice(strategy_matrix.shape[1], size=n, p=strategy_probs)
        race_strategy = strategy_matrix[np.arange(n), choice].copy()

        pace_delta = float(max(0.0, rival.get("race_pace_delta", 0.8)))
        grid_position = rival.get("grid_position")
        try:
            grid_position = int(grid_position)
        except Exception:
            grid_position = min(total_cars, j + 1)

        # The long-run delta is converted into race-time delta directly:
        # 0.20 s/lap over 57 laps = 11.4 s, not an arbitrary number of positions.
        race_strategy += pace_delta * inputs.circuit.race_laps
        race_strategy += (max(1, grid_position) - 1) * grid_time_factor

        # Driver/team execution variance: starts, traffic, stop execution and pace noise.
        race_strategy += rng.normal(
            0.0,
            1.25 + 0.75 * inputs.circuit.overtaking_difficulty,
            size=n,
        )

        rival_dnf_p = float(np.clip(
            0.035 + 0.040 * inputs.sc_probability + 0.075 * inputs.rain_probability,
            0.03,
            0.18,
        ))
        rival_dnf = rng.random(n) < rival_dnf_p
        race_strategy[rival_dnf] = np.inf
        rival_times[:, j] = race_strategy

    finishes = 1 + np.sum(rival_times < selected_time[:, None], axis=1)
    finishes = finishes.astype(int)

    # A selected-driver DNF is placed at the back of the simulated field.
    finishes[selected_dnf] = total_cars

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
    }
    return finishes, meta

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
    finishes, race_meta = _full_grid_finish_distribution(selected_costs, inputs, finish_rng)

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
            for pos in range(1, int(np.max(finishes)) + 1)
            if np.any(finishes == pos)
        },
        "competitors_modelled": race_meta["competitors_modelled"],
        "field_size": race_meta["field_size"],
        "race_model_confidence": race_meta["race_model_confidence"],
        "synthetic_grid": race_meta["synthetic_grid"],
        "selected_dnf_probability": race_meta["selected_dnf_probability"],
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
