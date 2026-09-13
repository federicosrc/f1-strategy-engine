from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import math
import numpy as np
import pandas as pd


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
    team_risk: float


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


def _sets_available(sets, compound):
    d = sets.get(compound, {})
    return int(d.get("new", 0)) + int(d.get("used", 0))


def _strategy_catalog(inputs: SimulationInputs):
    raw = [
        ("M-H", ["MEDIUM", "HARD"], [0.38, 0.62]),
        ("H-M", ["HARD", "MEDIUM"], [0.60, 0.40]),
        ("S-H", ["SOFT", "HARD"], [0.30, 0.70]),
        ("M-H-H", ["MEDIUM", "HARD", "HARD"], [0.25, 0.37, 0.38]),
        ("S-H-H", ["SOFT", "HARD", "HARD"], [0.22, 0.38, 0.40]),
        ("M-M-H", ["MEDIUM", "MEDIUM", "HARD"], [0.30, 0.32, 0.38]),
    ]
    out = []
    for name, compounds, fracs in raw:
        needs = {c: compounds.count(c) for c in set(compounds)}
        feasible = all(_sets_available(inputs.tyre_sets, c) >= n for c, n in needs.items())
        if feasible:
            out.append({"name": name, "compounds": compounds, "fractions": fracs})
    return out


def _pit_window(strategy, race_laps):
    cuts = np.cumsum(strategy["fractions"])[:-1]
    windows = []
    for c in cuts:
        lap = int(round(c * race_laps))
        windows.append(f"L{max(1, lap-3)}–{min(race_laps-1, lap+3)}")
    return " / ".join(windows) if windows else "No stop"


def _strategy_cost(strategy, inputs, rng, n):
    laps_total = inputs.circuit.race_laps
    costs = np.zeros(n, dtype=float)

    deg_shock = rng.lognormal(mean=0.0, sigma=0.16, size=n)
    traffic_shock = rng.gamma(shape=1.8, scale=1.0, size=n)
    sc = rng.random(n) < inputs.sc_probability
    rain = rng.random(n) < inputs.rain_probability

    for comp, frac in zip(strategy["compounds"], strategy["fractions"]):
        tyre = inputs.tyres[comp]
        stint_laps = max(1, int(round(frac * laps_total)))
        costs += tyre.pace_delta * stint_laps
        costs += tyre.degradation * deg_shock * stint_laps * (stint_laps - 1) / 2.0

    stops = len(strategy["compounds"]) - 1
    sc_alignment = min(0.88, 0.45 + 0.15 * stops)
    effective_pit = inputs.circuit.pit_loss_green - sc.astype(float) * sc_alignment * (
        inputs.circuit.pit_loss_green - inputs.circuit.pit_loss_sc
    )
    costs += stops * effective_pit

    costs += stops * inputs.circuit.overtaking_difficulty * (2.0 + 3.5 * traffic_shock)
    costs -= stops * inputs.circuit.undercut_power * (1.0 + rng.random(n) * 2.0)

    if strategy["compounds"][0] == "SOFT":
        costs -= 1.2 + rng.normal(0, 0.8, n)

    costs += rain.astype(float) * 18.0
    costs += inputs.driver.race_pace_delta * laps_total
    return costs


def _softmax(values, temperature=7.0):
    v = np.asarray(values, dtype=float)
    v = -(v - np.nanmin(v)) / temperature
    e = np.exp(v - np.max(v))
    return e / e.sum()


def run_monte_carlo(inputs: SimulationInputs) -> Dict[str, Any]:
    rng = np.random.default_rng(20260913)
    strategies = _strategy_catalog(inputs)
    if not strategies:
        raise ValueError("No feasible dry strategies with the current tyre-set availability.")

    n = int(inputs.simulations)
    costs = [_strategy_cost(s, inputs, rng, n) for s in strategies]
    cost_matrix = np.vstack(costs).T

    optimal_idx = np.argmin(cost_matrix, axis=1)
    optimal_probs = np.array([(optimal_idx == i).mean() for i in range(len(strategies))])
    expected_cost = cost_matrix.mean(axis=0)

    team_adjusted = []
    for i, s in enumerate(strategies):
        stops = len(s["compounds"]) - 1
        conservatism = (1.0 - inputs.driver.team_risk) * stops * 2.3
        track_position_bias = inputs.circuit.overtaking_difficulty * stops * 1.8
        team_adjusted.append(expected_cost[i] + conservatism + track_position_bias)
    team_probs = _softmax(team_adjusted, temperature=6.5)

    base_finish = inputs.driver.grid_position + inputs.driver.race_pace_delta * 3.5
    rows = []
    for i, s in enumerate(strategies):
        relative = expected_cost[i] - expected_cost.min()
        expected_finish = float(np.clip(base_finish + relative / 7.5, 1, 22))
        win = float(np.clip(0.62 * math.exp(-0.72 * (expected_finish - 1)), 0, 1))
        podium = float(np.clip(1 / (1 + math.exp(1.08 * (expected_finish - 3.4))), 0, 1))
        rows.append({
            "strategy": s["name"],
            "pit_window": _pit_window(s, inputs.circuit.race_laps),
            "expected_cost_s": round(float(expected_cost[i]), 1),
            "optimal_probability": float(optimal_probs[i]),
            "team_choice_probability": float(team_probs[i]),
            "expected_finish": round(expected_finish, 1),
            "win_probability": round(win * 100, 1),
            "podium_probability": round(podium * 100, 1),
        })

    best_opt_i = int(np.argmax(optimal_probs))
    best_team_i = int(np.argmax(team_probs))

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

    dry_one = [i for i, s in enumerate(strategies) if len(s["compounds"]) == 2]
    dry_two = [i for i, s in enumerate(strategies) if len(s["compounds"]) == 3]
    one_best = strategies[min(dry_one, key=lambda i: expected_cost[i])]["name"] if dry_one else strategies[best_opt_i]["name"]
    two_best = strategies[min(dry_two, key=lambda i: expected_cost[i])]["name"] if dry_two else strategies[best_opt_i]["name"]

    scenarios = [
        {"scenario": "Dry stable", "probability": probs["Dry stable"], "recommended_strategy": one_best,
         "trigger": "Degradation below forecast and no early neutralisation"},
        {"scenario": "Dry high degradation", "probability": probs["Dry high degradation"], "recommended_strategy": two_best,
         "trigger": "Observed degradation remains above the one-stop threshold"},
        {"scenario": "Dry + SC/VSC", "probability": probs["Dry + SC/VSC"],
         "recommended_strategy": two_best if inputs.circuit.pit_loss_green-inputs.circuit.pit_loss_sc > 7 else one_best,
         "trigger": "Neutralisation overlaps a viable tyre window"},
        {"scenario": "Mixed / Wet", "probability": probs["Mixed / Wet"], "recommended_strategy": "Weather reactive",
         "trigger": "Crossover time beats dry-tyre extension"},
        {"scenario": "Wet + SC/VSC", "probability": probs["Wet + SC/VSC"], "recommended_strategy": "Weather reactive + cheap stop",
         "trigger": "Rain + neutralisation"},
    ]

    radar = (
        f"{inputs.driver.driver_name}: model favours {strategies[best_team_i]['name']} "
        f"as the most likely team choice ({team_probs[best_team_i]:.0%}). "
        f"The mathematically optimal leader is {strategies[best_opt_i]['name']} "
        f"({optimal_probs[best_opt_i]:.0%}). Watch tyre degradation, the first pit window and any SC/VSC."
    )

    return {
        "strategies": rows,
        "predicted_team_strategy": strategies[best_team_i]["name"],
        "predicted_team_probability": float(team_probs[best_team_i]),
        "optimal_strategy": strategies[best_opt_i]["name"],
        "optimal_probability": float(optimal_probs[best_opt_i]),
        "scenarios": scenarios,
        "radar_message": radar,
    }
