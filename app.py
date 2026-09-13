from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import plotly.express as px

from data_sources import OpenF1Client, OpenMeteoClient, build_prerace_snapshot, build_live_snapshot
from strategy_engine import CircuitProfile, TyreModel, DriverContext, SimulationInputs, run_monte_carlo

st.set_page_config(page_title="F1 Strategy Engine", page_icon="🏎️", layout="wide")
st.title("F1 Strategy Engine")
st.caption("Pre-race prediction + live strategic radar. Automatic inputs first; manual overrides only when needed.")

@st.cache_resource
def clients():
    return OpenF1Client(), OpenMeteoClient()

openf1, meteo = clients()

with st.sidebar:
    st.header("Race setup")
    current_year = datetime.now(timezone.utc).year
    year = st.number_input("Season", min_value=2023, max_value=current_year + 1, value=current_year, step=1)

    try:
        meetings = openf1.meetings(int(year))
    except Exception as exc:
        meetings = []
        st.error(f"OpenF1 unavailable: {exc}")
    if not meetings:
        st.stop()

    meeting_labels = [f"{m.get('meeting_name', 'GP')} — {m.get('country_name','')}" for m in meetings]
    meeting_idx = st.selectbox(
        "Grand Prix",
        range(len(meetings)),
        index=max(0, len(meetings)-1),
        format_func=lambda i: meeting_labels[i]
    )
    meeting = meetings[meeting_idx]

    sessions = openf1.sessions(meeting["meeting_key"])
    race_sessions = [s for s in sessions if s.get("session_name") == "Race"]
    race_session = race_sessions[0] if race_sessions else (sessions[-1] if sessions else None)
    if not race_session:
        st.error("Race session not found.")
        st.stop()

    drivers = openf1.drivers(race_session["session_key"])
    if not drivers:
        for s in reversed(sessions):
            drivers = openf1.drivers(s["session_key"])
            if drivers:
                break

    driver_by_name = {d.get("full_name", str(d.get("driver_number"))): d for d in drivers}
    if not driver_by_name:
        st.error("Driver list unavailable.")
        st.stop()

    driver_name = st.selectbox("Driver", list(driver_by_name.keys()))
    driver = driver_by_name[driver_name]
    objective = st.selectbox("Objective", ["Best expected finish", "Podium probability", "Win probability"])
    live_mode = st.toggle("Live mode", value=False)
    sim_runs = st.slider("Monte Carlo simulations", 5000, 100000, 30000, 5000)

st.divider()

try:
    snapshot = (
        build_live_snapshot(openf1, meeting, race_session, driver)
        if live_mode
        else build_prerace_snapshot(openf1, meteo, meeting, sessions, race_session, driver)
    )
except Exception as exc:
    st.warning(f"Some automatic sources failed: {exc}")
    snapshot = {}

auto = snapshot.get("auto", {})
confidence = snapshot.get("confidence", {})
practice = snapshot.get("practice", {})

grid_position = int(auto.get("grid_position") or 10)
race_laps = int(auto.get("race_laps") or 57)
air_temp = float(auto.get("air_temperature") or 25.0)
track_temp = float(auto.get("track_temperature") or air_temp + 15)
rain_prob = float(auto.get("rain_probability") or 0.05)
pit_loss = float(auto.get("pit_loss_green") or 22.0)
pit_loss_sc = float(auto.get("pit_loss_sc") or max(8.0, pit_loss * 0.55))
sc_prob = float(auto.get("sc_probability") or 0.35)
overtaking = float(auto.get("overtaking_difficulty") or 0.55)
undercut = float(auto.get("undercut_power") or 0.60)
team_risk = float(auto.get("team_risk") or 0.50)
race_pace_delta = float(auto.get("race_pace_delta") or 0.0)

deg = practice.get("degradation", {})
soft_deg = float(deg.get("SOFT", 0.12))
medium_deg = float(deg.get("MEDIUM", 0.08))
hard_deg = float(deg.get("HARD", 0.055))

availability = snapshot.get("tyre_availability", {
    "SOFT": {"new": 1, "used": 1, "confidence": "low"},
    "MEDIUM": {"new": 1, "used": 1, "confidence": "low"},
    "HARD": {"new": 1, "used": 0, "confidence": "low"},
})

tyre_sets = {
    c: {
        "new": int(availability.get(c, {}).get("new", 1)),
        "used": int(availability.get(c, {}).get("used", 0)),
    }
    for c in ["SOFT", "MEDIUM", "HARD"]
}

with st.expander("Manual overrides — only if needed", expanded=False):
    st.caption("Leave the automatic value unless you have better information.")
    c1, c2, c3 = st.columns(3)
    with c1:
        grid_position = st.number_input("Grid position", 1, 30, grid_position)
        pit_loss = st.number_input("Pit loss green flag (s)", 10.0, 40.0, pit_loss, 0.1)
        sc_prob = st.slider("SC/VSC probability", 0.0, 1.0, sc_prob, 0.01)
    with c2:
        air_temp = st.number_input("Air temperature °C", -5.0, 50.0, air_temp, 0.5)
        track_temp = st.number_input("Track temperature °C", 0.0, 75.0, track_temp, 0.5)
        rain_prob = st.slider("Rain probability", 0.0, 1.0, rain_prob, 0.01)
    with c3:
        soft_deg = st.number_input("Soft degradation s/lap", 0.0, 0.6, soft_deg, 0.005)
        medium_deg = st.number_input("Medium degradation s/lap", 0.0, 0.6, medium_deg, 0.005)
        hard_deg = st.number_input("Hard degradation s/lap", 0.0, 0.6, hard_deg, 0.005)

    st.markdown("**Tyre sets available**")
    tcols = st.columns(3)
    for col, compound in zip(tcols, ["SOFT", "MEDIUM", "HARD"]):
        with col:
            tyre_sets[compound]["new"] = st.number_input(
                f"{compound.title()} new", 0, 5, tyre_sets[compound]["new"], key=f"{compound}_new"
            )
            tyre_sets[compound]["used"] = st.number_input(
                f"{compound.title()} used", 0, 5, tyre_sets[compound]["used"], key=f"{compound}_used"
            )

cols = st.columns(6)
cards = [
    ("Grid", f"P{grid_position}"),
    ("Weather", f"{air_temp:.0f}°C / {rain_prob:.0%} rain"),
    ("Track", f"{track_temp:.0f}°C"),
    ("Pit loss", f"{pit_loss:.1f}s"),
    ("SC/VSC", f"{sc_prob:.0%}"),
    ("Auto inputs", f"{snapshot.get('automatic_share', 0.0):.0%}"),
]
for col, (label, value) in zip(cols, cards):
    col.metric(label, value)

st.subheader("Automatic data quality")
if confidence:
    st.dataframe(
        pd.DataFrame([{"Variable": k, "Confidence": v} for k, v in confidence.items()]),
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("No source-confidence metadata available.")

tyres = {
    "SOFT": TyreModel("SOFT", -0.55, soft_deg),
    "MEDIUM": TyreModel("MEDIUM", 0.00, medium_deg),
    "HARD": TyreModel("HARD", 0.45, hard_deg),
}
circuit = CircuitProfile(race_laps, pit_loss, pit_loss_sc, overtaking, undercut, track_temp)
driver_ctx = DriverContext(
    driver_name,
    driver.get("team_name", ""),
    int(grid_position),
    race_pace_delta,
    team_risk
)
inputs = SimulationInputs(
    circuit,
    driver_ctx,
    tyres,
    tyre_sets,
    sc_prob,
    rain_prob,
    int(sim_runs),
    objective
)
result = run_monte_carlo(inputs)

st.subheader(f"Predicted strategy — {driver_name}")
a, b, c, d = st.columns(4)
a.metric("Most likely team choice", result["predicted_team_strategy"])
b.metric("Team choice probability", f"{result['predicted_team_probability']:.1%}")
c.metric("Most likely optimal", result["optimal_strategy"])
d.metric("Optimal probability", f"{result['optimal_probability']:.1%}")

strategy_df = pd.DataFrame(result["strategies"])
strategy_df["Optimal probability"] = strategy_df["optimal_probability"] * 100
strategy_df["Team choice probability"] = strategy_df["team_choice_probability"] * 100

left, right = st.columns([1.35, 1])
with left:
    fig = px.bar(
        strategy_df,
        x="strategy",
        y=["Optimal probability", "Team choice probability"],
        barmode="group",
        labels={"value": "Probability (%)", "strategy": "Strategy", "variable": ""}
    )
    fig.update_layout(height=420, legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)
with right:
    table = strategy_df[[
        "strategy", "pit_window", "expected_cost_s",
        "expected_finish", "win_probability", "podium_probability"
    ]].copy()
    table.columns = [
        "Strategy", "Pit window", "Expected cost (s)",
        "Expected finish", "Win %", "Podium %"
    ]
    st.dataframe(table, hide_index=True, use_container_width=True)

st.subheader("Race scenarios")
scenarios = pd.DataFrame(result["scenarios"])
scenarios["probability"] *= 100
scenarios.columns = [
    "Scenario", "Probability %",
    "Recommended strategy", "Trigger to watch"
]
st.dataframe(scenarios, hide_index=True, use_container_width=True)

st.subheader("Strategic radar")
st.info(result["radar_message"])

if live_mode:
    live = snapshot.get("live", {})
    st.subheader("Live feed")
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Lap", live.get("lap", "—"))
    l2.metric("Position", f"P{live.get('position','—')}")
    l3.metric("Compound", live.get("compound", "—"))
    l4.metric("Gap ahead", live.get("gap_ahead", "—"))
    st.caption("Live OpenF1 timing is subject to the provider's real-time access.")

st.divider()
st.caption("Research prototype: automatic data is confidence-scored. Low-confidence values remain overrideable.")
