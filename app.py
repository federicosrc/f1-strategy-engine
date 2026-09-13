from __future__ import annotations

from datetime import datetime, timezone
import math
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from official_sources import F1OfficialClient, PirelliCurrentSeason
from data_sources import (
    APIError,
    OpenF1Client,
    OpenMeteoClient,
    grid_position,
    infer_tyre_inventory,
    practice_dataset,
    race_pace_delta,
    resolve_meeting,
    select_practice_sessions,
    weekend_incident_risk,
)
from strategy_engine import (
    CircuitProfile,
    DriverContext,
    SimulationInputs,
    TyreModel,
    estimate_degradation_from_practice,
    run_monte_carlo,
)

st.set_page_config(
    page_title="Strategy Engine",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = r"""
<style>
:root {
  --bg:#080b0f; --panel:#0d1218; --panel2:#111820; --line:#26323d;
  --muted:#8d9aa7; --white:#f6f7f9; --red:#ff1e2d; --yellow:#ffd21f;
  --green:#37e77b; --blue:#39b8ff; --orange:#ff9a33;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--white); }
[data-testid="stAppViewContainer"] {
  background-image:
    linear-gradient(rgba(255,255,255,.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.015) 1px, transparent 1px),
    radial-gradient(circle at 80% 0%, rgba(255,30,45,.06), transparent 28%);
  background-size: 32px 32px, 32px 32px, auto;
}
section[data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none!important; }
[data-testid="stHeader"] button[kind="header"] { display:none!important; }
[data-testid="stHeader"] { background:rgba(8,11,15,.82); }
.block-container { padding-top:.85rem; padding-bottom:2rem; padding-left:1rem; padding-right:1rem; max-width:1900px; }
.setup-season { min-height:68px; display:flex; flex-direction:column; justify-content:center; padding:4px 2px; }
.setup-season .k { color:var(--muted); font-size:9px; letter-spacing:.13em; text-transform:uppercase; }
.setup-season .v { font-size:18px; font-weight:950; margin-top:3px; }
.setup-help { color:var(--muted); font-size:10px; margin-top:-2px; margin-bottom:3px; }
.circuit-stats { display:grid; grid-template-columns:1fr; gap:8px; }
.circuit-stat { border:1px solid var(--line); background:#0b1117; border-radius:6px; padding:9px 10px; min-height:72px; }
.circuit-stat .k { color:var(--muted); font-size:9px; text-transform:uppercase; letter-spacing:.1em; }
.circuit-stat .v { color:var(--white); font-size:25px; line-height:1.05; font-weight:950; margin-top:5px; white-space:nowrap; }
.circuit-stat .u { color:var(--muted); font-size:10px; margin-left:4px; font-weight:700; }
.circuit-stat .s { color:var(--muted); font-size:9px; margin-top:4px; }
h1,h2,h3 { letter-spacing:-.02em; }
.se-header {
  display:flex; align-items:center; justify-content:space-between; gap:18px;
  padding:13px 18px; border:1px solid var(--line); border-radius:8px;
  background:linear-gradient(135deg,#10161d,#090d12); margin-bottom:10px;
  box-shadow: inset 4px 0 0 var(--red);
}
.se-brand { font-size:28px; font-weight:900; letter-spacing:.02em; }
.se-brand span { color:var(--red); }
.se-sub { color:var(--muted); font-size:11px; letter-spacing:.15em; text-transform:uppercase; }
.se-headchips { display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }
.se-chip { border:1px solid var(--line); padding:8px 12px; border-radius:6px; background:#0c1117; min-width:120px; }
.se-chip .k { color:var(--muted); font-size:9px; text-transform:uppercase; letter-spacing:.12em; }
.se-chip .v { color:white; font-size:14px; font-weight:800; margin-top:2px; }
.panel-title { font-weight:900; font-size:16px; letter-spacing:.03em; text-transform:uppercase; margin-bottom:8px; }
.panel-title:before { content:""; display:inline-block; width:4px; height:16px; background:var(--red); margin-right:9px; vertical-align:-2px; }
.metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; }
.metric-card { border:1px solid var(--line); border-radius:6px; background:#0b1117; padding:10px 12px; min-height:70px; }
.metric-card .k { color:var(--muted); font-size:9px; text-transform:uppercase; letter-spacing:.1em; }
.metric-card .v { font-size:20px; font-weight:900; margin-top:4px; }
.metric-card .s { color:var(--muted); font-size:10px; margin-top:2px; }
.tyre-row { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
.tyre { border:1px solid var(--line); border-radius:7px; padding:11px 12px; background:#0b1016; text-align:center; }
.tyre.hard { box-shadow:inset 0 0 0 1px rgba(255,255,255,.28); }
.tyre.medium { border-color:#8a7413; box-shadow:inset 0 0 18px rgba(255,210,31,.05); }
.tyre.soft { border-color:#84242d; box-shadow:inset 0 0 18px rgba(255,30,45,.05); }
.tyre .name { font-size:12px; font-weight:900; letter-spacing:.08em; }
.tyre .compound { font-size:30px; font-weight:950; margin:6px 0 2px; }
.tyre.medium .compound { color:var(--yellow); }.tyre.soft .compound{color:var(--red);}
.tyre .small { color:var(--muted); font-size:9px; text-transform:uppercase; }
.strategy-hero { border:1px solid var(--line); border-radius:7px; padding:15px; background:linear-gradient(135deg,#0d141b,#090d12); }
.strategy-big { font-size:46px; font-weight:950; letter-spacing:.02em; color:var(--yellow); margin:2px 0; }
.strategy-label { color:var(--muted); font-size:9px; text-transform:uppercase; letter-spacing:.14em; }
.strategy-opt { border-color:#1d8d48; box-shadow:inset 0 0 24px rgba(55,231,123,.05); }
.strategy-opt .strategy-big { color:#f4f4f4; }
.barline { height:8px; background:#1b252e; border-radius:2px; overflow:hidden; }
.barline > div { height:100%; }
.quality { display:flex; align-items:center; justify-content:space-between; gap:10px; border-bottom:1px solid #1b252e; padding:7px 0; }
.qname { font-size:11px; }.qsource { color:var(--muted); font-size:9px; }
.badge { padding:3px 7px; border-radius:4px; font-size:9px; font-weight:900; text-transform:uppercase; }
.badge.high { background:rgba(55,231,123,.12); color:var(--green); border:1px solid #237b45; }
.badge.medium { background:rgba(255,210,31,.10); color:var(--yellow); border:1px solid #725f12; }
.badge.low,.badge.pending { background:rgba(255,154,51,.10); color:var(--orange); border:1px solid #73461d; }
.radar-row { display:grid; grid-template-columns:1fr 80px; gap:10px; align-items:center; padding:9px 0; border-bottom:1px solid #1b252e; }
.radar-name { font-size:12px; font-weight:800; }.radar-sub { color:var(--muted); font-size:9px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line)!important; background:rgba(11,16,22,.74); border-radius:8px!important; }
[data-testid="stMetric"] { background:#0b1016; border:1px solid var(--line); padding:10px 12px; border-radius:6px; }
[data-testid="stMetricLabel"] { color:var(--muted); }
.stButton > button { border-radius:5px; border:1px solid #ff3440; background:linear-gradient(180deg,#f32635,#d60f20); color:white; font-weight:900; letter-spacing:.04em; text-transform:uppercase; }
.stButton > button:hover { border-color:#ff6570; color:white; }
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input { background:#0b1117!important; }
@media (max-width: 900px) { .metric-grid { grid-template-columns:repeat(2,1fr); } .se-header{display:block;} .se-headchips{justify-content:flex-start;margin-top:10px;} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def clients():
    return F1OfficialClient(), PirelliCurrentSeason(), OpenF1Client(), OpenMeteoClient()


f1, pirelli, openf1, meteo = clients()
CURRENT_YEAR = datetime.now(timezone.utc).year


@st.cache_data(ttl=21600, show_spinner=False)
def official_calendar():
    return F1OfficialClient().calendar()


@st.cache_data(ttl=21600, show_spinner=False)
def official_event(event_key: str, event_payload: dict):
    return F1OfficialClient().event_details(event_payload)


@st.cache_data(ttl=21600, show_spinner=False)
def official_drivers():
    return F1OfficialClient().drivers()


def parse_race_datetime(details):
    txt = details.get("race_schedule_text")
    if not txt:
        return None
    for fmt in ("%d %b %H:%M",):
        try:
            dt = datetime.strptime(txt, fmt).replace(year=CURRENT_YEAR, tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def current_event_index(events):
    # Current-season only. Date hints come from the official-calendar fallback metadata.
    months = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    today = datetime.now(timezone.utc).date()
    best = len(events)-1 if events else 0
    for i, e in enumerate(events):
        d = e.get("dates", "")
        m = re.search(r"(?:\d{2}\s*[A-Za-z]*[–-])?(\d{2})\s+([A-Z][a-z]{2})", d)
        if not m:
            m = re.search(r"(\d{2}).*?([A-Z][a-z]{2})", d)
        if not m:
            continue
        day = int(m.group(1)); month = months.get(m.group(2))
        if not month:
            continue
        end_date = datetime(CURRENT_YEAR, month, day, tzinfo=timezone.utc).date()
        if end_date >= today:
            return i
        best = i
    return best


def panel_title(text):
    st.markdown(f'<div class="panel-title">{text}</div>', unsafe_allow_html=True)


def metric_html(items):
    cards = "".join(
        f'<div class="metric-card"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
        for k, v, s in items
    )
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)


def confidence_badge(level):
    level = (level or "low").lower()
    return f'<span class="badge {level}">{level}</span>'


def source_row(name, source, confidence):
    return (
        '<div class="quality">'
        f'<div><div class="qname">{name}</div><div class="qsource">{source}</div></div>'
        f'{confidence_badge(confidence)}</div>'
    )


calendar = official_calendar()
if not calendar:
    st.error("Current Formula 1 calendar could not be loaded.")
    st.stop()

def_idx = current_event_index(calendar)

drivers = official_drivers()
driver_index = drivers.index("Charles Leclerc") if "Charles Leclerc" in drivers else 0

with st.container(border=True):
    setup_cols = st.columns([0.80, 2.15, 1.70, 1.85, 1.45, 1.85], gap="small", vertical_alignment="bottom")
    with setup_cols[0]:
        st.markdown(
            f'<div class="setup-season"><div class="k">Current season</div><div class="v">{CURRENT_YEAR}</div></div>',
            unsafe_allow_html=True,
        )
    with setup_cols[1]:
        event_idx = st.selectbox(
            "Grand Prix",
            range(len(calendar)),
            index=min(def_idx, len(calendar)-1),
            format_func=lambda i: f"R{calendar[i].get('round','—')} · {calendar[i].get('name','Grand Prix')}",
        )
    event = calendar[event_idx]
    with setup_cols[2]:
        selected_driver = st.selectbox("Driver", drivers, index=driver_index)
    with setup_cols[3]:
        objective = st.selectbox("Objective", ["Best expected finish", "Podium probability", "Win probability"])
    with setup_cols[4]:
        simulations = st.selectbox(
            "Simulations",
            [5000, 10000, 20000, 30000, 50000, 75000, 100000],
            index=3,
            format_func=lambda x: f"{x:,}".replace(",", "."),
        )
    with setup_cols[5]:
        analyse = st.button("Run / refresh strategy", use_container_width=True)

st.markdown(
    '<div class="setup-help">Run once to load the current weekend. After that, changing Grand Prix or driver automatically recalculates the driver-specific model using cached session data whenever possible.</div>',
    unsafe_allow_html=True,
)

details = official_event(event["key"], event)
compound_info = pirelli.compounds(event["key"])
race_dt = parse_race_datetime(details)

weather = {}
try:
    geo = meteo.geocode(details.get("location", event.get("location", "")), details.get("country", event.get("country", "")))
    weather = meteo.forecast_at(geo["latitude"], geo["longitude"], race_dt)
except Exception:
    weather = {}

# Values available before any OpenF1 call.
air_temp = float(weather.get("temperature_2m", 25.0))
track_temp = float(weather.get("track_temperature_estimate", air_temp + 15.0))
rain_prob = float(weather.get("precipitation_probability", 5.0) or 0.0) / 100.0
wind = float(weather.get("wind_speed_10m", 0.0) or 0.0)
humidity = float(weather.get("relative_humidity_2m", 0.0) or 0.0)
race_laps = int(details.get("number_of_laps") or 57)
circuit_type = details.get("circuit_type", "Permanent")

# Strategic priors before current-weekend timing data.
if circuit_type == "Street":
    overtaking = 0.82; sc_prob = 0.48; pit_loss = 23.0
elif circuit_type == "Semi-permanent":
    overtaking = 0.70; sc_prob = 0.40; pit_loss = 23.5
else:
    overtaking = 0.56; sc_prob = 0.31; pit_loss = 22.0
pit_loss_sc = pit_loss * 0.55
undercut = min(0.90, 0.48 + overtaking * 0.35)

analysis_data = st.session_state.get("analysis_data")
analysis_key = st.session_state.get("analysis_key")
this_key = f"{CURRENT_YEAR}:{event['key']}:{selected_driver}"

# The first click activates current-weekend analytics. From that point on a GP/driver
# change automatically recalculates the selected driver. OpenF1 session payloads are
# cached, so switching from Leclerc to Hamilton normally filters the already-downloaded
# FP data instead of requesting the same laps/stints again.
if analyse:
    st.session_state["analysis_enabled"] = True

analysis_enabled = bool(st.session_state.get("analysis_enabled", False))
needs_analysis = analysis_enabled and (analyse or analysis_key != this_key)

if analysis_key != this_key:
    analysis_data = None

if needs_analysis:
    with st.spinner(f"Updating current-weekend model for {selected_driver}…"):
        try:
            meeting = resolve_meeting(openf1, details, CURRENT_YEAR)
            sessions = openf1.sessions(meeting["meeting_key"])
            practices = select_practice_sessions(sessions)
            if not practices:
                raise APIError("No completed practice session available yet")

            practice = practices[0]
            session_drivers = openf1.drivers(practice["session_key"])
            target = selected_driver.lower().strip()
            driver_row = None
            for d in session_drivers:
                full = str(d.get("full_name", "")).lower().strip()
                broadcast = str(d.get("broadcast_name", "")).lower().strip()
                if target == full or target == broadcast or target.split()[-1] in full:
                    driver_row = d
                    break
            if not driver_row:
                raise APIError("Selected driver not found in current weekend data")
            driver_number = int(driver_row["driver_number"])

            dataset, all_laps, raw_stints = practice_dataset(openf1, practice, driver_number)
            # If the latest practice did not contain a usable long run, try the next one.
            if len(dataset) < 5 and len(practices) > 1:
                for candidate in practices[1:]:
                    candidate_dataset, candidate_laps, candidate_stints = practice_dataset(openf1, candidate, driver_number)
                    if len(candidate_dataset) > len(dataset):
                        practice = candidate
                        dataset, all_laps, raw_stints = candidate_dataset, candidate_laps, candidate_stints
                    if len(dataset) >= 5:
                        break

            degradation = estimate_degradation_from_practice(dataset)
            pace_delta = race_pace_delta(all_laps, driver_number)
            gp = grid_position(openf1, sessions, driver_number)
            incident, incident_conf = weekend_incident_risk(openf1, practice)
            inventory = infer_tyre_inventory(raw_stints)

            analysis_data = {
                "meeting": meeting,
                "practice_name": practice.get("session_name", "Practice"),
                "driver_number": driver_number,
                "team_name": driver_row.get("team_name", ""),
                "grid_position": gp,
                "degradation": degradation,
                "pace_delta": pace_delta,
                "incident_risk": incident,
                "incident_conf": incident_conf,
                "inventory": inventory,
                "practice_rows": int(len(dataset)),
            }
            st.session_state["analysis_data"] = analysis_data
            st.session_state["analysis_key"] = this_key
            analysis_key = this_key
        except Exception as exc:
            st.session_state["analysis_data"] = None
            st.session_state["analysis_key"] = this_key
            analysis_data = None
            analysis_key = this_key
            st.warning(
                f"Current-weekend timing could not be fully loaded for {selected_driver}: {exc}. "
                "The dashboard remains usable with official and model fallbacks."
            )

# Current-weekend refined values.
grid = int((analysis_data or {}).get("grid_position") or 10)
pace_delta = float((analysis_data or {}).get("pace_delta") or 0.0)
incident_risk = float((analysis_data or {}).get("incident_risk") or sc_prob)
sc_prob = min(0.75, max(0.12, 0.55 * sc_prob + 0.45 * incident_risk))

deg = (analysis_data or {}).get("degradation", {})
soft_deg = float(deg.get("SOFT", 0.12))
medium_deg = float(deg.get("MEDIUM", 0.08))
hard_deg = float(deg.get("HARD", 0.055))
undercut = min(0.95, undercut + max(0, medium_deg - 0.06) * 0.7)

inventory = (analysis_data or {}).get("inventory") or {
    "SOFT": {"new": 1, "used": 1},
    "MEDIUM": {"new": 1, "used": 1},
    "HARD": {"new": 1, "used": 1},
}

# Only low-confidence values are exposed as overrides. They are now horizontal and
# scoped by GP + driver, so tyre values entered for Hamilton cannot leak into Leclerc.
with st.expander("Low-confidence overrides", expanded=False):
    st.caption("Change these only when you have better official information.")
    o1, o2, o3, o4, o5 = st.columns([1.15, 1.15, 1.25, 1.25, 1.25], gap="small")
    with o1:
        pit_loss = st.number_input(
            "Pit lane loss (s)", min_value=10.0, max_value=40.0,
            value=float(pit_loss), step=0.1,
            key=f"{event['key']}_pit_loss",
        )
    with o2:
        sc_prob = st.slider(
            "SC/VSC probability", 0.0, 1.0, float(sc_prob), 0.01,
            key=f"{event['key']}_sc_prob",
        )
    tyre_cols = {"SOFT": o3, "MEDIUM": o4, "HARD": o5}
    for c, col in tyre_cols.items():
        with col:
            st.markdown(f"**{c.title()} sets**")
            a, b = st.columns(2)
            base_key = f"{CURRENT_YEAR}:{event['key']}:{selected_driver}:{c}"
            inventory[c]["new"] = a.number_input(
                "New", 0, 5, int(inventory[c].get("new", 1)),
                key=f"{base_key}:new",
            )
            inventory[c]["used"] = b.number_input(
                "Used", 0, 5, int(inventory[c].get("used", 0)),
                key=f"{base_key}:used",
            )

# Header
st.markdown(
    f'''<div class="se-header">
      <div><div class="se-brand"><span>STRATEGY</span> ENGINE</div><div class="se-sub">Current-season race strategy dashboard</div></div>
      <div class="se-headchips">
        <div class="se-chip"><div class="k">Grand Prix</div><div class="v">{details.get('name', event.get('name'))}</div></div>
        <div class="se-chip"><div class="k">Driver</div><div class="v">{selected_driver}</div></div>
        <div class="se-chip"><div class="k">Model</div><div class="v">{'Driver-specific' if analysis_data else 'Baseline'}</div></div>
        <div class="se-chip"><div class="k">Forecast</div><div class="v">{air_temp:.0f}°C · {rain_prob:.0%} rain</div></div>
      </div>
    </div>''', unsafe_allow_html=True
)

# Top dashboard row
circuit_col, weather_col, tyre_col = st.columns([1.18, 1.0, 1.0], gap="small")
with circuit_col:
    with st.container(border=True):
        panel_title("Circuit info")
        map_col, info_col = st.columns([1.52, 0.82])
        with map_col:
            if details.get("track_image_url"):
                st.image(details["track_image_url"], use_container_width=True)
            else:
                st.markdown("#### Track map")
                st.caption("Official Formula 1 map will appear here when the page exposes a usable image URL.")
        with info_col:
            laps_display = details.get("number_of_laps", "—")
            length = details.get("circuit_length_km")
            dist = details.get("race_distance_km")
            length_prefix = "≈" if details.get("circuit_length_derived") else ""
            dist_prefix = "≈" if details.get("race_distance_derived") else ""
            length_display = f"{length_prefix}{length:.3f}" if isinstance(length, (float, int)) else "—"
            dist_display = f"{dist_prefix}{dist:.3f}" if isinstance(dist, (float, int)) else "—"
            length_note = "derived fallback" if details.get("circuit_length_derived") else "F1 official"
            dist_note = "derived fallback" if details.get("race_distance_derived") else "F1 official"
            stats_html = (
                f'<div class="circuit-stats">'
                f'<div class="circuit-stat"><div class="k">Laps</div><div class="v">{laps_display}</div><div class="s">F1 official</div></div>'
                f'<div class="circuit-stat"><div class="k">Circuit length</div><div class="v">{length_display}<span class="u">km</span></div><div class="s">{length_note}</div></div>'
                f'<div class="circuit-stat"><div class="k">Race distance</div><div class="v">{dist_display}<span class="u">km</span></div><div class="s">{dist_note}</div></div>'
                f'</div>'
            )
            st.markdown(stats_html, unsafe_allow_html=True)
        st.caption(f"{details.get('location', event.get('location',''))} · {circuit_type} · Source: Formula 1 official")

with weather_col:
    with st.container(border=True):
        panel_title("Weather & track")
        metric_html([
            ("Air temp", f"{air_temp:.0f}°C", "Open-Meteo forecast"),
            ("Track temp", f"{track_temp:.0f}°C", "estimated asphalt"),
            ("Rain probability", f"{rain_prob:.0%}", "race-time forecast"),
            ("Wind", f"{wind:.0f} km/h", f"humidity {humidity:.0f}%"),
        ])
        st.markdown("<br>", unsafe_allow_html=True)
        stress = min(1.0, max(0.05, (medium_deg / 0.16) + max(0, track_temp - 38) / 80))
        evolution = 0.78 if circuit_type in {"Street", "Semi-permanent"} else 0.55
        for label, val, color in [
            ("Tyre stress", stress, "#ffd21f"),
            ("Overtaking difficulty", overtaking, "#ff1e2d"),
            ("Undercut power", undercut, "#37e77b"),
            ("Track evolution", evolution, "#39b8ff"),
        ]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:10px;margin:6px 0 3px"><span>{label}</span><b>{val:.0%}</b></div><div class="barline"><div style="width:{val*100:.0f}%;background:{color}"></div></div>', unsafe_allow_html=True)

with tyre_col:
    with st.container(border=True):
        panel_title("Tyre compounds")
        st.markdown(
            f'''<div class="tyre-row">
              <div class="tyre hard"><div class="name">HARD</div><div class="compound">{compound_info['hard']}</div><div class="small">white</div></div>
              <div class="tyre medium"><div class="name">MEDIUM</div><div class="compound">{compound_info['medium']}</div><div class="small">yellow</div></div>
              <div class="tyre soft"><div class="name">SOFT</div><div class="compound">{compound_info['soft']}</div><div class="small">red</div></div>
            </div>''', unsafe_allow_html=True
        )
        st.caption(f"{compound_info['status']} · Pirelli official · {compound_info['confidence']} confidence")
        st.markdown("**Estimated race availability**")
        inv_text = " · ".join(f"{c[0]} {inventory[c]['new']}N/{inventory[c]['used']}U" for c in ["SOFT", "MEDIUM", "HARD"])
        st.code(inv_text, language=None)
        st.caption("Remaining sets stay overrideable because the authoritative Pirelli list is not yet machine-readable in this V1.3.")

# Build simulation.
tyres = {
    "SOFT": TyreModel("SOFT", -0.55, soft_deg),
    "MEDIUM": TyreModel("MEDIUM", 0.00, medium_deg),
    "HARD": TyreModel("HARD", 0.45, hard_deg),
}
inputs = SimulationInputs(
    circuit=CircuitProfile(
        race_laps=race_laps,
        pit_loss_green=float(pit_loss),
        pit_loss_sc=float(pit_loss_sc),
        overtaking_difficulty=float(overtaking),
        undercut_power=float(undercut),
        track_temperature=float(track_temp),
    ),
    driver=DriverContext(
        driver_name=selected_driver,
        team_name=(analysis_data or {}).get("team_name", ""),
        grid_position=grid,
        race_pace_delta=pace_delta,
        team_risk=0.50,
    ),
    tyres=tyres,
    tyre_sets=inventory,
    sc_probability=float(sc_prob),
    rain_probability=float(rain_prob),
    simulations=int(simulations),
    objective=objective,
)
try:
    result = run_monte_carlo(inputs)
except Exception as exc:
    st.error(f"Simulation error: {exc}")
    st.stop()

# Middle row
pred_col, chart_col, scenario_col = st.columns([1.15, 1.05, 0.90], gap="small")
with pred_col:
    with st.container(border=True):
        panel_title("Predicted strategy")
        a, b = st.columns(2)
        with a:
            st.markdown(f'''<div class="strategy-hero"><div class="strategy-label">Most likely team choice</div><div class="strategy-big">{result['predicted_team_strategy']}</div><div style="font-weight:900">{result['predicted_team_probability']:.0%} probability</div></div>''', unsafe_allow_html=True)
        with b:
            st.markdown(f'''<div class="strategy-hero strategy-opt"><div class="strategy-label">Mathematically optimal</div><div class="strategy-big">{result['optimal_strategy']}</div><div style="font-weight:900;color:#37e77b">{result['optimal_probability']:.0%} optimal</div></div>''', unsafe_allow_html=True)
        best = max(result["strategies"], key=lambda x: x["team_choice_probability"])
        metric_html([
            ("Ideal pit window", best["pit_window"], "predicted plan"),
            ("Expected finish", f"P{best['expected_finish']:.1f}", "model projection"),
            ("Podium probability", f"{best['podium_probability']:.0f}%", objective),
            ("Win probability", f"{best['win_probability']:.0f}%", objective),
        ])

with chart_col:
    with st.container(border=True):
        panel_title("Strategy comparison")
        sdf = pd.DataFrame(result["strategies"]).sort_values("expected_cost_s").head(5)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sdf["strategy"], y=sdf["expected_cost_s"],
            marker_color=["#ffd21f" if i == 0 else "#66727d" for i in range(len(sdf))],
            text=[f"{x:.1f}s" for x in sdf["expected_cost_s"]], textposition="outside",
        ))
        fig.update_layout(
            height=300, margin=dict(l=10,r=10,t=15,b=15),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dce2e8"), showlegend=False,
            yaxis=dict(title="Expected strategy cost", gridcolor="#202a33", zeroline=False),
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Lower expected strategy cost is better. Monte Carlo includes degradation, pit loss, SC and traffic shocks.")

with scenario_col:
    with st.container(border=True):
        panel_title("Scenario probabilities")
        for sc in sorted(result["scenarios"], key=lambda x: x["probability"], reverse=True):
            p = sc["probability"]
            color = "#37e77b" if "stable" in sc["scenario"].lower() else "#ffd21f" if "degradation" in sc["scenario"].lower() else "#ff1e2d" if "SC" in sc["scenario"] else "#39b8ff"
            st.markdown(
                f'<div style="display:grid;grid-template-columns:1fr 45px 65px;gap:8px;align-items:center;padding:8px 0;border-bottom:1px solid #1b252e">'
                f'<div><b style="font-size:11px">{sc["scenario"]}</b><div class="barline" style="margin-top:4px"><div style="width:{p*100:.0f}%;background:{color}"></div></div></div>'
                f'<div style="font-weight:900">{p:.0%}</div><div style="font-size:10px;color:#ffd21f;font-weight:900">{sc["recommended_strategy"]}</div></div>',
                unsafe_allow_html=True,
            )

# Bottom row
radar_col, quality_col = st.columns([1.35, 1.0], gap="small")
with radar_col:
    with st.container(border=True):
        panel_title("Strategic radar")
        degradation_level = "high" if medium_deg >= 0.12 else "medium" if medium_deg >= 0.075 else "low"
        undercut_level = "high" if undercut >= 0.75 else "medium" if undercut >= 0.55 else "low"
        sc_level = "high" if sc_prob >= 0.50 else "medium" if sc_prob >= 0.28 else "low"
        track_level = "high" if overtaking >= 0.72 else "medium" if overtaking >= 0.52 else "low"
        radar_rows = [
            ("Undercut threat", f"Power {undercut:.0%} · pit loss {pit_loss:.1f}s", undercut_level),
            ("Tyre degradation", f"Medium estimate {medium_deg:.3f}s/lap · {((analysis_data or {}).get('practice_name') or 'model prior')}", degradation_level),
            ("Neutralisation risk", f"SC/VSC prior {sc_prob:.0%} · current-weekend adjusted", sc_level),
            ("Track position importance", f"Overtaking difficulty {overtaking:.0%}", track_level),
        ]
        html_rows = "".join(
            f'<div class="radar-row"><div><div class="radar-name">{name}</div><div class="radar-sub">{sub}</div></div>{confidence_badge(level)}</div>'
            for name, sub, level in radar_rows
        )
        st.markdown(html_rows, unsafe_allow_html=True)
        st.info(result["radar_message"])

with quality_col:
    with st.container(border=True):
        panel_title("Data quality & confidence")
        rows = [
            ("Circuit / race distance", "Formula 1 official", "high" if details.get("circuit_length_km") and details.get("race_distance_km") else "medium"),
            ("Weather forecast", "Open-Meteo", "high" if weather else "low"),
            ("Tyre nomination", "Pirelli official", compound_info.get("confidence", "pending")),
            ("Grid position", "OpenF1 current weekend", "high" if (analysis_data or {}).get("grid_position") else "low"),
            ("Tyre degradation", f"OpenF1 {(analysis_data or {}).get('practice_name','practice')} long run", "high" if (analysis_data or {}).get("practice_rows",0) >= 10 else "medium" if analysis_data else "low"),
            ("Remaining tyre sets", "Current-weekend inference", "low"),
            ("SC/VSC risk", "Current circuit + weekend race control", "medium" if analysis_data else "low"),
        ]
        st.markdown("".join(source_row(*r) for r in rows), unsafe_allow_html=True)
        if analysis_data:
            st.success("Current-weekend analysis loaded. OpenF1 calls are cached and rate-limited.")
        else:
            st.caption("Press RUN / REFRESH STRATEGY once to load current-weekend timing. After that, changing driver or Grand Prix recalculates automatically using cached data whenever possible.")

st.caption(
    "Strategy Engine V1.3 · Current season only · Formula 1 official circuit/calendar data · "
    "Pirelli official compound nominations · Open-Meteo forecast · OpenF1 current-weekend analytics."
)
