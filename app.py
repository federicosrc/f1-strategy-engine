from __future__ import annotations

from datetime import datetime, timezone
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from official_sources import F1OfficialClient, PirelliCurrentSeason
from data_sources import FastF1DataClient, OpenMeteoClient
from strategy_engine_v20 import (
    CircuitProfile,
    DriverContext,
    SimulationInputs,
    TyreModel,
    estimate_degradation_from_practice,
    enumerate_legal_strategies,
    legal_next_compounds,
    scenario_probabilities,
    simulate_selected_strategy,
    total_sets,
    validate_strategy,
)

st.set_page_config(page_title="Strategy Engine", page_icon="🏁", layout="wide", initial_sidebar_state="collapsed")

CSS = r"""
<style>
:root {
  --bg:#080b0f; --panel:#0d1218; --line:#26323d; --muted:#8d9aa7;
  --white:#f6f7f9; --red:#ff1e2d; --yellow:#ffd21f; --green:#37e77b;
  --blue:#39b8ff; --orange:#ff9a33;
}
html, body, [data-testid="stAppViewContainer"] { background:var(--bg); color:var(--white); }
[data-testid="stAppViewContainer"] {
  background-image:linear-gradient(rgba(255,255,255,.015) 1px, transparent 1px),
                   linear-gradient(90deg,rgba(255,255,255,.015) 1px,transparent 1px),
                   radial-gradient(circle at 80% 0%,rgba(255,30,45,.06),transparent 28%);
  background-size:32px 32px,32px 32px,auto;
}
section[data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none!important; }
.block-container { padding-top:.7rem; padding-bottom:2rem; padding-left:1rem; padding-right:1rem; max-width:1900px; }
.se-header { display:flex; align-items:center; justify-content:space-between; gap:18px; padding:12px 18px;
  border:1px solid var(--line); border-radius:8px; background:linear-gradient(135deg,#10161d,#090d12);
  margin-bottom:9px; box-shadow:inset 4px 0 0 var(--red); }
.se-brand { font-size:28px; font-weight:900; }.se-brand span{color:var(--red)}
.se-sub { color:var(--muted); font-size:10px; letter-spacing:.15em; text-transform:uppercase; }
.se-headchips { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
.se-chip { border:1px solid var(--line); padding:7px 10px; border-radius:6px; background:#0c1117; min-width:108px; }
.se-chip .k { color:var(--muted); font-size:8px; text-transform:uppercase; letter-spacing:.12em; }
.se-chip .v { color:white; font-size:13px; font-weight:800; margin-top:2px; }
.panel-title { font-weight:900; font-size:15px; letter-spacing:.03em; text-transform:uppercase; margin-bottom:8px; }
.panel-title:before { content:""; display:inline-block; width:4px; height:15px; background:var(--red); margin-right:9px; vertical-align:-2px; }
.metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; }
.metric-card { border:1px solid var(--line); border-radius:6px; background:#0b1117; padding:9px 11px; min-height:68px; min-width:0; }
.metric-card .k { color:var(--muted); font-size:8px; text-transform:uppercase; letter-spacing:.1em; }
.metric-card .v { font-size:19px; font-weight:900; margin-top:4px; overflow-wrap:anywhere; }
.metric-card .s { color:var(--muted); font-size:9px; margin-top:2px; }
.circuit-stats { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; }
.circuit-stat { border:1px solid var(--line); background:#0b1117; border-radius:6px; padding:9px 10px; min-height:72px; }
.circuit-stat .k { color:var(--muted); font-size:8px; text-transform:uppercase; letter-spacing:.1em; }
.circuit-stat .v { color:var(--white); font-size:22px; line-height:1.05; font-weight:950; margin-top:5px; }
.circuit-stat .u { color:var(--muted); font-size:9px; margin-left:3px; font-weight:700; }
.circuit-stat .s { color:var(--muted); font-size:8px; margin-top:4px; }
.tyre-row { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
.tyre { border:1px solid var(--line); border-radius:7px; padding:10px 11px; background:#0b1016; text-align:center; }
.tyre.medium { border-color:#8a7413; }.tyre.soft { border-color:#84242d; }
.tyre .name { font-size:11px; font-weight:900; letter-spacing:.08em; }
.tyre .compound { font-size:27px; font-weight:950; margin:5px 0 2px; }
.tyre.medium .compound { color:var(--yellow); }.tyre.soft .compound{color:var(--red)}
.tyre .small { color:var(--muted); font-size:8px; text-transform:uppercase; }
.strategy-strip { display:flex; align-items:center; justify-content:center; gap:8px; flex-wrap:wrap;
  border:1px solid var(--line); background:#0a0f14; border-radius:6px; padding:9px 12px; margin:2px 0 8px; }
.strategy-pill { font-size:15px; font-weight:950; padding:4px 9px; border-radius:5px; border:1px solid var(--line); background:#111820; }
.strategy-arrow { color:var(--muted); font-weight:900; }
.rule-ok { color:var(--green); font-size:10px; font-weight:800; }.rule-bad{color:var(--orange);font-size:10px;font-weight:800}
.strategy-hero { border:1px solid var(--line); border-radius:7px; padding:14px; background:linear-gradient(135deg,#0d141b,#090d12); min-height:125px; }
.strategy-big { font-size:39px; font-weight:950; color:var(--yellow); margin:4px 0; }
.strategy-label { color:var(--muted); font-size:8px; text-transform:uppercase; letter-spacing:.14em; }
.strategy-opt { border-color:#1d8d48; box-shadow:inset 0 0 24px rgba(55,231,123,.05); }.strategy-opt .strategy-big{color:#f4f4f4}
.barline { height:8px; background:#1b252e; border-radius:2px; overflow:hidden; }.barline>div{height:100%}
.quality { display:flex; align-items:center; justify-content:space-between; gap:10px; border-bottom:1px solid #1b252e; padding:7px 0; }
.qname{font-size:11px}.qsource{color:var(--muted);font-size:9px}.badge{padding:3px 7px;border-radius:4px;font-size:9px;font-weight:900;text-transform:uppercase}
.badge.high{background:rgba(55,231,123,.12);color:var(--green);border:1px solid #237b45}.badge.medium{background:rgba(255,210,31,.10);color:var(--yellow);border:1px solid #725f12}.badge.low,.badge.pending{background:rgba(255,154,51,.10);color:var(--orange);border:1px solid #73461d}
[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line)!important; background:rgba(11,16,22,.74); border-radius:8px!important; }
.stButton>button { border-radius:5px; border:1px solid #ff3440; background:linear-gradient(180deg,#f32635,#d60f20); color:white; font-weight:900; text-transform:uppercase; min-height:42px; }
[data-baseweb="select"]>div,[data-testid="stNumberInput"] input{background:#0b1117!important}
@media(max-width:1000px){.metric-grid{grid-template-columns:repeat(2,1fr)}.circuit-stats{grid-template-columns:1fr}.se-header{display:block}.se-headchips{justify-content:flex-start;margin-top:10px}}

.final-outcome {
  display:grid;
  grid-template-columns: 1.25fr .8fr .8fr .8fr .8fr;
  gap:10px;
  border:1px solid #2b3944;
  border-left:5px solid #37e77b;
  border-radius:8px;
  background:linear-gradient(135deg,#0c1418,#081014);
  padding:14px;
  margin:8px 0 10px;
  box-shadow: inset 0 0 36px rgba(55,231,123,.035);
}
.final-main {
  min-height:126px; display:flex; flex-direction:column; justify-content:center; padding-left:8px;
}
.final-main .k { color:#8d9aa7; font-size:9px; text-transform:uppercase; letter-spacing:.16em; font-weight:800; }
.final-main .p { font-size:72px; line-height:.92; font-weight:1000; color:#f6f7f9; letter-spacing:-.06em; margin:8px 0 4px; }
.final-main .s { color:#37e77b; font-size:12px; font-weight:900; }
.outcome-mini {
  border:1px solid #26323d; border-radius:6px; background:#0b1117;
  min-height:126px; display:flex; flex-direction:column; justify-content:center; padding:11px 12px;
}
.outcome-mini .k { color:#8d9aa7; font-size:8px; text-transform:uppercase; letter-spacing:.11em; }
.outcome-mini .v { color:#f6f7f9; font-size:25px; font-weight:950; margin-top:7px; }
.outcome-mini .s { color:#8d9aa7; font-size:9px; margin-top:5px; }

.benchmark { display:grid; grid-template-columns:1fr auto 1fr; gap:10px; align-items:stretch; }
.bench-card { border:1px solid #26323d; border-radius:7px; background:#0b1117; padding:12px; }
.bench-card.best { border-color:#237b45; box-shadow:inset 0 0 22px rgba(55,231,123,.04); }
.bench-card .k { color:#8d9aa7; font-size:8px; text-transform:uppercase; letter-spacing:.12em; }
.bench-card .plan { font-size:27px; font-weight:1000; margin:6px 0 3px; }
.bench-card .cost { font-size:12px; color:#cbd3db; font-weight:800; }
.bench-vs { display:flex; flex-direction:column; align-items:center; justify-content:center; min-width:90px; padding:6px 8px; }
.bench-vs .delta { font-size:22px; font-weight:1000; color:#ffd21f; }
.bench-vs .label { color:#8d9aa7; font-size:8px; text-transform:uppercase; letter-spacing:.1em; text-align:center; }
.explain-box {
  margin-top:9px; padding:9px 11px; border:1px solid #26323d; border-radius:6px;
  background:#0a0f14; color:#aeb8c2; font-size:10px;
}
@media (max-width: 1100px) {
  .final-outcome { grid-template-columns:1fr 1fr; }
  .final-main { grid-column:1 / -1; }
  .benchmark { grid-template-columns:1fr; }
  .bench-vs { min-width:0; }
}


.projected-top {
  display:grid;
  grid-template-columns:1.25fr .8fr .8fr .8fr .8fr;
  gap:10px;
  margin:7px 0 10px;
  padding:12px 14px;
  border:1px solid #2b3944;
  border-left:6px solid #37e77b;
  border-radius:8px;
  background:linear-gradient(110deg,#0c1717,#0a1015 55%,#101116);
  box-shadow:0 8px 28px rgba(0,0,0,.16), inset 0 0 28px rgba(55,231,123,.035);
}
.projected-main { display:flex; align-items:center; gap:16px; min-height:98px; }
.projected-main .position { font-size:78px; line-height:.9; font-weight:1000; letter-spacing:-.07em; color:#f6f7f9; }
.projected-main .label { color:#8d9aa7; font-size:9px; letter-spacing:.15em; text-transform:uppercase; font-weight:900; }
.projected-main .expected { color:#37e77b; font-size:14px; font-weight:950; margin-top:5px; }
.projected-kpi { border-left:1px solid #26323d; padding:8px 12px; display:flex; flex-direction:column; justify-content:center; }
.projected-kpi .k { color:#8d9aa7; font-size:8px; text-transform:uppercase; letter-spacing:.1em; }
.projected-kpi .v { color:#f6f7f9; font-size:25px; font-weight:950; margin-top:5px; }
.projected-kpi .s { color:#8d9aa7; font-size:9px; margin-top:3px; }
@media(max-width:1100px){
  .projected-top{grid-template-columns:1fr 1fr}
  .projected-main{grid-column:1/-1}
}

.strategy-summary {
  border:1px solid #26323d;
  border-left:5px solid #ffd21f;
  border-radius:8px;
  background:linear-gradient(135deg,#10151a,#0a0e13);
  padding:16px;
  min-height:210px;
}
.strategy-summary .label { color:#8d9aa7; font-size:9px; text-transform:uppercase; letter-spacing:.15em; font-weight:900; }
.strategy-summary .sequence { color:#ffd21f; font-size:44px; line-height:1; font-weight:1000; margin:10px 0 14px; }
.strategy-details { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; }
.strategy-detail { border:1px solid #26323d; border-radius:6px; background:#0b1117; padding:10px; }
.strategy-detail .k { color:#8d9aa7; font-size:8px; text-transform:uppercase; letter-spacing:.1em; }
.strategy-detail .v { font-size:17px; font-weight:950; margin-top:4px; }

.standings { border:1px solid #26323d; border-radius:7px; overflow:hidden; background:#0a0f14; }
.stand-head,.stand-row {
  display:grid;
  grid-template-columns:44px minmax(155px,1.55fr) minmax(100px,1fr) 52px 78px 68px;
  gap:8px; align-items:center; padding:7px 10px;
}
.stand-head {
  color:#8d9aa7; font-size:8px; text-transform:uppercase; letter-spacing:.1em;
  background:#10161d; border-bottom:1px solid #26323d; font-weight:900;
}
.stand-row { font-size:10px; border-bottom:1px solid #182129; }
.stand-row:last-child { border-bottom:none; }
.stand-row.selected { background:rgba(255,210,31,.075); box-shadow:inset 4px 0 0 #ffd21f; }
.stand-pos { font-size:17px; font-weight:1000; }
.stand-driver { font-weight:900; font-size:11px; }
.stand-team { color:#8d9aa7; font-size:9px; }
.stand-num { font-weight:850; text-align:right; }
@media(max-width:1100px){
  .stand-head,.stand-row { grid-template-columns:38px minmax(120px,1.5fr) 48px 68px 58px; }
  .stand-team-col { display:none; }
}


.setup-title {
  margin:2px 0 8px;
  padding:8px 12px;
  border-left:4px solid #ff1e2d;
  background:linear-gradient(90deg,#111820,transparent);
  font-size:13px;
  font-weight:950;
  letter-spacing:.10em;
  text-transform:uppercase;
}
.setup-sub {
  display:block;
  margin-top:3px;
  color:#8d9aa7;
  font-size:9px;
  font-weight:600;
  letter-spacing:.02em;
  text-transform:none;
}
.weather-scenario {
  margin:7px 0 9px;
  display:inline-block;
  padding:4px 8px;
  border:1px solid #26323d;
  border-radius:4px;
  color:#dce2e8;
  background:#0b1117;
  font-size:9px;
  font-weight:900;
  letter-spacing:.08em;
  text-transform:uppercase;
}

.pitwall-title {
  display:flex; align-items:flex-end; justify-content:space-between; gap:20px;
  padding:13px 16px; margin:0 0 9px;
  border:1px solid #2a3540; border-left:5px solid #ff1e2d; border-radius:7px;
  background:linear-gradient(100deg,#111820,#090d12 60%,#140b0e);
}
.pitwall-title .big { font-size:19px; font-weight:1000; letter-spacing:.08em; text-transform:uppercase; }
.pitwall-title .big span { color:#ff1e2d; }
.pitwall-title .small { color:#8d9aa7; font-size:9px; letter-spacing:.08em; text-transform:uppercase; }
.section-band {
  margin:7px 0 7px; padding:6px 10px; border-bottom:1px solid #26323d;
  color:#cbd3db; font-size:10px; font-weight:950; text-transform:uppercase; letter-spacing:.14em;
}
.phase-chip {
  display:inline-flex; align-items:center; gap:6px; padding:4px 7px; margin:2px 4px 2px 0;
  border:1px solid #26323d; border-radius:4px; background:#0b1117; font-size:9px; font-weight:800;
}
.tyre-legend { display:flex; flex-wrap:wrap; gap:6px; margin:5px 0 2px; }
.tyre-legend span { padding:4px 7px; border-radius:4px; border:1px solid #26323d; font-size:9px; font-weight:950; }
@media(max-width:1100px){ .pitwall-title{display:block}.pitwall-title .small{margin-top:5px} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def clients():
    return F1OfficialClient(), PirelliCurrentSeason(), FastF1DataClient(), OpenMeteoClient()

f1, pirelli, fastf1_data, meteo = clients()
CURRENT_YEAR = datetime.now(timezone.utc).year
SIMULATION_RUNS = 30000


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
    try:
        return datetime.strptime(txt, "%d %b %H:%M").replace(year=CURRENT_YEAR, tzinfo=timezone.utc)
    except ValueError:
        return None


def current_event_index(events):
    months={"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    today=datetime.now(timezone.utc).date(); best=len(events)-1 if events else 0
    for i,e in enumerate(events):
        m=re.search(r"(\d{2}).*?([A-Z][a-z]{2})",e.get("dates",""))
        if not m: continue
        day,month=int(m.group(1)),months.get(m.group(2))
        if month and datetime(CURRENT_YEAR,month,day,tzinfo=timezone.utc).date()>=today: return i
        best=i
    return best


def panel_title(text):
    st.markdown(f'<div class="panel-title">{text}</div>', unsafe_allow_html=True)


def metric_html(items):
    cards="".join(f'<div class="metric-card"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>' for k,v,s in items)
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)


def confidence_badge(level):
    level=(level or "low").lower(); return f'<span class="badge {level}">{level}</span>'


def source_row(name,source,confidence):
    return f'<div class="quality"><div><div class="qname">{name}</div><div class="qsource">{source}</div></div>{confidence_badge(confidence)}</div>'


def load_driver_analysis(event, selected_driver):
    return fastf1_data.analyse_weekend(
        CURRENT_YEAR,
        int(event.get("round") or 1),
        selected_driver,
    )






calendar=official_calendar(); drivers=official_drivers()
if not calendar or not drivers:
    st.error("Current Formula 1 calendar or driver list could not be loaded.")
    st.stop()

def_idx=current_event_index(calendar)
default_driver_idx=drivers.index("Charles Leclerc") if "Charles Leclerc" in drivers else 0

st.markdown(
    '<div class="pitwall-title">'
    '<div><div class="big"><span>PIT WALL</span> · RACE SCENARIO ENGINE</div>'
    '<div class="small">Build the race, then test the strategy against the full simulated field</div></div>'
    '<div class="small">30,000 Monte Carlo races · current season data</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# RACE & STRATEGY
# ------------------------------------------------------------
st.markdown('<div class="section-band">01 · Race & strategy</div>', unsafe_allow_html=True)

race_cols=st.columns(
    [1.65,1.28,.70,.54,.70,.70,.62,.62],
    gap="small",
    vertical_alignment="bottom",
)

with race_cols[0]:
    event_idx=st.selectbox(
        "Grand Prix",
        range(len(calendar)),
        index=min(def_idx,len(calendar)-1),
        format_func=lambda i:f"R{calendar[i].get('round','—')} · {calendar[i].get('name','Grand Prix')}",
    )

event=calendar[event_idx]
details=official_event(event["key"],event)
race_laps=int(details.get("number_of_laps") or 57)

with race_cols[1]:
    selected_driver=st.selectbox("Driver",drivers,index=default_driver_idx)

with race_cols[3]:
    stops=st.selectbox("Stops",[1,2],index=0)

simulations=SIMULATION_RUNS
analysis_key=f"{CURRENT_YEAR}:{event['key']}:{selected_driver}"
analysis_data=st.session_state.get("analysis_data") if st.session_state.get("analysis_key")==analysis_key else None
compound_info=pirelli.compounds(event["key"])
race_dt=parse_race_datetime(details)

try:
    geo=meteo.geocode(
        details.get("location",event.get("location","")),
        details.get("country",event.get("country","")),
    )
    expected_weather=meteo.forecast_at(geo["latitude"],geo["longitude"],race_dt)
except Exception:
    expected_weather={}

expected_air=float(expected_weather.get("temperature_2m",25.0))
expected_track=float(expected_weather.get("track_temperature_estimate",expected_air+15.0))
expected_rain=float(expected_weather.get("precipitation_probability",5.0) or 0)/100.0
expected_wind=float(expected_weather.get("wind_speed_10m",0.0) or 0)
expected_humidity=float(expected_weather.get("relative_humidity_2m",55.0) or 55.0)

WEATHER_OPTIONS=["EXPECTED","DRY","HOT_DRY","COOL_DRY","CHANGEABLE","RAIN","HEAVY_RAIN"]
WEATHER_LABELS={
    "EXPECTED":"Expected conditions",
    "DRY":"Dry",
    "HOT_DRY":"Hot & dry",
    "COOL_DRY":"Cool & dry",
    "CHANGEABLE":"Changeable",
    "RAIN":"Rain",
    "HEAVY_RAIN":"Heavy rain",
}

def weather_values(mode):
    if mode=="EXPECTED":
        return {
            "air":expected_air,"track":expected_track,"rain":expected_rain,
            "wind":expected_wind,"humidity":expected_humidity,
            "source":"Open-Meteo race-time outlook",
        }
    if mode=="DRY":
        return {
            "air":expected_air,"track":expected_track,"rain":0.0,
            "wind":expected_wind,"humidity":min(expected_humidity,55.0),
            "source":"User scenario · expected temperatures",
        }
    if mode=="HOT_DRY":
        return {
            "air":max(32.0,expected_air+4.0),"track":max(48.0,expected_track+8.0),"rain":0.0,
            "wind":max(2.0,expected_wind),"humidity":min(45.0,expected_humidity),
            "source":"User scenario · high thermal stress",
        }
    if mode=="COOL_DRY":
        return {
            "air":min(20.0,expected_air-4.0),"track":min(30.0,expected_track-8.0),"rain":0.0,
            "wind":expected_wind,"humidity":max(50.0,expected_humidity),
            "source":"User scenario · low track temperature",
        }
    if mode=="CHANGEABLE":
        return {
            "air":min(expected_air,24.0),"track":min(expected_track,34.0),"rain":0.45,
            "wind":max(expected_wind,10.0),"humidity":max(expected_humidity,70.0),
            "source":"User scenario · intermittent rain",
        }
    if mode=="RAIN":
        return {
            "air":min(expected_air,21.0),"track":min(expected_track,27.0),"rain":0.90,
            "wind":max(expected_wind,12.0),"humidity":max(expected_humidity,85.0),
            "source":"User scenario · wet track",
        }
    return {
        "air":min(expected_air,19.0),"track":min(expected_track,24.0),"rain":0.98,
        "wind":max(expected_wind,15.0),"humidity":max(expected_humidity,92.0),
        "source":"User scenario · very wet track",
    }

# ------------------------------------------------------------
# SCENARIO EVOLUTION
# ------------------------------------------------------------
st.markdown('<div class="section-band">02 · Scenario evolution</div>', unsafe_allow_html=True)

scenario_cols=st.columns(
    [1.00,1.10,.66,1.10,.66,1.10,.92,.72],
    gap="small",
    vertical_alignment="bottom",
)

with scenario_cols[0]:
    weather_profile=st.selectbox(
        "Weather profile",
        ["STATIC","2_PHASES","3_PHASES"],
        format_func=lambda x:{
            "STATIC":"Static",
            "2_PHASES":"2 phases",
            "3_PHASES":"3 phases",
        }[x],
    )

with scenario_cols[1]:
    phase1=st.selectbox(
        "Phase 1",
        WEATHER_OPTIONS,
        format_func=lambda x:WEATHER_LABELS[x],
        index=0,
        key=f"phase1:{event['key']}",
    )

if weather_profile in {"2_PHASES","3_PHASES"}:
    switch1_options=list(range(4,max(5,race_laps-5)))
    default_switch1=max(4,min(race_laps-6,int(round(race_laps*.38))))
    with scenario_cols[2]:
        switch1=st.selectbox(
            "Phase 2 starts",
            switch1_options,
            index=switch1_options.index(default_switch1) if default_switch1 in switch1_options else 0,
            format_func=lambda x:f"L{x}",
            key=f"switch1:{event['key']}:{weather_profile}",
        )
    with scenario_cols[3]:
        phase2=st.selectbox(
            "Phase 2",
            WEATHER_OPTIONS,
            format_func=lambda x:WEATHER_LABELS[x],
            index=5 if weather_profile!="STATIC" else 0,
            key=f"phase2:{event['key']}:{weather_profile}",
        )
else:
    switch1=None
    phase2=None
    with scenario_cols[2]:
        st.selectbox("Phase 2 starts",["—"],disabled=True)
    with scenario_cols[3]:
        st.selectbox("Phase 2",["—"],disabled=True)

if weather_profile=="3_PHASES":
    switch2_options=list(range(switch1+4,max(switch1+5,race_laps-2)))
    default_switch2=max(switch1+4,min(race_laps-2,int(round(race_laps*.70))))
    with scenario_cols[4]:
        switch2=st.selectbox(
            "Phase 3 starts",
            switch2_options,
            index=switch2_options.index(default_switch2) if default_switch2 in switch2_options else 0,
            format_func=lambda x:f"L{x}",
            key=f"switch2:{event['key']}:{switch1}",
        )
    with scenario_cols[5]:
        phase3=st.selectbox(
            "Phase 3",
            WEATHER_OPTIONS,
            format_func=lambda x:WEATHER_LABELS[x],
            index=1,
            key=f"phase3:{event['key']}:{switch1}",
        )
else:
    switch2=None
    phase3=None
    with scenario_cols[4]:
        st.selectbox("Phase 3 starts",["—"],disabled=True)
    with scenario_cols[5]:
        st.selectbox("Phase 3",["—"],disabled=True)

with scenario_cols[6]:
    neutralisation_mode=st.selectbox(
        "Race control",
        ["NONE","SC","VSC"],
        format_func=lambda x:{
            "NONE":"No SC / VSC",
            "SC":"Safety Car",
            "VSC":"Virtual Safety Car",
        }[x],
    )

if neutralisation_mode in {"SC","VSC"}:
    neutral_options=["RANDOM"]+list(range(2,max(3,race_laps-1)))
    with scenario_cols[7]:
        neutral_lap_choice=st.selectbox(
            "Event lap",
            neutral_options,
            format_func=lambda x:"Random" if x=="RANDOM" else f"L{x}",
            key=f"neutral:{event['key']}:{neutralisation_mode}",
        )
    neutralisation_lap=None if neutral_lap_choice=="RANDOM" else int(neutral_lap_choice)
else:
    neutralisation_lap=None
    with scenario_cols[7]:
        st.selectbox("Event lap",["—"],disabled=True)

# Build weather timeline.
def make_phase(start_lap,end_lap,mode):
    vals=weather_values(mode)
    return {
        "start_lap":int(start_lap),
        "end_lap":int(end_lap),
        "mode":mode,
        "track_temp":float(vals["track"]),
        "rain_probability":float(vals["rain"]),
        "air_temp":float(vals["air"]),
        "wind":float(vals["wind"]),
        "humidity":float(vals["humidity"]),
        "source":vals["source"],
    }

if weather_profile=="STATIC":
    weather_timeline=[make_phase(1,race_laps,phase1)]
elif weather_profile=="2_PHASES":
    weather_timeline=[
        make_phase(1,switch1-1,phase1),
        make_phase(switch1,race_laps,phase2),
    ]
else:
    weather_timeline=[
        make_phase(1,switch1-1,phase1),
        make_phase(switch1,switch2-1,phase2),
        make_phase(switch2,race_laps,phase3),
    ]

weighted_laps=sum(p["end_lap"]-p["start_lap"]+1 for p in weather_timeline)
track_temp=sum((p["end_lap"]-p["start_lap"]+1)*p["track_temp"] for p in weather_timeline)/weighted_laps
air_temp=sum((p["end_lap"]-p["start_lap"]+1)*p["air_temp"] for p in weather_timeline)/weighted_laps
rain_prob=sum((p["end_lap"]-p["start_lap"]+1)*p["rain_probability"] for p in weather_timeline)/weighted_laps
wind=sum((p["end_lap"]-p["start_lap"]+1)*p["wind"] for p in weather_timeline)/weighted_laps
humidity=sum((p["end_lap"]-p["start_lap"]+1)*p["humidity"] for p in weather_timeline)/weighted_laps

wet_tyres_enabled=any(
    p["mode"] in {"CHANGEABLE","RAIN","HEAVY_RAIN"}
    or (p["mode"]=="EXPECTED" and p["rain_probability"]>=.20)
    for p in weather_timeline
)
available_compounds=["SOFT","MEDIUM","HARD"]+(
    ["INTERMEDIATE","WET"] if wet_tyres_enabled else []
)

phase1_mode=weather_timeline[0]["mode"]
default_start=(
    "WET" if phase1_mode=="HEAVY_RAIN"
    else "INTERMEDIATE" if phase1_mode=="RAIN"
    else "MEDIUM"
)
if default_start not in available_compounds:
    default_start="MEDIUM"

with race_cols[2]:
    start_compound=st.selectbox(
        "Start tyre",
        available_compounds,
        index=available_compounds.index(default_start),
        key=f"start:{event['key']}:{selected_driver}:{weather_profile}:{phase1_mode}",
    )

circuit_type=details.get("circuit_type","Permanent")
if circuit_type=="Street":
    overtaking,sc_prob,pit_loss=.82,.48,23.0
elif circuit_type=="Semi-permanent":
    overtaking,sc_prob,pit_loss=.70,.40,23.5
else:
    overtaking,sc_prob,pit_loss=.56,.31,22.0

grid=int((analysis_data or {}).get("grid_position") or 10)
pace_delta=float((analysis_data or {}).get("pace_delta") or 0.0)
incident=float((analysis_data or {}).get("incident_risk") or sc_prob)
sc_prob=min(.75,max(.12,.55*sc_prob+.45*incident))

deg=(analysis_data or {}).get("degradation",{})
soft_deg=float(deg.get("SOFT",.12))
medium_deg=float(deg.get("MEDIUM",.08))
hard_deg=float(deg.get("HARD",.055))
undercut=min(.95,.48+overtaking*.35+max(0,medium_deg-.06)*.7)

inventory=(analysis_data or {}).get("inventory") or {
    "SOFT":{"new":1,"used":1},
    "MEDIUM":{"new":1,"used":1},
    "HARD":{"new":1,"used":1},
}
inventory.setdefault("INTERMEDIATE",{"new":4,"used":0})
inventory.setdefault("WET",{"new":3,"used":0})

tyres={
    "SOFT":TyreModel("SOFT",-.55,soft_deg),
    "MEDIUM":TyreModel("MEDIUM",0.0,medium_deg),
    "HARD":TyreModel("HARD",.45,hard_deg),
    "INTERMEDIATE":TyreModel("INTERMEDIATE",.05,.035),
    "WET":TyreModel("WET",.25,.022),
}

provisional_inputs=SimulationInputs(
    CircuitProfile(race_laps,pit_loss,pit_loss*.55,overtaking,undercut,track_temp),
    DriverContext(selected_driver,(analysis_data or {}).get("team_name",""),grid,pace_delta),
    tyres,inventory,sc_prob,rain_prob,simulations,
    mandatory_race_compounds=("HARD","MEDIUM"),
    rivals=(analysis_data or {}).get("grid_model") or None,
    neutralisation_mode=neutralisation_mode,
    allowed_compounds=tuple(available_compounds),
    weather_mode=phase1_mode,
    weather_timeline=weather_timeline,
    neutralisation_lap=neutralisation_lap,
)

s2_options=legal_next_compounds(
    [start_compound],stops+1,provisional_inputs
) or available_compounds

with race_cols[4]:
    stint2=st.selectbox(
        "Stint 2",
        s2_options,
        key=f"s2:{analysis_key}:{weather_profile}:{start_compound}:{stops}:{switch1}",
    )

if stops==2:
    s3_options=legal_next_compounds(
        [start_compound,stint2],3,provisional_inputs
    ) or available_compounds
    with race_cols[5]:
        stint3=st.selectbox(
            "Stint 3",
            s3_options,
            key=f"s3:{analysis_key}:{weather_profile}:{start_compound}:{stint2}:{switch2}",
        )
else:
    stint3=None
    with race_cols[5]:
        st.selectbox("Stint 3",["—"],disabled=True)

pit1_max=max(4,race_laps-6 if stops==2 else race_laps-2)
pit1_default=max(3,min(pit1_max,int(round(race_laps*.38))))
pit1_options=list(range(3,pit1_max+1))
with race_cols[6]:
    pit_lap_1=st.selectbox(
        "Pit lap 1",
        pit1_options,
        index=pit1_options.index(pit1_default) if pit1_default in pit1_options else 0,
        format_func=lambda x:f"L{x}",
        key=f"pit1:{analysis_key}:{stops}",
    )

if stops==2:
    pit2_min=pit_lap_1+3
    pit2_max=max(pit2_min,race_laps-2)
    pit2_options=list(range(pit2_min,pit2_max+1))
    pit2_default=max(pit2_min,min(pit2_max,int(round(race_laps*.70))))
    with race_cols[7]:
        pit_lap_2=st.selectbox(
            "Pit lap 2",
            pit2_options,
            index=pit2_options.index(pit2_default) if pit2_default in pit2_options else 0,
            format_func=lambda x:f"L{x}",
            key=f"pit2:{analysis_key}:{pit_lap_1}",
        )
else:
    pit_lap_2=None
    with race_cols[7]:
        st.selectbox("Pit lap 2",["—"],disabled=True)

selected_compounds=[start_compound,stint2]+([stint3] if stops==2 else [])
selected_pit_laps=[pit_lap_1]+([pit_lap_2] if stops==2 else [])

button_cols=st.columns([3.0,1.6,1.6,3.0],gap="small")
with button_cols[1]:
    simulate_clicked=st.button("Simulate my strategy",use_container_width=True)
with button_cols[2]:
    optimal_clicked=st.button("Find optimal strategy",use_container_width=True)

short_name={"SOFT":"S","MEDIUM":"M","HARD":"H","INTERMEDIATE":"I","WET":"W"}
neutral_label={"NONE":"NO SC/VSC","SC":"SAFETY CAR","VSC":"VIRTUAL SAFETY CAR"}[neutralisation_mode]
valid_now,rule_reasons=validate_strategy(selected_compounds,provisional_inputs)

# Compact scenario strip.
phase_text=" · ".join(
    f'L{p["start_lap"]}–{p["end_lap"]} {WEATHER_LABELS[p["mode"]]}'
    for p in weather_timeline
)
st.markdown(
    f'<div class="strategy-strip">'
    f'<span style="font-size:10px;font-weight:900;color:#8d9aa7">SCENARIO</span>'
    f'<span style="font-size:10px;font-weight:900">{phase_text}</span>'
    f'<span style="font-size:10px;font-weight:900;color:#8d9aa7">'
    f'{" · ".join("PIT L"+str(x) for x in selected_pit_laps)} · {neutral_label}'
    f'{" RANDOM" if neutralisation_mode!="NONE" and neutralisation_lap is None else (" L"+str(neutralisation_lap) if neutralisation_lap else "")}'
    f'</span>'
    f'<span class="{"rule-ok" if valid_now else "rule-bad"}">'
    f'{"LEGAL STRATEGY" if valid_now else "CHECK STRATEGY"}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# SIMULATION
# ------------------------------------------------------------
if simulate_clicked or optimal_clicked:
    with st.spinner(f"Building current-weekend model for {selected_driver}…"):
        try:
            analysis_data=load_driver_analysis(event,selected_driver)
        except Exception:
            analysis_data={
                "available":False,"degradation":{},"pace_delta":0.0,
                "inventory":inventory,"grid_position":None,
                "practice_rows":0,"practice_name":"Fallback","grid_model":[],
            }
        st.session_state["analysis_data"]=analysis_data
        st.session_state["analysis_key"]=analysis_key

    grid=int((analysis_data or {}).get("grid_position") or 10)
    pace_delta=float((analysis_data or {}).get("pace_delta") or 0.0)
    incident=float((analysis_data or {}).get("incident_risk") or sc_prob)
    sc_prob=min(.75,max(.12,.55*sc_prob+.45*incident))

    deg=(analysis_data or {}).get("degradation",{})
    soft_deg=float(deg.get("SOFT",soft_deg))
    medium_deg=float(deg.get("MEDIUM",medium_deg))
    hard_deg=float(deg.get("HARD",hard_deg))
    loaded_inventory=(analysis_data or {}).get("inventory") or {}
    inventory.update(loaded_inventory)
    inventory.setdefault("INTERMEDIATE",{"new":4,"used":0})
    inventory.setdefault("WET",{"new":3,"used":0})

    tyres={
        "SOFT":TyreModel("SOFT",-.55,soft_deg),
        "MEDIUM":TyreModel("MEDIUM",0.0,medium_deg),
        "HARD":TyreModel("HARD",.45,hard_deg),
        "INTERMEDIATE":TyreModel("INTERMEDIATE",.05,.035),
        "WET":TyreModel("WET",.25,.022),
    }

    final_inputs=SimulationInputs(
        CircuitProfile(race_laps,pit_loss,pit_loss*.55,overtaking,undercut,track_temp),
        DriverContext(selected_driver,(analysis_data or {}).get("team_name",""),grid,pace_delta),
        tyres,inventory,sc_prob,rain_prob,simulations,
        mandatory_race_compounds=("HARD","MEDIUM"),
        rivals=(analysis_data or {}).get("grid_model") or None,
        neutralisation_mode=neutralisation_mode,
        allowed_compounds=tuple(available_compounds),
        weather_mode=phase1_mode,
        weather_timeline=weather_timeline,
        neutralisation_lap=neutralisation_lap,
    )

    valid,reasons=validate_strategy(selected_compounds,final_inputs)
    anchor=selected_compounds

    if not valid:
        legal=enumerate_legal_strategies(
            final_inputs,start_compound=start_compound,stops=(1,2)
        )
        if simulate_clicked:
            st.error("Selected strategy is not feasible: "+" ".join(reasons))
            st.session_state.pop("strategy_result",None)
            legal=[]
        if optimal_clicked and legal:
            anchor=legal[0]

    if valid or (optimal_clicked and anchor):
        try:
            result=simulate_selected_strategy(
                final_inputs,
                anchor,
                compare_same_start=True,
                pit_laps_override=selected_pit_laps if anchor==selected_compounds else None,
            )
            timeline_key="|".join(
                f'{p["start_lap"]}-{p["end_lap"]}-{p["mode"]}' for p in weather_timeline
            )
            result_key=(
                f"{analysis_key}:{'-'.join(selected_compounds)}:{timeline_key}:"
                f"{neutralisation_mode}:{neutralisation_lap}:"
                f"{'-'.join(str(x) for x in selected_pit_laps)}"
            )
            st.session_state["strategy_result"]=result
            st.session_state["strategy_result_key"]=result_key
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.pop("strategy_result",None)

timeline_key="|".join(
    f'{p["start_lap"]}-{p["end_lap"]}-{p["mode"]}' for p in weather_timeline
)
result_key_now=(
    f"{analysis_key}:{'-'.join(selected_compounds)}:{timeline_key}:"
    f"{neutralisation_mode}:{neutralisation_lap}:"
    f"{'-'.join(str(x) for x in selected_pit_laps)}"
)
result=st.session_state.get("strategy_result")
if st.session_state.get("strategy_result_key")!=result_key_now:
    result=None

# ------------------------------------------------------------
# OUTCOME HERO
# ------------------------------------------------------------
if result is not None:
    projected_position=int(result.get("projected_finish_position") or result["most_likely_finish"])
    model_conf=result.get("race_model_confidence","low").upper()
    competitors=result.get("competitors_modelled",0)
    st.markdown(
        f"""
        <div class="projected-top">
          <div class="projected-main">
            <div>
              <div class="label">Estimated final position</div>
              <div class="position">P{projected_position}</div>
            </div>
            <div>
              <div class="label">Monte Carlo expected finish</div>
              <div class="expected">P{result["expected_finish"]:.1f}</div>
              <div style="color:#8d9aa7;font-size:9px;margin-top:4px">
                {competitors} rivals · {model_conf} confidence
              </div>
            </div>
          </div>
          <div class="projected-kpi"><div class="k">Points</div><div class="v">{result["points_probability"]:.0%}</div><div class="s">P10 or better</div></div>
          <div class="projected-kpi"><div class="k">Top 5</div><div class="v">{result["top5_probability"]:.0%}</div><div class="s">P5 or better</div></div>
          <div class="projected-kpi"><div class="k">Podium</div><div class="v">{result["podium_probability"]:.0%}</div><div class="s">P1–P3</div></div>
          <div class="projected-kpi"><div class="k">Win</div><div class="v">{result["win_probability"]:.0%}</div><div class="s">P1</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# SCENARIO TIMELINE
# ------------------------------------------------------------
with st.container(border=True):
    panel_title("Race scenario timeline")

    TYRE_COLORS={
        "SOFT":"#ff1e2d","MEDIUM":"#ffd21f","HARD":"#e8edf2",
        "INTERMEDIATE":"#37e77b","WET":"#39b8ff",
    }
    WEATHER_COLORS={
        "EXPECTED":"#6b7680","DRY":"#a8b0b7","HOT_DRY":"#ff8d38",
        "COOL_DRY":"#67c7ef","CHANGEABLE":"#a06cd5",
        "RAIN":"#3988d1","HEAVY_RAIN":"#155087",
    }

    timeline_fig=go.Figure()

    for phase in weather_timeline:
        duration=phase["end_lap"]-phase["start_lap"]+1
        timeline_fig.add_trace(
            go.Bar(
                y=["Weather"],
                x=[duration],
                base=[phase["start_lap"]-1],
                orientation="h",
                marker_color=WEATHER_COLORS.get(phase["mode"],"#6b7680"),
                text=[WEATHER_LABELS[phase["mode"]]],
                textposition="inside",
                hovertemplate=(
                    f'{WEATHER_LABELS[phase["mode"]]}<br>'
                    f'L{phase["start_lap"]}–L{phase["end_lap"]}<br>'
                    f'Track {phase["track_temp"]:.0f}°C<extra></extra>'
                ),
                showlegend=False,
            )
        )

    bounds=[1]+[x+1 for x in selected_pit_laps]+[race_laps+1]
    for i,comp in enumerate(selected_compounds):
        start_lap=bounds[i]
        end_lap=bounds[i+1]-1
        timeline_fig.add_trace(
            go.Bar(
                y=["Tyres"],
                x=[end_lap-start_lap+1],
                base=[start_lap-1],
                orientation="h",
                marker_color=TYRE_COLORS.get(comp,"#8d9aa7"),
                text=[short_name.get(comp,comp[0])],
                textposition="inside",
                hovertemplate=f'{comp.title()} · L{start_lap}–L{end_lap}<extra></extra>',
                showlegend=False,
            )
        )

    for pit in selected_pit_laps:
        timeline_fig.add_vline(
            x=pit,
            line_width=1.4,
            line_dash="dot",
            line_color="#ffd21f",
            annotation_text=f"PIT L{pit}",
            annotation_position="top",
        )

    if neutralisation_mode in {"SC","VSC"} and neutralisation_lap is not None:
        duration=4 if neutralisation_mode=="SC" else 2
        timeline_fig.add_vrect(
            x0=neutralisation_lap,
            x1=min(race_laps,neutralisation_lap+duration),
            fillcolor="#ffbf3f",
            opacity=.12,
            line_width=0,
        )
        timeline_fig.add_vline(
            x=neutralisation_lap,
            line_width=2,
            line_color="#ffbf3f",
            annotation_text=f"{neutralisation_mode} L{neutralisation_lap}",
            annotation_position="top right",
        )

    timeline_fig.update_layout(
        height=245,
        barmode="overlay",
        bargap=.28,
        margin=dict(l=70,r=20,t=35,b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce2e8"),
        xaxis=dict(
            title="Race lap",
            range=[0,race_laps],
            dtick=max(5,int(round(race_laps/10))),
            gridcolor="#202a33",
            zeroline=False,
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=["Weather","Tyres"],
            autorange="reversed",
            gridcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
    )
    st.plotly_chart(timeline_fig,use_container_width=True,config={"displayModeBar":False})

    tyre_legend='<div class="tyre-legend">'+''.join(
        f'<span style="color:{TYRE_COLORS[c]}">{short_name[c]} · {c.title()}</span>'
        for c in available_compounds
    )+'</div>'
    st.markdown(tyre_legend,unsafe_allow_html=True)
    if neutralisation_mode in {"SC","VSC"} and neutralisation_lap is None:
        st.caption(f"{neutralisation_mode} timing is RANDOM in each simulated race.")

# ------------------------------------------------------------
# CORE DASHBOARD
# ------------------------------------------------------------
st.markdown(
    f"""<div class="se-header">
      <div>
        <div class="se-brand"><span>STRATEGY</span> ENGINE</div>
        <div class="se-sub">Pit-wall race scenario simulator</div>
      </div>
      <div class="se-headchips">
        <div class="se-chip"><div class="k">Grand Prix</div><div class="v">{details.get('name',event.get('name'))}</div></div>
        <div class="se-chip"><div class="k">Driver</div><div class="v">{selected_driver}</div></div>
        <div class="se-chip"><div class="k">Strategy</div><div class="v">{' → '.join(short_name.get(c,c[0]) for c in selected_compounds)}</div></div>
        <div class="se-chip"><div class="k">Weather phases</div><div class="v">{len(weather_timeline)}</div></div>
        <div class="se-chip"><div class="k">Race control</div><div class="v">{neutral_label}</div></div>
      </div>
    </div>""",
    unsafe_allow_html=True,
)

left_col,right_col=st.columns([1.06,1.06],gap="small")

with left_col:
    with st.container(border=True):
        panel_title("Circuit")
        if details.get("track_image_url"):
            st.image(details["track_image_url"],use_container_width=True)
        length=details.get("circuit_length_km")
        dist=details.get("race_distance_km")
        st.markdown(
            f'<div class="circuit-stats">'
            f'<div class="circuit-stat"><div class="k">Laps</div><div class="v">{race_laps}</div></div>'
            f'<div class="circuit-stat"><div class="k">Circuit</div><div class="v">{f"{length:.3f}" if isinstance(length,(float,int)) else "—"}<span class="u">km</span></div></div>'
            f'<div class="circuit-stat"><div class="k">Race</div><div class="v">{f"{dist:.3f}" if isinstance(dist,(float,int)) else "—"}<span class="u">km</span></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        panel_title("Weather evolution")
        for phase in weather_timeline:
            st.markdown(
                f'<span class="phase-chip">'
                f'L{phase["start_lap"]}–L{phase["end_lap"]} · {WEATHER_LABELS[phase["mode"]]} · '
                f'{phase["track_temp"]:.0f}°C track'
                f'</span>',
                unsafe_allow_html=True,
            )
        metric_html([
            ("Avg air",f"{air_temp:.0f}°C","weighted scenario"),
            ("Avg track",f"{track_temp:.0f}°C","weighted scenario"),
            ("Wet exposure",f"{rain_prob:.0%}","lap-weighted"),
            ("Wind",f"{wind:.0f} km/h",f"humidity {humidity:.0f}%"),
        ])

with right_col:
    with st.container(border=True):
        panel_title("Your strategy")
        strategy_sequence=" → ".join(short_name.get(c,c[0]) for c in selected_compounds)
        pit_text=" / ".join(f"L{x}" for x in selected_pit_laps)
        st.markdown(
            f"""<div class="strategy-summary">
              <div class="label">Selected race plan</div>
              <div class="sequence">{strategy_sequence}</div>
              <div class="strategy-details">
                <div class="strategy-detail"><div class="k">Driver</div><div class="v">{selected_driver}</div></div>
                <div class="strategy-detail"><div class="k">Pit laps</div><div class="v">{pit_text}</div></div>
                <div class="strategy-detail"><div class="k">Weather changes</div><div class="v">{len(weather_timeline)-1}</div></div>
                <div class="strategy-detail"><div class="k">Race control</div><div class="v">{neutral_label}</div></div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            f'Slick nomination: H {compound_info["hard"]} · M {compound_info["medium"]} · S {compound_info["soft"]}'
            + (" · I/W enabled" if wet_tyres_enabled else "")
        )

        if result is not None:
            metric_html([
                ("Grid",f"P{grid}","qualifying / fallback"),
                ("Projected",f'P{result["projected_finish_position"]}',"final classification"),
                ("Expected",f'P{result["expected_finish"]:.1f}',"Monte Carlo mean"),
                ("Strategy cost",f'{result["expected_cost_s"]:.1f}s',"lower is better"),
            ])

            panel_title("Finish-position distribution")
            ddf=pd.DataFrame([
                {"Position":f"P{k}","Probability":v*100}
                for k,v in result["finish_distribution"].items()
            ])
            fig2=go.Figure(
                go.Bar(
                    x=ddf["Position"],y=ddf["Probability"],
                    marker_color=[
                        "#37e77b" if int(p[1:])<=3
                        else "#ffd21f" if int(p[1:])<=10
                        else "#66727d" for p in ddf["Position"]
                    ],
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
                )
            )
            fig2.update_layout(
                height=270,margin=dict(l=8,r=8,t=10,b=10),
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#dce2e8"),showlegend=False,
                yaxis=dict(title="Probability %",gridcolor="#202a33",zeroline=False),
                xaxis=dict(title=""),
            )
            st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})

if result is None:
    with st.container(border=True):
        panel_title("Ready to simulate")
        st.info(
            "The timeline above is your race scenario. Press SIMULATE MY STRATEGY to calculate "
            "the full-grid outcome, or FIND OPTIMAL STRATEGY to benchmark the plan."
        )
else:
    optimal=result["optimal"]
    delta=float(result["delta_to_optimal_s"])
    bench_col,scenario_col=st.columns([1.28,.92],gap="small")

    with bench_col:
        with st.container(border=True):
            panel_title("Strategy benchmark")
            st.markdown(
                f"""<div class="benchmark">
                  <div class="bench-card">
                    <div class="k">Your plan</div>
                    <div class="plan">{result["strategy"]}</div>
                    <div class="cost">Pit: {result["pit_window"]}</div>
                    <div class="cost">Cost: {result["expected_cost_s"]:.1f}s</div>
                  </div>
                  <div class="bench-vs">
                    <div class="delta">+{max(0.0,delta):.1f}s</div>
                    <div class="label">vs best<br>lower is better</div>
                  </div>
                  <div class="bench-card best">
                    <div class="k">Best same-start plan</div>
                    <div class="plan">{optimal["strategy"]}</div>
                    <div class="cost">Pit window: {optimal["pit_window"]}</div>
                    <div class="cost">Cost: {optimal["expected_cost_s"]:.1f}s</div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

    with scenario_col:
        with st.container(border=True):
            panel_title("Scenario summary")
            st.markdown(
                f"""
                **Weather phases:** {len(weather_timeline)}  
                **Wet exposure:** {rain_prob:.0%}  
                **Average track temp:** {track_temp:.0f}°C  
                **Race control:** {neutral_label}  
                **Event timing:** {"Random" if neutralisation_mode!="NONE" and neutralisation_lap is None else ("L"+str(neutralisation_lap) if neutralisation_lap else "None")}  
                **Pit laps:** {' / '.join('L'+str(x) for x in selected_pit_laps)}
                """
            )

    with st.container(border=True):
        panel_title("Estimated final classification")
        standings=result.get("estimated_classification",[])
        rows_html=[]
        for row in standings:
            selected_class=" selected" if row.get("selected_driver") else ""
            team=row.get("team_name") or "—"
            grid_value=f'P{row["grid_position"]}' if row.get("grid_position") else "—"
            rows_html.append(
                f'<div class="stand-row{selected_class}">'
                f'<div class="stand-pos">P{row["projected_position"]}</div>'
                f'<div><div class="stand-driver">{row["driver_name"]}</div></div>'
                f'<div class="stand-team stand-team-col">{team}</div>'
                f'<div class="stand-num">{grid_value}</div>'
                f'<div class="stand-num">P{row["expected_finish"]:.1f}</div>'
                f'<div class="stand-num">{row["podium_probability"]:.0%}</div>'
                f'</div>'
            )
        st.markdown(
            '<div class="standings">'
            '<div class="stand-head">'
            '<div>Pos</div><div>Driver</div><div class="stand-team-col">Team</div>'
            '<div style="text-align:right">Grid</div>'
            '<div style="text-align:right">Avg</div>'
            '<div style="text-align:right">Podium</div>'
            '</div>'+''.join(rows_html)+'</div>',
            unsafe_allow_html=True,
        )

with st.expander("Low-confidence overrides",expanded=False):
    st.caption("Correct pit-lane loss or tyre availability if you have better official data.")
    cols=st.columns(6,gap="small")
    with cols[0]:
        pit_loss=st.number_input(
            "Pit loss (s)",10.0,40.0,float(pit_loss),0.1,key=f"{event['key']}:pit"
        )
    for comp,col in zip(("SOFT","MEDIUM","HARD","INTERMEDIATE","WET"),cols[1:]):
        with col:
            st.markdown(f"**{comp.title()}**")
            x,y=st.columns(2)
            base=f"{analysis_key}:{comp}"
            inventory[comp]["new"]=x.number_input("N",0,6,int(inventory[comp].get("new",0)),key=base+":n")
            inventory[comp]["used"]=y.number_input("U",0,6,int(inventory[comp].get("used",0)),key=base+":u")

st.caption(
    "Strategy Engine V2.0 · Multi-phase weather · exact/random SC/VSC timing · "
    "full-grid Monte Carlo · pit-wall timeline."
)
