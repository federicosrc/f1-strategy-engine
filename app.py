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
:root{
  --bg:#05090d; --panel:#081019; --panel2:#0a131d; --panel3:#0d1721;
  --line:#1f3442; --line2:#294655; --muted:#8d9ba7; --white:#f4f7fa;
  --red:#ff2638; --red2:#d81022; --yellow:#ffd21f; --green:#2ed47a;
  --blue:#2d9cff; --wet:#0c5da5; --orange:#ff9c3a; --cyan:#39c8ff;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--white)}
[data-testid="stAppViewContainer"]{
  background:
    linear-gradient(rgba(80,115,140,.025) 1px,transparent 1px),
    linear-gradient(90deg,rgba(80,115,140,.025) 1px,transparent 1px),
    radial-gradient(circle at 85% -10%,rgba(255,38,56,.06),transparent 32%);
  background-size:38px 38px,38px 38px,auto;
}
section[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important}
.block-container{padding:.55rem .85rem 2rem;max-width:1920px}

[data-testid="stVerticalBlockBorderWrapper"]{
  border-color:var(--line)!important;
  background:linear-gradient(145deg,rgba(7,15,22,.94),rgba(5,11,16,.92))!important;
  border-radius:7px!important;
}
[data-baseweb="select"]>div,[data-testid="stNumberInput"] input{
  background:#07111a!important;border-color:#223b4b!important;
}
[data-testid="stSelectbox"] label,[data-testid="stNumberInput"] label{
  color:#a9b6c1!important;font-size:9px!important;font-weight:850!important;
  text-transform:uppercase!important;letter-spacing:.08em!important;
}
.stButton>button{
  border-radius:5px;border:1px solid #314b5b;background:#0a141d;color:#dce5eb;
  min-height:36px;font-size:10px;font-weight:900;text-transform:uppercase;
}
.stButton>button:hover{border-color:#ff3344;color:white}
button[kind="primary"]{
  background:linear-gradient(180deg,#ff2638,#cf0d1f)!important;border-color:#ff4050!important;
}
hr{border-color:#1b2d39!important}

.topbar{
  display:grid;grid-template-columns:1.3fr 1.1fr .8fr;align-items:center;
  min-height:58px;padding:6px 2px 9px;margin-bottom:4px;border-bottom:1px solid var(--line);
}
.brand-title{font-size:25px;font-weight:1000;font-style:italic;letter-spacing:.02em}
.brand-title span{color:var(--red)}
.brand-sub{color:#91a1ad;font-size:9px;letter-spacing:.14em;text-transform:uppercase;margin-top:-3px}
.gp-head{display:flex;gap:13px;align-items:center;border-left:1px solid var(--line);padding-left:18px}
.round-box{color:#aab7c2;font-size:9px;font-weight:850;text-transform:uppercase;letter-spacing:.1em}
.gp-name{font-size:17px;font-weight:950;letter-spacing:.025em;text-transform:uppercase}
.gp-place{font-size:9px;color:#a3b1bc;margin-top:1px}
.head-meta{text-align:right;color:#82939f;font-size:9px;line-height:1.5}
.head-meta b{color:#d9e2e8}

.setup-shell{
  border:1px solid var(--line);border-radius:7px;background:rgba(7,14,20,.9);
  margin:4px 0 8px;padding:8px 12px 2px;
}
.setup-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:2px}
.setup-title{font-size:14px;font-weight:1000;text-transform:uppercase;letter-spacing:.04em}
.setup-caption{font-size:9px;color:#8c9aa5;margin-left:12px;font-weight:600;text-transform:none;letter-spacing:0}

.panel-title{
  font-size:13px;font-weight:1000;text-transform:uppercase;letter-spacing:.045em;margin-bottom:7px;
}
.panel-title:before{content:"";display:inline-block;width:3px;height:13px;background:var(--red);margin-right:8px;vertical-align:-1px}

.result-shell{
  display:grid;grid-template-columns:1.35fr .95fr;gap:9px;margin:7px 0 9px;
}
.result-card,.compare-card{
  border:1px solid var(--line);border-radius:7px;background:linear-gradient(135deg,#08121b,#050b10);
  padding:10px 13px;min-height:125px;
}
.result-title,.compare-title{font-size:13px;font-weight:1000;text-transform:uppercase;letter-spacing:.04em}
.result-flex{display:grid;grid-template-columns:180px repeat(5,1fr);align-items:center;margin-top:8px}
.result-pos{
  font-size:87px;font-weight:1000;font-style:italic;letter-spacing:-.08em;line-height:.85;
  background:linear-gradient(180deg,#ff505c,#ef1d31);-webkit-background-clip:text;color:transparent;
}
.result-kpi{border-left:1px solid #263946;padding:5px 13px;min-height:62px;display:flex;flex-direction:column;justify-content:center}
.result-kpi .k{font-size:9px;color:#91a0ab}
.result-kpi .v{font-size:25px;font-weight:1000;margin-top:2px}
.result-kpi .s{font-size:8px;color:#80909b;margin-top:2px}

.compare-grid{display:grid;grid-template-columns:1fr auto 1fr;gap:11px;align-items:center;margin-top:18px}
.compare-side .k{font-size:9px;color:#94a4af}
.compare-side .big{font-size:27px;font-weight:1000;margin-top:3px}
.compare-side .small{font-size:9px;color:#82929e;margin-top:3px}
.compare-delta{text-align:center;border-left:1px solid #263946;border-right:1px solid #263946;padding:6px 16px}
.compare-delta .v{font-size:19px;font-weight:1000;color:var(--green)}
.compare-delta .k{font-size:8px;color:#84949f;text-transform:uppercase}

.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px}
.metric-card{border:1px solid #1d3340;border-radius:5px;background:#07111a;padding:8px 9px;min-height:61px}
.metric-card .k{color:#81929e;font-size:8px;text-transform:uppercase;letter-spacing:.09em}
.metric-card .v{font-size:18px;font-weight:1000;margin-top:3px}
.metric-card .s{font-size:8px;color:#778995;margin-top:2px}

.timeline-note{font-size:8px;color:#8798a4;margin-left:8px;font-weight:500;text-transform:none}
.circuit-layout{display:grid;grid-template-columns:1.3fr .65fr;gap:9px;align-items:center}
.circuit-kpis{display:grid;gap:7px}
.circuit-kpi{border-bottom:1px solid #1d3340;padding:4px 0 7px}
.circuit-kpi:last-child{border-bottom:none}
.circuit-kpi .k{font-size:8px;color:#8596a2;text-transform:uppercase}
.circuit-kpi .v{font-size:16px;font-weight:950;margin-top:1px}

.insight{display:grid;grid-template-columns:28px 1fr;gap:8px;padding:9px 0;border-bottom:1px solid #1a2d38}
.insight:last-child{border-bottom:none}
.insight-icon{font-size:18px;text-align:center}
.insight .t{font-size:10px;font-weight:950}
.insight .s{font-size:9px;color:#8999a5;margin-top:2px;line-height:1.35}

.tyre-table{width:100%;border-collapse:collapse;font-size:9px}
.tyre-table th{background:#0d1a24;color:#8fa0ac;font-size:8px;text-transform:uppercase;text-align:left;padding:7px;border-bottom:1px solid #203845}
.tyre-table td{padding:7px;border-bottom:1px solid #142630}
.tyre-dot{
  display:inline-flex;width:22px;height:22px;border:2px solid currentColor;border-radius:50%;
  align-items:center;justify-content:center;font-size:9px;font-weight:1000;margin-right:6px;
}
.c-soft{color:#ff2638}.c-medium{color:#ffd21f}.c-hard{color:#f0f3f5}.c-intermediate{color:#2ed47a}.c-wet{color:#2d9cff}

.standings{border:1px solid #1c3340;border-radius:6px;overflow:hidden;background:#050c11}
.stand-head,.stand-row{
  display:grid;grid-template-columns:42px minmax(145px,1.3fr) minmax(95px,.9fr) 50px 83px 65px 65px;
  gap:6px;align-items:center;padding:6px 9px;
}
.stand-head{background:#0d1922;color:#81929d;font-size:8px;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid #243b48}
.stand-row{font-size:9px;border-bottom:1px solid #12242e}
.stand-row:last-child{border-bottom:none}
.stand-row.selected{background:rgba(255,210,31,.065);box-shadow:inset 0 0 0 1.5px #ffd21f}
.stand-pos{font-size:13px;font-weight:1000}
.stand-driver{font-weight:950;font-size:9px}
.stand-team{color:#8a9aa5}
.stand-num{text-align:right;font-weight:850}

.footerline{display:flex;justify-content:space-between;margin-top:8px;color:#70818c;font-size:8px}
.scenario-chip{
  display:inline-block;border:1px solid #26404f;background:#07111a;padding:5px 8px;border-radius:4px;
  color:#aebac3;font-size:8px;font-weight:900;margin-left:5px;text-transform:uppercase;
}
.scenario-chip.active{border-color:#ff2638;color:#ffb0b6;background:rgba(255,38,56,.05)}

@media(max-width:1200px){
  .topbar{grid-template-columns:1fr}.gp-head{border-left:none;padding-left:0;margin-top:7px}.head-meta{text-align:left;margin-top:6px}
  .result-shell{grid-template-columns:1fr}.result-flex{grid-template-columns:145px repeat(3,1fr)}.result-kpi:nth-child(n+5){display:none}
  .stand-head,.stand-row{grid-template-columns:38px minmax(120px,1.4fr) 48px 70px 58px 58px}.stand-team-col{display:none}
}
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

# First small control: choose GP so header can reflect it.
event_idx=st.selectbox(
    "Grand Prix",
    range(len(calendar)),
    index=min(def_idx,len(calendar)-1),
    format_func=lambda i:f"R{calendar[i].get('round','—')} · {calendar[i].get('name','Grand Prix')}",
    key="top_gp_hidden",
    label_visibility="collapsed",
)
event=calendar[event_idx]
details=official_event(event["key"],event)
race_laps=int(details.get("number_of_laps") or 57)

# Header - intentionally no F1 logo.
st.markdown(
    f"""
    <div class="topbar">
      <div>
        <div class="brand-title"><span>STRATEGY</span> ENGINE</div>
        <div class="brand-sub">Race simulator · pit-wall decision support</div>
      </div>
      <div class="gp-head">
        <div class="round-box">Round {event.get("round","—")}</div>
        <div>
          <div class="gp-name">{details.get("name",event.get("name","Grand Prix"))}</div>
          <div class="gp-place">{details.get("location",event.get("location",""))} · {event.get("dates","")}</div>
        </div>
      </div>
      <div class="head-meta">
        <b>Built with FastF1</b><br>
        Weather: Open-Meteo · 30,000 Monte Carlo races
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------
# SCENARIO SETUP
# ----------------------------------------------------------------
st.markdown(
    '<div class="setup-shell"><div class="setup-head">'
    '<div><span class="setup-title">Race scenario setup</span>'
    '<span class="setup-caption">Configure the race scenario and run the simulation</span></div>'
    '<div><span class="scenario-chip active">Scenario A</span>'
    '<span class="scenario-chip">Scenario B</span>'
    '<span class="scenario-chip">Scenario C</span></div>'
    '</div></div>',
    unsafe_allow_html=True,
)

setup_cols=st.columns(
    [1.30,1.05,.75,.58,.75,.75,1.00,.70,.70],
    gap="small",
    vertical_alignment="bottom",
)

with setup_cols[0]:
    selected_driver=st.selectbox("Driver",drivers,index=default_driver_idx,key="driver_main")
with setup_cols[1]:
    stops=st.selectbox("Race strategy",[1,2],index=1,format_func=lambda x:f"{x} stop" if x==1 else f"{x} stops",key="stops_main")

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
    "EXPECTED":"Expected conditions","DRY":"Dry","HOT_DRY":"Hot & dry","COOL_DRY":"Cool & dry",
    "CHANGEABLE":"Changeable","RAIN":"Rain","HEAVY_RAIN":"Heavy rain",
}
WEATHER_COLORS={
    "EXPECTED":"#687885","DRY":"#8e7d31","HOT_DRY":"#a75a23","COOL_DRY":"#316c86",
    "CHANGEABLE":"#684785","RAIN":"#195e9a","HEAVY_RAIN":"#0c426f",
}

def weather_values(mode):
    if mode=="EXPECTED":
        return dict(air=expected_air,track=expected_track,rain=expected_rain,wind=expected_wind,humidity=expected_humidity,source="Open-Meteo")
    if mode=="DRY":
        return dict(air=expected_air,track=expected_track,rain=0.0,wind=expected_wind,humidity=min(expected_humidity,55),source="Scenario")
    if mode=="HOT_DRY":
        return dict(air=max(32,expected_air+4),track=max(48,expected_track+8),rain=0.0,wind=max(2,expected_wind),humidity=min(45,expected_humidity),source="Scenario")
    if mode=="COOL_DRY":
        return dict(air=min(20,expected_air-4),track=min(30,expected_track-8),rain=0.0,wind=expected_wind,humidity=max(50,expected_humidity),source="Scenario")
    if mode=="CHANGEABLE":
        return dict(air=min(expected_air,24),track=min(expected_track,34),rain=.45,wind=max(expected_wind,10),humidity=max(expected_humidity,70),source="Scenario")
    if mode=="RAIN":
        return dict(air=min(expected_air,21),track=min(expected_track,27),rain=.90,wind=max(expected_wind,12),humidity=max(expected_humidity,85),source="Scenario")
    return dict(air=min(expected_air,19),track=min(expected_track,24),rain=.98,wind=max(expected_wind,15),humidity=max(expected_humidity,92),source="Scenario")

# Weather profile compact: static/2/3 phase.
with setup_cols[6]:
    weather_profile=st.selectbox(
        "Weather scenario",["STATIC","2_PHASES","3_PHASES"],index=2,
        format_func=lambda x:{"STATIC":"Static","2_PHASES":"2-phase","3_PHASES":"3-phase"}[x],
        key=f"wp:{event['key']}",
    )

# Build phase controls beneath setup, like the reference mockup.
phase_cols=st.columns([1.15,.7,1.15,.7,1.15,.95,.70],gap="small",vertical_alignment="bottom")

with phase_cols[0]:
    phase1=st.selectbox("Weather L1",WEATHER_OPTIONS,index=1,format_func=lambda x:WEATHER_LABELS[x],key=f"p1:{event['key']}")

if weather_profile in {"2_PHASES","3_PHASES"}:
    switch1_options=list(range(4,max(5,race_laps-5)))
    default_switch1=max(4,min(race_laps-6,int(round(race_laps*.38))))
    with phase_cols[1]:
        switch1=st.selectbox("Phase 2 starts",switch1_options,index=switch1_options.index(default_switch1),format_func=lambda x:f"L{x}",key=f"sw1:{event['key']}:{weather_profile}")
    with phase_cols[2]:
        phase2=st.selectbox("Weather phase 2",WEATHER_OPTIONS,index=5,format_func=lambda x:WEATHER_LABELS[x],key=f"p2:{event['key']}:{weather_profile}")
else:
    switch1=None; phase2=None
    with phase_cols[1]: st.selectbox("Phase 2 starts",["—"],disabled=True)
    with phase_cols[2]: st.selectbox("Weather phase 2",["—"],disabled=True)

if weather_profile=="3_PHASES":
    switch2_options=list(range(switch1+4,max(switch1+5,race_laps-2)))
    default_switch2=max(switch1+4,min(race_laps-2,int(round(race_laps*.70))))
    with phase_cols[3]:
        switch2=st.selectbox("Phase 3 starts",switch2_options,index=switch2_options.index(default_switch2),format_func=lambda x:f"L{x}",key=f"sw2:{event['key']}:{switch1}")
    with phase_cols[4]:
        phase3=st.selectbox("Weather phase 3",WEATHER_OPTIONS,index=6,format_func=lambda x:WEATHER_LABELS[x],key=f"p3:{event['key']}:{switch1}")
else:
    switch2=None; phase3=None
    with phase_cols[3]: st.selectbox("Phase 3 starts",["—"],disabled=True)
    with phase_cols[4]: st.selectbox("Weather phase 3",["—"],disabled=True)

with phase_cols[5]:
    neutralisation_mode=st.selectbox(
        "Safety Car / VSC",["NONE","SC","VSC"],
        format_func=lambda x:{"NONE":"No SC / VSC","SC":"Safety Car","VSC":"Virtual Safety Car"}[x],
        key=f"rc:{event['key']}",
    )

if neutralisation_mode in {"SC","VSC"}:
    neutral_opts=["RANDOM"]+list(range(2,max(3,race_laps-1)))
    with phase_cols[6]:
        neutral_choice=st.selectbox("Lap",neutral_opts,format_func=lambda x:"Random" if x=="RANDOM" else f"L{x}",key=f"rclap:{event['key']}:{neutralisation_mode}")
    neutralisation_lap=None if neutral_choice=="RANDOM" else int(neutral_choice)
else:
    neutralisation_lap=None
    with phase_cols[6]: st.selectbox("Lap",["—"],disabled=True)

def make_phase(start_lap,end_lap,mode):
    vals=weather_values(mode)
    return dict(
        start_lap=int(start_lap),end_lap=int(end_lap),mode=mode,
        track_temp=float(vals["track"]),rain_probability=float(vals["rain"]),
        air_temp=float(vals["air"]),wind=float(vals["wind"]),humidity=float(vals["humidity"]),source=vals["source"],
    )

if weather_profile=="STATIC":
    weather_timeline=[make_phase(1,race_laps,phase1)]
elif weather_profile=="2_PHASES":
    weather_timeline=[make_phase(1,switch1-1,phase1),make_phase(switch1,race_laps,phase2)]
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
    p["mode"] in {"CHANGEABLE","RAIN","HEAVY_RAIN"} or (p["mode"]=="EXPECTED" and p["rain_probability"]>=.20)
    for p in weather_timeline
)
available_compounds=["SOFT","MEDIUM","HARD"]+(["INTERMEDIATE","WET"] if wet_tyres_enabled else [])
phase1_mode=weather_timeline[0]["mode"]
default_start="WET" if phase1_mode=="HEAVY_RAIN" else "INTERMEDIATE" if phase1_mode=="RAIN" else "MEDIUM"
if default_start not in available_compounds: default_start="MEDIUM"

with setup_cols[2]:
    start_compound=st.selectbox("Tyre selection",available_compounds,index=available_compounds.index(default_start),key=f"start:{event['key']}:{phase1_mode}")

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
soft_deg=float(deg.get("SOFT",.12)); medium_deg=float(deg.get("MEDIUM",.08)); hard_deg=float(deg.get("HARD",.055))
undercut=min(.95,.48+overtaking*.35+max(0,medium_deg-.06)*.7)

inventory=(analysis_data or {}).get("inventory") or {
    "SOFT":{"new":1,"used":1},"MEDIUM":{"new":1,"used":1},"HARD":{"new":1,"used":1},
}
inventory.setdefault("INTERMEDIATE",{"new":4,"used":0}); inventory.setdefault("WET",{"new":3,"used":0})
tyres={
    "SOFT":TyreModel("SOFT",-.55,soft_deg),"MEDIUM":TyreModel("MEDIUM",0.0,medium_deg),
    "HARD":TyreModel("HARD",.45,hard_deg),"INTERMEDIATE":TyreModel("INTERMEDIATE",.05,.035),
    "WET":TyreModel("WET",.25,.022),
}

provisional_inputs=SimulationInputs(
    CircuitProfile(race_laps,pit_loss,pit_loss*.55,overtaking,undercut,track_temp),
    DriverContext(selected_driver,(analysis_data or {}).get("team_name",""),grid,pace_delta),
    tyres,inventory,sc_prob,rain_prob,simulations,
    mandatory_race_compounds=("HARD","MEDIUM"),
    rivals=(analysis_data or {}).get("grid_model") or None,
    neutralisation_mode=neutralisation_mode,allowed_compounds=tuple(available_compounds),
    weather_mode=phase1_mode,weather_timeline=weather_timeline,neutralisation_lap=neutralisation_lap,
)

s2_options=legal_next_compounds([start_compound],stops+1,provisional_inputs) or available_compounds
with setup_cols[4]:
    stint2=st.selectbox("Stint 2",s2_options,key=f"s2:{analysis_key}:{weather_profile}:{start_compound}:{stops}")
if stops==2:
    s3_options=legal_next_compounds([start_compound,stint2],3,provisional_inputs) or available_compounds
    with setup_cols[5]:
        stint3=st.selectbox("Stint 3",s3_options,key=f"s3:{analysis_key}:{weather_profile}:{start_compound}:{stint2}")
else:
    stint3=None
    with setup_cols[5]: st.selectbox("Stint 3",["—"],disabled=True)

pit1_max=max(4,race_laps-6 if stops==2 else race_laps-2)
pit1_default=max(3,min(pit1_max,int(round(race_laps*.38))))
pit1_options=list(range(3,pit1_max+1))
with setup_cols[7]:
    pit_lap_1=st.selectbox("Pit lap 1",pit1_options,index=pit1_options.index(pit1_default),format_func=lambda x:f"L{x}",key=f"pit1:{analysis_key}:{stops}")
if stops==2:
    pit2_min=pit_lap_1+3; pit2_max=max(pit2_min,race_laps-2); pit2_options=list(range(pit2_min,pit2_max+1))
    pit2_default=max(pit2_min,min(pit2_max,int(round(race_laps*.70))))
    with setup_cols[8]:
        pit_lap_2=st.selectbox("Pit lap 2",pit2_options,index=pit2_options.index(pit2_default),format_func=lambda x:f"L{x}",key=f"pit2:{analysis_key}:{pit_lap_1}")
else:
    pit_lap_2=None
    with setup_cols[8]: st.selectbox("Pit lap 2",["—"],disabled=True)

selected_compounds=[start_compound,stint2]+([stint3] if stops==2 else [])
selected_pit_laps=[pit_lap_1]+([pit_lap_2] if stops==2 else [])
short_name={"SOFT":"S","MEDIUM":"M","HARD":"H","INTERMEDIATE":"I","WET":"W"}
neutral_label={"NONE":"NO SC / VSC","SC":"SAFETY CAR","VSC":"VIRTUAL SAFETY CAR"}[neutralisation_mode]
valid_now,rule_reasons=validate_strategy(selected_compounds,provisional_inputs)

# Main actions.
act=st.columns([3.7,1.45,1.45,3.7],gap="small")
with act[1]:
    simulate_clicked=st.button("Simulate my strategy",use_container_width=True,type="primary")
with act[2]:
    optimal_clicked=st.button("Find optimal strategy",use_container_width=True)

if simulate_clicked or optimal_clicked:
    with st.spinner(f"Building current-weekend model for {selected_driver}…"):
        try:
            analysis_data=load_driver_analysis(event,selected_driver)
        except Exception:
            analysis_data={"available":False,"degradation":{},"pace_delta":0.0,"inventory":inventory,"grid_position":None,"grid_model":[]}
        st.session_state["analysis_data"]=analysis_data
        st.session_state["analysis_key"]=analysis_key

    grid=int((analysis_data or {}).get("grid_position") or 10)
    pace_delta=float((analysis_data or {}).get("pace_delta") or 0.0)
    incident=float((analysis_data or {}).get("incident_risk") or sc_prob)
    sc_prob=min(.75,max(.12,.55*sc_prob+.45*incident))
    deg=(analysis_data or {}).get("degradation",{})
    soft_deg=float(deg.get("SOFT",soft_deg)); medium_deg=float(deg.get("MEDIUM",medium_deg)); hard_deg=float(deg.get("HARD",hard_deg))
    loaded_inventory=(analysis_data or {}).get("inventory") or {}; inventory.update(loaded_inventory)
    inventory.setdefault("INTERMEDIATE",{"new":4,"used":0}); inventory.setdefault("WET",{"new":3,"used":0})
    tyres={
        "SOFT":TyreModel("SOFT",-.55,soft_deg),"MEDIUM":TyreModel("MEDIUM",0.0,medium_deg),
        "HARD":TyreModel("HARD",.45,hard_deg),"INTERMEDIATE":TyreModel("INTERMEDIATE",.05,.035),
        "WET":TyreModel("WET",.25,.022),
    }

    final_inputs=SimulationInputs(
        CircuitProfile(race_laps,pit_loss,pit_loss*.55,overtaking,undercut,track_temp),
        DriverContext(selected_driver,(analysis_data or {}).get("team_name",""),grid,pace_delta),
        tyres,inventory,sc_prob,rain_prob,simulations,
        mandatory_race_compounds=("HARD","MEDIUM"),
        rivals=(analysis_data or {}).get("grid_model") or None,
        neutralisation_mode=neutralisation_mode,allowed_compounds=tuple(available_compounds),
        weather_mode=phase1_mode,weather_timeline=weather_timeline,neutralisation_lap=neutralisation_lap,
    )

    valid,reasons=validate_strategy(selected_compounds,final_inputs)
    anchor=selected_compounds
    if not valid:
        legal=enumerate_legal_strategies(final_inputs,start_compound=start_compound,stops=(1,2))
        if simulate_clicked:
            st.error("Selected strategy is not feasible: "+" ".join(reasons)); st.session_state.pop("strategy_result",None); legal=[]
        if optimal_clicked and legal: anchor=legal[0]

    if valid or (optimal_clicked and anchor):
        try:
            result=simulate_selected_strategy(
                final_inputs,anchor,compare_same_start=True,
                pit_laps_override=selected_pit_laps if anchor==selected_compounds else None,
            )
            timeline_key="|".join(f'{p["start_lap"]}-{p["end_lap"]}-{p["mode"]}' for p in weather_timeline)
            result_key=f"{analysis_key}:{'-'.join(selected_compounds)}:{timeline_key}:{neutralisation_mode}:{neutralisation_lap}:{'-'.join(str(x) for x in selected_pit_laps)}"
            st.session_state["strategy_result"]=result; st.session_state["strategy_result_key"]=result_key
        except ValueError as exc:
            st.error(str(exc)); st.session_state.pop("strategy_result",None)

timeline_key="|".join(f'{p["start_lap"]}-{p["end_lap"]}-{p["mode"]}' for p in weather_timeline)
result_key_now=f"{analysis_key}:{'-'.join(selected_compounds)}:{timeline_key}:{neutralisation_mode}:{neutralisation_lap}:{'-'.join(str(x) for x in selected_pit_laps)}"
result=st.session_state.get("strategy_result")
if st.session_state.get("strategy_result_key")!=result_key_now: result=None

# ----------------------------------------------------------------
# HERO RESULT + STRATEGY COMPARISON
# ----------------------------------------------------------------
if result is not None:
    projected=int(result.get("projected_finish_position") or result["most_likely_finish"])
    best=result["optimal"]; delta=float(result["delta_to_optimal_s"])
    st.markdown(
        f"""
        <div class="result-shell">
          <div class="result-card">
            <div class="result-title">Estimated race result</div>
            <div class="result-flex">
              <div class="result-pos">P{projected}</div>
              <div class="result-kpi"><div class="k">Expected finish</div><div class="v">P{result["expected_finish"]:.1f}</div></div>
              <div class="result-kpi"><div class="k">Podium</div><div class="v">{result["podium_probability"]:.0%}</div></div>
              <div class="result-kpi"><div class="k">Top 5</div><div class="v">{result["top5_probability"]:.0%}</div></div>
              <div class="result-kpi"><div class="k">Points</div><div class="v">{result["points_probability"]:.0%}</div></div>
              <div class="result-kpi"><div class="k">Win</div><div class="v">{result["win_probability"]:.0%}</div></div>
            </div>
          </div>
          <div class="compare-card">
            <div class="compare-title">Strategy comparison</div>
            <div class="compare-grid">
              <div class="compare-side"><div class="k">Your plan</div><div class="big">P{result["expected_finish"]:.1f}</div><div class="small">{result["strategy"]} · Pit {result["pit_window"]}</div></div>
              <div class="compare-delta"><div class="v">+{max(0.0,delta):.1f}s</div><div class="k">vs optimal</div></div>
              <div class="compare-side"><div class="k">Best simulated plan</div><div class="big">{best["strategy"]}</div><div class="small">Pit {best["pit_window"]}</div></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="result-shell">
          <div class="result-card">
            <div class="result-title">Estimated race result</div>
            <div style="height:82px;display:flex;align-items:center;color:#738894;font-size:13px">
              Run the simulation to calculate the projected race result.
            </div>
          </div>
          <div class="compare-card">
            <div class="compare-title">Strategy comparison</div>
            <div style="height:82px;display:flex;align-items:center;color:#738894;font-size:13px">
              Your plan will be benchmarked against the best legal strategy.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------
# RACE TIMELINE + CIRCUIT INFO
# ----------------------------------------------------------------
mid=st.columns([2.9,1.0],gap="small")

TYRE_COLORS={"SOFT":"#ff2638","MEDIUM":"#ffd21f","HARD":"#eef2f4","INTERMEDIATE":"#2ed47a","WET":"#2d9cff"}

with mid[0]:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Race timeline <span class="timeline-note">Tyres, weather, pit stops and race events</span></div>',unsafe_allow_html=True)
        tf=go.Figure()
        for phase in weather_timeline:
            tf.add_trace(go.Bar(
                y=["Weather"],x=[phase["end_lap"]-phase["start_lap"]+1],base=[phase["start_lap"]-1],orientation="h",
                marker_color=WEATHER_COLORS.get(phase["mode"],"#687885"),
                text=[WEATHER_LABELS[phase["mode"]]],textposition="inside",
                hovertemplate=f'{WEATHER_LABELS[phase["mode"]]} · L{phase["start_lap"]}–L{phase["end_lap"]}<extra></extra>',
                showlegend=False,
            ))
        bounds=[1]+[x+1 for x in selected_pit_laps]+[race_laps+1]
        for i,comp in enumerate(selected_compounds):
            s=bounds[i]; e=bounds[i+1]-1
            tf.add_trace(go.Bar(
                y=["Your strategy"],x=[e-s+1],base=[s-1],orientation="h",marker_color=TYRE_COLORS.get(comp,"#8896a0"),
                text=[short_name.get(comp,comp[0])],textposition="inside",
                hovertemplate=f'{comp.title()} · L{s}–L{e}<extra></extra>',showlegend=False,
            ))
        if neutralisation_mode in {"SC","VSC"}:
            if neutralisation_lap is not None:
                duration=4 if neutralisation_mode=="SC" else 2
                tf.add_trace(go.Bar(
                    y=["Safety Car / VSC"],x=[duration],base=[neutralisation_lap-1],orientation="h",
                    marker_color="#ffd21f",text=[neutralisation_mode],textposition="inside",
                    hovertemplate=f'{neutralisation_mode} · L{neutralisation_lap}<extra></extra>',showlegend=False,
                ))
            else:
                tf.add_trace(go.Bar(
                    y=["Safety Car / VSC"],x=[race_laps],base=[0],orientation="h",
                    marker_color="rgba(255,210,31,.10)",text=["Random timing"],textposition="inside",
                    hovertemplate="Neutralisation timing sampled in Monte Carlo<extra></extra>",showlegend=False,
                ))
        for pit in selected_pit_laps:
            tf.add_vline(x=pit,line_width=1.5,line_dash="dot",line_color="#f8fbfd")
            tf.add_annotation(x=pit,y="Your strategy",text=f"PIT<br>L{pit}",showarrow=False,yshift=-23,font=dict(size=8,color="#e6edf2"))
        tf.update_layout(
            height=245,barmode="overlay",bargap=.25,margin=dict(l=105,r=12,t=23,b=30),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#dce5eb",size=9),
            xaxis=dict(title="Lap",range=[0,race_laps],dtick=max(5,int(round(race_laps/10))),gridcolor="#17303c",zeroline=False,side="top"),
            yaxis=dict(categoryarray=["Weather","Safety Car / VSC","Your strategy"],categoryorder="array",autorange="reversed",gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(tf,use_container_width=True,config={"displayModeBar":False})

with mid[1]:
    with st.container(border=True):
        panel_title("Circuit info")
        st.markdown(f'<div style="font-size:11px;color:#b9c4cc;margin-bottom:5px">{details.get("location",event.get("location",""))}</div>',unsafe_allow_html=True)
        if details.get("track_image_url"):
            st.image(details["track_image_url"],use_container_width=True)
        length=details.get("circuit_length_km"); dist=details.get("race_distance_km")
        st.markdown(
            f"""
            <div class="circuit-kpis">
              <div class="circuit-kpi"><div class="k">Length</div><div class="v">{f"{length:.3f} km" if isinstance(length,(float,int)) else "—"}</div></div>
              <div class="circuit-kpi"><div class="k">Laps</div><div class="v">{race_laps}</div></div>
              <div class="circuit-kpi"><div class="k">Race distance</div><div class="v">{f"{dist:.1f} km" if isinstance(dist,(float,int)) else "—"}</div></div>
              <div class="circuit-kpi"><div class="k">Circuit type</div><div class="v">{circuit_type}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------
# WEATHER CHART + TYRE PERFORMANCE + KEY INSIGHTS
# ----------------------------------------------------------------
lower=st.columns([1.45,.78,.78],gap="small")

with lower[0]:
    with st.container(border=True):
        panel_title("Weather & track evolution")
        laps=list(range(1,race_laps+1))
        temp_series=[]; rain_series=[]
        for lap in laps:
            ph=next((p for p in weather_timeline if p["start_lap"]<=lap<=p["end_lap"]),weather_timeline[-1])
            temp_series.append(ph["track_temp"])
            rain_series.append(ph["rain_probability"]*100)
        wf=go.Figure()
        wf.add_trace(go.Scatter(x=laps,y=temp_series,mode="lines",name="Track temperature",line=dict(color="#ff313c",width=2.2),hovertemplate="L%{x}: %{y:.0f}°C<extra></extra>"))
        wf.add_trace(go.Bar(x=laps,y=rain_series,name="Rain intensity",marker_color="#2479d1",opacity=.65,yaxis="y2",hovertemplate="L%{x}: %{y:.0f}% wet exposure<extra></extra>"))
        wf.update_layout(
            height=260,margin=dict(l=45,r=45,t=10,b=35),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#cbd5dc",size=8),
            xaxis=dict(title="Lap",gridcolor="#17303c",zeroline=False),
            yaxis=dict(title="Track temp °C",gridcolor="#17303c",zeroline=False),
            yaxis2=dict(title="Rain %",overlaying="y",side="right",range=[0,100],showgrid=False),
            legend=dict(orientation="h",y=1.12,x=.62,font=dict(size=8)),
        )
        st.plotly_chart(wf,use_container_width=True,config={"displayModeBar":False})

with lower[1]:
    with st.container(border=True):
        panel_title("Tyre performance (est.)")
        rows=[
            ("SOFT","S",soft_deg,"-0.55s","Dry / quali"),
            ("MEDIUM","M",medium_deg,"Reference","Dry"),
            ("HARD","H",hard_deg,"+0.45s","Dry / long run"),
        ]
        if wet_tyres_enabled:
            rows += [
                ("INTERMEDIATE","I",.035,"Wet-specialist","Damp / rain"),
                ("WET","W",.022,"Heavy-wet","Heavy rain"),
            ]
        tr=""
        for name,abbr,dg,pace,use in rows:
            cls="c-"+name.lower()
            tr+=f'<tr><td><span class="tyre-dot {cls}">{abbr}</span>{name.title()}</td><td>{dg:.3f}</td><td>{pace}</td><td>{use}</td></tr>'
        st.markdown(
            '<table class="tyre-table"><thead><tr><th>Compound</th><th>Deg s/lap</th><th>Relative pace</th><th>Best use</th></tr></thead>'
            f'<tbody>{tr}</tbody></table>',
            unsafe_allow_html=True,
        )

with lower[2]:
    with st.container(border=True):
        panel_title("Key insights")
        insights=[]
        wet_phases=[p for p in weather_timeline if p["mode"] in {"CHANGEABLE","RAIN","HEAVY_RAIN"}]
        if wet_phases:
            first_wet=wet_phases[0]
            insights.append(("🌧️",f'{WEATHER_LABELS[first_wet["mode"]]} from lap {first_wet["start_lap"]}',f'Optimal crossover window depends on tyre choice near L{max(1,first_wet["start_lap"]-2)}–{first_wet["start_lap"]+2}.'))
        else:
            insights.append(("☀️","Dry race profile","Strategy is driven mainly by degradation, undercut power and pit loss."))
        if neutralisation_mode!="NONE":
            timing="random timing" if neutralisation_lap is None else f"lap {neutralisation_lap}"
            insights.append(("🟨",f"{neutral_label.title()} · {timing}",f"Pit-loss benefit applies only when a stop aligns with the neutralisation window."))
        else:
            insights.append(("🟩","No neutralisation selected","All planned stops pay the normal green-flag pit loss."))
        if result is not None:
            delta=float(result["delta_to_optimal_s"])
            if delta>.5:
                insights.append(("⏱️",f"Strategy loses ~{delta:.1f}s vs optimum",f'Best same-start plan: {result["optimal"]["strategy"]}, pit {result["optimal"]["pit_window"]}.'))
            else:
                insights.append(("✅","Strategy close to optimum","Your fixed plan is within half a second of the model optimum."))
            insights.append(("📊",f'Top 5 in {result["top5_probability"]:.0%} of simulations',f'Points probability {result["points_probability"]:.0%}; podium {result["podium_probability"]:.0%}.'))
        for icon,title,sub in insights[:4]:
            st.markdown(f'<div class="insight"><div class="insight-icon">{icon}</div><div><div class="t">{title}</div><div class="s">{sub}</div></div></div>',unsafe_allow_html=True)

# ----------------------------------------------------------------
# ESTIMATED CLASSIFICATION
# ----------------------------------------------------------------
with st.container(border=True):
    panel_title("Estimated final classification (race simulation)")
    if result is None:
        st.info("Run the simulation to populate the projected classification.")
    else:
        standings=result.get("estimated_classification",[])
        rows_html=[]
        for row in standings:
            selected_class=" selected" if row.get("selected_driver") else ""
            team=row.get("team_name") or "—"
            grid_value=f'P{row["grid_position"]}' if row.get("grid_position") else "—"
            rows_html.append(
                f'<div class="stand-row{selected_class}">'
                f'<div class="stand-pos">{row["projected_position"]}</div>'
                f'<div class="stand-driver">{row["driver_name"]}</div>'
                f'<div class="stand-team stand-team-col">{team}</div>'
                f'<div class="stand-num">{grid_value}</div>'
                f'<div class="stand-num">P{row["expected_finish"]:.1f}</div>'
                f'<div class="stand-num">{row["podium_probability"]:.0%}</div>'
                f'<div class="stand-num">{row["points_probability"]:.0%}</div>'
                f'</div>'
            )
        st.markdown(
            '<div class="standings">'
            '<div class="stand-head"><div>Pos</div><div>Driver</div><div class="stand-team-col">Team</div>'
            '<div style="text-align:right">Grid</div><div style="text-align:right">Expected</div>'
            '<div style="text-align:right">Podium</div><div style="text-align:right">Points</div></div>'
            +''.join(rows_html)+'</div>',
            unsafe_allow_html=True,
        )

with st.expander("Low-confidence overrides",expanded=False):
    st.caption("Correct pit-lane loss or tyre availability only if you have more authoritative weekend data.")
    cols=st.columns(6,gap="small")
    with cols[0]:
        pit_loss=st.number_input("Pit loss (s)",10.0,40.0,float(pit_loss),0.1,key=f"{event['key']}:pit")
    for comp,col in zip(("SOFT","MEDIUM","HARD","INTERMEDIATE","WET"),cols[1:]):
        with col:
            st.markdown(f"**{comp.title()}**")
            x,y=st.columns(2); base=f"{analysis_key}:{comp}"
            inventory[comp]["new"]=x.number_input("N",0,6,int(inventory[comp].get("new",0)),key=base+":n")
            inventory[comp]["used"]=y.number_input("U",0,6,int(inventory[comp].get("used",0)),key=base+":u")

st.markdown(
    '<div class="footerline"><div>Strategy Engine V2.1 · Simulate. Analyse. Be ready.</div>'
    '<div>Data: FastF1 &nbsp;|&nbsp; Weather: Open-Meteo &nbsp;|&nbsp; Model: Monte Carlo</div></div>',
    unsafe_allow_html=True,
)
