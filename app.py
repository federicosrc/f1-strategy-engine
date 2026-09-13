from __future__ import annotations

from datetime import datetime, timezone
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from official_sources import F1OfficialClient, PirelliCurrentSeason
from data_sources import FastF1DataClient, OpenMeteoClient
from strategy_engine_v18 import (
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
    '<div class="setup-title">Race scenario setup'
    '<span class="setup-sub">Choose the race assumptions you want to test. Every selection below changes the simulation.</span>'
    '</div>',
    unsafe_allow_html=True,
)

# GP | Driver | Start | Stops | Stint 2 | Stint 3 | Weather | Neutralisation | Pit 1 | Pit 2
input_cols=st.columns(
    [1.72,1.38,.65,.56,.66,.66,1.05,1.00,.60,.60],
    gap="small",
    vertical_alignment="bottom",
)

with input_cols[0]:
    event_idx=st.selectbox(
        "Grand Prix",
        range(len(calendar)),
        index=min(def_idx,len(calendar)-1),
        format_func=lambda i:f"R{calendar[i].get('round','—')} · {calendar[i].get('name','Grand Prix')}",
    )

event=calendar[event_idx]
details=official_event(event["key"],event)
race_laps=int(details.get("number_of_laps") or 57)

with input_cols[1]:
    selected_driver=st.selectbox("Driver",drivers,index=default_driver_idx)
with input_cols[2]:
    start_compound=st.selectbox("Start tyre",["SOFT","MEDIUM","HARD"],index=1)
with input_cols[3]:
    stops=st.selectbox("Stops",[1,2],index=0)

with input_cols[6]:
    weather_mode=st.selectbox(
        "Weather",
        ["FORECAST","DRY","HOT_DRY","COOL_DRY","CHANGEABLE","RAIN_LIKELY"],
        format_func=lambda x:{
            "FORECAST":"Forecast",
            "DRY":"Dry",
            "HOT_DRY":"Hot & dry",
            "COOL_DRY":"Cool & dry",
            "CHANGEABLE":"Changeable",
            "RAIN_LIKELY":"Rain likely",
        }[x],
    )

with input_cols[7]:
    neutralisation_mode=st.selectbox(
        "Race control",
        ["NONE","SC","VSC"],
        format_func=lambda x:{
            "NONE":"No SC / VSC",
            "SC":"Safety Car",
            "VSC":"Virtual Safety Car",
        }[x],
    )

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
    forecast_weather=meteo.forecast_at(geo["latitude"],geo["longitude"],race_dt)
except Exception:
    forecast_weather={}

forecast_air=float(forecast_weather.get("temperature_2m",25.0))
forecast_track=float(forecast_weather.get("track_temperature_estimate",forecast_air+15.0))
forecast_rain=float(forecast_weather.get("precipitation_probability",5.0) or 0)/100.0
forecast_wind=float(forecast_weather.get("wind_speed_10m",0.0) or 0)
forecast_humidity=float(forecast_weather.get("relative_humidity_2m",55.0) or 55.0)

# Scenario override: forecast remains the baseline, then the user can stress-test realistic conditions.
if weather_mode=="FORECAST":
    air_temp=forecast_air
    track_temp=forecast_track
    rain_prob=forecast_rain
    wind=forecast_wind
    humidity=forecast_humidity
    weather_label="Forecast"
    weather_source="Open-Meteo race-time forecast"
elif weather_mode=="DRY":
    air_temp=forecast_air
    track_temp=forecast_track
    rain_prob=0.0
    wind=forecast_wind
    humidity=min(forecast_humidity,55.0)
    weather_label="Dry"
    weather_source="User scenario · forecast temperatures"
elif weather_mode=="HOT_DRY":
    air_temp=max(32.0,forecast_air+4.0)
    track_temp=max(48.0,forecast_track+8.0)
    rain_prob=0.0
    wind=max(2.0,forecast_wind)
    humidity=min(45.0,forecast_humidity)
    weather_label="Hot & dry"
    weather_source="User stress scenario"
elif weather_mode=="COOL_DRY":
    air_temp=min(20.0,forecast_air-4.0)
    track_temp=min(30.0,forecast_track-8.0)
    rain_prob=0.0
    wind=forecast_wind
    humidity=max(50.0,forecast_humidity)
    weather_label="Cool & dry"
    weather_source="User stress scenario"
elif weather_mode=="CHANGEABLE":
    air_temp=min(forecast_air,24.0)
    track_temp=min(forecast_track,34.0)
    rain_prob=0.45
    wind=max(forecast_wind,10.0)
    humidity=max(forecast_humidity,70.0)
    weather_label="Changeable"
    weather_source="User scenario · intermittent rain risk"
else:
    air_temp=min(forecast_air,22.0)
    track_temp=min(forecast_track,28.0)
    rain_prob=0.80
    wind=max(forecast_wind,12.0)
    humidity=max(forecast_humidity,80.0)
    weather_label="Rain likely"
    weather_source="User scenario · high rain probability"

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
tyres={
    "SOFT":TyreModel("SOFT",-.55,soft_deg),
    "MEDIUM":TyreModel("MEDIUM",0.0,medium_deg),
    "HARD":TyreModel("HARD",.45,hard_deg),
}

provisional_inputs=SimulationInputs(
    CircuitProfile(race_laps,pit_loss,pit_loss*.55,overtaking,undercut,track_temp),
    DriverContext(selected_driver,(analysis_data or {}).get("team_name",""),grid,pace_delta),
    tyres,inventory,sc_prob,rain_prob,simulations,
    mandatory_race_compounds=("HARD","MEDIUM"),
    rivals=(analysis_data or {}).get("grid_model") or None,
    neutralisation_mode=neutralisation_mode,
)

s2_options=legal_next_compounds(
    [start_compound],stops+1,provisional_inputs
) or [c for c in ("SOFT","MEDIUM","HARD") if total_sets(inventory,c)>0]

with input_cols[4]:
    stint2=st.selectbox(
        "Stint 2",s2_options,key=f"s2:{analysis_key}:{start_compound}:{stops}"
    )

if stops==2:
    s3_options=legal_next_compounds(
        [start_compound,stint2],3,provisional_inputs
    ) or [c for c in ("SOFT","MEDIUM","HARD") if total_sets(inventory,c)>0]
    with input_cols[5]:
        stint3=st.selectbox(
            "Stint 3",s3_options,key=f"s3:{analysis_key}:{start_compound}:{stint2}"
        )
else:
    stint3=None
    with input_cols[5]:
        st.selectbox(
            "Stint 3",["—"],index=0,disabled=True,
            key=f"s3-disabled:{analysis_key}:{start_compound}:{stops}",
        )

pit1_max=max(4,race_laps-6 if stops==2 else race_laps-2)
pit1_default=max(3,min(pit1_max,int(round(race_laps*.38))))
pit1_options=list(range(3,pit1_max+1))
with input_cols[8]:
    pit_lap_1=st.selectbox(
        "Pit lap 1",pit1_options,
        index=pit1_options.index(pit1_default) if pit1_default in pit1_options else 0,
        key=f"pit1:{analysis_key}:{stops}",
    )

if stops==2:
    pit2_min=pit_lap_1+3
    pit2_max=max(pit2_min,race_laps-2)
    pit2_options=list(range(pit2_min,pit2_max+1))
    pit2_default=max(pit2_min,min(pit2_max,int(round(race_laps*.70))))
    with input_cols[9]:
        pit_lap_2=st.selectbox(
            "Pit lap 2",pit2_options,
            index=pit2_options.index(pit2_default) if pit2_default in pit2_options else 0,
            key=f"pit2:{analysis_key}:{pit_lap_1}",
        )
else:
    pit_lap_2=None
    with input_cols[9]:
        st.selectbox(
            "Pit lap 2",["—"],index=0,disabled=True,
            key=f"pit2-disabled:{analysis_key}:{stops}",
        )

selected_compounds=[start_compound,stint2]+([stint3] if stops==2 else [])
selected_pit_laps=[pit_lap_1]+([pit_lap_2] if stops==2 else [])

button_cols=st.columns([3.2,1.55,1.55,3.2],gap="small")
with button_cols[1]:
    simulate_clicked=st.button("Simulate my strategy",use_container_width=True)
with button_cols[2]:
    optimal_clicked=st.button("Find optimal strategy",use_container_width=True)

valid_now,rule_reasons=validate_strategy(selected_compounds,provisional_inputs)
pills='<span class="strategy-arrow">→</span>'.join(
    f'<span class="strategy-pill">{c[0]}</span>' for c in selected_compounds
)
neutral_label={"NONE":"NO SC/VSC","SC":"SAFETY CAR","VSC":"VIRTUAL SAFETY CAR"}[neutralisation_mode]

st.markdown(
    f'<div class="strategy-strip">'
    f'<span style="color:#8d9aa7;font-size:10px;font-weight:800">SELECTED</span>'
    f'{pills}'
    f'<span style="color:#8d9aa7;font-size:10px;font-weight:800">'
    f'PIT {" / ".join("L"+str(x) for x in selected_pit_laps)} · {weather_label.upper()} · {neutral_label}</span>'
    f'<span class="{"rule-ok" if valid_now else "rule-bad"}">'
    f'{"LEGAL DRY STRATEGY" if valid_now else "CHECK STRATEGY"}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

if simulate_clicked or optimal_clicked:
    with st.spinner(f"Analysing current-weekend data for {selected_driver}…"):
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
    inventory=(analysis_data or {}).get("inventory") or inventory
    tyres={
        "SOFT":TyreModel("SOFT",-.55,soft_deg),
        "MEDIUM":TyreModel("MEDIUM",0.0,medium_deg),
        "HARD":TyreModel("HARD",.45,hard_deg),
    }
    undercut=min(.95,.48+overtaking*.35+max(0,medium_deg-.06)*.7)

    final_inputs=SimulationInputs(
        CircuitProfile(race_laps,pit_loss,pit_loss*.55,overtaking,undercut,track_temp),
        DriverContext(selected_driver,(analysis_data or {}).get("team_name",""),grid,pace_delta),
        tyres,inventory,sc_prob,rain_prob,simulations,
        mandatory_race_compounds=("HARD","MEDIUM"),
        rivals=(analysis_data or {}).get("grid_model") or None,
        neutralisation_mode=neutralisation_mode,
    )

    valid,reasons=validate_strategy(selected_compounds,final_inputs)
    anchor=selected_compounds

    if not valid:
        legal=enumerate_legal_strategies(
            final_inputs,start_compound=start_compound,stops=(1,2)
        )
        if simulate_clicked:
            st.error("Selected strategy is not feasible with the loaded data: "+" ".join(reasons))
            st.session_state.pop("strategy_result",None)
            legal=[]
        if optimal_clicked and legal:
            anchor=legal[0]

    if valid or (optimal_clicked and anchor):
        try:
            result=simulate_selected_strategy(
                final_inputs,anchor,compare_same_start=True,
                pit_laps_override=selected_pit_laps if anchor==selected_compounds else None,
            )
            result_key=(
                f"{analysis_key}:{'-'.join(selected_compounds)}:"
                f"{weather_mode}:{neutralisation_mode}:"
                f"{'-'.join(str(x) for x in selected_pit_laps)}"
            )
            st.session_state["strategy_result"]=result
            st.session_state["strategy_result_key"]=result_key
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.pop("strategy_result",None)

result=st.session_state.get("strategy_result")
result_key_now=(
    f"{analysis_key}:{'-'.join(selected_compounds)}:"
    f"{weather_mode}:{neutralisation_mode}:"
    f"{'-'.join(str(x) for x in selected_pit_laps)}"
)
if st.session_state.get("strategy_result_key")!=result_key_now:
    result=None

# Canonical projected position = the same position used in the final-classification table.
if result is not None:
    projected_position=int(
        result.get("projected_finish_position")
        or result.get("most_likely_finish")
    )
    model_conf=result.get("race_model_confidence","low").upper()
    competitors=result.get("competitors_modelled",0)
    projection_html=f"""
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
            Same ranking as final classification · {competitors} rivals · {model_conf} confidence
          </div>
        </div>
      </div>
      <div class="projected-kpi"><div class="k">Points</div><div class="v">{result["points_probability"]:.0%}</div><div class="s">P10 or better</div></div>
      <div class="projected-kpi"><div class="k">Top 5</div><div class="v">{result["top5_probability"]:.0%}</div><div class="s">P5 or better</div></div>
      <div class="projected-kpi"><div class="k">Podium</div><div class="v">{result["podium_probability"]:.0%}</div><div class="s">P1–P3</div></div>
      <div class="projected-kpi"><div class="k">Win</div><div class="v">{result["win_probability"]:.0%}</div><div class="s">P1</div></div>
    </div>
    """
    st.markdown(projection_html,unsafe_allow_html=True)

with st.expander("Low-confidence overrides",expanded=False):
    st.caption("Correct pit-lane loss or remaining tyre sets if you have better official data.")
    cols=st.columns(4,gap="small")
    with cols[0]:
        pit_loss=st.number_input(
            "Pit lane loss (s)",10.0,40.0,float(pit_loss),0.1,
            key=f"{event['key']}:pit",
        )
    for comp,col in zip(("SOFT","MEDIUM","HARD"),cols[1:]):
        with col:
            x,y=st.columns(2)
            base=f"{analysis_key}:{comp}"
            inventory[comp]["new"]=x.number_input(
                f"{comp[0]} new",0,5,int(inventory[comp].get("new",0)),key=base+":n"
            )
            inventory[comp]["used"]=y.number_input(
                f"{comp[0]} used",0,5,int(inventory[comp].get("used",0)),key=base+":u"
            )

st.markdown(
    f"""<div class="se-header">
      <div>
        <div class="se-brand"><span>STRATEGY</span> ENGINE</div>
        <div class="se-sub">Race scenario simulator · current season</div>
      </div>
      <div class="se-headchips">
        <div class="se-chip"><div class="k">Grand Prix</div><div class="v">{details.get('name',event.get('name'))}</div></div>
        <div class="se-chip"><div class="k">Driver</div><div class="v">{selected_driver}</div></div>
        <div class="se-chip"><div class="k">Strategy</div><div class="v">{' → '.join(c[0] for c in selected_compounds)}</div></div>
        <div class="se-chip"><div class="k">Weather</div><div class="v">{weather_label}</div></div>
        <div class="se-chip"><div class="k">Race control</div><div class="v">{neutral_label}</div></div>
      </div>
    </div>""",
    unsafe_allow_html=True,
)

left_col,right_col=st.columns([1.08,1.04],gap="small")

with left_col:
    with st.container(border=True):
        panel_title("Circuit info")
        if details.get("track_image_url"):
            st.image(details["track_image_url"],use_container_width=True)

        length=details.get("circuit_length_km")
        dist=details.get("race_distance_km")
        length_display=f"{length:.3f}" if isinstance(length,(float,int)) else "—"
        dist_display=f"{dist:.3f}" if isinstance(dist,(float,int)) else "—"

        st.markdown(
            f'<div class="circuit-stats">'
            f'<div class="circuit-stat"><div class="k">Laps</div><div class="v">{details.get("number_of_laps","—")}</div><div class="s">F1 official</div></div>'
            f'<div class="circuit-stat"><div class="k">Circuit length</div><div class="v">{length_display}<span class="u">km</span></div><div class="s">F1 official / derived</div></div>'
            f'<div class="circuit-stat"><div class="k">Race distance</div><div class="v">{dist_display}<span class="u">km</span></div><div class="s">F1 official / derived</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        panel_title("Weather & track")
        st.markdown(
            f'<div class="weather-scenario">Scenario: {weather_label}</div>',
            unsafe_allow_html=True,
        )
        metric_html([
            ("Air temp",f"{air_temp:.0f}°C",weather_source),
            ("Track temp",f"{track_temp:.0f}°C","scenario asphalt estimate"),
            ("Rain",f"{rain_prob:.0%}","simulation probability"),
            ("Wind",f"{wind:.0f} km/h",f"humidity {humidity:.0f}%"),
        ])

        indicators=[
            ("Tyre stress",min(1.0,max(.05,medium_deg/.16+max(0,track_temp-38)/80)),"#ffd21f"),
            ("Overtaking difficulty",overtaking,"#ff1e2d"),
            ("Undercut power",undercut,"#37e77b"),
        ]
        for label,val,color in indicators:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:10px;margin:6px 0 3px">'
                f'<span>{label}</span><b>{val:.0%}</b></div>'
                f'<div class="barline"><div style="width:{val*100:.0f}%;background:{color}"></div></div>',
                unsafe_allow_html=True,
            )

with right_col:
    with st.container(border=True):
        panel_title("Your strategy")
        strategy_sequence=" → ".join(c[0] for c in selected_compounds)
        pit_text=" / ".join(f"L{x}" for x in selected_pit_laps)

        st.markdown(
            f"""<div class="strategy-summary">
              <div class="label">Selected race plan</div>
              <div class="sequence">{strategy_sequence}</div>
              <div class="strategy-details">
                <div class="strategy-detail"><div class="k">Driver</div><div class="v">{selected_driver}</div></div>
                <div class="strategy-detail"><div class="k">Pit laps</div><div class="v">{pit_text}</div></div>
                <div class="strategy-detail"><div class="k">Weather</div><div class="v">{weather_label}</div></div>
                <div class="strategy-detail"><div class="k">Race control</div><div class="v">{neutral_label}</div></div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        if result is not None:
            metric_html([
                ("Grid",f"P{grid}","FastF1 / qualifying" if (analysis_data or {}).get("grid_position") else "fallback"),
                ("Projected",f'P{result.get("projected_finish_position",result["most_likely_finish"])}',"same as final classification"),
                ("Expected",f'P{result["expected_finish"]:.1f}',"Monte Carlo mean"),
                ("Strategy cost",f'{result["expected_cost_s"]:.1f}s',"lower is better"),
            ])

            # Requested placement: distribution directly below Your Strategy.
            panel_title("Finish-position distribution")
            ddf=pd.DataFrame([
                {"Position":f"P{k}","Probability":v*100}
                for k,v in result["finish_distribution"].items()
            ])
            fig2=go.Figure(
                go.Bar(
                    x=ddf["Position"],
                    y=ddf["Probability"],
                    marker_color=[
                        "#37e77b" if int(p[1:])<=3
                        else "#ffd21f" if int(p[1:])<=10
                        else "#66727d"
                        for p in ddf["Position"]
                    ],
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
                )
            )
            fig2.update_layout(
                height=275,
                margin=dict(l=8,r=8,t=10,b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#dce2e8"),
                showlegend=False,
                yaxis=dict(title="Probability %",gridcolor="#202a33",zeroline=False),
                xaxis=dict(title=""),
            )
            st.plotly_chart(
                fig2,use_container_width=True,config={"displayModeBar":False},
            )

if result is None:
    with st.container(border=True):
        panel_title("Strategy simulation")
        st.info(
            f"Selected {' → '.join(c[0] for c in selected_compounds)} for {selected_driver}, "
            f"pit {' / '.join('L'+str(x) for x in selected_pit_laps)}, {weather_label}, {neutral_label}. "
            "Press SIMULATE MY STRATEGY to calculate the race outcome."
        )
else:
    optimal=result["optimal"]
    delta=float(result["delta_to_optimal_s"])

    benchmark_col,scenario_col=st.columns([1.32,.88],gap="small")

    with benchmark_col:
        with st.container(border=True):
            panel_title("Strategy benchmark")
            delta_text=f"+{max(0.0,delta):.1f}s"
            st.markdown(
                f"""<div class="benchmark">
                  <div class="bench-card">
                    <div class="k">Your plan</div>
                    <div class="plan">{result["strategy"]}</div>
                    <div class="cost">Pit: {result["pit_window"]}</div>
                    <div class="cost">Expected strategy cost: {result["expected_cost_s"]:.1f}s</div>
                  </div>
                  <div class="bench-vs">
                    <div class="delta">{delta_text}</div>
                    <div class="label">vs best plan<br>lower is better</div>
                  </div>
                  <div class="bench-card best">
                    <div class="k">Best legal plan · same start tyre</div>
                    <div class="plan">{optimal["strategy"]}</div>
                    <div class="cost">Pit window: {optimal["pit_window"]}</div>
                    <div class="cost">Expected strategy cost: {optimal["expected_cost_s"]:.1f}s</div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

    with scenario_col:
        with st.container(border=True):
            panel_title("Scenario assumptions")
            st.markdown(
                f"""
                **Weather:** {weather_label}  
                **Rain probability:** {rain_prob:.0%}  
                **Track temperature:** {track_temp:.0f}°C  
                **Race control:** {neutral_label}  
                **Pit laps:** {' / '.join('L'+str(x) for x in selected_pit_laps)}
                """
            )
            st.caption(
                "Weather and race-control assumptions are shared across each simulated race, "
                "so every driver in that Monte Carlo run experiences the same scenario."
            )

    # Full-width estimated final classification.
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
            '</div>'
            +''.join(rows_html)+
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "This is the canonical projected classification. The large Estimated Final Position above "
            "uses the selected driver's position in this same table."
        )

st.caption(
    "Strategy Engine V1.8 · Race-scenario simulator · Shared weather/race-control assumptions · "
    "Full-grid Monte Carlo classification."
)
