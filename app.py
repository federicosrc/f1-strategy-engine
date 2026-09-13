from __future__ import annotations

from datetime import datetime, timezone
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from official_sources import F1OfficialClient, PirelliCurrentSeason
from data_sources import FastF1DataClient, OpenMeteoClient
from strategy_engine_v21 import (
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
    find_target_outcome,
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
.setup-section{margin:10px 0 6px;color:#d8e1e7;font-size:10px;font-weight:950;text-transform:uppercase;letter-spacing:.10em}
.setup-section span{color:#7e909d;font-weight:700;margin-left:8px;letter-spacing:0;text-transform:none;font-size:9px}

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

.footerline{display:flex;justify-content:center;margin-top:8px;color:#70818c;font-size:8px}
.scenario-chip{
  display:inline-block;border:1px solid #26404f;background:#07111a;padding:5px 8px;border-radius:4px;
  color:#aebac3;font-size:8px;font-weight:900;margin-left:5px;text-transform:uppercase;
}
.scenario-chip.active{border-color:#ff2638;color:#ffb0b6;background:rgba(255,38,56,.05)}

.setup-block{margin:12px 0 14px}
.setup-info-panel{
  min-height:188px;border:1px solid var(--line);border-radius:10px;padding:14px 14px 14px 16px;
  background:linear-gradient(160deg,rgba(9,18,28,.98),rgba(6,11,18,.95));
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.02);
}
.setup-info-panel.blue{box-shadow:inset 4px 0 0 0 #39a8ff}
.setup-info-panel.red{box-shadow:inset 4px 0 0 0 #ff4254}
.setup-info-panel.yellow{box-shadow:inset 4px 0 0 0 #ffd21f}
.setup-badge{
  width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;
  color:white;font-weight:1000;font-size:20px;margin-bottom:14px;border:1px solid rgba(255,255,255,.10)
}
.setup-badge.blue{background:linear-gradient(180deg,#2f9dff,#1d69cf)}
.setup-badge.red{background:linear-gradient(180deg,#ff5867,#dc1d31)}
.setup-badge.yellow{background:linear-gradient(180deg,#ffe14a,#efbe00);color:#14181c}
.setup-info-title{font-size:14px;font-weight:1000;text-transform:uppercase;letter-spacing:.03em}
.setup-info-sub{font-size:9px;color:#8fa0ab;line-height:1.5;margin-top:10px;max-width:190px}
.setup-controls-card{
  min-height:188px;border:1px solid var(--line);border-radius:10px;padding:14px 14px 10px;
  background:linear-gradient(135deg,rgba(8,18,27,.98),rgba(5,11,17,.95));
}
.setup-card-title{font-size:10px;font-weight:1000;text-transform:uppercase;letter-spacing:.12em;color:#c9d2d9;margin-bottom:10px}
.setup-subgrid{margin-top:8px;padding-top:8px;border-top:1px solid #18303c}
.summary-col{display:grid;gap:8px}
.summary-card{border:1px solid #1b3341;border-radius:8px;background:#07111a;padding:10px 11px;min-height:52px}
.summary-card .k{font-size:8px;color:#8597a3;text-transform:uppercase;letter-spacing:.10em}
.summary-card .v{font-size:18px;font-weight:1000;margin-top:3px}
.summary-card .s{font-size:8px;color:#728492;margin-top:2px}
.segment-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:4px}
.tyre-plan-display{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding-top:10px}
.tyre-bubble{
  width:46px;height:46px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-weight:1000;font-size:22px;background:#07111a;border:3px solid currentColor;box-shadow:0 0 0 1px rgba(255,255,255,.04) inset;
}
.tyre-arrow{color:#7f909c;font-size:22px;font-weight:800}
.tyre-caption{font-size:9px;color:#7f909c;text-transform:uppercase;letter-spacing:.10em;margin-top:2px}
.info-note-box{border:1px solid #1b3341;border-radius:8px;background:#07111a;padding:12px 12px;min-height:52px}
.info-note-box .k{font-size:8px;color:#8798a4;text-transform:uppercase;letter-spacing:.10em}
.info-note-box .v{font-size:12px;font-weight:850;margin-top:5px;color:#dde5eb;line-height:1.35}
.setup-actions{display:flex;justify-content:center;align-items:center;gap:14px;margin:10px 0 2px}
.setup-spacer{height:2px}
.section-grid{margin:10px 0 16px}
.section-controls{padding-top:2px}
.inline-summary{margin-top:12px}
.strategy-inline-grid{display:grid;grid-template-columns:1.35fr 1fr 1fr;gap:10px;margin-top:12px}
.strategy-plan-card,.inline-card{
  border:1px solid #1b3341;border-radius:8px;background:#07111a;padding:11px 12px;min-height:62px;
}
.strategy-plan-card .k,.inline-card .k{font-size:8px;color:#8798a4;text-transform:uppercase;letter-spacing:.10em}
.strategy-plan-card .v,.inline-card .v{font-size:12px;font-weight:850;margin-top:6px;color:#dde5eb;line-height:1.35}
.strategy-plan-card .s,.inline-card .s{font-size:8px;color:#728492;margin-top:4px}
@media(max-width:1200px){
  .setup-info-panel,.setup-controls-card{min-height:auto}
  .summary-col{grid-template-columns:1fr 1fr}
  .strategy-inline-grid{grid-template-columns:1fr}
  .topbar{grid-template-columns:1fr}
  .gp-head{border-left:none;padding-left:0;margin-top:7px}
  .head-meta{text-align:left;margin-top:6px}
  .result-shell{grid-template-columns:1fr}
  .result-flex{grid-template-columns:145px repeat(3,1fr)}
  .result-kpi:nth-child(n+5){display:none}
  .stand-head,.stand-row{grid-template-columns:38px minmax(120px,1.4fr) 48px 70px 58px 58px}
  .stand-team-col{display:none}
}

@media(max-width:768px){
  .block-container{padding:.35rem .45rem 4rem;max-width:100%}
  [data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;gap:.45rem!important}
  [data-testid="column"]{width:100%!important;flex:1 1 100%!important;min-width:100%!important}
  [data-testid="stVerticalBlockBorderWrapper"]{border-radius:8px!important}
  [data-baseweb="select"]>div,[data-testid="stNumberInput"] input{min-height:46px!important}
  [data-testid="stSelectbox"] label,[data-testid="stNumberInput"] label{font-size:10px!important}
  .stButton>button{min-height:44px;font-size:11px;width:100%}
  .topbar{min-height:auto;padding:4px 0 10px;gap:6px}
  .brand-title{font-size:20px}
  .brand-sub{font-size:8px;line-height:1.4}
  .gp-name{font-size:15px}
  .gp-place,.head-meta{font-size:9px}
  .setup-shell{padding:8px 10px 4px}
  .setup-head{flex-direction:column;align-items:flex-start;gap:8px}
  .setup-title{font-size:13px}
  .setup-caption{display:block;margin:4px 0 0 0}
  .scenario-chip{margin:0 6px 6px 0}
  .setup-info-panel{padding:12px 12px 12px 14px}
  .setup-badge{width:38px;height:38px;font-size:18px;margin-bottom:10px}
  .setup-info-title{font-size:13px}
  .setup-info-sub{max-width:none;font-size:9px}
  .setup-card-title{margin-bottom:8px}
  .summary-card .v{font-size:16px}
  .result-flex{grid-template-columns:1fr!important;gap:0;margin-top:6px}
  .result-pos{font-size:58px;line-height:.9;margin-bottom:6px}
  .result-kpi{border-left:none;border-top:1px solid #263946;padding:8px 0 8px 0;min-height:auto}
  .compare-grid{grid-template-columns:1fr;gap:10px;margin-top:12px}
  .compare-delta{border:none;border-top:1px solid #263946;border-bottom:1px solid #263946;padding:10px 0}
  .compare-side .big{font-size:22px}
  .panel-title,.result-title,.compare-title{font-size:12px}
  .metric-grid{grid-template-columns:1fr 1fr}
  .circuit-layout{grid-template-columns:1fr}
  .tyre-bubble{width:40px;height:40px;font-size:19px}
  .tyre-arrow{font-size:18px}
  .strategy-plan-card .v,.inline-card .v,.info-note-box .v{font-size:11px}
  .standings{overflow-x:auto;-webkit-overflow-scrolling:touch}
  .stand-head,.stand-row{min-width:620px}
}

.mode-wrap{margin:6px 0 10px;padding:7px 10px;border:1px solid #1b3341;border-radius:7px;background:#07111a}
.target-hero{display:grid;grid-template-columns:1.05fr 1.6fr;gap:10px;margin:8px 0 10px}
.target-score,.target-status{border:1px solid #1f3442;border-radius:8px;background:linear-gradient(145deg,#08131c,#050b10);padding:14px 16px}
.target-score .k,.target-status .k{font-size:9px;color:#8798a4;text-transform:uppercase;letter-spacing:.11em;font-weight:900}
.target-score .v{font-size:52px;font-weight:1000;color:#ff3546;line-height:1;margin-top:5px}
.target-score .s,.target-status .s{font-size:9px;color:#83939e;margin-top:5px;line-height:1.45}
.target-status .v{font-size:22px;font-weight:1000;margin-top:7px}
.target-card{border:1px solid #1f3442;border-radius:8px;background:#07111a;padding:13px 14px}
.target-card .title{font-size:12px;font-weight:1000;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px}
.target-bigplan{font-size:29px;font-weight:1000;color:#ffd21f;margin:4px 0 10px}
.target-facts{display:grid;grid-template-columns:repeat(2,1fr);gap:7px}
.target-fact{border:1px solid #18313d;border-radius:6px;padding:8px;background:#060e15}
.target-fact .k{font-size:8px;color:#81929e;text-transform:uppercase;letter-spacing:.08em}
.target-fact .v{font-size:13px;font-weight:900;margin-top:4px}
.req-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:8px 0}
.req-card{border:1px solid #1c3340;border-radius:7px;background:#07111a;padding:10px 11px}
.req-card .k{font-size:8px;color:#8798a4;text-transform:uppercase;letter-spacing:.08em}
.req-card .v{font-size:10px;font-weight:850;margin-top:5px;line-height:1.4}
.target-scenarios{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.target-scenario{border:1px solid #1b3341;border-radius:7px;background:#07111a;padding:10px}
.target-scenario .rank{font-size:8px;color:#ff5d69;font-weight:1000;text-transform:uppercase}
.target-scenario .prob{font-size:24px;font-weight:1000;margin-top:4px}
.target-scenario .plan{font-size:12px;font-weight:950;color:#ffd21f;margin-top:4px}
.target-scenario .meta{font-size:8px;color:#84949f;line-height:1.45;margin-top:5px}
@media(max-width:768px){
  .target-hero{grid-template-columns:1fr}
  .req-grid{grid-template-columns:1fr 1fr}
  .target-scenarios{grid-template-columns:1fr}
  .target-score .v{font-size:44px}
  .target-facts{grid-template-columns:1fr 1fr}
}


/* V2.5 compact / A4-focused cockpit */
.block-container{padding:.35rem .65rem 1.0rem;max-width:1500px}
.topbar{min-height:46px;padding:2px 2px 6px;margin-bottom:2px}
.brand-title{font-size:21px;line-height:1}
.brand-sub{font-size:8px}
.gp-name{font-size:15px}
.gp-place,.head-meta,.round-box{font-size:8px}
.mode-wrap{margin:3px 0 6px;padding:4px 8px}
.setup-shell{padding:6px 10px 0;margin:3px 0 6px}
.setup-head{margin-bottom:0}
.setup-title{font-size:12px}
.setup-caption{font-size:8px}
.compact-zone{border:1px solid var(--line);border-radius:8px;background:linear-gradient(145deg,rgba(8,16,24,.97),rgba(5,11,17,.94));padding:8px 10px 6px;margin:5px 0}
.compact-zone-title{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.compact-zone-title .n{width:24px;height:24px;border-radius:8px;display:inline-flex;align-items:center;justify-content:center;font-weight:1000;font-size:13px;color:#fff}
.compact-zone-title .n.blue{background:linear-gradient(180deg,#2f9dff,#1d69cf)}
.compact-zone-title .n.red{background:linear-gradient(180deg,#ff5867,#dc1d31)}
.compact-zone-title .n.yellow{background:linear-gradient(180deg,#ffe14a,#efbe00);color:#11151a}
.compact-zone-title .t{font-size:11px;font-weight:1000;text-transform:uppercase;letter-spacing:.08em}
.compact-zone-title .s{font-size:8px;color:#7f909c}
.compact-divider{height:1px;background:#17303c;margin:6px 0}
.compact-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:6px}
.mini-kpi{border:1px solid #18313d;border-radius:7px;background:#07111a;padding:7px 8px}
.mini-kpi .k{font-size:7px;color:#80919d;text-transform:uppercase;letter-spacing:.08em}
.mini-kpi .v{font-size:14px;font-weight:1000;margin-top:2px}
.mini-kpi .s{font-size:7px;color:#738592;margin-top:1px}

/* radio pills */
[data-testid="stRadio"] [role="radiogroup"]{gap:6px;flex-wrap:wrap}
[data-testid="stRadio"] label{background:#07111a;border:1px solid #203744;border-radius:999px;padding:5px 10px!important;min-height:30px}
[data-testid="stRadio"] label p{font-size:11px!important;font-weight:700!important}
[data-testid="stRadio"] label:has(input:checked){background:rgba(255,38,56,.10);border-color:#ff3042;box-shadow:0 0 0 1px rgba(255,38,56,.22) inset}
[data-testid="stRadio"] label:hover{border-color:#ff3042}

/* tabs more compact */
.stTabs [data-baseweb="tab-list"]{gap:6px}
.stTabs [data-baseweb="tab"]{height:34px;border-radius:999px;background:#07111a;border:1px solid #223847;padding:0 14px}
.stTabs [aria-selected="true"]{background:rgba(255,38,56,.08)!important;border-color:#ff3042!important}

.result-shell{margin:5px 0 7px}
.result-card,.compare-card{padding:9px 11px;min-height:102px}
.result-title,.compare-title{font-size:12px}
.result-flex{margin-top:6px}
.result-pos{font-size:74px}
.result-kpi .v{font-size:20px}
.compare-grid{margin-top:12px}
.summary-card .v{font-size:15px}
.strategy-plan-card,.inline-card,.info-note-box{padding:9px 10px;min-height:54px}
.strategy-plan-card .v,.inline-card .v,.info-note-box .v{font-size:11px}
.panel-tight [data-testid="stVerticalBlockBorderWrapper"]{padding-bottom:0!important}
.footerline{margin-top:6px}

@media(max-width:1300px){
  .compact-kpis{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:768px){
  .block-container{max-width:100%;padding:.25rem .4rem 1.2rem}
  .compact-kpis{grid-template-columns:1fr 1fr}
}



/* V2.6 racing visual layer */
:root{--glow-red:rgba(255,38,56,.28);--glow-blue:rgba(45,156,255,.22);--glow-yellow:rgba(255,210,31,.18)}
[data-testid="stAppViewContainer"]{
  background:
    linear-gradient(rgba(80,115,140,.022) 1px,transparent 1px),
    linear-gradient(90deg,rgba(80,115,140,.022) 1px,transparent 1px),
    linear-gradient(135deg,rgba(255,255,255,.018) 25%,transparent 25%) 0 0/18px 18px,
    linear-gradient(315deg,rgba(255,255,255,.012) 25%,transparent 25%) 0 0/18px 18px,
    radial-gradient(circle at 84% -12%,rgba(255,38,56,.12),transparent 30%),
    radial-gradient(circle at 12% 115%,rgba(45,156,255,.07),transparent 26%),
    #05090d;
}
.topbar.racing-topbar{
  border:1px solid #203543;border-radius:12px;padding:10px 14px 11px;margin-bottom:6px;
  background:
    linear-gradient(90deg,rgba(255,38,56,.08),transparent 18%,transparent 82%,rgba(45,156,255,.08)),
    linear-gradient(145deg,rgba(7,14,21,.98),rgba(5,10,15,.96));
  box-shadow:0 8px 30px rgba(0,0,0,.28), inset 0 0 0 1px rgba(255,255,255,.02);
  position:relative;overflow:hidden;
}
.topbar.racing-topbar:before{
  content:"";position:absolute;left:0;right:0;top:0;height:4px;
  background:linear-gradient(90deg,var(--red) 0 26%, transparent 26% 30%, #ffffff 30% 56%, transparent 56% 60%, var(--cyan) 60% 100%);
  opacity:.9;
}
.brand-block{display:flex;flex-direction:column;gap:3px}
.brand-title{font-size:24px!important;font-style:italic;letter-spacing:.04em;text-shadow:0 0 18px rgba(255,38,56,.12)}
.brand-title span{position:relative}
.brand-title span:after{content:"";position:absolute;left:0;right:-14px;bottom:-3px;height:2px;background:linear-gradient(90deg,var(--red),transparent)}
.brand-sub{letter-spacing:.16em;color:#9eacb7!important}
.start-lights{display:flex;gap:7px;margin-top:4px}
.start-lights .light{width:12px;height:12px;border-radius:50%;background:#1a232c;border:1px solid #3b4c59;box-shadow:inset 0 0 0 1px rgba(255,255,255,.02)}
.start-lights .light.on{background:radial-gradient(circle at 35% 35%,#ff8892 0 15%,#ff3143 40%,#9d0d1a 82%);box-shadow:0 0 10px rgba(255,38,56,.55),0 0 18px rgba(255,38,56,.28)}
.gp-head{border-left:1px solid rgba(255,255,255,.08)!important;padding-left:16px!important}
.round-box{display:inline-flex;align-items:center;gap:6px;background:#08131d;border:1px solid #27404f;border-radius:999px;padding:4px 10px;color:#e9eff3!important}
.round-box:before{content:"⬢";color:var(--red);font-size:9px}
.gp-name{font-size:18px!important;letter-spacing:.04em;text-shadow:0 0 12px rgba(255,255,255,.03)}
.racing-meta{display:flex;justify-content:flex-end;gap:7px;align-items:center;flex-wrap:wrap;text-align:right!important}
.meta-chip{padding:6px 10px;border:1px solid #263c4a;border-radius:999px;background:#08131d;color:#b8c3cb;font-size:8px;font-weight:900;text-transform:uppercase;letter-spacing:.11em}
.meta-chip.accent{color:#ffd7db;border-color:#a12935;background:rgba(255,38,56,.09);box-shadow:0 0 0 1px rgba(255,38,56,.18) inset}
.mode-wrap{border-radius:12px!important;background:linear-gradient(145deg,rgba(8,18,27,.98),rgba(5,11,17,.96))!important;box-shadow:0 8px 26px rgba(0,0,0,.20)}
[data-testid="stRadio"] label{border-radius:999px!important;background:linear-gradient(180deg,#0a141d,#07111a)!important;box-shadow:inset 0 -1px 0 rgba(255,255,255,.02)}
[data-testid="stRadio"] label:has(input:checked){background:linear-gradient(180deg,rgba(255,38,56,.18),rgba(255,38,56,.07))!important;border-color:#ff3042!important;box-shadow:0 0 0 1px rgba(255,38,56,.20) inset, 0 0 18px rgba(255,38,56,.10)}
.stButton>button{
  border-radius:8px!important;
  border:1px solid #314b5b!important;
  background:linear-gradient(180deg,#0c1822,#08121a)!important;
  box-shadow:0 7px 18px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.04);
  letter-spacing:.08em!important;
}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 10px 22px rgba(0,0,0,.28), 0 0 0 1px rgba(255,38,56,.15) inset}
button[kind="primary"]{
  background:linear-gradient(180deg,#ff4253 0%, #ef1d31 45%, #b80b1a 100%)!important;
  border-color:#ff5c6a!important;color:#fff!important;
  box-shadow:0 10px 22px rgba(184,11,26,.35), inset 0 1px 0 rgba(255,255,255,.15)!important;
}
button[kind="primary"]:hover{box-shadow:0 12px 26px rgba(184,11,26,.42),0 0 20px rgba(255,38,56,.22)!important}
[data-testid="stVerticalBlockBorderWrapper"],.result-card,.compare-card,.summary-card,.strategy-plan-card,.inline-card,.info-note-box,.target-card,.target-score,.target-status,.req-card,.target-scenario,.standings{
  position:relative;overflow:hidden;
}
[data-testid="stVerticalBlockBorderWrapper"]:before,.result-card:before,.compare-card:before,.summary-card:before,.strategy-plan-card:before,.inline-card:before,.info-note-box:before,.target-card:before,.target-score:before,.target-status:before,.req-card:before,.target-scenario:before,.standings:before{
  content:"";position:absolute;left:0;top:0;width:100%;height:2px;background:linear-gradient(90deg,var(--red),transparent 55%,var(--cyan));opacity:.85;
}
.panel-title:before{width:4px!important;border-radius:2px;box-shadow:0 0 12px rgba(255,38,56,.32)}
.summary-card .v,.mini-kpi .v,.metric-card .v,.target-fact .v{font-style:italic}
.setup-shell,.compact-zone,.result-card,.compare-card,.target-score,.target-status,.target-card{box-shadow:0 10px 22px rgba(0,0,0,.18), inset 0 0 0 1px rgba(255,255,255,.02)}
.scenario-chip{border-radius:999px!important;padding:6px 10px!important;background:#09131b!important}
.scenario-chip.active{box-shadow:0 0 0 1px rgba(255,38,56,.2) inset,0 0 14px rgba(255,38,56,.08)}
.compact-zone-title .n{box-shadow:0 0 14px rgba(0,0,0,.20)}
.compact-zone-title .n.blue{background:linear-gradient(180deg,#3eb4ff,#1659cf)!important}
.compact-zone-title .n.red{background:linear-gradient(180deg,#ff5c69,#c40a1b)!important}
.compact-zone-title .n.yellow{background:linear-gradient(180deg,#ffe875,#ffbf08)!important}
.summary-card,.mini-kpi,.metric-card,.strategy-plan-card,.inline-card,.info-note-box{background:linear-gradient(180deg,#07111a,#061019)!important}
.tyre-bubble{box-shadow:0 0 0 1px rgba(255,255,255,.04) inset,0 0 18px rgba(255,255,255,.03)}
.tyre-bubble[style*="#ff2638"]{box-shadow:0 0 0 1px rgba(255,255,255,.04) inset,0 0 18px rgba(255,38,56,.15)}
.tyre-bubble[style*="#ffd21f"]{box-shadow:0 0 0 1px rgba(255,255,255,.04) inset,0 0 18px rgba(255,210,31,.12)}
.tyre-bubble[style*="#2d9cff"],.tyre-bubble[style*="#2ed47a"]{box-shadow:0 0 0 1px rgba(255,255,255,.04) inset,0 0 18px rgba(45,156,255,.13)}
.stTabs [data-baseweb="tab-list"]{padding:2px;background:rgba(8,17,25,.8);border:1px solid #203543;border-radius:999px}
.stTabs [data-baseweb="tab"]{height:36px!important;background:transparent!important;border-radius:999px!important;border:1px solid transparent!important;font-weight:900!important;letter-spacing:.05em!important}
.stTabs [aria-selected="true"]{background:linear-gradient(180deg,rgba(255,38,56,.18),rgba(255,38,56,.08))!important;border-color:#ff3042!important;box-shadow:0 0 18px rgba(255,38,56,.10)}
.result-pos{background:linear-gradient(180deg,#fff7f8,#ff6d7a 28%,#ff2638 68%,#bf0e20)!important;-webkit-background-clip:text!important}
.compare-delta .v{color:#50e79b!important;text-shadow:0 0 10px rgba(46,212,122,.20)}
.insight{padding:10px 0!important}
.insight-icon{width:30px;height:30px;display:flex;align-items:center;justify-content:center;border-radius:50%;background:#09131b;border:1px solid #213846}
.stand-row.selected{background:linear-gradient(90deg,rgba(255,210,31,.11),rgba(255,210,31,.03))!important;box-shadow:inset 0 0 0 1px #ffd21f, inset 5px 0 0 #ffd21f!important}
.footerline{font-size:8px;color:#7d8d98!important;text-transform:uppercase;letter-spacing:.11em}
@media(max-width:768px){
  .racing-meta{justify-content:flex-start}
  .gp-name{font-size:16px!important}
  .brand-title{font-size:21px!important}
  .start-lights .light{width:10px;height:10px}
}



/* V2.7 — deeper racing / pit-wall skin */
:root{
  --race-red:#ff2038;
  --race-red-dark:#9f0917;
  --race-yellow:#ffd21f;
  --race-cyan:#39c8ff;
  --race-green:#31df8a;
  --race-panel:#071018;
  --race-panel-2:#0a141e;
}

[data-testid="stAppViewContainer"]{
  background:
    repeating-linear-gradient(135deg, rgba(255,255,255,.013) 0 6px, transparent 6px 12px),
    linear-gradient(rgba(80,115,140,.021) 1px,transparent 1px),
    linear-gradient(90deg,rgba(80,115,140,.021) 1px,transparent 1px),
    radial-gradient(circle at 83% -10%,rgba(255,32,56,.15),transparent 26%),
    radial-gradient(circle at 10% 105%,rgba(57,200,255,.07),transparent 25%),
    #04080c!important;
  background-size:auto,34px 34px,34px 34px,auto,auto,auto!important;
}

.block-container{max-width:1540px!important}

/* top pit-wall masthead */
.racing-topbar{
  clip-path:polygon(0 0,98.9% 0,100% 24%,100% 100%,1.1% 100%,0 76%);
  border-color:#29414f!important;
  box-shadow:0 14px 34px rgba(0,0,0,.34),0 0 0 1px rgba(255,32,56,.035) inset!important;
}
.racing-topbar:after{
  content:"";position:absolute;right:14px;bottom:7px;width:90px;height:10px;opacity:.30;
  background:
    linear-gradient(45deg,#fff 25%,transparent 25%) 0 0/10px 10px,
    linear-gradient(45deg,transparent 75%,#fff 75%) 0 0/10px 10px,
    linear-gradient(45deg,transparent 75%,#fff 75%) 5px -5px/10px 10px,
    linear-gradient(45deg,#fff 25%,transparent 25%) 5px 5px/10px 10px;
}
.brand-title{font-size:25px!important;letter-spacing:.065em!important;text-transform:uppercase;transform:skewX(-7deg)}
.brand-title span{color:#ff2638!important;text-shadow:0 0 16px rgba(255,32,56,.23)}
.brand-sub{font-style:italic;color:#a7b5bf!important}
.start-lights{gap:5px!important}
.start-lights .light{width:11px!important;height:11px!important;border-radius:3px!important;transform:skewX(-8deg)}
.gp-name{font-style:italic;transform:skewX(-4deg);transform-origin:left center}
.meta-chip{border-radius:4px!important;transform:skewX(-6deg);padding:5px 9px!important}
.meta-chip>*{transform:skewX(6deg)}
.meta-chip.accent{background:linear-gradient(180deg,rgba(255,32,56,.22),rgba(140,7,20,.15))!important}

/* mode selector = steering-wheel switch feel */
.mode-wrap{position:relative;overflow:hidden!important}
.mode-wrap:before{content:"MODE SELECT";position:absolute;right:10px;top:6px;font-size:7px;color:#536774;letter-spacing:.16em;font-weight:900}
[data-testid="stRadio"] label{border-radius:5px!important;transform:skewX(-4deg);padding:5px 12px!important}
[data-testid="stRadio"] label p{transform:skewX(4deg);letter-spacing:.035em}
[data-testid="stRadio"] label:has(input:checked){box-shadow:inset 4px 0 0 var(--race-red),0 0 18px rgba(255,32,56,.11)!important}

/* setup / cards */
.setup-shell,.compact-zone,[data-testid="stVerticalBlockBorderWrapper"]{
  border-color:#223a48!important;
}
.setup-shell{position:relative;overflow:hidden}
.setup-shell:after{content:"";position:absolute;right:-28px;top:0;width:110px;height:100%;background:repeating-linear-gradient(135deg,rgba(255,32,56,.06) 0 7px,transparent 7px 14px);pointer-events:none}
.setup-title,.panel-title,.result-title,.compare-title,.target-card .title{font-style:italic;letter-spacing:.08em!important}
.compact-zone-title .t{font-style:italic;font-size:11.5px!important}
.compact-zone-title .n{border-radius:4px!important;transform:skewX(-7deg)}
.compact-zone-title .n.red{box-shadow:0 0 16px rgba(255,32,56,.18)!important}
.compact-zone-title .n.blue{box-shadow:0 0 16px rgba(57,200,255,.15)!important}
.compact-zone-title .n.yellow{box-shadow:0 0 16px rgba(255,210,31,.15)!important}

[data-testid="stVerticalBlockBorderWrapper"]{
  box-shadow:0 8px 20px rgba(0,0,0,.22),inset 0 0 0 1px rgba(255,255,255,.015)!important;
}
[data-testid="stVerticalBlockBorderWrapper"]:after{
  content:"";position:absolute;right:0;top:0;width:34px;height:34px;
  background:linear-gradient(135deg,transparent 49%,rgba(255,32,56,.18) 50%,rgba(255,32,56,.18) 53%,transparent 54%);
  pointer-events:none;
}

/* Input controls closer to race console */
[data-baseweb="select"]>div,[data-testid="stNumberInput"] input{
  border-radius:5px!important;
  background:linear-gradient(180deg,#0b151e,#071019)!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.025)!important;
}
[data-baseweb="select"]>div:focus-within{border-color:#ff3042!important;box-shadow:0 0 0 1px rgba(255,48,66,.12),0 0 14px rgba(255,48,66,.08)!important}

/* Racing CTA */
.stButton>button{font-style:italic!important;clip-path:polygon(0 0,96% 0,100% 30%,100% 100%,4% 100%,0 70%);text-shadow:0 1px 0 rgba(0,0,0,.35)}
button[kind="primary"]{background:linear-gradient(135deg,#ff4254 0%,#ed1830 45%,#9e0716 100%)!important}
button[kind="primary"]:before{content:"";display:inline-block;width:14px;height:8px;margin-right:7px;background:repeating-linear-gradient(90deg,#fff 0 3px,transparent 3px 6px);opacity:.75}

/* Result hero = timing screen */
.result-card,.compare-card,.target-score,.target-status{
  border-top:1px solid #3a5361!important;
  background:linear-gradient(160deg,#09151f,#050a0f 72%)!important;
}
.result-card{box-shadow:inset 5px 0 0 var(--race-red),0 12px 28px rgba(0,0,0,.24)!important}
.result-pos{font-family:Arial Narrow,Arial,sans-serif!important;letter-spacing:-.095em!important;font-style:italic!important;text-shadow:0 0 24px rgba(255,32,56,.14)}
.result-kpi .k,.target-fact .k,.summary-card .k,.mini-kpi .k{letter-spacing:.12em!important}
.result-kpi .v{font-style:italic}
.compare-card{box-shadow:inset 4px 0 0 #39c8ff,0 12px 28px rgba(0,0,0,.20)!important}
.compare-delta{position:relative}
.compare-delta:before{content:"DELTA";display:block;font-size:6px;letter-spacing:.14em;color:#647985;margin-bottom:2px}

/* Tyres = stronger motorsport cues */
.tyre-bubble{background:radial-gradient(circle at 38% 34%,#111d27,#050b10 68%)!important;border-width:4px!important;font-style:italic}
.tyre-arrow{color:#96a7b2!important}
.c-soft,.tyre-bubble[style*="#ff2638"]{filter:drop-shadow(0 0 5px rgba(255,32,56,.25))}
.c-medium,.tyre-bubble[style*="#ffd21f"]{filter:drop-shadow(0 0 5px rgba(255,210,31,.18))}
.c-intermediate{filter:drop-shadow(0 0 5px rgba(49,223,138,.18))}
.c-wet{filter:drop-shadow(0 0 5px rgba(45,156,255,.20))}

/* tabs resemble telemetry pages */
.stTabs [data-baseweb="tab-list"]{border-radius:5px!important;background:#060d13!important;padding:3px!important}
.stTabs [data-baseweb="tab"]{border-radius:4px!important;font-style:italic!important}
.stTabs [aria-selected="true"]{box-shadow:inset 4px 0 0 var(--race-red),0 0 16px rgba(255,32,56,.08)!important}

/* timing tower / classification */
.stand-head{background:linear-gradient(90deg,#101b24,#081018)!important;border-bottom:1px solid #2c4654!important}
.stand-row{transition:background .15s ease}
.stand-row:hover{background:#0a151e!important}
.stand-pos{font-style:italic;font-size:14px!important}
.stand-driver{letter-spacing:.03em}
.stand-row.selected .stand-driver{color:#ffe777}

/* target outcome */
.target-score{box-shadow:inset 5px 0 0 var(--race-red),0 12px 28px rgba(0,0,0,.22)!important}
.target-score .v{font-style:italic;text-shadow:0 0 18px rgba(255,32,56,.18)}
.target-status{box-shadow:inset 5px 0 0 var(--race-yellow)!important}
.target-scenario{clip-path:polygon(0 0,97% 0,100% 11%,100% 100%,3% 100%,0 89%)}
.target-scenario .rank{font-style:italic;letter-spacing:.12em}
.target-scenario .plan{font-style:italic}

/* subtle dynamic effect: start lights + selected controls */
@keyframes racePulse{0%,100%{filter:brightness(1)}50%{filter:brightness(1.22)}}
@keyframes redSweep{0%{transform:translateX(-130%)}100%{transform:translateX(160%)}}
.start-lights .light.on:nth-child(1){animation:racePulse 2.6s ease-in-out infinite .05s}
.start-lights .light.on:nth-child(2){animation:racePulse 2.6s ease-in-out infinite .15s}
.start-lights .light.on:nth-child(3){animation:racePulse 2.6s ease-in-out infinite .25s}
.start-lights .light.on:nth-child(4){animation:racePulse 2.6s ease-in-out infinite .35s}
.start-lights .light.on:nth-child(5){animation:racePulse 2.6s ease-in-out infinite .45s}
button[kind="primary"]{position:relative;overflow:hidden}
button[kind="primary"]:after{content:"";position:absolute;top:0;bottom:0;width:35%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.16),transparent);animation:redSweep 3.6s ease-in-out infinite}

/* Checkered micro-strip on footer */
.footerline:before{content:"";display:inline-block;width:72px;height:8px;margin-right:10px;vertical-align:-1px;opacity:.35;background:conic-gradient(#fff 25%,transparent 0 50%,#fff 0 75%,transparent 0) 0 0/8px 8px}

@media(max-width:768px){
  .racing-topbar{clip-path:none}
  .racing-topbar:after{display:none}
  .brand-title{transform:none!important}
  .gp-name{transform:none!important}
  .meta-chip{transform:none!important}
  .stButton>button{clip-path:none}
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

@st.cache_data(ttl=21600, show_spinner=False)
def official_driver_standings():
    return F1OfficialClient().driver_standings(CURRENT_YEAR)


def target_driver_prior_from_standings(target_analysis, selected_driver: str, event_key: str):
    """Inject a driver-specific prior only when weekend/FastF1 evidence is unavailable."""
    context = str((target_analysis or {}).get("target_context") or "generic_fallback")
    if context != "generic_fallback":
        return target_analysis

    standings = official_driver_standings()
    if not standings:
        return target_analysis

    grid_rows = F1OfficialClient().event_grid_prior(event_key, standings)
    if not grid_rows:
        return target_analysis

    max_points = max(float(r.get("points") or 0.0) for r in grid_rows) or 1.0
    grid_model = []
    selected = None
    for row in grid_rows:
        rank = int(row.get("position") or row.get("grid_position") or 22)
        points = float(row.get("points") or 0.0)
        deficit = max(0.0, 1.0 - points / max_points)
        # Season-strength prior: enough spread to preserve genuine driver/team differences
        # without pretending championship points are literal lap-time measurements.
        pace_delta = float(min(1.65, 0.48 * deficit + 0.012 * max(0, rank - 1)))
        item = {
            "driver_name": str(row.get("driver") or ""),
            "abbreviation": "",
            "team_name": str(row.get("team") or ""),
            "grid_position": int(row.get("grid_position") or rank),
            "race_pace_delta": pace_delta,
            "pace_confidence": "low",
            "grid_confidence": "medium" if event_key == "spain" and str(row.get("driver")) in {"Lando Norris","Kimi Antonelli","Max Verstappen","Lewis Hamilton","Charles Leclerc","George Russell","Oscar Piastri","Liam Lawson","Franco Colapinto","Arvid Lindblad"} else "low",
        }
        grid_model.append(item)
        if item["driver_name"].strip().lower() == selected_driver.strip().lower():
            selected = item

    if selected is None:
        return target_analysis

    out = dict(target_analysis or {})
    out.update({
        "available": True,
        "practice_name": "Current-season strength prior",
        "team_name": selected["team_name"],
        "grid_position": selected["grid_position"],
        "grid_confidence": selected["grid_confidence"],
        "pace_delta": selected["race_pace_delta"],
        "pace_confidence": "low",
        "grid_model": grid_model,
        "grid_model_confidence": "low",
        "target_context": "official_standings_prior",
    })
    return out


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

event_state_key="event_main_idx"
if event_state_key not in st.session_state:
    st.session_state[event_state_key]=min(def_idx,len(calendar)-1)
event_idx=int(st.session_state.get(event_state_key,min(def_idx,len(calendar)-1)))
event_idx=max(0,min(event_idx,len(calendar)-1))
event=calendar[event_idx]
details=official_event(event["key"],event)
race_laps=int(details.get("number_of_laps") or 57)

# Header - intentionally no F1 logo and no visible data-source labels.
st.markdown(
    f"""
    <div class="topbar racing-topbar">
      <div class="brand-block">
        <div class="brand-title"><span>STRATEGY</span> ENGINE</div>
        <div class="brand-sub">Race simulator · pit-wall decision support</div>
        <div class="start-lights" aria-hidden="true">
          <span class="light on"></span><span class="light on"></span><span class="light on"></span><span class="light on"></span><span class="light on"></span>
        </div>
      </div>
      <div class="gp-head">
        <div class="round-box">Round {event.get("round","—")}</div>
        <div>
          <div class="gp-name">{details.get("name",event.get("name","Grand Prix"))}</div>
          <div class="gp-place">{details.get("location",event.get("location",""))} · {event.get("dates","")}</div>
        </div>
      </div>
      <div class="head-meta racing-meta">
        <div class="meta-chip">Current season</div>
        <div class="meta-chip">Pre-race model</div>
        <div class="meta-chip accent">Attack mode</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------
# APP MODE
# ----------------------------------------------------------------
st.markdown('<div class="mode-wrap"><div style="font-size:9px;color:#80919d;text-transform:uppercase;letter-spacing:.10em;margin-bottom:4px">Mode</div></div>', unsafe_allow_html=True)
app_mode_ui=st.radio(
    "Mode",
    ["🏎️ Simulate race","🎯 Target outcome"],
    horizontal=True,
    label_visibility="collapsed",
    key="app_mode_ui",
)
app_mode = "SIMULATE RACE" if app_mode_ui.startswith("🏎️") else "TARGET OUTCOME"

if app_mode=="TARGET OUTCOME":
    st.markdown(
        '<div class="setup-shell"><div class="setup-head">'
        '<div><span class="setup-title">Target outcome</span>'
        '<span class="setup-caption">Choose the result you want to reach. The engine searches the conditions and strategy that make it most achievable.</span></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    target_cols=st.columns([1.55,1.25,1.15,.90],gap="small",vertical_alignment="bottom")
    with target_cols[0]:
        target_event_idx=st.selectbox(
            "1. Grand Prix",
            range(len(calendar)),
            index=event_idx,
            format_func=lambda i:f'R{calendar[i].get("round","—")} · {calendar[i].get("name","Grand Prix")}',
            key=event_state_key,
        )
    with target_cols[1]:
        target_driver=st.selectbox("2. Driver",drivers,index=default_driver_idx,key="target_driver")
    with target_cols[2]:
        target_goal_ui=st.radio(
            "3. Target",
            ["🥇 Win","🏆 Podium","⭐ Top 5","✅ Points"],
            horizontal=True,
            key="target_goal_ui",
        )
        target_goal={"🥇 Win":"WIN","🏆 Podium":"PODIUM","⭐ Top 5":"TOP5","✅ Points":"POINTS"}[target_goal_ui]
    with target_cols[3]:
        target_run=st.button("🎯 Find path",use_container_width=True,type="primary")

    target_key=f'target-v3:{CURRENT_YEAR}:{event["key"]}:{target_driver}:{target_goal}'

    if target_run:
        with st.spinner(f"Searching realistic paths to {target_goal} for {target_driver}…"):
            try:
                target_analysis=fastf1_data.analyse_target_context(CURRENT_YEAR, int(event.get("round") or 1), target_driver)
            except Exception:
                target_analysis={"available":False,"degradation":{},"pace_delta":0.0,"inventory":{},"grid_position":None,"grid_model":[],"target_context":"generic_fallback"}

            target_analysis=target_driver_prior_from_standings(target_analysis,target_driver,event["key"])
            if str((target_analysis or {}).get("target_context")) == "generic_fallback":
                st.error("Driver-specific performance context is unavailable. Target Outcome will not use a generic identical-driver fallback. Try again when current-season data are reachable.")
                st.session_state.pop("target_result",None)
                st.stop()

            target_race_dt=parse_race_datetime(details)
            try:
                target_geo=meteo.geocode(
                    details.get("location",event.get("location","")),
                    details.get("country",event.get("country","")),
                )
                target_weather=meteo.forecast_at(target_geo["latitude"],target_geo["longitude"],target_race_dt)
            except Exception:
                target_weather={}

            t_air=float(target_weather.get("temperature_2m",25.0))
            t_track=float(target_weather.get("track_temperature_estimate",t_air+15.0))
            t_rain=float(target_weather.get("precipitation_probability",5.0) or 0)/100.0

            circuit_type_target=details.get("circuit_type","Permanent")
            if circuit_type_target=="Street":
                t_overtaking,t_sc,t_pit=.82,.48,23.0
            elif circuit_type_target=="Semi-permanent":
                t_overtaking,t_sc,t_pit=.70,.40,23.5
            else:
                t_overtaking,t_sc,t_pit=.56,.31,22.0

            t_grid=int((target_analysis or {}).get("grid_position") or 10)
            t_pace=float((target_analysis or {}).get("pace_delta") or 0.0)
            t_incident=float((target_analysis or {}).get("incident_risk") or t_sc)
            t_sc=min(.75,max(.12,.55*t_sc+.45*t_incident))
            t_deg=(target_analysis or {}).get("degradation",{})
            t_soft=float(t_deg.get("SOFT",.12)); t_med=float(t_deg.get("MEDIUM",.08)); t_hard=float(t_deg.get("HARD",.055))
            t_undercut=min(.95,.48+t_overtaking*.35+max(0,t_med-.06)*.7)
            t_inventory=(target_analysis or {}).get("inventory") or {
                "SOFT":{"new":2,"used":1},"MEDIUM":{"new":2,"used":1},"HARD":{"new":2,"used":1},
            }
            t_inventory.setdefault("INTERMEDIATE",{"new":4,"used":0})
            t_inventory.setdefault("WET",{"new":3,"used":0})
            t_tyres={
                "SOFT":TyreModel("SOFT",-.55,t_soft),
                "MEDIUM":TyreModel("MEDIUM",0.0,t_med),
                "HARD":TyreModel("HARD",.45,t_hard),
                "INTERMEDIATE":TyreModel("INTERMEDIATE",.05,.035),
                "WET":TyreModel("WET",.25,.022),
            }
            target_inputs=SimulationInputs(
                CircuitProfile(race_laps,t_pit,t_pit*.55,t_overtaking,t_undercut,t_track),
                DriverContext(target_driver,(target_analysis or {}).get("team_name",""),t_grid,t_pace),
                t_tyres,t_inventory,t_sc,t_rain,30000,
                mandatory_race_compounds=("HARD","MEDIUM"),
                rivals=(target_analysis or {}).get("grid_model") or None,
                neutralisation_mode="NONE",
                allowed_compounds=("SOFT","MEDIUM","HARD","INTERMEDIATE","WET"),
                weather_mode="EXPECTED",
                weather_timeline=[{
                    "start_lap":1,"end_lap":race_laps,"mode":"EXPECTED",
                    "track_temp":t_track,"rain_probability":t_rain,
                }],
                neutralisation_lap=None,
            )
            try:
                target_result=find_target_outcome(target_inputs,target_goal)
                st.session_state["target_result"]=target_result
                st.session_state["target_result_key"]=target_key
            except Exception as exc:
                st.error(f"Target search could not be completed: {exc}")
                st.session_state.pop("target_result",None)

    target_result=st.session_state.get("target_result")
    if st.session_state.get("target_result_key")!=target_key:
        target_result=None

    if target_result is None:
        with st.container(border=True):
            panel_title("What this mode does")
            st.markdown(
                "The engine keeps the driver's current-weekend pace and grid evidence fixed, then searches plausible weather evolution, Safety Car / VSC timing, tyre strategy and pit-stop timing. The strongest candidates are confirmed with 30,000 simulated races."
            )
    else:
        best=target_result["best_case"]
        robust=target_result.get("robust_path") or {}
        goal_label={"WIN":"WIN","PODIUM":"PODIUM","TOP5":"TOP 5","POINTS":"POINTS"}[target_goal]
        goal_range={"WIN":"P1 only","PODIUM":"P1–P3","TOP5":"P1–P5","POINTS":"P1–P10"}[target_goal]
        prob_bundle=target_result.get("best_case_probabilities",{})
        st.markdown(
            f'''
            <div class="target-hero">
              <div class="target-score">
                <div class="k">{target_driver} · chance of {goal_label}</div>
                <div class="v">{target_result["max_achievable_probability"]:.0%}</div>
                <div class="s">Probability of finishing {goal_range} in the best confirmed scenario. This percentage is not the projected finishing position.</div>
              </div>
              <div class="target-status">
                <div class="k">Assessment</div>
                <div class="v">{target_result["status"]}</div>
                <div class="s">{target_result["status_text"]}</div>
              </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        baseline_context=(target_analysis or {}).get("target_context","current_weekend") if "target_analysis" in locals() else "cached"
        baseline_label={"current_weekend":"Weekend evidence","current_season_prior":"FastF1 season prior","official_standings_prior":"Season strength prior","generic_fallback":"Generic fallback","cached":"Saved search"}.get(baseline_context,"Driver-specific model")
        baseline_grid = f"P{t_grid}" if "t_grid" in locals() else "—"
        baseline_pace = f"+{t_pace:.2f}s/lap" if "t_pace" in locals() else "—"
        st.markdown(
            f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:-2px 0 8px">'
            f'<span class="scenario-chip active">{target_driver}</span>'
            f'<span class="scenario-chip">Baseline {baseline_grid}</span>'
            f'<span class="scenario-chip">Pace {baseline_pace}</span>'
            f'<span class="scenario-chip">{baseline_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        target_main=st.columns([1.15,1.0],gap="small")
        with target_main[0]:
            pits_best=' / '.join('L'+str(x) for x in best["pit_laps"])
            st.markdown(
                f'''
                <div class="target-card">
                  <div class="title">Best path to target</div>
                  <div class="target-bigplan">{best["strategy"]}</div>
                  <div class="target-facts">
                    <div class="target-fact"><div class="k">🥇 Win · P1</div><div class="v">{prob_bundle.get("WIN",best["win_probability"]):.0%}</div></div>
                    <div class="target-fact"><div class="k">🏆 Podium · P1–P3</div><div class="v">{prob_bundle.get("PODIUM",best["podium_probability"]):.0%}</div></div>
                    <div class="target-fact"><div class="k">⭐ Top 5 · P1–P5</div><div class="v">{prob_bundle.get("TOP5",best["top5_probability"]):.0%}</div></div>
                    <div class="target-fact"><div class="k">✅ Points · P1–P10</div><div class="v">{prob_bundle.get("POINTS",best["points_probability"]):.0%}</div></div>
                    <div class="target-fact"><div class="k">Expected finish</div><div class="v">P{best["expected_finish"]:.1f}</div></div>
                    <div class="target-fact"><div class="k">Pit laps</div><div class="v">{pits_best}</div></div>
                    <div class="target-fact"><div class="k">Race control</div><div class="v">{best["neutralisation_label"]}</div></div>
                    <div class="target-fact"><div class="k">Weather</div><div class="v">{best["weather_label"]}</div></div>
                  </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
        with target_main[1]:
            if robust:
                robust_pits=' / '.join('L'+str(x) for x in robust.get("typical_pit_laps",[])) or '—'
                st.markdown(
                    f'''
                    <div class="target-card">
                      <div class="title">Most robust path</div>
                      <div class="target-bigplan">{robust.get("strategy","—")}</div>
                      <div class="target-facts">
                        <div class="target-fact"><div class="k">Average target chance</div><div class="v">{robust.get("robust_probability",0):.0%}</div></div>
                        <div class="target-fact"><div class="k">Downside case</div><div class="v">{robust.get("downside_probability",0):.0%}</div></div>
                        <div class="target-fact"><div class="k">Typical pit laps</div><div class="v">{robust_pits}</div></div>
                        <div class="target-fact"><div class="k">Scenario coverage</div><div class="v">{robust.get("coverage",0):.0%}</div></div>
                      </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

        reqs=target_result.get("minimum_requirements",[])
        if reqs:
            panel_title("Minimum requirements")
            req_html=''.join(
                f'<div class="req-card"><div class="k">{r["name"]}</div><div class="v">{r["value"]}</div></div>'
                for r in reqs
            )
            st.markdown(f'<div class="req-grid">{req_html}</div>',unsafe_allow_html=True)

        sens=target_result.get("sensitivity",[])
        if sens:
            sensitivity_col,scenario_col=st.columns([1.0,1.25],gap="small")
            with sensitivity_col:
                with st.container(border=True):
                    panel_title("What changes the target probability")
                    sdf=pd.DataFrame(sens).sort_values("impact_pp")
                    sf=go.Figure(go.Bar(
                        x=sdf["impact_pp"],y=sdf["factor"],orientation="h",
                        marker_color=["#2ed47a" if x>=0 else "#ff4254" for x in sdf["impact_pp"]],
                        hovertemplate="%{y}: %{x:+.1f} pp<extra></extra>",
                    ))
                    sf.update_layout(
                        height=330,margin=dict(l=10,r=12,t=8,b=25),
                        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#dce5eb",size=9),
                        xaxis=dict(title="Impact on target probability (pp)",gridcolor="#17303c",zeroline=True,zerolinecolor="#6b7b86"),
                        yaxis=dict(title="",automargin=True),showlegend=False,
                    )
                    st.plotly_chart(sf,use_container_width=True,config={"displayModeBar":False})
            with scenario_col:
                with st.container(border=True):
                    panel_title("Top scenario paths")
                    rows=target_result.get("top_scenarios",[])[:3]
                    html=[]
                    for i,row in enumerate(rows,1):
                        pits=' / '.join('L'+str(x) for x in row["pit_laps"])
                        html.append(
                            f'<div class="target-scenario"><div class="rank">#{i} path</div>'
                            f'<div class="prob">{row["target_probability"]:.0%}</div>'
                            f'<div class="plan">{row["strategy"]}</div>'
                            f'<div class="meta">Pit {pits}<br>{row["weather_label"]}<br>{row["neutralisation_label"]}</div></div>'
                        )
                    st.markdown('<div class="target-scenarios">'+''.join(html)+'</div>',unsafe_allow_html=True)

        st.caption(
            f'Search: {target_result["coarse_candidates_evaluated"]} candidate plans. The leading paths were re-run with {target_result["final_simulations"]:,} Monte Carlo races. The result describes scenario-dependent probability, not a guarantee.'
        )

    st.stop()

# ----------------------------------------------------------------
# SCENARIO SETUP
# ----------------------------------------------------------------
st.markdown(
    '<div class="setup-shell"><div class="setup-head">'
    '<div><span class="setup-title">Race scenario setup</span>'
    '<span class="setup-caption">Everything important stays above the fold: define conditions, strategy and race control, then run the model.</span></div>'
    '<div><span class="scenario-chip active">A</span><span class="scenario-chip">B</span><span class="scenario-chip">C</span></div>'
    '</div></div>',
    unsafe_allow_html=True,
)

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
WEATHER_ICONS={
    "EXPECTED":"📡","DRY":"☀️","HOT_DRY":"🔥","COOL_DRY":"🧊",
    "CHANGEABLE":"🌦️","RAIN":"🌧️","HEAVY_RAIN":"⛈️",
}
WEATHER_UI={k:f'{WEATHER_ICONS[k]} {WEATHER_LABELS[k]}' for k in WEATHER_OPTIONS}
PROFILE_UI={"STATIC":"1 phase","2_PHASES":"2 phases","3_PHASES":"3 phases"}
PROFILE_INV={v:k for k,v in PROFILE_UI.items()}

TYRE_ACCENT={"SOFT":"#ff2638","MEDIUM":"#ffd21f","HARD":"#eef2f4","INTERMEDIATE":"#2ed47a","WET":"#2d9cff"}
TYRE_SHORT={"SOFT":"S","MEDIUM":"M","HARD":"H","INTERMEDIATE":"I","WET":"W"}
TYRE_ICON={"SOFT":"🔴","MEDIUM":"🟡","HARD":"⚪","INTERMEDIATE":"🟢","WET":"🔵"}

def weather_values(mode):
    if mode=="EXPECTED":
        return dict(air=expected_air,track=expected_track,rain=expected_rain,wind=expected_wind,humidity=expected_humidity,source="Hidden")
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

def summary_card(label, value, sub=""):
    st.markdown(
        f'<div class="summary-card"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )

def note_box(label, value):
    st.markdown(
        f'<div class="info-note-box"><div class="k">{label}</div><div class="v">{value}</div></div>',
        unsafe_allow_html=True,
    )

def inline_card(label, value, sub=""):
    st.markdown(
        f'<div class="inline-card"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )

def strategy_plan_html(compounds):
    pieces=[]
    for i,c in enumerate(compounds):
        pieces.append(f'<span class="tyre-bubble" style="color:{TYRE_ACCENT[c]}">{TYRE_SHORT[c]}</span>')
        if i < len(compounds)-1:
            pieces.append('<span class="tyre-arrow">→</span>')
    return ''.join(pieces)

def mini_kpi(label,value,sub=""):
    st.markdown(f'<div class="mini-kpi"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)

# WEATHER
with st.container(border=True):
    st.markdown('<div class="compact-zone-title"><span class="n blue">1</span><div><div class="t">Weather conditions</div><div class="s">Circuit + weather evolution</div></div></div>', unsafe_allow_html=True)
    top_weather=st.columns([1.55,.85,1.10,.75,1.10,.75,1.10], gap="small", vertical_alignment="bottom")
    with top_weather[0]:
        event_idx=st.selectbox(
            "Grand Prix / Circuit",
            range(len(calendar)),
            index=event_idx,
            format_func=lambda i:f'R{calendar[i].get("round","—")} · {calendar[i].get("name","Grand Prix")}',
            key=event_state_key,
        )
    with top_weather[1]:
        weather_profile_ui=st.selectbox("Weather layout", list(PROFILE_UI.values()), index=2, key=f"wpui:{event['key']}")
        weather_profile=PROFILE_INV[weather_profile_ui]
    with top_weather[2]:
        phase1=st.selectbox("Phase 1",WEATHER_OPTIONS,index=1,format_func=lambda x:WEATHER_UI[x],key=f"p1:{event['key']}")
    if weather_profile in {"2_PHASES","3_PHASES"}:
        switch1_options=list(range(4,max(5,race_laps-5)))
        default_switch1=max(4,min(race_laps-6,int(round(race_laps*.38))))
        with top_weather[3]:
            switch1=st.selectbox("Phase 2 starts",switch1_options,index=switch1_options.index(default_switch1),format_func=lambda x:f"L{x}",key=f"sw1:{event['key']}:{weather_profile}")
        with top_weather[4]:
            phase2=st.selectbox("Phase 2",WEATHER_OPTIONS,index=5,format_func=lambda x:WEATHER_UI[x],key=f"p2:{event['key']}:{weather_profile}")
    else:
        switch1=None; phase2=None
        with top_weather[3]: st.selectbox("Phase 2 starts", ["—"], disabled=True)
        with top_weather[4]: st.selectbox("Phase 2", ["—"], disabled=True)
    if weather_profile=="3_PHASES":
        switch2_options=list(range(switch1+4,max(switch1+5,race_laps-2)))
        default_switch2=max(switch1+4,min(race_laps-2,int(round(race_laps*.70))))
        with top_weather[5]:
            switch2=st.selectbox("Phase 3 starts",switch2_options,index=switch2_options.index(default_switch2),format_func=lambda x:f"L{x}",key=f"sw2:{event['key']}:{switch1}")
        with top_weather[6]:
            phase3=st.selectbox("Phase 3",WEATHER_OPTIONS,index=6,format_func=lambda x:WEATHER_UI[x],key=f"p3:{event['key']}:{switch1}")
    else:
        switch2=None; phase3=None
        with top_weather[5]: st.selectbox("Phase 3 starts", ["—"], disabled=True)
        with top_weather[6]: st.selectbox("Phase 3", ["—"], disabled=True)

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
weather_compact=' • '.join([f'{WEATHER_ICONS[p["mode"]]} L{p["start_lap"]}–L{p["end_lap"]}' for p in weather_timeline])

with st.container():
    weather_summary=st.columns(4,gap="small")
    with weather_summary[0]: summary_card("Track temp",f"{track_temp:.0f}°C","weighted")
    with weather_summary[1]: summary_card("Rain",f"{rain_prob:.0%}","race average")
    with weather_summary[2]: summary_card("Air temp",f"{air_temp:.0f}°C","ambient")
    with weather_summary[3]: summary_card("Weather map",weather_compact,"scenario flow")

wet_tyres_enabled=any(
    p["mode"] in {"CHANGEABLE","RAIN","HEAVY_RAIN"} or (p["mode"]=="EXPECTED" and p["rain_probability"]>=.20)
    for p in weather_timeline
)
available_compounds=["SOFT","MEDIUM","HARD"]+(["INTERMEDIATE","WET"] if wet_tyres_enabled else [])
phase1_mode=weather_timeline[0]["mode"]
default_start="WET" if phase1_mode=="HEAVY_RAIN" else "INTERMEDIATE" if phase1_mode=="RAIN" else "MEDIUM"
if default_start not in available_compounds: default_start="MEDIUM"

simulations=SIMULATION_RUNS

# STRATEGY
with st.container(border=True):
    st.markdown('<div class="compact-zone-title"><span class="n red">2</span><div><div class="t">Driver strategy</div><div class="s">Driver + stops + tyre sequence</div></div></div>', unsafe_allow_html=True)
    strategy_section=st.columns([1.45,.72,.82,.82,.82,.78,.78], gap="small", vertical_alignment="bottom")
    with strategy_section[0]:
        selected_driver=st.selectbox("Driver",drivers,index=default_driver_idx,key="driver_main")
    with strategy_section[1]:
        stops_ui=st.radio("Stops",["1 stop","2 stops"],horizontal=True,key="stops_ui")
        stops=1 if stops_ui.startswith("1") else 2

analysis_key=f"{CURRENT_YEAR}:{event['key']}:{selected_driver}"
analysis_data=st.session_state.get("analysis_data") if st.session_state.get("analysis_key")==analysis_key else None

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

neutralisation_mode=st.session_state.get(f"rc:{event['key']}", "NONE")
neutralisation_lap=st.session_state.get(f"rclap_store:{event['key']}")

provisional_inputs=SimulationInputs(
    CircuitProfile(race_laps,pit_loss,pit_loss*.55,overtaking,undercut,track_temp),
    DriverContext(selected_driver,(analysis_data or {}).get("team_name",""),grid,pace_delta),
    tyres,inventory,sc_prob,rain_prob,simulations,
    mandatory_race_compounds=("HARD","MEDIUM"),
    rivals=(analysis_data or {}).get("grid_model") or None,
    neutralisation_mode=neutralisation_mode,allowed_compounds=tuple(available_compounds),
    weather_mode=phase1_mode,weather_timeline=weather_timeline,neutralisation_lap=neutralisation_lap,
)

fmt_tyre=lambda x:f'{TYRE_ICON.get(x,"⚪")} {x.title()}'
with strategy_section[2]:
    start_compound=st.selectbox("Start tyre",available_compounds,index=available_compounds.index(default_start),format_func=fmt_tyre,key=f"start:{event['key']}:{phase1_mode}")

s2_options=legal_next_compounds([start_compound],stops+1,provisional_inputs) or available_compounds
with strategy_section[3]:
    stint2=st.selectbox("Stint 2",s2_options,format_func=fmt_tyre,key=f"s2:{analysis_key}:{weather_profile}:{start_compound}:{stops}")
if stops==2:
    s3_options=legal_next_compounds([start_compound,stint2],3,provisional_inputs) or available_compounds
    with strategy_section[4]:
        stint3=st.selectbox("Stint 3",s3_options,format_func=fmt_tyre,key=f"s3:{analysis_key}:{weather_profile}:{start_compound}:{stint2}")
else:
    stint3=None
    with strategy_section[4]: st.selectbox("Stint 3", ["—"], disabled=True)

pit1_max=max(4,race_laps-6 if stops==2 else race_laps-2)
pit1_default=max(3,min(pit1_max,int(round(race_laps*.38))))
pit1_options=list(range(3,pit1_max+1))
with strategy_section[5]:
    pit_lap_1=st.selectbox("Pit 1",pit1_options,index=pit1_options.index(pit1_default),format_func=lambda x:f"L{x}",key=f"pit1:{analysis_key}:{stops}")
if stops==2:
    pit2_min=pit_lap_1+3; pit2_max=max(pit2_min,race_laps-2); pit2_options=list(range(pit2_min,pit2_max+1))
    pit2_default=max(pit2_min,min(pit2_max,int(round(race_laps*.70))))
    with strategy_section[6]:
        pit_lap_2=st.selectbox("Pit 2",pit2_options,index=pit2_options.index(pit2_default),format_func=lambda x:f"L{x}",key=f"pit2:{analysis_key}:{pit_lap_1}")
else:
    pit_lap_2=None
    with strategy_section[6]: st.selectbox("Pit 2", ["—"], disabled=True)

selected_compounds_preview=[start_compound,stint2]+([stint3] if stops==2 else [])
stint_text = f'L1–L{pit_lap_1} · ' + (f'L{pit_lap_1+1}–L{pit_lap_2} · L{pit_lap_2+1}–L{race_laps}' if stops==2 else f'L{pit_lap_1+1}–L{race_laps}')
strategy_preview=st.columns([1.5,1.15,1.0],gap="small")
with strategy_preview[0]:
    st.markdown(
        f'<div class="strategy-plan-card"><div class="k">Your strategy</div><div class="v" style="margin-top:8px">{strategy_plan_html(selected_compounds_preview)}</div><div class="s">Tyre sequence</div></div>',
        unsafe_allow_html=True,
    )
with strategy_preview[1]:
    inline_card("Stint windows", stint_text, "planned usage")
with strategy_preview[2]:
    inline_card("Plan type", f'{stops} stop' + ('s' if stops==2 else '') + ' · {selected_driver}', "current choice")

# SAFETY / VSC
with st.container(border=True):
    st.markdown('<div class="compact-zone-title"><span class="n yellow">3</span><div><div class="t">Safety Car / VSC</div><div class="s">Neutralisation type and timing</div></div></div>', unsafe_allow_html=True)
    safety_section=st.columns([1.1,.9,2.2], gap="small", vertical_alignment="bottom")
    with safety_section[0]:
        neutralisation_ui=st.radio("Safety / VSC",["🟢 None","🚗 Safety Car","⚠️ VSC"],horizontal=True,key=f"rcui:{event['key']}")
        neutralisation_mode={"🟢 None":"NONE","🚗 Safety Car":"SC","⚠️ VSC":"VSC"}[neutralisation_ui]
        st.session_state[f"rc:{event['key']}"]=neutralisation_mode
    if neutralisation_mode in {"SC","VSC"}:
        neutral_opts=["RANDOM"]+list(range(2,max(3,race_laps-1)))
        with safety_section[1]:
            neutral_choice=st.selectbox("Event lap",neutral_opts,format_func=lambda x:"Random" if x=="RANDOM" else f"L{x}",key=f"rclap:{event['key']}:{neutralisation_mode}")
        neutralisation_lap=None if neutral_choice=="RANDOM" else int(neutral_choice)
    else:
        neutralisation_lap=None
        with safety_section[1]: st.selectbox("Event lap", ["—"], disabled=True)
    st.session_state[f"rclap_store:{event['key']}"]=neutralisation_lap
    note_text = (
        "Green-flag race assumed unless the scenario changes"
        if neutralisation_mode=="NONE" else
        f'{"Safety Car" if neutralisation_mode=="SC" else "Virtual Safety Car"} sampled at random timing'
        if neutralisation_lap is None else
        f'{"Safety Car" if neutralisation_mode=="SC" else "Virtual Safety Car"} triggered on lap {neutralisation_lap}'
    )
    with safety_section[2]:
        note_box("Race control note", note_text)

    safety_summary=st.columns(3,gap="small")
    items = [
        ("Neutralisation", {"NONE":"None","SC":"Safety Car","VSC":"VSC"}[neutralisation_mode], "event type"),
        ("Timing", "Random" if neutralisation_mode!="NONE" and neutralisation_lap is None else (f"L{neutralisation_lap}" if neutralisation_lap is not None else "—"), "trigger lap"),
        ("Pit-loss effect", "Reduced" if neutralisation_mode!="NONE" else "Normal", "during neutralisation"),
    ]
    for c,it in zip(safety_summary, items):
        with c: summary_card(*it)

selected_compounds=[start_compound,stint2]+([stint3] if stops==2 else [])
selected_pit_laps=[pit_lap_1]+([pit_lap_2] if stops==2 else [])
short_name={"SOFT":"S","MEDIUM":"M","HARD":"H","INTERMEDIATE":"I","WET":"W"}
neutral_label={"NONE":"NO SC / VSC","SC":"SAFETY CAR","VSC":"VIRTUAL SAFETY CAR"}[neutralisation_mode]
valid_now,rule_reasons=validate_strategy(selected_compounds,provisional_inputs)

# Main actions.
act=st.columns([3.2,1.65,1.65,3.2],gap="small")
with act[1]:
    simulate_clicked=st.button("▶ Simulate race",use_container_width=True,type="primary")
with act[2]:
    optimal_clicked=st.button("⚡ Find optimum",use_container_width=True)

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
# DETAIL TABS (compact, no long scrolling)
# ----------------------------------------------------------------
detail_tabs=st.tabs(["🏁 Overview","📈 Timeline","🌦️ Weather & tyres","📋 Final classification"])

with detail_tabs[0]:
    ov=st.columns([1.05,.95,1.0],gap="small")
    with ov[0]:
        with st.container(border=True):
            panel_title("Circuit info")
            st.markdown(f'<div style="font-size:10px;color:#b9c4cc;margin-bottom:5px">{details.get("location",event.get("location",""))}</div>',unsafe_allow_html=True)
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
    with ov[1]:
        with st.container(border=True):
            panel_title("Key insights")
            insights=[]
            wet_phases=[p for p in weather_timeline if p["mode"] in {"CHANGEABLE","RAIN","HEAVY_RAIN"}]
            if wet_phases:
                first_wet=wet_phases[0]
                insights.append(("🌧️",f'{WEATHER_LABELS[first_wet["mode"]]} from lap {first_wet["start_lap"]}',f'Cross-over window near L{max(1,first_wet["start_lap"]-2)}–{first_wet["start_lap"]+2}.'))
            else:
                insights.append(("☀️","Dry race profile","Strategy is driven mainly by degradation, undercut power and pit loss."))
            if neutralisation_mode!="NONE":
                timing="random timing" if neutralisation_lap is None else f"lap {neutralisation_lap}"
                insights.append(("🟨",f"{neutral_label.title()} · {timing}","Pit-loss benefit applies only if the stop aligns with the neutralisation window."))
            else:
                insights.append(("🟩","No neutralisation selected","All planned stops pay the normal green-flag pit loss."))
            if result is not None:
                delta=float(result["delta_to_optimal_s"])
                if delta>.5:
                    insights.append(("⏱️",f"~{delta:.1f}s slower than optimum",f'Best same-start plan: {result["optimal"]["strategy"]}, pit {result["optimal"]["pit_window"]}.'))
                else:
                    insights.append(("✅","Strategy close to optimum","Your fixed plan is within half a second of the model optimum."))
                insights.append(("📊",f'Top 5 in {result["top5_probability"]:.0%} of simulations',f'Points {result["points_probability"]:.0%}; podium {result["podium_probability"]:.0%}.'))
            for icon,title,sub in insights[:4]:
                st.markdown(f'<div class="insight"><div class="insight-icon">{icon}</div><div><div class="t">{title}</div><div class="s">{sub}</div></div></div>',unsafe_allow_html=True)
    with ov[2]:
        with st.container(border=True):
            panel_title("Weather snapshot")
            grid=st.columns(2,gap="small")
            with grid[0]: summary_card("Track temp",f"{track_temp:.0f}°C","weighted")
            with grid[1]: summary_card("Rain",f"{rain_prob:.0%}","race average")
            with grid[0]: summary_card("Air temp",f"{air_temp:.0f}°C","ambient")
            with grid[1]: summary_card("Wind",f"{wind:.0f} km/h",f"humidity {humidity:.0f}%")
            if result is not None:
                st.markdown('<div class="compact-divider"></div>', unsafe_allow_html=True)
                inline_card("Finish position distribution", f'Projected finish around P{result["expected_finish"]:.1f}', f'Most likely outcome: P{result["most_likely_finish"]}')

with detail_tabs[1]:
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
                y=["Your strategy"],x=[e-s+1],base=[s-1],orientation="h",marker_color=TYRE_ACCENT.get(comp,"#8896a0"),
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
            height=220,barmode="overlay",bargap=.25,margin=dict(l=95,r=12,t=23,b=30),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#dce5eb",size=9),
            xaxis=dict(title="Lap",range=[0,race_laps],dtick=max(5,int(round(race_laps/10))),gridcolor="#17303c",zeroline=False,side="top"),
            yaxis=dict(categoryarray=["Weather","Safety Car / VSC","Your strategy"],categoryorder="array",autorange="reversed",gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(tf,use_container_width=True,config={"displayModeBar":False})

with detail_tabs[2]:
    lower=st.columns([1.4,.9],gap="small")
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
            wf.add_trace(go.Scatter(x=laps,y=temp_series,mode="lines",name="Track temperature",line=dict(color="#ff313c",width=2.0),hovertemplate="L%{x}: %{y:.0f}°C<extra></extra>"))
            wf.add_trace(go.Bar(x=laps,y=rain_series,name="Rain intensity",marker_color="#2479d1",opacity=.65,yaxis="y2",hovertemplate="L%{x}: %{y:.0f}% wet exposure<extra></extra>"))
            wf.update_layout(
                height=240,margin=dict(l=45,r=45,t=10,b=32),
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#cbd5dc",size=8),
                xaxis=dict(title="Lap",gridcolor="#17303c",zeroline=False),
                yaxis=dict(title="Track temp °C",gridcolor="#17303c",zeroline=False),
                yaxis2=dict(title="Rain %",overlaying="y",side="right",range=[0,100],showgrid=False),
                legend=dict(orientation="h",y=1.10,x=.50,font=dict(size=8)),
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
                '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch"><table class="tyre-table"><thead><tr><th>Compound</th><th>Deg s/lap</th><th>Relative pace</th><th>Best use</th></tr></thead>'
                f'<tbody>{tr}</tbody></table></div>',
                unsafe_allow_html=True,
            )

with detail_tabs[3]:
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
                '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch"><div class="standings">'
                '<div class="stand-head"><div>Pos</div><div>Driver</div><div class="stand-team-col">Team</div>'
                '<div style="text-align:right">Grid</div><div style="text-align:right">Expected</div>'
                '<div style="text-align:right">Podium</div><div style="text-align:right">Points</div></div>'
                +''.join(rows_html)+'</div></div>',
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
    '<div class="footerline"><div>Strategy Engine V2.7.1 · Driver-specific Target Outcome.</div></div>',
    unsafe_allow_html=True,
)
