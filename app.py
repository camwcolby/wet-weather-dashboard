from __future__ import annotations
from datetime import timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Hull Wet Weather Operations",page_icon="🌊",layout="wide",initial_sidebar_state="collapsed")

from components.style import inject_css
from components.header import render_header
from config.theme import NAVY,BLUE,GREEN,LIME,AMBER,RED,MUTED_BLUE
from services.data_loader import load_collection,load_influent,load_process_summary,load_station_runtimes,latest_snapshot,load_asset_locations
from services.weather import historical_precip,nws_bundle
from services.tides import tide_predictions,historical_tides
from services.marine import marine_forecast
from models.storm_selection import rank_events
from utils.formatting import fmt,status_from_utilization

inject_css()
with st.spinner("Loading 2026 operating data..."):
    collection=load_collection(); influent=load_influent(); process=load_process_summary(); runtimes=load_station_runtimes()
    min_date=min(collection.timestamp.min().date(),process.date.min().date()); max_date=max(collection.timestamp.max().date(),process.date.max().date(),runtimes.date.max().date())
    rain=historical_precip(min_date,max_date)
    ranked,significant=rank_events(process,runtimes,rain)

latest_storm = significant.iloc[0].event_date if not significant.empty else pd.Timestamp(max_date)
mode=st.sidebar.radio("Operating view",["Latest significant storm","Latest available data","Custom date"],index=0)
if mode=="Latest significant storm": selected_day=pd.Timestamp(latest_storm)
elif mode=="Latest available data": selected_day=pd.Timestamp(max_date)
else: selected_day=pd.Timestamp(st.sidebar.date_input("Date",value=latest_storm,min_value=min_date,max_value=max_date))
window=st.sidebar.select_slider("Playback window",options=[6,12,24,48,72],value=24,format_func=lambda x:f"{x} hours")
event_row = ranked[ranked.event_date.dt.normalize()==selected_day.normalize()]
response_day = pd.Timestamp(event_row.iloc[0].response_date) if not event_row.empty and pd.notna(event_row.iloc[0].response_date) else selected_day
as_of=response_day+pd.Timedelta(hours=23,minutes=59)
render_header(f"Historical Playback · Rain {selected_day:%b %d} → Response {response_day:%b %d, %Y}")
st.page_link("pages/2_Wet_Weather_Analytics.py", label="Open Wet Weather Analytics", icon="🌧️")

row=ranked[ranked.event_date.dt.normalize()==selected_day.normalize()]
row=row.iloc[0] if not row.empty else pd.Series(dtype=float)
rain_val=row.get("rain_in",np.nan); plant_flow=row.get("plant_peak_mgd",np.nan); storm_score=row.get("storm_score",np.nan); response_lag=row.get("response_lag_hr",np.nan)
snap=latest_snapshot(collection,as_of)
max_level=snap.level_in.max() if not snap.empty else np.nan
running=int((snap.get("pump1_status",0).fillna(0)+snap.get("pump2_status",0).fillna(0)>0).sum()) if not snap.empty else 0
severity="ALARM" if storm_score>=.8 else "ELEVATED" if storm_score>=.55 else "WATCH" if storm_score>=.3 else "NORMAL"
sev_color={"NORMAL":GREEN,"WATCH":LIME,"ELEVATED":AMBER,"ALARM":RED}[severity]
st.markdown(f'<div class="status-strip" style="border-left-color:{sev_color}"><b style="color:{NAVY}">{severity} WET WEATHER STATUS</b> &nbsp; Storm response score {fmt(storm_score*100,0,"%")} · {fmt(rain_val,2," in")} rainfall · plant response peak +{fmt(response_lag,0," hr")} · {running}/7 telemetry stations operating at playback time</div>',unsafe_allow_html=True)

kpis=st.columns(6)
items=[("Rainfall trigger",fmt(rain_val,2,' in'),"Event-day total"),("Peak plant influent",fmt(plant_flow,2,' MGD'),"Trigger day + next 24 hr"),("Highest wet well",fmt(max_level,1,' in'),"Across stations"),("Stations running",f"{running} / 7","Telemetry reporting"),("System runtime",fmt(row.get('total_runtime_48h',np.nan),1,' hr'),"Daily combined"),("Storm score",fmt(storm_score*100,0,'%'),"Rain + hydraulic response")]
for c,(lab,val,sub) in zip(kpis,items): c.markdown(f'<div class="kpi"><div class="kpi-label">{lab}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>',unsafe_allow_html=True)

left,right=st.columns([2.25,1],gap="medium")
with left:
    assets=load_asset_locations()
    if not snap.empty:
        assets=assets.merge(snap[[c for c in ["asset_id","flow_gpm","level_in","pump1_status","pump2_status","interceptor_level"] if c in snap]],on="asset_id",how="left")
    else:
        for c in ["flow_gpm","level_in","pump1_status","pump2_status","interceptor_level"]: assets[c]=np.nan
    assets["utilization"]=(assets.level_in/84).clip(0,1.2)
    assets["status"]=assets.utilization.apply(status_from_utilization)
    assets.loc[assets.asset_type=="Treatment Plant","status"]="Plant"
    colors={"Normal":GREEN,"Watch":LIME,"Warning":AMBER,"Alarm":RED,"No Data":"#98A5B3","Plant":NAVY}
    assets["marker_color"]=assets.status.map(colors)
    assets["hover"]="<b>"+assets.display_name+"</b><br>"+assets.address+"<br>Flow: "+assets.flow_gpm.fillna(0).round(0).astype(int).astype(str)+" gpm<br>Wet well: "+assets.level_in.round(1).astype(str)+" in<br>Status: "+assets.status
    fig=px.scatter_map(assets,lat="lat",lon="lon",color="status",color_discrete_map=colors,size=assets.asset_type.map({"Treatment Plant":26,"Pump Station":16}).fillna(16),hover_name="display_name",hover_data={"address":True,"flow_gpm":':.0f',"level_in":':.1f',"lat":False,"lon":False,"status":False},zoom=11.4,height=570)
    fig.update_layout(map_style="open-street-map",margin=dict(l=0,r=0,t=0,b=0),legend=dict(orientation="h",y=1.01,x=.01,bgcolor="rgba(255,255,255,.8)"),clickmode="event+select")
    event=st.plotly_chart(fig,use_container_width=True,on_select="rerun",selection_mode="points",key="system_map")
    selected_asset=st.session_state.get("selected_asset","PS 3")
    try:
        pts=event.selection.points if event and event.selection else []
        if pts:
            name=pts[0].get("hovertext") or pts[0].get("customdata",[None])[0]
            match=assets.loc[assets.display_name==name]
            if not match.empty: selected_asset=match.iloc[0].asset_id; st.session_state.selected_asset=selected_asset
    except Exception: pass
with right:
    choices=assets.asset_id.tolist()
    selected_asset=st.selectbox("Selected asset",choices,index=choices.index(selected_asset) if selected_asset in choices else 0,label_visibility="collapsed")
    st.session_state.selected_asset=selected_asset
    a=assets.loc[assets.asset_id==selected_asset].iloc[0]
    st.markdown(f'<div class="panel"><div class="station-title">{a.display_name}</div><div style="color:#73808c;font-size:.82rem;margin:3px 0 10px">{a.address}</div><span class="pill">{a.status}</span><hr style="border:none;border-top:1px solid #E8EDF2;margin:12px 0"><b>Current snapshot</b><br><br>Wet well <b style="float:right">{fmt(a.level_in,1," in")}</b><br>Flow <b style="float:right">{fmt(a.flow_gpm,0," gpm")}</b><br>Pumps running <b style="float:right">{int((a.pump1_status or 0)+(a.pump2_status or 0)) if pd.notna(a.get("pump1_status")) else "—"}</b><br>Design capacity <b style="float:right">{fmt(a.capacity_gpm,0," gpm")}</b></div>',unsafe_allow_html=True)
    st.page_link("pages/1_Pump_Station_Detail.py",label="Open dedicated asset page →",icon="🔎",use_container_width=True)
    st.markdown("#### What needs attention")
    alerts=[]
    for _,x in assets[assets.asset_type=="Pump Station"].sort_values("utilization",ascending=False).head(4).iterrows():
        alerts.append(f"**{x.asset_id}** · {x.status} · {fmt(x.level_in,1,' in')} wet well")
    st.info("\n\n".join(alerts) if alerts else "No station data available")

st.markdown("### Coordinated storm response")
start=min(selected_day, as_of-pd.Timedelta(hours=window))
c=collection[(collection.timestamp>=start)&(collection.timestamp<=as_of)].groupby("timestamp",as_index=False).agg(collection_flow_gpm=("flow_gpm","sum"),max_wetwell_in=("level_in","max"))
i=influent[(influent.timestamp>=start)&(influent.timestamp<=as_of)][["timestamp","influent_total_mgd"]]
t=historical_tides(start.date(),as_of.date()+timedelta(days=1))
fig=go.Figure()
fig.add_trace(go.Scatter(x=c.timestamp,y=c.collection_flow_gpm,name="Collection flow",line=dict(color=BLUE,width=2)))
fig.add_trace(go.Scatter(x=i.timestamp,y=i.influent_total_mgd*694.444,name="Plant influent (gpm eq.)",line=dict(color=NAVY,width=2)))
fig.add_trace(go.Scatter(x=c.timestamp,y=c.max_wetwell_in*35,name="Max wet well (scaled)",line=dict(color=AMBER,width=1.5,dash="dot")))
if not t.empty: fig.add_trace(go.Scatter(x=t.t,y=t.v*250,name="Tide (scaled)",line=dict(color=MUTED_BLUE,width=1,dash="dash")))
fig.update_layout(height=360,margin=dict(l=15,r=15,t=10,b=20),paper_bgcolor="white",plot_bgcolor="white",legend=dict(orientation="h",y=1.13),hovermode="x unified",xaxis=dict(showgrid=False),yaxis=dict(title="Operational response index",gridcolor="#EDF1F5"))
st.plotly_chart(fig,use_container_width=True)

with st.expander("External API status and live context"):
    weather=nws_bundle(); tides=tide_predictions(); marine=marine_forecast()
    a,b,c=st.columns(3)
    a.write("**National Weather Service**"); a.success(f"Connected · {len(weather['hourly'])} hourly periods") if weather.get("ok") else a.warning("Unavailable; historical dashboard remains operational")
    b.write("**NOAA Tides & Currents**"); b.success(f"Connected · {len(tides)} tide events") if not tides.empty else b.warning("Unavailable; cached/historical view used")
    c.write("**Marine forecast**"); c.success(f"Connected · {len(marine)} hourly periods") if not marine.empty else c.warning("Unavailable; dashboard degrades gracefully")
