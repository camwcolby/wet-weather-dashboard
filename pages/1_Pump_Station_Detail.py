from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
st.set_page_config(page_title="Hull Asset Detail",page_icon="⚙️",layout="wide")
from components.style import inject_css
from components.header import render_header
from config.assets import ASSETS
from config.theme import NAVY,BLUE,AMBER,GREEN
from services.data_loader import load_collection,load_station_runtimes
inject_css(); render_header("Asset Drill-down")
collection=load_collection(); runtimes=load_station_runtimes()
station_ids=[a["asset_id"] for a in ASSETS if a["asset_type"]=="Pump Station"]
default=st.session_state.get("selected_asset","PS 3"); default=default if default in station_ids else "PS 3"
station=st.selectbox("Pump station",station_ids,index=station_ids.index(default)); st.session_state.selected_asset=station
a=next(x for x in ASSETS if x["asset_id"]==station)
s=collection[collection.asset_id==station].copy(); max_ts=s.timestamp.max(); start=max_ts-pd.Timedelta(days=7); s=s[s.timestamp>=start]
st.markdown(f"## {a['display_name']}  "); st.caption(f"{a['address']} · Design capacity {a['capacity_gpm']:,} gpm · Force main {a['force_main']}")
latest=s.iloc[-1]
c1,c2,c3,c4=st.columns(4); c1.metric("Wet-well level",f"{latest.level_in:.1f} in"); c2.metric("Flow",f"{latest.flow_gpm:,.0f} gpm"); c3.metric("Pump 1","RUNNING" if latest.pump1_status else "OFF"); c4.metric("Pump 2","RUNNING" if latest.pump2_status else "OFF")
fig=go.Figure(); fig.add_trace(go.Scatter(x=s.timestamp,y=s.level_in,name="Wet well (in)",line=dict(color=AMBER))); fig.add_trace(go.Scatter(x=s.timestamp,y=s.flow_gpm,name="Flow (gpm)",yaxis="y2",line=dict(color=BLUE)))
fig.update_layout(height=430,hovermode="x unified",paper_bgcolor="white",plot_bgcolor="white",yaxis=dict(title="Wet well (in)",gridcolor="#EDF1F5"),yaxis2=dict(title="Flow (gpm)",overlaying="y",side="right"),legend=dict(orientation="h",y=1.1),margin=dict(l=20,r=20,t=20,b=20)); st.plotly_chart(fig,use_container_width=True)
rt=runtimes[runtimes.asset_id==station].tail(14)
st.markdown("### Daily runtime and pumped volume")
fig2=go.Figure(); fig2.add_bar(x=rt.date,y=rt.total_runtime_hr,name="Runtime (hr)",marker_color=NAVY); fig2.add_trace(go.Scatter(x=rt.date,y=rt.flow_kgal,name="Flow (kgal)",yaxis="y2",line=dict(color=GREEN,width=3)))
fig2.update_layout(height=340,yaxis2=dict(overlaying="y",side="right"),hovermode="x unified",margin=dict(l=20,r=20,t=20,b=20),legend=dict(orientation="h",y=1.12)); st.plotly_chart(fig2,use_container_width=True)
st.page_link("app.py",label="← Return to system overview")
