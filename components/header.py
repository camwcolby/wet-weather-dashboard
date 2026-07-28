from pathlib import Path
import base64
import streamlit as st

def render_header(mode_label):
    logo=Path(__file__).resolve().parents[1]/"assets"/"logos"/"inframark_azuria-tag-logo_0226-2048x593.png"
    b64=base64.b64encode(logo.read_bytes()).decode() if logo.exists() else ""
    st.markdown(f'''<div class="topbar"><div style="display:flex;align-items:center;justify-content:space-between;gap:18px"><div style="display:flex;align-items:center;gap:20px"><img src="data:image/png;base64,{b64}" style="height:42px;filter:brightness(0) invert(1)"><div><div class="eyebrow">Operational Intelligence Platform</div><div class="title">Hull Wet Weather Operations</div><div class="subtitle">Collections System + Wastewater Treatment Facility</div></div></div><div style="text-align:right"><div class="eyebrow">Operating View</div><div style="font-size:1rem;font-weight:700">{mode_label}</div></div></div></div>''',unsafe_allow_html=True)
