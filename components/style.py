from pathlib import Path
import streamlit as st

def inject_css():
    st.markdown("""
    <style>
    :root{--navy:#0D004C;--blue:#008FD5;--lime:#AFE327;--green:#5EB95E;--charcoal:#45484D;--bg:#F4F7FA;}
    .stApp{background:var(--bg)}
    [data-testid="stHeader"]{background:transparent}
    .block-container{max-width:1600px;padding-top:.7rem;padding-bottom:2rem}
    .topbar{background:linear-gradient(105deg,#0D004C 0%,#17104f 65%,#008FD5 145%);padding:16px 22px;border-radius:0 0 16px 16px;color:white;margin:-.7rem -1rem 14px;box-shadow:0 8px 24px rgba(13,0,76,.18)}
    .eyebrow{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:#AFE327;font-weight:700}
    .title{font-size:1.55rem;font-weight:750;margin-top:3px}.subtitle{opacity:.78;font-size:.88rem}
    .status-strip{padding:10px 16px;border-radius:11px;background:#fff;border-left:6px solid #5EB95E;box-shadow:0 3px 14px rgba(20,30,55,.08);margin-bottom:12px}
    .panel{background:white;border:1px solid #E3E9EF;border-radius:14px;padding:14px 16px;box-shadow:0 4px 18px rgba(35,50,80,.06)}
    .section-intro{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin:8px 2px 12px}
    .section-kicker{font-size:.82rem;color:#0D004C;font-weight:760;letter-spacing:.02em}
    .section-note{font-size:.76rem;color:#73808c;margin-top:2px}
    .status-badge{border:1px solid;border-radius:999px;padding:5px 10px;font-size:.72rem;font-weight:760;background:#fff;white-space:nowrap}
    .section-divider{height:1px;background:#DCE4EA;margin:10px 2px 12px}
    .kpi{background:white;border:1px solid #E3E9EF;border-radius:14px;padding:15px 17px;box-shadow:0 2px 10px rgba(35,50,80,.045);height:100%;box-sizing:border-box}
    .kpi-primary{min-height:178px}
    .kpi-secondary{min-height:132px}
    .kpi-label{font-size:.70rem;color:#617181;text-transform:uppercase;letter-spacing:.09em;font-weight:760;margin-bottom:10px}
    .kpi-value{font-size:1.65rem;line-height:1.1;color:#0D004C;font-weight:760;margin:0}
    .kpi-value-compact{font-size:1.22rem;line-height:1.28;max-width:95%}
    .kpi-value-secondary{font-size:1.28rem;line-height:1.2}
    .kpi-sub{font-size:.75rem;line-height:1.45;color:#73808c;margin-top:10px}
    .kpi-sub-strong{font-weight:650;color:#617181}
    .split-count{display:flex;align-items:center;gap:16px;margin:0 0 12px}
    .split-count>div{display:flex;align-items:baseline;gap:5px}
    .count-number{font-size:1.65rem;line-height:1;color:#0D004C;font-weight:780}
    .count-label{font-size:.70rem;color:#617181;font-weight:760;letter-spacing:.06em}
    .count-divider{width:1px;height:26px;background:#DCE4EA}
    .status-list{font-size:.72rem;line-height:1.42;color:#73808c;margin-top:5px}
    .status-key{display:inline-block;min-width:27px;margin-right:6px;color:#0D004C;font-size:.66rem;font-weight:780;letter-spacing:.05em}
    .value-denominator{font-size:.92rem;font-weight:650}
    .response-pill{display:inline-block;border:1px solid;border-radius:999px;padding:4px 8px;margin-top:12px;font-size:.70rem;font-weight:760;background:rgba(255,255,255,.58)}
    .station-title{font-size:1.15rem;color:#0D004C;font-weight:750}.pill{display:inline-block;padding:4px 9px;border-radius:999px;font-size:.72rem;font-weight:750;background:#EAF7EA;color:#23762C}
    div[data-testid="stMetric"]{background:#fff;border:1px solid #E3E9EF;padding:10px 12px;border-radius:12px}
    .stButton>button,.stLinkButton>a{border-radius:9px;font-weight:650}
    </style>""",unsafe_allow_html=True)
