# dashboard.py — Streamlit live forecasting dashboard
# Run scheduler.py in a separate terminal to keep data fresh.

import json
import calendar
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path

from database import (read_actuals, read_predictions, read_errors, init_db)
from config   import PEAK_THRESHOLD_PATH, FORECAST_HOURS

# Shared file that tells the scheduler which model to use.
# Written here whenever the radio button changes; read by scheduler.py
# at the top of every hourly job.
SELECTED_MODEL_FILE = Path("selected_model.json")


def write_selected_model(model_name: str):
    """Persist the user radio selection so the scheduler picks it up."""
    try:
        SELECTED_MODEL_FILE.write_text(
            json.dumps({"selected_model": model_name}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        st.warning(f"Could not save model selection: {exc}")

st.set_page_config(
    page_title="⚡ Electricity Demand Forecast",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state: theme ──────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode

# ═══════════════════════════════════════════════════════════════
# THEME TOKENS
# ═══════════════════════════════════════════════════════════════
if dark:
    T = dict(
        page_bg       = "#0f1117",
        sidebar_bg    = "#0d1117",
        sidebar_border= "#1f2937",
        sidebar_text  = "#a8c5b5",
        sidebar_muted = "#374151",
        sidebar_input = "#111827",
        card_bg       = "#1a1d27",
        card_border   = "#1f2937",
        card_shadow   = "0 4px 20px rgba(0,0,0,0.3)",
        tile_bg       = "#111827",
        tile_border   = "#1f2937",
        title_color   = "#f9fafb",
        text_color    = "#e2e8f0",
        sub_color     = "#4b5563",
        muted         = "#374151",
        plot_bg       = "#111827",
        plot_grid     = "#1f2937",
        plot_tick     = "#4b5563",
        plot_legend   = "#9ca3af",
        prog_track    = "#111827",
        ph_bg         = "#111827",
        ph_border     = "#1f2937",
        ph_title      = "#4b5563",
        ph_text       = "#374151",
        divider       = "#1f2937",
        # KPI schemes
        kpi1_bg  = "linear-gradient(135deg,#1e40af,#3b82f6)", kpi1_fg="#fff", kpi1_label="rgba(255,255,255,0.6)", kpi1_delta="rgba(255,255,255,0.55)",
        kpi2_bg  = "linear-gradient(135deg,#0e7490,#22d3ee)", kpi2_fg="#fff", kpi2_label="rgba(255,255,255,0.6)", kpi2_delta="rgba(255,255,255,0.55)",
        kpi3a_bg = "linear-gradient(135deg,#065f46,#10b981)", kpi3a_fg="#fff",
        kpi3b_bg = "linear-gradient(135deg,#7f1d1d,#ef4444)", kpi3b_fg="#fff",
        kpi4_bg  = "linear-gradient(135deg,#4c1d95,#8b5cf6)", kpi4_fg="#fff", kpi4_label="rgba(255,255,255,0.6)", kpi4_delta="rgba(255,255,255,0.55)",
        kpi5_bg  = "linear-gradient(135deg,#065f46,#10b981)", kpi5_fg="#fff", kpi5_label="rgba(255,255,255,0.6)", kpi5_delta="rgba(255,255,255,0.55)",
        # Chart colors
        c_main   = "#3b82f6", c_main_fill="rgba(59,130,246,0.08)",
        c_fore   = "#22d3ee", c_peak_fill="rgba(239,68,68,0.1)",
        c_thresh = "#f59e0b", c_now="#374151",
        c_solar  = "#f59e0b", c_wind="#10b981",
        c_price  = "#f59e0b", c_price_fill="rgba(245,158,11,0.08)",
        c_temp   = "#ef4444", c_temp_fill="rgba(239,68,68,0.06)",
        c_hum    = "#3b82f6", c_hum_fill="rgba(59,130,246,0.06)",
        c_ratio  = "#10b981", c_ratio_fill="rgba(16,185,129,0.1)",
        c_acc1   = "#ef4444", c_acc1_fill="rgba(239,68,68,0.07)",
        c_acc2   = "#8b5cf6", c_acc2_fill="rgba(139,92,246,0.07)",
        c_spend  = "#10b981", c_spend_red="#ef4444",
        c_spend_fill="rgba(16,185,129,0.07)", c_spend_fill_red="rgba(239,68,68,0.07)",
        c_goal_line="#3b82f6", c_goal_bar="rgba(59,130,246,0.25)",
        c_bar_border="#3b82f6",
        fin_card_bg="#1a1d27", fin_card_border="#1f2937",
        fin_val="#f9fafb", fin_sub="#4b5563",
        fin_pos="#10b981", fin_neg="#ef4444",
        alert_ok_bg="rgba(16,185,129,0.1)", alert_ok_border="rgba(16,185,129,0.3)",
        alert_ok_left="#10b981", alert_ok_text="#6ee7b7",
        alert_warn_bg="rgba(245,158,11,0.1)", alert_warn_border="rgba(245,158,11,0.3)",
        alert_warn_left="#f59e0b", alert_warn_text="#fcd34d",
        alert_err_bg="rgba(239,68,68,0.1)", alert_err_border="rgba(239,68,68,0.3)",
        alert_err_left="#ef4444", alert_err_text="#fca5a5",
        header_bg="linear-gradient(135deg,#0f172a,#1e293b,#0f172a)",
        header_border="#1e3a5f",
        header_glow1="rgba(59,130,246,0.15)", header_glow2="rgba(16,185,129,0.08)",
        header_title="#f9fafb",
        badge_bg="rgba(245,200,66,0.15)", badge_border="rgba(245,200,66,0.35)", badge_color="#f5c842",
        badge2_bg="rgba(74,222,128,0.12)", badge2_border="rgba(74,222,128,0.3)", badge2_color="#4ade80",
        header_meta="#4b5563",
        table_peak="rgba(239,68,68,0.08)",
        toggle_icon="🌙",
    )
else:
    T = dict(
        page_bg       = "#f0f4f0",
        sidebar_bg    = "#1a3a2e",
        sidebar_border= "#2d5a42",
        sidebar_text  = "#a8c5b5",
        sidebar_muted = "#4a8a6a",
        sidebar_input = "#0f2419",
        card_bg       = "#ffffff",
        card_border   = "#e8f0eb",
        card_shadow   = "0 2px 16px rgba(0,0,0,0.06)",
        tile_bg       = "#f8faf9",
        tile_border   = "#e5ede8",
        title_color   = "#1a3a2e",
        text_color    = "#374151",
        sub_color     = "#9ca3af",
        muted         = "#d1d5db",
        plot_bg       = "#f8faf9",
        plot_grid     = "#e8f0eb",
        plot_tick     = "#9ca3af",
        plot_legend   = "#374151",
        prog_track    = "#e8f0eb",
        ph_bg         = "#f8faf9",
        ph_border     = "#d1e8da",
        ph_title      = "#6b7280",
        ph_text       = "#9ca3af",
        divider       = "#e8f0eb",
        kpi1_bg  = "#1a3a2e", kpi1_fg="#e8f5ee", kpi1_label="#6b9e88", kpi1_delta="#6b9e88",
        kpi2_bg  = "#f5c842", kpi2_fg="#1a2e1a", kpi2_label="rgba(26,46,26,0.55)", kpi2_delta="rgba(26,46,26,0.55)",
        kpi3a_bg = "#1a3a2e", kpi3a_fg="#e8f5ee",
        kpi3b_bg = "#dc2626", kpi3b_fg="#fff",
        kpi4_bg  = "#3d7a5e", kpi4_fg="#e8f5ee", kpi4_label="#a8c5b5", kpi4_delta="#a8c5b5",
        kpi5_bg  = "#d1fae5", kpi5_fg="#065f46", kpi5_label="#34d399", kpi5_delta="#34d399",
        c_main   = "#1a3a2e", c_main_fill="rgba(26,58,46,0.07)",
        c_fore   = "#d97706", c_peak_fill="rgba(220,38,38,0.07)",
        c_thresh = "#dc2626", c_now="#9ca3af",
        c_solar  = "#f5c842", c_wind="#1a3a2e",
        c_price  = "#d97706", c_price_fill="rgba(215,119,0,0.08)",
        c_temp   = "#dc2626", c_temp_fill="rgba(220,38,38,0.06)",
        c_hum    = "#2563eb", c_hum_fill="rgba(37,99,235,0.06)",
        c_ratio  = "#16a34a", c_ratio_fill="rgba(22,163,74,0.09)",
        c_acc1   = "#dc2626", c_acc1_fill="rgba(220,38,38,0.07)",
        c_acc2   = "#1a3a2e", c_acc2_fill="rgba(26,58,46,0.07)",
        c_spend  = "#16a34a", c_spend_red="#dc2626",
        c_spend_fill="rgba(22,163,74,0.07)", c_spend_fill_red="rgba(220,38,38,0.07)",
        c_goal_line="#1a3a2e", c_goal_bar="rgba(26,58,46,0.15)",
        c_bar_border="#1a3a2e",
        fin_card_bg="#ffffff", fin_card_border="#e8f0eb",
        fin_val="#1a3a2e", fin_sub="#9ca3af",
        fin_pos="#16a34a", fin_neg="#dc2626",
        alert_ok_bg="#f0fdf4", alert_ok_border="#bbf7d0",
        alert_ok_left="#16a34a", alert_ok_text="#15803d",
        alert_warn_bg="#fffbeb", alert_warn_border="#fde68a",
        alert_warn_left="#d97706", alert_warn_text="#92400e",
        alert_err_bg="#fef2f2", alert_err_border="#fecaca",
        alert_err_left="#dc2626", alert_err_text="#991b1b",
        header_bg="#1a3a2e",
        header_border="#2d5a42",
        header_glow1="rgba(245,200,66,0.18)", header_glow2="rgba(74,222,128,0.1)",
        header_title="#f0f9f4",
        badge_bg="rgba(245,200,66,0.15)", badge_border="rgba(245,200,66,0.35)", badge_color="#f5c842",
        badge2_bg="rgba(74,222,128,0.12)", badge2_border="rgba(74,222,128,0.3)", badge2_color="#4ade80",
        header_meta="#6b9e88",
        table_peak="rgba(220,38,38,0.06)",
        toggle_icon="☀️",
    )

# ═══════════════════════════════════════════════════════════════
# CSS — generated from theme tokens
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {{ font-family:'Inter',sans-serif; background:{T['page_bg']}; }}
#MainMenu,footer,header {{ visibility:hidden; }}
.block-container {{ padding:1.5rem 2rem 3rem !important; max-width:100% !important; background:{T['page_bg']}; }}

/* Sidebar */
section[data-testid="stSidebar"] > div:first-child {{
    background:{T['sidebar_bg']}; border-right:1px solid {T['sidebar_border']};
}}
[data-testid="stSidebar"] * {{ color:{T['sidebar_text']} !important; }}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {{ color:#f0f9f4 !important; }}
[data-testid="stSidebar"] hr {{ border-color:{T['sidebar_border']} !important; }}
[data-testid="stSidebar"] a {{ color:#4ade80 !important; }}
[data-testid="stSidebar"] .stNumberInput input {{
    background:{T['sidebar_input']} !important; border:1px solid {T['sidebar_border']} !important;
    color:#f0f9f4 !important; border-radius:8px !important;
}}
[data-testid="stSidebar"] button {{
    background:{T['sidebar_border']} !important; color:#f0f9f4 !important;
    border:none !important; border-radius:8px !important;
}}

/* KPI cards */
.kpi-card {{
    border-radius:16px; padding:1.3rem 1.5rem;
    position:relative; overflow:hidden;
    box-shadow:0 4px 24px rgba(0,0,0,0.12);
    height:115px; display:flex; flex-direction:column; justify-content:space-between;
}}
.kpi-card::after {{
    content:''; position:absolute; bottom:-25px; right:-25px;
    width:80px; height:80px; border-radius:50%; background:rgba(255,255,255,0.1);
}}
.kpi-label {{ font-size:0.6rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; }}
.kpi-value {{ font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:700; line-height:1; white-space:nowrap; }}
.kpi-delta {{ font-size:0.7rem; font-weight:500; margin-top:2px; }}

.kpi1 {{ background:{T['kpi1_bg']}; color:{T['kpi1_fg']}; }}
.kpi1 .kpi-label {{ color:{T['kpi1_label']}; }} .kpi1 .kpi-delta {{ color:{T['kpi1_delta']}; }}
.kpi2 {{ background:{T['kpi2_bg']}; color:{T['kpi2_fg']}; }}
.kpi2 .kpi-label {{ color:{T['kpi2_label']}; }} .kpi2 .kpi-delta {{ color:{T['kpi2_delta']}; }}
.kpi3a {{ background:{T['kpi3a_bg']}; color:{T['kpi3a_fg']}; }}
.kpi3b {{ background:{T['kpi3b_bg']}; color:{T['kpi3b_fg']}; }}
.kpi4 {{ background:{T['kpi4_bg']}; color:{T['kpi4_fg']}; }}
.kpi4 .kpi-label {{ color:{T['kpi4_label']}; }} .kpi4 .kpi-delta {{ color:{T['kpi4_delta']}; }}
.kpi5 {{ background:{T['kpi5_bg']}; color:{T['kpi5_fg']}; }}
.kpi5 .kpi-label {{ color:{T['kpi5_label']}; }} .kpi5 .kpi-delta {{ color:{T['kpi5_delta']}; }}

/* Section cards */
.card {{ background:{T['card_bg']}; border:1px solid {T['card_border']}; border-radius:16px; padding:1.4rem 1.6rem; box-shadow:{T['card_shadow']}; margin-bottom:1rem; }}
.card-title {{ font-family:'Space Grotesk',sans-serif; font-size:0.92rem; font-weight:700; color:{T['title_color']}; margin-bottom:2px; }}
.card-sub   {{ font-size:0.7rem; color:{T['sub_color']}; margin-bottom:0.9rem; }}

/* Stat tiles */
.stat-tile {{ background:{T['tile_bg']}; border:1px solid {T['tile_border']}; border-radius:10px; padding:0.75rem 1rem; text-align:center; }}
.stat-tile-label {{ font-size:0.6rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:{T['sub_color']}; margin-bottom:3px; }}
.stat-tile-value {{ font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:700; color:{T['title_color']}; white-space:nowrap; }}

/* Header banner */
.dash-header {{
    background:{T['header_bg']}; border:1px solid {T['header_border']};
    border-radius:18px; padding:1.6rem 2rem; margin-bottom:1.5rem;
    position:relative; overflow:hidden; box-shadow:0 8px 32px rgba(0,0,0,0.2);
}}
.dash-header::before {{
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px; border-radius:50%;
    background:radial-gradient(circle,{T['header_glow1']} 0%,transparent 70%);
}}
.dash-header::after {{
    content:''; position:absolute; bottom:-40px; left:25%;
    width:160px; height:160px; border-radius:50%;
    background:radial-gradient(circle,{T['header_glow2']} 0%,transparent 70%);
}}
.dash-title {{ font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:800; color:{T['header_title']}; letter-spacing:-0.02em; margin-bottom:6px; }}
.dash-badge  {{ display:inline-block; background:{T['badge_bg']}; border:1px solid {T['badge_border']}; color:{T['badge_color']}; font-size:0.62rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; border-radius:20px; padding:2px 10px; margin-right:5px; }}
.dash-badge2 {{ display:inline-block; background:{T['badge2_bg']}; border:1px solid {T['badge2_border']}; color:{T['badge2_color']}; font-size:0.62rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; border-radius:20px; padding:2px 10px; margin-right:5px; }}
.dash-meta {{ font-size:0.73rem; color:{T['header_meta']}; margin-top:8px; }}

/* Financial cards */
.fin-card {{ background:{T['fin_card_bg']}; border:1px solid {T['fin_card_border']}; border-radius:14px; padding:1.2rem 1.4rem; box-shadow:{T['card_shadow']}; }}
.fin-label {{ font-size:0.6rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:{T['sub_color']}; margin-bottom:3px; }}
.fin-value {{ font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; color:{T['fin_val']}; line-height:1.1; }}
.fin-pos   {{ font-size:0.73rem; color:{T['fin_pos']}; font-weight:500; margin-top:3px; }}
.fin-neg   {{ font-size:0.73rem; color:{T['fin_neg']}; font-weight:500; margin-top:3px; }}
.fin-sub   {{ font-size:0.73rem; color:{T['fin_sub']}; margin-top:3px; }}

/* Alerts */
.alert-ok   {{ background:{T['alert_ok_bg']};   border:1px solid {T['alert_ok_border']};   border-left:4px solid {T['alert_ok_left']};   border-radius:10px; padding:0.85rem 1.1rem; color:{T['alert_ok_text']};   font-size:0.85rem; font-weight:500; }}
.alert-warn {{ background:{T['alert_warn_bg']}; border:1px solid {T['alert_warn_border']}; border-left:4px solid {T['alert_warn_left']}; border-radius:10px; padding:0.85rem 1.1rem; color:{T['alert_warn_text']}; font-size:0.85rem; font-weight:500; }}
.alert-err  {{ background:{T['alert_err_bg']};  border:1px solid {T['alert_err_border']};  border-left:4px solid {T['alert_err_left']};  border-radius:10px; padding:0.85rem 1.1rem; color:{T['alert_err_text']};  font-size:0.85rem; font-weight:500; }}

/* Progress bars */
.prog-wrap  {{ background:{T['prog_track']}; border-radius:6px; height:9px; overflow:hidden; margin:5px 0 2px; }}
.prog-green {{ height:9px; border-radius:6px; background:linear-gradient(90deg,#16a34a,#4ade80); }}
.prog-amber {{ height:9px; border-radius:6px; background:linear-gradient(90deg,#d97706,#f5c842); }}
.prog-red   {{ height:9px; border-radius:6px; background:linear-gradient(90deg,#dc2626,#f87171); }}
.prog-label {{ font-size:0.78rem; font-weight:600; color:{T['title_color']}; margin-bottom:2px; }}
.prog-pct   {{ font-size:0.75rem; color:{T['sub_color']}; font-weight:400; }}
.prog-sub   {{ font-size:0.68rem; color:{T['sub_color']}; margin-top:2px; }}

/* Placeholder */
.placeholder {{ background:{T['ph_bg']}; border:1.5px dashed {T['ph_border']}; border-radius:12px; padding:2.5rem; text-align:center; }}
.placeholder-icon  {{ font-size:2rem; margin-bottom:0.5rem; }}
.placeholder-title {{ font-family:'Space Grotesk',sans-serif; font-weight:600; color:{T['ph_title']}; margin-bottom:4px; font-size:0.9rem; }}
.placeholder-text  {{ font-size:0.78rem; color:{T['ph_text']}; }}

hr {{ border-color:{T['divider']} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=T["plot_bg"],
    font=dict(family="Inter", color=T["plot_tick"], size=11),
    xaxis=dict(gridcolor=T["plot_grid"], linecolor=T["plot_grid"],
               zerolinecolor=T["plot_grid"], tickfont=dict(color=T["plot_tick"])),
    yaxis=dict(gridcolor=T["plot_grid"], linecolor=T["plot_grid"],
               zerolinecolor=T["plot_grid"], tickfont=dict(color=T["plot_tick"])),
    margin=dict(l=0, r=0, t=30, b=0),
)

# ── Init ──────────────────────────────────────────────────────
@st.cache_resource
def load_threshold():
    try:    return joblib.load(PEAK_THRESHOLD_PATH)
    except: return 30_000.0

PEAK_THRESHOLD = load_threshold()
init_db()

def safe_last(s):
    v = s.dropna(); return float(v.iloc[-1]) if not v.empty else None
def safe_nth(s, n):
    v = s.dropna(); return float(v.iloc[-n]) if len(v) >= n else None
def has_col(df, col):
    return col in df.columns and df[col].notna().any()
def ph(icon, title, text):
    return f"""<div class="placeholder">
        <div class="placeholder-icon">{icon}</div>
        <div class="placeholder-title">{title}</div>
        <div class="placeholder-text">{text}</div>
    </div>"""


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0.4rem 0 1rem">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;color:#f0f9f4">
            ⚡ ElecForecast
        </div>
        <div style="font-size:0.68rem;color:#4a8a6a;margin-top:2px;letter-spacing:0.06em;text-transform:uppercase">
            University of Houston · ENTSO-E
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── THEME TOGGLE ──────────────────────────────────────────
    col_tog1, col_tog2 = st.columns([1, 1])
    with col_tog1:
        st.markdown(f'<p style="font-size:0.62rem;font-weight:600;letter-spacing:0.1em;color:#4a8a6a;text-transform:uppercase;margin-top:10px">Theme</p>', unsafe_allow_html=True)
    with col_tog2:
        toggled = st.toggle(
            label=T["toggle_icon"],
            value=st.session_state.dark_mode,
            key="theme_toggle",
            help="Switch between Light and Dark mode",
        )
        if toggled != st.session_state.dark_mode:
            st.session_state.dark_mode = toggled
            st.rerun()

    st.markdown(f'<p style="font-size:0.72rem;color:#4a8a6a;margin:-4px 0 8px">{"🌙 Dark mode" if dark else "☀️ Light mode"}</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<p style="font-size:0.6rem;font-weight:600;letter-spacing:0.1em;color:#4a8a6a;text-transform:uppercase;margin-bottom:6px">Display</p>', unsafe_allow_html=True)
    refresh_interval = st.slider("Auto-refresh (s)", 30, 300, 60, 30)
    history_days     = st.slider("History window (days)", 1, 14, 7)
    show_raw         = st.checkbox("Show raw data tables", value=False)

    st.divider()
    st.markdown('<p style="font-size:0.6rem;font-weight:600;letter-spacing:0.1em;color:#4a8a6a;text-transform:uppercase;margin-bottom:6px">💰 Financial Goals</p>', unsafe_allow_html=True)
    monthly_goal     = st.number_input("Monthly Budget (€)",  min_value=0.0, value=150.0,  step=10.0,  format="%.2f")
    annual_goal      = st.number_input("Annual Budget (€)",   min_value=0.0, value=1800.0, step=50.0,  format="%.2f")
    user_kwh_monthly = st.number_input("Monthly Usage (kWh)", min_value=1.0, value=300.0,  step=10.0,  format="%.0f")

    st.divider()
    st.markdown('<p style="font-size:0.6rem;font-weight:600;letter-spacing:0.1em;color:#4a8a6a;text-transform:uppercase;margin-bottom:6px">Data Sources</p>', unsafe_allow_html=True)
    st.markdown("🔋 [ENTSO-E](https://transparency.entsoe.eu)  \n🌤 [OpenWeatherMap](https://openweathermap.org)")
    st.caption("Forecasts refresh hourly via scheduler.py")
    st.divider()
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()

st.markdown(f'<meta http-equiv="refresh" content="{refresh_interval}">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=55)
def load_data(hd):
    return (
        read_actuals(hours_back=hd * 24),
        read_predictions(hours_ahead=FORECAST_HOURS),
        read_errors(days_back=hd),
    )

actuals, preds_raw, errors_raw = load_data(history_days)


# ═══════════════════════════════════════════════════════════════
# MODEL SELECTOR — XGBoost / LightGBM forecast view
# ═══════════════════════════════════════════════════════════════
def normalize_prediction_models(pred_df):
    """
    Supports either format:
    1) Long format: columns include model, predicted_load
    2) Wide format: columns include xgboost_prediction/lightgbm_prediction
       or predicted_load_xgboost/predicted_load_lightgbm

    The dashboard will use the selected model and keep the rest of your
    existing charts/tables unchanged.
    """
    if pred_df.empty:
        return pred_df.copy()

    df = pred_df.copy()

    if "model" in df.columns and "predicted_load" in df.columns:
        df["model"] = df["model"].astype(str)
        return df

    possible_pairs = {
        "LightGBM": [
            "lightgbm_prediction",
            "predicted_load_lightgbm",
            "lightgbm_predicted_load",
            "lgbm_prediction",
        ],
        "XGBoost": [
            "xgboost_prediction",
            "predicted_load_xgboost",
            "xgboost_predicted_load",
            "xgb_prediction",
        ],
    }

    rows = []
    for model_name, possible_cols in possible_pairs.items():
        pred_col = next((c for c in possible_cols if c in df.columns), None)
        if pred_col is None:
            continue

        temp = df.copy()
        temp["model"] = model_name
        temp["predicted_load"] = temp[pred_col]

        if "is_peak" not in temp.columns:
            temp["is_peak"] = (temp["predicted_load"] >= PEAK_THRESHOLD).astype(int)

        rows.append(temp)

    if rows:
        return pd.concat(rows, ignore_index=True)

    # Fallback for old scheduler/database that only stores one prediction column.
    # It lets the dashboard still run, but both buttons will show the same old prediction
    # until scheduler.py/database.py are updated to save both model outputs.
    if "predicted_load" in df.columns:
        df["model"] = "XGBoost"
        return df

    return df


preds_all_models = normalize_prediction_models(preds_raw)

if "selected_forecast_model" not in st.session_state:
    st.session_state.selected_forecast_model = "LightGBM"

# Always show both options — the radio must never collapse to one item
# even if only one model has predictions in the DB yet.
MODEL_OPTIONS = ["LightGBM", "XGBoost"]

if st.session_state.selected_forecast_model not in MODEL_OPTIONS:
    st.session_state.selected_forecast_model = "LightGBM"

# Keep the selected model from the radio button state even though the radio UI
# is displayed later, below the comparison section.
if "model_radio" in st.session_state and st.session_state.model_radio in MODEL_OPTIONS:
    st.session_state.selected_forecast_model = st.session_state.model_radio
else:
    st.session_state.model_radio = st.session_state.selected_forecast_model

selected_model = st.session_state.selected_forecast_model
write_selected_model(selected_model)

# Scheduler/database may save the model column as model_name.
# Normalize it so the dashboard can filter correctly.
if not preds_all_models.empty and "model_name" in preds_all_models.columns and "model" not in preds_all_models.columns:
    preds_all_models["model"] = preds_all_models["model_name"]


def filter_by_model(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Return only rows for the selected model. Works for both preds and errors."""
    if df.empty:
        return df.copy()
    col = "model" if "model" in df.columns else ("model_name" if "model_name" in df.columns else None)
    if col:
        return df[df[col].astype(str).str.lower() == model_name.lower()].copy()
    return df.copy()


preds  = filter_by_model(preds_all_models, selected_model)
errors = filter_by_model(errors_raw, selected_model)


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
now_utc = pd.Timestamp.now(tz="UTC")
st.markdown(f"""
<div class="dash-header">
    <div class="dash-title">⚡ Live Electricity Demand Forecast</div>
    <div style="margin-top:8px">
        <span class="dash-badge2">● Live</span>
        <span class="dash-badge">LightGBM</span>
        <span class="dash-badge">XGBoost</span>
        <span class="dash-badge">Spain · ENTSO-E</span>
        <span class="dash-badge">Peak ≥ {PEAK_THRESHOLD:,.0f} MW</span>
    </div>
    <div class="dash-meta">
        Refreshes every {refresh_interval}s &nbsp;·&nbsp;
        {now_utc.strftime('%A, %d %B %Y &nbsp;·&nbsp; %H:%M UTC')}
    </div>
</div>
""", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════
# KPI ROW
# ═══════════════════════════════════════════════════════════════
latest = safe_last(actuals["total_load_actual"]) if not actuals.empty else None
prev   = safe_nth(actuals["total_load_actual"], 2) if not actuals.empty else None
delta  = (latest - prev) if (latest is not None and prev is not None) else None
nxt    = float(preds["predicted_load"].iloc[0]) if not preds.empty else None
pk     = int(preds["is_peak"].sum()) if not preds.empty and "is_peak" in preds.columns else 0
mae    = errors["abs_error"].mean() if not errors.empty else None
rr     = safe_last(actuals["renewable_ratio"]) if not actuals.empty and has_col(actuals, "renewable_ratio") else None

delta_str = f'{"↑" if delta and delta > 0 else "↓"} {abs(delta):,.0f} MW vs prev hour' if delta else "Live reading"
kpi3_cls  = "kpi3b" if pk > 0 else "kpi3a"
kpi3_val  = f"🔴 {pk} Peak Hours" if pk > 0 else "🟢 All Clear"

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="kpi-card kpi1">
        <div class="kpi-label">Current Load</div>
        <div class="kpi-value">{f"{latest:,.0f}" if latest else "—"}<span style="font-size:1rem;opacity:0.5"> MW</span></div>
        <div class="kpi-delta">{delta_str}</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card kpi2">
        <div class="kpi-label">Next Hour Forecast</div>
        <div class="kpi-value">{f"{nxt:,.0f}" if nxt else "—"}<span style="font-size:1rem;opacity:0.5"> MW</span></div>
        <div class="kpi-delta">{selected_model} prediction</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card {kpi3_cls}">
        <div class="kpi-label" style="opacity:0.65;font-size:0.6rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase">24h Peak Alert</div>
        <div class="kpi-value" style="font-size:1.15rem">{kpi3_val}</div>
        <div class="kpi-delta" style="opacity:0.6">Next {FORECAST_HOURS} hours</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card kpi4">
        <div class="kpi-label">{history_days}-Day Avg MAE</div>
        <div class="kpi-value">{f"{mae:,.0f}" if mae else "—"}<span style="font-size:1rem;opacity:0.5"> MW</span></div>
        <div class="kpi-delta">Model accuracy</div>
    </div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card kpi5">
        <div class="kpi-label">Renewable Ratio</div>
        <div class="kpi-value">{f"{rr:.1%}" if rr else "—"}</div>
        <div class="kpi-delta">Solar + Wind share</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)




# ═══════════════════════════════════════════════════════════════
# MODEL COMPARISON RESULTS
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="card-title">🤖 Model Comparison Results</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="card-sub">Compares Linear Regression, Random Forest, LightGBM, and XGBoost using MAE, RMSE, and R²</div>',
    unsafe_allow_html=True
)

comparison_file = Path("model_comparison_results.csv")

if comparison_file.exists():
    comparison_df = pd.read_csv(comparison_file)

    required_cols = {"Model", "MAE", "RMSE", "R2"}

    if required_cols.issubset(set(comparison_df.columns)):
        comparison_df = comparison_df.sort_values("RMSE", ascending=True).reset_index(drop=True)
        best_model = comparison_df.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""<div class="stat-tile">
                <div class="stat-tile-label">Best Model</div>
                <div class="stat-tile-value">{best_model['Model']}</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""<div class="stat-tile">
                <div class="stat-tile-label">Lowest RMSE</div>
                <div class="stat-tile-value">{best_model['RMSE']:,.0f} MW</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""<div class="stat-tile">
                <div class="stat-tile-label">MAE</div>
                <div class="stat-tile-value">{best_model['MAE']:,.0f} MW</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""<div class="stat-tile">
                <div class="stat-tile-label">R² Score</div>
                <div class="stat-tile-value">{best_model['R2']:.4f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 RMSE / MAE Chart", "📈 R² Chart", "📋 Results Table"])

        with tab1:
            fig_model_error = go.Figure()

            fig_model_error.add_trace(go.Bar(
                x=comparison_df["Model"],
                y=comparison_df["RMSE"],
                name="RMSE",
                marker_color=T["c_main"],
                text=comparison_df["RMSE"].round(0),
                textposition="outside"
            ))

            fig_model_error.add_trace(go.Bar(
                x=comparison_df["Model"],
                y=comparison_df["MAE"],
                name="MAE",
                marker_color=T["c_fore"],
                text=comparison_df["MAE"].round(0),
                textposition="outside"
            ))

            fig_model_error.update_layout(
                height=360,
                barmode="group",
                xaxis_title="Model",
                yaxis_title="Error (MW)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.01,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=T["plot_legend"])
                ),
                **PL
            )

            st.plotly_chart(fig_model_error, use_container_width=True)

        with tab2:
            fig_r2 = go.Figure()

            fig_r2.add_trace(go.Bar(
                x=comparison_df["Model"],
                y=comparison_df["R2"],
                name="R² Score",
                marker_color=T["c_ratio"],
                text=comparison_df["R2"].round(4),
                textposition="outside"
            ))

            fig_r2.update_layout(
                height=330,
                xaxis_title="Model",
                yaxis_title="R² Score",
                showlegend=False,
                **PL
            )

            # IMPORTANT:
            # PL already contains a yaxis setting.
            # So we update the y-axis range separately to avoid:
            # TypeError: update_layout() got multiple values for keyword argument 'yaxis'
            fig_r2.update_yaxes(
                range=[0, max(1.0, comparison_df["R2"].max() + 0.05)]
            )

            st.plotly_chart(fig_r2, use_container_width=True)

        with tab3:
            display_df = comparison_df.copy()
            display_df["MAE"] = display_df["MAE"].map(lambda x: f"{x:,.2f}")
            display_df["RMSE"] = display_df["RMSE"].map(lambda x: f"{x:,.2f}")
            display_df["R2"] = display_df["R2"].map(lambda x: f"{x:.4f}")
            st.dataframe(display_df, use_container_width=True, height=180)

        st.markdown(
            f'<div class="alert-ok">✓ <b>{best_model["Model"]}</b> is selected as the best model because it has the lowest RMSE.</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            ph(
                "⚠️",
                "Model comparison file has missing columns",
                "Expected columns: Model, MAE, RMSE, R2. Re-run train.py to regenerate model_comparison_results.csv."
            ),
            unsafe_allow_html=True
        )
else:
    st.markdown(
        ph(
            "🤖",
            "Model Comparison Not Available Yet",
            "Run python train.py first. It will create model_comparison_results.csv, then this dashboard section will appear."
        ),
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════
# FORECAST MODEL RADIO SELECTOR — placed below comparison
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="card-title">🤖 Forecast Model Selection</div>', unsafe_allow_html=True)
st.markdown('<div class="card-sub">Select a model below the comparison results. The main chart and forecast table update based on this choice.</div>', unsafe_allow_html=True)

# Style the radio to look clean and prominent
st.markdown(f"""
<style>
div[data-testid="stRadio"] > label {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {T['sub_color']};
    margin-bottom: 6px;
    display: block;
}}
div[data-testid="stRadio"] div[role="radiogroup"] {{
    display: flex;
    flex-direction: row;
    gap: 1rem;
}}
div[data-testid="stRadio"] label[data-baseweb="radio"] {{
    background: {T['tile_bg']};
    border: 1px solid {T['tile_border']};
    border-radius: 10px;
    padding: 0.75rem 1.4rem;
    cursor: pointer;
    transition: all 0.15s ease;
    flex: 1;
}}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{
    border: 2px solid #16a34a;
    background: linear-gradient(135deg, rgba(22,163,74,0.12), rgba(74,222,128,0.06));
    box-shadow: 0 3px 12px rgba(22,163,74,0.2);
}}
div[data-testid="stRadio"] span[data-testid="stMarkdownContainer"] p {{
    font-size: 0.88rem;
    font-weight: 600;
    color: {T['text_color']};
    margin: 0;
}}
</style>
""", unsafe_allow_html=True)

selected_model = st.radio(
    label="Prediction model",
    options=MODEL_OPTIONS,
    index=MODEL_OPTIONS.index(st.session_state.selected_forecast_model),
    format_func=lambda m: "🌿 LightGBM — Gradient Boosted Trees (fast, memory-efficient)" if m == "LightGBM"
                     else "⚡ XGBoost — Extreme Gradient Boosting (robust, high-accuracy)",
    horizontal=True,
    label_visibility="collapsed",
    key="model_radio",
)

# Sync session state and persist to disk so scheduler/dashboard use the same selection
st.session_state.selected_forecast_model = selected_model
write_selected_model(selected_model)

# Re-filter after the radio is rendered so the chart/table below immediately use the selection
preds  = filter_by_model(preds_all_models, selected_model)
errors = filter_by_model(errors_raw, selected_model)

if preds.empty and not preds_all_models.empty:
    st.markdown(
        f'<div class="alert-warn">&#9888; No predictions found for <b>{selected_model}</b>. '
        f'Selection saved. Make sure <code>train.py</code> and <code>scheduler.py</code> have run after both model files were created.</div>',
        unsafe_allow_html=True
    )
else:
    _model_color = "#16a34a" if selected_model == "LightGBM" else "#1e40af"
    st.markdown(
        f'<div class="alert-ok">&#10003; Dashboard is showing '
        f'<b style="color:{_model_color}">{selected_model}</b> predictions. '
        f'The main chart and forecast table below use this model.</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

st.divider()


# ═══════════════════════════════════════════════════════════════
# MAIN CHART
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="card-title">📈 Demand: Recent History + 24-Hour Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="card-sub">Actual load vs model forecast with peak-hour shading</div>', unsafe_allow_html=True)

if not actuals.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actuals["time"], y=actuals["total_load_actual"],
        mode="lines", name="Actual Demand",
        line=dict(color=T["c_main"], width=2.5),
        fill="tozeroy", fillcolor=T["c_main_fill"],
    ))
    if not preds.empty:
        fig.add_trace(go.Scatter(
            x=preds["target_time"], y=preds["predicted_load"],
            mode="lines+markers", name="Forecast",
            line=dict(color=T["c_fore"], width=2, dash="dash"),
            marker=dict(size=5, color=T["c_fore"]),
        ))
        for _, row in preds[preds["is_peak"] == 1].iterrows():
            t = pd.Timestamp(row["target_time"])
            fig.add_vrect(
                x0=(t - pd.Timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
                x1=(t + pd.Timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
                fillcolor=T["c_peak_fill"], line_width=0,
                annotation_text="⚠ Peak", annotation_position="top left",
                annotation_font=dict(color=T["c_thresh"], size=10),
            )
    fig.add_hline(y=PEAK_THRESHOLD, line_dash="dot",
                  line_color=T["c_thresh"], line_width=1.5,
                  annotation_text=f"Peak threshold · {PEAK_THRESHOLD:,.0f} MW",
                  annotation_position="bottom right",
                  annotation_font=dict(color=T["c_thresh"], size=10))
    now_s = now_utc.strftime("%Y-%m-%d %H:%M:%S")
    fig.add_shape(type="line", x0=now_s, x1=now_s, y0=0, y1=1,
                  xref="x", yref="paper", line=dict(color=T["c_now"], width=1, dash="dot"))
    fig.add_annotation(x=now_s, y=0.98, xref="x", yref="paper",
                       text="Now", showarrow=False, yanchor="top",
                       font=dict(color=T["c_now"], size=10))
    fig.update_layout(height=380, xaxis_title="Time (UTC)", yaxis_title="Total Load (MW)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                  bgcolor="rgba(0,0,0,0)", font=dict(color=T["plot_legend"])),
                      hovermode="x unified", **PL)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown(ph("⏳", "No Data Yet", "scheduler.py may still be fetching the first batch."), unsafe_allow_html=True)

st.divider()


# ═══════════════════════════════════════════════════════════════
# ROW 2 — Accuracy + Renewables
# ═══════════════════════════════════════════════════════════════
col_acc, col_ren = st.columns(2)

with col_acc:
    st.markdown('<div class="card-title">🎯 Forecast Accuracy</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Prediction error over selected history window</div>', unsafe_allow_html=True)
    if not errors.empty:
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("Absolute Error (MW)", "% Error"),
                              vertical_spacing=0.14)
        fig2.add_trace(go.Scatter(x=errors["time"], y=errors["abs_error"], mode="lines",
            line=dict(color=T["c_acc1"], width=1.5),
            fill="tozeroy", fillcolor=T["c_acc1_fill"]), row=1, col=1)
        fig2.add_trace(go.Scatter(x=errors["time"], y=errors["pct_error"], mode="lines",
            line=dict(color=T["c_acc2"], width=1.5),
            fill="tozeroy", fillcolor=T["c_acc2_fill"]), row=2, col=1)
        fig2.update_layout(height=260, showlegend=False, **PL)
        st.plotly_chart(fig2, use_container_width=True)
        s1, s2, s3 = st.columns(3)
        for col_w, lbl, val in [
            (s1, "Mean MAE",   f"{errors['abs_error'].mean():,.0f} MW"),
            (s2, "Max Error",  f"{errors['abs_error'].max():,.0f} MW"),
            (s3, "Mean % Err", f"{errors['pct_error'].mean():.2f}%"),
        ]:
            col_w.markdown(f"""<div class="stat-tile">
                <div class="stat-tile-label">{lbl}</div>
                <div class="stat-tile-value">{val}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(ph("📊", "No Error Data Yet",
            "Errors are computed once forecast hours have passed and actuals arrive."), unsafe_allow_html=True)

with col_ren:
    st.markdown('<div class="card-title">🌱 Renewable Generation</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Solar and wind output over selected window</div>', unsafe_allow_html=True)
    has_solar = has_col(actuals, "generation_solar")
    has_wind  = has_col(actuals, "generation_wind_onshore")
    has_ratio = has_col(actuals, "renewable_ratio")
    if not actuals.empty and (has_solar or has_wind):
        fig3 = go.Figure()
        if has_solar:
            fig3.add_trace(go.Bar(x=actuals["time"], y=actuals["generation_solar"],
                name="Solar", marker_color=T["c_solar"], opacity=0.9))
        if has_wind:
            fig3.add_trace(go.Scatter(x=actuals["time"], y=actuals["generation_wind_onshore"],
                mode="lines", name="Wind Onshore", line=dict(color=T["c_wind"], width=2)))
        fig3.update_layout(height=200, barmode="overlay", xaxis_title="Time (UTC)",
            yaxis_title="Generation (MW)",
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        bgcolor="rgba(0,0,0,0)", font=dict(color=T["plot_legend"])),
            hovermode="x unified", **PL)
        st.plotly_chart(fig3, use_container_width=True)
        if has_ratio:
            rdf = actuals.dropna(subset=["renewable_ratio"]).copy()
            fig3b = go.Figure()
            fig3b.add_trace(go.Scatter(x=rdf["time"], y=rdf["renewable_ratio"],
                mode="lines", line=dict(color=T["c_ratio"], width=1.5),
                fill="tozeroy", fillcolor=T["c_ratio_fill"]))
            fig3b.update_layout(height=90, showlegend=False,
                                xaxis_title="", yaxis_title="Ratio", **PL)
            st.plotly_chart(fig3b, use_container_width=True)
    else:
        st.markdown(ph("🌱", "Awaiting Generation Data",
            "Solar and wind data will appear once scheduler connects to ENTSO-E."), unsafe_allow_html=True)

st.divider()


# ═══════════════════════════════════════════════════════════════
# ROW 3 — Price + Weather
# ═══════════════════════════════════════════════════════════════
col_price, col_weather = st.columns(2)

with col_price:
    st.markdown('<div class="card-title">💶 Day-Ahead Price</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Market electricity price (€/MWh)</div>', unsafe_allow_html=True)
    if not actuals.empty and has_col(actuals, "price_day_ahead"):
        pdf = actuals.dropna(subset=["price_day_ahead"])
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=pdf["time"], y=pdf["price_day_ahead"],
            mode="lines", line=dict(color=T["c_price"], width=2),
            fill="tozeroy", fillcolor=T["c_price_fill"]))
        fig4.update_layout(height=200, xaxis_title="Time (UTC)",
                           yaxis_title="€ / MWh", showlegend=False, **PL)
        st.plotly_chart(fig4, use_container_width=True)
        lp  = safe_last(pdf["price_day_ahead"])
        avg = float(pdf["price_day_ahead"].mean())
        mx  = float(pdf["price_day_ahead"].max())
        mn  = float(pdf["price_day_ahead"].min())
        p1, p2, p3, p4 = st.columns(4)
        for col_w, lbl, val in [
            (p1, "Latest",              f"€{lp:.2f}" if lp else "—"),
            (p2, f"{history_days}d Avg", f"€{avg:.2f}"),
            (p3, f"{history_days}d High", f"€{mx:.2f}"),
            (p4, f"{history_days}d Low",  f"€{mn:.2f}"),
        ]:
            col_w.markdown(f"""<div class="stat-tile">
                <div class="stat-tile-label">{lbl}</div>
                <div class="stat-tile-value">{val}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(ph("💶", "Price Data Unavailable",
            "Day-ahead prices will populate once scheduler connects to ENTSO-E."), unsafe_allow_html=True)

with col_weather:
    st.markdown('<div class="card-title">🌡️ Weather · 5-City Average</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Temperature and humidity across Spain</div>', unsafe_allow_html=True)
    if not actuals.empty and has_col(actuals, "temp"):
        wd = actuals.dropna(subset=["temp"]).copy()
        wd["temp_c"] = wd["temp"] - 273.15
        fig5 = make_subplots(rows=1, cols=2,
                              subplot_titles=("Temperature (°C)", "Humidity (%)"),
                              horizontal_spacing=0.1)
        fig5.add_trace(go.Scatter(x=wd["time"], y=wd["temp_c"], mode="lines",
            line=dict(color=T["c_temp"], width=1.5),
            fill="tozeroy", fillcolor=T["c_temp_fill"]), row=1, col=1)
        if has_col(actuals, "humidity"):
            fig5.add_trace(go.Scatter(x=wd["time"], y=wd["humidity"], mode="lines",
                line=dict(color=T["c_hum"], width=1.5),
                fill="tozeroy", fillcolor=T["c_hum_fill"]), row=1, col=2)
        fig5.update_layout(height=200, showlegend=False, **PL)
        st.plotly_chart(fig5, use_container_width=True)
        lw   = wd.iloc[-1]
        hum  = lw.get("humidity", None)
        wspd = lw.get("wind_speed", None)
        w1, w2, w3 = st.columns(3)
        for col_w, lbl, val in [
            (w1, "Temperature", f"{lw['temp_c']:.1f} °C"),
            (w2, "Humidity",    f"{hum:.0f}%"    if hum  is not None and pd.notna(hum)  else "—"),
            (w3, "Wind Speed",  f"{wspd:.1f} m/s" if wspd is not None and pd.notna(wspd) else "—"),
        ]:
            col_w.markdown(f"""<div class="stat-tile">
                <div class="stat-tile-label">{lbl}</div>
                <div class="stat-tile-value">{val}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(ph("🌤️", "Weather Data Unavailable",
            "Temperature data will appear once scheduler fetches from OpenWeatherMap."), unsafe_allow_html=True)

st.divider()


# ═══════════════════════════════════════════════════════════════
# FORECAST TABLE
# ═══════════════════════════════════════════════════════════════
st.markdown(f'<div class="card-title">🔮 Next {FORECAST_HOURS}-Hour Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="card-sub">Hourly load predictions with peak classification</div>', unsafe_allow_html=True)

if not preds.empty:
    forecast_cols = ["target_time", "predicted_load", "is_peak", "made_at"]
    if "model" in preds.columns:
        forecast_cols.insert(0, "model")
    dp = preds[forecast_cols].copy()
    dp["target_time"]    = dp["target_time"].dt.strftime("%Y-%m-%d %H:%M UTC")
    dp["predicted_load"] = dp["predicted_load"].round(0).astype(int)
    dp["Status"]         = dp["is_peak"].apply(lambda x: "🔴 PEAK" if x else "🟢 Normal")
    dp = dp.rename(columns={"model": "Model",
                             "target_time": "Time (UTC)",
                             "predicted_load": "Predicted Load (MW)",
                             "made_at": "Forecast Made At"}).drop(columns=["is_peak"])
    st.dataframe(
        dp.style.apply(lambda row: [f"background-color:{T['table_peak']}"
                                    if row["Status"] == "🔴 PEAK" else "" for _ in row], axis=1),
        use_container_width=True, height=380,
    )
else:
    st.markdown(ph("🔮", "Forecast Not Yet Generated",
        "Wait for the scheduler to complete its first run."), unsafe_allow_html=True)

st.divider()


# ═══════════════════════════════════════════════════════════════
# 💰 FINANCIAL GOAL TRACKER
# ═══════════════════════════════════════════════════════════════
def compute_financial(adf, mg, ag, kwh):
    if adf.empty or not has_col(adf, "price_day_ahead"):
        return None
    now = pd.Timestamp.now(tz="UTC")
    mdf = adf[
        (adf["time"].dt.year  == now.year) &
        (adf["time"].dt.month == now.month)
    ].dropna(subset=["price_day_ahead"])
    if mdf.empty: return None
    avg_mwh     = float(mdf["price_day_ahead"].mean())
    avg_kwh     = avg_mwh / 1000.0
    dim         = calendar.monthrange(now.year, now.month)[1]
    h_elapsed   = len(mdf)
    d_elapsed   = h_elapsed / 24.0
    kwh_hr      = kwh / (dim * 24)
    cost_so_far = kwh_hr * h_elapsed * avg_kwh
    proj_m      = (cost_so_far / d_elapsed * dim) if d_elapsed > 0 else kwh * avg_kwh
    proj_a      = proj_m * 12
    hdf         = mdf.sort_values("time").copy()
    hdf["hr_cost"]   = kwh_hr * (hdf["price_day_ahead"] / 1000)
    hdf["cum_cost"]  = hdf["hr_cost"].cumsum()
    hdf["goal_pace"] = (mg / (dim * 24)) * (np.arange(len(hdf)) + 1)
    return dict(avg_mwh=avg_mwh, avg_kwh=avg_kwh, cost_so_far=cost_so_far,
                proj_m=proj_m, proj_a=proj_a, d_elapsed=d_elapsed, dim=dim,
                pct_m=proj_m/mg*100 if mg>0 else 0,
                pct_a=proj_a/ag*100 if ag>0 else 0, hdf=hdf)

st.markdown('<div class="card-title">💰 Financial Goal Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="card-sub">Personal electricity cost estimate · Adjust targets in the sidebar</div>', unsafe_allow_html=True)

fin = compute_financial(actuals, monthly_goal, annual_goal, user_kwh_monthly)

if fin is None:
    st.markdown(ph("💰", "Financial Data Not Available",
        "Price data will appear after the first scheduler run."), unsafe_allow_html=True)
else:
    over_m = fin["proj_m"] > monthly_goal
    over_a = fin["proj_a"] > annual_goal

    if over_m or over_a:
        parts = []
        if over_m: parts.append(f"Monthly projected <b>€{fin['proj_m']:.2f}</b> exceeds €{monthly_goal:.2f} goal by <b>€{fin['proj_m']-monthly_goal:.2f}</b>.")
        if over_a: parts.append(f"Annual projection <b>€{fin['proj_a']:.2f}</b> exceeds €{annual_goal:.2f} goal by <b>€{fin['proj_a']-annual_goal:.2f}</b>.")
        st.markdown(f'<div class="alert-err">⚠ <b>Budget Alert — </b>{" ".join(parts)}</div>', unsafe_allow_html=True)
    elif fin["pct_m"] > 80 or fin["pct_a"] > 80:
        st.markdown(f'<div class="alert-warn">⚠ <b>Approaching Limit — </b>Monthly projection at <b>{fin["pct_m"]:.0f}%</b> of goal.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-ok">✓ <b>On Track — </b>Monthly projection at <b>{fin["pct_m"]:.0f}%</b> of your €{monthly_goal:.2f} goal.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    days_left  = max(0, fin["dim"] - fin["d_elapsed"])
    daily_left = (monthly_goal - fin["cost_so_far"]) / days_left if days_left > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="fin-card">
            <div class="fin-label">Spent So Far</div>
            <div class="fin-value">€{fin['cost_so_far']:.2f}</div>
            <div class="fin-sub">{fin['d_elapsed']:.1f} of {fin['dim']} days elapsed</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        dm  = fin["proj_m"] - monthly_goal
        cls = "fin-neg" if dm > 0 else "fin-pos"
        st.markdown(f"""<div class="fin-card">
            <div class="fin-label">Projected This Month</div>
            <div class="fin-value">€{fin['proj_m']:.2f}</div>
            <div class="{cls}">{'↑' if dm>0 else '↓'} €{abs(dm):.2f} vs goal</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        da  = fin["proj_a"] - annual_goal
        cls = "fin-neg" if da > 0 else "fin-pos"
        st.markdown(f"""<div class="fin-card">
            <div class="fin-label">Projected This Year</div>
            <div class="fin-value">€{fin['proj_a']:.2f}</div>
            <div class="{cls}">{'↑' if da>0 else '↓'} €{abs(da):.2f} vs goal</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="fin-card">
            <div class="fin-label">Daily Budget Left</div>
            <div class="fin-value">€{daily_left:.2f}</div>
            <div class="fin-sub">€{fin['avg_kwh']:.4f}/kWh · {days_left:.0f} days left</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    def pcls(p): return "prog-red" if p>=100 else ("prog-amber" if p>=80 else "prog-green")
    pb1, pb2 = st.columns(2)
    with pb1:
        w = min(fin["pct_m"], 100)
        st.markdown(f"""
        <p class="prog-label">Monthly Budget &nbsp;<span class="prog-pct">{fin['pct_m']:.1f}% used</span></p>
        <div class="prog-wrap"><div class="{pcls(fin['pct_m'])}" style="width:{w}%"></div></div>
        <p class="prog-sub">€{fin['cost_so_far']:.2f} spent · €{monthly_goal:.2f} goal</p>
        """, unsafe_allow_html=True)
    with pb2:
        w = min(fin["pct_a"], 100)
        st.markdown(f"""
        <p class="prog-label">Annual Budget &nbsp;<span class="prog-pct">{fin['pct_a']:.1f}% projected</span></p>
        <div class="prog-wrap"><div class="{pcls(fin['pct_a'])}" style="width:{w}%"></div></div>
        <p class="prog-sub">€{fin['proj_a']:.2f} projected · €{annual_goal:.2f} goal</p>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ch1, ch2 = st.columns([3, 2])
    with ch1:
        st.markdown('<div class="card-title">Cumulative Spend vs Budget Pace</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Actual cost accumulation vs linear budget target this month</div>', unsafe_allow_html=True)
        hdf = fin["hdf"]
        sc  = T["c_spend_red"] if over_m else T["c_spend"]
        fr  = T["c_spend_fill_red"] if over_m else T["c_spend_fill"]
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=hdf["time"], y=hdf["goal_pace"], mode="lines",
            name="Budget Pace", line=dict(color=T["sub_color"], width=1.5, dash="dot")))
        fig_c.add_trace(go.Scatter(x=hdf["time"], y=hdf["cum_cost"], mode="lines",
            name="Cumulative Spend", line=dict(color=sc, width=2.5),
            fill="tozeroy", fillcolor=fr))
        fig_c.add_hline(y=monthly_goal, line_dash="dash",
            line_color=T["c_goal_line"], line_width=1.2,
            annotation_text=f"Goal €{monthly_goal:.2f}",
            annotation_position="top right",
            annotation_font=dict(color=T["c_goal_line"], size=10))
        eom = pd.Timestamp(year=now_utc.year, month=now_utc.month,
                           day=fin["dim"], hour=23, tz="UTC")
        fig_c.add_trace(go.Scatter(x=[eom], y=[fin["proj_m"]], mode="markers+text",
            name="EOM Projection",
            marker=dict(color=sc, size=9, symbol="diamond",
                        line=dict(color="white", width=1.5)),
            text=[f"  €{fin['proj_m']:.2f}"],
            textposition="middle right",
            textfont=dict(color=T["text_color"], size=11)))
        fig_c.update_layout(height=280, xaxis_title="Date", yaxis_title="Cost (€)",
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        bgcolor="rgba(0,0,0,0)", font=dict(color=T["plot_legend"])),
            hovermode="x unified", **PL)
        st.plotly_chart(fig_c, use_container_width=True)

    with ch2:
        st.markdown('<div class="card-title">Goal vs Projection</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Monthly and annual comparison</div>', unsafe_allow_html=True)
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(name="Goal", x=["Monthly", "Annual"],
            y=[monthly_goal, annual_goal],
            marker_color=T["c_goal_bar"],
            marker_line=dict(color=T["c_bar_border"], width=1.5),
            text=[f"€{monthly_goal:.0f}", f"€{annual_goal:.0f}"],
            textposition="outside", textfont=dict(color=T["text_color"], size=11)))
        fig_b.add_trace(go.Bar(name="Projected", x=["Monthly", "Annual"],
            y=[fin["proj_m"], fin["proj_a"]],
            marker_color=[T["c_spend_red"] if over_m else T["c_spend"],
                          T["c_spend_red"] if over_a else T["c_spend"]],
            text=[f"€{fin['proj_m']:.0f}", f"€{fin['proj_a']:.0f}"],
            textposition="outside", textfont=dict(color=T["text_color"], size=11)))
        fig_b.update_layout(barmode="group", height=280, yaxis_title="€",
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        bgcolor="rgba(0,0,0,0)", font=dict(color=T["plot_legend"])),
            **PL)
        st.plotly_chart(fig_b, use_container_width=True)

    st.caption(f"Methodology: avg market price €{fin['avg_mwh']:.2f}/MWh × your {user_kwh_monthly:.0f} kWh/month estimate.")


# ═══════════════════════════════════════════════════════════════
# RAW DATA
# ═══════════════════════════════════════════════════════════════
if show_raw:
    st.divider()
    st.markdown('<div class="card-title">🗄️ Raw Data</div>', unsafe_allow_html=True)
    with st.expander("Actuals"):
        st.dataframe(actuals, use_container_width=True)
    with st.expander("Predictions"):
        st.dataframe(preds, use_container_width=True)
    with st.expander("Errors"):
        st.dataframe(errors, use_container_width=True)