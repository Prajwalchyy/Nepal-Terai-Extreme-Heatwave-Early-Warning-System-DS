"""
================================================================================
Nepal Terai Extreme Heatwave Prediction System
Production Interactive Meteorological & Disaster Early-Warning Dashboard
================================================================================
Domain: Disaster Risk Reduction (DRR) & Meteorological Forecasting
Target Region: 21 Lowland Terai Districts, Nepal
Engine: Multi-Champion ML Pipeline (Random Forest + LightGBM + Ridge + Specialist)
================================================================================
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Ensure UTF-8 output encoding across Windows shells
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.predict import (
    predict_heatwave,
    VALID_DISTRICTS,
    DISTRICT_METADATA,
    HEATWAVE_ALERT_TEMP,
    TIER_1_MODERATE_MIN,
    TIER_2_SEVERE_MIN,
    TIER_3_EXTREME_MIN,
)

# ------------------------------------------------------------------------------
# 1. Page Configuration & Professional Styling System
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Nepal Terai Extreme Heatwave Prediction System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def load_css(file_path: Path):
    """Loads external CSS styling into Streamlit application."""
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(BASE_DIR / "style.css")

# ------------------------------------------------------------------------------
# 2. Helpers: Physical & Bio-Meteorological Calculations
# ------------------------------------------------------------------------------
def compute_feels_like(temp_c: float, humidity_pct: float, wind_speed_ms: float) -> float:
    """Calculates the Steadman Heat Index (Apparent Feels-Like Temperature)."""
    e_a = (humidity_pct / 100.0) * 6.1078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    feels_like = temp_c + (0.33 * e_a) - (0.70 * wind_speed_ms) - 4.00
    return round(float(feels_like), 1)

def evaluate_today_observed_status(today_max: float) -> dict:
    """Evaluates Today's ground-truth observed heatwave status using physical DHM thresholds."""
    if today_max >= 45.0:
        return {
            "status_text": "Hot (Active Heatwave)",
            "status_short": "Hot",
            "is_hot": True,
            "threat_name": "Extreme Heatwave",
            "threat_label": "Extreme Heatwave (Red Alert)",
            "tier_num": 3,
            "color": "#dc2626",
            "bg": "#fef2f2",
            "border": "#fecaca",
            "text": "#991b1b",
            "benchmark": "≥ 45.0°C (Catastrophic Danger)",
        }
    elif today_max >= 42.0:
        return {
            "status_text": "Hot (Active Heatwave)",
            "status_short": "Hot",
            "is_hot": True,
            "threat_name": "Severe Heatwave",
            "threat_label": "Severe Heatwave (Orange Alert)",
            "tier_num": 2,
            "color": "#ea580c",
            "bg": "#fff7ed",
            "border": "#fed7aa",
            "text": "#9a3412",
            "benchmark": "42.0°C – 44.9°C (High Risk)",
        }
    elif today_max >= 40.0:
        return {
            "status_text": "Hot (Active Heatwave)",
            "status_short": "Hot",
            "is_hot": True,
            "threat_name": "Moderate Heatwave",
            "threat_label": "Moderate Heatwave (Yellow Alert)",
            "tier_num": 1,
            "color": "#d97706",
            "bg": "#fffbeb",
            "border": "#fde68a",
            "text": "#92400e",
            "benchmark": "40.0°C – 41.9°C (Elevated Risk)",
        }
    else:
        return {
            "status_text": "Normal Conditions",
            "status_short": "Normal",
            "is_hot": False,
            "threat_name": "Normal",
            "threat_label": "Normal (Low Thermal Stress)",
            "tier_num": 0,
            "color": "#059669",
            "bg": "#ecfdf5",
            "border": "#a7f3d0",
            "text": "#065f46",
            "benchmark": "< 40.0°C (Non-Heatwave)",
        }

def get_threat_profile(tier_num: int) -> dict:
    """Returns standardized threat metadata."""
    profiles = {
        0: {
            "name": "Normal",
            "label": "Normal (Low Thermal Stress)",
            "color": "#059669",
            "bg": "#ecfdf5",
            "border": "#a7f3d0",
            "text": "#065f46",
            "benchmark": "< 40.0°C (Non-Heatwave)",
        },
        1: {
            "name": "Moderate Heatwave",
            "label": "Moderate Heatwave (Yellow Alert)",
            "color": "#d97706",
            "bg": "#fffbeb",
            "border": "#fde68a",
            "text": "#92400e",
            "benchmark": "40.0°C – 41.9°C (Elevated Risk)",
        },
        2: {
            "name": "Severe Heatwave",
            "label": "Severe Heatwave (Orange Alert)",
            "color": "#ea580c",
            "bg": "#fff7ed",
            "border": "#fed7aa",
            "text": "#9a3412",
            "benchmark": "42.0°C – 44.9°C (High Danger)",
        },
        3: {
            "name": "Extreme Heatwave",
            "label": "Extreme Heatwave (Red Alert)",
            "color": "#dc2626",
            "bg": "#fef2f2",
            "border": "#fecaca",
            "text": "#991b1b",
            "benchmark": "≥ 45.0°C (Catastrophic Danger)",
        },
    }
    return profiles.get(tier_num, profiles[0])

# Current time in Nepal Time (NPT: UTC+5:45)
npt_time = datetime.utcnow() + timedelta(hours=5, minutes=45)
live_time_str = npt_time.strftime("%A, %d %B %Y • %H:%M:%S NPT")

# ------------------------------------------------------------------------------
# 3. Project Header
# ------------------------------------------------------------------------------
st.html(
    f"""
    <div class="ews-header">
        <h1 class="ews-title">Nepal Terai Extreme Heatwave Prediction System</h1>
        <div class="ews-clock-badge">
            <div id="ews-live-clock" class="ews-live-time mono">{live_time_str}</div>
        </div>
    </div>
    <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3C/svg%3E" style="display:none;" onload="
        (function() {{
            function updateClock() {{
                var el = document.getElementById('ews-live-clock');
                if (!el) return;
                var now = new Date();
                var utc = now.getTime() + (now.getTimezoneOffset() * 60000);
                var npt = new Date(utc + 20700000);
                var days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
                var months = ['January','February','March','April','May','June','July','August','September','October','November','December'];
                var h = String(npt.getHours()).padStart(2, '0');
                var m = String(npt.getMinutes()).padStart(2, '0');
                var s = String(npt.getSeconds()).padStart(2, '0');
                el.textContent = days[npt.getDay()] + ', ' + npt.getDate() + ' ' + months[npt.getMonth()] + ' ' + npt.getFullYear() + ' • ' + h + ':' + m + ':' + s + ' NPT';
            }}
            updateClock();
            if (window._ews_clock_timer) clearInterval(window._ews_clock_timer);
            window._ews_clock_timer = setInterval(updateClock, 1000);
        }})();
    " />
    <script>
        (function() {{
            function updateClock() {{
                var el = document.getElementById('ews-live-clock');
                if (!el) return;
                var now = new Date();
                var utc = now.getTime() + (now.getTimezoneOffset() * 60000);
                var npt = new Date(utc + 20700000);
                var days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
                var months = ['January','February','March','April','May','June','July','August','September','October','November','December'];
                var h = String(npt.getHours()).padStart(2, '0');
                var m = String(npt.getMinutes()).padStart(2, '0');
                var s = String(npt.getSeconds()).padStart(2, '0');
                el.textContent = days[npt.getDay()] + ', ' + npt.getDate() + ' ' + months[npt.getMonth()] + ' ' + npt.getFullYear() + ' • ' + h + ':' + m + ':' + s + ' NPT';
            }}
            updateClock();
            if (window._ews_clock_timer) clearInterval(window._ews_clock_timer);
            window._ews_clock_timer = setInterval(updateClock, 1000);
        }})();
    </script>
    """,
    unsafe_allow_javascript=True,
)

# ------------------------------------------------------------------------------
# 4. Two Prediction Modes (Clearly Distinguishable Tabs)
# ------------------------------------------------------------------------------
district_labels = {
    d: f"{d} ({DISTRICT_METADATA[d]['Regional_Hub']}, {DISTRICT_METADATA[d]['Province']})"
    for d in VALID_DISTRICTS
}

mode_tab1, mode_tab2 = st.tabs([
    "📍 Mode 1: District-Based Prediction (Live Telemetry)",
    "🧪 Mode 2: Custom Weather Input (Scenario Analysis)",
])

# State initialization
if "active_district" not in st.session_state:
    st.session_state["active_district"] = "Banke"
if "pred_mode" not in st.session_state:
    st.session_state["pred_mode"] = "live"

# --- MODE 1: District-Based Prediction ---
with mode_tab1:
    col_d1, col_d2, col_d3 = st.columns([2.5, 1.5, 1.2])
    
    with col_d1:
        sel_dist_mode1 = st.selectbox(
            "Select District",
            options=VALID_DISTRICTS,
            format_func=lambda d: district_labels[d],
            index=VALID_DISTRICTS.index(st.session_state["active_district"]) if st.session_state["active_district"] in VALID_DISTRICTS else 0,
            key="dist_mode1",
            help="Pulls automated weather observations and 14-day history directly from satellite & station feeds.",
        )
    
    with col_d2:
        d_meta = DISTRICT_METADATA[sel_dist_mode1]
        st.html(
            f"""
            <div style="padding-top:28px; font-size:12.5px; color:#475569; line-height:1.4;">
                <b>Sub-Region:</b> {d_meta['Terai_Region']}<br>
                <b>GPS:</b> <span class="mono">{d_meta['Latitude']}°N, {d_meta['Longitude']}°E</span>
            </div>
            """
        )
        
    with col_d3:
        st.html("<div style='height: 28px;'></div>")
        btn_mode1 = st.button("🚀 Run Forecast", type="primary", use_container_width=True, key="btn_mode1")

# --- MODE 2: Custom Weather Input ---
with mode_tab2:
    st.html("<div style='font-size:13px; color:#475569; margin-bottom:12px;'>Enter custom atmospheric observations to test model predictions under extreme climate stress scenarios:</div>")
    
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        sel_dist_mode2 = st.selectbox(
            "Target District",
            options=VALID_DISTRICTS,
            format_func=lambda d: district_labels[d],
            index=VALID_DISTRICTS.index(st.session_state["active_district"]) if st.session_state["active_district"] in VALID_DISTRICTS else 0,
            key="dist_mode2",
        )
    with col_c2:
        custom_date = st.date_input("Observation Date", value=datetime.now().date(), key="custom_date")

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        custom_max_t = st.slider("Maximum Temp (°C)", min_value=20.0, max_value=55.0, value=43.5, step=0.1, key="cust_max_t")
    with col_w2:
        custom_min_t = st.slider("Minimum Temp (°C)", min_value=10.0, max_value=38.0, value=28.0, step=0.1, key="cust_min_t")
    with col_w3:
        custom_rh = st.slider("Humidity (%)", min_value=5.0, max_value=100.0, value=35.0, step=1.0, key="cust_rh")

    col_w4, col_w5, col_btn_wrap = st.columns([1.2, 1.2, 1.0])
    with col_w4:
        custom_wind = st.slider("Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=3.5, step=0.1, key="cust_wind")
    with col_w5:
        custom_precip = st.slider("Precipitation (mm)", min_value=0.0, max_value=150.0, value=0.0, step=0.5, key="cust_precip")
    with col_btn_wrap:
        st.html("<div style='height: 28px;'></div>")
        btn_mode2 = st.button("🧪 Predict Custom Scenario", type="primary", use_container_width=True, key="btn_mode2")

# ------------------------------------------------------------------------------
# 5. Prediction Execution & State Management
# ------------------------------------------------------------------------------
# Determine which mode was triggered
if btn_mode1:
    st.session_state["pred_mode"] = "live"
    st.session_state["active_district"] = sel_dist_mode1
elif btn_mode2:
    st.session_state["pred_mode"] = "custom"
    st.session_state["active_district"] = sel_dist_mode2

# Execute prediction if triggered or not in session state
run_prediction = (btn_mode1 or btn_mode2) or ("prediction_result" not in st.session_state)

if run_prediction:
    target_district = st.session_state["active_district"]
    use_live = (st.session_state.get("pred_mode", "live") == "live")

    # Mode 2 Input Validation
    if not use_live:
        if custom_min_t >= custom_max_t:
            st.error(f"⚠️ Validation Error: Minimum temperature ({custom_min_t}°C) cannot be greater than or equal to Maximum temperature ({custom_max_t}°C).")
            st.stop()
        arg_date = custom_date.strftime("%Y-%m-%d")
        arg_max_t = custom_max_t
        arg_min_t = custom_min_t
        arg_rh = custom_rh
        arg_wind = custom_wind
        arg_precip = custom_precip
    else:
        arg_date = datetime.now().strftime("%Y-%m-%d")
        arg_max_t = None
        arg_min_t = None
        arg_rh = None
        arg_wind = None
        arg_precip = None

    with st.spinner(f"Ingesting atmospheric parameters and executing multi-champion models for {target_district}..."):
        try:
            res = predict_heatwave(
                district=target_district,
                date_str=arg_date,
                max_temp_c=arg_max_t,
                min_temp_c=arg_min_t,
                humidity_pct=arg_rh,
                wind_speed_ms=arg_wind,
                precipitation_mm=arg_precip,
                use_live_api=use_live,
            )
            st.session_state["prediction_result"] = res
            st.session_state["last_district"] = target_district
        except Exception as e:
            st.error(f"Forecast Generation Failed: {e}")
            st.stop()

result = st.session_state["prediction_result"]
w = result["input_weather"]
district_name = result["district"]
d_info = DISTRICT_METADATA[district_name]

# Dates
obs_date = datetime.strptime(result["forecast_date"], "%Y-%m-%d").date()
tomorrow_date = obs_date + timedelta(days=1)
today_str = obs_date.strftime("%d %b %Y")
tomorrow_str = tomorrow_date.strftime("%d %b %Y")

# Today's deterministic ground-truth status
today_max = w["max_temp_c"]
today_eval = evaluate_today_observed_status(today_max)

# Tomorrow's AI Model predictions
is_alert_tomorrow = result["heatwave_alert"]
prob_tomorrow = result["alert_probability"]
tier_tomorrow = result["severity_tier"]
pred_temp_tomorrow = result["predicted_max_temp_c"]
tomorrow_threat = get_threat_profile(tier_tomorrow)

# Temperature shift
delta_t = round(pred_temp_tomorrow - today_max, 2)
delta_sign = "+" if delta_t >= 0 else ""
delta_str = f"{delta_sign}{delta_t:.1f}°C"

# ------------------------------------------------------------------------------
# 6. Three Main Prediction Results (Main Visual Focus)
# ------------------------------------------------------------------------------
st.markdown("---")
st.html("<div class='section-title-wrap'>🎯 Tomorrow's Heatwave Forecast (24-Hour AI Prediction)</div>")

col_r1, col_r2, col_r3 = st.columns(3)

# --- Heatwave Status ---
with col_r1:
    if is_alert_tomorrow:
        hw_status_text = "HOT"
        hw_sub_label = "Heatwave Alert Active"
        hw_color = "#dc2626"
        hw_bg = "#fef2f2"
        hw_border = "#fecaca"
        hw_icon = "⚠️"
    else:
        hw_status_text = "NORMAL"
        hw_sub_label = "No Heatwave Expected"
        hw_color = "#059669"
        hw_bg = "#ecfdf5"
        hw_border = "#a7f3d0"
        hw_icon = "✅"

    st.html(
        f"""
        <div class="result-card" style="border-top: 5px solid {hw_color};">
            <div class="result-card-kicker">Heatwave Status • Tomorrow ({tomorrow_str})</div>
            <div class="status-display-large" style="color: {hw_color};">
                <span>{hw_icon}</span>
                <span>{hw_status_text}</span>
            </div>
            <div style="font-size:13.5px; font-weight:700; color:{hw_color}; margin-bottom:12px;">
                {hw_sub_label}
            </div>
            <div style="font-size:12.5px; font-weight:600; color:#475569;">
                Heatwave Probability: <b class="mono" style="color:{hw_color};">{prob_tomorrow * 100:.1f}%</b>
            </div>
        </div>
        """
    )

# --- Heatwave Threat Level ---
with col_r2:
    st.html(
        f"""
        <div class="result-card" style="border-top: 5px solid {tomorrow_threat['color']};">
            <div class="result-card-kicker">Heatwave Threat Level • Tomorrow ({tomorrow_str})</div>
            <div style="margin-top: 14px;">
                <div class="threat-badge-pill" style="background:{tomorrow_threat['bg']}; color:{tomorrow_threat['text']}; border:1.5px solid {tomorrow_threat['border']}; margin-bottom:0;">
                    <span>●</span> <span>{tomorrow_threat['label']}</span>
                </div>
            </div>
        </div>
        """
    )

# --- Predicted Temperature ---
with col_r3:
    temp_color = tomorrow_threat['color']
    trend_bg = "#fee2e2" if delta_t > 0.5 else ("#ecfdf5" if delta_t < -0.5 else "#f1f5f9")
    trend_color = "#991b1b" if delta_t > 0.5 else ("#065f46" if delta_t < -0.5 else "#475569")

    st.html(
        f"""
        <div class="result-card" style="border-top: 5px solid {temp_color};">
            <div class="result-card-kicker">Predicted High Temperature • Tomorrow ({tomorrow_str})</div>
            <div class="temp-hero-display">
                <span style="font-size:32px; margin-right:4px;">🌡️</span>
                <span class="temp-val-hero mono" style="color: {temp_color};">{pred_temp_tomorrow:.2f}</span>
                <span class="temp-unit-hero">°C</span>
            </div>
            <div class="trend-indicator-pill" style="background:{trend_bg}; color:{trend_color};">
                <span>24h Trend: <b>{delta_str}</b> vs today ({today_max:.1f}°C)</span>
            </div>
        </div>
        """
    )

# ------------------------------------------------------------------------------
# 7. Today vs Tomorrow Comparison Section
# ------------------------------------------------------------------------------
today_status_badge = (
    f'<span style="background:{today_eval["bg"]}; color:{today_eval["text"]}; '
    f'border:1px solid {today_eval["border"]}; padding:4px 10px; border-radius:999px; '
    f'font-weight:800; font-size:12px;">'
    f'{"⚠️ " + today_eval["status_short"] if today_eval["is_hot"] else "✅ " + today_eval["status_short"]}'
    f'</span>'
)

tomorrow_status_badge = (
    f'<span style="background:{"#fef2f2" if is_alert_tomorrow else "#ecfdf5"}; '
    f'color:{"#991b1b" if is_alert_tomorrow else "#065f46"}; '
    f'border:1px solid {"#fecaca" if is_alert_tomorrow else "#a7f3d0"}; '
    f'padding:4px 10px; border-radius:999px; font-weight:800; font-size:12px;">'
    f'{"⚠️ HOT" if is_alert_tomorrow else "✅ NORMAL"} ({prob_tomorrow * 100:.1f}%)'
    f'</span>'
)

today_threat_badge = (
    f'<span style="background:{today_eval["bg"]}; color:{today_eval["text"]}; '
    f'border:1px solid {today_eval["border"]}; padding:4px 10px; border-radius:999px; '
    f'font-weight:700; font-size:12px;">'
    f'{today_eval["threat_label"]}'
    f'</span>'
)

tomorrow_threat_badge = (
    f'<span style="background:{tomorrow_threat["bg"]}; color:{tomorrow_threat["text"]}; '
    f'border:1px solid {tomorrow_threat["border"]}; padding:4px 10px; border-radius:999px; '
    f'font-weight:700; font-size:12px;">'
    f'{tomorrow_threat["label"]}'
    f'</span>'
)

st.html(
    f"""
    <div class="comparison-container">
        <div class="comparison-header">
            <div>⚖️ Today vs Tomorrow Meteorological Comparison • {district_name}</div>
        </div>
        <div class="comparison-grid">
            <div class="comp-pane" style="border-left: 4px solid {today_eval['color']};">
                <div class="comp-pane-title">
                    <span>📅 TODAY ({today_str})</span>
                </div>
                <div class="comp-row">
                    <span class="comp-label">Heatwave Status</span>
                    <span class="comp-val">{today_status_badge}</span>
                </div>
                <div class="comp-row">
                    <span class="comp-label">Heatwave Threat Level</span>
                    <span class="comp-val">{today_threat_badge}</span>
                </div>
                <div class="comp-row">
                    <span class="comp-label">Max Temperature</span>
                    <span class="comp-val mono" style="font-size:16px; color:{today_eval['color']}; font-weight:800;">{today_max:.1f} °C</span>
                </div>
            </div>
            <div class="comp-pane" style="border-left: 4px solid {tomorrow_threat['color']};">
                <div class="comp-pane-title">
                    <span>🔮 TOMORROW PREDICTION ({tomorrow_str})</span>
                </div>
                <div class="comp-row">
                    <span class="comp-label">Heatwave Status</span>
                    <span class="comp-val">{tomorrow_status_badge}</span>
                </div>
                <div class="comp-row">
                    <span class="comp-label">Heatwave Threat Level</span>
                    <span class="comp-val">{tomorrow_threat_badge}</span>
                </div>
                <div class="comp-row">
                    <span class="comp-label">Max Temperature</span>
                    <span class="comp-val mono" style="font-size:16px; color:{tomorrow_threat['color']}; font-weight:800;">{pred_temp_tomorrow:.2f} °C <span style="font-size:12px; font-weight:600; color:{trend_color};">({delta_str})</span></span>
                </div>
            </div>
        </div>
    </div>
    """
)

# ------------------------------------------------------------------------------
# 8. Today's Weather Conditions
# ------------------------------------------------------------------------------
feels_like_max = compute_feels_like(w["max_temp_c"], w["humidity_pct"], w["wind_speed_ms"])
diurnal_diff = round(w["max_temp_c"] - w["min_temp_c"], 1)

st.html(
    f"""
    <div class="conditions-container">
        <div class="section-title-wrap">
            <span>📊 Today's Weather Conditions • {district_name} ({today_str})</span>
        </div>
        <div class="weather-tiles-grid">
            <div class="weather-tile">
                <div class="tile-kicker">Max Temperature</div>
                <div class="tile-value-num mono">{w['max_temp_c']:.1f} <span style="font-size:15px; color:#64748b;">°C</span></div>
                <div class="tile-footnote">Daytime peak thermometer</div>
            </div>
            <div class="weather-tile">
                <div class="tile-kicker">Min Temperature</div>
                <div class="tile-value-num mono">{w['min_temp_c']:.1f} <span style="font-size:15px; color:#64748b;">°C</span></div>
                <div class="tile-footnote">Diurnal Gap: <b class="mono">+{diurnal_diff}°C</b></div>
            </div>
            <div class="weather-tile">
                <div class="tile-kicker">Feels-Like Temp</div>
                <div class="tile-value-num mono" style="color:#ea580c;">{feels_like_max:.1f} <span style="font-size:15px; color:#64748b;">°C</span></div>
                <div class="tile-footnote">Steadman Apparent Index</div>
            </div>
            <div class="weather-tile">
                <div class="tile-kicker">Relative Humidity</div>
                <div class="tile-value-num mono">{w['humidity_pct']:.1f} <span style="font-size:15px; color:#64748b;">%</span></div>
                <div class="tile-footnote">Peak afternoon moisture</div>
            </div>
            <div class="weather-tile">
                <div class="tile-kicker">Wind Speed</div>
                <div class="tile-value-num mono">{w['wind_speed_ms']:.2f} <span style="font-size:15px; color:#64748b;">m/s</span></div>
                <div class="tile-footnote">Atmospheric circulation</div>
            </div>
            <div class="weather-tile">
                <div class="tile-kicker">Precipitation</div>
                <div class="tile-value-num mono">{w['precipitation_mm']:.1f} <span style="font-size:15px; color:#64748b;">mm</span></div>
                <div class="tile-footnote">24-hour recorded rainfall</div>
            </div>
        </div>
    </div>
    """
)

# ------------------------------------------------------------------------------
# 9. Operational Footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.html(
    """
    <div style="text-align: center; color: #64748b; font-size: 12.5px; line-height: 1.6; margin-bottom: 24px;">
        <b>Nepal Terai Extreme Heatwave Prediction System (2000–2026)</b><br>
        Department of Hydrology & Meteorology (DHM) and National Disaster Risk Reduction & Management Authority (NDRRMA) Operations Alignment.<br>
        Multi-Champion Architecture: Random Forest (Flag) · LightGBM (Severity) · Ridge (Temperature) · Specialist Consistency Enforcer.
    </div>
    """
)
