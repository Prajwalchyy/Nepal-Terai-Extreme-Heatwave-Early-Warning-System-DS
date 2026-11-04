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
from datetime import datetime, timedelta, timezone

# Ensure UTF-8 output encoding across Windows shells safely
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import streamlit as st
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.predict import (
    predict_heatwave,
    fetch_live_weather,
    DISTRICT_CLIMATOLOGY,
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

@st.cache_data(ttl=600, show_spinner=False)
def fetch_today_api_weather(district: str) -> dict:
    """
    Fetches today's live atmospheric parameters from Open-Meteo API.
    Cached for 10 minutes to maintain snappy dashboard responsiveness.
    Falls back gracefully to 26-year district climatological normals if offline.
    """
    try:
        data = fetch_live_weather(district)
        return {
            "max_temp_c": float(data["max_temp_c"]),
            "min_temp_c": float(data["min_temp_c"]),
            "humidity_pct": float(data["humidity_pct"]),
            "wind_speed_ms": float(data["wind_speed_ms"]),
            "precipitation_mm": float(data["precipitation_mm"]),
            "date_str": str(data.get("date_str", datetime.now().strftime("%Y-%m-%d"))),
            "is_live": True,
            "source": "Open-Meteo Live API",
        }
    except Exception as e:
        month_str = str(datetime.now().month)
        clim = DISTRICT_CLIMATOLOGY.get(district, {}).get(month_str, {})
        default_max = float(clim.get("Clim_Max_Mean", 43.0))
        default_min = float(clim.get("Clim_Min_Mean", 28.0))
        return {
            "max_temp_c": round(default_max, 1),
            "min_temp_c": round(default_min, 1),
            "humidity_pct": 35.0,
            "wind_speed_ms": 3.5,
            "precipitation_mm": 0.0,
            "date_str": datetime.now().strftime("%Y-%m-%d"),
            "is_live": False,
            "source": "Climatological Normals (Offline Fallback)",
        }


@st.cache_data(ttl=600, show_spinner=False)
def fetch_terai_regional_summary():
    """
    Computes 24-hour tactical predictions across all 21 Terai districts concurrently.
    Cached for 10 minutes to maintain snappy, instant dashboard responsiveness.
    """
    def _predict_single(d: str):
        try:
            return predict_heatwave(d, use_live_api=True)
        except Exception:
            try:
                return predict_heatwave(d, use_live_api=False)
            except Exception:
                return None

    with ThreadPoolExecutor(max_workers=7) as executor:
        results = [r for r in executor.map(_predict_single, VALID_DISTRICTS) if r is not None]
    return results


# Current time in Nepal Time (NPT: UTC+5:45)
npt_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=45)
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
# 4. Prediction Dashboard Renderer (Scoped per Mode)
# ------------------------------------------------------------------------------
def render_prediction_dashboard(result: dict, is_custom_mode: bool = False):
    """Renders forecast results, today's weather inputs, and comparison cards for a mode."""
    w = result["input_weather"]
    district_name = result["district"]
    d_info = DISTRICT_METADATA[district_name]

    # Dates
    obs_date = datetime.strptime(result["forecast_date"], "%Y-%m-%d").date()
    tomorrow_date = obs_date + timedelta(days=1)
    today_str = obs_date.strftime("%d %b %Y")
    tomorrow_str = tomorrow_date.strftime("%d %b %Y")

    is_real_tomorrow = (tomorrow_date == datetime.now().date() + timedelta(days=1))
    forecast_section_title = (
        "🎯 Tomorrow's Heatwave Forecast (24-Hour AI Prediction)"
        if is_real_tomorrow
        else f"🎯 Heatwave Forecast for {tomorrow_str} (24-Hour AI Prediction)"
    )
    pred_kicker = f"Tomorrow ({tomorrow_str})" if is_real_tomorrow else f"Target Forecast ({tomorrow_str})"
    obs_kicker = f"Today ({today_str})" if (obs_date == datetime.now().date()) else f"Observed ({today_str})"

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

    # Tomorrow's Heatwave Forecast (24-Hour AI Prediction)
    st.markdown("---")
    st.html(f"<div class='section-title-wrap' style='margin-top: 10px;'>{forecast_section_title}</div>")

    col_r1, col_r2, col_r3 = st.columns(3)

    # Heatwave Status
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
                <div class="result-card-kicker">Heatwave Status • {pred_kicker}</div>
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

    # Heatwave Threat Level
    with col_r2:
        st.html(
            f"""
            <div class="result-card" style="border-top: 5px solid {tomorrow_threat['color']};">
                <div class="result-card-kicker">Heatwave Threat Level • {pred_kicker}</div>
                <div style="margin-top: 14px;">
                    <div class="threat-badge-pill" style="background:{tomorrow_threat['bg']}; color:{tomorrow_threat['text']}; border:1.5px solid {tomorrow_threat['border']}; margin-bottom:0;">
                        <span>●</span> <span>{tomorrow_threat['label']}</span>
                    </div>
                </div>
            </div>
            """
        )

    # Predicted Temperature
    with col_r3:
        temp_color = tomorrow_threat['color']
        trend_bg = "#fee2e2" if delta_t > 0.5 else ("#ecfdf5" if delta_t < -0.5 else "#f1f5f9")
        trend_color = "#991b1b" if delta_t > 0.5 else ("#065f46" if delta_t < -0.5 else "#475569")

        st.html(
            f"""
            <div class="result-card" style="border-top: 5px solid {temp_color};">
                <div class="result-card-kicker">Predicted High Temperature • {pred_kicker}</div>
                <div class="temp-hero-display">
                    <span style="font-size:32px; margin-right:4px;">🌡️</span>
                    <span class="temp-val-hero mono" style="color: {temp_color};">{pred_temp_tomorrow:.2f}</span>
                    <span class="temp-unit-hero">°C</span>
                </div>
                <div class="trend-indicator-pill" style="background:{trend_bg}; color:{trend_color};">
                    <span>24h Trend: <b>{delta_str}</b> vs baseline ({today_max:.1f}°C)</span>
                </div>
            </div>
            """
        )

    # Today's Weather Conditions
    feels_like_max = compute_feels_like(w["max_temp_c"], w["humidity_pct"], w["wind_speed_ms"])
    diurnal_diff = round(w["max_temp_c"] - w["min_temp_c"], 1)

    mode_tag = (
        '<span style="background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; font-size:11.5px; font-weight:700; padding:2px 8px; border-radius:999px; margin-left:8px;">🧪 Custom Scenario</span>'
        if is_custom_mode
        else '<span style="background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0; font-size:11.5px; font-weight:700; padding:2px 8px; border-radius:999px; margin-left:8px;">📡 Live Telemetry</span>'
    )

    st.html(
        f"""
        <div class="conditions-container" style="margin-top: 26px;">
            <div class="section-title-wrap">
                <span>📊 Meteorological Observations • {district_name} ({obs_kicker})</span>
                {mode_tag}
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

    # Today vs Tomorrow Comparison Section
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
        <div class="comparison-container" style="margin-top: 26px;">
            <div class="comparison-header">
                <div>⚖️ Meteorological Comparison: Observed Baseline vs 24h Forecast • {district_name}</div>
            </div>
            <div class="comparison-grid">
                <div class="comp-pane" style="border-left: 4px solid {today_eval['color']};">
                    <div class="comp-pane-title">
                        <span>📅 OBSERVED BASELINE ({today_str})</span>
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
                        <span>🔮 24-HOUR FORECAST ({tomorrow_str})</span>
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
# 5. 21-District Terai Regional Macro Situational Bar & Synoptic Overview
# ------------------------------------------------------------------------------
def render_terai_regional_overview():
    """Renders the 21-District Regional Situational Bar & Macro Early Warning Matrix."""
    summary_data = fetch_terai_regional_summary()
    if not summary_data:
        return

    REGIONS = [
        {"name": "Western Terai", "desc": "Lumbini & Sudurpashchim (8 Districts)"},
        {"name": "Central Terai", "desc": "Bagmati & Gandaki (2 Districts)"},
        {"name": "Eastern Terai", "desc": "Koshi & Madhesh (11 Districts)"},
    ]

    total_alerts = sum(1 for r in summary_data if r["heatwave_alert"])
    overall_max_t = max(r["predicted_max_temp_c"] for r in summary_data)
    hottest_dist = [r["district"] for r in summary_data if r["predicted_max_temp_c"] == overall_max_t][0]

    macro_status_html = (
        f"<span class='mono' style='font-weight:700; font-size:12px; color:#dc2626;'>⚠️ {total_alerts} Districts in Active Heatwave • National Peak: {overall_max_t:.1f}°C ({hottest_dist})</span>"
        if total_alerts > 0
        else f"<span class='mono' style='font-weight:700; font-size:12px; color:#059669;'>🟢 All 21 Districts Safe (&lt; 40.0°C) • National Peak: {overall_max_t:.1f}°C ({hottest_dist})</span>"
    )

    st.html(
        f"""
        <div class="regional-deck">
            <div class="regional-deck-title">
                <span>🌐 21-District Terai Regional Synoptic Overview • 24-Hour Early Warning</span>
                {macro_status_html}
            </div>
        </div>
        """
    )

    reg_cols = st.columns(3)
    for idx, reg_info in enumerate(REGIONS):
        reg_name = reg_info["name"]
        items = [r for r in summary_data if DISTRICT_METADATA[r["district"]]["Terai_Region"] == reg_name]
        if not items:
            continue

        max_t = max(r["predicted_max_temp_c"] for r in items)
        peak_d = [r["district"] for r in items if r["predicted_max_temp_c"] == max_t][0]
        max_tier = max(r["severity_tier"] for r in items)
        alert_ds = [r["district"] for r in items if r["heatwave_alert"]]
        threat = get_threat_profile(max_tier)

        tier_icon = "🟢" if max_tier == 0 else ("🟡" if max_tier == 1 else ("🟠" if max_tier == 2 else "🔴"))
        tier_sub = "Normal / Safe" if max_tier == 0 else f"Active Alert • Tier {max_tier}"

        with reg_cols[idx]:
            st.html(
                f"""
                <div class="regional-card">
                    <div class="regional-card-header">
                        <span class="regional-card-name">{reg_name}</span>
                        <span class="regional-card-count">{len(items)} Districts</span>
                    </div>
                    <div style="font-size:11.5px; color:#64748b; margin-bottom:4px;">{reg_info['desc']}</div>
                    <div class="regional-status-badge" style="background:{threat['bg']}; color:{threat['text']}; border:1px solid {threat['border']};">
                        <span>{tier_icon}</span>
                        <span>{threat['name']} ({tier_sub})</span>
                    </div>
                    <div class="regional-stat-row">
                        <span>Regional Peak:</span>
                        <span class="regional-stat-val mono" style="color:{threat['color']};">{max_t:.1f}°C <span style="font-size:11px; font-weight:normal; color:#64748b;">({peak_d})</span></span>
                    </div>
                    <div class="regional-stat-row">
                        <span>Active Alerts:</span>
                        <span class="regional-stat-val" style="color:{'#dc2626' if alert_ds else '#059669'};">
                            {len(alert_ds)} of {len(items)} districts
                        </span>
                    </div>
                    <div style="font-size:11px; color:{'#b91c1c' if alert_ds else '#475569'}; margin-top:6px; line-height:1.35; padding-top:4px; border-top:1px dashed #e2e8f0;">
                        {'⚠️ <b>Under Alert:</b> ' + ', '.join(alert_ds) if alert_ds else '✓ All districts within safe thermal limits'}
                    </div>
                </div>
                """
            )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    with st.expander("📋 View Complete 21-District Macro Early Warning Matrix", expanded=False):
        df_rows = []
        for r in sorted(summary_data, key=lambda x: (x["severity_tier"], x["predicted_max_temp_c"]), reverse=True):
            d_name = r["district"]
            meta = DISTRICT_METADATA[d_name]
            tier = r["severity_tier"]
            tp = get_threat_profile(tier)
            df_rows.append({
                "District": d_name,
                "Terai Sub-Region": meta["Terai_Region"],
                "Province": meta["Province"],
                "Regional Hub": meta["Regional_Hub"],
                "Predicted Max Temp": f"{r['predicted_max_temp_c']:.1f}°C",
                "Threat Classification": f"{tp['name']} (Tier {tier})",
                "Alert Status": "⚠️ ACTIVE" if r["heatwave_alert"] else "✓ Safe",
                "Alert Probability": f"{r['alert_probability'] * 100:.1f}%",
            })
        df_macro = pd.DataFrame(df_rows)
        st.dataframe(df_macro, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)


render_terai_regional_overview()


# ------------------------------------------------------------------------------
# 6. Two-Option Segmented Control & 3D Physical Card-Flip Interface
# ------------------------------------------------------------------------------
district_labels = {
    d: f"{d} ({DISTRICT_METADATA[d]['Regional_Hub']}, {DISTRICT_METADATA[d]['Province']})"
    for d in VALID_DISTRICTS
}

MODE_1_LABEL = "Mode 1: District-Based Prediction"
MODE_2_LABEL = "Mode 2: Custom Input Prediction"
MODE_OPTIONS = [MODE_1_LABEL, MODE_2_LABEL]

# Independent state initialization for both modes
if "dist_mode1" not in st.session_state:
    st.session_state["dist_mode1"] = "Banke"
if "dist_mode2" not in st.session_state:
    st.session_state["dist_mode2"] = "Banke"
if "active_prediction_mode" not in st.session_state:
    st.session_state["active_prediction_mode"] = MODE_1_LABEL

# Two-option segmented control
col_s1, col_s2, col_s3 = st.columns([1, 2.8, 1])
with col_s2:
    selected_mode = st.segmented_control(
        "Prediction Mode",
        options=MODE_OPTIONS,
        default=st.session_state["active_prediction_mode"],
        format_func=lambda x: ("📍 " if "Mode 1" in x else "🧪 ") + x,
        key="mode_segmented_control",
        label_visibility="collapsed",
    )
    if not selected_mode:
        selected_mode = st.session_state["active_prediction_mode"]
    st.session_state["active_prediction_mode"] = selected_mode

is_mode1 = (selected_mode == MODE_1_LABEL)
active_class = "flip-mode-1" if is_mode1 else "flip-mode-2"

# Dynamic face positioning and smooth 3D rotation styling
flip_css = f"""
<style>
.st-key-mode1_face {{
    transform: rotateY({'0deg' if is_mode1 else '-180deg'}) !important;
    opacity: {'1' if is_mode1 else '0'} !important;
    visibility: {'visible' if is_mode1 else 'hidden'} !important;
    pointer-events: {'auto' if is_mode1 else 'none'} !important;
    position: {'relative' if is_mode1 else 'absolute'} !important;
    z-index: {'2' if is_mode1 else '1'} !important;
    top: 0 !important;
    left: 0 !important;
}}
.st-key-mode2_face {{
    transform: rotateY({'180deg' if is_mode1 else '0deg'}) !important;
    opacity: {'0' if is_mode1 else '1'} !important;
    visibility: {'hidden' if is_mode1 else 'visible'} !important;
    pointer-events: {'none' if is_mode1 else 'auto'} !important;
    position: {'absolute' if is_mode1 else 'relative'} !important;
    z-index: {'1' if is_mode1 else '2'} !important;
    top: 0 !important;
    left: 0 !important;
}}
</style>
"""
st.html(flip_css)

# Synchronize 3D flip card state and immediate click listener for instant feedback
st.html(
    f"""
    <div id="mode-flip-detector" data-mode="{active_class}" style="display:none;"></div>
    <script>
    (function() {{
        function applyFlip() {{
            var stage = document.querySelector('.st-key-prediction_flip_stage');
            var det = document.getElementById('mode-flip-detector');
            if (stage && det) {{
                var targetMode = det.getAttribute('data-mode');
                if (targetMode === 'flip-mode-1') {{
                    stage.classList.remove('flip-mode-2');
                    stage.classList.add('flip-mode-1');
                }} else {{
                    stage.classList.remove('flip-mode-1');
                    stage.classList.add('flip-mode-2');
                }}
            }}
        }}
        applyFlip();
        setTimeout(applyFlip, 40);
        setTimeout(applyFlip, 120);

        if (!window._flipListenerAttached) {{
            window._flipListenerAttached = true;
            document.addEventListener('click', function(e) {{
                var btn = e.target.closest('[data-testid="stSegmentedControl"] button, [data-testid="stSegmentedControl"] [role="radio"]');
                if (!btn) return;
                var text = (btn.textContent || '').trim();
                var stage = document.querySelector('.st-key-prediction_flip_stage');
                if (!stage) return;
                if (text.indexOf('Mode 2') !== -1 || text.indexOf('Custom') !== -1) {{
                    stage.classList.remove('flip-mode-1');
                    stage.classList.add('flip-mode-2');
                }} else if (text.indexOf('Mode 1') !== -1 || text.indexOf('District') !== -1) {{
                    stage.classList.remove('flip-mode-2');
                    stage.classList.add('flip-mode-1');
                }}
            }}, true);
        }}
    }})();
    </script>
    """,
    unsafe_allow_javascript=True,
)

# 3D Flip Card Perspective Stage
flip_stage = st.container(key="prediction_flip_stage")
with flip_stage:
    # --- CARD FACE 1: Mode 1 (District-Based Prediction) ---
    with st.container(key="mode1_face"):
        col_d1, col_d2, col_d3 = st.columns([2.5, 1.5, 1.2])
        
        with col_d1:
            sel_dist_mode1 = st.selectbox(
                "Select District",
                options=VALID_DISTRICTS,
                format_func=lambda d: district_labels[d],
                index=VALID_DISTRICTS.index(st.session_state["dist_mode1"]) if st.session_state["dist_mode1"] in VALID_DISTRICTS else 0,
                key="dist_mode1",
                help="Pulls automated weather observations and 14-day history directly from satellite & station feeds.",
            )
        
        with col_d2:
            d_meta1 = DISTRICT_METADATA[sel_dist_mode1]
            st.html(
                f"""
                <div style="padding-top:28px; font-size:12.5px; color:#475569; line-height:1.4;">
                    <b>Sub-Region:</b> {d_meta1['Terai_Region']}<br>
                    <b>GPS:</b> <span class="mono">{d_meta1['Latitude']}°N, {d_meta1['Longitude']}°E</span>
                </div>
                """
            )
            
        with col_d3:
            st.html("<div style='height: 28px;'></div>")
            btn_mode1 = st.button("🚀 Run Forecast", type="primary", use_container_width=True, key="btn_mode1")

        # Mode 1 execution: automatically runs when district changes, button is pressed, or initial load
        dist_changed_m1 = (st.session_state.get("mode1_predicted_district") != sel_dist_mode1)
        if btn_mode1 or ("mode1_result" not in st.session_state) or dist_changed_m1:
            st.session_state["mode1_predicted_district"] = sel_dist_mode1
            with st.spinner(f"Ingesting atmospheric parameters and executing multi-champion models for {sel_dist_mode1}..."):
                try:
                    res1 = predict_heatwave(
                        district=sel_dist_mode1,
                        date_str=datetime.now().strftime("%Y-%m-%d"),
                        use_live_api=True,
                    )
                    st.session_state["mode1_result"] = res1
                    st.session_state["mode1_error"] = None
                except Exception as e:
                    st.session_state["mode1_error"] = f"Forecast Generation Failed: {e}"

        # Render Mode 1's isolated results and inputs
        if st.session_state.get("mode1_error"):
            st.error(st.session_state["mode1_error"])
        elif st.session_state.get("mode1_result"):
            render_prediction_dashboard(st.session_state["mode1_result"], is_custom_mode=False)

    # --- CARD FACE 2: Mode 2 (Custom Weather Input) ---
    with st.container(key="mode2_face"):
        col_c1, col_c2 = st.columns([2, 1.2])
        with col_c1:
            sel_dist_mode2 = st.selectbox(
                "Target District",
                options=VALID_DISTRICTS,
                format_func=lambda d: district_labels[d],
                index=VALID_DISTRICTS.index(st.session_state["dist_mode2"]) if st.session_state["dist_mode2"] in VALID_DISTRICTS else 0,
                key="dist_mode2",
            )

        today_date = datetime.now().date()
        tomorrow_date = today_date + timedelta(days=1)
        today_str = today_date.strftime("%d %b %Y")
        tomorrow_str = tomorrow_date.strftime("%d %b %Y")

        with col_c2:
            st.html(
                f"""
                <div style="padding-top:28px;">
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:6px 12px; font-size:12.5px; color:#334155; line-height:1.4;">
                        <span style="font-weight:700; color:#1e293b;">🎯 Forecast Target:</span> Tomorrow (<span class="mono" style="font-weight:700; color:#2563eb;">{tomorrow_str}</span>)<br>
                        <span style="color:#64748b; font-size:11.5px;">24-Hour Tactical Early Warning (Day <i>t+1</i>)</span>
                    </div>
                </div>
                """
            )

        # Automatically fetch initial inputs from the API on initial load, refresh, or district switch
        if st.session_state.get("custom_inputs_loaded_for") != sel_dist_mode2:
            live_today = fetch_today_api_weather(sel_dist_mode2)
            st.session_state["cust_max_t"] = float(np.clip(round(live_today["max_temp_c"], 1), 15.0, 55.0))
            # Ensure min < max
            valid_min = min(round(live_today["min_temp_c"], 1), st.session_state["cust_max_t"] - 1.0)
            st.session_state["cust_min_t"] = float(np.clip(valid_min, 5.0, 38.0))
            st.session_state["cust_rh"] = float(np.clip(round(live_today["humidity_pct"], 1), 5.0, 100.0))
            st.session_state["cust_wind"] = float(np.clip(round(live_today["wind_speed_ms"], 1), 0.0, 25.0))
            st.session_state["cust_precip"] = float(np.clip(round(round(live_today["precipitation_mm"] * 2) / 2, 1), 0.0, 150.0))
            st.session_state["custom_inputs_loaded_for"] = sel_dist_mode2

        st.caption(
            f"Set custom atmospheric parameters for **Today ({today_str}, Day *t*)** to simulate early-warning heatwave conditions for **Tomorrow ({tomorrow_str}, Day *t+1*)**:"
        )

        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            custom_max_t = st.slider(
                "Maximum Temp (°C)",
                min_value=15.0,
                max_value=55.0,
                step=0.1,
                key="cust_max_t",
            )
        with col_w2:
            custom_min_t = st.slider(
                "Minimum Temp (°C)",
                min_value=5.0,
                max_value=38.0,
                step=0.1,
                key="cust_min_t",
            )
        with col_w3:
            custom_rh = st.slider(
                "Humidity (%)",
                min_value=5.0,
                max_value=100.0,
                step=1.0,
                key="cust_rh",
            )

        col_w4, col_w5 = st.columns(2)
        with col_w4:
            custom_wind = st.slider(
                "Wind Speed (m/s)",
                min_value=0.0,
                max_value=25.0,
                step=0.1,
                key="cust_wind",
            )
        with col_w5:
            custom_precip = st.slider(
                "Precipitation (mm)",
                min_value=0.0,
                max_value=150.0,
                step=0.5,
                key="cust_precip",
            )

        st.html("<div style='height: 12px;'></div>")
        col_b1, col_b2 = st.columns([1, 2.5])
        with col_b1:
            if st.button("🔄 Sync Live API", use_container_width=True, key="btn_resync_api", help="Re-sync inputs with today's live API telemetry"):
                st.session_state["custom_inputs_loaded_for"] = None
                st.rerun()
        with col_b2:
            btn_mode2 = st.button("🧪 Predict Custom Scenario", type="primary", use_container_width=True, key="btn_mode2")

        # Mode 2 execution: automatically runs when district changes, button is pressed, or initial load
        dist_changed_m2 = (st.session_state.get("mode2_predicted_district") != sel_dist_mode2)
        if btn_mode2 or ("mode2_result" not in st.session_state) or dist_changed_m2:
            st.session_state["mode2_predicted_district"] = sel_dist_mode2
            if custom_min_t >= custom_max_t:
                st.session_state["mode2_error"] = f"⚠️ Validation Error: Minimum temperature ({custom_min_t}°C) cannot be greater than or equal to Maximum temperature ({custom_max_t}°C)."
                st.session_state["mode2_result"] = None
            else:
                with st.spinner(f"Ingesting custom atmospheric parameters and executing multi-champion models for {sel_dist_mode2}..."):
                    try:
                        res2 = predict_heatwave(
                            district=sel_dist_mode2,
                            date_str=today_date.strftime("%Y-%m-%d"),
                            max_temp_c=custom_max_t,
                            min_temp_c=custom_min_t,
                            humidity_pct=custom_rh,
                            wind_speed_ms=custom_wind,
                            precipitation_mm=custom_precip,
                            use_live_api=True,
                        )
                        st.session_state["mode2_result"] = res2
                        st.session_state["mode2_error"] = None
                    except Exception as e:
                        st.session_state["mode2_error"] = f"Forecast Generation Failed: {e}"

        # Render Mode 2's isolated results and inputs
        if st.session_state.get("mode2_error"):
            st.error(st.session_state["mode2_error"])
        elif st.session_state.get("mode2_result"):
            render_prediction_dashboard(st.session_state["mode2_result"], is_custom_mode=True)

# ------------------------------------------------------------------------------
# 6. Operational Footer
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
