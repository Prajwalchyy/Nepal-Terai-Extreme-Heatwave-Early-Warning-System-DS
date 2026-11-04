# Nepal Terai Extreme Heatwave Early-Warning System (2000–2026)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![ML-Scikit--Learn-LightGBM](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20LightGBM-orange.svg)](https://scikit-learn.org/)
[![Status-Operational](https://img.shields.io/badge/Status-Operational%20%26%20Audited-brightgreen.svg)]()
[![Target-21%20Terai%20Districts](https://img.shields.io/badge/Coverage-21%20Terai%20Districts-success.svg)]()

An operational, machine learning Disaster Risk Reduction (DRR) forecasting system predicting extreme thermal events, hazard severity tiers, and peak daytime temperatures 24 hours in advance across all 21 lowland Terai districts of Nepal.

The system combines 26+ years of daily meteorological reanalysis data (NASA POWER), live satellite and station telemetry (Open-Meteo API), and a multi-champion machine learning cascade enforced by physical consistency logic.

---

## 1. Operational Overview & Problem Context

During pre-monsoon and summer periods (April to June), Nepal's southern lowland (the Terai belt) experiences lethal heatwaves where afternoon temperatures consistently surpass $40.0^\circ\text{C}$ to $45.0^\circ\text{C}$. Advective hot and dry westerly winds (*Loo*) combine with high humidity along the Indo-Gangetic Plain to trigger severe heat distress, hospital surges, and crop desiccation.

Traditional meteorological bulletins in the region have historically relied on static temperature thresholds ($40.0^\circ\text{C}$) that neglect microclimates, humidity, wind desiccation, and soil moisture memory. Furthermore, historical heatwave days account for only **6.45% of days** (a 14.5:1 class imbalance), while extreme emergency days ($\ge 45.0^\circ\text{C}$) comprise just **0.20% of records**. Standard classifiers optimize for overall accuracy and miss dangerous anomalies.

This platform resolves these challenges through:
- **Disaster-calibrated modeling**: Priority on life-safety recall rather than deceptive overall accuracy.
- **Physical consistency enforcement**: A post-inference cascade that guarantees mathematical alignment between binary alerts, multi-tier hazard levels, and continuous temperature predictions.
- **24-Hour Tactical Horizon ($t+1$)**: Real-time integration of live satellite feeds with 14-day antecedent physical memory.

---

## 2. System Architecture & Multi-Champion Cascade

The production inference pipeline executes three specialized champion models followed by an enforcer module to prevent contradictory outputs:

```
                          Incoming 24-Hour Weather Telemetry
                      (Live Open-Meteo API or Mode 2 Simulation)
                                         │
                                         ▼
                 ┌─────────────────────────────────────────────────┐
                 │    Feature Engineering & Antecedent History     │
                 │      (63 Physical, Lag & Climatology Vars)      │
                 └───────────────────────┬─────────────────────────┘
                                         │
                 ┌───────────────────────┼─────────────────────────┐
                 │                       │                         │
                 ▼                       ▼                         ▼
      ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
      │  Target 1: Alert    │ │ Target 2: Severity  │ │ Target 3: Max Temp  │
      │    Random Forest    │ │      LightGBM       │ │  Ridge Regression   │
      │ (Threshold = 0.30)  │ │ (num_leaves = 63)   │ │    (alpha = 1.0)    │
      └──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
                 │                       │                         │
                 │   Binary Alert        │  Severity Tier          │  Predicted Max
                 │  (True / False)       │    (Tier 0–3)           │   Temp (°C)
                 │                       │                         │
                 └───────────────────────┼─────────────────────────┘
                                         │
                                         ▼
                 ┌─────────────────────────────────────────────────┐
                 │   Hybrid Multi-Target Consistency Enforcer      │
                 │   - Validates threshold invariants              │
                 │   - Conditioned Specialist Model Re-prediction  │
                 │   - Resolves 100% of physical discrepancies     │
                 └───────────────────────┬─────────────────────────┘
                                         │
                                         ▼
                 ┌─────────────────────────────────────────────────┐
                 │     Unified Disaster Risk Reduction Output      │
                 │  - Validated Alert & DHM Threat Badge           │
                 │  - Calibrated Next-Day High Temperature         │
                 │  - Multi-Sector Institutional DRR Advisories    │
                 └─────────────────────────────────────────────────┘
```

### The Champion Models

1. **Target 1: Binary Heatwave Alert ($T_{\max, t+1} \ge 40.0^\circ\text{C}$)**
   - **Champion**: Tuned `RandomForestClassifier` (`n_estimators=100`, `class_weight='balanced'`).
   - **Decision Threshold**: $\tau = 0.30$ (optimized via precision-recall sweep).
   - **Performance**: **85.26% disaster recall**, **73.60% PR-AUC**, Brier score 0.0151.

2. **Target 2: Hazard Severity Tier (4-Tier Department of Hydrology and Meteorology / DHM Schema)**
   - **Champion**: Tuned `LGBMClassifier` (`num_leaves=63`, `n_estimators=200`, `learning_rate=0.10`, `class_weight='balanced'`).
   - **Tiers**:
     - *Tier 0*: Normal / Low Stress ($< 40.0^\circ\text{C}$)
     - *Tier 1*: Moderate Warning ($40.0^\circ\text{C} - 41.9^\circ\text{C}$)
     - *Tier 2*: Severe Danger ($42.0^\circ\text{C} - 44.9^\circ\text{C}$)
     - *Tier 3*: Extreme Emergency ($\ge 45.0^\circ\text{C}$)
   - **Performance**: **66.50% Macro $F_1$-score**, 97.21% overall accuracy. Captured **50.0% of rare Tier 3 emergencies** where unweighted algorithms collapsed to 0.0% recall.

3. **Target 3: Continuous Next-Day Maximum Temperature ($T_{\max, t+1}$)**
   - **Champion**: Regularized `Ridge` Regression ($\alpha = 1.0$).
   - **Performance**: **MAE of 0.9482°C**, **RMSE of 1.2978°C**, **$R^2 = 94.87\%$**.
   - **Advantage**: Retains open linear extrapolation into record-breaking heat anomalies without the step-function clipping inherent to tree models. Trains in 0.13 seconds.

4. **Consistency Enforcer: Conditioned Specialist Model**
   - **Architecture**: Hierarchical logical enforcer paired with a secondary `Ridge` model trained on combined weather and validated disaster indicators.
   - **Result**: Eliminates 1,240 out of 1,240 test contradictions (**100% resolution rate**), improving overall MAE by 0.02°C on conflicted records.

---

## 3. Performance Benchmark Summary

Evaluated on strict chronological out-of-time test partitions (preserving the physical arrow of time; zero data leakage):

| Forecasting Task | Model / Method | Accuracy / $R^2$ | Precision / MAE | Recall / RMSE | $F_1$ / Macro $F_1$ | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Binary Alert** ($t+1$) | Logistic Regression | 90.07% | 37.48% | 97.22% | 54.10% | 61.12% |
| **Binary Alert** ($t+1$) | **Random Forest ($\tau = 0.30$)** | **94.86%** | **50.96%** | **85.26%** | **63.82%** | **73.60%** |
| **Severity Tiers** (0–3) | XGBoost (Default) | 97.53% | 76.54% | 59.87% | 65.73% | — |
| **Severity Tiers** (0–3) | **LightGBM (Balanced)** | **97.21%** | **70.21%** | **64.70%** | **66.50%** | — |
| **Max Temperature** ($t+1$) | Random Forest Regressor | 94.87% ($R^2$) | 0.9479°C (MAE) | 1.2995°C (RMSE) | — | — |
| **Max Temperature** ($t+1$) | **Ridge Regression ($\alpha=1.0$)** | **94.87% ($R^2$)** | **0.9482°C (MAE)** | **1.2978°C (RMSE)**| — | — |

---

## 4. Geographical Scope: 21 Terai Districts

The system covers all 21 administrative districts across 6 provinces along Nepal's southern border:

| Region | Provinces | Districts Covered | Climate Characteristics |
| :--- | :--- | :--- | :--- |
| **Western Terai** | Lumbini, Sudurpashchim | Banke, Bardiya, Kailali, Kanchanpur, Dang, Kapilvastu, Rupandehi, Parasi | Highest thermal intensity, direct exposure to advective *Loo* winds, dry spells |
| **Central Terai** | Bagmati, Gandaki, Madhesh | Chitwan, Nawalpur, Parsa, Bara, Rautahat, Sarlahi | High humidity river basin, intense heat dome accumulation, early pre-monsoon shifts |
| **Eastern Terai** | Koshi, Madhesh | Dhanusha, Mahottari, Siraha, Saptari, Sunsari, Morang, Jhapa | Higher atmospheric moisture, early monsoon onset, elevated Steadman heat index |

---

## 5. Engineered Features (63 Dimensions)

Ablation testing demonstrated that omitting thermodynamic and autoregressive features causes continuous temperature $R^2$ to drop from 94.87% to 88.14% and reduces binary alert PR-AUC by 4.2%.

The 63 features are grouped into five physical domains:

1. **Thermodynamic Indicators**:
   - Diurnal Temperature Range ($T_{\max} - T_{\min}$)
   - NOAA Steadman Heat Index ("Feels-Like" Temperature)
   - Stull Wet-Bulb Temperature ($T_w$) and Wet-Bulb Depression
   - Advective *Loo* Desiccation Index
   - Tropical Night Indicator ($T_{\min} \ge 25.0^\circ\text{C}$)
2. **Autoregressive Lags**:
   - 1-Day and 2-Day backward lags for temperature, precipitation, and relative humidity
3. **Momentum & Dynamic Deltas**:
   - 1-Day and 2-Day temperature velocity ($\Delta T_{\max}, \Delta T_{\min}$)
4. **Antecedent Rolling Windows & Extremes**:
   - 3-day and 7-day rolling means and extrema
   - 7-day and 14-day cumulative precipitation deficits
   - Consecutive dry spell counts and heatwave streaks
5. **Climatology & Geographic Embeddings**:
   - 26-year day-of-year district climatological normals ($\mu, \sigma$) and Z-scores
   - Cyclical solar calendar transforms ($\sin, \cos$ of Day-of-Year and Month)
   - Spatial coordinates (Latitude, Longitude) and regional zone indicators

---

## 6. Interactive Web Application (`app.py`)

The production interface is built in Streamlit with custom styling (`style.css`), a real-time Nepal Standard Time (NPT) clock, and responsive 3D card-flip mode transitions:

- **Mode 1: Automated Live Telemetry**
  - Select any of the 21 districts to instantly query the Open-Meteo API.
  - Automatically retrieves today's observed weather and 14 days of antecedent satellite telemetry.
  - Generates 24-hour tactical early warnings ($t+1$) with sub-second latency.
- **Mode 2: What-If Custom Simulation**
  - Allows disaster management officials to manipulate Day $t$ atmospheric sliders (Max Temp, Min Temp, Humidity, Wind, Rain).
  - Preserves underlying 14-day antecedent soil and weather memory to model simulated next-day heatwave risk.
  - Includes a one-click **🔄 Sync Live API** button to reset values back to real-time observations.
- **21-District Macro Regional Situational Bar**
  - Multi-threaded concurrent batch inference (`ThreadPoolExecutor`) across all 21 districts.
  - Cached with `@st.cache_data(ttl=600)` for immediate regional risk assessment across Western, Central, and Eastern Terai.
- **Operational DRR Advisories**
  - Dynamic guidance tailored for municipal disaster committees, agriculture, schools, and outdoor laborers based on predicted severity tiers.

---

## 7. Project Structure

```
terai_heatwave_dataset/
├── app.py                             # Streamlit operational web dashboard
├── style.css                          # Custom enterprise styling & animations
├── requirements.txt                   # Production dependency specifications
├── README.md                          # Project documentation
│
├── src/                               # Production Python modules
│   ├── __init__.py
│   ├── data_processing.py             # Data cleaning & validation pipeline
│   ├── feature_engineering.py         # 63-feature transformation engine
│   ├── model_train.py                 # Multi-model training & consistency fitting
│   └── predict.py                     # Production inference engine & live API caller
│
├── models/                            # Serialized production model artifacts
│   ├── heatwave_flag.pkl              # Target 1: Random Forest Classifier
│   ├── heatwave_severity.pkl          # Target 2: LightGBM Multi-Class Classifier
│   ├── heatwave_temp.pkl              # Target 3: Ridge Regressor
│   ├── heatwave_repredict_temp_model.pkl # Target 4: Conditioned Specialist Model
│   ├── preprocessing.pkl              # Scalers, imputers & pipeline transformers
│   ├── pipeline_metadata.json         # Feature list, thresholds & export records
│   ├── district_climatology.json      # 26-year day-of-year normal baselines
│   └── district_metadata.json         # Coordinates, elevations & regional hubs
│
├── data/                              # Meteorological datasets
│   ├── download_terai_data.py         # NASA POWER automated ingestion script
│   ├── raw/
│   │   └── terai_heatwave_raw.csv     # Raw NASA POWER daily dataset (2000–2026)
│   └── processed/
│       ├── clean_terai_heatwave.csv   # Cleaned dataset (204,897 records)
│       ├── terai_features_engineered.csv
│       └── terai_features_engineered.parquet
│
└── notebooks/                         # Development & research notebooks
    ├── 01_data_load_clean.ipynb       # Data cleaning & validation
    ├── 02_eda.ipynb                   # Exploratory data analysis & climate trends
    ├── 03_feature_eng.ipynb           # Feature formulation & physics validation
    └── 04_ml_test_train.ipynb         # Model benchmarking, tuning & consistency audit
```

---

## 8. Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Prajwalchyy/Nepal-Terai-Extreme-Heatwave-Early-Warning-System-DS.git
cd Nepal-Terai-Extreme-Heatwave-Early-Warning-System-DS
```

### 2. Set Up Python Environment
Recommended: Python 3.10, 3.11, or 3.12.
```bash
# Using conda
conda create -n terai_env python=3.11 -y
conda activate terai_env

# Or using venv
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run CLI Inference Engine
You can execute predictions directly from the command line:
```bash
python -m src.predict
```

Or invoke the Python API in your own scripts:
```python
from src.predict import predict_heatwave

# Scenario A: Automated live forecast using real-time satellite feeds
result = predict_heatwave(district="Banke", use_live_api=True)
print(f"Target Date: {result['forecast_date']}")
print(f"Heatwave Alert: {result['heatwave_alert']} (p={result['alert_probability']:.2f})")
print(f"Hazard Severity: {result['severity_label']}")
print(f"Predicted High Temp: {result['predicted_max_temp_c']} °C")

# Scenario B: What-if custom weather input
sim_result = predict_heatwave(
    district="Parsa",
    use_live_api=False,
    max_temp_c=42.0,
    min_temp_c=29.0,
    humidity_pct=55.0,
    wind_speed_ms=4.2,
    precipitation_mm=0.0
)
print(f"Simulated Outcome: {sim_result['severity_label']} ({sim_result['predicted_max_temp_c']} °C)")
```

---

## 9. Physical Invariants & Scientific Integrity

- **Chronological Time Splitting**: Time-series cross-validation and evaluation strictly preserve temporal sequence (Train: 2000–2020; Validation: 2021–2022; Test: 2023–2026). No future observations are leaked into past feature spaces.
- **Zero-Contradiction Guarantee**: An alert cannot fire without a corresponding hazard tier ($\ge 1$), and a Tier 0 classification cannot output a temperature exceeding the danger threshold ($\ge 40.0^\circ\text{C}$).
- **Thermodynamic Grounding**: Feature relationships obey physical boundaries ($T_{\min} \le T_{\max}$, $T_w \le T$, Relative Humidity $\in [0, 100]\%$).

---

## 10. Data Sources & Acknowledgments

- **NASA POWER Project**: Agroclimatology and Surface Meteorology Daily Dataset (2000–2026), NASA Langley Research Center.
- **Open-Meteo API**: High-resolution global numerical weather prediction (ECMWF, GFS) providing real-time telemetry and 14-day antecedent feeds.
- **Department of Hydrology and Meteorology (DHM), Nepal**: Institutional heat hazard warning thresholds and meteorological references.

---

## 11. Authors & Maintainers

- **Prajwal Chaudhary** — Lead Researcher & Machine Learning Developer
- **Repository**: [Nepal-Terai-Extreme-Heatwave-Early-Warning-System-DS](https://github.com/Prajwalchyy/Nepal-Terai-Extreme-Heatwave-Early-Warning-System-DS.git)
