"""
================================================================================
Nepal Terai Extreme Heatwave Prediction System (2000-2026)
Production Model Training & Artifact Serialization Pipeline
================================================================================
Domain: Climate Disaster Risk Reduction (DRR) & Multi-Task Forecasting
Target Region: 21 Lowland Terai Districts, Nepal
Training Scope: Complete Historical Corpus (2000-2026, N=204,582 records)

Champion Architectures & Optimal Hyperparameters:
  - Target 1 (Heatwave Alert): Random Forest Classifier (n_estimators=100, class_weight='balanced')
  - Target 2 (Severity Tier):  LightGBM Classifier (num_leaves=63, lr=0.10, n_estimators=200, class_weight='balanced')
  - Target 3 (Max Temperature): Ridge Regression (alpha=1.0)
Specialist:
  - Conditioned Temperature Model: Ridge Regression (alpha=1.0) on [Weather + Alert + Severity]
================================================================================
"""

from pathlib import Path
import sys
import time
import json
from typing import Tuple, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# ------------------------------------------------------------------------------
# Configuration & Paths
# ------------------------------------------------------------------------------
OPTIMAL_THRESHOLD_FLAG = 0.30
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "terai_features_engineered.parquet"
MODELS_DIR = BASE_DIR / "models"


# ==============================================================================
# 1. Fast Data Ingestion
# ==============================================================================
def load_training_data(data_path: Path = DATA_PATH) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Loads engineered parquet dataset and isolates feature matrix X and targets."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_parquet(data_path)
    drop_cols = [
        "Date", "District", "Province", "Regional_Hub",
        "Next_Day_Max_Temp", "Heatwave_Flag", "Heatwave_Severity"
    ]
    X = pd.get_dummies(df.drop(columns=drop_cols), columns=["Season", "Terai_Region"], dtype="int8")

    y_flag = df["Heatwave_Flag"].values
    y_sev = df["Heatwave_Severity"].values
    y_mtemp = df["Next_Day_Max_Temp"].values

    return X, y_flag, y_sev, y_mtemp


# ==============================================================================
# 2. Champion Model Training
# ==============================================================================
def train_champion_models(
    X_scaled: np.ndarray,
    y_flag: np.ndarray,
    y_sev: np.ndarray,
    y_mtemp: np.ndarray,
) -> Dict[str, Any]:
    """Trains the crowned champions and the secondary specialist on full dataset."""
    t0 = time.time()

    print("1/4 Training Target 1 Champion: Random Forest (n=100, balanced)...")
    champion_flag = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    champion_flag.fit(X_scaled, y_flag)

    print("2/4 Training Target 2 Champion: LightGBM (leaves=63, lr=0.10, balanced)...")
    champion_severity = LGBMClassifier(
        num_leaves=63,
        learning_rate=0.10,
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    champion_severity.fit(X_scaled, y_sev)

    print("3/4 Training Target 3 Champion: Ridge Regression (alpha=1.0)...")
    champion_temp = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    champion_temp.fit(X_scaled, y_mtemp)

    print("4/4 Training Conditioned Specialist: Ridge on [Weather + Alert + Severity]...")
    X_cond = np.column_stack([X_scaled, y_flag, y_sev])
    conditioned_temp_model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    conditioned_temp_model.fit(X_cond, y_mtemp)

    elapsed = time.time() - t0
    print(f"✅ All champion models trained successfully in {elapsed:.2f} seconds.")

    return {
        "champion_flag": champion_flag,
        "champion_severity": champion_severity,
        "champion_temp": champion_temp,
        "conditioned_temp_model": conditioned_temp_model,
        "optimal_threshold_flag": OPTIMAL_THRESHOLD_FLAG,
    }


# ==============================================================================
# 3. Model Artifact Serialization
# ==============================================================================
def save_production_artifacts(
    models_dir: Path,
    scaler: StandardScaler,
    models_bundle: Dict[str, Any],
    feature_names: list,
) -> None:
    """Serializes production models, scaler, and pipeline metadata to models/."""
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, models_dir / "preprocessing.pkl")
    joblib.dump(models_bundle["champion_flag"], models_dir / "heatwave_flag.pkl")
    joblib.dump(models_bundle["champion_severity"], models_dir / "heatwave_severity.pkl")
    joblib.dump(models_bundle["champion_temp"], models_dir / "heatwave_temp.pkl")
    joblib.dump(models_bundle["conditioned_temp_model"], models_dir / "heatwave_repredict_temp_model.pkl")

    metadata = {
        "system_name": "Nepal Terai Extreme Heatwave Early-Warning System",
        "optimal_threshold_flag": models_bundle.get("optimal_threshold_flag", OPTIMAL_THRESHOLD_FLAG),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "training_scope": "Full Dataset Retrained (2000-01-15 to 2026-09-16, N=204,582 daily district records)",
        "champion_models": {
            "target_1_alert": "RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)",
            "target_2_severity": "LGBMClassifier(num_leaves=63, learning_rate=0.10, n_estimators=200, class_weight='balanced', random_state=42)",
            "target_3_temperature": "Ridge(alpha=1.0, random_state=42)",
            "conditioned_specialist": "Ridge(alpha=1.0, random_state=42) on [Weather + Alert + Severity]",
        },
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(models_dir / "pipeline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"📦 Production artifacts successfully saved to: {models_dir.resolve()}")


# ==============================================================================
# 4. Main Training Orchestration
# ==============================================================================
def main() -> None:
    print("=" * 80)
    print(">>> STARTING FULL-HISTORY CHAMPION MODEL TRAINING (2000-2026)")
    print("=" * 80)

    # 1. Ingestion
    print(f"Loading feature matrix from: {DATA_PATH.name}...")
    X, y_flag, y_sev, y_mtemp = load_training_data(DATA_PATH)
    print(f"Loaded {len(X):,} records with {len(X.columns)} features.")

    # 2. Fit Scaler across full dataset
    print("Fitting StandardScaler on full 2000-2026 dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train Champion Models
    models_bundle = train_champion_models(X_scaled, y_flag, y_sev, y_mtemp)

    # 4. Serialize Artifacts
    save_production_artifacts(MODELS_DIR, scaler, models_bundle, X.columns.tolist())

    print("\n" + "=" * 80)
    print(">>> MODEL TRAINING & SERIALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
