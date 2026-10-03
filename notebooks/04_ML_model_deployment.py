from pathlib import Path
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    max_error,
)


# ==============================================================================
# 1. Data Ingestion & Preprocessing
# ==============================================================================
def load_and_preprocess_feature_matrix(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ingests engineered feature matrix, decouples administrative metadata,
    encodes categorical variables, and isolates target variables.
    """
    df = pd.read_parquet(data_path)

    meta_cols = ["Date", "District", "Province", "Regional_Hub"]
    df_meta = df[meta_cols].copy()

    cat_cols = ["Season", "Terai_Region"]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype="int8")

    target_cols = ["Next_Day_Max_Temp", "Heatwave_Flag", "Heatwave_Severity"]
    y_targets = df[target_cols].copy()

    drop_cols = meta_cols + target_cols
    X = df_encoded.drop(columns=drop_cols)

    return df_meta, X, y_targets


# ==============================================================================
# 2. Chronological 3-Way Partitioning & Scaling
# ==============================================================================
def split_and_scale_dataset(
    df_meta: pd.DataFrame,
    X: pd.DataFrame,
    y_targets: pd.DataFrame,
    models_dir: Path,
) -> dict:
    """
    Applies strict chronological partition masks (Train <= 2020, Val 2021-2022, Test >= 2023),
    fits StandardScaler strictly on training observations, and serializes the preprocessing pipeline.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    train_mask = df_meta["Date"] <= "2020-12-31"
    val_mask = df_meta["Date"].between("2021-01-01", "2022-12-31")
    test_mask = df_meta["Date"] >= "2023-01-01"

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]

    y_train_flag = y_targets.loc[train_mask, "Heatwave_Flag"]
    y_val_flag = y_targets.loc[val_mask, "Heatwave_Flag"]
    y_test_flag = y_targets.loc[test_mask, "Heatwave_Flag"]

    y_train_sev = y_targets.loc[train_mask, "Heatwave_Severity"]
    y_val_sev = y_targets.loc[val_mask, "Heatwave_Severity"]
    y_test_sev = y_targets.loc[test_mask, "Heatwave_Severity"]

    y_train_mtemp = y_targets.loc[train_mask, "Next_Day_Max_Temp"]
    y_val_mtemp = y_targets.loc[val_mask, "Next_Day_Max_Temp"]
    y_test_mtemp = y_targets.loc[test_mask, "Next_Day_Max_Temp"]

    # Calculate class imbalance ratio (~14.5:1)
    pos_weight = float((len(y_train_flag) - y_train_flag.sum()) / y_train_flag.sum())

    # Fit and serialize StandardScaler strictly on X_train
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, models_dir / "preprocessing.pkl")

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Chronologically sort training observations for walk-forward CV
    sort_idx = df_meta[train_mask].sort_values(by=["Date", "District"]).index
    X_train_chrono = X_train.loc[sort_idx].reset_index(drop=True)
    y_train_flag_chrono = y_train_flag.loc[sort_idx].reset_index(drop=True)
    X_train_scaled_chrono = scaler.transform(X_train_chrono)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_chrono": X_train_chrono,
        "X_train_scaled_chrono": X_train_scaled_chrono,
        "y_train_flag": y_train_flag,
        "y_val_flag": y_val_flag,
        "y_test_flag": y_test_flag,
        "y_train_flag_chrono": y_train_flag_chrono,
        "y_train_sev": y_train_sev,
        "y_val_sev": y_val_sev,
        "y_test_sev": y_test_sev,
        "y_train_mtemp": y_train_mtemp,
        "y_val_mtemp": y_val_mtemp,
        "y_test_mtemp": y_test_mtemp,
        "pos_weight": pos_weight,
        "scaler": scaler,
    }


# ==============================================================================
# 3. Model Zoo Initializers (Factories)
# ==============================================================================
def get_target1_classification_models(pos_weight: float) -> dict:
    """Returns candidate model zoo for Target 1: Heatwave_Flag (Binary)."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42
        ),
        "LightGBM": LGBMClassifier(
            scale_pos_weight=pos_weight,
            n_estimators=150,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=pos_weight,
            n_estimators=150,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def get_target2_severity_models() -> dict:
    """Returns candidate model zoo for Target 2: Heatwave_Severity (Multi-Class)."""
    return {
        "Logistic Regression (Multinomial)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42
        ),
        "LightGBM": LGBMClassifier(
            objective="multiclass",
            num_class=4,
            class_weight="balanced",
            n_estimators=150,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            n_estimators=150,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1,
        ),
    }


def get_target3_temperature_models() -> dict:
    """Returns candidate model zoo for Target 3: Next_Day_Max_Temp (Continuous Regression)."""
    return {
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.1, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=42
        ),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(
            random_state=42
        ),
        "LightGBM Regressor": LGBMRegressor(
            n_estimators=150, random_state=42, verbosity=-1, n_jobs=-1
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
    }


# ==============================================================================
# 4. Batch Training & Validation Benchmarking Engines
# ==============================================================================
def benchmark_target1_models(
    models: dict,
    X_train: pd.DataFrame,
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_val_scaled: np.ndarray,
    y_val: pd.Series,
) -> pd.DataFrame:
    """Trains and evaluates Target 1 models on validation partition, returning ranked leaderboard."""
    results = []
    for name, model in models.items():
        t0 = time.time()
        X_tr = X_train_scaled if "Logistic" in name else X_train
        X_va = X_val_scaled if "Logistic" in name else X_val

        model.fit(X_tr, y_train)
        train_time = time.time() - t0

        val_preds = model.predict(X_va)
        val_probs = model.predict_proba(X_va)[:, 1]

        results.append({
            "Model": name,
            "Train Time (s)": round(train_time, 2),
            "Precision": round(float(precision_score(y_val, val_preds, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_val, val_preds, zero_division=0)), 4),
            "F1-Score": round(float(f1_score(y_val, val_preds, zero_division=0)), 4),
            "PR-AUC": round(float(average_precision_score(y_val, val_probs)), 4),
            "ROC-AUC": round(float(roc_auc_score(y_val, val_probs)), 4),
            "Brier Score": round(float(brier_score_loss(y_val, val_probs)), 4),
        })

    return pd.DataFrame(results).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)


def benchmark_target2_models(
    models: dict,
    X_train: pd.DataFrame,
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_val_scaled: np.ndarray,
    y_val: pd.Series,
) -> pd.DataFrame:
    """Trains and evaluates Target 2 models on validation partition, returning ranked leaderboard."""
    results = []
    for name, model in models.items():
        t0 = time.time()
        X_tr = X_train_scaled if "Logistic" in name else X_train
        X_va = X_val_scaled if "Logistic" in name else X_val

        model.fit(X_tr, y_train)
        train_time = time.time() - t0

        val_preds = model.predict(X_va)
        val_probs = model.predict_proba(X_va)

        tier_recalls = recall_score(y_val, val_preds, average=None, zero_division=0)
        rec_t0 = float(tier_recalls[0]) if len(tier_recalls) > 0 else 0.0
        rec_t1 = float(tier_recalls[1]) if len(tier_recalls) > 1 else 0.0
        rec_t2 = float(tier_recalls[2]) if len(tier_recalls) > 2 else 0.0
        rec_t3 = float(tier_recalls[3]) if len(tier_recalls) > 3 else 0.0

        try:
            roc_ovr = float(roc_auc_score(y_val, val_probs, multi_class="ovr", average="macro"))
        except Exception:
            roc_ovr = float("nan")

        results.append({
            "Model": name,
            "Train Time (s)": round(train_time, 2),
            "Accuracy": round(float(accuracy_score(y_val, val_preds)), 4),
            "Macro F1": round(float(f1_score(y_val, val_preds, average="macro", zero_division=0)), 4),
            "Weighted F1": round(float(f1_score(y_val, val_preds, average="weighted", zero_division=0)), 4),
            "Tier 0 Recall": round(rec_t0, 4),
            "Tier 1 Recall": round(rec_t1, 4),
            "Tier 2 Recall": round(rec_t2, 4),
            "Tier 3 Recall": round(rec_t3, 4),
            "ROC-AUC (OvR)": round(roc_ovr, 4),
        })

    return pd.DataFrame(results).sort_values(by="Macro F1", ascending=False).reset_index(drop=True)


def benchmark_target3_models(
    models: dict,
    X_train: pd.DataFrame,
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_val_scaled: np.ndarray,
    y_val: pd.Series,
) -> pd.DataFrame:
    """Trains and evaluates Target 3 models on validation partition, returning ranked leaderboard."""
    results = []
    for name, model in models.items():
        t0 = time.time()
        is_linear = any(term in name for term in ["Ridge", "Lasso", "Linear"])
        X_tr = X_train_scaled if is_linear else X_train
        X_va = X_val_scaled if is_linear else X_val

        model.fit(X_tr, y_train)
        train_time = time.time() - t0

        val_preds = model.predict(X_va)

        mae = float(mean_absolute_error(y_val, val_preds))
        rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
        r2 = float(r2_score(y_val, val_preds))
        worst_miss = float(max_error(y_val, val_preds))

        results.append({
            "Model": name,
            "Train Time (s)": round(train_time, 2),
            "MAE (°C)": round(mae, 4),
            "RMSE (°C)": round(rmse, 4),
            "R² Score": round(r2, 4),
            "Max Error (°C)": round(worst_miss, 2),
        })

    return pd.DataFrame(results).sort_values(by="MAE (°C)", ascending=True).reset_index(drop=True)


# ==============================================================================
# 5. Full Pipeline Execution
# ==============================================================================
def run_model_deployment_pipeline(base_dir: Path | None = None) -> dict:
    """
    Executes the entire model deployment pipeline:
    data ingestion, splitting, scaling, model zoo benchmarking, and leaderboard generation.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    data_path = base_dir / "data" / "processed" / "terai_features_engineered.parquet"
    models_dir = base_dir / "models"

    df_meta, X, y_targets = load_and_preprocess_feature_matrix(data_path)
    data = split_and_scale_dataset(df_meta, X, y_targets, models_dir)

    models_target1 = get_target1_classification_models(data["pos_weight"])
    models_target2 = get_target2_severity_models()
    models_target3 = get_target3_temperature_models()

    leaderboard_target1 = benchmark_target1_models(
        models_target1,
        data["X_train"],
        data["X_train_scaled"],
        data["y_train_flag"],
        data["X_val"],
        data["X_val_scaled"],
        data["y_val_flag"],
    )

    leaderboard_target2 = benchmark_target2_models(
        models_target2,
        data["X_train"],
        data["X_train_scaled"],
        data["y_train_sev"],
        data["X_val"],
        data["X_val_scaled"],
        data["y_val_sev"],
    )

    leaderboard_target3 = benchmark_target3_models(
        models_target3,
        data["X_train"],
        data["X_train_scaled"],
        data["y_train_mtemp"],
        data["X_val"],
        data["X_val_scaled"],
        data["y_val_mtemp"],
    )

    return {
        "leaderboard_target1": leaderboard_target1,
        "leaderboard_target2": leaderboard_target2,
        "leaderboard_target3": leaderboard_target3,
        "models_target1": models_target1,
        "models_target2": models_target2,
        "models_target3": models_target3,
        "data": data,
    }


if __name__ == "__main__":
    _ = run_model_deployment_pipeline()
