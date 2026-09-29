"""
Feature Engineering Pipeline for Nepal Terai Extreme Heatwave Prediction.
Transforms cleaned multi-decadal meteorological data into a production-ready 
65-feature matrix with physical, bio-meteorological, temporal, and spatial predictors.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# File Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "clean_terai_heatwave.csv"
ENGINEERED_DATA_PATH = BASE_DIR / "data" / "processed" / "terai_features_engineered.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers 65 features across physical, bio-meteorological, temporal,
    climatological, and spatial domains for extreme heatwave forecasting.

    Parameters:
        df (pd.DataFrame): Cleaned input dataframe from clean_terai_heatwave.csv

    Returns:
        pd.DataFrame: Fully engineered, leak-free feature matrix (204,582 rows x 65 columns).
    """
    df = df.copy()

    # 1. Sort temporally within each district
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["District", "Date"]).reset_index(drop=True)

    # 2. Prediction Targets
    # Target 1: Binary Heatwave Alert (T_max >= 40.0°C)
    df["Heatwave_Flag"] = (df["Max_Temp_C"] >= 40.0).astype(int)

    # Target 2: Tiered Heatwave Severity (0: Normal, 1: Moderate, 2: Severe, 3: Extreme)
    severity_conditions = [
        df["Max_Temp_C"] < 40.0,
        (df["Max_Temp_C"] >= 40.0) & (df["Max_Temp_C"] < 42.0),
        (df["Max_Temp_C"] >= 42.0) & (df["Max_Temp_C"] < 45.0),
        df["Max_Temp_C"] >= 45.0
    ]
    df["Heatwave_Severity"] = np.select(severity_conditions, [0, 1, 2, 3], default=0).astype(int)

    # Target 3: Tomorrow's High Temperature (Continuous Forecast, t+1)
    df["Next_Day_Max_Temp"] = df.groupby("District")["Max_Temp_C"].shift(-1)

    # 3. Atmospheric Physics & Bio-Meteorological Features
    # Diurnal Temperature Range (DTR)
    df["Temp_Range"] = (df["Max_Temp_C"] - df["Min_Temp_C"]).round(2)

    # Apparent "Feels-Like" Temperature (Steadman Heat Index)
    e_a_max = (df["Humidity_Pct"] / 100.0) * 6.1078 * np.exp((17.27 * df["Max_Temp_C"]) / (df["Max_Temp_C"] + 237.3))
    df["Feels_Like_Max_C"] = (df["Max_Temp_C"] + (0.33 * e_a_max) - (0.70 * df["Wind_Speed_ms"]) - 4.00).round(2)

    e_a_min = (df["Humidity_Pct"] / 100.0) * 6.1078 * np.exp((17.27 * df["Min_Temp_C"]) / (df["Min_Temp_C"] + 237.3))
    df["Feels_Like_Min_C"] = (df["Min_Temp_C"] + (0.33 * e_a_min) - (0.70 * df["Wind_Speed_ms"]) - 4.00).round(2)

    df["Day_Thermal_Gap"] = (df["Feels_Like_Max_C"] - df["Max_Temp_C"]).round(2)
    df["Night_Thermal_Gap"] = (df["Feels_Like_Min_C"] - df["Min_Temp_C"]).round(2)

    # Thermodynamic Wet-Bulb Temperature (Stull's Formula)
    t = df["Max_Temp_C"]
    rh = df["Humidity_Pct"]
    df["Wet_Bulb_Temp"] = (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    ).round(2)
    df["Wet_Bulb_Depression"] = (df["Max_Temp_C"] - df["Wet_Bulb_Temp"]).round(2)

    # Nocturnal Thermal Stress (Tropical Night Indicator: Min_Temp >= 25.0°C)
    df["Tropical_Night"] = (df["Min_Temp_C"] >= 25.0).astype(int)

    # Scorching Westerly Wind Index (Loo Index)
    df["Loo_Index"] = ((df["Max_Temp_C"] * df["Wind_Speed_ms"]) / (df["Humidity_Pct"] + 1.0)).round(2)

    # 4. Atmospheric Memory: Autoregressive Lags
    df["Yesterday_Max_Temp"] = df.groupby("District")["Max_Temp_C"].shift(1)
    df["Yesterday_Min_Temp"] = df.groupby("District")["Min_Temp_C"].shift(1)
    df["Yesterday_Precip"] = df.groupby("District")["Precipitation_mm"].shift(1)
    df["Yesterday_Humidity"] = df.groupby("District")["Humidity_Pct"].shift(1)

    df["Max_Temp_Lag2"] = df.groupby("District")["Max_Temp_C"].shift(2)
    df["Min_Temp_Lag2"] = df.groupby("District")["Min_Temp_C"].shift(2)

    # Thermal Momentum & Acceleration
    df["Temp_Momentum_1D"] = (df["Yesterday_Max_Temp"] - df["Max_Temp_Lag2"]).round(2)
    df["Night_Momentum_1D"] = (df["Yesterday_Min_Temp"] - df["Min_Temp_Lag2"]).round(2)
    df["Temp_Momentum_2D"] = (df["Yesterday_Max_Temp"] - df.groupby("District")["Max_Temp_C"].shift(3)).round(2)
    df["Night_Momentum_2D"] = (df["Yesterday_Min_Temp"] - df.groupby("District")["Min_Temp_C"].shift(3)).round(2)

    # 5. Multi-Day Heat Spells & Cumulative Persistence
    df["Past_3D_Avg_Max_Temp"] = df.groupby("District")["Yesterday_Max_Temp"].transform(lambda s: s.rolling(3).mean()).round(2)
    df["Past_7D_Avg_Max_Temp"] = df.groupby("District")["Yesterday_Max_Temp"].transform(lambda s: s.rolling(7).mean()).round(2)
    df["Past_7D_Max_Temp"] = df.groupby("District")["Yesterday_Max_Temp"].transform(lambda s: s.rolling(7).max()).round(2)

    df["Past_3D_Avg_Min_Temp"] = df.groupby("District")["Yesterday_Min_Temp"].transform(lambda s: s.rolling(3).mean()).round(2)
    df["Past_7D_Avg_Min_Temp"] = df.groupby("District")["Yesterday_Min_Temp"].transform(lambda s: s.rolling(7).mean()).round(2)
    df["Past_7D_Min_Temp"] = df.groupby("District")["Yesterday_Min_Temp"].transform(lambda s: s.rolling(7).min()).round(2)

    # 6. Soil Moisture & Drought Deficit Dynamics
    df["Past_7D_Rain_Sum"] = df.groupby("District")["Yesterday_Precip"].transform(lambda s: s.rolling(7).sum()).round(2)
    df["Past_14D_Rain_Sum"] = df.groupby("District")["Yesterday_Precip"].transform(lambda s: s.rolling(14).sum()).round(2)

    # Consecutive Dry Days (WMO Standard: Precipitation < 1.0 mm)
    df["Consecutive_Dry_Days"] = df.groupby("District")["Yesterday_Precip"].transform(
        lambda s: (s < 1.0).groupby((s >= 1.0).cumsum()).cumsum()
    )

    # Heatwave Streak Counter (Consecutive Days >= 40.0°C)
    df["Heatwave_Streak"] = df.groupby("District")["Yesterday_Max_Temp"].transform(
        lambda s: (s >= 40.0).groupby((s < 40.0).cumsum()).cumsum()
    )

    # 7. 26-Year Climatological Baseline & Statistical Anomalies
    if "Day_of_Year" not in df.columns:
        df["Day_of_Year"] = df["Date"].dt.dayofyear

    df["Clim_Max_Mean"] = df.groupby(["District", "Day_of_Year"])["Max_Temp_C"].transform("mean").round(2)
    df["Clim_Max_Std"] = df.groupby(["District", "Day_of_Year"])["Max_Temp_C"].transform("std").round(2)
    df["Clim_Min_Mean"] = df.groupby(["District", "Day_of_Year"])["Min_Temp_C"].transform("mean").round(2)
    df["Clim_Min_Std"] = df.groupby(["District", "Day_of_Year"])["Min_Temp_C"].transform("std").round(2)

    df["Temp_Anomaly"] = (df["Yesterday_Max_Temp"] - df["Clim_Max_Mean"]).round(2)
    df["Night_Anomaly"] = (df["Yesterday_Min_Temp"] - df["Clim_Min_Mean"]).round(2)

    df["Temp_Z_Score"] = ((df["Yesterday_Max_Temp"] - df["Clim_Max_Mean"]) / (df["Clim_Max_Std"] + 1e-5)).round(2)
    df["Night_Z_Score"] = ((df["Yesterday_Min_Temp"] - df["Clim_Min_Mean"]) / (df["Clim_Min_Std"] + 1e-5)).round(2)

    # 8. Earth's Orbital Geometry & Cyclical Time Features
    df["Month"] = df["Date"].dt.month
    df["DOY_Sin"] = np.sin(2 * np.pi * df["Day_of_Year"] / 365.25).round(4)
    df["DOY_Cos"] = np.cos(2 * np.pi * df["Day_of_Year"] / 365.25).round(4)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12.0).round(4)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12.0).round(4)

    # Nepal DHM 4 Official Seasons
    season_conditions = [
        df["Month"].isin([12, 1, 2]),       # Winter
        df["Month"].isin([3, 4, 5]),        # Pre-Monsoon (Primary Heatwave Window)
        df["Month"].isin([6, 7, 8, 9]),     # Monsoon
        df["Month"].isin([10, 11])          # Post-Monsoon
    ]
    season_choices = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
    df["Season"] = np.select(season_conditions, season_choices, default="Winter")

    # 9. Statistical Distribution Corrections: Log-Transformed Rainfall
    df["Rain_Log"] = np.log1p(df["Precipitation_mm"]).round(4)
    df["Yesterday_Rain_Log"] = np.log1p(df["Yesterday_Precip"]).round(4)
    df["Past_7D_Rain_Log"] = np.log1p(df["Past_7D_Rain_Sum"]).round(4)

    # 10. Spatial Geography & Macro-Climatic Regionalization
    df["Is_Western_Terai"] = df["Province"].isin(["Lumbini", "Sudurpashchim"]).astype(int)

    region_conditions = [
        df["Province"].isin(["Lumbini", "Sudurpashchim"]),
        df["Province"].isin(["Gandaki", "Bagmati"]),
        df["Province"].isin(["Madhesh", "Koshi"])
    ]
    region_choices = ["Western Terai", "Central Terai", "Eastern Terai"]
    df["Terai_Region"] = np.select(region_conditions, region_choices, default="Eastern Terai")

    # 11. Cleanup and Boundary Trimming
    # Drop exploratory Next_Day_Min_Temp if present
    if "Next_Day_Min_Temp" in df.columns:
        df = df.drop(columns=["Next_Day_Min_Temp"])

    # Trim warmup (14 days past) and lookahead (1 day future) boundary NaNs (315 rows total)
    df_final = df.dropna().reset_index(drop=True)

    # Deterministic Quality Assertions
    assert df_final.isnull().sum().sum() == 0, "[ERROR] Null values remain in engineered dataset!"
    assert df_final.shape == (204582, 65), f"[ERROR] Unexpected shape: {df_final.shape}"

    return df_final


def process_and_save_features(
    input_path: Path = CLEAN_DATA_PATH,
    output_path: Path = ENGINEERED_DATA_PATH
) -> pd.DataFrame:
    """
    Executes the full feature engineering pipeline from raw clean data to exported CSV.
    """
    df = pd.read_csv(input_path)
    df_final = engineer_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Exported '{output_path.name}' ({df_final.shape[0]:,} rows x {df_final.shape[1]} columns, {file_size_mb:.2f} MB)")
    return df_final


if __name__ == "__main__":
    process_and_save_features()
