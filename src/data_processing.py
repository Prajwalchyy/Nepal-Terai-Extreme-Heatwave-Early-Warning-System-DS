from pathlib import Path
import pandas as pd

# File Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "terai_heatwave_raw.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "clean_terai_heatwave.csv"

# Column mapping: NASA codes -> Standard names
COLUMN_MAP = {
    "YEAR": "Year",
    "DOY": "Day_of_Year",
    "T2M": "Mean_Temp_C",
    "T2M_MAX": "Max_Temp_C",
    "T2M_MIN": "Min_Temp_C",
    "RH2M": "Humidity_Pct",
    "WS2M": "Wind_Speed_ms",
    "PRECTOTCORR": "Precipitation_mm"
}


def load_data(filepath=RAW_DATA_PATH):
    """Load raw dataset from CSV."""
    return pd.read_csv(filepath)


def clean_terai_data(df):
    """
    Standardize and clean Terai climate dataset:
    - Standardize and cast string columns (Province, District, Regional_Hub) to category
    - Construct ISO Date from Year and Day_of_Year (datetime64)
    - Drop missing values (NaN)
    - Remove duplicates
    - Apply meteorological sanity checks
    - Sort and order columns
    """
    df = df.copy()

    # 1. Rename columns to standardized names
    df = df.rename(columns=COLUMN_MAP)

    # 2. Standardize text and convert to categorical dtype
    for col in ["Province", "District", "Regional_Hub"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title().astype("category")

    # 3. Construct ISO Date from Year + Day_of_Year
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Day_of_Year"].astype(str).str.zfill(3),
        format="%Y-%j"
    )

    # 4. Drop rows containing NaN values
    df = df.dropna().reset_index(drop=True)

    # 5. Remove duplicate records
    df = df.drop_duplicates(subset=["Date", "District"]).reset_index(drop=True)

    # 6. Physical meteorological sanity checks
    df = df[df["Min_Temp_C"] <= df["Max_Temp_C"]]
    df = df[(df["Humidity_Pct"] >= 0.0) & (df["Humidity_Pct"] <= 100.0)]
    df = df[(df["Wind_Speed_ms"] >= 0.0) & (df["Precipitation_mm"] >= 0.0)]

    # 7. Sort: Date descending, Province / Hub / District ascending
    df = df.sort_values(
        by=["Date", "Province", "Regional_Hub", "District"],
        ascending=[False, True, True, True]
    ).reset_index(drop=True)

    # 8. Clean column layout
    ordered_cols = [
        "Date", "Year", "Day_of_Year", "Province", "District", "Regional_Hub",
        "Latitude", "Longitude", "Max_Temp_C", "Min_Temp_C", "Mean_Temp_C",
        "Humidity_Pct", "Wind_Speed_ms", "Precipitation_mm"
    ]
    return df[[col for col in ordered_cols if col in df.columns]]


def process_and_save():
    """Load raw data, execute cleaning pipeline, and save clean dataset."""
    df = load_data()
    clean_df = clean_terai_data(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_DATA_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(clean_df):,} clean records to {PROCESSED_DATA_PATH}")
    return clean_df


if __name__ == "__main__":
    process_and_save()
