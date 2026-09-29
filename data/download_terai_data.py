"""
NASA POWER API - Nepal Terai Climate Downloader (All 21 Districts)
Official Portal: https://power.larc.nasa.gov
"""

import datetime
import io
import os
import time
import urllib.request
import numpy as np
import pandas as pd

# Directory where the raw combined CSV file will be saved
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw", "terai_heatwave_raw.csv")

# All 21 Terai Districts of Nepal across 6 Provinces (East to West)
TERAI_DISTRICTS = [
    {"district": "Jhapa", "hub": "Bhadrapur", "province": "Koshi", "lat": 26.54, "lon": 88.08},
    {"district": "Morang", "hub": "Biratnagar", "province": "Koshi", "lat": 26.45, "lon": 87.27},
    {"district": "Sunsari", "hub": "Inaruwa", "province": "Koshi", "lat": 26.60, "lon": 87.15},
    {"district": "Saptari", "hub": "Rajbiraj", "province": "Madhesh", "lat": 26.54, "lon": 86.75},
    {"district": "Siraha", "hub": "Lahan", "province": "Madhesh", "lat": 26.65, "lon": 86.20},
    {"district": "Dhanusha", "hub": "Janakpur", "province": "Madhesh", "lat": 26.73, "lon": 85.92},
    {"district": "Mahottari", "hub": "Jaleshwar", "province": "Madhesh", "lat": 26.64, "lon": 85.80},
    {"district": "Sarlahi", "hub": "Malangwa", "province": "Madhesh", "lat": 26.86, "lon": 85.56},
    {"district": "Rautahat", "hub": "Gaur", "province": "Madhesh", "lat": 26.76, "lon": 85.28},
    {"district": "Bara", "hub": "Kalaiya", "province": "Madhesh", "lat": 27.03, "lon": 85.00},
    {"district": "Parsa", "hub": "Birgunj", "province": "Madhesh", "lat": 27.01, "lon": 84.87},
    {"district": "Chitwan", "hub": "Bharatpur", "province": "Bagmati", "lat": 27.68, "lon": 84.43},
    {"district": "Nawalpur", "hub": "Kawasoti", "province": "Gandaki", "lat": 27.65, "lon": 84.12},
    {"district": "Parasi", "hub": "Ramgram", "province": "Lumbini", "lat": 27.53, "lon": 83.67},
    {"district": "Rupandehi", "hub": "Bhairahawa", "province": "Lumbini", "lat": 27.50, "lon": 83.45},
    {"district": "Kapilvastu", "hub": "Taulihawa", "province": "Lumbini", "lat": 27.55, "lon": 83.05},
    {"district": "Dang", "hub": "Ghorahi", "province": "Lumbini", "lat": 28.04, "lon": 82.49},
    {"district": "Banke", "hub": "Nepalgunj", "province": "Lumbini", "lat": 28.05, "lon": 81.62},
    {"district": "Bardiya", "hub": "Gulariya", "province": "Lumbini", "lat": 28.23, "lon": 81.33},
    {"district": "Kailali", "hub": "Dhangadhi", "province": "Sudurpashchim", "lat": 28.68, "lon": 80.59},
    {"district": "Kanchanpur", "hub": "Bhimdatta", "province": "Sudurpashchim", "lat": 28.97, "lon": 80.18}
]

START_DATE = "20000101"
END_DATE = datetime.datetime.now().strftime("%Y%m%d")
PARAMETERS = "T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,PRECTOTCORR"
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_district_dataframe(dist_info, current_idx, total_count):
    """Queries NASA POWER API for a single district and returns a pandas DataFrame."""
    district = dist_info["district"]
    hub = dist_info["hub"]
    province = dist_info["province"]
    lat = dist_info["lat"]
    lon = dist_info["lon"]

    url = (
        f"{BASE_URL}"
        f"?parameters={PARAMETERS}"
        f"&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={START_DATE}"
        f"&end={END_DATE}"
        f"&format=CSV"
    )

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read().decode("utf-8")

        lines = content.splitlines()
        header_end_idx = 0
        for idx, line in enumerate(lines):
            if "-END HEADER-" in line:
                header_end_idx = idx + 1
                break

        csv_body = "\n".join(lines[header_end_idx:])
        if not csv_body.strip():
            raise ValueError("Empty data returned from NASA API.")

        df = pd.read_csv(io.StringIO(csv_body))

        df["District"] = district
        df["Province"] = province
        df["Regional_Hub"] = hub
        df["Latitude"] = lat
        df["Longitude"] = lon

        print(f"[{current_idx}/{total_count}] Fetched {district} ({len(df):,} records)")
        return df

    except Exception as e:
        print(f"[{current_idx}/{total_count}] Error fetching {district}: {e}")
        return None


def main():
    total_districts = len(TERAI_DISTRICTS)
    print(f"Downloading NASA POWER climate data for {total_districts} Terai districts (2000 to present)...")

    district_dfs = []
    for idx, dist_info in enumerate(TERAI_DISTRICTS, 1):
        df_dist = fetch_district_dataframe(dist_info, idx, total_districts)
        if df_dist is not None:
            district_dfs.append(df_dist)
        time.sleep(1)

    if not district_dfs:
        print("Error: No data was retrieved.")
        return

    raw_df = pd.concat(district_dfs, ignore_index=True)

    # Sort descending by date
    raw_df = raw_df.sort_values(
        by=["YEAR", "DOY", "Province", "Regional_Hub", "District"],
        ascending=[False, False, True, True, True]
    ).reset_index(drop=True)

    # Replace NASA missing/fill values (-999.0 and -999) with NaN
    raw_df.replace([-999.0, -999], np.nan, inplace=True)

    # Standard raw column layout
    raw_columns = [
        "YEAR",
        "DOY",
        "Province",
        "District",
        "Regional_Hub",
        "Latitude",
        "Longitude",
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "RH2M",
        "WS2M",
        "PRECTOTCORR"
    ]
    raw_df = raw_df[raw_columns]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    raw_df.to_csv(OUTPUT_FILE, index=False, na_rep="NaN")
    print(f"Saved {len(raw_df):,} raw records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
