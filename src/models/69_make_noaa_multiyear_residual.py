from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_WATER_DIR = ROOT / "data" / "raw" / "NOAA_hourly_water_level_2023_2025"
RAW_TIDE_DIR = ROOT / "data" / "raw" / "NOAA_hourly_tide_predictions_2023_2025"
STATIONS_PATH = ROOT / "data" / "processed" / "stations.csv"
OUT_DIR = ROOT / "data" / "processed_multiyear_2023_2025"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_station_pair(station_id: str) -> pd.DataFrame:
    water_path = RAW_WATER_DIR / f"CO-OPS__{station_id}__hr_2023_2025.csv"
    tide_path = RAW_TIDE_DIR / f"CO-OPS__{station_id}__pr_2023_2025.csv"
    if not water_path.exists():
        raise FileNotFoundError(water_path)
    if not tide_path.exists():
        raise FileNotFoundError(tide_path)

    water = clean_columns(pd.read_csv(water_path))
    tide = clean_columns(pd.read_csv(tide_path))

    water["datetime"] = pd.to_datetime(water["Date Time"])
    tide["datetime"] = pd.to_datetime(tide["Date Time"])

    water = water.rename(
        columns={
            "Water Level": "water_level",
            "Sigma": "sigma",
            "I": "quality_i",
            "L": "quality_l",
        }
    )
    tide = tide.rename(columns={"Prediction": "tide"})

    keep_water = ["datetime", "water_level", "sigma", "quality_i", "quality_l"]
    keep_tide = ["datetime", "tide"]
    df = water[keep_water].merge(tide[keep_tide], on="datetime", how="inner")
    df["station_id"] = station_id

    for col in ["water_level", "sigma", "quality_i", "quality_l", "tide"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["residual"] = df["water_level"] - df["tide"]
    return df[
        [
            "datetime",
            "station_id",
            "water_level",
            "tide",
            "residual",
            "sigma",
            "quality_i",
            "quality_l",
        ]
    ]


def make_matrix(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    matrix = (
        long_df.pivot(index="datetime", columns="station_id", values=value_col)
        .sort_index()
        .reset_index()
    )
    matrix.columns.name = None
    return matrix


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(STATIONS_PATH)
    station_ids = [str(sid) for sid in stations["station_id"]]

    frames = [read_station_pair(station_id) for station_id in station_ids]
    long_df = pd.concat(frames, ignore_index=True).sort_values(["datetime", "station_id"])

    long_df.to_csv(OUT_DIR / "water_tide_residual_long.csv", index=False)
    make_matrix(long_df, "water_level").to_csv(OUT_DIR / "water_level_matrix.csv", index=False)
    make_matrix(long_df, "tide").to_csv(OUT_DIR / "tide_matrix.csv", index=False)
    make_matrix(long_df, "residual").to_csv(OUT_DIR / "residual_matrix.csv", index=False)
    stations.to_csv(OUT_DIR / "station_order.csv", index=False)

    summary = (
        long_df.groupby("station_id")
        .agg(
            rows=("datetime", "size"),
            start=("datetime", "min"),
            end=("datetime", "max"),
            missing_water=("water_level", lambda s: int(s.isna().sum())),
            missing_tide=("tide", lambda s: int(s.isna().sum())),
            missing_residual=("residual", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "noaa_multiyear_residual_summary.csv", index=False)

    print(f"Saved NOAA multi-year processed data to: {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
