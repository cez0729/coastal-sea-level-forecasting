from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import shutil
import tempfile

import numpy as np
import pandas as pd
import xarray as xr


ROOT = REPO_ROOT
ERA5_DIR = ROOT / "data" / "raw" / "ERA5_2023_2025"
STATIONS_PATH = ROOT / "data" / "processed" / "stations.csv"
OUT_DIR = ROOT / "data" / "processed_multiyear_2023_2025"


def copy_to_ascii_temp(paths: list[Path]) -> list[Path]:
    temp_dir = Path(tempfile.gettempdir()) / "sea_level_multiyear_era5"
    temp_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for idx, path in enumerate(paths):
        target = temp_dir / f"era5_{idx:02d}_{path.stem[-6:]}.nc"
        if not target.exists() or target.stat().st_size != path.stat().st_size:
            shutil.copy2(path, target)
        out.append(target)
    return out


def nearest_grid_series(ds: xr.Dataset, station: pd.Series) -> pd.DataFrame:
    lat_name = "latitude"
    lon_name = "longitude"
    time_name = "valid_time" if "valid_time" in ds.coords else "time"

    point = ds.sel(
        {
            lat_name: float(station["lat"]),
            lon_name: float(station["lon"]),
        },
        method="nearest",
    )
    df = point[["u10", "v10", "msl"]].to_dataframe().reset_index()
    df = df.rename(columns={time_name: "datetime"})
    df["station_id"] = station["station_id"]
    df["station_name"] = station["station_name"]
    df["station_lat"] = station["lat"]
    df["station_lon"] = station["lon"]
    df["era5_lat"] = float(point[lat_name].values)
    df["era5_lon"] = float(point[lon_name].values)
    df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    df["wind_stress_u_proxy"] = df["u10"] * df["wind_speed"]
    df["wind_stress_v_proxy"] = df["v10"] * df["wind_speed"]
    return df


def main() -> None:
    paths = sorted(ERA5_DIR.glob("era5_single_levels_u10_v10_msl_*.nc"))
    if len(paths) != 36:
        raise FileNotFoundError(f"Expected 36 ERA5 monthly files, found {len(paths)} in {ERA5_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ascii_paths = copy_to_ascii_temp(paths)
    datasets = [xr.open_dataset(path) for path in ascii_paths]
    ds = xr.concat(datasets, dim="valid_time" if "valid_time" in datasets[0].coords else "time").sortby(
        "valid_time" if "valid_time" in datasets[0].coords else "time"
    )

    stations = pd.read_csv(STATIONS_PATH)
    rows = [nearest_grid_series(ds, station) for _, station in stations.iterrows()]
    out = pd.concat(rows, ignore_index=True).sort_values(["datetime", "station_id"])
    columns = [
        "datetime",
        "station_id",
        "station_name",
        "station_lat",
        "station_lon",
        "era5_lat",
        "era5_lon",
        "u10",
        "v10",
        "msl",
        "wind_speed",
        "wind_stress_u_proxy",
        "wind_stress_v_proxy",
    ]
    out[columns].to_csv(OUT_DIR / "era5_station_hourly.csv", index=False)

    summary = (
        out.groupby("station_id")
        .agg(
            rows=("datetime", "size"),
            start=("datetime", "min"),
            end=("datetime", "max"),
            missing_u10=("u10", lambda s: int(s.isna().sum())),
            missing_v10=("v10", lambda s: int(s.isna().sum())),
            missing_msl=("msl", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "era5_station_hourly_summary.csv", index=False)
    print(f"Saved {len(out):,} rows to {OUT_DIR / 'era5_station_hourly.csv'}")
    print(summary.to_string(index=False))

    ds.close()
    for dataset in datasets:
        dataset.close()


if __name__ == "__main__":
    main()
