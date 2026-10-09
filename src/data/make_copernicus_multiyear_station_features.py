from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import shutil
import tempfile

import numpy as np
import pandas as pd
import xarray as xr


ROOT = REPO_ROOT
STATIONS_PATH = ROOT / "data" / "processed" / "stations.csv"
DEPTH_PATH = ROOT / "data" / "processed" / "gebco_station_depth.csv"
CURRENT_PATH = (
    ROOT
    / "data"
    / "raw"
    / "copernicus_currents_2023_2025"
    / "cmems_mod_glo_phy_my_0.083deg_P1D-m_20230101_20251231.nc"
)
WAVE_PATH = (
    ROOT
    / "data"
    / "raw"
    / "copernicus_waves_2023_2025"
    / "cmems_mod_glo_wav_my_0.2deg_PT3H-i_20230101_20251231.nc"
)
OUT_DIR = ROOT / "data" / "processed_multiyear_2023_2025"

GRAVITY = 9.80665


def open_dataset_via_ascii_temp(path: Path, temp_name: str) -> xr.Dataset:
    """netCDF4 can fail on this Windows Chinese project path; open an ASCII temp copy."""
    temp_dir = Path(tempfile.gettempdir()) / "sea_level_multiyear_copernicus"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / temp_name
    if not temp_path.exists() or temp_path.stat().st_size != path.stat().st_size:
        shutil.copy2(path, temp_path)
    return xr.open_dataset(temp_path)


def direction_to_degrees_toward(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0


def nearest_valid_point(ds: xr.Dataset, var_name: str, lat: float, lon: float) -> xr.Dataset:
    valid = np.isfinite(ds[var_name].mean(dim="time", skipna=True))
    if "depth" in valid.dims:
        valid = valid.isel(depth=0)

    lat_values = ds["latitude"].values.astype(np.float64)
    lon_values = ds["longitude"].values.astype(np.float64)
    valid_values = valid.values.astype(bool)

    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    distance2 = (lat_grid - lat) ** 2 + (lon_grid - lon) ** 2
    distance2 = np.where(valid_values, distance2, np.inf)
    if not np.isfinite(distance2).any():
        raise ValueError(f"No valid ocean point found for variable {var_name}.")

    lat_idx, lon_idx = np.unravel_index(np.argmin(distance2), distance2.shape)
    return ds.isel(latitude=lat_idx, longitude=lon_idx)


def solve_wavenumber(period_seconds: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    period_seconds = np.asarray(period_seconds, dtype=np.float64)
    depth_m = np.maximum(np.asarray(depth_m, dtype=np.float64), 0.1)
    omega = 2.0 * np.pi / np.maximum(period_seconds, 0.1)
    k = np.maximum((omega**2) / GRAVITY, 1e-6)
    for _ in range(30):
        kh = k * depth_m
        tanh_kh = np.tanh(kh)
        f = GRAVITY * k * tanh_kh - omega**2
        df = GRAVITY * (tanh_kh + kh * (1.0 - tanh_kh**2))
        k = np.maximum(k - f / np.maximum(df, 1e-12), 1e-8)
    return k


def phase_speed(period_seconds: pd.Series, depth_m: pd.Series) -> np.ndarray:
    period = period_seconds.to_numpy(dtype=np.float64)
    depth = depth_m.to_numpy(dtype=np.float64)
    omega = 2.0 * np.pi / np.maximum(period, 0.1)
    k = solve_wavenumber(period, depth)
    return omega / k


def make_currents(stations: pd.DataFrame) -> pd.DataFrame:
    if not CURRENT_PATH.exists():
        raise FileNotFoundError(CURRENT_PATH)

    ds = open_dataset_via_ascii_temp(CURRENT_PATH, "currents_20230101_20251231.nc")
    if "depth" in ds.dims:
        ds = ds.sel(depth=ds["depth"].min(), method="nearest")

    rows = []
    for station in stations.itertuples(index=False):
        point = nearest_valid_point(ds, "uo", float(station.lat), float(station.lon))
        df = point[["uo", "vo"]].to_dataframe().reset_index()
        df = df.rename(columns={"time": "datetime"})
        df["station_id"] = station.station_id
        df["station_name"] = station.station_name
        df["station_lat"] = station.lat
        df["station_lon"] = station.lon
        df["current_lat"] = float(point["latitude"].values)
        df["current_lon"] = float(point["longitude"].values)
        df["current_speed_mps"] = np.sqrt(df["uo"] ** 2 + df["vo"] ** 2)
        df["current_direction_deg_toward"] = direction_to_degrees_toward(
            df["uo"].to_numpy(dtype=np.float64),
            df["vo"].to_numpy(dtype=np.float64),
        )
        rows.append(df)
    ds.close()

    out = pd.concat(rows, ignore_index=True)
    columns = [
        "datetime",
        "station_id",
        "station_name",
        "station_lat",
        "station_lon",
        "current_lat",
        "current_lon",
        "uo",
        "vo",
        "current_speed_mps",
        "current_direction_deg_toward",
    ]
    return out[columns].sort_values(["datetime", "station_id"])


def make_waves(stations: pd.DataFrame) -> pd.DataFrame:
    if not WAVE_PATH.exists():
        raise FileNotFoundError(WAVE_PATH)

    depth = pd.read_csv(DEPTH_PATH, dtype={"station_id": str})[["station_id", "depth"]]
    depth["station_id"] = depth["station_id"].astype(str)
    ds = open_dataset_via_ascii_temp(WAVE_PATH, "waves_20230101_20251231.nc")
    wave_vars = [
        "VHM0",
        "VTPK",
        "VTM10",
        "VMDR",
        "VSDX",
        "VSDY",
        "VPED",
    ]
    wave_vars = [v for v in wave_vars if v in ds.data_vars]

    rows = []
    for station in stations.itertuples(index=False):
        point = nearest_valid_point(ds, "VHM0", float(station.lat), float(station.lon))
        df = point[wave_vars].to_dataframe().reset_index()
        df = df.rename(
            columns={
                "time": "datetime",
                "VHM0": "wave_height",
                "VTPK": "wave_peak_period",
                "VTM10": "wave_mean_period",
                "VMDR": "wave_direction",
                "VSDX": "wave_stokes_drift_x",
                "VSDY": "wave_stokes_drift_y",
                "VPED": "wave_peak_directional_spread",
            }
        )
        df["station_id"] = station.station_id
        df["station_name"] = station.station_name
        df["station_lat"] = station.lat
        df["station_lon"] = station.lon
        df["wave_lat"] = float(point["latitude"].values)
        df["wave_lon"] = float(point["longitude"].values)
        rows.append(df)
    ds.close()

    out = pd.concat(rows, ignore_index=True)
    out = out.merge(depth, on="station_id", how="left")
    out["wave_speed_from_peak_period_mps"] = phase_speed(
        out["wave_peak_period"], out["depth"]
    )
    out["wave_speed_from_mean_period_mps"] = phase_speed(
        out["wave_mean_period"], out["depth"]
    )
    direction_rad = np.deg2rad(out["wave_direction"].to_numpy(dtype=np.float64))
    out["wave_dir_x_from_mps"] = out["wave_speed_from_peak_period_mps"] * np.sin(direction_rad)
    out["wave_dir_y_from_mps"] = out["wave_speed_from_peak_period_mps"] * np.cos(direction_rad)

    columns = [
        "datetime",
        "station_id",
        "station_name",
        "station_lat",
        "station_lon",
        "wave_lat",
        "wave_lon",
        "depth",
        "wave_height",
        "wave_peak_period",
        "wave_mean_period",
        "wave_direction",
        "wave_stokes_drift_x",
        "wave_stokes_drift_y",
        "wave_peak_directional_spread",
        "wave_speed_from_peak_period_mps",
        "wave_speed_from_mean_period_mps",
        "wave_dir_x_from_mps",
        "wave_dir_y_from_mps",
    ]
    return out[columns].sort_values(["datetime", "station_id"])


def write_summary(df: pd.DataFrame, path: Path, value_cols: list[str]) -> None:
    summary = (
        df.groupby("station_id")
        .agg(
            rows=("datetime", "size"),
            start=("datetime", "min"),
            end=("datetime", "max"),
            **{
                f"missing_{col}": (col, lambda s: int(s.isna().sum()))
                for col in value_cols
                if col in df.columns
            },
        )
        .reset_index()
    )
    summary.to_csv(path, index=False)
    print(summary.to_string(index=False))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(STATIONS_PATH)

    currents = make_currents(stations)
    currents_path = OUT_DIR / "surface_currents_station_daily.csv"
    currents.to_csv(currents_path, index=False)
    print(f"Saved {len(currents):,} rows to {currents_path}")
    write_summary(
        currents,
        OUT_DIR / "surface_currents_station_daily_summary.csv",
        ["uo", "vo", "current_speed_mps"],
    )

    waves = make_waves(stations)
    waves_path = OUT_DIR / "wave_direction_speed_station_3hourly.csv"
    waves.to_csv(waves_path, index=False)
    print(f"Saved {len(waves):,} rows to {waves_path}")
    write_summary(
        waves,
        OUT_DIR / "wave_direction_speed_station_3hourly_summary.csv",
        ["wave_height", "wave_peak_period", "wave_mean_period", "wave_direction"],
    )


if __name__ == "__main__":
    main()
