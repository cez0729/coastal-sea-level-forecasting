from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
CURRENT_PATH = (
    ROOT
    / "data"
    / "raw"
    / "copernicus_currents"
    / "surface_currents_uo_vo_20250401_20251230.nc"
)
CURRENT_DIR = ROOT / "data" / "raw" / "copernicus_currents"
CURRENT_PATTERN = "surface_currents_uo_vo_20250401_20251230_*.nc"
STATIONS_PATH = ROOT / "data" / "processed" / "stations.csv"
OUT_PATH = ROOT / "data" / "processed" / "surface_currents_station_daily.csv"


def direction_to_degrees_from(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return ocean-current direction in degrees toward which the water flows."""
    return (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0


def nearest_valid_ocean_point(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    valid = np.isfinite(ds["uo"].mean(dim="time", skipna=True))
    if "depth" in valid.dims:
        valid = valid.isel(depth=0)

    lat_values = ds["latitude"].values.astype(np.float64)
    lon_values = ds["longitude"].values.astype(np.float64)
    valid_values = valid.values.astype(bool)

    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    distance2 = (lat_grid - lat) ** 2 + (lon_grid - lon) ** 2
    distance2 = np.where(valid_values, distance2, np.inf)

    if not np.isfinite(distance2).any():
        raise ValueError("No valid ocean current grid point found in subset.")

    lat_idx, lon_idx = np.unravel_index(np.argmin(distance2), distance2.shape)
    return ds.isel(latitude=lat_idx, longitude=lon_idx)


def main() -> None:
    paths = sorted(CURRENT_DIR.glob(CURRENT_PATTERN))
    if CURRENT_PATH.exists():
        paths = [CURRENT_PATH]

    if not paths:
        raise FileNotFoundError(
            f"Missing {CURRENT_PATH} or {CURRENT_PATTERN}. "
            "Run 65_download_copernicus_surface_currents.py first."
        )

    stations = pd.read_csv(STATIONS_PATH)

    temp_dir = Path(tempfile.gettempdir()) / "sea_level_copernicus_currents"
    temp_dir.mkdir(parents=True, exist_ok=True)
    ascii_paths = []
    for idx, path in enumerate(paths):
        target = temp_dir / f"current_{idx:02d}.nc"
        shutil.copy2(path, target)
        ascii_paths.append(target)

    datasets = [xr.open_dataset(path) for path in ascii_paths]
    ds = xr.concat(datasets, dim="time").sortby("time")

    if "depth" in ds.dims:
        ds = ds.sel(depth=ds["depth"].min(), method="nearest")

    time_name = "time" if "time" in ds.coords else "valid_time"
    rows = []

    for station in stations.itertuples(index=False):
        point = nearest_valid_ocean_point(ds, float(station.lat), float(station.lon))

        df = point[["uo", "vo"]].to_dataframe().reset_index()
        df = df.rename(columns={time_name: "datetime"})
        df["station_id"] = station.station_id
        df["station_name"] = station.station_name
        df["station_lat"] = station.lat
        df["station_lon"] = station.lon
        df["current_lat"] = float(point["latitude"].values)
        df["current_lon"] = float(point["longitude"].values)
        df["current_speed_mps"] = np.sqrt(df["uo"] ** 2 + df["vo"] ** 2)
        df["current_direction_deg_toward"] = direction_to_degrees_from(
            df["uo"].to_numpy(dtype=np.float64),
            df["vo"].to_numpy(dtype=np.float64),
        )
        rows.append(df)

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
    out[columns].to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out):,} rows to {OUT_PATH}")
    ds.close()


if __name__ == "__main__":
    main()
