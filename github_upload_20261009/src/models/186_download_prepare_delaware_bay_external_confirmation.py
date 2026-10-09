"""Download and prepare the preregistered Delaware Bay external region.

This script may inspect data availability, but it never trains or scores a
forecast model. Station retention is determined only by the frozen coverage
rule in configs/delaware_bay_external_confirmation_20260812.json.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import importlib.util
import json
import math
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "delaware_bay_external_confirmation_20260812.json"
BASE_DIR = ROOT / "data" / "external_region_delaware_bay_2023_2025"
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
NOAA_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_META = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


marine = load_module("delaware_marine", HERE / "71_make_copernicus_multiyear_station_features.py")
era5_impl = load_module("delaware_era5", HERE / "72_make_era5_multiyear_station_features.py")
coops_impl = load_module("delaware_coops", HERE / "75_download_noaa_ndbc_enhanced_forcing.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, retries: int = 5, timeout: int = 90) -> bytes:
    last_error = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "sea-level-external-confirmation/1.0"})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}: {last_error}")


def month_ranges(start_year: int, end_year: int):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            last = calendar.monthrange(year, month)[1]
            yield date(year, month, 1), date(year, month, last)


def noaa_url(station: str, product: str, begin: date, end: date) -> str:
    params = {
        "station": station,
        "begin_date": begin.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "product": product,
        "datum": "MLLW",
        "time_zone": "gmt",
        "units": "metric",
        "format": "csv",
        "application": "delaware_external_confirmation",
    }
    if product != "hourly_height":
        params["interval"] = "h"
    return NOAA_API + "?" + urlencode(params)


def download_noaa_job(job: tuple[str, str, date, date]) -> dict:
    station, product, begin, end = job
    target_dir = RAW_DIR / "coops" / product / station
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{station}_{product}_{begin:%Y%m}.csv"
    if target.exists() and target.stat().st_size > 0:
        return {"station_id": station, "product": product, "month": f"{begin:%Y-%m}", "status": "skipped", "path": str(target.relative_to(ROOT))}
    url = noaa_url(station, product, begin, end)
    payload = fetch(url)
    text = payload.decode("utf-8-sig", errors="replace")
    if "Date Time" not in text[:500]:
        raise RuntimeError(f"No data columns: {station} {product} {begin:%Y-%m}: {text[:240]}")
    target.write_bytes(payload)
    return {"station_id": station, "product": product, "month": f"{begin:%Y-%m}", "status": "downloaded", "path": str(target.relative_to(ROOT)), "sha256": sha256(target)}


def read_noaa_product(station: str, product: str) -> pd.DataFrame:
    value = "Water Level" if product == "hourly_height" else "Prediction"
    frames = []
    for path in sorted((RAW_DIR / "coops" / product / station).glob("*.csv")):
        frame = pd.read_csv(path, skipinitialspace=True)
        frame.columns = [str(column).strip() for column in frame.columns]
        if "Date Time" not in frame or value not in frame:
            continue
        keep = frame[["Date Time", value]].copy()
        keep["datetime"] = pd.to_datetime(keep.pop("Date Time"), errors="coerce")
        keep[value] = pd.to_numeric(keep[value], errors="coerce")
        frames.append(keep.dropna(subset=["datetime"]))
    if not frames:
        return pd.DataFrame(columns=["datetime", value])
    return pd.concat(frames, ignore_index=True).drop_duplicates("datetime", keep="last").sort_values("datetime")


def verify_metadata(config: dict) -> pd.DataFrame:
    rows = []
    for expected in config["stations"]:
        station = expected["station_id"]
        payload = json.loads(fetch(f"{NOAA_META}/{station}.json").decode("utf-8"))
        actual = payload["stations"][0]
        distance = math.hypot(float(actual["lat"]) - expected["lat"], float(actual["lng"]) - expected["lon"])
        if str(actual["id"]) != station or distance > 0.01:
            raise RuntimeError(f"NOAA metadata mismatch for preregistered station {station}")
        rows.append({
            "station_id": station,
            "station_name": str(actual["name"]),
            "lat": float(actual["lat"]),
            "lon": float(actual["lng"]),
            "state": str(actual.get("state", "")),
            "tidal": bool(actual.get("tidal", False)),
            "observedst": bool(actual.get("observedst", False)),
        })
    return pd.DataFrame(rows)


def build_water_table(stations: pd.DataFrame) -> pd.DataFrame:
    hours = pd.date_range("2023-01-01", "2025-12-31 23:00:00", freq="h")
    rows = []
    for station in stations.itertuples(index=False):
        water = read_noaa_product(station.station_id, "hourly_height").rename(columns={"Water Level": "water_level"})
        tide = read_noaa_product(station.station_id, "predictions").rename(columns={"Prediction": "tide"})
        frame = pd.DataFrame({"datetime": hours}).merge(water, on="datetime", how="left").merge(tide, on="datetime", how="left")
        frame["station_id"] = station.station_id
        frame["residual"] = frame["water_level"] - frame["tide"]
        frame["sigma"] = np.nan
        frame["quality_i"] = np.nan
        frame["quality_l"] = np.nan
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)[["datetime", "station_id", "water_level", "tide", "residual", "sigma", "quality_i", "quality_l"]]


def coverage_gate(water: pd.DataFrame, stations: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = {
        "train": (pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01")),
        "validation": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-01")),
        "test": (pd.Timestamp("2025-07-01"), pd.Timestamp("2026-01-01")),
    }
    rows = []
    for station in stations["station_id"]:
        frame = water[water["station_id"] == station]
        for period, (start, end) in periods.items():
            part = frame[(frame["datetime"] >= start) & (frame["datetime"] < end)]
            rows.append({
                "station_id": station,
                "period": period,
                "hours": len(part),
                "water_coverage": float(part["water_level"].notna().mean()),
                "tide_coverage": float(part["tide"].notna().mean()),
                "residual_coverage": float(part["residual"].notna().mean()),
            })
    coverage = pd.DataFrame(rows)
    threshold = float(config["station_exclusion_rule"]["allowed_reason"].split("below ")[1].split(" ")[0])
    keep = coverage.groupby("station_id")[["water_coverage", "tide_coverage"]].min().min(axis=1) >= threshold
    retained_ids = keep[keep].index.tolist()
    retained = stations[stations["station_id"].isin(retained_ids)].copy()
    minimum = int(config["station_exclusion_rule"]["minimum_retained_stations"])
    if len(retained) < minimum:
        raise RuntimeError(f"Coverage gate failed: retained {len(retained)} stations; minimum={minimum}")
    return retained, coverage


def extract_depth(stations: pd.DataFrame) -> pd.DataFrame:
    source = next((ROOT / "data" / "raw" / "GEBCO_gebco_unzip").glob("*.nc"))
    temp = Path(tempfile.gettempdir()) / "delaware_gebco.nc"
    if not temp.exists() or temp.stat().st_size != source.stat().st_size:
        shutil.copy2(source, temp)
    ds = xr.open_dataset(temp)
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    variable = "elevation" if "elevation" in ds.data_vars else next(iter(ds.data_vars))
    elevation = ds[variable].values
    lat = ds[lat_name].values.astype(float)
    lon = ds[lon_name].values.astype(float)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    wet = elevation < 0
    rows = []
    for station in stations.itertuples(index=False):
        distance2 = (lat_grid - station.lat) ** 2 + ((lon_grid - station.lon) * np.cos(np.deg2rad(station.lat))) ** 2
        distance2 = np.where(wet, distance2, np.inf)
        iy, ix = np.unravel_index(np.argmin(distance2), distance2.shape)
        distance_km = 111.0 * math.sqrt(float(distance2[iy, ix]))
        rows.append({
            "station_id": station.station_id,
            "station_name": station.station_name,
            "station_lat": station.lat,
            "station_lon": station.lon,
            "gebco_lat": float(lat[iy]),
            "gebco_lon": float(lon[ix]),
            "elevation": float(elevation[iy, ix]),
            "depth": max(0.5, -float(elevation[iy, ix])),
            "distance_km": distance_km,
        })
    ds.close()
    return pd.DataFrame(rows)


def build_era5(stations: pd.DataFrame) -> pd.DataFrame:
    paths = sorted((ROOT / "data" / "raw" / "ERA5_2023_2025").glob("*.nc"))
    if len(paths) != 36:
        raise RuntimeError(f"Expected 36 existing ERA5 files, found {len(paths)}")
    ascii_paths = era5_impl.copy_to_ascii_temp(paths)
    datasets = [xr.open_dataset(path) for path in ascii_paths]
    time_name = "valid_time" if "valid_time" in datasets[0].coords else "time"
    ds = xr.concat(datasets, dim=time_name).sortby(time_name)
    rows = [era5_impl.nearest_grid_series(ds, station) for _, station in stations.iterrows()]
    ds.close()
    for dataset in datasets:
        dataset.close()
    return pd.concat(rows, ignore_index=True).sort_values(["datetime", "station_id"])


def met_url(station: str, product: str, begin: date, end: date) -> str:
    params = {
        "station": station,
        "begin_date": begin.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "product": product,
        "time_zone": "gmt",
        "units": "metric",
        "format": "csv",
        "application": "delaware_external_confirmation",
    }
    return NOAA_API + "?" + urlencode(params)


def download_met_job(job: tuple[str, str, date, date], cached_only: bool = False) -> tuple[dict, pd.DataFrame]:
    station, product, begin, end = job
    target_dir = RAW_DIR / "coops_met" / station
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{station}_{product}_{begin:%Y%m}.csv"
    if cached_only and (not target.exists() or target.stat().st_size == 0):
        row = {"station_id": station, "product": product, "month": f"{begin:%Y-%m}", "rows": 0, "status": "missing_after_bounded_download_attempt"}
        return row, pd.DataFrame()
    try:
        payload = target.read_bytes() if target.exists() and target.stat().st_size else fetch(met_url(station, product, begin, end), retries=2, timeout=30)
        if not target.exists():
            target.write_bytes(payload)
        parsed = coops_impl.parse_coops_csv(payload, station, product, begin.year)
        row = {"station_id": station, "product": product, "month": f"{begin:%Y-%m}", "rows": len(parsed), "status": "ok" if len(parsed) else "no_data"}
        return row, parsed
    except Exception as exc:
        row = {"station_id": station, "product": product, "month": f"{begin:%Y-%m}", "rows": 0, "status": f"error:{type(exc).__name__}"}
        return row, pd.DataFrame()


def build_local_met(stations: pd.DataFrame, cached_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    products = ["wind", "air_pressure", "air_temperature", "water_temperature"]
    inventory = []
    long_frames = []
    jobs = [(station, product, begin, end) for station in stations["station_id"] for product in products for begin, end in month_ranges(2023, 2025)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download_met_job, job, cached_only) for job in jobs]
        for future in as_completed(futures):
            row, parsed = future.result()
            inventory.append(row)
            if not parsed.empty:
                long_frames.append(parsed)
    long = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame(columns=["datetime", "station_id"])
    hours = pd.date_range("2023-01-01", "2025-12-31 23:00:00", freq="h")
    value_cols = [
        "coops_wind_speed", "coops_wind_gust", "coops_air_pressure",
        "coops_air_temperature", "coops_water_temperature",
    ]
    frames = []
    for station in stations["station_id"]:
        base = pd.DataFrame({"datetime": hours, "station_id": station})
        if not long.empty:
            part = long[long["station_id"] == station].copy()
            available = [column for column in value_cols if column in part]
            part = part[["datetime", "station_id", *available]].groupby(["datetime", "station_id"], as_index=False).mean(numeric_only=True)
            base = base.merge(part, on=["datetime", "station_id"], how="left")
        for column in value_cols:
            if column not in base:
                base[column] = np.nan
        pressure_observed = base["coops_air_pressure"].notna()
        pressure = base["coops_air_pressure"].ffill(limit=6)
        prior_mean = pressure.expanding(min_periods=1).mean().shift(1)
        base["coops_pressure_anom"] = (pressure - prior_mean).where(pressure_observed)
        base["coops_pressure_tendency_3h"] = pressure.diff(3).where(pressure_observed)
        wind_observed = base["coops_wind_speed"].notna()
        wind = base["coops_wind_speed"].ffill(limit=6)
        base["coops_wind_speed_tendency_3h"] = wind.diff(3).where(wind_observed)
        frames.append(base)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(inventory)


def write_processed(config: dict, stations: pd.DataFrame, water: pd.DataFrame, cached_local_met: bool = False) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stations.to_csv(PROCESSED_DIR / "station_order.csv", index=False)
    water = water[water["station_id"].isin(stations["station_id"])].copy()
    water.to_csv(PROCESSED_DIR / "water_tide_residual_long.csv", index=False)
    for value, name in (("water_level", "water_level_matrix.csv"), ("tide", "tide_matrix.csv"), ("residual", "residual_matrix.csv")):
        water.pivot(index="datetime", columns="station_id", values=value).reset_index().to_csv(PROCESSED_DIR / name, index=False)

    depth = extract_depth(stations)
    depth.to_csv(PROCESSED_DIR / "gebco_station_depth.csv", index=False)
    era5 = build_era5(stations)
    era5.to_csv(PROCESSED_DIR / "era5_station_hourly.csv", index=False)

    marine.STATIONS_PATH = PROCESSED_DIR / "station_order.csv"
    marine.DEPTH_PATH = PROCESSED_DIR / "gebco_station_depth.csv"
    marine.OUT_DIR = PROCESSED_DIR
    currents = marine.make_currents(stations)
    waves = marine.make_waves(stations)
    currents.to_csv(PROCESSED_DIR / "surface_currents_station_daily.csv", index=False)
    waves.to_csv(PROCESSED_DIR / "wave_direction_speed_station_3hourly.csv", index=False)

    local_met, met_inventory = build_local_met(stations, cached_only=cached_local_met)
    local_met.to_csv(PROCESSED_DIR / "noaa_coops_met_station_hourly.csv", index=False)
    met_inventory.to_csv(PROCESSED_DIR / "coops_met_download_inventory.csv", index=False)

    source_files = [
        PROCESSED_DIR / "station_order.csv", PROCESSED_DIR / "water_tide_residual_long.csv",
        PROCESSED_DIR / "era5_station_hourly.csv", PROCESSED_DIR / "surface_currents_station_daily.csv",
        PROCESSED_DIR / "wave_direction_speed_station_3hourly.csv", PROCESSED_DIR / "gebco_station_depth.csv",
        PROCESSED_DIR / "noaa_coops_met_station_hourly.csv",
    ]
    manifest = {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": sha256(CONFIG_PATH),
        "retained_station_ids": stations["station_id"].tolist(),
        "files": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in source_files],
    }
    (PROCESSED_DIR / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cached-local-met", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stations = verify_metadata(config)
    stations.to_csv(PROCESSED_DIR / "official_station_metadata_snapshot.csv", index=False)

    jobs = [(station, product, begin, end) for station in stations["station_id"] for product in ("hourly_height", "predictions") for begin, end in month_ranges(2023, 2025)]
    inventory = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_noaa_job, job) for job in jobs]
        for future in as_completed(futures):
            try:
                inventory.append(future.result())
            except Exception as exc:
                failures.append(str(exc))
    pd.DataFrame(inventory).to_csv(PROCESSED_DIR / "noaa_target_download_inventory.csv", index=False)
    if failures:
        (PROCESSED_DIR / "download_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"NOAA target download had {len(failures)} failures")

    water = build_water_table(stations)
    retained, coverage = coverage_gate(water, stations, config)
    coverage.to_csv(PROCESSED_DIR / "target_coverage_by_split.csv", index=False)
    excluded = stations[~stations["station_id"].isin(retained["station_id"])]
    excluded.to_csv(PROCESSED_DIR / "stations_excluded_by_frozen_coverage_rule.csv", index=False)
    write_processed(config, retained, water, cached_local_met=args.cached_local_met)
    print(coverage.to_string(index=False))
    print(f"Prepared {len(retained)} retained stations under {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
