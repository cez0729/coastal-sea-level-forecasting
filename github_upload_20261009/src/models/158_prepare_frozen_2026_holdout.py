from __future__ import annotations

"""Download and assemble the frozen 2026 H1 chronological holdout inputs."""

import argparse
import importlib.util
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
START = pd.Timestamp("2026-01-01 00:00:00")
END = pd.Timestamp("2026-06-30 23:00:00")
STATIONS = ["8461490", "8510560", "8516945", "8518750", "8531680", "8534720", "8536110"]
RAW = ROOT / "data" / "raw" / "frozen_holdout_2026_h1"
SEGMENT = ROOT / "data" / "processed_2026_h1_frozen"
EXTENDED = ROOT / "data" / "processed_multiyear_2023_2026_h1_frozen"
COPERNICUS = ROOT / ".venv" / "Scripts" / "copernicusmarine.exe"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


noaa = load_module("frozen_noaa", HERE / "67_download_noaa_multiyear_raw.py")
era = load_module("frozen_era", HERE / "70_download_era5_multiyear.py")
marine = load_module("frozen_marine", HERE / "71_make_copernicus_multiyear_station_features.py")
local = load_module("frozen_local", HERE / "75_download_noaa_ndbc_enhanced_forcing.py")


def month_ranges():
    for month in range(1, 7):
        begin = pd.Timestamp(2026, month, 1)
        end = begin + pd.offsets.MonthEnd(1)
        yield begin, end


def download_noaa_water_tide(skip_existing: bool) -> None:
    for station in STATIONS:
        for product, suffix in (("hourly_height", "water"), ("predictions", "tide")):
            path = RAW / "noaa" / f"{station}_{suffix}_2026_h1.csv"
            if skip_existing and path.exists() and path.stat().st_size > 0:
                continue
            chunks = []
            for begin, end in month_ranges():
                params = {
                    "station": station,
                    "begin_date": begin.strftime("%Y%m%d"),
                    "end_date": end.strftime("%Y%m%d"),
                    "product": product,
                    "datum": "MLLW",
                    "time_zone": "gmt",
                    "units": "metric",
                    "format": "csv",
                    "application": "frozen_2026_holdout",
                }
                if product == "predictions":
                    params["interval"] = "h"
                chunks.append(noaa.request_csv(params))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(noaa.strip_duplicate_header(chunks), encoding="utf-8", newline="\n")


def process_noaa_water_tide() -> None:
    frames = []
    for station in STATIONS:
        water = pd.read_csv(RAW / "noaa" / f"{station}_water_2026_h1.csv")
        tide = pd.read_csv(RAW / "noaa" / f"{station}_tide_2026_h1.csv")
        water.columns = [str(col).strip() for col in water.columns]
        tide.columns = [str(col).strip() for col in tide.columns]
        water["datetime"] = pd.to_datetime(water["Date Time"])
        tide["datetime"] = pd.to_datetime(tide["Date Time"])
        water = water.rename(columns={"Water Level": "water_level", "Sigma": "sigma", "I": "quality_i", "L": "quality_l"})
        tide = tide.rename(columns={"Prediction": "tide"})
        frame = water[["datetime", "water_level", "sigma", "quality_i", "quality_l"]].merge(
            tide[["datetime", "tide"]], on="datetime", how="inner"
        )
        for col in ("water_level", "sigma", "quality_i", "quality_l", "tide"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["station_id"] = station
        frame["residual"] = frame["water_level"] - frame["tide"]
        frames.append(frame[["datetime", "station_id", "water_level", "tide", "residual", "sigma", "quality_i", "quality_l"]])
    SEGMENT.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).sort_values(["datetime", "station_id"]).to_csv(
        SEGMENT / "water_tide_residual_long.csv", index=False
    )


def download_local_met(skip_existing: bool) -> None:
    frames = []
    inventory = []
    for station in STATIONS:
        for product in local.COOPS_PRODUCTS:
            path = RAW / "coops" / f"{station}_{product}_2026_h1.csv"
            url = (
                "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
                f"begin_date=20260101&end_date=20260630&station={station}&product={product}"
                "&time_zone=gmt&interval=h&units=metric&format=csv&application=frozen_2026_holdout"
            )
            try:
                if not (skip_existing and path.exists() and path.stat().st_size > 0):
                    raw = local.fetch_bytes(url, timeout=90, tries=4)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                raw = path.read_bytes()
                parsed = local.parse_coops_csv(raw, station, product, 2026)
                if not parsed.empty:
                    frames.append(parsed)
                inventory.append({"station_id": station, "product": product, "rows": len(parsed), "status": "ok" if len(parsed) else "no_data", "url": url})
            except Exception as exc:
                inventory.append({"station_id": station, "product": product, "rows": 0, "status": type(exc).__name__, "url": url})
    long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["datetime", "station_id"])
    hours = pd.date_range(START, END, freq="h")
    value_cols = [
        "coops_wind_speed", "coops_wind_gust", "coops_wind_direction_deg", "coops_wind_u", "coops_wind_v",
        "coops_air_pressure", "coops_air_temperature", "coops_water_temperature",
    ]
    station_frames = []
    for station in STATIONS:
        base = pd.DataFrame({"datetime": hours, "station_id": station})
        sub = long[long["station_id"] == station].copy()
        keep = ["datetime", "station_id"] + [col for col in value_cols if col in sub.columns]
        if len(keep) > 2:
            sub = sub[keep].groupby(["datetime", "station_id"], as_index=False).mean(numeric_only=True)
            base = base.merge(sub, on=["datetime", "station_id"], how="left")
        station_frames.append(base)
    out = pd.concat(station_frames, ignore_index=True).sort_values(["station_id", "datetime"])
    numeric = [col for col in out.columns if col not in {"datetime", "station_id"}]
    if numeric:
        out[numeric] = out.groupby("station_id", group_keys=False)[numeric].apply(lambda frame: frame.interpolate(limit=6, limit_direction="both"))
    historical = pd.read_csv(ROOT / "data" / "processed_multiyear_2023_2025" / "noaa_coops_met_station_hourly.csv")
    pressure_reference = float(historical["coops_air_pressure"].mean(skipna=True))
    if "coops_air_pressure" in out:
        out["coops_pressure_anom"] = out["coops_air_pressure"] - pressure_reference
        out["coops_pressure_tendency_3h"] = out.groupby("station_id")["coops_air_pressure"].diff(3)
    if "coops_wind_speed" in out:
        out["coops_wind_speed_tendency_3h"] = out.groupby("station_id")["coops_wind_speed"].diff(3)
    out.to_csv(SEGMENT / "noaa_coops_met_station_hourly.csv", index=False)
    pd.DataFrame(inventory).to_csv(SEGMENT / "noaa_coops_download_inventory.csv", index=False)


def download_era5(skip_existing: bool) -> None:
    out = RAW / "era5"
    for month in range(1, 7):
        target = out / f"era5_single_levels_u10_v10_msl_2026{month:02d}.nc"
        if skip_existing and target.exists() and target.stat().st_size > 0:
            continue
        era.download_month(2026, month, out)


def process_era5() -> None:
    paths = sorted((RAW / "era5").glob("era5_single_levels_u10_v10_msl_2026*.nc"))
    if len(paths) != 6:
        raise FileNotFoundError(f"Expected six ERA5 files, found {len(paths)}")
    temp_dir = Path(tempfile.gettempdir()) / "sea_level_frozen_2026_era5"
    temp_dir.mkdir(parents=True, exist_ok=True)
    ascii_paths = []
    for path in paths:
        target = temp_dir / path.name
        if not target.exists() or target.stat().st_size != path.stat().st_size:
            shutil.copy2(path, target)
        ascii_paths.append(target)
    datasets = [xr.open_dataset(path) for path in ascii_paths]
    time_name = "valid_time" if "valid_time" in datasets[0].coords else "time"
    ds = xr.concat(datasets, dim=time_name).sortby(time_name)
    stations = pd.read_csv(ROOT / "data" / "processed" / "stations.csv")
    rows = []
    for station in stations.itertuples(index=False):
        point = ds.sel(latitude=float(station.lat), longitude=float(station.lon), method="nearest")
        frame = point[["u10", "v10", "msl"]].to_dataframe().reset_index().rename(columns={time_name: "datetime"})
        frame["station_id"] = str(station.station_id)
        frame["station_name"] = station.station_name
        frame["station_lat"], frame["station_lon"] = station.lat, station.lon
        frame["era5_lat"], frame["era5_lon"] = float(point.latitude.values), float(point.longitude.values)
        frame["wind_speed"] = np.sqrt(frame["u10"] ** 2 + frame["v10"] ** 2)
        frame["wind_stress_u_proxy"] = frame["u10"] * frame["wind_speed"]
        frame["wind_stress_v_proxy"] = frame["v10"] * frame["wind_speed"]
        rows.append(frame)
    pd.concat(rows, ignore_index=True).sort_values(["datetime", "station_id"]).to_csv(SEGMENT / "era5_station_hourly.csv", index=False)
    ds.close()
    for dataset in datasets:
        dataset.close()


def copernicus_subset(dataset: str, variables: list[str], target: Path, skip_existing: bool) -> None:
    if skip_existing and target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(COPERNICUS), "subset", "-i", dataset,
        "-x", "-76", "-X", "-70", "-y", "38", "-Y", "42.5",
        "-t", "2026-01-01", "-T", "2026-06-30T23:59:59",
        "-o", str(target.parent), "-f", target.name,
        "--file-format", "netcdf", "--coordinates-selection-method", "nearest",
        "--overwrite", "--disable-progress-bar",
    ]
    for variable in variables:
        command.extend(["-v", variable])
    if "phy-cur" in dataset:
        command.extend(["-z", "0", "-Z", "1"])
    subprocess.run(command, check=True)


def download_and_process_marine(skip_existing: bool) -> None:
    current = RAW / "copernicus" / "currents_2026_h1.nc"
    wave = RAW / "copernicus" / "waves_2026_h1.nc"
    copernicus_subset("cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", ["uo", "vo"], current, skip_existing)
    copernicus_subset(
        "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
        ["VHM0", "VTPK", "VTM10", "VMDR", "VSDX", "VSDY", "VPED"],
        wave,
        skip_existing,
    )
    marine.CURRENT_PATH = current
    marine.WAVE_PATH = wave
    currents = marine.make_currents(pd.read_csv(marine.STATIONS_PATH))
    waves = marine.make_waves(pd.read_csv(marine.STATIONS_PATH))
    currents.to_csv(SEGMENT / "surface_currents_station_daily.csv", index=False)
    waves.to_csv(SEGMENT / "wave_direction_speed_station_3hourly.csv", index=False)


def append_table(name: str) -> None:
    old = pd.read_csv(ROOT / "data" / "processed_multiyear_2023_2025" / name)
    new = pd.read_csv(SEGMENT / name)
    old["datetime"] = pd.to_datetime(old["datetime"])
    new["datetime"] = pd.to_datetime(new["datetime"])
    old["station_id"], new["station_id"] = old["station_id"].astype(str), new["station_id"].astype(str)
    columns = list(dict.fromkeys([*old.columns, *new.columns]))
    combined = pd.concat([old.reindex(columns=columns), new.reindex(columns=columns)], ignore_index=True)
    combined = combined.drop_duplicates(["datetime", "station_id"], keep="last")
    combined = combined[combined["datetime"] <= END].sort_values(["datetime", "station_id"])
    EXTENDED.mkdir(parents=True, exist_ok=True)
    combined.to_csv(EXTENDED / name, index=False)


def assemble() -> None:
    names = [
        "water_tide_residual_long.csv", "era5_station_hourly.csv", "surface_currents_station_daily.csv",
        "wave_direction_speed_station_3hourly.csv", "noaa_coops_met_station_hourly.csv",
    ]
    for name in names:
        append_table(name)
    inventory = []
    for path in sorted(EXTENDED.glob("*.csv")):
        frame = pd.read_csv(path, usecols=lambda col: col in {"datetime", "station_id"})
        inventory.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "rows": len(frame),
                "start": str(pd.to_datetime(frame["datetime"]).min()),
                "end": str(pd.to_datetime(frame["datetime"]).max()),
                "stations": int(frame["station_id"].astype(str).nunique()),
            }
        )
    pd.DataFrame(inventory).to_csv(EXTENDED / "HOLDOUT_DATA_INVENTORY.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the frozen 2026 H1 holdout data")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-downloads", action="store_true")
    args = parser.parse_args()
    SEGMENT.mkdir(parents=True, exist_ok=True)
    if not args.skip_downloads:
        download_noaa_water_tide(args.skip_existing)
        download_local_met(args.skip_existing)
        download_era5(args.skip_existing)
        download_and_process_marine(args.skip_existing)
    process_noaa_water_tide()
    process_era5()
    if not (SEGMENT / "surface_currents_station_daily.csv").exists():
        download_and_process_marine(True)
    assemble()
    (EXTENDED / "HOLDOUT_BOUNDARY.json").write_text(
        json.dumps({"start": str(START), "end": str(END), "model_frozen_before_download": True}, indent=2),
        encoding="utf-8",
    )
    print(pd.read_csv(EXTENDED / "HOLDOUT_DATA_INVENTORY.csv").to_string(index=False))


if __name__ == "__main__":
    main()
