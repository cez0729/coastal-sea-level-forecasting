from __future__ import annotations

"""Download and assemble the frozen July 2026 chronological holdout inputs."""

import argparse
import importlib.util
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
START = pd.Timestamp("2026-07-01 00:00:00")
END = pd.Timestamp("2026-07-31 23:00:00")
STATIONS = ["8461490", "8510560", "8516945", "8518750", "8531680", "8534720", "8536110"]
BASE = ROOT / "data" / "processed_multiyear_2023_2026_h1_frozen"
RAW = ROOT / "data" / "raw" / "frozen_holdout_2026_july"
SEGMENT = ROOT / "data" / "processed_2026_july_frozen"
EXTENDED = ROOT / "data" / "processed_multiyear_2023_2026_july_frozen"
COPERNICUS = ROOT / ".venv" / "Scripts" / "copernicusmarine.exe"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


noaa = load_module("july_noaa", HERE / "67_download_noaa_multiyear_raw.py")
era = load_module("july_era", HERE / "70_download_era5_multiyear.py")
marine = load_module("july_marine", HERE / "71_make_copernicus_multiyear_station_features.py")
local = load_module("july_local", HERE / "75_download_noaa_ndbc_enhanced_forcing.py")


def csv_has_api_error(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return True
    return any(line.lstrip().startswith("Error:") for line in path.read_text(encoding="utf-8-sig").splitlines())


def download_noaa_water_tide(skip_existing: bool) -> None:
    for station in STATIONS:
        for product, suffix in (("hourly_height", "water"), ("predictions", "tide")):
            path = RAW / "noaa" / f"{station}_{suffix}_2026_july.csv"
            params = {
                "station": station,
                "begin_date": "20260701",
                "end_date": "20260731",
                "product": product,
                "datum": "MLLW",
                "time_zone": "gmt",
                "units": "metric",
                "format": "csv",
                "application": "frozen_2026_july_holdout",
            }
            if product == "predictions":
                params["interval"] = "h"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not (skip_existing and path.exists() and path.stat().st_size > 0):
                path.write_text(noaa.strip_duplicate_header([noaa.request_csv(params)]), encoding="utf-8", newline="\n")
            if product == "hourly_height" and csv_has_api_error(path):
                fallback = RAW / "noaa" / f"{station}_water_6min_2026_july.csv"
                fallback_params = dict(params)
                fallback_params["product"] = "water_level"
                if not (skip_existing and fallback.exists() and fallback.stat().st_size > 0 and not csv_has_api_error(fallback)):
                    fallback.write_text(
                        noaa.strip_duplicate_header([noaa.request_csv(fallback_params)]),
                        encoding="utf-8",
                        newline="\n",
                    )
                if csv_has_api_error(fallback):
                    raise RuntimeError(f"NOAA returned no hourly or 6-minute water level for station {station}")


def process_noaa_water_tide() -> None:
    frames = []
    for station in STATIONS:
        hourly_path = RAW / "noaa" / f"{station}_water_2026_july.csv"
        water_source = hourly_path if not csv_has_api_error(hourly_path) else RAW / "noaa" / f"{station}_water_6min_2026_july.csv"
        water = pd.read_csv(water_source)
        tide = pd.read_csv(RAW / "noaa" / f"{station}_tide_2026_july.csv")
        water.columns = [str(col).strip() for col in water.columns]
        tide.columns = [str(col).strip() for col in tide.columns]
        water["datetime"] = pd.to_datetime(water["Date Time"])
        tide["datetime"] = pd.to_datetime(tide["Date Time"])
        water = water.rename(columns={"Water Level": "water_level", "Sigma": "sigma", "I": "quality_i", "L": "quality_l"})
        tide = tide.rename(columns={"Prediction": "tide"})
        for col in ("water_level", "sigma", "quality_i", "quality_l"):
            if col not in water:
                water[col] = np.nan
            water[col] = pd.to_numeric(water[col], errors="coerce")
        if water["datetime"].duplicated().any() or len(water) > 24 * 32:
            water["datetime"] = water["datetime"].dt.floor("h")
            water = water.groupby("datetime", as_index=False)[["water_level", "sigma", "quality_i", "quality_l"]].mean()
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
    frames, inventory = [], []
    for station in STATIONS:
        for product in local.COOPS_PRODUCTS:
            path = RAW / "coops" / f"{station}_{product}_2026_july.csv"
            url = (
                "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
                f"begin_date=20260701&end_date=20260731&station={station}&product={product}"
                "&time_zone=gmt&interval=h&units=metric&format=csv&application=frozen_2026_july_holdout"
            )
            try:
                if not (skip_existing and path.exists() and path.stat().st_size > 0):
                    raw = local.fetch_bytes(url, timeout=90, tries=4)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                parsed = local.parse_coops_csv(path.read_bytes(), station, product, 2026)
                if not parsed.empty:
                    frames.append(parsed)
                inventory.append({"station_id": station, "product": product, "rows": len(parsed), "status": "ok" if len(parsed) else "no_data", "url": url})
            except Exception as exc:
                inventory.append({"station_id": station, "product": product, "rows": 0, "status": type(exc).__name__, "url": url})
    long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["datetime", "station_id"])
    station_frames = []
    value_cols = [
        "coops_wind_speed", "coops_wind_gust", "coops_wind_direction_deg", "coops_wind_u", "coops_wind_v",
        "coops_air_pressure", "coops_air_temperature", "coops_water_temperature",
    ]
    for station in STATIONS:
        base = pd.DataFrame({"datetime": pd.date_range(START, END, freq="h"), "station_id": station})
        sub = long[long["station_id"] == station].copy()
        keep = ["datetime", "station_id"] + [col for col in value_cols if col in sub.columns]
        if len(keep) > 2:
            sub = sub[keep].groupby(["datetime", "station_id"], as_index=False).mean(numeric_only=True)
            base = base.merge(sub, on=["datetime", "station_id"], how="left")
        station_frames.append(base)
    out = pd.concat(station_frames, ignore_index=True).sort_values(["station_id", "datetime"])
    numeric = [col for col in out.columns if col not in {"datetime", "station_id"}]
    if numeric:
        out[numeric] = out.groupby("station_id", group_keys=False)[numeric].apply(
            lambda frame: frame.interpolate(limit=6, limit_direction="both")
        )
    historical = pd.read_csv(BASE / "noaa_coops_met_station_hourly.csv")
    if "coops_air_pressure" in out:
        out["coops_pressure_anom"] = out["coops_air_pressure"] - float(historical["coops_air_pressure"].mean(skipna=True))
        out["coops_pressure_tendency_3h"] = out.groupby("station_id")["coops_air_pressure"].diff(3)
    if "coops_wind_speed" in out:
        out["coops_wind_speed_tendency_3h"] = out.groupby("station_id")["coops_wind_speed"].diff(3)
    out.to_csv(SEGMENT / "noaa_coops_met_station_hourly.csv", index=False)
    pd.DataFrame(inventory).to_csv(SEGMENT / "noaa_coops_download_inventory.csv", index=False)


def download_era5(skip_existing: bool) -> None:
    target = RAW / "era5" / "era5_single_levels_u10_v10_msl_202607.nc"
    if not (skip_existing and target.exists() and target.stat().st_size > 0):
        era.download_month(2026, 7, target.parent)


def process_era5() -> None:
    source = RAW / "era5" / "era5_single_levels_u10_v10_msl_202607.nc"
    if not source.exists():
        raise FileNotFoundError(source)
    ascii_path = Path(tempfile.gettempdir()) / "sea_level_frozen_2026_july_era5.nc"
    if not ascii_path.exists() or ascii_path.stat().st_size != source.stat().st_size:
        shutil.copy2(source, ascii_path)
    ds = xr.open_dataset(ascii_path)
    time_name = "valid_time" if "valid_time" in ds.coords else "time"
    stations = pd.read_csv(ROOT / "data" / "processed" / "stations.csv")
    rows = []
    for station in stations.itertuples(index=False):
        point = ds.sel(latitude=float(station.lat), longitude=float(station.lon), method="nearest")
        frame = point[["u10", "v10", "msl"]].to_dataframe().reset_index().rename(columns={time_name: "datetime"})
        frame["station_id"], frame["station_name"] = str(station.station_id), station.station_name
        frame["station_lat"], frame["station_lon"] = station.lat, station.lon
        frame["era5_lat"], frame["era5_lon"] = float(point.latitude.values), float(point.longitude.values)
        frame["wind_speed"] = np.sqrt(frame["u10"] ** 2 + frame["v10"] ** 2)
        frame["wind_stress_u_proxy"] = frame["u10"] * frame["wind_speed"]
        frame["wind_stress_v_proxy"] = frame["v10"] * frame["wind_speed"]
        rows.append(frame)
    pd.concat(rows, ignore_index=True).sort_values(["datetime", "station_id"]).to_csv(SEGMENT / "era5_station_hourly.csv", index=False)
    ds.close()


def copernicus_subset(dataset: str, variables: list[str], target: Path, skip_existing: bool) -> None:
    if skip_existing and target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(COPERNICUS), "subset", "-i", dataset,
        "-x", "-76", "-X", "-70", "-y", "38", "-Y", "42.5",
        "-t", "2026-07-01", "-T", "2026-07-31T23:59:59",
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
    current = RAW / "copernicus" / "currents_2026_july.nc"
    wave = RAW / "copernicus" / "waves_2026_july.nc"
    copernicus_subset("cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", ["uo", "vo"], current, skip_existing)
    copernicus_subset(
        "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
        ["VHM0", "VTPK", "VTM10", "VMDR", "VSDX", "VSDY", "VPED"], wave, skip_existing,
    )
    ascii_dir = Path(tempfile.gettempdir()) / "sea_level_frozen_2026_july_marine"
    ascii_dir.mkdir(parents=True, exist_ok=True)
    ascii_current, ascii_wave = ascii_dir / current.name, ascii_dir / wave.name
    shutil.copy2(current, ascii_current)
    shutil.copy2(wave, ascii_wave)
    marine.CURRENT_PATH, marine.WAVE_PATH = ascii_current, ascii_wave
    stations = pd.read_csv(marine.STATIONS_PATH)
    marine.make_currents(stations).to_csv(SEGMENT / "surface_currents_station_daily.csv", index=False)
    marine.make_waves(stations).to_csv(SEGMENT / "wave_direction_speed_station_3hourly.csv", index=False)


def append_table(name: str) -> None:
    old, new = pd.read_csv(BASE / name), pd.read_csv(SEGMENT / name)
    old["datetime"], new["datetime"] = pd.to_datetime(old["datetime"]), pd.to_datetime(new["datetime"])
    old["station_id"], new["station_id"] = old["station_id"].astype(str), new["station_id"].astype(str)
    columns = list(dict.fromkeys([*old.columns, *new.columns]))
    combined = pd.concat([old.reindex(columns=columns), new.reindex(columns=columns)], ignore_index=True)
    combined = combined.drop_duplicates(["datetime", "station_id"], keep="last")
    combined = combined[combined["datetime"] <= END].sort_values(["datetime", "station_id"])
    EXTENDED.mkdir(parents=True, exist_ok=True)
    combined.to_csv(EXTENDED / name, index=False)


def assemble() -> None:
    for name in (
        "water_tide_residual_long.csv", "era5_station_hourly.csv", "surface_currents_station_daily.csv",
        "wave_direction_speed_station_3hourly.csv", "noaa_coops_met_station_hourly.csv",
    ):
        append_table(name)
    inventory = []
    for path in sorted(EXTENDED.glob("*.csv")):
        frame = pd.read_csv(path, usecols=lambda col: col in {"datetime", "station_id"})
        inventory.append(
            {
                "file": path.name, "bytes": path.stat().st_size, "rows": len(frame),
                "start": str(pd.to_datetime(frame["datetime"]).min()),
                "end": str(pd.to_datetime(frame["datetime"]).max()),
                "stations": int(frame["station_id"].astype(str).nunique()),
            }
        )
    pd.DataFrame(inventory).to_csv(EXTENDED / "HOLDOUT_DATA_INVENTORY.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the frozen July 2026 holdout data")
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
        json.dumps(
            {
                "start": str(START), "end": str(END),
                "gate_frozen_before_download": True,
                "h1_used_for_gate_development": True,
                "target_residual_imputation_for_scoring": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.read_csv(EXTENDED / "HOLDOUT_DATA_INVENTORY.csv").to_string(index=False))


if __name__ == "__main__":
    main()
