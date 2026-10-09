from __future__ import annotations

import argparse
import calendar
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "neuralcora_surge"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
MANIFEST_ROOT = DATA_ROOT / "manifests"

STATIONS = {
    "8461490": "New London",
    "8510560": "Montauk",
    "8516945": "Kings Point",
    "8518750": "The Battery",
    "8531680": "Sandy Hook",
    "8534720": "Atlantic City",
    "8536110": "Cape May",
}


def read_noaa_csvs(paths: list[Path], value_column: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, skipinitialspace=True)
        frame.columns = [column.strip() for column in frame.columns]
        if "Date Time" not in frame.columns or value_column not in frame.columns:
            raise ValueError(f"Unexpected NOAA columns in {path}: {list(frame.columns)}")
        keep = ["Date Time", value_column]
        for optional in ("Sigma", "I", "L"):
            if optional in frame.columns:
                keep.append(optional)
        frame = frame[keep].copy()
        frame["time"] = pd.to_datetime(frame.pop("Date Time"), utc=True, errors="coerce")
        frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["time", value_column])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["time"]).sort_values("time")
    combined = combined.drop_duplicates("time", keep="last")
    return combined


def expected_hours(year: int) -> int:
    return (366 if calendar.isleap(year) else 365) * 24


def build_station_data(start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_index = pd.date_range(
        f"{start_year}-01-01T00:00:00Z",
        f"{end_year}-12-31T23:00:00Z",
        freq="h",
    )
    station_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    for station_id, station_name in STATIONS.items():
        water_paths = sorted((RAW_ROOT / "noaa" / "hourly_height" / station_id).glob("*.csv"))
        prediction_dir = RAW_ROOT / "noaa" / "tide_predictions" / station_id
        prediction_paths = [prediction_dir / f"{station_id}_predictions_{year}.csv" for year in range(start_year, end_year + 1)]
        prediction_paths = [path for path in prediction_paths if path.exists()]

        water = read_noaa_csvs(water_paths, "Water Level").rename(columns={"Water Level": "observed_msl_m"})
        tide = read_noaa_csvs(prediction_paths, "Prediction").rename(columns={"Prediction": "tide_msl_m"})
        merged = pd.DataFrame({"time": full_index}).merge(water, on="time", how="left")
        merged = merged.merge(tide[["time", "tide_msl_m"]], on="time", how="left")
        merged.insert(0, "station_name", station_name)
        merged.insert(0, "station_id", station_id)
        merged["residual_m"] = merged["observed_msl_m"] - merged["tide_msl_m"]
        station_frames.append(merged)

        for year in range(start_year, end_year + 1):
            group = merged.loc[merged["time"].dt.year == year]
            expected = expected_hours(year)
            observed_count = int(group["observed_msl_m"].notna().sum())
            tide_count = int(group["tide_msl_m"].notna().sum())
            residual_count = int(group["residual_m"].notna().sum())
            audit_rows.append(
                {
                    "station_id": station_id,
                    "station_name": station_name,
                    "year": year,
                    "expected_hours": expected,
                    "observed_hours": observed_count,
                    "tide_hours": tide_count,
                    "residual_hours": residual_count,
                    "observed_coverage": observed_count / expected,
                    "tide_coverage": tide_count / expected,
                    "residual_coverage": residual_count / expected,
                }
            )

    long_data = pd.concat(station_frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    return long_data, audit


def write_processed_tables(long_data: pd.DataFrame, audit: pd.DataFrame) -> None:
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    long_path = PROCESSED_ROOT / "noaa_station_water_tide_residual_1999_2022.csv.gz"
    long_data.to_csv(long_path, index=False, compression="gzip", date_format="%Y-%m-%dT%H:%M:%SZ")
    audit.to_csv(PROCESSED_ROOT / "noaa_station_year_coverage_1999_2022.csv", index=False)

    for value, filename in (
        ("observed_msl_m", "observed_matrix_1999_2022.csv.gz"),
        ("tide_msl_m", "tide_matrix_1999_2022.csv.gz"),
        ("residual_m", "residual_matrix_1999_2022.csv.gz"),
    ):
        matrix = long_data.pivot(index="time", columns="station_name", values=value)
        matrix.to_csv(PROCESSED_ROOT / filename, compression="gzip", date_format="%Y-%m-%dT%H:%M:%SZ")


def parse_coordinate(value: str) -> float:
    value = value.strip()
    number = float(value[:-1])
    return -number if value[-1] in {"S", "W"} else number


def parse_hurdat(start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = sorted((RAW_ROOT / "events").glob("hurdat2-1851-*.txt"))
    if not candidates:
        raise FileNotFoundError("HURDAT2 file not found; run script 109 stage events")
    source = candidates[-1]
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    point_rows: list[dict[str, object]] = []
    current_id = ""
    current_name = ""
    remaining = 0
    for line in lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3 and re_is_storm_header(fields[0]):
            current_id = fields[0]
            current_name = fields[1]
            remaining = int(fields[2])
            continue
        if remaining <= 0 or len(fields) < 8:
            continue
        remaining -= 1
        timestamp = pd.to_datetime(fields[0] + fields[1], format="%Y%m%d%H%M", utc=True)
        if timestamp.year < start_year or timestamp.year > end_year:
            continue
        lat = parse_coordinate(fields[4])
        lon = parse_coordinate(fields[5])
        point_rows.append(
            {
                "storm_id": current_id,
                "storm_name": current_name,
                "time": timestamp,
                "record_identifier": fields[2],
                "status": fields[3],
                "lat": lat,
                "lon": lon,
                "max_wind_kt": pd.to_numeric(fields[6], errors="coerce"),
                "min_pressure_mb": pd.to_numeric(fields[7], errors="coerce"),
            }
        )

    points = pd.DataFrame(point_rows)
    if points.empty:
        return points, pd.DataFrame()
    # Broad Northeast corridor; final event inclusion must be checked against water-level response.
    corridor = points.loc[
        points["lat"].between(35.0, 45.0) & points["lon"].between(-80.0, -65.0)
    ].copy()
    summaries = (
        corridor.groupby(["storm_id", "storm_name"], as_index=False)
        .agg(
            start=("time", "min"),
            end=("time", "max"),
            track_points=("time", "size"),
            max_wind_kt=("max_wind_kt", "max"),
            min_pressure_mb=("min_pressure_mb", "min"),
            min_lat=("lat", "min"),
            max_lat=("lat", "max"),
            min_lon=("lon", "min"),
            max_lon=("lon", "max"),
        )
        .sort_values("start")
    )
    return points, summaries


def re_is_storm_header(value: str) -> bool:
    return len(value) == 8 and value[:2].isalpha() and value[2:].isdigit()


def validate_netcdf() -> dict[str, object]:
    report: dict[str, object] = {}
    era5_files = sorted((RAW_ROOT / "era5").glob("*.nc"))
    if era5_files:
        with xr.open_dataset(era5_files[0], engine="h5netcdf") as ds:
            report["era5_sample"] = {
                "path": str(era5_files[0].relative_to(DATA_ROOT)),
                "sizes": dict(ds.sizes),
                "variables": list(ds.data_vars),
                "time_start": str(ds.valid_time.values[0] if "valid_time" in ds else ds.time.values[0]),
                "time_end": str(ds.valid_time.values[-1] if "valid_time" in ds else ds.time.values[-1]),
            }
    cora_files = sorted((RAW_ROOT / "cora" / "events").glob("**/cora_500m_*.nc"))
    if cora_files:
        with xr.open_dataset(cora_files[0], engine="h5netcdf") as ds:
            report["cora_sample"] = {
                "path": str(cora_files[0].relative_to(DATA_ROOT)),
                "sizes": dict(ds.sizes),
                "variables": list(ds.data_vars),
                "time_start": str(ds.time.values[0]),
                "time_end": str(ds.time.values[-1]),
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aligned NeuralCORA-Surge station data")
    parser.add_argument("--start-year", type=int, default=1999)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        audit = pd.read_csv(PROCESSED_ROOT / "noaa_station_year_coverage_1999_2022.csv")
        event_summary = pd.read_csv(PROCESSED_ROOT / "hurdat2_northeast_candidate_events_1999_2022.csv")
        row_count = int(audit["expected_hours"].sum())
        observed_coverage = float(audit["observed_hours"].sum() / row_count)
        tide_coverage = float(audit["tide_hours"].sum() / row_count)
        residual_coverage = float(audit["residual_hours"].sum() / row_count)
    else:
        long_data, audit = build_station_data(args.start_year, args.end_year)
        write_processed_tables(long_data, audit)
        points, event_summary = parse_hurdat(args.start_year, args.end_year)
        points.to_csv(PROCESSED_ROOT / "hurdat2_atlantic_track_points_1999_2022.csv.gz", index=False, compression="gzip")
        event_summary.to_csv(PROCESSED_ROOT / "hurdat2_northeast_candidate_events_1999_2022.csv", index=False)
        row_count = int(len(long_data))
        observed_coverage = float(long_data["observed_msl_m"].notna().mean())
        tide_coverage = float(long_data["tide_msl_m"].notna().mean())
        residual_coverage = float(long_data["residual_m"].notna().mean())

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "period": [args.start_year, args.end_year],
        "rows": row_count,
        "stations": len(STATIONS),
        "overall_observed_coverage": observed_coverage,
        "overall_tide_coverage": tide_coverage,
        "overall_residual_coverage": residual_coverage,
        "northeast_tropical_candidates": int(len(event_summary)),
        "netcdf_samples": validate_netcdf(),
    }
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    target = MANIFEST_ROOT / "data_build_report.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
