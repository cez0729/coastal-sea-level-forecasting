from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "neuralcora_surge"
PROCESSED_ROOT = DATA_ROOT / "processed"
MANIFEST_ROOT = DATA_ROOT / "manifests"

STATION_COORDS = {
    "New London": (41.371666, -72.09556),
    "Montauk": (41.048332, -71.95944),
    "Kings Point": (40.8103, -73.7649),
    "The Battery": (40.700554, -74.01417),
    "Sandy Hook": (40.4669, -74.0094),
    "Atlantic City": (39.356667, -74.41805),
    "Cape May": (38.9683, -74.96),
}


def split_name(timestamp: pd.Timestamp) -> str:
    if timestamp <= pd.Timestamp("2016-12-31T23:00:00Z"):
        return "train"
    if timestamp <= pd.Timestamp("2019-12-31T23:00:00Z"):
        return "validation"
    return "test"


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1_r = np.deg2rad(lat1)
    lat2_r = np.deg2rad(lat2)
    dlat = lat2_r - lat1_r
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def load_hurdat_near_stations() -> pd.DataFrame:
    path = PROCESSED_ROOT / "hurdat2_atlantic_track_points_1999_2022.csv.gz"
    tracks = pd.read_csv(path, parse_dates=["time"])
    minimum = np.full(len(tracks), np.inf)
    for lat, lon in STATION_COORDS.values():
        minimum = np.minimum(minimum, haversine_km(tracks["lat"].to_numpy(), tracks["lon"].to_numpy(), lat, lon))
    tracks["nearest_station_distance_km"] = minimum
    return tracks.loc[tracks["nearest_station_distance_km"] <= 500].copy()


def merge_active_hours(active: pd.Series, max_gap_hours: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    times = active.index[active.fillna(False)]
    if len(times) == 0:
        return []
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = times[0]
    previous = times[0]
    for timestamp in times[1:]:
        if (timestamp - previous) > pd.Timedelta(hours=max_gap_hours):
            intervals.append((start, previous))
            start = timestamp
        previous = timestamp
    intervals.append((start, previous))
    return intervals


def match_hurdat(
    tracks: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[str, str, float | None]:
    nearby = tracks.loc[
        tracks["time"].between(start - pd.Timedelta(days=3), end + pd.Timedelta(days=3))
    ]
    if nearby.empty:
        return "", "", None
    best = nearby.sort_values("nearest_station_distance_km").iloc[0]
    return str(best["storm_id"]), str(best["storm_name"]), float(best["nearest_station_distance_km"])


def build_catalog(train_end: str, max_gap_hours: int, padding_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    residual_path = PROCESSED_ROOT / "residual_matrix_1999_2022.csv.gz"
    residual = pd.read_csv(residual_path, index_col=0, parse_dates=True)
    residual.index = pd.to_datetime(residual.index, utc=True)
    train = residual.loc[: pd.Timestamp(train_end)]
    thresholds = pd.DataFrame(
        {
            "station_name": residual.columns,
            "positive_q95_m": [train[column].quantile(0.95) for column in residual.columns],
            "positive_q99_m": [train[column].quantile(0.99) for column in residual.columns],
            "training_start": train.index.min(),
            "training_end": train.index.max(),
        }
    )
    q95 = thresholds.set_index("station_name")["positive_q95_m"]
    q99 = thresholds.set_index("station_name")["positive_q99_m"]
    above_q95 = residual.ge(q95, axis="columns")
    above_q99 = residual.ge(q99, axis="columns")
    active = above_q99.any(axis=1) | (above_q95.sum(axis=1) >= 2)
    intervals = merge_active_hours(active, max_gap_hours)
    tracks = load_hurdat_near_stations()

    rows: list[dict[str, object]] = []
    for event_number, (core_start, core_end) in enumerate(intervals, start=1):
        window_start = core_start - pd.Timedelta(hours=padding_hours)
        window_end = core_end + pd.Timedelta(hours=padding_hours)
        segment = residual.loc[window_start:window_end]
        if segment.empty or not np.isfinite(segment.to_numpy()).any():
            continue
        stacked = segment.stack().dropna()
        peak_time, peak_station = stacked.idxmax()
        peak_value = float(stacked.max())
        triggered = sorted(set(above_q95.loc[core_start:core_end].columns[above_q95.loc[core_start:core_end].any(axis=0)]))
        storm_id, storm_name, storm_distance = match_hurdat(tracks, window_start, window_end)
        rows.append(
            {
                "event_id": f"NCS{event_number:04d}",
                "split": split_name(peak_time),
                "core_start": core_start,
                "core_end": core_end,
                "window_start": window_start,
                "window_end": window_end,
                "duration_hours": int((core_end - core_start) / pd.Timedelta(hours=1)) + 1,
                "peak_time": peak_time,
                "peak_station": peak_station,
                "peak_residual_m": peak_value,
                "triggered_station_count": len(triggered),
                "triggered_stations": ";".join(triggered),
                "hurdat_storm_id": storm_id,
                "hurdat_storm_name": storm_name,
                "hurdat_min_station_distance_km": storm_distance,
            }
        )
    return pd.DataFrame(rows), thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train-locked storm-driven residual event catalog")
    parser.add_argument("--train-end", default="2016-12-31T23:00:00Z")
    parser.add_argument("--max-gap-hours", type=int, default=72)
    parser.add_argument("--padding-hours", type=int, default=72)
    args = parser.parse_args()

    catalog, thresholds = build_catalog(args.train_end, args.max_gap_hours, args.padding_hours)
    catalog_path = PROCESSED_ROOT / "storm_driven_event_catalog_train_locked_1999_2022.csv"
    threshold_path = PROCESSED_ROOT / "storm_event_thresholds_train_1999_2016.csv"
    catalog.to_csv(catalog_path, index=False)
    thresholds.to_csv(threshold_path, index=False)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "threshold_source": "1999-01-01 through 2016-12-31 only",
        "event_rule": "any station above train q99 OR at least two stations above train q95",
        "max_gap_hours": args.max_gap_hours,
        "padding_hours": args.padding_hours,
        "events": int(len(catalog)),
        "events_by_split": catalog["split"].value_counts().to_dict(),
        "events_with_hurdat_match": int(catalog["hurdat_storm_id"].fillna("").ne("").sum()),
    }
    target = MANIFEST_ROOT / "event_catalog_report.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
