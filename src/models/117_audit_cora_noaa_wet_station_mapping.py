from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "neuralcora_surge"
DEFAULT_CORA_ROOT = DATA_ROOT / "raw" / "cora" / "catalog_days"
DEFAULT_NOAA = DATA_ROOT / "processed" / "noaa_station_water_tide_residual_1999_2022.csv.gz"
DEFAULT_OUTPUT = ROOT / "results" / "neuralcora_surge_station_audit"

STATIONS = {
    "8461490": ("New London", 41.371666, -72.09556),
    "8510560": ("Montauk", 41.048332, -71.95944),
    "8516945": ("Kings Point", 40.8103, -73.7649),
    "8518750": ("The Battery", 40.700554, -74.01417),
    "8531680": ("Sandy Hook", 40.4669, -74.0094),
    "8534720": ("Atlantic City", 39.356667, -74.41805),
    "8536110": ("Cape May", 38.9683, -74.96),
}


def date_from_path(path: Path) -> str:
    token = path.stem.rsplit("_", 1)[-1]
    return pd.to_datetime(token, format="%Y%m%d").date().isoformat()


def available_files(cora_root: Path) -> pd.DataFrame:
    paths = sorted(cora_root.glob("*/*.nc"))
    if not paths:
        raise FileNotFoundError(f"No completed CORA files found below {cora_root}")
    date_index = pd.read_csv(cora_root / "catalog_date_index.csv")
    split_by_date = dict(zip(date_index["date"].astype(str), date_index["splits"].astype(str)))
    rows = []
    for path in paths:
        current = date_from_path(path)
        memberships = split_by_date.get(current, "unknown").split(";")
        rows.append({"date": current, "split": memberships[0], "path": path})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def candidate_indices(lat: np.ndarray, lon: np.ndarray, k: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    mean_lat = float(np.mean([value[1] for value in STATIONS.values()]))
    lon_scale = float(np.cos(np.deg2rad(mean_lat)))
    tree = cKDTree(np.column_stack([lat, lon * lon_scale]))
    station_candidates: dict[str, np.ndarray] = {}
    for station_id, (_, station_lat, station_lon) in STATIONS.items():
        _, indices = tree.query([station_lat, station_lon * lon_scale], k=min(k, len(lat)))
        station_candidates[station_id] = np.atleast_1d(indices).astype(np.int64)
    union = np.unique(np.concatenate(list(station_candidates.values())))
    return station_candidates, union


def read_nodes(path: Path, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        values = ds["zeta"].isel(nodes=indices).transpose("time", "nodes").values.astype(np.float32)
        times = ds["time"].values.astype("datetime64[ns]")
    return values, times


def stable_wet_mapping(files: pd.DataFrame, k: int, min_train_valid: float) -> pd.DataFrame:
    with xr.open_dataset(files.iloc[0]["path"], engine="h5netcdf") as ds:
        lat = ds["lat"].values.astype(np.float64)
        lon = ds["lon"].values.astype(np.float64)
    station_candidates, union = candidate_indices(lat, lon, k)
    union_position = {int(node): position for position, node in enumerate(union)}
    valid_count = np.zeros(len(union), dtype=np.int64)
    total_count = 0
    train_files = files[files["split"] == "train"]
    if train_files.empty:
        raise ValueError("At least one completed train CORA file is required to select wet nodes")
    for path in train_files["path"]:
        values, _ = read_nodes(path, union)
        valid_count += np.isfinite(values).sum(axis=0)
        total_count += values.shape[0]
    validity = valid_count / max(total_count, 1)

    rows: list[dict[str, object]] = []
    for station_id, (name, station_lat, station_lon) in STATIONS.items():
        candidates = station_candidates[station_id]
        candidate_validity = np.asarray([validity[union_position[int(node)]] for node in candidates])
        eligible = candidates[candidate_validity >= min_train_valid]
        relaxed = False
        if eligible.size == 0:
            best_validity = float(candidate_validity.max())
            eligible = candidates[candidate_validity == best_validity]
            relaxed = True
        scale = np.cos(np.deg2rad(station_lat))
        distance_degrees = np.sqrt((lat[eligible] - station_lat) ** 2 + ((lon[eligible] - station_lon) * scale) ** 2)
        selected = int(eligible[int(np.argmin(distance_degrees))])
        position = union_position[selected]
        distance_km = float(distance_degrees.min() * 111.195)
        rows.append(
            {
                "station_id": station_id,
                "station_name": name,
                "station_lat": station_lat,
                "station_lon": station_lon,
                "cora_node_index": selected,
                "cora_lat": float(lat[selected]),
                "cora_lon": float(lon[selected]),
                "distance_km": distance_km,
                "train_valid_fraction": float(validity[position]),
                "selection_relaxed": relaxed,
                "train_files_used": int(len(train_files)),
                "train_hours_used": int(total_count),
            }
        )
    return pd.DataFrame(rows)


def extract_mapped_series(files: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    indices = mapping["cora_node_index"].to_numpy(dtype=np.int64)
    parts: list[pd.DataFrame] = []
    for row in files.itertuples(index=False):
        values, times = read_nodes(row.path, indices)
        for station_index, station in enumerate(mapping.itertuples(index=False)):
            parts.append(
                pd.DataFrame(
                    {
                        "station_id": str(station.station_id),
                        "station_name": station.station_name,
                        "time": pd.to_datetime(times, utc=True),
                        "split": row.split,
                        "cora_total_m": values[:, station_index],
                    }
                )
            )
    combined = pd.concat(parts, ignore_index=True)
    return combined.drop_duplicates(["station_id", "time"], keep="last")


def fit_calibration(matched: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for station_id, group in matched.groupby("station_id", sort=False):
        group = group.copy()
        train = group[(group["split"] == "train") & group["cora_total_m"].notna() & group["observed_msl_m"].notna()]
        if len(train) < 2:
            intercept, slope = 0.0, 1.0
        else:
            slope, intercept = np.polyfit(train["cora_total_m"], train["observed_msl_m"], deg=1)
        bias = float((train["observed_msl_m"] - train["cora_total_m"]).mean()) if len(train) else 0.0
        group["cora_bias_corrected_m"] = group["cora_total_m"] + bias
        group["cora_affine_corrected_m"] = intercept + slope * group["cora_total_m"]
        group["train_bias_m"] = bias
        group["train_affine_intercept_m"] = intercept
        group["train_affine_slope"] = slope
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def score_pair(observed: pd.Series, predicted: pd.Series, prefix: str) -> dict[str, float | int]:
    valid = observed.notna() & predicted.notna()
    y = observed[valid].to_numpy(dtype=float)
    p = predicted[valid].to_numpy(dtype=float)
    if len(y) == 0:
        return {f"{prefix}_n": 0, f"{prefix}_rmse_m": np.nan, f"{prefix}_mae_m": np.nan, f"{prefix}_r2": np.nan, f"{prefix}_correlation": np.nan}
    error = p - y
    denominator = float(np.sum((y - y.mean()) ** 2))
    correlation = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0 else np.nan
    return {
        f"{prefix}_n": int(len(y)),
        f"{prefix}_rmse_m": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_mae_m": float(np.mean(np.abs(error))),
        f"{prefix}_r2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else np.nan,
        f"{prefix}_correlation": correlation,
    }


def metrics_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [("all", "all", frame)]
    groups.extend((split, "all", group) for split, group in frame.groupby("split"))
    groups.extend((split, station, group) for (split, station), group in frame.groupby(["split", "station_name"]))
    for split, station, group in groups:
        row: dict[str, object] = {"split": split, "station_name": station, "rows": int(len(group))}
        row.update(score_pair(group["observed_msl_m"], group["cora_total_m"], "total_raw"))
        row.update(score_pair(group["observed_msl_m"], group["cora_bias_corrected_m"], "total_bias"))
        row.update(score_pair(group["observed_msl_m"], group["cora_affine_corrected_m"], "total_affine"))
        row.update(score_pair(group["residual_m"], group["cora_total_m"], "residual_mismatch"))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train-selected stable wet CORA nodes against NOAA total water level")
    parser.add_argument("--cora-root", type=Path, default=DEFAULT_CORA_ROOT)
    parser.add_argument("--noaa-csv", type=Path, default=DEFAULT_NOAA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--min-train-valid", type=float, default=0.95)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = available_files(args.cora_root)
    mapping = stable_wet_mapping(files, args.candidates, args.min_train_valid)
    cora = extract_mapped_series(files, mapping)
    noaa = pd.read_csv(args.noaa_csv, compression="infer", usecols=["station_id", "time", "observed_msl_m", "residual_m"])
    noaa["station_id"] = noaa["station_id"].astype(str)
    noaa["time"] = pd.to_datetime(noaa["time"], utc=True)
    matched = cora.merge(noaa, on=["station_id", "time"], how="left", validate="one_to_one")
    matched = fit_calibration(matched)
    metrics = metrics_table(matched)

    mapping.to_csv(args.output_dir / "station_wet_node_mapping.csv", index=False)
    matched.to_csv(args.output_dir / "cora_noaa_hourly_matched.csv.gz", index=False, compression="gzip")
    metrics.to_csv(args.output_dir / "cora_noaa_metrics.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_target": "CORA zeta is audited against NOAA observed total water level; residual comparison is mismatch diagnostic only",
        "mapping_selection": "nearest node among candidates meeting train-only finite-fraction threshold",
        "completed_cora_files": int(len(files)),
        "files_by_split": files.groupby("split").size().astype(int).to_dict(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "audit_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(mapping.to_string(index=False))
    print(metrics[metrics["station_name"] == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
