from __future__ import annotations

import argparse
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "neuralcora_surge"
REFERENCE_TEMPLATE = (
    "https://noaa-nos-cora-pds.s3.amazonaws.com/"
    "cora_gec/500m_grid/water_levels/zarr/500m_grid_zeta_{date}.nc.zarr"
)

STATIONS = {
    "8461490": ("New London", 41.371666, -72.09556),
    "8510560": ("Montauk", 41.048332, -71.95944),
    "8516945": ("Kings Point", 40.8103, -73.7649),
    "8518750": ("The Battery", 40.700554, -74.01417),
    "8531680": ("Sandy Hook", 40.4669, -74.0094),
    "8534720": ("Atlantic City", 39.356667, -74.41805),
    "8536110": ("Cape May", 38.9683, -74.96),
}


def open_reference(reference_url: str) -> xr.Dataset:
    reference_fs = fsspec.filesystem(
        "reference",
        fo=reference_url,
        remote_protocol="s3",
        remote_options={"anon": True, "asynchronous": True},
        asynchronous=True,
    )
    return xr.open_zarr(reference_fs.get_mapper(""), consolidated=False, chunks=None)


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_mapping(ds: xr.Dataset) -> pd.DataFrame:
    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    rows: list[dict[str, object]] = []
    for station_id, (name, station_lat, station_lon) in STATIONS.items():
        scale = np.cos(np.deg2rad(station_lat))
        distance_sq = (lat - station_lat) ** 2 + ((lon - station_lon) * scale) ** 2
        index = int(np.nanargmin(distance_sq))
        rows.append(
            {
                "station_id": station_id,
                "station_name": name,
                "station_lat": station_lat,
                "station_lon": station_lon,
                "cora_node_index": index,
                "cora_lat": float(lat[index]),
                "cora_lon": float(lon[index]),
                "distance_degrees": float(np.sqrt(distance_sq[index])),
            }
        )
    return pd.DataFrame(rows)


def write_mapping(sample_date: date) -> Path:
    reference_url = REFERENCE_TEMPLATE.format(date=sample_date.strftime("%Y%m%d"))
    with open_reference(reference_url) as ds:
        mapping = build_mapping(ds)
    target = DATA_ROOT / "raw" / "cora" / "cora_500m_station_mapping.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(target, index=False)
    print(mapping.to_string(index=False))
    print(f"Saved: {target}")
    return target


def extract_day(current: date, bbox: tuple[float, float, float, float], output_dir: Path) -> dict[str, object]:
    target = output_dir / f"cora_500m_{current.strftime('%Y%m%d')}.nc"
    if target.exists() and target.stat().st_size > 0:
        return {"date": current.isoformat(), "status": "skipped", "path": str(target)}

    reference_url = REFERENCE_TEMPLATE.format(date=current.strftime("%Y%m%d"))
    with open_reference(reference_url) as ds:
        west, east, south, north = bbox
        node_index = np.flatnonzero(
            (ds["lon"].values >= west)
            & (ds["lon"].values <= east)
            & (ds["lat"].values >= south)
            & (ds["lat"].values <= north)
        )
        if node_index.size == 0:
            raise RuntimeError(f"No CORA nodes found in bbox {bbox}")
        subset = ds[["zeta"]].isel(nodes=node_index).load().astype({"zeta": "float32"})
    subset.attrs.update(
        {
            "source_reference": reference_url,
            "bbox_west_east_south_north": json.dumps(bbox),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".nc.part")
    encoding = {"zeta": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    subset.to_netcdf(temporary, engine="h5netcdf", encoding=encoding)
    temporary.replace(target)
    return {
        "date": current.isoformat(),
        "status": "downloaded",
        "path": str(target),
        "nodes": int(node_index.size),
        "bytes": target.stat().st_size,
    }


def extract_event(
    event_id: str,
    start: date,
    end: date,
    bbox: tuple[float, float, float, float],
) -> None:
    output_dir = DATA_ROOT / "raw" / "cora" / "events" / event_id
    records: list[dict[str, object]] = []
    for current in daterange(start, end):
        print(f"Extracting CORA {event_id}: {current}")
        records.append(extract_day(current, bbox, output_dir))
    manifest = {
        "event_id": event_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "bbox": bbox,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "days": records,
    }
    target = output_dir / "event_manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved: {target}")


def catalog_date_index(catalog_csv: Path, splits: list[str]) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_csv)
    required = {"event_id", "split", "window_start", "window_end"}
    missing = sorted(required - set(catalog.columns))
    if missing:
        raise ValueError(f"Event catalog is missing columns: {missing}")
    catalog = catalog[catalog["split"].isin(splits)].copy()
    rows: list[dict[str, object]] = []
    for row in catalog.itertuples(index=False):
        start = pd.Timestamp(row.window_start).floor("D")
        end = pd.Timestamp(row.window_end).ceil("D")
        for current in pd.date_range(start, end, freq="D"):
            rows.append(
                {
                    "date": current.date().isoformat(),
                    "split": row.split,
                    "event_id": row.event_id,
                }
            )
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        raise ValueError("No event dates matched the requested splits")
    return (
        expanded.groupby("date", as_index=False)
        .agg(
            splits=("split", lambda values: ";".join(sorted(set(values)))),
            event_ids=("event_id", lambda values: ";".join(sorted(set(values)))),
            event_count=("event_id", "nunique"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def balanced_catalog_order(date_index: pd.DataFrame, splits: list[str]) -> pd.DataFrame:
    """Interleave chronological split queues without changing split membership."""
    queues: dict[str, deque[int]] = {split: deque() for split in splits}
    fallback: deque[int] = deque()
    for index, value in date_index["splits"].items():
        memberships = str(value).split(";")
        owner = next((split for split in splits if split in memberships), None)
        (queues[owner] if owner is not None else fallback).append(index)

    ordered: list[int] = []
    while any(queues.values()):
        for split in splits:
            if queues[split]:
                ordered.append(queues[split].popleft())
    ordered.extend(fallback)
    return date_index.loc[ordered].reset_index(drop=True)


def extract_day_with_retry(
    current: date,
    bbox: tuple[float, float, float, float],
    output_root: Path,
    retries: int,
) -> dict[str, object]:
    output_dir = output_root / f"{current.year:04d}"
    for attempt in range(1, retries + 1):
        try:
            return extract_day(current, bbox, output_dir)
        except Exception as exc:
            if attempt >= retries:
                return {
                    "date": current.isoformat(),
                    "status": "failed",
                    "path": str(output_dir / f"cora_500m_{current.strftime('%Y%m%d')}.nc"),
                    "error": repr(exc),
                }
            time.sleep(min(30, 2**attempt))
    raise RuntimeError("Unreachable retry state")


def extract_catalog(
    catalog_csv: Path,
    splits: list[str],
    bbox: tuple[float, float, float, float],
    workers: int,
    retries: int,
    limit_days: int | None,
    schedule: str,
) -> None:
    output_root = DATA_ROOT / "raw" / "cora" / "catalog_days"
    output_root.mkdir(parents=True, exist_ok=True)
    date_index = catalog_date_index(catalog_csv, splits)
    if schedule == "balanced":
        date_index = balanced_catalog_order(date_index, splits)
    if limit_days is not None:
        date_index = date_index.iloc[:limit_days].copy()
    date_index.to_csv(output_root / "catalog_date_index.csv", index=False)
    dates = [date.fromisoformat(value) for value in date_index["date"]]
    records: list[dict[str, object]] = []
    manifest_path = output_root / "download_manifest.csv"
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(extract_day_with_retry, current, bbox, output_root, retries): current
            for current in dates
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(f"[{completed}/{len(futures)}] {record['date']}: {record['status']}", flush=True)
            if completed % 25 == 0 or completed == len(futures):
                pd.DataFrame(records).sort_values("date").to_csv(manifest_path, index=False)
    failed = [record for record in records if record["status"] == "failed"]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_csv": str(catalog_csv),
        "splits": splits,
        "bbox": bbox,
        "unique_days": len(dates),
        "downloaded_or_skipped": len(records) - len(failed),
        "failed": len(failed),
    }
    (output_root / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failed:
        raise RuntimeError(f"CORA catalog extraction finished with {len(failed)} failed days")


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CORA V1.1 subsets for NeuralCORA-Surge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mapping_parser = subparsers.add_parser("mapping")
    mapping_parser.add_argument("--sample-date", type=parse_date, default=date(2022, 1, 1))

    event_parser = subparsers.add_parser("event")
    event_parser.add_argument("--event-id", required=True)
    event_parser.add_argument("--start", type=parse_date, required=True)
    event_parser.add_argument("--end", type=parse_date, required=True)
    event_parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "EAST", "SOUTH", "NORTH"),
        default=(-75.25, -71.25, 38.75, 41.60),
    )
    catalog_parser = subparsers.add_parser("catalog")
    catalog_parser.add_argument(
        "--catalog-csv",
        type=Path,
        default=DATA_ROOT / "processed" / "storm_driven_event_catalog_train_locked_1999_2022.csv",
    )
    catalog_parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "validation", "test"])
    catalog_parser.add_argument("--workers", type=int, default=4)
    catalog_parser.add_argument("--retries", type=int, default=4)
    catalog_parser.add_argument("--limit-days", type=int)
    catalog_parser.add_argument(
        "--schedule",
        choices=["chronological", "balanced"],
        default="balanced",
        help="balanced interleaves split dates so validation/test data become available early",
    )
    catalog_parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "EAST", "SOUTH", "NORTH"),
        default=(-75.25, -71.25, 38.75, 41.60),
    )
    args = parser.parse_args()

    if args.command == "mapping":
        write_mapping(args.sample_date)
    elif args.command == "event":
        extract_event(args.event_id, args.start, args.end, tuple(args.bbox))
    else:
        extract_catalog(
            args.catalog_csv,
            args.splits,
            tuple(args.bbox),
            args.workers,
            args.retries,
            args.limit_days,
            args.schedule,
        )


if __name__ == "__main__":
    main()
