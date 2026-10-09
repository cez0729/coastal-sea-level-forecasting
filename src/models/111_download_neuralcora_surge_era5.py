from __future__ import annotations

import argparse
import calendar
from collections import defaultdict
from datetime import date
from pathlib import Path

import cdsapi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "neuralcora_surge" / "raw" / "era5"
DEFAULT_CATALOG = ROOT / "data" / "neuralcora_surge" / "processed" / "publication_event_catalog_1999_2022.csv"


def iter_months(start: date, end: date):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1


def download_month(client: cdsapi.Client, year: int, month: int, output_dir: Path) -> None:
    target = output_dir / f"era5_u10_v10_msl_{year}{month:02d}.nc"
    if target.exists() and target.stat().st_size > 0:
        print(f"Skipping existing: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
        ],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}" for day in range(1, calendar.monthrange(year, month)[1] + 1)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        # Covers all seven stations with a buffer for regional forcing.
        "area": [42.5, -76.0, 38.0, -70.0],
    }
    temporary = target.with_suffix(".nc.part")
    print(f"Submitting ERA5 request: {year}-{month:02d}")
    client.retrieve("reanalysis-era5-single-levels", request, str(temporary))
    temporary.replace(target)
    print(f"Saved: {target}")


def catalog_days_by_month(catalog_csv: Path) -> dict[tuple[int, int], list[int]]:
    import pandas as pd

    catalog = pd.read_csv(catalog_csv)
    required = {"window_start", "window_end"}
    missing = sorted(required - set(catalog.columns))
    if missing:
        raise ValueError(f"Event catalog is missing columns: {missing}")
    grouped: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in catalog.itertuples(index=False):
        start = pd.Timestamp(row.window_start).floor("D")
        end = pd.Timestamp(row.window_end).ceil("D")
        for current in pd.date_range(start, end, freq="D"):
            grouped[(current.year, current.month)].add(current.day)
    return {key: sorted(days) for key, days in grouped.items()}


def ordered_catalog_months(catalog_csv: Path, schedule: str) -> list[tuple[tuple[int, int], list[int]]]:
    import pandas as pd

    catalog = pd.read_csv(catalog_csv)
    grouped_days = catalog_days_by_month(catalog_csv)
    if schedule == "chronological":
        return sorted(grouped_days.items())
    split_months: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in catalog.itertuples(index=False):
        start = pd.Timestamp(row.window_start).floor("D")
        end = pd.Timestamp(row.window_end).ceil("D")
        for current in pd.date_range(start, end, freq="D"):
            split_months[str(row.split)].add((current.year, current.month))
    queues = {split: list(sorted(split_months[split])) for split in ("train", "validation", "test")}
    ordered: list[tuple[int, int]] = []
    while any(queues.values()):
        for split in ("train", "validation", "test"):
            if queues[split]:
                ordered.append(queues[split].pop(0))
    seen: set[tuple[int, int]] = set()
    unique_order: list[tuple[int, int]] = []
    for item in ordered:
        if item not in seen:
            seen.add(item)
            unique_order.append(item)
    return [(item, grouped_days[item]) for item in unique_order]


def download_event_month(
    client: cdsapi.Client,
    year: int,
    month: int,
    days: list[int],
    output_dir: Path,
) -> None:
    full_month = output_dir / f"era5_u10_v10_msl_{year}{month:02d}.nc"
    target = output_dir / f"era5_u10_v10_msl_events_{year}{month:02d}.nc"
    if full_month.exists() and full_month.stat().st_size > 0:
        print(f"Skipping event month covered by full month: {full_month}")
        return
    if target.exists() and target.stat().st_size > 0:
        print(f"Skipping existing: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
        ],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}" for day in days],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [42.5, -76.0, 38.0, -70.0],
    }
    temporary = target.with_suffix(".nc.part")
    print(f"Submitting ERA5 event request: {year}-{month:02d}, {len(days)} days")
    client.retrieve("reanalysis-era5-single-levels", request, str(temporary))
    temporary.replace(target)
    print(f"Saved: {target}")


def download_year(client: cdsapi.Client, year: int, output_dir: Path) -> None:
    target = output_dir / f"era5_u10_v10_msl_{year}.nc"
    if target.exists() and target.stat().st_size > 0:
        print(f"Skipping existing: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
        ],
        "year": [str(year)],
        "month": [f"{month:02d}" for month in range(1, 13)],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [42.5, -76.0, 38.0, -70.0],
    }
    temporary = target.with_suffix(".nc.part")
    print(f"Submitting ERA5 annual request: {year}")
    client.retrieve("reanalysis-era5-single-levels", request, str(temporary))
    temporary.replace(target)
    print(f"Saved: {target}")


def download_quarter(client: cdsapi.Client, year: int, quarter: int, output_dir: Path) -> None:
    months = list(range((quarter - 1) * 3 + 1, quarter * 3 + 1))
    target = output_dir / f"era5_u10_v10_msl_{year}Q{quarter}.nc"
    if target.exists() and target.stat().st_size > 0:
        print(f"Skipping existing: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
        ],
        "year": [str(year)],
        "month": [f"{month:02d}" for month in months],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [42.5, -76.0, 38.0, -70.0],
    }
    temporary = target.with_suffix(".nc.part")
    print(f"Submitting ERA5 quarterly request: {year} Q{quarter}")
    client.retrieve("reanalysis-era5-single-levels", request, str(temporary))
    temporary.replace(target)
    print(f"Saved: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ERA5 forcing for NeuralCORA-Surge")
    parser.add_argument("--start", type=date.fromisoformat, default=date(1979, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2022, 12, 31))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--granularity", choices=["event", "month", "quarter", "year"], default="month")
    parser.add_argument("--catalog-csv", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--schedule", choices=["chronological", "balanced"], default="balanced")
    args = parser.parse_args()

    client = cdsapi.Client()
    if args.granularity == "event":
        for (year, month), days in ordered_catalog_months(args.catalog_csv, args.schedule):
            if (year, month) < (args.start.year, args.start.month) or (year, month) > (args.end.year, args.end.month):
                continue
            download_event_month(client, year, month, days, args.out_dir)
    elif args.granularity == "year":
        for year in range(args.start.year, args.end.year + 1):
            download_year(client, year, args.out_dir)
    elif args.granularity == "quarter":
        for year in range(args.start.year, args.end.year + 1):
            for quarter in range(1, 5):
                download_quarter(client, year, quarter, args.out_dir)
    else:
        for year, month in iter_months(args.start, args.end):
            download_month(client, year, month, args.out_dir)


if __name__ == "__main__":
    main()
