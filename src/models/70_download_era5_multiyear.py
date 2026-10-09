from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import cdsapi


ROOT = Path(__file__).resolve().parents[1]


def download_month(year: int, month: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"era5_single_levels_u10_v10_msl_{year}{month:02d}.nc"
    if target.exists() and target.stat().st_size > 0:
        print(f"Skipping existing: {target}")
        return

    last_day = calendar.monthrange(year, month)[1]
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
        ],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, last_day + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        # North, West, South, East.
        "area": [42.5, -76.0, 38.0, -70.0],
    }

    print(f"Submitting ERA5 request for {year}-{month:02d}: {target}")
    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-single-levels", request, str(target))
    print(f"Saved: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ERA5 u10/v10/msl for 2023-2025 project region.")
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--out-dir", default="data/raw/ERA5_2023_2025")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            download_month(year, month, out_dir)


if __name__ == "__main__":
    main()
