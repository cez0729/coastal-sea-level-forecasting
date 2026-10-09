from __future__ import annotations

import argparse
import calendar
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = REPO_ROOT
API_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

STATIONS = [
    "8461490",  # New London, CT
    "8510560",  # Montauk, NY
    "8516945",  # Kings Point, NY
    "8518750",  # The Battery, NY
    "8531680",  # Sandy Hook, NJ
    "8534720",  # Atlantic City, NJ
    "8536110",  # Cape May, NJ
]


def month_ranges(start_year: int, end_year: int):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            last = calendar.monthrange(year, month)[1]
            yield date(year, month, 1), date(year, month, last)


def request_csv(params: dict[str, str], retries: int = 4) -> str:
    url = f"{API_BASE}?{urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=60) as response:
                text = response.read().decode("utf-8-sig")
            if "Error" in text[:300] and "Date Time" not in text[:300]:
                raise RuntimeError(text[:500])
            return text
        except Exception as exc:  # NOAA occasionally throttles or drops long calls.
            last_error = exc
            time.sleep(2.0 + attempt * 2.0)
    raise RuntimeError(f"NOAA request failed after {retries} retries: {url}\n{last_error}")


def strip_duplicate_header(chunks: list[str]) -> str:
    lines: list[str] = []
    header: str | None = None
    for chunk in chunks:
        chunk_lines = [line for line in chunk.splitlines() if line.strip()]
        if not chunk_lines:
            continue
        if header is None:
            header = chunk_lines[0]
            lines.append(header)
        lines.extend(line for line in chunk_lines[1:] if line != header)
    return "\n".join(lines) + "\n"


def download_station_years(
    station: str,
    start_year: int,
    end_year: int,
    product: str,
    out_path: Path,
) -> None:
    chunks: list[str] = []
    for begin, end in month_ranges(start_year, end_year):
        params = {
            "station": station,
            "begin_date": begin.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "product": product,
            "datum": "MLLW",
            "time_zone": "gmt",
            "units": "metric",
            "format": "csv",
            "application": "sea_level_multistate_ode_project",
        }
        if product == "predictions":
            params["interval"] = "h"
        print(f"Downloading NOAA {product}: station={station} {begin} to {end}")
        chunks.append(request_csv(params))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(strip_duplicate_header(chunks), encoding="utf-8", newline="\n")
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download multi-year NOAA hourly water level and tide prediction raw CSV files."
    )
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--stations", nargs="*", default=STATIONS)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    tag = f"{args.start_year}_{args.end_year}"
    water_dir = ROOT / "data" / "raw" / f"NOAA_hourly_water_level_{tag}"
    tide_dir = ROOT / "data" / "raw" / f"NOAA_hourly_tide_predictions_{tag}"

    for station in args.stations:
        water_path = water_dir / f"CO-OPS__{station}__hr_{tag}.csv"
        tide_path = tide_dir / f"CO-OPS__{station}__pr_{tag}.csv"

        if not (args.skip_existing and water_path.exists()):
            download_station_years(
                station=station,
                start_year=args.start_year,
                end_year=args.end_year,
                product="hourly_height",
                out_path=water_path,
            )
        else:
            print(f"Skipping existing: {water_path}")

        if not (args.skip_existing and tide_path.exists()):
            download_station_years(
                station=station,
                start_year=args.start_year,
                end_year=args.end_year,
                product="predictions",
                out_path=tide_path,
            )
        else:
            print(f"Skipping existing: {tide_path}")


if __name__ == "__main__":
    main()
