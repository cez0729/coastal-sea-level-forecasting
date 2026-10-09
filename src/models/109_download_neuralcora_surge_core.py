from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timezone, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "neuralcora_surge"
NOAA_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_MDAPI = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"
NHC_DATA_PAGE = "https://www.nhc.noaa.gov/data/"

STATIONS = {
    "8461490": "New London",
    "8510560": "Montauk",
    "8516945": "Kings Point",
    "8518750": "The Battery",
    "8531680": "Sandy Hook",
    "8534720": "Atlantic City",
    "8536110": "Cape May",
}

SOURCE_FILES = {
    "CORA_V1.1_intake.yml": "https://noaa-nos-cora-pds.s3.amazonaws.com/CORA_V1.1_intake.yml",
    "CORA_V1.1_intake_zarr3.yml": "https://noaa-nos-cora-pds.s3.amazonaws.com/CORA_V1.1_intake_zarr3.yml",
    "cora_bucket_root.xml": "https://noaa-nos-cora-pds.s3.amazonaws.com/?list-type=2&delimiter=/",
    "noaa_storm_surge_definition.html": "https://oceanservice.noaa.gov/facts/stormsurge-stormtide.html",
}


def fetch_bytes(url: str, retries: int = 5) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "NeuralCORA-Surge/1.0"})
            with urlopen(request, timeout=90) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Download failed after {retries} attempts: {url}\n{last_error}")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)


def download_source_metadata() -> None:
    output_dir = DATA_ROOT / "raw" / "source_metadata"
    for name, url in SOURCE_FILES.items():
        target = output_dir / name
        if target.exists() and target.stat().st_size > 0:
            print(f"Skipping existing: {target}")
            continue
        print(f"Downloading {url}")
        atomic_write(target, fetch_bytes(url))


def download_station_metadata() -> None:
    output_dir = DATA_ROOT / "raw" / "noaa" / "station_metadata"
    for station_id in STATIONS:
        target = output_dir / f"{station_id}.json"
        if target.exists() and target.stat().st_size > 0:
            print(f"Skipping existing: {target}")
            continue
        url = f"{NOAA_MDAPI}/stations/{station_id}.json?expand=details,sensors,products,datums"
        print(f"Downloading NOAA metadata: {station_id}")
        atomic_write(target, fetch_bytes(url))


def download_jobs(start_year: int, end_year: int, products: list[str]) -> list[tuple[str, str, int, int | None]]:
    jobs: list[tuple[str, str, int, int | None]] = []
    for station_id in STATIONS:
        for product in products:
            for year in range(start_year, end_year + 1):
                if product == "hourly_height":
                    jobs.extend((station_id, product, year, month) for month in range(1, 13))
                else:
                    # NOAA permits hourly tide predictions to be requested one year at a time.
                    jobs.append((station_id, product, year, None))
    return jobs


def download_noaa_period(job: tuple[str, str, int, int | None], datum: str) -> tuple[str, str]:
    station_id, product, year, month = job
    product_dir = "hourly_height" if product == "hourly_height" else "tide_predictions"
    output_dir = DATA_ROOT / "raw" / "noaa" / product_dir / station_id
    period_tag = f"{year}{month:02d}" if month is not None else str(year)
    target = output_dir / f"{station_id}_{product}_{period_tag}.csv"
    missing = target.with_suffix(".missing.json")
    if target.exists() and target.stat().st_size > 0:
        return "skipped", str(target)
    if missing.exists():
        return "missing", str(missing)

    if month is None:
        begin = date(year, 1, 1)
        end = date(year, 12, 31)
    else:
        begin = date(year, month, 1)
        end = date(year, month, calendar.monthrange(year, month)[1])
    params = {
        "station": station_id,
        "begin_date": begin.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "product": product,
        "datum": datum,
        "time_zone": "gmt",
        "units": "metric",
        "format": "csv",
        "application": "NeuralCORA_Surge",
    }
    if product == "predictions":
        params["interval"] = "h"
    url = f"{NOAA_API}?{urlencode(params)}"

    try:
        payload = fetch_bytes(url)
        text = payload.decode("utf-8-sig", errors="replace")
        if "Error" in text[:500] and "Date Time" not in text[:500]:
            missing_payload = {
                "station": station_id,
                "product": product,
                "year": year,
                "month": month,
                "url": url,
                "response": text[:1000],
            }
            atomic_write(missing, json.dumps(missing_payload, indent=2).encode("utf-8"))
            return "missing", str(missing)
        atomic_write(target, payload)
        return "downloaded", str(target)
    except Exception as exc:
        return "failed", f"{station_id} {product} {period_tag}: {exc}"


def download_noaa(
    start_year: int,
    end_year: int,
    products: list[str],
    datum: str,
    workers: int,
) -> None:
    jobs = download_jobs(start_year, end_year, products)
    counts = {"downloaded": 0, "skipped": 0, "missing": 0, "failed": 0}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_noaa_period, job, datum): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            status, detail = future.result()
            counts[status] += 1
            if status == "failed":
                failures.append(detail)
            if index % 100 == 0 or status == "failed":
                print(f"NOAA progress {index}/{len(jobs)}: {counts}")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "start_year": start_year,
        "end_year": end_year,
        "datum": datum,
        "products": products,
        "counts": counts,
        "failures": failures,
    }
    target = DATA_ROOT / "manifests" / f"noaa_download_{start_year}_{end_year}.json"
    atomic_write(target, json.dumps(report, indent=2).encode("utf-8"))
    if failures:
        raise RuntimeError(f"{len(failures)} NOAA requests failed; see {target}")


def download_hurdat2() -> None:
    page = fetch_bytes(NHC_DATA_PAGE).decode("utf-8", errors="replace")
    match = re.search(r'href="(/data/hurdat/hurdat2-1851-\d{4}-\d+\.txt)"', page)
    if not match:
        raise RuntimeError("Could not locate the current Atlantic HURDAT2 link")
    relative_url = match.group(1)
    url = f"https://www.nhc.noaa.gov{relative_url}"
    target = DATA_ROOT / "raw" / "events" / Path(relative_url).name
    if target.exists() and target.stat().st_size > 0:
        print(f"Skipping existing: {target}")
        return
    print(f"Downloading HURDAT2: {url}")
    atomic_write(target, fetch_bytes(url))


def write_manifest() -> None:
    rows: list[dict[str, object]] = []
    target = DATA_ROOT / "manifests" / "file_manifest.csv"
    for path in sorted(DATA_ROOT.rglob("*")):
        if not path.is_file() or path.name.endswith(".part") or path == target:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "relative_path": path.relative_to(DATA_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
            }
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256", "modified_utc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote manifest: {target} ({len(rows)} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NeuralCORA-Surge core public data")
    parser.add_argument(
        "stage",
        choices=["metadata", "noaa", "events", "manifest", "all"],
        help="Download stage to execute",
    )
    parser.add_argument("--start-year", type=int, default=1999)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument(
        "--products",
        nargs="+",
        choices=["hourly_height", "predictions"],
        default=["hourly_height", "predictions"],
    )
    parser.add_argument("--datum", default="MSL")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if args.stage in {"metadata", "all"}:
        download_source_metadata()
        download_station_metadata()
    if args.stage in {"noaa", "all"}:
        download_noaa(args.start_year, args.end_year, args.products, args.datum, args.workers)
    if args.stage in {"events", "all"}:
        download_hurdat2()
    if args.stage in {"manifest", "all"}:
        write_manifest()


if __name__ == "__main__":
    main()
