"""Download two nearby NOAA gauges for an external spatial-transfer audit.

The external-node experiment is intentionally separate from the 34-feature
seven-station model: it uses only observed hourly level, NOAA tide prediction,
and residual history.  This avoids inventing unavailable wave/current features
for new gauges while testing whether the learned coastal signal transfers.
"""
from __future__ import annotations

import argparse
import calendar
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external_nodes_noaa_2023_2025"
STATIONS = {"8537121": "Ship John Shoal", "8551762": "Delaware City"}
API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
MD_API = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"


def fetch(url: str, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "coastal-residual-spatial-audit/1.0"})
            with urlopen(req, timeout=90) as response:
                return response.read()
        except Exception as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Download failed: {url}: {last}")


def request_url(station: str, product: str, begin: date, end: date) -> str:
    params = {
        "station": station, "begin_date": begin.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"), "product": product,
        "datum": "MSL", "time_zone": "gmt", "units": "metric",
        "format": "csv", "application": "spatial_residual_audit",
    }
    if product == "predictions":
        params["interval"] = "h"
    return API + "?" + urlencode(params)


def download_job(job):
    station, product, year, month = job
    folder = OUT / "raw" / product / station
    folder.mkdir(parents=True, exist_ok=True)
    tag = f"{year}{month:02d}" if month else str(year)
    target = folder / f"{station}_{product}_{tag}.csv"
    if target.exists() and target.stat().st_size > 0:
        return "skipped", str(target)
    begin = date(year, month, 1) if month else date(year, 1, 1)
    if month:
        end = date(year, month, calendar.monthrange(year, month)[1])
    else:
        end = date(year, 12, 31)
    payload = fetch(request_url(station, product, begin, end))
    text = payload.decode("utf-8-sig", errors="replace")
    if "Date Time" not in text[:500]:
        raise RuntimeError(f"NOAA response did not contain data columns: {station} {product} {tag}: {text[:300]}")
    target.write_bytes(payload)
    return "downloaded", str(target)


def read_product(station: str, product: str) -> pd.DataFrame:
    frames = []
    value = "Water Level" if product == "hourly_height" else "Prediction"
    for path in sorted((OUT / "raw" / product / station).glob("*.csv")):
        frame = pd.read_csv(path, skipinitialspace=True)
        frame.columns = [str(c).strip() for c in frame.columns]
        if "Date Time" not in frame.columns or value not in frame.columns:
            continue
        keep = frame[["Date Time", value]].copy()
        keep["time"] = pd.to_datetime(keep.pop("Date Time"), utc=True, errors="coerce")
        keep[value] = pd.to_numeric(keep[value], errors="coerce")
        frames.append(keep)
    if not frames:
        return pd.DataFrame(columns=["time", "value"])
    out = pd.concat(frames, ignore_index=True).dropna(subset=["time"]).drop_duplicates("time").sort_values("time")
    return out.rename(columns={value: "value"})


def build_processed(start_year: int, end_year: int):
    rows = []
    for station, name in STATIONS.items():
        observed = read_product(station, "hourly_height").rename(columns={"value": "observed_msl_m"})
        tide = read_product(station, "predictions").rename(columns={"value": "tide_msl_m"})
        frame = observed.merge(tide, on="time", how="outer").sort_values("time")
        frame.insert(0, "station_name", name)
        frame.insert(0, "station_id", station)
        frame["residual_m"] = frame["observed_msl_m"] - frame["tide_msl_m"]
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    out = OUT / "processed"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "external_nodes_water_tide_residual_2023_2025.csv.gz", index=False, compression="gzip", date_format="%Y-%m-%dT%H:%M:%SZ")
    coverage = result.groupby(["station_id", "station_name"], as_index=False).agg(
        rows=("time", "size"), observed_hours=("observed_msl_m", "count"),
        tide_hours=("tide_msl_m", "count"), residual_hours=("residual_m", "count"),
    )
    coverage.to_csv(out / "coverage.csv", index=False)
    return coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    jobs = []
    for station in STATIONS:
        for year in range(args.start_year, args.end_year + 1):
            jobs.extend((station, "hourly_height", year, month) for month in range(1, 13))
            jobs.append((station, "predictions", year, None))
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_job, job) for job in jobs]
        for future in as_completed(futures):
            try:
                status, detail = future.result()
                counts[status] += 1
            except Exception as exc:
                counts["failed"] += 1
                failures.append(str(exc))
    coverage = build_processed(args.start_year, args.end_year)
    (OUT / "download_manifest.json").write_text(json.dumps({"stations": STATIONS, "years": [args.start_year, args.end_year], "counts": counts, "failures": failures, "coverage": coverage.to_dict(orient="records")}, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "coverage": coverage.to_dict(orient="records"), "failures": failures[:5]}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
