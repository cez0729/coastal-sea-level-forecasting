from __future__ import annotations

import gzip
import io
import math
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "NOAA_NDBC_enhanced_forcing_2023_2025"
OUT_DIR = ROOT / "data" / "processed_multiyear_2023_2025"
STATION_META_PATH = OUT_DIR / "station_order.csv"

YEARS = [2023, 2024, 2025]
COOPS_PRODUCTS = ["wind", "air_pressure", "air_temperature", "water_temperature"]
APPLICATION = "SeaLevelPhysicsLossV3"
NDBC_BBOX = {"lat_min": 38.0, "lat_max": 42.5, "lon_min": -76.0, "lon_max": -71.0}


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "coops").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "ndbc_stdmet").mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_bytes(url: str, timeout: int = 45, tries: int = 3) -> bytes:
    last_error: Exception | None = None
    headers = {"User-Agent": "SeaLevelPhysicsLossV3/1.0 (research data download)"}
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed after {tries} tries: {url}") from last_error


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def read_station_meta() -> pd.DataFrame:
    meta = pd.read_csv(STATION_META_PATH)
    meta["station_id"] = meta["station_id"].astype(str)
    return meta


def coops_url(station_id: str, product: str, year: int) -> str:
    return (
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
        f"begin_date={year}0101&end_date={year}1231"
        f"&station={station_id}&product={product}&time_zone=gmt"
        "&interval=h&units=metric&format=csv"
        f"&application={APPLICATION}"
    )


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9eE+\-.]", "", regex=True), errors="coerce")


def parse_coops_csv(raw: bytes, station_id: str, product: str, year: int) -> pd.DataFrame:
    text = raw.decode("utf-8", errors="ignore")
    if "Error:" in text[:500] or not text.strip() or "Date Time" not in text.splitlines()[0]:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(text))
    if "Date Time" not in df.columns:
        return pd.DataFrame()
    df["datetime"] = pd.to_datetime(df["Date Time"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df["station_id"] = station_id

    out = df[["datetime", "station_id"]].copy()
    cols = list(df.columns)
    if product == "wind":
        # CO-OPS wind CSV often has duplicate "Direction" names. Pandas renames the second one.
        speed_col = next((c for c in cols if c.lower().strip() == "speed"), None)
        gust_col = next((c for c in cols if c.lower().strip() == "gust"), None)
        direction_candidates = [c for c in cols if c.lower().startswith("direction")]
        if speed_col:
            out["coops_wind_speed"] = clean_numeric(df[speed_col])
        if gust_col:
            out["coops_wind_gust"] = clean_numeric(df[gust_col])
        if direction_candidates:
            # Prefer the numeric direction column if there are cardinal and degree columns.
            best = None
            best_non_null = -1
            for c in direction_candidates:
                numeric = clean_numeric(df[c])
                non_null = int(numeric.notna().sum())
                if non_null > best_non_null:
                    best = numeric
                    best_non_null = non_null
            out["coops_wind_direction_deg"] = best
            rad = np.deg2rad(out["coops_wind_direction_deg"].astype(float))
            spd = out.get("coops_wind_speed", pd.Series(np.nan, index=out.index)).astype(float)
            # Meteorological "from" direction converted to approximate east/north vector.
            out["coops_wind_u"] = -spd * np.sin(rad)
            out["coops_wind_v"] = -spd * np.cos(rad)
    elif product == "air_pressure":
        pressure_col = next((c for c in cols if "pressure" in c.lower()), None)
        if pressure_col:
            out["coops_air_pressure"] = clean_numeric(df[pressure_col])
    elif product == "air_temperature":
        temp_col = next((c for c in cols if "temperature" in c.lower()), None)
        if temp_col:
            out["coops_air_temperature"] = clean_numeric(df[temp_col])
    elif product == "water_temperature":
        temp_col = next((c for c in cols if "temperature" in c.lower()), None)
        if temp_col:
            out["coops_water_temperature"] = clean_numeric(df[temp_col])
    else:
        return pd.DataFrame()

    value_cols = [c for c in out.columns if c not in {"datetime", "station_id"}]
    out = out.dropna(subset=value_cols, how="all")
    out["source_product"] = product
    out["year"] = year
    return out


def download_coops(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    inventory = []
    for station_id in meta["station_id"]:
        for product in COOPS_PRODUCTS:
            product_frames = []
            for year in YEARS:
                url = coops_url(station_id, product, year)
                raw_path = RAW_DIR / "coops" / f"COOPS__{station_id}__{product}__{year}.csv"
                try:
                    raw = fetch_bytes(url)
                    raw_path.write_bytes(raw)
                    parsed = parse_coops_csv(raw, station_id, product, year)
                    status = "ok" if len(parsed) else "no_data"
                    product_frames.append(parsed)
                    inventory.append({
                        "source": "NOAA CO-OPS",
                        "station_id": station_id,
                        "product": product,
                        "year": year,
                        "status": status,
                        "rows": len(parsed),
                        "raw_path": str(raw_path.relative_to(ROOT)),
                        "url": url,
                    })
                except Exception as exc:
                    inventory.append({
                        "source": "NOAA CO-OPS",
                        "station_id": station_id,
                        "product": product,
                        "year": year,
                        "status": f"error: {type(exc).__name__}",
                        "rows": 0,
                        "raw_path": str(raw_path.relative_to(ROOT)),
                        "url": url,
                    })
            non_empty = [p for p in product_frames if not p.empty]
            product_df = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
            if not product_df.empty:
                rows.append(product_df)
    if rows:
        long = pd.concat(rows, ignore_index=True)
    else:
        long = pd.DataFrame(columns=["datetime", "station_id"])
    return long, pd.DataFrame(inventory)


def pivot_coops(long: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    hours = pd.date_range("2023-01-01 00:00:00", "2025-12-31 23:00:00", freq="h")
    frames = []
    value_cols = [
        "coops_wind_speed",
        "coops_wind_gust",
        "coops_wind_direction_deg",
        "coops_wind_u",
        "coops_wind_v",
        "coops_air_pressure",
        "coops_air_temperature",
        "coops_water_temperature",
    ]
    for sid in meta["station_id"]:
        base = pd.DataFrame({"datetime": hours, "station_id": sid})
        if not long.empty:
            sub = long[long["station_id"] == sid].copy()
            keep = ["datetime", "station_id"] + [c for c in value_cols if c in sub.columns]
            sub = sub[keep].groupby(["datetime", "station_id"], as_index=False).mean(numeric_only=True)
            base = base.merge(sub, on=["datetime", "station_id"], how="left")
        frames.append(base)
    out = pd.concat(frames, ignore_index=True)
    numeric_cols = [c for c in out.columns if c not in {"datetime", "station_id"}]
    if numeric_cols:
        out[numeric_cols] = out.groupby("station_id", group_keys=False)[numeric_cols].apply(
            lambda x: x.interpolate(limit=6, limit_direction="both")
        )
        if "coops_air_pressure" in out.columns:
            out["coops_pressure_anom"] = out["coops_air_pressure"] - out["coops_air_pressure"].mean(skipna=True)
            out["coops_pressure_tendency_3h"] = out.groupby("station_id")["coops_air_pressure"].diff(3)
        if "coops_wind_speed" in out.columns:
            out["coops_wind_speed_tendency_3h"] = out.groupby("station_id")["coops_wind_speed"].diff(3)
    return out


def download_ndbc_station_metadata() -> pd.DataFrame:
    url = "https://www.ndbc.noaa.gov/activestations.xml"
    raw = fetch_bytes(url)
    (RAW_DIR / "ndbc_activestations.xml").write_bytes(raw)
    root = ET.fromstring(raw)
    rows = []
    for station in root.findall("station"):
        try:
            sid = station.attrib["id"]
            lat = float(station.attrib["lat"])
            lon = float(station.attrib["lon"])
        except Exception:
            continue
        if (
            NDBC_BBOX["lat_min"] <= lat <= NDBC_BBOX["lat_max"]
            and NDBC_BBOX["lon_min"] <= lon <= NDBC_BBOX["lon_max"]
        ):
            rows.append({
                "ndbc_station": sid,
                "ndbc_name": station.attrib.get("name", ""),
                "ndbc_lat": lat,
                "ndbc_lon": lon,
                "owner": station.attrib.get("owner", ""),
                "pgm": station.attrib.get("pgm", ""),
            })
    return pd.DataFrame(rows)


def list_ndbc_historical_files() -> set[str]:
    url = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
    raw = fetch_bytes(url)
    (RAW_DIR / "ndbc_stdmet_index.html").write_bytes(raw)
    html = raw.decode("utf-8", errors="ignore")
    return set(re.findall(r'href="([^"]+h20(?:23|24|25)\.txt\.gz)"', html))


def parse_ndbc_stdmet(raw: bytes, station_id: str, year: int) -> pd.DataFrame:
    try:
        text = gzip.decompress(raw).decode("utf-8", errors="ignore")
    except gzip.BadGzipFile:
        text = raw.decode("utf-8", errors="ignore")
    if not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", comment=None)
    # Header can use #YY or YY depending on file.
    df.columns = [c.lstrip("#") for c in df.columns]
    required = {"YY", "MM", "DD", "hh"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    minute = df["mm"] if "mm" in df.columns else 0
    df["datetime"] = pd.to_datetime(
        {
            "year": clean_numeric(df["YY"]).astype("Int64") + 2000,
            "month": clean_numeric(df["MM"]).astype("Int64"),
            "day": clean_numeric(df["DD"]).astype("Int64"),
            "hour": clean_numeric(df["hh"]).astype("Int64"),
            "minute": clean_numeric(pd.Series(minute)).fillna(0).astype("Int64"),
        },
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).copy()
    out = pd.DataFrame({"datetime": df["datetime"], "ndbc_station": station_id, "year": year})
    mapping = {
        "WDIR": "ndbc_wind_direction_deg",
        "WSPD": "ndbc_wind_speed",
        "GST": "ndbc_wind_gust",
        "WVHT": "ndbc_wave_height",
        "DPD": "ndbc_dominant_wave_period",
        "APD": "ndbc_average_wave_period",
        "MWD": "ndbc_mean_wave_direction_deg",
        "PRES": "ndbc_air_pressure",
        "ATMP": "ndbc_air_temperature",
        "WTMP": "ndbc_water_temperature",
        "DEWP": "ndbc_dewpoint",
    }
    for src, dst in mapping.items():
        if src in df.columns:
            val = clean_numeric(df[src])
            val = val.mask(val >= 999)
            out[dst] = val
    value_cols = [c for c in out.columns if c not in {"datetime", "ndbc_station", "year"}]
    out = out.dropna(subset=value_cols, how="all")
    if "ndbc_wind_speed" in out.columns and "ndbc_wind_direction_deg" in out.columns:
        rad = np.deg2rad(out["ndbc_wind_direction_deg"].astype(float))
        spd = out["ndbc_wind_speed"].astype(float)
        out["ndbc_wind_u"] = -spd * np.sin(rad)
        out["ndbc_wind_v"] = -spd * np.cos(rad)
    if "ndbc_wave_height" in out.columns:
        out["ndbc_wave_energy"] = out["ndbc_wave_height"] ** 2
    return out


def download_ndbc(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ndbc_meta = download_ndbc_station_metadata()
    available_files = list_ndbc_historical_files()
    inventory = []
    frames = []
    for _, station in ndbc_meta.iterrows():
        sid = station["ndbc_station"]
        for year in YEARS:
            filename = f"{sid}h{year}.txt.gz"
            if filename not in available_files:
                inventory.append({
                    "source": "NOAA NDBC",
                    "ndbc_station": sid,
                    "year": year,
                    "status": "missing_from_historical_index",
                    "rows": 0,
                    "url": f"https://www.ndbc.noaa.gov/data/historical/stdmet/{filename}",
                })
                continue
            url = f"https://www.ndbc.noaa.gov/data/historical/stdmet/{filename}"
            raw_path = RAW_DIR / "ndbc_stdmet" / filename
            try:
                raw = fetch_bytes(url)
                raw_path.write_bytes(raw)
                parsed = parse_ndbc_stdmet(raw, sid, year)
                parsed = parsed.merge(ndbc_meta, on="ndbc_station", how="left")
                frames.append(parsed)
                inventory.append({
                    "source": "NOAA NDBC",
                    "ndbc_station": sid,
                    "year": year,
                    "status": "ok" if len(parsed) else "no_data",
                    "rows": len(parsed),
                    "raw_path": str(raw_path.relative_to(ROOT)),
                    "url": url,
                })
            except Exception as exc:
                inventory.append({
                    "source": "NOAA NDBC",
                    "ndbc_station": sid,
                    "year": year,
                    "status": f"error: {type(exc).__name__}",
                    "rows": 0,
                    "raw_path": str(raw_path.relative_to(ROOT)),
                    "url": url,
                })
    long = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    return long, ndbc_meta, pd.DataFrame(inventory)


def make_nearest_ndbc_station_hourly(long: pd.DataFrame, ndbc_meta: pd.DataFrame, station_meta: pd.DataFrame) -> pd.DataFrame:
    hours = pd.date_range("2023-01-01 00:00:00", "2025-12-31 23:00:00", freq="h")
    if long.empty or ndbc_meta.empty:
        return pd.concat([pd.DataFrame({"datetime": hours, "station_id": sid}) for sid in station_meta["station_id"]], ignore_index=True)

    value_cols = [c for c in long.columns if c.startswith("ndbc_") and c not in {"ndbc_station", "ndbc_name", "ndbc_lat", "ndbc_lon"}]
    long = (
        long[["datetime", "ndbc_station"] + value_cols]
        .groupby(["datetime", "ndbc_station"], as_index=False)
        .mean(numeric_only=True)
    )
    nearest_rows = []
    for _, st in station_meta.iterrows():
        candidates = []
        for _, buoy in ndbc_meta.iterrows():
            dist = haversine_km(st["lat"], st["lon"], buoy["ndbc_lat"], buoy["ndbc_lon"])
            candidates.append((dist, buoy["ndbc_station"]))
        candidates = sorted(candidates)
        # Use up to three nearest buoys; if the nearest is sparse, the next ones can fill gaps.
        nearest_rows.append({
            "station_id": st["station_id"],
            "nearest_ndbc_1": candidates[0][1] if len(candidates) > 0 else "",
            "nearest_ndbc_1_distance_km": candidates[0][0] if len(candidates) > 0 else np.nan,
            "nearest_ndbc_2": candidates[1][1] if len(candidates) > 1 else "",
            "nearest_ndbc_2_distance_km": candidates[1][0] if len(candidates) > 1 else np.nan,
            "nearest_ndbc_3": candidates[2][1] if len(candidates) > 2 else "",
            "nearest_ndbc_3_distance_km": candidates[2][0] if len(candidates) > 2 else np.nan,
        })
    nearest = pd.DataFrame(nearest_rows)

    frames = []
    for _, row in nearest.iterrows():
        sid = row["station_id"]
        base = pd.DataFrame({"datetime": hours, "station_id": sid})
        selected = [row["nearest_ndbc_1"], row["nearest_ndbc_2"], row["nearest_ndbc_3"]]
        selected = [x for x in selected if isinstance(x, str) and x]
        pieces = []
        for rank, buoy_id in enumerate(selected, start=1):
            sub = long[long["ndbc_station"] == buoy_id].copy()
            if sub.empty:
                continue
            sub = sub.drop(columns=["ndbc_station"])
            rename = {c: f"{c}_r{rank}" for c in value_cols if c in sub.columns}
            sub = sub.rename(columns=rename)
            pieces.append(sub)
        for piece in pieces:
            base = base.merge(piece, on="datetime", how="left")
        for col in value_cols:
            rank_cols = [f"{col}_r{r}" for r in range(1, 4) if f"{col}_r{r}" in base.columns]
            if rank_cols:
                base[col] = base[rank_cols].bfill(axis=1).iloc[:, 0]
        keep = ["datetime", "station_id"] + value_cols
        base = base[[c for c in keep if c in base.columns]]
        base = base.merge(nearest[nearest["station_id"] == sid], on="station_id", how="left")
        frames.append(base)
    out = pd.concat(frames, ignore_index=True)
    numeric_cols = [c for c in out.columns if c.startswith("ndbc_") and c not in {"ndbc_station", "ndbc_name"}]
    out[numeric_cols] = out.groupby("station_id", group_keys=False)[numeric_cols].apply(
        lambda x: x.interpolate(limit=6, limit_direction="both")
    )
    if "ndbc_air_pressure" in out.columns:
        out["ndbc_pressure_anom"] = out["ndbc_air_pressure"] - out["ndbc_air_pressure"].mean(skipna=True)
        out["ndbc_pressure_tendency_3h"] = out.groupby("station_id")["ndbc_air_pressure"].diff(3)
    if "ndbc_wind_speed" in out.columns:
        out["ndbc_wind_speed_tendency_3h"] = out.groupby("station_id")["ndbc_wind_speed"].diff(3)
    if "ndbc_wave_height" in out.columns and "ndbc_dominant_wave_period" in out.columns:
        out["ndbc_wave_energy_flux"] = out["ndbc_wave_height"] ** 2 * out["ndbc_dominant_wave_period"]
    return out


def summarize_completeness(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    value_cols = [c for c in df.columns if c not in {"datetime", "station_id"} and not c.endswith("_distance_km") and not c.startswith("nearest_")]
    for col in value_cols:
        rows.append({
            "dataset": prefix,
            "column": col,
            "non_missing": int(df[col].notna().sum()),
            "total": int(len(df)),
            "coverage": float(df[col].notna().mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    station_meta = read_station_meta()

    print("Downloading NOAA CO-OPS station meteorology...")
    coops_long, coops_inventory = download_coops(station_meta)
    coops_long.to_csv(OUT_DIR / "noaa_coops_met_long.csv", index=False)
    coops_hourly = pivot_coops(coops_long, station_meta)
    coops_hourly.to_csv(OUT_DIR / "noaa_coops_met_station_hourly.csv", index=False)

    print("Downloading NOAA NDBC offshore buoy observations...")
    ndbc_long, ndbc_meta, ndbc_inventory = download_ndbc(station_meta)
    ndbc_meta.to_csv(OUT_DIR / "ndbc_candidate_buoys_metadata.csv", index=False)
    ndbc_long.to_csv(OUT_DIR / "ndbc_stdmet_long.csv", index=False)
    ndbc_hourly = make_nearest_ndbc_station_hourly(ndbc_long, ndbc_meta, station_meta)
    ndbc_hourly.to_csv(OUT_DIR / "ndbc_nearest_buoy_station_hourly.csv", index=False)

    inventory = pd.concat([coops_inventory, ndbc_inventory], ignore_index=True)
    inventory.to_csv(OUT_DIR / "enhanced_forcing_download_inventory.csv", index=False)
    completeness = pd.concat(
        [
            summarize_completeness(coops_hourly, "NOAA_COOPS_station_met"),
            summarize_completeness(ndbc_hourly, "NOAA_NDBC_nearest_buoy"),
        ],
        ignore_index=True,
    )
    completeness.to_csv(OUT_DIR / "enhanced_forcing_completeness_summary.csv", index=False)

    print("Saved enhanced forcing datasets:")
    print(f"  {OUT_DIR / 'noaa_coops_met_station_hourly.csv'} rows={len(coops_hourly)}")
    print(f"  {OUT_DIR / 'ndbc_nearest_buoy_station_hourly.csv'} rows={len(ndbc_hourly)}")
    print(f"  {OUT_DIR / 'enhanced_forcing_download_inventory.csv'} rows={len(inventory)}")
    print("\nBest coverage columns:")
    if not completeness.empty:
        print(completeness.sort_values("coverage", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
