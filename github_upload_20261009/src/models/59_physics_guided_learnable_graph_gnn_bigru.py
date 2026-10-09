import argparse
import json
import math
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import netcdf_file
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# ============================================================
# NOAA stations
# ============================================================
STATION_IDS = [
    "8461490",
    "8510560",
    "8516945",
    "8518750",
    "8531680",
    "8534720",
    "8536110",
]


# ============================================================
# Seed
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Basic utilities
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    return radius * 2.0 * np.arcsin(np.sqrt(a))


def haar_denoise(values: np.ndarray, levels: int = 2, threshold_scale: float = 0.35):
    x = np.asarray(values, dtype=np.float64)
    n_original = len(x)

    n_power = 1 << int(np.ceil(np.log2(max(2, n_original))))
    padded = np.pad(x, (0, n_power - n_original), mode="edge")

    coeffs = []
    current = padded

    for _ in range(levels):
        if len(current) < 2:
            break

        avg = (current[0::2] + current[1::2]) / math.sqrt(2.0)
        detail = (current[0::2] - current[1::2]) / math.sqrt(2.0)

        coeffs.append(detail)
        current = avg

    if coeffs:
        sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
        threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(len(padded)))

        coeffs = [
            np.sign(c) * np.maximum(np.abs(c) - threshold, 0.0)
            for c in coeffs
        ]

    for detail in reversed(coeffs):
        up = np.empty(detail.size * 2, dtype=np.float64)
        up[0::2] = (current + detail) / math.sqrt(2.0)
        up[1::2] = (current - detail) / math.sqrt(2.0)
        current = up

    return current[:n_original]


def find_dir(root: Path, keywords):
    if isinstance(keywords, str):
        keywords = [keywords]

    if not root.exists():
        return None

    candidates = []

    for p in root.rglob("*"):
        if p.is_dir():
            name = p.name.lower()
            if all(k.lower() in name for k in keywords):
                candidates.append(p)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: len(str(x)))
    return candidates[0]


def find_file(root: Path, patterns):
    if root is None:
        return None

    if isinstance(patterns, str):
        patterns = [patterns]

    files = []

    for pattern in patterns:
        files.extend(root.rglob(pattern))

    files = [p for p in files if p.is_file()]

    if not files:
        return None

    files = sorted(files, key=lambda x: len(str(x)))
    return files[0]


# ============================================================
# Dataset path
# ============================================================
def resolve_dataset_root(data_root: Path) -> Path:
    cwd = Path.cwd()

    candidates = [
        data_root,
        data_root / "raw",
        data_root / "data" / "raw",
        data_root / "海平面预测数据",
        data_root / "sea_level_data" / "海平面预测数据",
        cwd,
        cwd / "data" / "raw",
        cwd / ".." / "data" / "raw",
        cwd / "海平面预测数据",
        cwd / "sea_level_data",
        cwd / "sea_level_data" / "海平面预测数据",
    ]

    checked = []

    for root in candidates:
        root = root.resolve()
        checked.append(str(root))

        meta_dir = root / "站台元数据"
        hr_dir = root / "2025小时级实测水位"
        pr_dir = root / "2025 年小时级天文潮"

        if meta_dir.exists() and hr_dir.exists() and pr_dir.exists():
            return root

        meta_dir_auto = find_dir(root, "站台元数据")
        hr_dir_auto = find_dir(root, "实测水位")
        pr_dir_auto = find_dir(root, "天文潮")

        if meta_dir_auto and hr_dir_auto and pr_dir_auto:
            common_parent = meta_dir_auto.parent

            if hr_dir_auto.parent == common_parent and pr_dir_auto.parent == common_parent:
                return common_parent

    raise FileNotFoundError(
        "找不到数据集根目录。代码需要找到这些文件夹：\n"
        "  站台元数据\n"
        "  2025小时级实测水位\n"
        "  2025 年小时级天文潮\n\n"
        "已经检查过这些路径：\n"
        + "\n".join(f"  {p}" for p in checked)
    )


def get_dataset_paths(data_root: Path):
    root = resolve_dataset_root(data_root)

    meta_dir = root / "站台元数据"
    hr_dir = root / "2025小时级实测水位"
    pr_dir = root / "2025 年小时级天文潮"

    if not meta_dir.exists():
        meta_dir = find_dir(root, "站台元数据")
    if not hr_dir.exists():
        hr_dir = find_dir(root, "实测水位")
    if not pr_dir.exists():
        pr_dir = find_dir(root, "天文潮")

    era5_dir = find_dir(root, "era5")
    gebco_dir = find_dir(root, "gebco")

    if era5_dir is None:
        raise FileNotFoundError(f"找不到 ERA5 文件夹，请检查：{root}")

    if gebco_dir is None:
        print(f"Warning: 找不到 GEBCO 文件夹，将直接在 root 下递归搜索 GEBCO .nc 文件：{root}")

    era5_path = find_file(
        era5_dir,
        [
            "*.nc",
            "*.nc4",
            "*.h5",
            "*.hdf5",
        ],
    )

    gebco_path = find_file(
        root,
        [
            "*gebco*.nc",
            "*GEBCO*.nc",
            "*gebco*.nc4",
            "*GEBCO*.nc4",
        ],
    )

    wave_path = find_file(
        root,
        [
            "*海浪*.nc",
            "*wave*.nc",
            "*Wave*.nc",
            "*WAVE*.nc",
        ],
    )

    typhoon_path = find_file(
        root,
        [
            "*台风*.csv",
            "*typhoon*.csv",
            "*Typhoon*.csv",
            "*TYPHOON*.csv",
        ],
    )

    required = {
        "root": root,
        "meta_dir": meta_dir,
        "hr_dir": hr_dir,
        "pr_dir": pr_dir,
        "era5_path": era5_path,
        "gebco_path": gebco_path,
        "wave_path": wave_path,
        "typhoon_path": typhoon_path,
    }

    for key, value in required.items():
        if value is None or not Path(value).exists():
            raise FileNotFoundError(f"{key} 不存在：{value}")

    print("\nUsing dataset paths:")
    for key, value in required.items():
        print(f"  {key}: {value}")

    return required


# ============================================================
# Read raw data
# ============================================================
def read_station_metadata(meta_dir: Path):
    rows = []

    for station_id in STATION_IDS:
        path = meta_dir / f"{station_id}.json"

        if not path.exists():
            path = find_file(meta_dir, f"*{station_id}*.json")

        if path is None or not path.exists():
            raise FileNotFoundError(f"找不到站点元数据：{station_id}.json in {meta_dir}")

        with path.open("r", encoding="utf-8") as f:
            station = json.load(f)["stations"][0]

        rows.append(
            {
                "station_id": station_id,
                "name": station["name"],
                "lat": float(station["lat"]),
                "lon": float(station["lng"]),
            }
        )

    return pd.DataFrame(rows)


def read_water_and_tide(hr_dir: Path, pr_dir: Path, station_id: str):
    water_path = hr_dir / f"CO-OPS__{station_id}__hr.csv"
    tide_path = pr_dir / f"CO-OPS__{station_id}__pr.csv"

    if not water_path.exists():
        water_path = find_file(hr_dir, f"*{station_id}*hr*.csv")

    if not tide_path.exists():
        tide_path = find_file(pr_dir, f"*{station_id}*pr*.csv")

    if water_path is None or not water_path.exists():
        raise FileNotFoundError(f"找不到实测水位 CSV：station={station_id}, dir={hr_dir}")

    if tide_path is None or not tide_path.exists():
        raise FileNotFoundError(f"找不到天文潮 CSV：station={station_id}, dir={pr_dir}")

    water = pd.read_csv(water_path)
    tide = pd.read_csv(tide_path)

    water.columns = [c.strip() for c in water.columns]
    tide.columns = [c.strip() for c in tide.columns]

    water["time"] = pd.to_datetime(water["Date Time"])
    tide["time"] = pd.to_datetime(tide["Date Time"])

    df = water[["time", "Water Level", "Sigma", "I"]].merge(
        tide[["time", "Prediction"]],
        on="time",
        how="inner",
    )

    df = df.rename(
        columns={
            "Water Level": "water_level",
            "Prediction": "tide",
            "Sigma": "sigma",
            "I": "quality_i",
        }
    )

    for col in ["water_level", "tide", "sigma", "quality_i"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["residual"] = df["water_level"] - df["tide"]
    df["residual_wavelet"] = haar_denoise(df["residual"].to_numpy(), levels=2)

    return df


def nearest_grid_series_h5(nc_path: Path, station_meta: pd.DataFrame, variables, time_key: str):
    out = {sid: None for sid in station_meta["station_id"]}

    with h5py.File(nc_path, "r") as f:
        if time_key not in f:
            raise KeyError(
                f"{nc_path} 里面找不到时间变量：{time_key}，实际变量有：{list(f.keys())}"
            )

        times = pd.to_datetime(f[time_key][:], unit="s", utc=True).tz_localize(None)

        lats = f["latitude"][:]
        lons = f["longitude"][:]

        lon_grid, lat_grid = np.meshgrid(lons, lats)

        for _, row in station_meta.iterrows():
            data = {"time": times}

            for var in variables:
                if var not in f:
                    raise KeyError(
                        f"{nc_path} 里面找不到变量：{var}，实际变量有：{list(f.keys())}"
                    )

                arr = f[var][:]

                valid = np.isfinite(arr).any(axis=0)

                if valid.any():
                    dist = haversine_km(
                        row["lat"],
                        row["lon"],
                        lat_grid,
                        lon_grid,
                    )

                    dist = np.where(valid, dist, np.inf)

                    lat_idx, lon_idx = np.unravel_index(
                        np.argmin(dist),
                        dist.shape,
                    )
                else:
                    lat_idx = int(np.argmin(np.abs(lats - row["lat"])))
                    lon_idx = int(np.argmin(np.abs(lons - row["lon"])))

                data[var.lower()] = arr[:, lat_idx, lon_idx].astype(np.float64)

            out[row["station_id"]] = pd.DataFrame(data)

    return out


def read_gebco_depth(gebco_path: Path, station_meta: pd.DataFrame):
    depths = {}

    with netcdf_file(str(gebco_path), "r", mmap=False) as f:
        lats = f.variables["lat"].data.copy()
        lons = f.variables["lon"].data.copy()
        elev = f.variables["elevation"].data

        for _, row in station_meta.iterrows():
            lat_idx = int(np.argmin(np.abs(lats - row["lat"])))
            lon_idx = int(np.argmin(np.abs(lons - row["lon"])))

            depths[row["station_id"]] = float(elev[lat_idx, lon_idx])

    return depths


def read_typhoon_features(typhoon_path: Path, station_meta: pd.DataFrame):
    usecols = [
        "SEASON",
        "BASIN",
        "ISO_TIME",
        "LAT",
        "LON",
        "USA_WIND",
        "USA_PRES",
        "WMO_WIND",
        "WMO_PRES",
    ]

    ty = pd.read_csv(typhoon_path, usecols=usecols, low_memory=False)

    ty["ISO_TIME"] = pd.to_datetime(
        ty["ISO_TIME"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    ty = ty[
        (ty["SEASON"] == 2025)
        & (ty["BASIN"].astype(str).str.strip() == "NA")
    ].copy()

    ty["lat"] = pd.to_numeric(ty["LAT"], errors="coerce")
    ty["lon"] = pd.to_numeric(ty["LON"], errors="coerce")

    ty["wind"] = pd.to_numeric(ty["USA_WIND"], errors="coerce").fillna(
        pd.to_numeric(ty["WMO_WIND"], errors="coerce")
    )

    ty["pres"] = pd.to_numeric(ty["USA_PRES"], errors="coerce").fillna(
        pd.to_numeric(ty["WMO_PRES"], errors="coerce")
    )

    ty = ty.dropna(subset=["ISO_TIME", "lat", "lon"])

    full_index = pd.date_range("2025-01-01 00:00", "2025-12-31 23:00", freq="h")

    out = {}

    if ty.empty:
        for sid in station_meta["station_id"]:
            out[sid] = pd.DataFrame(
                {
                    "time": full_index,
                    "ty_dist": 9999.0,
                    "ty_wind": 0.0,
                    "ty_pres": 0.0,
                }
            )

        return out

    for _, station in station_meta.iterrows():
        distances = haversine_km(
            station["lat"],
            station["lon"],
            ty["lat"].to_numpy(),
            ty["lon"].to_numpy(),
        )

        tmp = ty[["ISO_TIME", "wind", "pres"]].copy()
        tmp["ty_dist"] = distances

        tmp = tmp.sort_values(["ISO_TIME", "ty_dist"]).drop_duplicates("ISO_TIME")
        tmp = tmp.rename(
            columns={
                "ISO_TIME": "time",
                "wind": "ty_wind",
                "pres": "ty_pres",
            }
        )

        tmp = tmp.set_index("time").sort_index()
        tmp = tmp.reindex(full_index)
        tmp = tmp.interpolate(method="time", limit=3)
        tmp = tmp.ffill(limit=3).bfill(limit=3)

        tmp["ty_dist"] = tmp["ty_dist"].fillna(9999.0)
        tmp["ty_wind"] = tmp["ty_wind"].fillna(0.0)
        tmp["ty_pres"] = tmp["ty_pres"].fillna(0.0)

        tmp.loc[tmp["ty_dist"] > 800.0, ["ty_wind", "ty_pres"]] = 0.0

        tmp = tmp.reset_index().rename(columns={"index": "time"})

        out[station["station_id"]] = tmp

    return out


def build_node_frames(data_root: Path):
    paths = get_dataset_paths(data_root)

    station_meta = read_station_metadata(paths["meta_dir"])

    print("\nReading wave data...")
    wave = nearest_grid_series_h5(
        paths["wave_path"],
        station_meta,
        ["VHM0", "VTPK", "VTM10", "VMDR"],
        "time",
    )

    print("Reading ERA5 data...")
    era5 = nearest_grid_series_h5(
        paths["era5_path"],
        station_meta,
        ["u10", "v10", "msl"],
        "valid_time",
    )

    print("Reading GEBCO elevation...")
    gebco = read_gebco_depth(paths["gebco_path"], station_meta)

    print("Reading typhoon features...")
    typhoon = read_typhoon_features(paths["typhoon_path"], station_meta)

    frames = {}

    for station_id in STATION_IDS:
        print(f"Building station frame: {station_id}")

        df = read_water_and_tide(
            paths["hr_dir"],
            paths["pr_dir"],
            station_id,
        )

        df = df.merge(wave[station_id], on="time", how="left")
        df = df.merge(era5[station_id], on="time", how="inner")
        df = df.merge(typhoon[station_id], on="time", how="left")

        df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)

        meta = station_meta[station_meta["station_id"] == station_id].iloc[0]

        df["lat"] = float(meta["lat"])
        df["lon"] = float(meta["lon"])
        df["elevation"] = gebco[station_id]

        df = df.set_index("time").sort_index()

        for col in ["vhm0", "vtpk", "vtm10", "vmdr"]:
            if col in df.columns:
                df[col] = df[col].interpolate(method="time")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0.0)

        frames[station_id] = df.reset_index()

    return STATION_IDS, frames, station_meta


# ============================================================
# Feature groups
# ============================================================
def get_feature_groups():
    return {
        "residual_only": [
            "residual",
        ],

        "meteo_core": [
            "residual",
            "u10",
            "v10",
            "msl",
        ],

        "meteo_wave": [
            "residual",
            "u10",
            "v10",
            "msl",
            "vhm0",
        ],

        "meteo_wave_depth": [
            "residual",
            "u10",
            "v10",
            "msl",
            "vhm0",
            "elevation",
        ],

        "full_no_static": [
            "residual",
            "residual_wavelet",
            "u10",
            "v10",
            "wind_speed",
            "msl",
            "vhm0",
            "vtpk",
            "vtm10",
            "vmdr",
            "ty_dist",
            "ty_wind",
            "ty_pres",
            "elevation",
        ],

        "full": [
            "residual",
            "residual_wavelet",
            "tide",
            "sigma",
            "u10",
            "v10",
            "wind_speed",
            "msl",
            "vhm0",
            "vtpk",
            "vtm10",
            "vmdr",
            "ty_dist",
            "ty_wind",
            "ty_pres",
            "lat",
            "lon",
            "elevation",
        ],
    }


def get_selected_feature_group_by_horizon(horizon):
    selected = {
        1: "full_no_static",
        12: "meteo_wave_depth",
        24: "meteo_wave",
    }

    if horizon not in selected:
        raise ValueError(f"Only support selected configs for horizons 1, 12, 24. Got {horizon}")

    return selected[horizon]


def get_graph_prior_by_horizon(horizon):
    if horizon == 1:
        return {
            "identity": 0.90,
            "distance": 0.05,
            "corr": 0.05,
        }

    if horizon == 12:
        return {
            "identity": 0.15,
            "distance": 0.75,
            "corr": 0.10,
        }

    if horizon == 24:
        return {
            "identity": 0.15,
            "distance": 0.10,
            "corr": 0.75,
        }

    raise ValueError(f"Only support horizons 1, 12, 24. Got {horizon}")


def get_physics_graph_name_by_horizon(horizon):
    if horizon == 1:
        return "identity"
    if horizon == 12:
        return "distance"
    if horizon == 24:
        return "corr"

    raise ValueError(f"Only support horizons 1, 12, 24. Got {horizon}")


# ============================================================
# Arrays
# ============================================================
def make_model_arrays(frames, feature_cols):
    common_times = None

    for sid in STATION_IDS:
        times = set(frames[sid]["time"])
        common_times = times if common_times is None else common_times & times

    common_times = sorted(common_times)

    if len(common_times) == 0:
        raise ValueError("没有找到 7 个站点共同时间。请检查数据时间范围。")

    features = []
    residuals = []
    tides = []

    for sid in STATION_IDS:
        df = frames[sid].set_index("time").loc[common_times]

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"站点 {sid} 缺少特征列：{missing}")

        features.append(df[feature_cols].to_numpy(dtype=np.float32))
        residuals.append(df["residual"].to_numpy(dtype=np.float32))
        tides.append(df["tide"].to_numpy(dtype=np.float32))

    x = np.stack(features, axis=1).astype(np.float32)
    residual = np.stack(residuals, axis=1).astype(np.float32)
    tide = np.stack(tides, axis=1).astype(np.float32)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
    tide = np.nan_to_num(tide, nan=0.0, posinf=0.0, neginf=0.0)

    return np.array(common_times), x, residual, tide


def make_physics_arrays(frames):
    common_times = None

    for sid in STATION_IDS:
        times = set(frames[sid]["time"])
        common_times = times if common_times is None else common_times & times

    common_times = sorted(common_times)

    required_cols = [
        "residual",
        "tide",
        "u10",
        "v10",
        "msl",
        "vhm0",
        "elevation",
    ]

    arrays = {col: [] for col in required_cols}

    for sid in STATION_IDS:
        df = frames[sid].set_index("time").loc[common_times]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"站点 {sid} 缺少物理模型列：{missing}")

        for col in required_cols:
            arrays[col].append(df[col].to_numpy(dtype=np.float32))

    out = {}

    for col in required_cols:
        out[col] = np.stack(arrays[col], axis=1).astype(np.float32)
        out[col] = np.nan_to_num(out[col], nan=0.0, posinf=0.0, neginf=0.0)

    out["time"] = np.array(common_times)
    out["wind_speed"] = np.sqrt(out["u10"] ** 2 + out["v10"] ** 2).astype(np.float32)
    out["wind_stress_u"] = (out["u10"] * out["wind_speed"]).astype(np.float32)
    out["wind_stress_v"] = (out["v10"] * out["wind_speed"]).astype(np.float32)

    return out


# ============================================================
# Graphs
# ============================================================
def row_normalize(adj):
    adj = adj.astype(np.float64)
    row_sum = adj.sum(axis=1, keepdims=True)
    return (adj / (row_sum + 1e-8)).astype(np.float32)


def sym_normalize_adj(adj):
    adj = adj.astype(np.float64)
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt
    return adj_norm.astype(np.float32)


def build_graphs(station_meta: pd.DataFrame, residual: np.ndarray):
    n = len(STATION_IDS)

    coords = (
        station_meta
        .set_index("station_id")
        .loc[STATION_IDS][["lat", "lon"]]
        .to_numpy()
    )

    dist = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            dist[i, j] = haversine_km(
                coords[i, 0],
                coords[i, 1],
                coords[j, 0],
                coords[j, 1],
            )

    sigma = np.median(dist[dist > 0])
    distance_adj = np.exp(-dist / sigma)
    np.fill_diagonal(distance_adj, 1.0)

    corr = np.nan_to_num(np.corrcoef(residual.T), nan=0.0)
    corr_adj = np.maximum(corr, 0.0)
    np.fill_diagonal(corr_adj, 1.0)

    identity_adj = np.eye(n, dtype=np.float64)
    fusion_adj = 0.5 * distance_adj + 0.5 * corr_adj
    np.fill_diagonal(fusion_adj, 1.0)

    physics_graphs = {
        "identity": row_normalize(identity_adj),
        "distance": row_normalize(distance_adj),
        "corr": row_normalize(corr_adj),
        "fusion": row_normalize(fusion_adj),
    }

    gnn_graphs = {
        "identity": sym_normalize_adj(identity_adj),
        "distance": sym_normalize_adj(distance_adj),
        "corr": sym_normalize_adj(corr_adj),
    }

    return physics_graphs, gnn_graphs


# ============================================================
# Physics ODE model
# ============================================================
class PhysicsODEModel:
    """
    Physics ODE baseline:

    d eta / dt = c
                 + a eta
                 + kappa (A eta - eta)
                 + beta X

    Forecast:
    eta[t+1] = eta[t] + d eta / dt
    """

    def __init__(
        self,
        graph_adj,
        feature_terms,
        ridge_alpha=1e-2,
        dt_hours=1.0,
        clip_residual=2.0,
    ):
        self.graph_adj = graph_adj
        self.feature_terms = feature_terms
        self.ridge_alpha = ridge_alpha
        self.dt_hours = dt_hours
        self.clip_residual = clip_residual

        self.msl_ref = 101325.0
        self.scaler = StandardScaler()
        self.regressor = Ridge(alpha=ridge_alpha)
        self.fitted = False

    def _build_features_one_time(self, eta, data, t):
        features = []

        for term in self.feature_terms:
            if term == "constant":
                value = np.ones_like(eta)

            elif term == "eta":
                value = eta

            elif term == "graph_delta":
                value = self.graph_adj @ eta - eta

            elif term == "graph_eta":
                value = self.graph_adj @ eta

            elif term == "u10":
                value = data["u10"][t]

            elif term == "v10":
                value = data["v10"][t]

            elif term == "wind_speed":
                value = data["wind_speed"][t]

            elif term == "wind_stress_u":
                value = data["wind_stress_u"][t]

            elif term == "wind_stress_v":
                value = data["wind_stress_v"][t]

            elif term == "pressure_anom":
                value = data["msl"][t] - self.msl_ref

            elif term == "vhm0":
                value = data["vhm0"][t]

            elif term == "elevation":
                value = data["elevation"][t]

            elif term == "inverse_depth":
                depth = np.maximum(-data["elevation"][t], 1.0)
                value = 1.0 / depth

            else:
                raise ValueError(f"Unknown physics term: {term}")

            features.append(np.asarray(value, dtype=np.float32)[:, None])

        return np.concatenate(features, axis=1).astype(np.float32)

    def fit(self, data, train_end):
        residual = data["residual"]

        self.msl_ref = float(np.mean(data["msl"][:train_end]))

        x_rows = []
        y_rows = []

        for t in range(0, train_end - 1):
            eta_t = residual[t]
            eta_next = residual[t + 1]

            d_eta = (eta_next - eta_t) / self.dt_hours

            x_t = self._build_features_one_time(
                eta=eta_t,
                data=data,
                t=t,
            )

            x_rows.append(x_t)
            y_rows.append(d_eta[:, None])

        x_train = np.concatenate(x_rows, axis=0)
        y_train = np.concatenate(y_rows, axis=0).reshape(-1)

        x_train_scaled = self.scaler.fit_transform(x_train)

        self.regressor.fit(x_train_scaled, y_train)
        self.fitted = True

        return self

    def derivative(self, eta, data, t):
        if not self.fitted:
            raise RuntimeError("PhysicsODEModel is not fitted.")

        x = self._build_features_one_time(
            eta=eta,
            data=data,
            t=t,
        )

        x_scaled = self.scaler.transform(x)
        d_eta = self.regressor.predict(x_scaled)

        return d_eta.astype(np.float32)

    def step(self, eta, data, t):
        d_eta = self.derivative(eta, data, t)
        eta_next = eta + self.dt_hours * d_eta

        if self.clip_residual is not None and self.clip_residual > 0:
            eta_next = np.clip(
                eta_next,
                -abs(self.clip_residual),
                abs(self.clip_residual),
            )

        return eta_next.astype(np.float32)

    def forecast_sequence(self, data, start_t, horizon):
        residual = data["residual"]

        eta = residual[start_t - 1].copy()
        preds = []

        for s in range(horizon):
            forcing_t = start_t + s - 1

            if forcing_t < 0:
                forcing_t = 0

            if forcing_t >= len(residual):
                forcing_t = len(residual) - 1

            eta = self.step(
                eta=eta,
                data=data,
                t=forcing_t,
            )

            preds.append(eta.copy())

        pred = np.stack(preds, axis=1)

        return pred.astype(np.float32)

    def coefficients_table(self):
        return pd.DataFrame(
            {
                "term": self.feature_terms,
                "coef_scaled_feature": self.regressor.coef_,
            }
        )


def build_physics_model(physics_graph, ridge_alpha, clip_residual):
    feature_terms = [
        "constant",
        "eta",
        "graph_delta",
        "wind_stress_u",
        "wind_stress_v",
        "pressure_anom",
        "vhm0",
        "elevation",
        "inverse_depth",
    ]

    return PhysicsODEModel(
        graph_adj=physics_graph,
        feature_terms=feature_terms,
        ridge_alpha=ridge_alpha,
        dt_hours=1.0,
        clip_residual=clip_residual,
    )


def make_indices(n_time, window, horizon, start, end):
    return np.arange(start + window, end - horizon + 1)


def precompute_physics_forecasts(physics_model, physics_data, indices, horizon):
    preds = []

    for t in indices:
        pred = physics_model.forecast_sequence(
            data=physics_data,
            start_t=int(t),
            horizon=horizon,
        )
        preds.append(pred[None, :, :])

    return np.concatenate(preds, axis=0).astype(np.float32)


# ============================================================
# Dataset
# ============================================================
class PhysicsGuidedDataset(Dataset):
    def __init__(
        self,
        x,
        residual,
        tide,
        window,
        horizon,
        indices,
        physics_pred,
    ):
        self.x = x
        self.residual = residual
        self.tide = tide
        self.window = window
        self.horizon = horizon
        self.indices = indices
        self.physics_pred = physics_pred

        if len(self.indices) != len(self.physics_pred):
            raise ValueError(
                f"indices length {len(self.indices)} != physics_pred length {len(self.physics_pred)}"
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = int(self.indices[idx])

        xs = self.x[t - self.window:t]
        ys = self.residual[t:t + self.horizon].T
        ts = self.tide[t:t + self.horizon].T
        ps = self.physics_pred[idx]

        return (
            torch.tensor(xs, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.float32),
            torch.tensor(ts, dtype=torch.float32),
            torch.tensor(ps, dtype=torch.float32),
        )


# ============================================================
# Learnable graph GNN-BiGRU correction model
# ============================================================
class PriorLearnableGraphFusion(nn.Module):
    def __init__(self, base_graphs: dict, prior_weights: dict, temperature: float = 1.0):
        super().__init__()

        graph_names = ["identity", "distance", "corr"]
        graph_stack = np.stack([base_graphs[name] for name in graph_names], axis=0)

        prior = np.array([prior_weights[name] for name in graph_names], dtype=np.float32)
        prior = prior / prior.sum()

        self.graph_names = graph_names
        self.temperature = temperature

        self.register_buffer(
            "graph_stack",
            torch.tensor(graph_stack, dtype=torch.float32),
        )

        self.register_buffer(
            "prior_weights",
            torch.tensor(prior, dtype=torch.float32),
        )

        init_logits = torch.log(torch.tensor(prior, dtype=torch.float32) + 1e-8)
        self.logits = nn.Parameter(init_logits)

    def forward(self):
        weights = torch.softmax(self.logits / self.temperature, dim=0)
        adj = torch.sum(weights[:, None, None] * self.graph_stack, dim=0)
        return adj

    def regularization_loss(self):
        eps = 1e-8
        weights = torch.softmax(self.logits / self.temperature, dim=0)
        prior = self.prior_weights
        kl = torch.sum(weights * (torch.log(weights + eps) - torch.log(prior + eps)))
        return kl

    def get_weights(self):
        weights = torch.softmax((self.logits / self.temperature).detach().cpu(), dim=0).numpy()
        return {
            name: float(weights[i])
            for i, name in enumerate(self.graph_names)
        }


class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        if x.shape[2] != adj.shape[0]:
            raise ValueError(
                f"Node mismatch: x has {x.shape[2]} nodes, adj has {adj.shape[0]} nodes."
            )

        x = torch.einsum("ij,btjf->btif", adj, x)
        return self.linear(x)


class PhysicsGuidedLearnableGraphGNNBiGRU(nn.Module):
    """
    Final prediction:

    y_final = y_physics + correction_GNN

    GNN-BiGRU learns only the correction term.
    The graph inside GNN is still learnable:
    A = w_id A_id + w_distance A_distance + w_corr A_corr
    """

    def __init__(
        self,
        input_dim,
        gnn_hidden,
        gru_hidden,
        horizon,
        dropout,
        base_graphs,
        prior_weights,
        graph_temperature,
    ):
        super().__init__()

        self.horizon = horizon

        self.graph_fusion = PriorLearnableGraphFusion(
            base_graphs=base_graphs,
            prior_weights=prior_weights,
            temperature=graph_temperature,
        )

        self.gcn1 = GraphConvolution(input_dim, gnn_hidden)
        self.gcn2 = GraphConvolution(gnn_hidden, gnn_hidden)

        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=gnn_hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.correction_head = nn.Sequential(
            nn.Linear(gru_hidden * 2 + horizon, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, horizon),
        )

    def forward(self, x, physics_pred):
        adj = self.graph_fusion()

        h = torch.relu(self.gcn1(x, adj))
        h = self.dropout(torch.relu(self.gcn2(h, adj)))

        bsz, steps, nodes, hidden = h.shape

        h = h.permute(0, 2, 1, 3)
        h = h.reshape(bsz * nodes, steps, hidden)

        out, _ = self.gru(h)

        last = out[:, -1, :]
        last = last.reshape(bsz, nodes, -1)

        correction_input = torch.cat([last, physics_pred], dim=-1)

        correction = self.correction_head(correction_input)

        final_pred = physics_pred + correction

        return final_pred, correction

    def graph_regularization_loss(self):
        return self.graph_fusion.regularization_loss()

    def get_graph_weights(self):
        return self.graph_fusion.get_weights()


# ============================================================
# Training helpers
# ============================================================
def build_optimizer(model, lr, graph_lr_mult, weight_decay):
    graph_params = list(model.graph_fusion.parameters())
    graph_param_ids = set(id(p) for p in graph_params)

    other_params = [
        p for p in model.parameters()
        if id(p) not in graph_param_ids
    ]

    optimizer = torch.optim.Adam(
        [
            {
                "params": other_params,
                "lr": lr,
                "weight_decay": weight_decay,
            },
            {
                "params": graph_params,
                "lr": lr * graph_lr_mult,
                "weight_decay": 0.0,
            },
        ]
    )

    return optimizer


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    graph_reg_lambda,
    correction_reg_lambda,
):
    model.train()

    total_loss = 0.0
    total_pred_loss = 0.0
    total_graph_reg = 0.0
    total_corr_reg = 0.0

    for xb, yb, _, pb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pb = pb.to(device)

        optimizer.zero_grad()

        pred, correction = model(xb, pb)

        pred_loss = criterion(pred, yb)
        graph_reg = model.graph_regularization_loss()
        correction_reg = torch.mean(correction ** 2)

        loss = (
            pred_loss
            + graph_reg_lambda * graph_reg
            + correction_reg_lambda * correction_reg
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_pred_loss += pred_loss.item() * xb.size(0)
        total_graph_reg += graph_reg.item() * xb.size(0)
        total_corr_reg += correction_reg.item() * xb.size(0)

    n = max(1, len(loader.dataset))

    return {
        "loss": total_loss / n,
        "pred_loss": total_pred_loss / n,
        "graph_reg": total_graph_reg / n,
        "correction_reg": total_corr_reg / n,
    }


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    for xb, yb, _, pb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pb = pb.to(device)

        pred, _ = model(xb, pb)
        loss = criterion(pred, yb)

        total_loss += loss.item() * xb.size(0)

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    pred_list = []
    true_list = []
    tide_list = []
    physics_list = []
    correction_list = []

    for xb, yb, tb, pb in loader:
        xb = xb.to(device)
        pb_device = pb.to(device)

        pred, correction = model(xb, pb_device)

        pred_list.append(pred.cpu().numpy())
        true_list.append(yb.numpy())
        tide_list.append(tb.numpy())
        physics_list.append(pb.numpy())
        correction_list.append(correction.cpu().numpy())

    return (
        np.concatenate(pred_list, axis=0),
        np.concatenate(true_list, axis=0),
        np.concatenate(tide_list, axis=0),
        np.concatenate(physics_list, axis=0),
        np.concatenate(correction_list, axis=0),
    )


# ============================================================
# Metrics
# ============================================================
def regression_metrics(y_true, y_pred):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    mse = mean_squared_error(yt, yp)
    mae = mean_absolute_error(yt, yp)
    rmse = math.sqrt(mse)
    r2 = r2_score(yt, yp)
    bias = float(np.mean(yp - yt))

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "R2": r2,
    }


def collect_metrics(prefix, true_residual, pred_residual, tide):
    pred_level = pred_residual + tide
    true_level = true_residual + tide

    seq_residual = regression_metrics(true_residual, pred_residual)
    seq_level = regression_metrics(true_level, pred_level)

    last_true_residual = true_residual[:, :, -1:]
    last_pred_residual = pred_residual[:, :, -1:]

    last_true_level = true_level[:, :, -1:]
    last_pred_level = pred_level[:, :, -1:]

    last_residual = regression_metrics(last_true_residual, last_pred_residual)
    last_level = regression_metrics(last_true_level, last_pred_level)

    out = {}

    for k, v in seq_residual.items():
        out[f"{prefix}_seq_residual_{k}"] = v
    for k, v in seq_level.items():
        out[f"{prefix}_seq_sea_level_{k}"] = v
    for k, v in last_residual.items():
        out[f"{prefix}_last_residual_{k}"] = v
    for k, v in last_level.items():
        out[f"{prefix}_last_sea_level_{k}"] = v

    return out


# ============================================================
# One experiment
# ============================================================
def run_one_experiment(
    x_scaled,
    residual,
    tide,
    physics_data,
    physics_graph,
    gnn_graphs,
    prior_weights,
    feature_group,
    feature_cols,
    horizon,
    window,
    epochs,
    batch_size,
    lr,
    graph_lr_mult,
    weight_decay,
    gnn_hidden,
    gru_hidden,
    dropout,
    graph_temperature,
    graph_reg_lambda,
    correction_reg_lambda,
    ridge_alpha,
    clip_residual,
    device,
    output_root,
    scheduler_patience,
    scheduler_factor,
    min_lr,
    early_stop_patience,
    min_delta,
    resume=False,
):
    print("\n" + "=" * 100)
    print("PHYSICS-GUIDED LEARNABLE GRAPH GNN-BIGRU")
    print(f"Horizon       : {horizon}h")
    print(f"Feature group : {feature_group}")
    print(f"Feature cols  : {feature_cols}")
    print(f"Prior weights : {prior_weights}")
    print(f"Physics graph : fixed ODE graph")
    print("=" * 100)

    n_time = len(residual)

    train_end = int(n_time * 0.7)
    val_end = int(n_time * 0.85)

    train_indices = make_indices(
        n_time=n_time,
        window=window,
        horizon=horizon,
        start=0,
        end=train_end,
    )

    val_indices = make_indices(
        n_time=n_time,
        window=window,
        horizon=horizon,
        start=train_end,
        end=val_end,
    )

    test_indices = make_indices(
        n_time=n_time,
        window=window,
        horizon=horizon,
        start=val_end,
        end=n_time,
    )

    print("\nFitting physics ODE model on training split only...")

    physics_model = build_physics_model(
        physics_graph=physics_graph,
        ridge_alpha=ridge_alpha,
        clip_residual=clip_residual,
    )

    physics_model.fit(
        data=physics_data,
        train_end=train_end,
    )

    exp_name = f"physics_guided_{horizon}h_{feature_group}"
    exp_dir = output_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    physics_model.coefficients_table().to_csv(
        exp_dir / "physics_ode_coefficients.csv",
        index=False,
    )

    print("Precomputing physics forecasts...")

    train_physics_pred = precompute_physics_forecasts(
        physics_model=physics_model,
        physics_data=physics_data,
        indices=train_indices,
        horizon=horizon,
    )

    val_physics_pred = precompute_physics_forecasts(
        physics_model=physics_model,
        physics_data=physics_data,
        indices=val_indices,
        horizon=horizon,
    )

    test_physics_pred = precompute_physics_forecasts(
        physics_model=physics_model,
        physics_data=physics_data,
        indices=test_indices,
        horizon=horizon,
    )

    print(
        f"Samples: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
    )

    train_ds = PhysicsGuidedDataset(
        x=x_scaled,
        residual=residual,
        tide=tide,
        window=window,
        horizon=horizon,
        indices=train_indices,
        physics_pred=train_physics_pred,
    )

    val_ds = PhysicsGuidedDataset(
        x=x_scaled,
        residual=residual,
        tide=tide,
        window=window,
        horizon=horizon,
        indices=val_indices,
        physics_pred=val_physics_pred,
    )

    test_ds = PhysicsGuidedDataset(
        x=x_scaled,
        residual=residual,
        tide=tide,
        window=window,
        horizon=horizon,
        indices=test_indices,
        physics_pred=test_physics_pred,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = PhysicsGuidedLearnableGraphGNNBiGRU(
        input_dim=x_scaled.shape[-1],
        gnn_hidden=gnn_hidden,
        gru_hidden=gru_hidden,
        horizon=horizon,
        dropout=dropout,
        base_graphs=gnn_graphs,
        prior_weights=prior_weights,
        graph_temperature=graph_temperature,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = build_optimizer(
        model=model,
        lr=lr,
        graph_lr_mult=graph_lr_mult,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )

    best_model_path = exp_dir / "best_model.pt"
    last_ckpt_path = exp_dir / "last_checkpoint.pt"
    history_path = exp_dir / "training_history.csv"

    best_val = float("inf")
    start_epoch = 1
    bad_epochs = 0
    history_rows = []

    if resume and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device)

        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])

        best_val = ckpt["best_val"]
        start_epoch = ckpt["epoch"] + 1
        bad_epochs = ckpt.get("bad_epochs", 0)

        if history_path.exists():
            history_rows = pd.read_csv(history_path).to_dict("records")

        print(
            f"Resumed from checkpoint: epoch={ckpt['epoch']}, "
            f"best_val={best_val:.6f}, bad_epochs={bad_epochs}"
        )

    for epoch in range(start_epoch, epochs + 1):
        train_info = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            graph_reg_lambda=graph_reg_lambda,
            correction_reg_lambda=correction_reg_lambda,
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)

        weights_now = model.get_graph_weights()

        main_lr_now = optimizer.param_groups[0]["lr"]
        graph_lr_now = optimizer.param_groups[1]["lr"]

        improved = val_loss < best_val - min_delta

        if improved:
            best_val = val_loss
            bad_epochs = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_val": best_val,
                    "epoch": epoch,
                    "feature_group": feature_group,
                    "horizon": horizon,
                    "graph_weights": weights_now,
                    "prior_weights": prior_weights,
                },
                best_model_path,
            )
        else:
            bad_epochs += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_total_loss": train_info["loss"],
                "train_pred_mse": train_info["pred_loss"],
                "train_graph_reg": train_info["graph_reg"],
                "train_correction_reg": train_info["correction_reg"],
                "val_mse": val_loss,
                "best_val_mse": best_val,
                "main_lr": main_lr_now,
                "graph_lr": graph_lr_now,
                "w_identity": weights_now["identity"],
                "w_distance": weights_now["distance"],
                "w_corr": weights_now["corr"],
                "bad_epochs": bad_epochs,
            }
        )

        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val": best_val,
                "bad_epochs": bad_epochs,
                "feature_group": feature_group,
                "horizon": horizon,
                "graph_weights": weights_now,
                "prior_weights": prior_weights,
            },
            last_ckpt_path,
        )

        print(
            f"[physics_guided | {horizon}h | {feature_group}] "
            f"epoch={epoch:03d} "
            f"train_mse={train_info['pred_loss']:.6f} "
            f"val_mse={val_loss:.6f} "
            f"best={best_val:.6f} "
            f"main_lr={main_lr_now:.1e} "
            f"graph_lr={graph_lr_now:.1e} "
            f"w_id={weights_now['identity']:.3f} "
            f"w_dist={weights_now['distance']:.3f} "
            f"w_corr={weights_now['corr']:.3f} "
            f"bad={bad_epochs}/{early_stop_patience}"
        )

        if bad_epochs >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best val mse={best_val:.6f}"
            )
            break

    if best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    learned_weights = model.get_graph_weights()

    pred_residual, true_residual, target_tide, physics_pred, correction_pred = predict(
        model=model,
        loader=test_loader,
        device=device,
    )

    final_metrics = collect_metrics(
        prefix="final",
        true_residual=true_residual,
        pred_residual=pred_residual,
        tide=target_tide,
    )

    physics_metrics = collect_metrics(
        prefix="physics",
        true_residual=true_residual,
        pred_residual=physics_pred,
        tide=target_tide,
    )

    correction_metrics = regression_metrics(
        true_residual - physics_pred,
        correction_pred,
    )

    row = {
        "horizon": horizon,
        "feature_group": feature_group,
        "window": window,
        "num_features": len(feature_cols),
        "features": "|".join(feature_cols),
        "best_val_mse": best_val,

        "learned_w_identity": learned_weights["identity"],
        "learned_w_distance": learned_weights["distance"],
        "learned_w_corr": learned_weights["corr"],

        "prior_w_identity": prior_weights["identity"],
        "prior_w_distance": prior_weights["distance"],
        "prior_w_corr": prior_weights["corr"],

        "ridge_alpha": ridge_alpha,
        "clip_residual": clip_residual,
    }

    row.update(physics_metrics)
    row.update(final_metrics)

    row["correction_MSE"] = correction_metrics["MSE"]
    row["correction_MAE"] = correction_metrics["MAE"]
    row["correction_RMSE"] = correction_metrics["RMSE"]
    row["correction_Bias"] = correction_metrics["Bias"]
    row["correction_R2"] = correction_metrics["R2"]

    pd.DataFrame([row]).to_csv(exp_dir / "metrics.csv", index=False)

    np.save(exp_dir / "pred_residual_final.npy", pred_residual)
    np.save(exp_dir / "true_residual.npy", true_residual)
    np.save(exp_dir / "physics_pred_residual.npy", physics_pred)
    np.save(exp_dir / "correction_pred.npy", correction_pred)
    np.save(exp_dir / "target_tide.npy", target_tide)

    print("\nLearned graph weights in correction GNN:")
    print(f"  identity : {learned_weights['identity']:.6f}")
    print(f"  distance : {learned_weights['distance']:.6f}")
    print(f"  corr     : {learned_weights['corr']:.6f}")

    print("\nPhysics ODE metrics:")
    print(f"  seq residual RMSE  : {row['physics_seq_residual_RMSE']:.6f}")
    print(f"  seq residual R2    : {row['physics_seq_residual_R2']:.6f}")
    print(f"  last residual RMSE : {row['physics_last_residual_RMSE']:.6f}")
    print(f"  last residual R2   : {row['physics_last_residual_R2']:.6f}")

    print("\nFinal physics-guided GNN metrics:")
    print(f"  seq residual RMSE  : {row['final_seq_residual_RMSE']:.6f}")
    print(f"  seq residual R2    : {row['final_seq_residual_R2']:.6f}")
    print(f"  last residual RMSE : {row['final_last_residual_RMSE']:.6f}")
    print(f"  last residual R2   : {row['final_last_residual_R2']:.6f}")

    print(f"\nSaved to: {exp_dir}")

    return row


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default="..\\data\\raw")

    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 12, 24],
    )

    parser.add_argument(
        "--feature-strategy",
        choices=["selected", "unified"],
        default="selected",
    )

    parser.add_argument(
        "--unified-feature-group",
        default="meteo_wave_depth",
    )

    parser.add_argument("--window", type=int, default=24)

    parser.add_argument("--gnn-hidden", type=int, default=64)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)

    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--graph-lr-mult", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--graph-temperature", type=float, default=1.0)
    parser.add_argument("--graph-reg-lambda", type=float, default=1e-3)

    parser.add_argument(
        "--correction-reg-lambda",
        type=float,
        default=1e-5,
        help="Small regularization on neural correction magnitude.",
    )

    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-2,
        help="Ridge alpha for fitting physics ODE coefficients.",
    )

    parser.add_argument(
        "--clip-residual",
        type=float,
        default=2.0,
        help="Clip physics recursive residual prediction. Use 0 to disable.",
    )

    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)

    parser.add_argument("--early-stop-patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-6)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output-dir",
        default="outputs/physics_guided_learnable_graph_gnn_bigru",
    )

    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    clip_residual = args.clip_residual
    if clip_residual <= 0:
        clip_residual = None

    feature_groups = get_feature_groups()

    if args.feature_strategy == "unified":
        if args.unified_feature_group not in feature_groups:
            raise ValueError(
                f"未知 unified_feature_group: {args.unified_feature_group}. "
                f"可选: {list(feature_groups.keys())}"
            )

    print("\nLoading real NOAA data once...")
    station_ids, frames, station_meta = build_node_frames(data_root)

    print("\nBuilding physics arrays...")
    physics_data = make_physics_arrays(frames)

    print("\nBuilding graphs...")
    physics_graphs, gnn_graphs = build_graphs(
        station_meta=station_meta,
        residual=physics_data["residual"],
    )

    all_rows = []

    for horizon in args.horizons:
        if args.feature_strategy == "selected":
            feature_group = get_selected_feature_group_by_horizon(horizon)
        else:
            feature_group = args.unified_feature_group

        feature_cols = feature_groups[feature_group]

        physics_graph_name = get_physics_graph_name_by_horizon(horizon)
        physics_graph = physics_graphs[physics_graph_name]

        prior_weights = get_graph_prior_by_horizon(horizon)

        print("\n" + "#" * 100)
        print(f"Horizon {horizon}h physics-guided learnable graph GNN:")
        print(f"  feature_strategy  = {args.feature_strategy}")
        print(f"  feature_group     = {feature_group}")
        print(f"  physics_graph     = {physics_graph_name}")
        print(f"  gnn_graph_prior   = {prior_weights}")
        print("#" * 100)

        times, x, residual, tide = make_model_arrays(
            frames=frames,
            feature_cols=feature_cols,
        )

        n_time, nodes, feats = x.shape

        if nodes != len(STATION_IDS):
            raise ValueError(f"x 节点数错误: {nodes}, expected={len(STATION_IDS)}")

        train_end = int(n_time * 0.7)

        scaler = StandardScaler()
        scaler.fit(x[:train_end].reshape(-1, feats))

        x_scaled = scaler.transform(
            x.reshape(-1, feats)
        ).reshape(n_time, nodes, feats).astype(np.float32)

        residual = residual.astype(np.float32)
        tide = tide.astype(np.float32)

        row = run_one_experiment(
            x_scaled=x_scaled,
            residual=residual,
            tide=tide,
            physics_data=physics_data,
            physics_graph=physics_graph,
            gnn_graphs=gnn_graphs,
            prior_weights=prior_weights,
            feature_group=feature_group,
            feature_cols=feature_cols,
            horizon=horizon,
            window=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            graph_lr_mult=args.graph_lr_mult,
            weight_decay=args.weight_decay,
            gnn_hidden=args.gnn_hidden,
            gru_hidden=args.gru_hidden,
            dropout=args.dropout,
            graph_temperature=args.graph_temperature,
            graph_reg_lambda=args.graph_reg_lambda,
            correction_reg_lambda=args.correction_reg_lambda,
            ridge_alpha=args.ridge_alpha,
            clip_residual=clip_residual,
            device=device,
            output_root=output_root,
            scheduler_patience=args.scheduler_patience,
            scheduler_factor=args.scheduler_factor,
            min_lr=args.min_lr,
            early_stop_patience=args.early_stop_patience,
            min_delta=args.min_delta,
            resume=args.resume,
        )

        all_rows.append(row)

        pd.DataFrame(all_rows).to_csv(
            output_root / "physics_guided_metrics.csv",
            index=False,
        )

    overall = pd.DataFrame(all_rows)
    overall.to_csv(output_root / "physics_guided_metrics.csv", index=False)

    print("\n" + "=" * 100)
    print("PHYSICS-GUIDED LEARNABLE GRAPH GNN-BIGRU FINISHED")
    print("=" * 100)

    display_cols = [
        "horizon",
        "feature_group",
        "physics_seq_residual_RMSE",
        "physics_seq_residual_R2",
        "physics_last_residual_RMSE",
        "physics_last_residual_R2",
        "final_seq_residual_RMSE",
        "final_seq_residual_R2",
        "final_last_residual_RMSE",
        "final_last_residual_R2",
        "learned_w_identity",
        "learned_w_distance",
        "learned_w_corr",
    ]

    print(overall[display_cols])

    print(f"\nSaved metrics to: {output_root / 'physics_guided_metrics.csv'}")


if __name__ == "__main__":
    main()