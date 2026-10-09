import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.io import netcdf_file
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


STATION_IDS = [
    "8461490",
    "8510560",
    "8516945",
    "8518750",
    "8531680",
    "8534720",
    "8536110",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return radius * 2.0 * np.arcsin(np.sqrt(a))


def haar_denoise(values: np.ndarray, levels: int = 2, threshold_scale: float = 0.35) -> np.ndarray:
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
        coeffs = [np.sign(c) * np.maximum(np.abs(c) - threshold, 0.0) for c in coeffs]

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


def resolve_dataset_root(data_root: Path) -> Path:
    """
    自动适配你的图片结构：

    情况 A:
      当前目录/
        RAO_GNN_BiGRU.py
        海平面预测数据/

    情况 B:
      当前目录/
        RAO_GNN_BiGRU.py
        sea_level_data/
          海平面预测数据/

    情况 C:
      直接传入：
        --data-root ./海平面预测数据
    """
    cwd = Path.cwd()

    candidates = [
        data_root,
        data_root / "海平面预测数据",
        data_root / "sea_level_data" / "海平面预测数据",
        cwd,
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
    """
    更稳健的数据路径搜索函数。

    这版专门修复你之前遇到的 GEBCO 搜索问题：
    你的目录里既有 “GEBCO 水深 海底地形”，又有 “GEBCO_gebco_unzip”，
    旧代码可能先找到前者但里面没有 .nc，所以 gebco_path=None。
    这里优先使用真实 .nc 文件路径；找不到再全目录递归搜索。
    """
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

    # ERA5
    era5_dir = find_dir(root, "era5")
    era5_path = None
    if era5_dir is not None:
        era5_path = find_file(era5_dir, ["*.nc", "*.nc4", "*.h5", "*.hdf5"])
    if era5_path is None:
        era5_path = find_file(root, ["*era5*.nc", "*ERA5*.nc", "*气象*.nc"])

    # GEBCO：优先写死你当前真实存在的路径，然后再兜底搜索
    gebco_path = root / "GEBCO_gebco_unzip" / "gebco_2025_n42.5_s38.0_w-76.0_e-70.0.nc"
    if not gebco_path.exists():
        gebco_path = find_file(root, ["*gebco*.nc", "*GEBCO*.nc", "*elevation*.nc", "*bathymetry*.nc"])

    # Wave
    wave_path = root / "海浪数据.nc"
    if not wave_path.exists():
        wave_path = find_file(root, ["*海浪*.nc", "*wave*.nc", "*Wave*.nc"])

    # Typhoon
    typhoon_path = root / "台风数据.csv"
    if not typhoon_path.exists():
        typhoon_path = find_file(root, ["*台风*.csv", "*typhoon*.csv", "*Typhoon*.csv"])

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

    print("Using dataset paths:")
    for key, value in required.items():
        print(f"  {key}: {value}")

    for key, value in required.items():
        if value is None or not Path(value).exists():
            raise FileNotFoundError(f"{key} 不存在：{value}")

    return required


def read_station_metadata(meta_dir: Path) -> pd.DataFrame:
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


def read_water_and_tide(hr_dir: Path, pr_dir: Path, station_id: str) -> pd.DataFrame:
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


def nearest_grid_series_h5(nc_path: Path, station_meta: pd.DataFrame, variables: list[str], time_key: str):
    out = {sid: None for sid in station_meta["station_id"]}

    with h5py.File(nc_path, "r") as f:
        if time_key not in f:
            raise KeyError(f"{nc_path} 里面找不到时间变量：{time_key}，实际变量有：{list(f.keys())}")

        times = pd.to_datetime(f[time_key][:], unit="s", utc=True).tz_localize(None)
        lats = f["latitude"][:]
        lons = f["longitude"][:]
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        for _, row in station_meta.iterrows():
            data = {"time": times}

            for var in variables:
                if var not in f:
                    raise KeyError(f"{nc_path} 里面找不到变量：{var}，实际变量有：{list(f.keys())}")

                arr = f[var][:]
                valid = np.isfinite(arr).any(axis=0)

                if valid.any():
                    dist = haversine_km(row["lat"], row["lon"], lat_grid, lon_grid)
                    dist = np.where(valid, dist, np.inf)
                    lat_idx, lon_idx = np.unravel_index(np.argmin(dist), dist.shape)
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

    ty = ty[(ty["SEASON"] == 2025) & (ty["BASIN"].astype(str).str.strip() == "NA")].copy()

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
        tmp = tmp.rename(columns={"ISO_TIME": "time", "wind": "ty_wind", "pres": "ty_pres"})
        tmp = tmp.set_index("time").sort_index()

        tmp = tmp.reindex(full_index).interpolate(method="time", limit=3).ffill(limit=3).bfill(limit=3)

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

    wave = nearest_grid_series_h5(
        paths["wave_path"],
        station_meta,
        ["VHM0", "VTPK", "VTM10", "VMDR"],
        "time",
    )

    era5 = nearest_grid_series_h5(
        paths["era5_path"],
        station_meta,
        ["u10", "v10", "msl"],
        "valid_time",
    )

    gebco = read_gebco_depth(paths["gebco_path"], station_meta)
    typhoon = read_typhoon_features(paths["typhoon_path"], station_meta)

    frames = {}

    for station_id in STATION_IDS:
        df = read_water_and_tide(paths["hr_dir"], paths["pr_dir"], station_id)

        df = df.merge(wave[station_id], on="time", how="left")
        df = df.merge(era5[station_id], on="time", how="inner")
        df = df.merge(typhoon[station_id], on="time", how="left")

        df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)

        meta = station_meta[station_meta["station_id"] == station_id].iloc[0]
        df["lat"] = float(meta["lat"])
        df["lon"] = float(meta["lon"])
        df["elevation"] = gebco[station_id]

        df = df.set_index("time").sort_index()

        df[["vhm0", "vtpk", "vtm10", "vmdr"]] = df[
            ["vhm0", "vtpk", "vtm10", "vmdr"]
        ].interpolate(method="time")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0.0)

        frames[station_id] = df.reset_index()

    return STATION_IDS, frames, station_meta


def build_adjacency(station_meta: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> np.ndarray:
    n = len(STATION_IDS)
    coords = station_meta.set_index("station_id").loc[STATION_IDS][["lat", "lon"]].to_numpy()

    dist = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            dist[i, j] = haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])

    sigma = np.median(dist[dist > 0])
    dist_weight = np.exp(-dist / sigma)

    residuals = np.vstack([frames[sid]["residual"].to_numpy() for sid in STATION_IDS])
    corr = np.nan_to_num(np.corrcoef(residuals), nan=0.0)
    corr_weight = np.maximum(corr, 0.0)

    adj = 0.6 * dist_weight + 0.4 * corr_weight
    np.fill_diagonal(adj, 1.0)

    degree = adj.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))

    return d_inv_sqrt @ adj @ d_inv_sqrt


def make_arrays(frames: dict[str, pd.DataFrame], feature_cols: list[str], target_col: str):
    common_times = None

    for sid in STATION_IDS:
        times = set(frames[sid]["time"])
        common_times = times if common_times is None else common_times & times

    common_times = sorted(common_times)

    features = []
    targets = []
    tides = []

    for sid in STATION_IDS:
        df = frames[sid].set_index("time").loc[common_times]

        features.append(df[feature_cols].to_numpy(dtype=np.float32))
        targets.append(df[target_col].to_numpy(dtype=np.float32))
        tides.append(df["tide"].to_numpy(dtype=np.float32))

    x = np.stack(features, axis=1)
    y = np.stack(targets, axis=1)
    tide = np.stack(tides, axis=1)

    return np.array(common_times), x, y, tide



# ============================================================
# 61. Physics-informed loss GNN-BiGRU
# 核心区别：物理微分方程不再只作为 baseline，而是作为 loss 约束：
# Loss = MSE(y_hat, y_true) + lambda_phys * || d_eta_pred/dt - f_ODE(eta_pred, X, A) ||^2
# ============================================================

FEATURE_GROUPS = {
    "residual_only": [
        "residual", "residual_wavelet", "tide", "sigma",
    ],
    "meteo_core": [
        "residual", "residual_wavelet", "tide", "sigma",
        "u10", "v10", "wind_speed", "msl",
    ],
    "meteo_wave": [
        "residual", "residual_wavelet", "tide", "sigma",
        "u10", "v10", "wind_speed", "msl",
        "vhm0", "vtpk", "vtm10", "vmdr",
    ],
    "meteo_wave_depth": [
        "residual", "residual_wavelet", "tide", "sigma",
        "u10", "v10", "wind_speed", "msl",
        "vhm0", "vtpk", "vtm10", "vmdr", "elevation",
    ],
    "full_no_static": [
        "residual", "residual_wavelet", "tide", "sigma",
        "u10", "v10", "wind_speed", "msl",
        "vhm0", "vtpk", "vtm10", "vmdr",
        "ty_dist", "ty_wind", "ty_pres",
    ],
    "full": [
        "residual", "residual_wavelet", "tide", "sigma",
        "u10", "v10", "wind_speed", "msl",
        "vhm0", "vtpk", "vtm10", "vmdr",
        "ty_dist", "ty_wind", "ty_pres",
        "lat", "lon", "elevation",
    ],
}

# 物理 loss 中使用的外部强迫项。这里使用标准化后的 forcing，避免不同变量量纲差异太大。
PHYSICS_FORCING_COLS = [
    "wind_speed", "msl", "vhm0", "vtpk", "vtm10", "vmdr", "elevation", "ty_dist", "ty_wind", "ty_pres",
]

# 根据你前面实验得到的最优配置，自动为不同 horizon 选特征和图初始化先验。
DEFAULT_HORIZON_CONFIG = {
    1: {
        "feature_group": "full_no_static",
        "graph_init_weights": [0.90, 0.05, 0.05],  # identity, distance, corr
    },
    12: {
        "feature_group": "meteo_wave_depth",
        "graph_init_weights": [0.15, 0.75, 0.10],
    },
    24: {
        "feature_group": "meteo_wave",
        "graph_init_weights": [0.15, 0.10, 0.75],
    },
}


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.float64)
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    adj = np.maximum(adj, 0.0)
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
    return d_inv_sqrt @ adj @ d_inv_sqrt


def build_graph_priors(station_meta: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """构造三种图先验：identity / distance / residual correlation。"""
    n = len(STATION_IDS)
    coords = station_meta.set_index("station_id").loc[STATION_IDS][["lat", "lon"]].to_numpy()

    identity = np.eye(n, dtype=np.float64)

    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dist[i, j] = haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
    sigma = np.median(dist[dist > 0])
    distance = np.exp(-dist / (sigma + 1e-8))
    np.fill_diagonal(distance, 1.0)

    residuals = np.vstack([frames[sid]["residual"].to_numpy(dtype=np.float64) for sid in STATION_IDS])
    corr = np.nan_to_num(np.corrcoef(residuals), nan=0.0)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 1.0)

    return {
        "identity": normalize_adjacency(identity),
        "distance": normalize_adjacency(distance),
        "corr": normalize_adjacency(corr),
    }


def make_feature_arrays(frames: dict[str, pd.DataFrame], feature_cols: list[str], physics_cols: list[str]):
    """对齐所有站点时间，并返回模型输入、residual、tide 和物理 loss 使用的 forcing。"""
    common_times = None
    for sid in STATION_IDS:
        times = set(frames[sid]["time"])
        common_times = times if common_times is None else common_times & times
    common_times = sorted(common_times)

    missing = []
    for sid in STATION_IDS:
        for c in feature_cols + physics_cols + ["residual", "tide"]:
            if c not in frames[sid].columns:
                missing.append((sid, c))
    if missing:
        shown = ", ".join([f"{sid}:{c}" for sid, c in missing[:20]])
        raise KeyError(f"数据中缺少必要列，前 20 个缺失项：{shown}")

    x_list, y_list, tide_list, phys_list = [], [], [], []
    for sid in STATION_IDS:
        df = frames[sid].set_index("time").loc[common_times]
        x_list.append(df[feature_cols].to_numpy(dtype=np.float32))
        y_list.append(df["residual"].to_numpy(dtype=np.float32))
        tide_list.append(df["tide"].to_numpy(dtype=np.float32))
        phys_list.append(df[physics_cols].to_numpy(dtype=np.float32))

    # [time, node, feature]
    x = np.stack(x_list, axis=1)
    y = np.stack(y_list, axis=1)
    tide = np.stack(tide_list, axis=1)
    phys = np.stack(phys_list, axis=1)
    return np.array(common_times), x, y, tide, phys


class SeaLevelPhysicsWindowDataset(Dataset):
    """
    返回：
      xb:          [window, node, feature]，模型输入
      yb:          [node, horizon]，真实 residual
      tb:          [node, horizon]，对应天文潮，用于重构总水位
      eta0:        [node]，预测起点前一小时 residual
      phys_seq:    [node, horizon, physics_feature]，物理 loss 中的外部强迫项

    physics_forcing_mode:
      last_input: 使用预测起点前一小时的 forcing，并复制到整个 horizon，严格避免未来 forcing 信息。
      future:     使用目标时段真实/再分析 forcing。若你有业务预报 forcing，可以用这个；否则更建议 last_input。
    """
    def __init__(
        self,
        x_scaled: np.ndarray,
        y_residual: np.ndarray,
        tide: np.ndarray,
        physics_scaled: np.ndarray,
        window: int,
        horizon: int,
        start: int,
        end: int,
        physics_forcing_mode: str = "last_input",
    ):
        self.x_scaled = x_scaled
        self.y = y_residual
        self.tide = tide
        self.physics_scaled = physics_scaled
        self.window = int(window)
        self.horizon = int(horizon)
        self.physics_forcing_mode = physics_forcing_mode
        self.indices = np.arange(start + window, end - horizon + 1)

        if physics_forcing_mode not in {"last_input", "future"}:
            raise ValueError("physics_forcing_mode must be 'last_input' or 'future'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = int(self.indices[idx])

        xb = self.x_scaled[t - self.window: t]                  # [window, node, feature]
        yb = self.y[t: t + self.horizon].T                      # [node, horizon]
        tb = self.tide[t: t + self.horizon].T                   # [node, horizon]
        eta0 = self.y[t - 1]                                    # [node]

        if self.physics_forcing_mode == "future":
            phys_seq = self.physics_scaled[t: t + self.horizon].transpose(1, 0, 2)
        else:
            phys_seq = np.repeat(self.physics_scaled[t - 1: t], self.horizon, axis=0).transpose(1, 0, 2)

        return (
            torch.from_numpy(xb.astype(np.float32)),
            torch.from_numpy(yb.astype(np.float32)),
            torch.from_numpy(tb.astype(np.float32)),
            torch.from_numpy(eta0.astype(np.float32)),
            torch.from_numpy(phys_seq.astype(np.float32)),
        )


class LearnableGraphFusion(nn.Module):
    def __init__(self, graph_priors: dict[str, np.ndarray], init_weights: Optional[list[float]] = None):
        super().__init__()
        self.graph_names = ["identity", "distance", "corr"]
        priors = np.stack([graph_priors[name] for name in self.graph_names], axis=0).astype(np.float32)
        self.register_buffer("priors", torch.tensor(priors, dtype=torch.float32))

        if init_weights is None:
            init_weights = [1.0 / len(self.graph_names)] * len(self.graph_names)
        init_weights = np.asarray(init_weights, dtype=np.float32)
        init_weights = init_weights / init_weights.sum()
        logits = np.log(init_weights + 1e-8)
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def weights(self):
        return torch.softmax(self.logits, dim=0)

    def forward(self):
        w = self.weights()
        adj = torch.einsum("g,gij->ij", w, self.priors)
        return adj

    def weight_dict(self):
        w = self.weights().detach().cpu().numpy()
        return {name: float(w[i]) for i, name in enumerate(self.graph_names)}


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: [batch, time, node, feature]
        x = torch.einsum("ij,btjf->btif", adj, x)
        return self.linear(x)


class PhysicsInformedGNNBiGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        graph_priors: dict[str, np.ndarray],
        graph_init_weights: list[float],
        gnn_hidden: int,
        gru_hidden: int,
        horizon: int,
        dropout: float,
    ):
        super().__init__()
        self.graph = LearnableGraphFusion(graph_priors, graph_init_weights)
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
        self.head = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, horizon),
        )

    def forward(self, x):
        adj = self.graph()
        h = torch.relu(self.gcn1(x, adj))
        h = self.dropout(torch.relu(self.gcn2(h, adj)))

        bsz, steps, nodes, hidden = h.shape
        h = h.permute(0, 2, 1, 3).reshape(bsz * nodes, steps, hidden)
        out, _ = self.gru(h)
        last = out[:, -1, :]
        pred = self.head(last)
        return pred.reshape(bsz, nodes, -1)


class PhysicsODEConstraint(nn.Module):
    """
    物理微分方程约束：

      dη/dt = b - λη + κ(Aη - η) + βX

    其中：
      η: residual
      b: 站点偏置项
      λ: 衰减系数，使用 softplus 保证非负
      κ: 图扩散系数，使用 softplus 保证非负
      Aη - η: 空间传播/扩散项
      X: 风、气压、波浪、水深、台风等 forcing，已标准化
      β: forcing 系数，可学习

    physics_loss = mean((discrete_dη_pred - ODE_RHS)^2)
    """
    def __init__(self, num_nodes: int, num_forcing: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_nodes))
        self.raw_decay = nn.Parameter(torch.tensor(-2.5))     # softplus 后约 0.079
        self.raw_kappa = nn.Parameter(torch.tensor(-3.0))     # softplus 后约 0.049
        self.beta = nn.Parameter(torch.zeros(num_forcing))

    def coefficients(self):
        return {
            "mean_bias": float(self.bias.detach().cpu().mean()),
            "decay_lambda": float(torch.nn.functional.softplus(self.raw_decay).detach().cpu()),
            "diffusion_kappa": float(torch.nn.functional.softplus(self.raw_kappa).detach().cpu()),
            **{f"beta_{i}": float(v) for i, v in enumerate(self.beta.detach().cpu().numpy())},
        }

    def forward(self, pred_residual, eta0, phys_seq, adj):
        # pred_residual: [B, N, H]
        # eta0:          [B, N]
        # phys_seq:      [B, N, H, P]
        # adj:           [N, N]
        eta_prev = torch.cat([eta0.unsqueeze(-1), pred_residual[:, :, :-1]], dim=-1)  # [B,N,H]
        lhs = pred_residual - eta_prev                                                # discrete dη/dt, dt=1 hour

        spatial = torch.einsum("ij,bjh->bih", adj, eta_prev) - eta_prev              # Aη - η
        forcing = torch.einsum("p,bnhp->bnh", self.beta, phys_seq)                   # βX

        decay = torch.nn.functional.softplus(self.raw_decay)
        kappa = torch.nn.functional.softplus(self.raw_kappa)
        rhs = self.bias.view(1, -1, 1) - decay * eta_prev + kappa * spatial + forcing

        residual = lhs - rhs
        return torch.mean(residual ** 2), residual


class CombinedPhysicsLoss(nn.Module):
    """
    优化后的 physics-informed loss。

    相比旧版的关键变化：
    1. physics loss 不再固定从第 1 个 epoch 就强约束，而是由 train_model 动态传入 lambda。
    2. physics residual 会除以训练集 residual 差分标准差 physics_scale，避免量纲/尺度不匹配。
    3. 支持 mse 或 huber 形式的 physics loss；默认 huber 更稳，不容易把 GNN 拉偏。
    4. best model 默认按 validation data_loss 选择，而不是 total_loss，保证预测精度优先。
    """
    def __init__(
        self,
        physics_ode: PhysicsODEConstraint,
        physics_scale: float,
        last_step_weight: float = 0.0,
        physics_loss_type: str = "huber",
        ode_coef_l2: float = 1e-5,
    ):
        super().__init__()
        self.physics_ode = physics_ode
        self.physics_scale = float(max(physics_scale, 1e-6))
        self.last_step_weight = float(last_step_weight)
        self.physics_loss_type = physics_loss_type
        self.ode_coef_l2 = float(ode_coef_l2)
        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss(beta=1.0)

    def compute_physics_loss(self, pred, eta0, phys_seq, adj):
        _, physics_residual = self.physics_ode(pred, eta0, phys_seq, adj)
        scaled = physics_residual / self.physics_scale
        if self.physics_loss_type == "mse":
            physics_loss = torch.mean(scaled ** 2)
        elif self.physics_loss_type == "huber":
            physics_loss = self.huber(scaled, torch.zeros_like(scaled))
        else:
            raise ValueError("physics_loss_type must be 'mse' or 'huber'")
        return physics_loss

    def ode_regularization(self):
        reg = torch.mean(self.physics_ode.beta ** 2)
        reg = reg + torch.mean(self.physics_ode.bias ** 2)
        reg = reg + torch.nn.functional.softplus(self.physics_ode.raw_decay) ** 2
        reg = reg + torch.nn.functional.softplus(self.physics_ode.raw_kappa) ** 2
        return reg

    def forward(self, pred, target, eta0, phys_seq, adj, physics_lambda: float):
        data_loss = self.mse(pred, target)
        last_loss = self.mse(pred[:, :, -1], target[:, :, -1])
        physics_loss = self.compute_physics_loss(pred, eta0, phys_seq, adj)
        ode_reg = self.ode_regularization()
        total = (
            data_loss
            + self.last_step_weight * last_loss
            + float(physics_lambda) * physics_loss
            + self.ode_coef_l2 * ode_reg
        )
        return total, {
            "data_loss": float(data_loss.detach().cpu()),
            "last_loss": float(last_loss.detach().cpu()),
            "physics_loss": float(physics_loss.detach().cpu()),
            "ode_reg": float(ode_reg.detach().cpu()),
            "physics_lambda": float(physics_lambda),
            "total_loss": float(total.detach().cpu()),
        }


def physics_lambda_for_epoch(epoch: int, lambda_max: float, warmup_epochs: int, ramp_epochs: int) -> float:
    """
    物理 loss 权重调度：
    - 前 warmup_epochs 个 epoch：lambda=0，让 GNN 先学会基本预测。
    - 接着 ramp_epochs 个 epoch：从 0 线性增加到 lambda_max。
    - 后面保持 lambda_max。

    这是根据你这次结果做的优化：旧版一开始就用 0.05 强约束，24h 反而明显变差。
    """
    if lambda_max <= 0:
        return 0.0
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(lambda_max)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / float(ramp_epochs)))
    return float(lambda_max) * progress


@torch.no_grad()
def evaluate_combined_loss(model, criterion, loader, device, physics_lambda: float):
    model.eval()
    totals = {
        "total_loss": 0.0,
        "data_loss": 0.0,
        "last_loss": 0.0,
        "physics_loss": 0.0,
        "ode_reg": 0.0,
        "physics_lambda": 0.0,
    }
    n_batches = 0
    for xb, yb, _, eta0, phys_seq in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        eta0 = eta0.to(device)
        phys_seq = phys_seq.to(device)
        pred = model(xb)
        adj = model.graph()
        _, parts = criterion(pred, yb, eta0, phys_seq, adj, physics_lambda)
        for k in totals:
            totals[k] += parts[k]
        n_batches += 1
    for k in totals:
        totals[k] /= max(1, n_batches)
    return totals


def train_model(
    model,
    physics_ode,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    graph_lr_mult: float,
    physics_lr_mult: float,
    weight_decay: float,
    physics_lambda_max: float,
    physics_warmup_epochs: int,
    physics_ramp_epochs: int,
    physics_scale: float,
    physics_loss_type: str,
    ode_coef_l2: float,
    selection_metric: str,
    last_step_weight: float,
    patience: int,
    device,
):
    criterion = CombinedPhysicsLoss(
        physics_ode=physics_ode,
        physics_scale=physics_scale,
        last_step_weight=last_step_weight,
        physics_loss_type=physics_loss_type,
        ode_coef_l2=ode_coef_l2,
    )

    graph_params = list(model.graph.parameters())
    graph_param_ids = {id(p) for p in graph_params}
    base_params = [p for p in model.parameters() if id(p) not in graph_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": lr},
            {"params": graph_params, "lr": lr * graph_lr_mult},
            {"params": physics_ode.parameters(), "lr": lr * physics_lr_mult},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    if selection_metric not in {"val_data_loss", "val_total_loss", "val_last_loss"}:
        raise ValueError("selection_metric must be val_data_loss, val_total_loss, or val_last_loss")

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        current_lambda = physics_lambda_for_epoch(
            epoch=epoch,
            lambda_max=physics_lambda_max,
            warmup_epochs=physics_warmup_epochs,
            ramp_epochs=physics_ramp_epochs,
        )

        model.train()
        physics_ode.train()
        train_sum = {
            "total_loss": 0.0,
            "data_loss": 0.0,
            "last_loss": 0.0,
            "physics_loss": 0.0,
            "ode_reg": 0.0,
            "physics_lambda": 0.0,
        }
        n_batches = 0

        for xb, yb, _, eta0, phys_seq in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            eta0 = eta0.to(device)
            phys_seq = phys_seq.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            adj = model.graph()
            loss, parts = criterion(pred, yb, eta0, phys_seq, adj, current_lambda)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(physics_ode.parameters()), 1.0)
            optimizer.step()

            for k in train_sum:
                train_sum[k] += parts[k]
            n_batches += 1

        for k in train_sum:
            train_sum[k] /= max(1, n_batches)

        val_parts = evaluate_combined_loss(model, criterion, val_loader, device, current_lambda)
        if selection_metric == "val_data_loss":
            val_score = val_parts["data_loss"]
        elif selection_metric == "val_last_loss":
            val_score = val_parts["last_loss"]
        else:
            val_score = val_parts["total_loss"]
        scheduler.step(val_score)

        weights = model.graph.weight_dict()
        row = {
            "epoch": epoch,
            "selection_metric": selection_metric,
            "selection_score": val_score,
            **{f"train_{k}": v for k, v in train_sum.items()},
            **{f"val_{k}": v for k, v in val_parts.items()},
            "w_identity": weights["identity"],
            "w_distance": weights["distance"],
            "w_corr": weights["corr"],
            **physics_ode.coefficients(),
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"lambda={current_lambda:.6f} "
            f"train_data={train_sum['data_loss']:.6f} "
            f"train_phys={train_sum['physics_loss']:.6f} "
            f"val_data={val_parts['data_loss']:.6f} "
            f"val_last={val_parts['last_loss']:.6f} "
            f"val_phys={val_parts['physics_loss']:.6f} "
            f"select={val_score:.6f} "
            f"w=[id {weights['identity']:.3f}, dist {weights['distance']:.3f}, corr {weights['corr']:.3f}]"
        )

        if val_score < best_val - 1e-8:
            best_val = val_score
            bad_epochs = 0
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "physics_ode": {k: v.detach().cpu().clone() for k, v in physics_ode.state_dict().items()},
            }
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best_{selection_metric}={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        physics_ode.load_state_dict(best_state["physics_ode"])

    return pd.DataFrame(history), best_val


@torch.no_grad()
def predict(model, physics_ode, loader, device):
    model.eval()
    physics_ode.eval()
    preds, ys, tides = [], [], []
    physics_losses = []

    for xb, yb, tb, eta0, phys_seq in loader:
        xb = xb.to(device)
        eta0_device = eta0.to(device)
        phys_seq_device = phys_seq.to(device)
        pred = model(xb)
        adj = model.graph()
        phys_loss, _ = physics_ode(pred, eta0_device, phys_seq_device, adj)
        physics_losses.append(float(phys_loss.detach().cpu()))

        preds.append(pred.cpu().numpy())
        ys.append(yb.numpy())
        tides.append(tb.numpy())

    return np.concatenate(preds), np.concatenate(ys), np.concatenate(tides), float(np.mean(physics_losses))


def regression_metrics(y_true, y_pred):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    mse = mean_squared_error(yt, yp)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def summarize_metrics(true_residual, pred_residual, tide):
    pred_level = pred_residual + tide
    true_level = true_residual + tide

    seq_res = regression_metrics(true_residual, pred_residual)
    seq_level = regression_metrics(true_level, pred_level)
    last_res = regression_metrics(true_residual[:, :, -1], pred_residual[:, :, -1])
    last_level = regression_metrics(true_level[:, :, -1], pred_level[:, :, -1])

    out = {}
    for k, v in seq_res.items():
        out[f"seq_residual_{k}"] = v
    for k, v in last_res.items():
        out[f"last_residual_{k}"] = v
    for k, v in seq_level.items():
        out[f"seq_sea_level_{k}"] = v
    for k, v in last_level.items():
        out[f"last_sea_level_{k}"] = v
    return out


def run_one_horizon(
    horizon: int,
    args,
    frames,
    station_meta,
    graph_priors,
    output_dir: Path,
    device,
):
    if args.feature_group == "auto":
        feature_group = DEFAULT_HORIZON_CONFIG.get(horizon, DEFAULT_HORIZON_CONFIG[24])["feature_group"]
    else:
        feature_group = args.feature_group

    if feature_group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {feature_group}; choices={list(FEATURE_GROUPS)}")

    feature_cols = FEATURE_GROUPS[feature_group]
    physics_cols = [c for c in PHYSICS_FORCING_COLS if c in frames[STATION_IDS[0]].columns]

    if args.graph_init == "auto":
        init_weights = DEFAULT_HORIZON_CONFIG.get(horizon, DEFAULT_HORIZON_CONFIG[24])["graph_init_weights"]
    elif args.graph_init == "uniform":
        init_weights = [1 / 3, 1 / 3, 1 / 3]
    elif args.graph_init == "identity":
        init_weights = [0.90, 0.05, 0.05]
    elif args.graph_init == "distance":
        init_weights = [0.10, 0.85, 0.05]
    elif args.graph_init == "corr":
        init_weights = [0.10, 0.05, 0.85]
    else:
        raise ValueError("--graph-init must be auto/uniform/identity/distance/corr")

    print("\n" + "=" * 80)
    print(f"Running horizon={horizon}h | feature_group={feature_group} | physics_lambda_max={args.physics_lambda_max}")
    print(f"Graph init weights [identity, distance, corr] = {init_weights}")
    print(f"Model features ({len(feature_cols)}): {feature_cols}")
    print(f"Physics forcing features ({len(physics_cols)}): {physics_cols}")

    times, x_raw, y_residual, tide, phys_raw = make_feature_arrays(frames, feature_cols, physics_cols)

    n_time, nodes, feats = x_raw.shape
    train_end = int(n_time * args.train_ratio)
    val_end = int(n_time * (args.train_ratio + args.val_ratio))

    x_scaler = StandardScaler()
    x_scaler.fit(x_raw[:train_end].reshape(-1, feats))
    x_scaled = x_scaler.transform(x_raw.reshape(-1, feats)).reshape(n_time, nodes, feats).astype(np.float32)

    phys_scaler = StandardScaler()
    phys_scaler.fit(phys_raw[:train_end].reshape(-1, len(physics_cols)))
    phys_scaled = phys_scaler.transform(phys_raw.reshape(-1, len(physics_cols))).reshape(n_time, nodes, len(physics_cols)).astype(np.float32)

    y_residual = y_residual.astype(np.float32)
    tide = tide.astype(np.float32)

    train_ds = SeaLevelPhysicsWindowDataset(
        x_scaled, y_residual, tide, phys_scaled,
        args.window, horizon, 0, train_end,
        physics_forcing_mode=args.physics_forcing_mode,
    )
    val_ds = SeaLevelPhysicsWindowDataset(
        x_scaled, y_residual, tide, phys_scaled,
        args.window, horizon, train_end, val_end,
        physics_forcing_mode=args.physics_forcing_mode,
    )
    test_ds = SeaLevelPhysicsWindowDataset(
        x_scaled, y_residual, tide, phys_scaled,
        args.window, horizon, val_end, n_time,
        physics_forcing_mode=args.physics_forcing_mode,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(f"Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Input array: time={n_time}, nodes={nodes}, features={feats}, device={device}")

    model = PhysicsInformedGNNBiGRU(
        input_dim=feats,
        graph_priors=graph_priors,
        graph_init_weights=init_weights,
        gnn_hidden=args.gnn_hidden,
        gru_hidden=args.gru_hidden,
        horizon=horizon,
        dropout=args.dropout,
    ).to(device)

    physics_ode = PhysicsODEConstraint(num_nodes=nodes, num_forcing=len(physics_cols)).to(device)

    train_dy = y_residual[1:train_end] - y_residual[:train_end - 1]
    physics_scale = float(np.std(train_dy) + 1e-6)
    print(f"Physics residual scale from training d_eta std: {physics_scale:.6f}")

    history, best_val = train_model(
        model=model,
        physics_ode=physics_ode,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        graph_lr_mult=args.graph_lr_mult,
        physics_lr_mult=args.physics_lr_mult,
        weight_decay=args.weight_decay,
        physics_lambda_max=args.physics_lambda_max,
        physics_warmup_epochs=args.physics_warmup_epochs,
        physics_ramp_epochs=args.physics_ramp_epochs,
        physics_scale=physics_scale,
        physics_loss_type=args.physics_loss_type,
        ode_coef_l2=args.ode_coef_l2,
        selection_metric=args.selection_metric,
        last_step_weight=args.last_step_weight,
        patience=args.patience,
        device=device,
    )

    pred_residual, true_residual, target_tide, test_physics_loss = predict(model, physics_ode, test_loader, device)
    metrics = summarize_metrics(true_residual, pred_residual, target_tide)

    weights = model.graph.weight_dict()
    ode_coef = physics_ode.coefficients()

    row = {
        "model": "optimized_physics_informed_loss_gnn_bigru",
        "horizon": horizon,
        "feature_group": feature_group,
        "physics_lambda_max": args.physics_lambda_max,
        "physics_warmup_epochs": args.physics_warmup_epochs,
        "physics_ramp_epochs": args.physics_ramp_epochs,
        "physics_loss_type": args.physics_loss_type,
        "selection_metric": args.selection_metric,
        "physics_scale": physics_scale,
        "physics_forcing_mode": args.physics_forcing_mode,
        "best_val_total_loss": best_val,
        "test_physics_loss": test_physics_loss,
        "learned_w_identity": weights["identity"],
        "learned_w_distance": weights["distance"],
        "learned_w_corr": weights["corr"],
        **ode_coef,
        **metrics,
    }

    horizon_dir = output_dir / f"horizon_{horizon}h"
    horizon_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(horizon_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(horizon_dir / "metrics.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "physics_ode_state_dict": physics_ode.state_dict(),
            "feature_group": feature_group,
            "feature_cols": feature_cols,
            "physics_cols": physics_cols,
            "graph_init_weights": init_weights,
            "args": vars(args),
        },
        horizon_dir / "model.pt",
    )

    target_start_times = np.array([str(times[int(i)]) for i in test_ds.indices])
    np.savez_compressed(
        horizon_dir / "predictions.npz",
        pred_residual=pred_residual,
        true_residual=true_residual,
        target_tide=target_tide,
        target_start_times=target_start_times,
        station_ids=np.array(STATION_IDS),
    )

    print("\nTest metrics:")
    important_keys = [
        "seq_residual_RMSE", "seq_residual_R2", "last_residual_RMSE", "last_residual_R2",
        "seq_sea_level_RMSE", "seq_sea_level_R2", "last_sea_level_RMSE", "last_sea_level_R2",
    ]
    for k in important_keys:
        print(f"  {k}: {row[k]:.6f}")
    print(f"  learned graph weights: identity={weights['identity']:.4f}, distance={weights['distance']:.4f}, corr={weights['corr']:.4f}")
    print(f"Saved horizon outputs to: {horizon_dir}")

    return row


def main():
    parser = argparse.ArgumentParser(description="Optimized physics-informed-loss GNN-BiGRU for NOAA residual prediction")
    parser.add_argument("--data-root", default=".", help="数据根目录，例如 ..\\data\\raw")
    parser.add_argument("--output-dir", default="outputs/optimized_physics_informed_loss_gnn_bigru")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 12, 24])
    parser.add_argument("--feature-group", default="auto", choices=["auto"] + list(FEATURE_GROUPS.keys()))
    parser.add_argument("--graph-init", default="auto", choices=["auto", "uniform", "identity", "distance", "corr"])

    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--gnn-hidden", type=int, default=64)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--graph-lr-mult", type=float, default=3.0)
    parser.add_argument("--physics-lr-mult", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument("--physics-lambda-max", type=float, default=0.005,
                        help="物理微分方程 loss 的最大权重。根据这次结果，默认从 0.005 开始，不建议直接 0.05。")
    parser.add_argument("--physics-lambda", type=float, default=None,
                        help="兼容旧命令。如果提供，会覆盖 --physics-lambda-max。")
    parser.add_argument("--physics-warmup-epochs", type=int, default=20,
                        help="前多少个 epoch 不加入物理 loss，让 GNN 先学习数据规律。")
    parser.add_argument("--physics-ramp-epochs", type=int, default=30,
                        help="物理 loss 从 0 线性增加到最大值所用的 epoch 数。")
    parser.add_argument("--physics-loss-type", default="huber", choices=["huber", "mse"],
                        help="physics residual 的 loss 类型。huber 更稳，mse 更强。")
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5,
                        help="ODE 参数 L2 正则，防止物理项系数异常变大。")
    parser.add_argument("--selection-metric", default="val_data_loss", choices=["val_data_loss", "val_total_loss", "val_last_loss"],
                        help="early stopping 和最佳模型选择指标。默认按预测误差 val_data_loss 选，而不是 total loss。")
    parser.add_argument("--last-step-weight", type=float, default=0.0,
                        help="额外加强最后一步目标时效误差的权重；默认不加。")
    parser.add_argument("--physics-forcing-mode", default="last_input", choices=["last_input", "future"],
                        help="物理 loss 用的 forcing。last_input 更严格；future 适合你有未来气象预报 forcing 的情况。")

    args = parser.parse_args()
    if args.physics_lambda is not None:
        args.physics_lambda_max = float(args.physics_lambda)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data once...")
    station_ids, frames, station_meta = build_node_frames(Path(args.data_root))
    graph_priors = build_graph_priors(station_meta, frames)

    all_rows = []
    for horizon in args.horizons:
        row = run_one_horizon(
            horizon=horizon,
            args=args,
            frames=frames,
            station_meta=station_meta,
            graph_priors=graph_priors,
            output_dir=output_dir,
            device=device,
        )
        all_rows.append(row)

    summary = pd.DataFrame(all_rows)
    summary_path = output_dir / "optimized_physics_informed_loss_metrics.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("All horizons finished.")
    print(f"Saved summary metrics to: {summary_path}")
    print("\nRecommended columns to compare with previous results:")
    print(summary[[
        "horizon", "feature_group", "physics_lambda_max",
        "last_residual_RMSE", "last_residual_R2",
        "seq_residual_RMSE", "seq_residual_R2",
        "learned_w_identity", "learned_w_distance", "learned_w_corr",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
