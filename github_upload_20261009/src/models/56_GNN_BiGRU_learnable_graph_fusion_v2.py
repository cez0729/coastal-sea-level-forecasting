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
# Utilities
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


# ============================================================
# Graphs
# ============================================================
def normalize_adj(adj: np.ndarray):
    adj = adj.astype(np.float64)
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt
    return adj_norm.astype(np.float32)


def build_base_graphs(station_meta: pd.DataFrame, frames):
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

    residuals = np.vstack(
        [
            frames[sid]["residual"].to_numpy()
            for sid in STATION_IDS
        ]
    )

    corr = np.nan_to_num(np.corrcoef(residuals), nan=0.0)
    corr_adj = np.maximum(corr, 0.0)
    np.fill_diagonal(corr_adj, 1.0)

    identity_adj = np.eye(n, dtype=np.float64)

    graphs = {
        "identity": normalize_adj(identity_adj),
        "distance": normalize_adj(distance_adj),
        "corr": normalize_adj(corr_adj),
    }

    return graphs


# ============================================================
# Arrays
# ============================================================
def make_arrays(frames, feature_cols, target_col="residual"):
    common_times = None

    for sid in STATION_IDS:
        times = set(frames[sid]["time"])
        common_times = times if common_times is None else common_times & times

    common_times = sorted(common_times)

    if len(common_times) == 0:
        raise ValueError("没有找到 7 个站点共同时间。请检查数据时间范围。")

    features = []
    targets = []
    tides = []

    for sid in STATION_IDS:
        df = frames[sid].set_index("time").loc[common_times]

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"站点 {sid} 缺少特征列：{missing}")

        features.append(df[feature_cols].to_numpy(dtype=np.float32))
        targets.append(df[target_col].to_numpy(dtype=np.float32))
        tides.append(df["tide"].to_numpy(dtype=np.float32))

    x = np.stack(features, axis=1)
    y = np.stack(targets, axis=1)
    tide = np.stack(tides, axis=1)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    tide = np.nan_to_num(tide, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return np.array(common_times), x, y, tide


# ============================================================
# Dataset
# ============================================================
class SeaLevelWindowDataset(Dataset):
    def __init__(
        self,
        x,
        y,
        tide,
        window,
        horizon,
        start,
        end,
        target_mode="multi",
    ):
        self.x = x
        self.y = y
        self.tide = tide
        self.window = window
        self.horizon = horizon
        self.target_mode = target_mode

        self.indices = np.arange(start + window, end - horizon + 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        xs = self.x[t - self.window:t]

        if self.target_mode == "multi":
            ys = self.y[t:t + self.horizon].T
            ts = self.tide[t:t + self.horizon].T

        elif self.target_mode == "direct":
            ys = self.y[t + self.horizon - 1]
            ts = self.tide[t + self.horizon - 1]

            ys = ys[:, None]
            ts = ts[:, None]

        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")

        return (
            torch.tensor(xs, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.float32),
            torch.tensor(ts, dtype=torch.float32),
        )


# ============================================================
# Learnable graph model
# ============================================================
class LearnableGraphFusion(nn.Module):
    def __init__(self, base_graphs: dict):
        super().__init__()

        graph_names = ["identity", "distance", "corr"]
        graph_stack = np.stack([base_graphs[name] for name in graph_names], axis=0)

        self.graph_names = graph_names
        self.register_buffer(
            "graph_stack",
            torch.tensor(graph_stack, dtype=torch.float32),
        )

        self.logits = nn.Parameter(torch.zeros(len(graph_names), dtype=torch.float32))

    def forward(self):
        weights = torch.softmax(self.logits, dim=0)
        adj = torch.sum(weights[:, None, None] * self.graph_stack, dim=0)
        return adj

    def get_weights(self):
        weights = torch.softmax(self.logits.detach().cpu(), dim=0).numpy()
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


class LearnableGraphGNNBiGRU(nn.Module):
    def __init__(
        self,
        input_dim,
        gnn_hidden,
        gru_hidden,
        output_steps,
        dropout,
        base_graphs,
    ):
        super().__init__()

        self.graph_fusion = LearnableGraphFusion(base_graphs)

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
            nn.Linear(gru_hidden, output_steps),
        )

    def forward(self, x):
        adj = self.graph_fusion()

        h = torch.relu(self.gcn1(x, adj))
        h = self.dropout(torch.relu(self.gcn2(h, adj)))

        bsz, steps, nodes, hidden = h.shape

        h = h.permute(0, 2, 1, 3)
        h = h.reshape(bsz * nodes, steps, hidden)

        out, _ = self.gru(h)

        last = out[:, -1, :]

        pred = self.head(last)

        return pred.reshape(bsz, nodes, -1)

    def get_graph_weights(self):
        return self.graph_fusion.get_weights()


# ============================================================
# Train / evaluate / predict
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        pred = model(xb)
        loss = criterion(pred, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = criterion(pred, yb)

        total_loss += loss.item() * xb.size(0)

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    preds = []
    ys = []
    tides = []

    for xb, yb, tb in loader:
        xb = xb.to(device)

        pred = model(xb).cpu().numpy()

        preds.append(pred)
        ys.append(yb.numpy())
        tides.append(tb.numpy())

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(tides, axis=0),
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


# ============================================================
# Optimizer with separate graph LR
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
                "name": "main_params",
            },
            {
                "params": graph_params,
                "lr": lr * graph_lr_mult,
                "weight_decay": 0.0,
                "name": "graph_params",
            },
        ]
    )

    return optimizer


# ============================================================
# One experiment
# ============================================================
def run_one_experiment(
    x_scaled,
    y_residual,
    tide,
    base_graphs,
    feature_group,
    feature_cols,
    horizon,
    target_mode,
    window,
    epochs,
    batch_size,
    lr,
    graph_lr_mult,
    weight_decay,
    gnn_hidden,
    gru_hidden,
    dropout,
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
    print("LEARNABLE GRAPH FUSION EXPERIMENT V2")
    print(f"Horizon      : {horizon}h")
    print(f"Feature group: {feature_group}")
    print(f"Target mode  : {target_mode}")
    print(f"Feature cols : {feature_cols}")
    print(f"x shape      : {x_scaled.shape}")
    print(f"epochs       : {epochs}")
    print(f"lr           : {lr}")
    print(f"graph lr     : {lr * graph_lr_mult}")
    print("=" * 100)

    n_time = len(x_scaled)

    train_end = int(n_time * 0.7)
    val_end = int(n_time * 0.85)

    train_ds = SeaLevelWindowDataset(
        x=x_scaled,
        y=y_residual,
        tide=tide,
        window=window,
        horizon=horizon,
        start=0,
        end=train_end,
        target_mode=target_mode,
    )

    val_ds = SeaLevelWindowDataset(
        x=x_scaled,
        y=y_residual,
        tide=tide,
        window=window,
        horizon=horizon,
        start=train_end,
        end=val_end,
        target_mode=target_mode,
    )

    test_ds = SeaLevelWindowDataset(
        x=x_scaled,
        y=y_residual,
        tide=tide,
        window=window,
        horizon=horizon,
        start=val_end,
        end=n_time,
        target_mode=target_mode,
    )

    print(
        f"Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
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

    output_steps = horizon if target_mode == "multi" else 1

    model = LearnableGraphGNNBiGRU(
        input_dim=x_scaled.shape[-1],
        gnn_hidden=gnn_hidden,
        gru_hidden=gru_hidden,
        output_steps=output_steps,
        dropout=dropout,
        base_graphs=base_graphs,
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

    exp_name = f"learnable_graph_v2_{horizon}h_{feature_group}_{target_mode}"
    exp_dir = output_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = exp_dir / "best_model.pt"
    last_ckpt_path = exp_dir / "last_checkpoint.pt"
    history_path = exp_dir / "graph_weights_history.csv"

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
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
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
                    "target_mode": target_mode,
                    "graph_weights": weights_now,
                },
                best_model_path,
            )
        else:
            bad_epochs += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
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
                "target_mode": target_mode,
                "graph_weights": weights_now,
            },
            last_ckpt_path,
        )

        print(
            f"[learnable_graph_v2 | {horizon}h | {feature_group}] "
            f"epoch={epoch:03d} "
            f"train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f} "
            f"best_val_mse={best_val:.6f} "
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

    pred_residual, true_residual, target_tide = predict(
        model=model,
        loader=test_loader,
        device=device,
    )

    pred_level = pred_residual + target_tide
    true_level = true_residual + target_tide

    # sequence metrics
    seq_residual_metrics = regression_metrics(true_residual, pred_residual)
    seq_level_metrics = regression_metrics(true_level, pred_level)

    # last-step metrics
    last_true_residual = true_residual[:, :, -1:]
    last_pred_residual = pred_residual[:, :, -1:]

    last_true_level = true_level[:, :, -1:]
    last_pred_level = pred_level[:, :, -1:]

    last_residual_metrics = regression_metrics(last_true_residual, last_pred_residual)
    last_level_metrics = regression_metrics(last_true_level, last_pred_level)

    row = {
        "horizon": horizon,
        "target_mode": target_mode,
        "feature_group": feature_group,
        "window": window,
        "num_features": len(feature_cols),
        "features": "|".join(feature_cols),
        "best_val_mse": best_val,

        "learned_w_identity": learned_weights["identity"],
        "learned_w_distance": learned_weights["distance"],
        "learned_w_corr": learned_weights["corr"],

        "seq_residual_MSE": seq_residual_metrics["MSE"],
        "seq_residual_MAE": seq_residual_metrics["MAE"],
        "seq_residual_RMSE": seq_residual_metrics["RMSE"],
        "seq_residual_Bias": seq_residual_metrics["Bias"],
        "seq_residual_R2": seq_residual_metrics["R2"],

        "seq_sea_level_MSE": seq_level_metrics["MSE"],
        "seq_sea_level_MAE": seq_level_metrics["MAE"],
        "seq_sea_level_RMSE": seq_level_metrics["RMSE"],
        "seq_sea_level_Bias": seq_level_metrics["Bias"],
        "seq_sea_level_R2": seq_level_metrics["R2"],

        "last_residual_MSE": last_residual_metrics["MSE"],
        "last_residual_MAE": last_residual_metrics["MAE"],
        "last_residual_RMSE": last_residual_metrics["RMSE"],
        "last_residual_Bias": last_residual_metrics["Bias"],
        "last_residual_R2": last_residual_metrics["R2"],

        "last_sea_level_MSE": last_level_metrics["MSE"],
        "last_sea_level_MAE": last_level_metrics["MAE"],
        "last_sea_level_RMSE": last_level_metrics["RMSE"],
        "last_sea_level_Bias": last_level_metrics["Bias"],
        "last_sea_level_R2": last_level_metrics["R2"],
    }

    pd.DataFrame([row]).to_csv(exp_dir / "metrics.csv", index=False)

    np.save(exp_dir / "pred_residual.npy", pred_residual)
    np.save(exp_dir / "true_residual.npy", true_residual)
    np.save(exp_dir / "target_tide.npy", target_tide)

    print("\nLearned graph weights:")
    print(f"  identity : {learned_weights['identity']:.6f}")
    print(f"  distance : {learned_weights['distance']:.6f}")
    print(f"  corr     : {learned_weights['corr']:.6f}")

    print("\nSequence residual metrics:")
    for k, v in seq_residual_metrics.items():
        print(f"  {k}: {v:.6f}")

    print("\nLast-step residual metrics:")
    for k, v in last_residual_metrics.items():
        print(f"  {k}: {v:.6f}")

    print(f"\nSaved to: {exp_dir}")
    print(f"Saved graph weight history to: {history_path}")

    return row


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default="..\\data\\raw")

    parser.add_argument("--target-mode", choices=["direct", "multi"], default="multi")

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
        help="selected: use best feature group per horizon; unified: use one feature group for all horizons",
    )

    parser.add_argument(
        "--unified-feature-group",
        default="meteo_wave_depth",
        help="used only when --feature-strategy unified",
    )

    parser.add_argument("--window", type=int, default=24)

    parser.add_argument("--gnn-hidden", type=int, default=64)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)

    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--graph-lr-mult", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)

    parser.add_argument("--early-stop-patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-6)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output-dir",
        default="outputs/gnn_bigru_learnable_graph_fusion_v2",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from last checkpoint if available",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    data_root = Path(args.data_root)

    feature_groups = get_feature_groups()

    if args.feature_strategy == "unified":
        if args.unified_feature_group not in feature_groups:
            raise ValueError(
                f"未知 unified_feature_group: {args.unified_feature_group}. "
                f"可选: {list(feature_groups.keys())}"
            )

    print("\nLoading real NOAA data once...")
    station_ids, frames, station_meta = build_node_frames(data_root)

    print("\nBuilding base graphs: identity, distance, corr...")
    base_graphs = build_base_graphs(station_meta, frames)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for horizon in args.horizons:
        if args.feature_strategy == "selected":
            feature_group = get_selected_feature_group_by_horizon(horizon)
        else:
            feature_group = args.unified_feature_group

        feature_cols = feature_groups[feature_group]

        print("\n" + "#" * 100)
        print(f"Horizon {horizon}h learnable graph fusion V2:")
        print(f"  feature_strategy = {args.feature_strategy}")
        print(f"  feature_group    = {feature_group}")
        print("#" * 100)

        times, x, y_residual, tide = make_arrays(
            frames=frames,
            feature_cols=feature_cols,
            target_col="residual",
        )

        print(f"Raw x shape for horizon {horizon}h: {x.shape}")

        n_time, nodes, feats = x.shape

        if nodes != len(STATION_IDS):
            raise ValueError(f"x 节点数错误: {nodes}, expected={len(STATION_IDS)}")

        train_end = int(n_time * 0.7)

        scaler = StandardScaler()
        scaler.fit(x[:train_end].reshape(-1, feats))

        x_scaled = scaler.transform(
            x.reshape(-1, feats)
        ).reshape(n_time, nodes, feats).astype(np.float32)

        y_residual = y_residual.astype(np.float32)
        tide = tide.astype(np.float32)

        row = run_one_experiment(
            x_scaled=x_scaled,
            y_residual=y_residual,
            tide=tide,
            base_graphs=base_graphs,
            feature_group=feature_group,
            feature_cols=feature_cols,
            horizon=horizon,
            target_mode=args.target_mode,
            window=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            graph_lr_mult=args.graph_lr_mult,
            weight_decay=args.weight_decay,
            gnn_hidden=args.gnn_hidden,
            gru_hidden=args.gru_hidden,
            dropout=args.dropout,
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
            output_root / "learnable_graph_metrics.csv",
            index=False,
        )

    overall = pd.DataFrame(all_rows)
    overall.to_csv(output_root / "learnable_graph_metrics.csv", index=False)

    print("\n" + "=" * 100)
    print("LEARNABLE GRAPH FUSION V2 EXPERIMENTS FINISHED")
    print("=" * 100)

    display_cols = [
        "horizon",
        "target_mode",
        "feature_group",
        "learned_w_identity",
        "learned_w_distance",
        "learned_w_corr",
        "seq_residual_RMSE",
        "seq_residual_R2",
        "last_residual_RMSE",
        "last_residual_R2",
        "seq_sea_level_RMSE",
        "seq_sea_level_R2",
        "last_sea_level_RMSE",
        "last_sea_level_R2",
    ]

    print(overall[display_cols])

    print(f"\nSaved metrics to: {output_root / 'learnable_graph_metrics.csv'}")


if __name__ == "__main__":
    main()