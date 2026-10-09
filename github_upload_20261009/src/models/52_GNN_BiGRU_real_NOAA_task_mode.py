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
# 7 个 NOAA 潮位站
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
# 随机种子
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 工具函数
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
# 自动定位数据集根目录
# ============================================================
def resolve_dataset_root(data_root: Path) -> Path:
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

    # 这里必须先定义 era5_dir 和 gebco_dir
    era5_dir = find_dir(root, "era5")
    gebco_dir = find_dir(root, "gebco")

    wave_path = root / "海浪数据.nc"
    if not wave_path.exists():
        wave_path = find_file(root, ["*海浪*.nc", "*wave*.nc", "*.nc"])

    typhoon_path = root / "台风数据.csv"
    if not typhoon_path.exists():
        typhoon_path = find_file(root, ["*台风*.csv", "*typhoon*.csv"])

    if era5_dir is None:
        raise FileNotFoundError(f"找不到 era5 文件夹，请检查：{root}")

    if gebco_dir is None:
        print(f"Warning: 找不到 gebco 文件夹，将直接在 root 下递归搜索 GEBCO .nc 文件：{root}")

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
# 读取站点元数据
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


# ============================================================
# 读取实测水位和天文潮，计算 residual
# ============================================================
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


# ============================================================
# 从 nc/h5 网格中取最近格点时间序列
# ============================================================
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


# ============================================================
# 读取 GEBCO 水深
# ============================================================
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


# ============================================================
# 读取台风特征
# ============================================================
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


# ============================================================
# 构建每个站点 DataFrame
# ============================================================
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

    print("Reading GEBCO depth...")
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
# 构造固定融合图：0.6 distance + 0.4 corr
# ============================================================
def build_adjacency(station_meta: pd.DataFrame, frames):
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
    dist_weight = np.exp(-dist / sigma)

    residuals = np.vstack(
        [
            frames[sid]["residual"].to_numpy()
            for sid in STATION_IDS
        ]
    )

    corr = np.nan_to_num(np.corrcoef(residuals), nan=0.0)
    corr_weight = np.maximum(corr, 0.0)

    adj = 0.6 * dist_weight + 0.4 * corr_weight
    np.fill_diagonal(adj, 1.0)

    degree = adj.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))

    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt

    return adj_norm.astype(np.float32)


# ============================================================
# 对齐所有站点时间，构造 x/y/tide
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

    x = np.stack(features, axis=1)       # [T, N, F]
    y = np.stack(targets, axis=1)        # [T, N]
    tide = np.stack(tides, axis=1)       # [T, N]

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    tide = np.nan_to_num(tide, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return np.array(common_times), x, y, tide


# ============================================================
# Dataset：direct / multi 两种任务
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
        target_mode="direct",
    ):
        """
        x:    [T, N, F]
        y:    [T, N]
        tide: [T, N]

        direct:
            只预测未来第 horizon 小时 residual
            输出 [N, 1]

        multi:
            预测未来 1 到 horizon 小时 residual 序列
            输出 [N, horizon]
        """

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

        xs = self.x[t - self.window:t]  # [window, N, F]

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
# Graph Convolution，修复版
# ============================================================
class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        """
        x:   [B, T, N, F]
        adj: [N, N]
        """

        if x.shape[2] != adj.shape[0]:
            raise ValueError(
                f"Node mismatch: x has {x.shape[2]} nodes, "
                f"but adj has {adj.shape[0]} nodes."
            )

        x = torch.einsum("ij,btjf->btif", adj, x)

        return self.linear(x)


# ============================================================
# GNN-BiGRU
# ============================================================
class GNNBiGRU(nn.Module):
    def __init__(
        self,
        input_dim,
        gnn_hidden,
        gru_hidden,
        output_steps,
        dropout,
    ):
        super().__init__()

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

    def forward(self, x, adj):
        h = torch.relu(self.gcn1(x, adj))
        h = self.dropout(torch.relu(self.gcn2(h, adj)))

        bsz, steps, nodes, hidden = h.shape

        h = h.permute(0, 2, 1, 3)
        h = h.reshape(bsz * nodes, steps, hidden)

        out, _ = self.gru(h)

        last = out[:, -1, :]

        pred = self.head(last)

        return pred.reshape(bsz, nodes, -1)


# ============================================================
# 训练、评估、预测
# ============================================================
def train_one_epoch(model, loader, adj, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        pred = model(xb, adj)

        loss = criterion(pred, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate_loss(model, loader, adj, criterion, device):
    model.eval()

    total_loss = 0.0

    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb, adj)

        loss = criterion(pred, yb)

        total_loss += loss.item() * xb.size(0)

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def predict(model, loader, adj, device):
    model.eval()

    preds = []
    ys = []
    tides = []

    for xb, yb, tb in loader:
        xb = xb.to(device)

        pred = model(xb, adj).cpu().numpy()

        preds.append(pred)
        ys.append(yb.numpy())
        tides.append(tb.numpy())

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(tides, axis=0),
    )


# ============================================================
# 指标
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
# 单个实验
# ============================================================
def run_one_experiment(
    x_scaled,
    y_residual,
    tide,
    adj_np,
    horizon,
    target_mode,
    window,
    epochs,
    batch_size,
    lr,
    weight_decay,
    gnn_hidden,
    gru_hidden,
    dropout,
    device,
    output_root,
):
    print("\n" + "=" * 80)
    print(f"Experiment: target_mode={target_mode}, horizon={horizon}h")
    print(f"x_scaled shape  = {x_scaled.shape}")
    print(f"y_residual shape= {y_residual.shape}")
    print(f"tide shape      = {tide.shape}")
    print(f"adj shape       = {adj_np.shape}")
    print("=" * 80)

    if x_scaled.shape[1] != len(STATION_IDS):
        raise ValueError(
            f"x 的节点数不是 7。x.shape={x_scaled.shape}, STATION_IDS={len(STATION_IDS)}"
        )

    if adj_np.shape != (len(STATION_IDS), len(STATION_IDS)):
        raise ValueError(
            f"adj 不是 7x7。adj.shape={adj_np.shape}"
        )

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
        f"Samples: train={len(train_ds)}, "
        f"val={len(val_ds)}, "
        f"test={len(test_ds)}"
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

    model = GNNBiGRU(
        input_dim=x_scaled.shape[-1],
        gnn_hidden=gnn_hidden,
        gru_hidden=gru_hidden,
        output_steps=output_steps,
        dropout=dropout,
    ).to(device)

    adj = torch.tensor(adj_np, dtype=torch.float32).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            adj=adj,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            adj=adj,
            criterion=criterion,
            device=device,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"[{target_mode}-{horizon}h] "
            f"epoch={epoch:03d} "
            f"train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f} "
            f"best_val_mse={best_val:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_residual, true_residual, target_tide = predict(
        model=model,
        loader=test_loader,
        adj=adj,
        device=device,
    )

    pred_level = pred_residual + target_tide
    true_level = true_residual + target_tide

    residual_metrics = regression_metrics(true_residual, pred_residual)
    level_metrics = regression_metrics(true_level, pred_level)

    exp_name = f"gnn_bigru_{target_mode}_{horizon}h"
    exp_dir = output_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), exp_dir / "model.pt")

    metrics_row = {
        "target_mode": target_mode,
        "horizon": horizon,
        "window": window,
        "output_steps": output_steps,
        "best_val_mse": best_val,

        "residual_MSE": residual_metrics["MSE"],
        "residual_MAE": residual_metrics["MAE"],
        "residual_RMSE": residual_metrics["RMSE"],
        "residual_Bias": residual_metrics["Bias"],
        "residual_R2": residual_metrics["R2"],

        "sea_level_MSE": level_metrics["MSE"],
        "sea_level_MAE": level_metrics["MAE"],
        "sea_level_RMSE": level_metrics["RMSE"],
        "sea_level_Bias": level_metrics["Bias"],
        "sea_level_R2": level_metrics["R2"],
    }

    pd.DataFrame([metrics_row]).to_csv(exp_dir / "metrics.csv", index=False)

    np.save(exp_dir / "pred_residual.npy", pred_residual)
    np.save(exp_dir / "true_residual.npy", true_residual)
    np.save(exp_dir / "target_tide.npy", target_tide)

    print("\nTest residual metrics:")
    for k, v in residual_metrics.items():
        print(f"  {k}: {v:.6f}")

    print("\nTest sea level metrics:")
    for k, v in level_metrics.items():
        print(f"  {k}: {v:.6f}")

    print(f"\nSaved experiment results to: {exp_dir}")

    return metrics_row


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default=".")
    parser.add_argument("--window", type=int, default=24)

    parser.add_argument(
        "--target-mode",
        choices=["direct", "multi"],
        default="direct",
    )

    parser.add_argument(
        "--run-all",
        action="store_true",
        help="run direct and multi for 1h, 12h, 24h",
    )

    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 12, 24],
    )

    parser.add_argument("--gnn-hidden", type=int, default=64)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output-dir",
        default="outputs/gnn_bigru_real_noaa_task_mode",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    data_root = Path(args.data_root)

    # ========================================================
    # 当前先用 full features
    # 后面特征消融时再扩展这里
    # ========================================================
    feature_cols = [
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
    ]

    print("\nLoading and aligning real NOAA data...")

    station_ids, frames, station_meta = build_node_frames(data_root)

    print("\nBuilding adjacency...")
    adj_np = build_adjacency(station_meta, frames)

    print("\nMaking aligned arrays...")
    times, x, y_residual, tide = make_arrays(
        frames=frames,
        feature_cols=feature_cols,
        target_col="residual",
    )

    print("\nRaw data shape:")
    print(f"  times       = {len(times)}")
    print(f"  x           = {x.shape}")
    print(f"  y_residual  = {y_residual.shape}")
    print(f"  tide        = {tide.shape}")
    print(f"  adj         = {adj_np.shape}")
    print(f"  features    = {len(feature_cols)}")

    if x.shape[1] != len(STATION_IDS):
        raise ValueError(
            f"x 节点维度错误：x.shape={x.shape}，应该是 [T, 7, F]"
        )

    # ========================================================
    # 标准化：只用 train 段 fit
    # ========================================================
    n_time, nodes, feats = x.shape
    train_end = int(n_time * 0.7)

    scaler = StandardScaler()
    scaler.fit(x[:train_end].reshape(-1, feats))

    x_scaled = scaler.transform(
        x.reshape(-1, feats)
    ).reshape(n_time, nodes, feats).astype(np.float32)

    y_residual = y_residual.astype(np.float32)
    tide = tide.astype(np.float32)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    modes = ["direct", "multi"] if args.run_all else [args.target_mode]

    all_rows = []

    for mode in modes:
        for horizon in args.horizons:
            row = run_one_experiment(
                x_scaled=x_scaled,
                y_residual=y_residual,
                tide=tide,
                adj_np=adj_np,
                horizon=horizon,
                target_mode=mode,
                window=args.window,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                gnn_hidden=args.gnn_hidden,
                gru_hidden=args.gru_hidden,
                dropout=args.dropout,
                device=device,
                output_root=output_root,
            )

            all_rows.append(row)

            pd.DataFrame(all_rows).to_csv(
                output_root / "overall_metrics.csv",
                index=False,
            )

    overall = pd.DataFrame(all_rows)
    overall.to_csv(output_root / "overall_metrics.csv", index=False)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 80)

    print(overall[
        [
            "target_mode",
            "horizon",
            "window",
            "residual_RMSE",
            "residual_R2",
            "sea_level_RMSE",
            "sea_level_R2",
        ]
    ])

    print(f"\nSaved overall metrics to: {output_root / 'overall_metrics.csv'}")


if __name__ == "__main__":
    main()