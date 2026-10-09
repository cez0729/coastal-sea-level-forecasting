import argparse
import json
import math
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


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
# Read data
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
# Arrays
# ============================================================
def make_physics_arrays(frames):
    common_times = None

    for sid in STATION_IDS:
        times = set(frames[sid]["time"])
        common_times = times if common_times is None else common_times & times

    common_times = sorted(common_times)

    if len(common_times) == 0:
        raise ValueError("没有找到 7 个站点共同时间。请检查数据时间范围。")

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
            raise KeyError(f"站点 {sid} 缺少物理 ODE 所需列：{missing}")

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
    distance_adj = row_normalize(distance_adj)

    corr = np.nan_to_num(np.corrcoef(residual.T), nan=0.0)
    corr_adj = np.maximum(corr, 0.0)
    np.fill_diagonal(corr_adj, 1.0)
    corr_adj = row_normalize(corr_adj)

    identity_adj = np.eye(n, dtype=np.float32)

    fusion_adj = row_normalize(0.5 * distance_adj + 0.5 * corr_adj)

    graphs = {
        "identity": identity_adj,
        "distance": distance_adj,
        "corr": corr_adj,
        "fusion": fusion_adj,
    }

    return graphs


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
# Physics ODE model
# ============================================================
class PhysicsODEModel:
    """
    Linear ODE baseline:

    d eta / dt = f(eta, forcing, graph)

    Fit by:
    eta[t+1] - eta[t] = Ridge(features[t])

    Then forecast recursively:
    eta_hat[t+s+1] = eta_hat[t+s] + d_eta_hat[t+s]
    """

    def __init__(
        self,
        model_name,
        feature_terms,
        graph_adj=None,
        ridge_alpha=1e-2,
        dt_hours=1.0,
        msl_ref=101325.0,
        clip_residual=None,
    ):
        self.model_name = model_name
        self.feature_terms = feature_terms
        self.graph_adj = graph_adj
        self.ridge_alpha = ridge_alpha
        self.dt_hours = dt_hours
        self.msl_ref = msl_ref
        self.clip_residual = clip_residual

        self.scaler = StandardScaler()
        self.regressor = Ridge(alpha=ridge_alpha)

        self.fitted = False

    def _build_features_one_time(self, eta, data, t):
        """
        eta: shape [N]
        data arrays: each [T, N]
        t: forcing time index
        return feature matrix [N, F]
        """

        features = []

        for term in self.feature_terms:
            if term == "eta":
                value = eta

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

            elif term == "depth_positive":
                value = np.maximum(-data["elevation"][t], 1.0)

            elif term == "inverse_depth":
                depth = np.maximum(-data["elevation"][t], 1.0)
                value = 1.0 / depth

            elif term == "graph_delta":
                if self.graph_adj is None:
                    raise ValueError("graph_delta term requires graph_adj.")
                value = self.graph_adj @ eta - eta

            elif term == "graph_eta":
                if self.graph_adj is None:
                    raise ValueError("graph_eta term requires graph_adj.")
                value = self.graph_adj @ eta

            elif term == "constant":
                value = np.ones_like(eta)

            else:
                raise ValueError(f"Unknown feature term: {term}")

            value = np.asarray(value, dtype=np.float32)
            features.append(value[:, None])

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
            raise RuntimeError("Model is not fitted.")

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

        if self.clip_residual is not None:
            eta_next = np.clip(
                eta_next,
                -abs(self.clip_residual),
                abs(self.clip_residual),
            )

        return eta_next.astype(np.float32)

    def forecast_sequence(self, data, start_t, horizon):
        """
        start_t: forecast origin.
        Initial state = residual[start_t - 1]
        Predict residual[start_t], ..., residual[start_t + horizon - 1]
        """

        residual = data["residual"]

        eta = residual[start_t - 1].copy()

        preds = []

        for s in range(horizon):
            forcing_t = start_t + s - 1

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
        coef = self.regressor.coef_

        return pd.DataFrame(
            {
                "term": self.feature_terms,
                "coef_scaled_feature": coef,
            }
        )


# ============================================================
# Baseline: persistence
# ============================================================
def persistence_forecast(data, start_t, horizon):
    eta0 = data["residual"][start_t - 1].copy()
    pred = np.repeat(eta0[:, None], horizon, axis=1)
    return pred.astype(np.float32)


# ============================================================
# Experiment
# ============================================================
def make_model_specs(graphs, ridge_alpha, clip_residual):
    specs = []

    specs.append(
        {
            "model_name": "ode_decay",
            "feature_terms": [
                "constant",
                "eta",
            ],
            "graph_adj": None,
        }
    )

    specs.append(
        {
            "model_name": "ode_local_forcing",
            "feature_terms": [
                "constant",
                "eta",
                "u10",
                "v10",
                "wind_speed",
                "pressure_anom",
                "vhm0",
            ],
            "graph_adj": None,
        }
    )

    for graph_name in ["distance", "corr", "fusion"]:
        specs.append(
            {
                "model_name": f"ode_graph_diffusion_{graph_name}",
                "feature_terms": [
                    "constant",
                    "eta",
                    "graph_delta",
                ],
                "graph_adj": graphs[graph_name],
            }
        )

    for graph_name in ["distance", "corr", "fusion"]:
        specs.append(
            {
                "model_name": f"ode_forced_graph_{graph_name}",
                "feature_terms": [
                    "constant",
                    "eta",
                    "graph_delta",
                    "u10",
                    "v10",
                    "wind_speed",
                    "pressure_anom",
                    "vhm0",
                ],
                "graph_adj": graphs[graph_name],
            }
        )

    for graph_name in ["distance", "corr", "fusion"]:
        specs.append(
            {
                "model_name": f"ode_bathymetry_forced_graph_{graph_name}",
                "feature_terms": [
                    "constant",
                    "eta",
                    "graph_delta",
                    "wind_stress_u",
                    "wind_stress_v",
                    "pressure_anom",
                    "vhm0",
                    "elevation",
                    "inverse_depth",
                ],
                "graph_adj": graphs[graph_name],
            }
        )

    models = []

    for spec in specs:
        model = PhysicsODEModel(
            model_name=spec["model_name"],
            feature_terms=spec["feature_terms"],
            graph_adj=spec["graph_adj"],
            ridge_alpha=ridge_alpha,
            dt_hours=1.0,
            clip_residual=clip_residual,
        )
        models.append(model)

    return models


def evaluate_model_on_horizon(model, data, horizon, test_start, test_end):
    residual = data["residual"]
    tide = data["tide"]

    pred_residual_list = []
    true_residual_list = []
    target_tide_list = []

    for start_t in range(test_start, test_end - horizon + 1):
        pred_residual = model.forecast_sequence(
            data=data,
            start_t=start_t,
            horizon=horizon,
        )

        true_residual = residual[start_t:start_t + horizon].T
        target_tide = tide[start_t:start_t + horizon].T

        pred_residual_list.append(pred_residual[None, :, :])
        true_residual_list.append(true_residual[None, :, :])
        target_tide_list.append(target_tide[None, :, :])

    pred_residual = np.concatenate(pred_residual_list, axis=0)
    true_residual = np.concatenate(true_residual_list, axis=0)
    target_tide = np.concatenate(target_tide_list, axis=0)

    pred_level = pred_residual + target_tide
    true_level = true_residual + target_tide

    seq_residual_metrics = regression_metrics(true_residual, pred_residual)
    seq_level_metrics = regression_metrics(true_level, pred_level)

    last_true_residual = true_residual[:, :, -1:]
    last_pred_residual = pred_residual[:, :, -1:]

    last_true_level = true_level[:, :, -1:]
    last_pred_level = pred_level[:, :, -1:]

    last_residual_metrics = regression_metrics(last_true_residual, last_pred_residual)
    last_level_metrics = regression_metrics(last_true_level, last_pred_level)

    return {
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


def evaluate_persistence_on_horizon(data, horizon, test_start, test_end):
    residual = data["residual"]
    tide = data["tide"]

    pred_residual_list = []
    true_residual_list = []
    target_tide_list = []

    for start_t in range(test_start, test_end - horizon + 1):
        pred_residual = persistence_forecast(
            data=data,
            start_t=start_t,
            horizon=horizon,
        )

        true_residual = residual[start_t:start_t + horizon].T
        target_tide = tide[start_t:start_t + horizon].T

        pred_residual_list.append(pred_residual[None, :, :])
        true_residual_list.append(true_residual[None, :, :])
        target_tide_list.append(target_tide[None, :, :])

    pred_residual = np.concatenate(pred_residual_list, axis=0)
    true_residual = np.concatenate(true_residual_list, axis=0)
    target_tide = np.concatenate(target_tide_list, axis=0)

    pred_level = pred_residual + target_tide
    true_level = true_residual + target_tide

    seq_residual_metrics = regression_metrics(true_residual, pred_residual)
    seq_level_metrics = regression_metrics(true_level, pred_level)

    last_true_residual = true_residual[:, :, -1:]
    last_pred_residual = pred_residual[:, :, -1:]

    last_true_level = true_level[:, :, -1:]
    last_pred_level = pred_level[:, :, -1:]

    last_residual_metrics = regression_metrics(last_true_residual, last_pred_residual)
    last_level_metrics = regression_metrics(last_true_level, last_pred_level)

    return {
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


def run_experiment(data, graphs, horizons, ridge_alpha, clip_residual, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    n_time = len(data["residual"])

    train_end = int(n_time * 0.7)
    val_end = int(n_time * 0.85)
    test_start = val_end
    test_end = n_time

    print("\nSplit:")
    print(f"  n_time     : {n_time}")
    print(f"  train_end  : {train_end}")
    print(f"  val_end    : {val_end}")
    print(f"  test_start : {test_start}")
    print(f"  test_end   : {test_end}")

    models = make_model_specs(
        graphs=graphs,
        ridge_alpha=ridge_alpha,
        clip_residual=clip_residual,
    )

    fitted_models = []

    print("\nFitting physics ODE models on training data...")

    for model in models:
        print(f"  Fitting {model.model_name}...")
        model.fit(data=data, train_end=train_end)
        fitted_models.append(model)

        coef_path = output_dir / f"{model.model_name}_coefficients.csv"
        model.coefficients_table().to_csv(coef_path, index=False)

    rows = []

    for horizon in horizons:
        print("\n" + "=" * 100)
        print(f"Evaluating horizon: {horizon}h")
        print("=" * 100)

        persistence_metrics = evaluate_persistence_on_horizon(
            data=data,
            horizon=horizon,
            test_start=test_start,
            test_end=test_end,
        )

        row = {
            "model_name": "persistence_deta_dt_zero",
            "equation_type": "deta_dt_zero",
            "horizon": horizon,
            "ridge_alpha": 0.0,
            "feature_terms": "none",
        }
        row.update(persistence_metrics)
        rows.append(row)

        print(
            f"[{horizon}h] persistence "
            f"seq_RMSE={row['seq_residual_RMSE']:.6f}, "
            f"seq_R2={row['seq_residual_R2']:.6f}, "
            f"last_RMSE={row['last_residual_RMSE']:.6f}, "
            f"last_R2={row['last_residual_R2']:.6f}"
        )

        for model in fitted_models:
            metrics = evaluate_model_on_horizon(
                model=model,
                data=data,
                horizon=horizon,
                test_start=test_start,
                test_end=test_end,
            )

            row = {
                "model_name": model.model_name,
                "equation_type": model.model_name,
                "horizon": horizon,
                "ridge_alpha": ridge_alpha,
                "feature_terms": "|".join(model.feature_terms),
            }
            row.update(metrics)
            rows.append(row)

            print(
                f"[{horizon}h] {model.model_name} "
                f"seq_RMSE={row['seq_residual_RMSE']:.6f}, "
                f"seq_R2={row['seq_residual_R2']:.6f}, "
                f"last_RMSE={row['last_residual_RMSE']:.6f}, "
                f"last_R2={row['last_residual_R2']:.6f}"
            )

        result_df = pd.DataFrame(rows)
        result_df.to_csv(output_dir / "physics_ode_metrics.csv", index=False)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_dir / "physics_ode_metrics.csv", index=False)

    best_seq = (
        result_df
        .sort_values(["horizon", "seq_residual_RMSE"])
        .groupby("horizon")
        .head(3)
    )

    best_last = (
        result_df
        .sort_values(["horizon", "last_residual_RMSE"])
        .groupby("horizon")
        .head(3)
    )

    best_seq.to_csv(output_dir / "best_physics_models_by_sequence_rmse.csv", index=False)
    best_last.to_csv(output_dir / "best_physics_models_by_last_step_rmse.csv", index=False)

    print("\n" + "=" * 100)
    print("PHYSICS ODE BASELINE FINISHED")
    print("=" * 100)

    display_cols = [
        "horizon",
        "model_name",
        "seq_residual_RMSE",
        "seq_residual_R2",
        "last_residual_RMSE",
        "last_residual_R2",
    ]

    print("\nTop models by sequence RMSE:")
    print(best_seq[display_cols])

    print("\nTop models by last-step RMSE:")
    print(best_last[display_cols])

    print(f"\nSaved metrics to: {output_dir / 'physics_ode_metrics.csv'}")
    print(f"Saved best sequence models to: {output_dir / 'best_physics_models_by_sequence_rmse.csv'}")
    print(f"Saved best last-step models to: {output_dir / 'best_physics_models_by_last_step_rmse.csv'}")


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
        "--ridge-alpha",
        type=float,
        default=1e-2,
        help="Ridge regularization for fitting ODE coefficients.",
    )

    parser.add_argument(
        "--clip-residual",
        type=float,
        default=2.0,
        help="Clip recursive ODE residual prediction to avoid numerical explosion. Use 0 to disable.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/physics_multi_ode_baselines",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    clip_residual = args.clip_residual
    if clip_residual <= 0:
        clip_residual = None

    print("\nLoading NOAA data for physics ODE baselines...")
    station_ids, frames, station_meta = build_node_frames(data_root)

    print("\nBuilding physics arrays...")
    data = make_physics_arrays(frames)

    print("\nBuilding graphs...")
    graphs = build_graphs(
        station_meta=station_meta,
        residual=data["residual"],
    )

    print("\nAvailable physics ODE models:")
    print("  1. persistence_deta_dt_zero")
    print("  2. ode_decay")
    print("  3. ode_local_forcing")
    print("  4. ode_graph_diffusion_distance")
    print("  5. ode_graph_diffusion_corr")
    print("  6. ode_graph_diffusion_fusion")
    print("  7. ode_forced_graph_distance")
    print("  8. ode_forced_graph_corr")
    print("  9. ode_forced_graph_fusion")
    print("  10. ode_bathymetry_forced_graph_distance")
    print("  11. ode_bathymetry_forced_graph_corr")
    print("  12. ode_bathymetry_forced_graph_fusion")

    run_experiment(
        data=data,
        graphs=graphs,
        horizons=args.horizons,
        ridge_alpha=args.ridge_alpha,
        clip_residual=clip_residual,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()