from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed_multiyear_2023_2025"
DEPTH_PATH = ROOT / "data" / "processed" / "gebco_station_depth.csv"
OUT_DIR = ROOT / "数据整理" / "outputs" / "multistate_physics_baseline"

STATION_IDS = [
    "8461490",
    "8510560",
    "8516945",
    "8518750",
    "8531680",
    "8534720",
    "8536110",
]

HORIZONS = [1, 6, 12, 24]
GRAVITY = 9.80665


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def row_normalize(adj: np.ndarray) -> np.ndarray:
    row_sum = adj.sum(axis=1, keepdims=True)
    return (adj / np.maximum(row_sum, 1e-8)).astype(np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "Bias": float(np.mean(y_pred - y_true)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_long_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["station_id"] = df["station_id"].astype(str)
    return df


def make_hourly_feature_table() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    water = load_long_csv("water_tide_residual_long.csv")
    era5 = load_long_csv("era5_station_hourly.csv")
    currents = load_long_csv("surface_currents_station_daily.csv")
    waves = load_long_csv("wave_direction_speed_station_3hourly.csv")
    depth = pd.read_csv(DEPTH_PATH)
    depth["station_id"] = depth["station_id"].astype(str)

    # Build an hourly station-time grid from NOAA residuals and align lower-frequency forcings.
    base = water[["datetime", "station_id", "water_level", "tide", "residual", "sigma"]].copy()
    base = base.sort_values(["station_id", "datetime"])

    era5_cols = [
        "datetime",
        "station_id",
        "u10",
        "v10",
        "msl",
        "wind_speed",
        "wind_stress_u_proxy",
        "wind_stress_v_proxy",
    ]
    base = base.merge(era5[era5_cols], on=["datetime", "station_id"], how="left")

    current_cols = ["datetime", "station_id", "uo", "vo", "current_speed_mps"]
    wave_cols = [
        "datetime",
        "station_id",
        "wave_height",
        "wave_peak_period",
        "wave_mean_period",
        "wave_direction",
        "wave_stokes_drift_x",
        "wave_stokes_drift_y",
        "wave_speed_from_peak_period_mps",
        "wave_dir_x_from_mps",
        "wave_dir_y_from_mps",
    ]

    # Time interpolation is done within each station on the NOAA hourly grid.
    frames = []
    for sid in STATION_IDS:
        b = base[base["station_id"] == sid].set_index("datetime").sort_index()
        c = currents[currents["station_id"] == sid][current_cols].set_index("datetime").sort_index()
        w = waves[waves["station_id"] == sid][wave_cols].set_index("datetime").sort_index()

        b = b.join(c.drop(columns=["station_id"]), how="left")
        b = b.join(w.drop(columns=["station_id"]), how="left")

        interp_cols = [
            "uo",
            "vo",
            "current_speed_mps",
            "wave_height",
            "wave_peak_period",
            "wave_mean_period",
            "wave_direction",
            "wave_stokes_drift_x",
            "wave_stokes_drift_y",
            "wave_speed_from_peak_period_mps",
            "wave_dir_x_from_mps",
            "wave_dir_y_from_mps",
        ]
        b[interp_cols] = b[interp_cols].interpolate(method="time").ffill().bfill()
        b["station_id"] = sid
        frames.append(b.reset_index())

    df = pd.concat(frames, ignore_index=True)
    df = df.merge(
        depth[["station_id", "depth", "station_lat", "station_lon"]],
        on="station_id",
        how="left",
    )

    msl_ref = float(df["msl"].mean())
    df["pressure_anom"] = df["msl"] - msl_ref
    df["inverse_depth"] = 1.0 / np.maximum(df["depth"], 0.5)
    df["wave_energy"] = df["wave_height"] ** 2
    df["wave_energy_flux"] = df["wave_height"] ** 2 * df["wave_speed_from_peak_period_mps"]
    df["wind_wave_alignment"] = (
        df["wind_stress_u_proxy"] * df["wave_dir_x_from_mps"]
        + df["wind_stress_v_proxy"] * df["wave_dir_y_from_mps"]
    )
    df["wave_setup_proxy"] = df["wave_energy"] * df["inverse_depth"]

    station_meta = (
        df[["station_id", "station_lat", "station_lon", "depth"]]
        .drop_duplicates("station_id")
        .set_index("station_id")
        .loc[STATION_IDS]
        .reset_index()
    )
    adj = build_distance_graph(station_meta)
    return df, station_meta, adj


def build_distance_graph(station_meta: pd.DataFrame) -> np.ndarray:
    coords = station_meta[["station_lat", "station_lon"]].to_numpy(dtype=float)
    n = len(coords)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
    sigma = np.median(dist[dist > 0])
    adj = np.exp(-dist / sigma)
    np.fill_diagonal(adj, 0.0)
    return row_normalize(adj)


def build_arrays(df: pd.DataFrame, adj: np.ndarray) -> dict[str, np.ndarray]:
    common_times = sorted(set.intersection(*[
        set(df.loc[df["station_id"] == sid, "datetime"]) for sid in STATION_IDS
    ]))
    time_index = pd.to_datetime(common_times)

    arrays = {"time": np.array(time_index)}
    cols = [
        "water_level",
        "tide",
        "residual",
        "u10",
        "v10",
        "msl",
        "wind_speed",
        "wind_stress_u_proxy",
        "wind_stress_v_proxy",
        "pressure_anom",
        "uo",
        "vo",
        "current_speed_mps",
        "wave_height",
        "wave_energy",
        "wave_energy_flux",
        "wave_setup_proxy",
        "wave_stokes_drift_x",
        "wave_stokes_drift_y",
        "wave_dir_x_from_mps",
        "wave_dir_y_from_mps",
        "wind_wave_alignment",
        "depth",
        "inverse_depth",
    ]

    for col in cols:
        values = []
        for sid in STATION_IDS:
            s = (
                df[df["station_id"] == sid]
                .set_index("datetime")
                .loc[time_index, col]
                .to_numpy(dtype=np.float32)
            )
            values.append(s)
        arrays[col] = np.stack(values, axis=1).astype(np.float32)
        arrays[col] = np.nan_to_num(arrays[col], nan=0.0, posinf=0.0, neginf=0.0)

    arrays["adj"] = adj.astype(np.float32)
    arrays["graph_eta"] = arrays["adj"] @ arrays["residual"].T
    arrays["graph_eta"] = arrays["graph_eta"].T.astype(np.float32)
    return arrays


@dataclass
class OdeSpec:
    name: str
    state_names: list[str]
    target_names: list[str]
    eta_terms: list[str]
    u_terms: list[str] | None = None
    v_terms: list[str] | None = None
    w_terms: list[str] | None = None


class LinearMultistateODE:
    def __init__(self, spec: OdeSpec, ridge_alpha: float = 1.0, clip_eta: float = 3.0):
        self.spec = spec
        self.ridge_alpha = ridge_alpha
        self.clip_eta = clip_eta
        self.scalers: dict[str, StandardScaler] = {}
        self.models: dict[str, Ridge] = {}
        self.fitted = False

    def _features_for_target(self, state: dict[str, np.ndarray], data: dict[str, np.ndarray], t: int, target: str) -> np.ndarray:
        terms = {
            "eta": self.spec.eta_terms,
            "u": self.spec.u_terms or [],
            "v": self.spec.v_terms or [],
            "W": self.spec.w_terms or [],
        }[target]

        eta = state["eta"]
        u = state.get("u", data["uo"][t])
        v = state.get("v", data["vo"][t])
        w = state.get("W", data["wave_setup_proxy"][t])
        adj = data["adj"]
        graph_eta = adj @ eta
        graph_delta_eta = graph_eta - eta
        graph_u = adj @ u
        graph_v = adj @ v
        graph_w = adj @ w

        lookup = {
            "constant": np.ones_like(eta),
            "eta": eta,
            "u": u,
            "v": v,
            "W": w,
            "graph_eta": graph_eta,
            "graph_delta_eta": graph_delta_eta,
            "graph_u_delta": graph_u - u,
            "graph_v_delta": graph_v - v,
            "graph_W_delta": graph_w - w,
            "transport_div_proxy": graph_u + graph_v - u - v,
            "pressure_anom": data["pressure_anom"][t],
            "wind_stress_u": data["wind_stress_u_proxy"][t],
            "wind_stress_v": data["wind_stress_v_proxy"][t],
            "wind_speed": data["wind_speed"][t],
            "inverse_depth": data["inverse_depth"][t],
            "depth": data["depth"][t],
            "wave_energy": data["wave_energy"][t],
            "wave_energy_flux": data["wave_energy_flux"][t],
            "wave_setup_proxy": data["wave_setup_proxy"][t],
            "wave_stokes_drift_x": data["wave_stokes_drift_x"][t],
            "wave_stokes_drift_y": data["wave_stokes_drift_y"][t],
            "wave_dir_x": data["wave_dir_x_from_mps"][t],
            "wave_dir_y": data["wave_dir_y_from_mps"][t],
            "wind_wave_alignment": data["wind_wave_alignment"][t],
            "pressure_x_depth": data["pressure_anom"][t] * data["inverse_depth"][t],
            "wind_u_x_depth": data["wind_stress_u_proxy"][t] * data["inverse_depth"][t],
            "wind_v_x_depth": data["wind_stress_v_proxy"][t] * data["inverse_depth"][t],
            "friction_u": -u * np.maximum(np.sqrt(u**2 + v**2), 1e-6) * data["inverse_depth"][t],
            "friction_v": -v * np.maximum(np.sqrt(u**2 + v**2), 1e-6) * data["inverse_depth"][t],
            "pressure_gradient_proxy": graph_delta_eta * GRAVITY,
        }
        return np.stack([lookup[term] for term in terms], axis=1).astype(np.float32)

    def fit(self, data: dict[str, np.ndarray], train_end: int) -> None:
        for target in self.spec.target_names:
            x_rows = []
            y_rows = []
            for t in range(train_end - 1):
                state = self.state_from_data(data, t)
                x_rows.append(self._features_for_target(state, data, t, target))
                y_rows.append(self.target_delta_from_data(data, t, target)[:, None])
            x = np.concatenate(x_rows, axis=0)
            y = np.concatenate(y_rows, axis=0).reshape(-1)
            scaler = StandardScaler()
            model = Ridge(alpha=self.ridge_alpha)
            model.fit(scaler.fit_transform(x), y)
            self.scalers[target] = scaler
            self.models[target] = model
        self.fitted = True

    def state_from_data(self, data: dict[str, np.ndarray], t: int) -> dict[str, np.ndarray]:
        state = {"eta": data["residual"][t].copy()}
        if "u" in self.spec.state_names:
            state["u"] = data["uo"][t].copy()
        if "v" in self.spec.state_names:
            state["v"] = data["vo"][t].copy()
        if "W" in self.spec.state_names:
            state["W"] = data["wave_setup_proxy"][t].copy()
        return state

    def target_delta_from_data(self, data: dict[str, np.ndarray], t: int, target: str) -> np.ndarray:
        if target == "eta":
            return data["residual"][t + 1] - data["residual"][t]
        if target == "u":
            return data["uo"][t + 1] - data["uo"][t]
        if target == "v":
            return data["vo"][t + 1] - data["vo"][t]
        if target == "W":
            return data["wave_setup_proxy"][t + 1] - data["wave_setup_proxy"][t]
        raise ValueError(target)

    def step(self, state: dict[str, np.ndarray], data: dict[str, np.ndarray], t: int) -> dict[str, np.ndarray]:
        next_state = {k: v.copy() for k, v in state.items()}
        for target in self.spec.target_names:
            x = self._features_for_target(state, data, t, target)
            dx = self.models[target].predict(self.scalers[target].transform(x)).astype(np.float32)
            key = "eta" if target == "eta" else target
            next_state[key] = state[key] + dx
        next_state["eta"] = np.clip(next_state["eta"], -self.clip_eta, self.clip_eta)
        return next_state

    def forecast_eta(self, data: dict[str, np.ndarray], start_t: int, horizon: int) -> np.ndarray:
        state = self.state_from_data(data, start_t - 1)
        preds = []
        for s in range(horizon):
            forcing_t = min(start_t + s - 1, len(data["residual"]) - 2)
            state = self.step(state, data, forcing_t)
            preds.append(state["eta"].copy())
        return np.stack(preds, axis=1)

    def coefficient_table(self) -> pd.DataFrame:
        rows = []
        target_terms = {
            "eta": self.spec.eta_terms,
            "u": self.spec.u_terms or [],
            "v": self.spec.v_terms or [],
            "W": self.spec.w_terms or [],
        }
        for target, model in self.models.items():
            for term, coef in zip(target_terms[target], model.coef_):
                rows.append({"model_name": self.spec.name, "target": target, "term": term, "coef_scaled_feature": coef})
        return pd.DataFrame(rows)


def make_specs() -> list[OdeSpec]:
    eta_local = [
        "constant",
        "eta",
        "graph_delta_eta",
        "pressure_anom",
        "wind_stress_u",
        "wind_stress_v",
        "wind_speed",
        "wave_energy",
    ]
    eta_uv = eta_local + [
        "u",
        "v",
        "transport_div_proxy",
        "pressure_x_depth",
        "wind_u_x_depth",
        "wind_v_x_depth",
    ]
    eta_uvw = eta_uv + [
        "W",
        "graph_W_delta",
        "wave_energy_flux",
        "wind_wave_alignment",
        "wave_stokes_drift_x",
        "wave_stokes_drift_y",
    ]
    u_terms = [
        "constant",
        "u",
        "v",
        "graph_u_delta",
        "pressure_gradient_proxy",
        "wind_u_x_depth",
        "friction_u",
        "wave_stokes_drift_x",
        "wave_dir_x",
    ]
    v_terms = [
        "constant",
        "v",
        "u",
        "graph_v_delta",
        "pressure_gradient_proxy",
        "wind_v_x_depth",
        "friction_v",
        "wave_stokes_drift_y",
        "wave_dir_y",
    ]
    w_terms = [
        "constant",
        "W",
        "graph_W_delta",
        "wave_energy",
        "wave_energy_flux",
        "wind_wave_alignment",
        "inverse_depth",
    ]
    return [
        OdeSpec(
            name="eta_forced_graph_ode",
            state_names=["eta"],
            target_names=["eta"],
            eta_terms=eta_local,
        ),
        OdeSpec(
            name="eta_uv_graph_swe_ode",
            state_names=["eta", "u", "v"],
            target_names=["eta", "u", "v"],
            eta_terms=eta_uv,
            u_terms=u_terms,
            v_terms=v_terms,
        ),
        OdeSpec(
            name="eta_uv_wave_graph_swe_ode",
            state_names=["eta", "u", "v", "W"],
            target_names=["eta", "u", "v", "W"],
            eta_terms=eta_uvw,
            u_terms=u_terms,
            v_terms=v_terms,
            w_terms=w_terms,
        ),
    ]


def evaluate_model(model: LinearMultistateODE, data: dict[str, np.ndarray], horizon: int, test_start: int, test_end: int) -> dict:
    pred_list = []
    true_list = []
    tide_list = []
    for start_t in range(test_start, test_end - horizon + 1, horizon):
        pred = model.forecast_eta(data, start_t, horizon)
        true = data["residual"][start_t:start_t + horizon].T
        tide = data["tide"][start_t:start_t + horizon].T
        pred_list.append(pred[None, :, :])
        true_list.append(true[None, :, :])
        tide_list.append(tide[None, :, :])

    pred_res = np.concatenate(pred_list, axis=0)
    true_res = np.concatenate(true_list, axis=0)
    tide = np.concatenate(tide_list, axis=0)
    pred_level = pred_res + tide
    true_level = true_res + tide

    seq_res = regression_metrics(true_res, pred_res)
    last_res = regression_metrics(true_res[:, :, -1], pred_res[:, :, -1])
    seq_level = regression_metrics(true_level, pred_level)
    last_level = regression_metrics(true_level[:, :, -1], pred_level[:, :, -1])

    row = {
        "model_name": model.spec.name,
        "horizon": horizon,
        "n_samples": pred_res.shape[0],
    }
    for prefix, metrics in [
        ("seq_residual", seq_res),
        ("last_residual", last_res),
        ("seq_sea_level", seq_level),
        ("last_sea_level", last_level),
    ]:
        for k, v in metrics.items():
            row[f"{prefix}_{k}"] = v
    return row


def evaluate_persistence(data: dict[str, np.ndarray], horizon: int, test_start: int, test_end: int) -> dict:
    pred_list = []
    true_list = []
    tide_list = []
    for start_t in range(test_start, test_end - horizon + 1, horizon):
        eta0 = data["residual"][start_t - 1]
        pred = np.repeat(eta0[:, None], horizon, axis=1)
        true = data["residual"][start_t:start_t + horizon].T
        tide = data["tide"][start_t:start_t + horizon].T
        pred_list.append(pred[None, :, :])
        true_list.append(true[None, :, :])
        tide_list.append(tide[None, :, :])
    pred_res = np.concatenate(pred_list, axis=0)
    true_res = np.concatenate(true_list, axis=0)
    tide = np.concatenate(tide_list, axis=0)
    row = {
        "model_name": "persistence_deta_dt_zero",
        "horizon": horizon,
        "n_samples": pred_res.shape[0],
    }
    for prefix, yt, yp in [
        ("seq_residual", true_res, pred_res),
        ("last_residual", true_res[:, :, -1], pred_res[:, :, -1]),
        ("seq_sea_level", true_res + tide, pred_res + tide),
        ("last_sea_level", true_res[:, :, -1] + tide[:, :, -1], pred_res[:, :, -1] + tide[:, :, -1]),
    ]:
        for k, v in regression_metrics(yt, yp).items():
            row[f"{prefix}_{k}"] = v
    return row


def plot_metrics(metrics: pd.DataFrame, out_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    label_map = {
        "persistence_deta_dt_zero": "Persistence",
        "eta_forced_graph_ode": "Eta ODE",
        "eta_uv_graph_swe_ode": "Eta+Current SWE ODE",
        "eta_uv_wave_graph_swe_ode": "Eta+Current+Wave SWE ODE",
    }
    color_map = {
        "Persistence": "#6B7280",
        "Eta ODE": "#2563EB",
        "Eta+Current SWE ODE": "#059669",
        "Eta+Current+Wave SWE ODE": "#DC2626",
    }
    metrics = metrics.copy()
    metrics["label"] = metrics["model_name"].map(label_map)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    for label, sub in metrics.groupby("label"):
        sub = sub.sort_values("horizon")
        axes[0].plot(sub["horizon"], sub["last_residual_RMSE"], marker="o", linewidth=2.2, label=label, color=color_map[label])
        axes[1].plot(sub["horizon"], sub["last_sea_level_RMSE"], marker="o", linewidth=2.2, label=label, color=color_map[label])

    axes[0].set_title("Last-step residual RMSE")
    axes[1].set_title("Last-step sea-level RMSE")
    for ax in axes:
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel("RMSE (m)")
        ax.set_xticks(HORIZONS)
        ax.legend(frameon=True, fontsize=8)
    fig.suptitle("Pure Physics ODE Baseline Comparison (2023-2025 Test Split)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "physics_baseline_rmse_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    pivot = metrics.pivot(index="label", columns="horizon", values="last_residual_RMSE")
    pivot = pivot.loc[[label_map[k] for k in label_map]]
    im = ax.imshow(pivot.values, cmap="YlGnBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[f"{h}h" for h in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Residual RMSE Heatmap: lower is better")
    fig.colorbar(im, ax=ax, label="RMSE (m)")
    fig.tight_layout()
    fig.savefig(out_dir / "physics_baseline_residual_rmse_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and aligning multi-year features...")
    table, station_meta, adj = make_hourly_feature_table()
    table.to_csv(OUT_DIR / "aligned_multistate_hourly_features.csv", index=False)
    station_meta.to_csv(OUT_DIR / "station_meta_used.csv", index=False)
    np.savetxt(OUT_DIR / "distance_graph_adjacency.csv", adj, delimiter=",")

    data = build_arrays(table, adj)
    n_time = len(data["time"])
    train_end = int(n_time * args.train_ratio)
    val_end = int(n_time * (args.train_ratio + args.val_ratio))
    test_start = val_end
    test_end = n_time
    print(f"Time steps: {n_time}, train_end={train_end}, val_end={val_end}, test={test_end - test_start}")

    rows = []
    for horizon in HORIZONS:
        rows.append(evaluate_persistence(data, horizon, test_start, test_end))

    coef_tables = []
    for spec in make_specs():
        print(f"Fitting {spec.name}...")
        model = LinearMultistateODE(spec=spec, ridge_alpha=args.ridge_alpha, clip_eta=args.clip_eta)
        model.fit(data, train_end=train_end)
        coef_tables.append(model.coefficient_table())
        for horizon in HORIZONS:
            row = evaluate_model(model, data, horizon, test_start, test_end)
            rows.append(row)
            print(
                f"  {spec.name} {horizon}h "
                f"last_res_RMSE={row['last_residual_RMSE']:.4f} "
                f"last_res_R2={row['last_residual_R2']:.4f}"
            )

    metrics = pd.DataFrame(rows).sort_values(["horizon", "last_residual_RMSE"])
    metrics.to_csv(OUT_DIR / "physics_baseline_metrics.csv", index=False)
    pd.concat(coef_tables, ignore_index=True).to_csv(OUT_DIR / "physics_baseline_coefficients.csv", index=False)

    best = metrics.sort_values(["horizon", "last_residual_RMSE"]).groupby("horizon").head(1)
    best.to_csv(OUT_DIR / "best_physics_baseline_by_horizon.csv", index=False)
    plot_metrics(metrics, OUT_DIR)

    print("\nBest by horizon:")
    print(best[["horizon", "model_name", "last_residual_RMSE", "last_residual_R2", "last_sea_level_RMSE"]].to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure multistate graph shallow-water ODE baselines.")
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--clip-eta", type=float, default=3.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
