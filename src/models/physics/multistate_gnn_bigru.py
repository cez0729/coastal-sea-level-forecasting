from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = REPO_ROOT
DATA_DIR = ROOT / "data" / "processed_multiyear_2023_2025"
DEPTH_PATH = ROOT / "data" / "processed" / "gebco_station_depth.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "multistate_physics_loss_gnn_bigru_v2"

STATION_IDS = [
    "8461490",
    "8510560",
    "8516945",
    "8518750",
    "8531680",
    "8534720",
    "8536110",
]

STATE_NAMES = ["eta", "u", "v", "W"]
GRAVITY = 9.80665


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_long_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["station_id"] = df["station_id"].astype(str)
    return df


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


def make_hourly_feature_table() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    water = load_long_csv("water_tide_residual_long.csv")
    era5 = load_long_csv("era5_station_hourly.csv")
    currents = load_long_csv("surface_currents_station_daily.csv")
    waves = load_long_csv("wave_direction_speed_station_3hourly.csv")
    depth = pd.read_csv(DEPTH_PATH)
    depth["station_id"] = depth["station_id"].astype(str)

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
    df = df.merge(depth[["station_id", "depth", "station_lat", "station_lon"]], on="station_id", how="left")

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


def build_arrays(df: pd.DataFrame, adj: np.ndarray) -> dict[str, np.ndarray]:
    common_times = sorted(set.intersection(*[set(df.loc[df["station_id"] == sid, "datetime"]) for sid in STATION_IDS]))
    time_index = pd.to_datetime(common_times)
    arrays: dict[str, np.ndarray] = {"time": np.array(time_index)}

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
        "wave_peak_period",
        "wave_mean_period",
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
        arr = np.stack(values, axis=1).astype(np.float32)
        arrays[col] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    arrays["adj_distance"] = adj.astype(np.float32)
    return arrays


def make_graph_priors(arrays: dict[str, np.ndarray], train_end: int) -> dict[str, np.ndarray]:
    n = len(STATION_IDS)
    identity = np.eye(n, dtype=np.float32)
    distance = arrays["adj_distance"].astype(np.float32)

    residual = arrays["residual"][:train_end]
    corr = np.corrcoef(residual.T)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    corr = row_normalize(corr).astype(np.float32)

    return {"identity": identity, "distance": distance, "corr": corr}


class MultistateWindowDataset(Dataset):
    def __init__(
        self,
        x_scaled: np.ndarray,
        states: np.ndarray,
        tide: np.ndarray,
        physics_scaled: np.ndarray,
        window: int,
        horizon: int,
        start: int,
        end: int,
        physics_forcing_mode: str,
        stride: int = 1,
    ):
        if physics_forcing_mode not in {"last_input", "future"}:
            raise ValueError("physics_forcing_mode must be last_input or future")
        self.x_scaled = x_scaled
        self.states = states
        self.tide = tide
        self.physics_scaled = physics_scaled
        self.window = int(window)
        self.horizon = int(horizon)
        self.physics_forcing_mode = physics_forcing_mode
        self.indices = np.arange(start + window, end - horizon + 1, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = int(self.indices[idx])
        xb = self.x_scaled[t - self.window: t]
        target_states = self.states[t: t + self.horizon].transpose(1, 0, 2)
        tide = self.tide[t: t + self.horizon].T
        init_states = self.states[t - 1]
        if self.physics_forcing_mode == "future":
            phys_seq = self.physics_scaled[t: t + self.horizon].transpose(1, 0, 2)
        else:
            phys_seq = np.repeat(self.physics_scaled[t - 1: t], self.horizon, axis=0).transpose(1, 0, 2)
        return (
            torch.from_numpy(xb.astype(np.float32)),
            torch.from_numpy(target_states.astype(np.float32)),
            torch.from_numpy(tide.astype(np.float32)),
            torch.from_numpy(init_states.astype(np.float32)),
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
        init = np.asarray(init_weights, dtype=np.float32)
        init = init / max(float(init.sum()), 1e-8)
        self.logits = nn.Parameter(torch.tensor(np.log(init + 1e-8), dtype=torch.float32))

    def weights(self):
        return torch.softmax(self.logits, dim=0)

    def forward(self):
        return torch.einsum("g,gij->ij", self.weights(), self.priors)

    def weight_dict(self):
        w = self.weights().detach().cpu().numpy()
        return {name: float(w[i]) for i, name in enumerate(self.graph_names)}


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        x = torch.einsum("ij,btjf->btif", adj, x)
        return self.linear(x)


def bidirectional_final_hidden(h_n: torch.Tensor) -> torch.Tensor:
    """Concatenate the final forward and backward states of the top BiGRU layer."""
    if h_n.ndim != 3 or h_n.shape[0] < 2 or h_n.shape[0] % 2 != 0:
        raise ValueError(f"Expected bidirectional GRU state [2*layers, batch, hidden], got {tuple(h_n.shape)}")
    return torch.cat([h_n[-2], h_n[-1]], dim=-1)


class MultistateGNNBiGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        graph_priors: dict[str, np.ndarray],
        graph_init_weights: list[float],
        gnn_hidden: int,
        gru_hidden: int,
        horizon: int,
        dropout: float,
        num_states: int = 4,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.num_states = int(num_states)
        self.graph = LearnableGraphFusion(graph_priors, graph_init_weights)
        self.gcn1 = GraphConvolution(input_dim, gnn_hidden)
        self.gcn2 = GraphConvolution(gnn_hidden, gnn_hidden)
        self.norm = nn.LayerNorm(gnn_hidden)
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
            nn.Linear(gru_hidden, horizon * num_states),
        )

    def forward(self, x):
        adj = self.graph()
        h = torch.relu(self.gcn1(x, adj))
        h = torch.relu(self.gcn2(h, adj))
        h = self.dropout(self.norm(h))
        bsz, steps, nodes, hidden = h.shape
        h = h.permute(0, 2, 1, 3).reshape(bsz * nodes, steps, hidden)
        _, h_n = self.gru(h)
        pred = self.head(bidirectional_final_hidden(h_n))
        return pred.reshape(bsz, nodes, self.horizon, self.num_states)


class MultistatePhysicsODE(nn.Module):
    def __init__(self, num_nodes: int, num_forcing: int, num_states: int = 4):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_forcing = num_forcing
        self.num_states = num_states
        self.extra_names = [
            "u",
            "v",
            "W",
            "transport_div_proxy",
            "pressure_gradient_proxy",
        ]
        self.bias = nn.Parameter(torch.zeros(num_states, num_nodes))
        self.raw_decay = nn.Parameter(torch.full((num_states,), -2.7))
        self.raw_kappa = nn.Parameter(torch.full((num_states,), -3.2))
        self.beta = nn.Parameter(torch.zeros(num_states, len(self.extra_names) + num_forcing))

    def coefficients(self) -> dict[str, float]:
        out = {}
        decay = torch.nn.functional.softplus(self.raw_decay).detach().cpu().numpy()
        kappa = torch.nn.functional.softplus(self.raw_kappa).detach().cpu().numpy()
        beta = self.beta.detach().cpu().numpy()
        for s, name in enumerate(STATE_NAMES):
            out[f"{name}_mean_bias"] = float(self.bias[s].detach().cpu().mean())
            out[f"{name}_decay"] = float(decay[s])
            out[f"{name}_kappa"] = float(kappa[s])
            for i, value in enumerate(beta[s]):
                out[f"{name}_beta_{i}"] = float(value)
        return out

    def forward(self, pred_states, init_states, phys_seq, adj):
        # pred_states: [B, N, H, S], init_states: [B, N, S]
        prev = torch.cat([init_states.unsqueeze(2), pred_states[:, :, :-1, :]], dim=2)
        lhs = pred_states - prev

        spatial = torch.einsum("ij,bjhs->bihs", adj, prev) - prev
        eta = prev[..., 0]
        u = prev[..., 1]
        v = prev[..., 2]
        W = prev[..., 3]
        graph_u = torch.einsum("ij,bjh->bih", adj, u)
        graph_v = torch.einsum("ij,bjh->bih", adj, v)
        transport = (graph_u - u) + (graph_v - v)
        pressure_gradient = GRAVITY * spatial[..., 0]

        extra = torch.stack([u, v, W, transport, pressure_gradient], dim=-1)
        extra = torch.cat([extra, phys_seq], dim=-1)
        forcing = torch.einsum("sk,bnhk->bnhs", self.beta, extra)

        decay = torch.nn.functional.softplus(self.raw_decay).view(1, 1, 1, -1)
        kappa = torch.nn.functional.softplus(self.raw_kappa).view(1, 1, 1, -1)
        rhs = self.bias.T.view(1, self.num_nodes, 1, self.num_states) - decay * prev + kappa * spatial + forcing
        residual = lhs - rhs
        return torch.mean(residual ** 2), residual


class CombinedMultistateLoss(nn.Module):
    def __init__(
        self,
        physics_ode: MultistatePhysicsODE,
        state_scale: np.ndarray,
        delta_scale: np.ndarray,
        aux_weight: float,
        physics_loss_type: str,
        physics_state_weights: list[float],
        last_step_weight: float,
        ode_coef_l2: float,
    ):
        super().__init__()
        self.physics_ode = physics_ode
        self.register_buffer("state_scale", torch.tensor(state_scale.reshape(1, 1, 1, -1), dtype=torch.float32))
        self.register_buffer("delta_scale", torch.tensor(delta_scale.reshape(1, 1, 1, -1), dtype=torch.float32))
        self.register_buffer("state_weights", torch.tensor(physics_state_weights, dtype=torch.float32).view(1, 1, 1, -1))
        self.aux_weight = float(aux_weight)
        self.physics_loss_type = physics_loss_type
        self.last_step_weight = float(last_step_weight)
        self.ode_coef_l2 = float(ode_coef_l2)
        self.huber = nn.SmoothL1Loss(beta=1.0, reduction="none")

    def data_loss(self, pred, target):
        scaled = (pred - target) / self.state_scale
        eta_loss = torch.mean(scaled[..., 0] ** 2)
        aux_loss = torch.mean(scaled[..., 1:] ** 2)
        last_loss = torch.mean(scaled[:, :, -1, 0] ** 2)
        return eta_loss + self.aux_weight * aux_loss, eta_loss, aux_loss, last_loss

    def physics_loss(self, pred, init_states, phys_seq, adj):
        _, residual = self.physics_ode(pred, init_states, phys_seq, adj)
        scaled = residual / self.delta_scale
        if self.physics_loss_type == "huber":
            loss_by_item = self.huber(scaled, torch.zeros_like(scaled))
        elif self.physics_loss_type == "mse":
            loss_by_item = scaled ** 2
        else:
            raise ValueError("physics_loss_type must be huber or mse")
        return torch.mean(loss_by_item * self.state_weights)

    def ode_regularization(self):
        reg = torch.mean(self.physics_ode.beta ** 2)
        reg = reg + torch.mean(self.physics_ode.bias ** 2)
        reg = reg + torch.mean(torch.nn.functional.softplus(self.physics_ode.raw_decay) ** 2)
        reg = reg + torch.mean(torch.nn.functional.softplus(self.physics_ode.raw_kappa) ** 2)
        return reg

    def forward(self, pred, target, init_states, phys_seq, adj, physics_lambda: float):
        data_loss, eta_loss, aux_loss, last_loss = self.data_loss(pred, target)
        physics_loss = self.physics_loss(pred, init_states, phys_seq, adj)
        ode_reg = self.ode_regularization()
        total = data_loss + self.last_step_weight * last_loss + float(physics_lambda) * physics_loss + self.ode_coef_l2 * ode_reg
        return total, {
            "data_loss": float(data_loss.detach().cpu()),
            "eta_data_loss": float(eta_loss.detach().cpu()),
            "aux_data_loss": float(aux_loss.detach().cpu()),
            "last_loss": float(last_loss.detach().cpu()),
            "physics_loss": float(physics_loss.detach().cpu()),
            "ode_reg": float(ode_reg.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
            "physics_lambda": float(physics_lambda),
        }


def physics_lambda_for_epoch(epoch: int, lambda_max: float, warmup_epochs: int, ramp_epochs: int) -> float:
    if lambda_max <= 0:
        return 0.0
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(lambda_max)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / float(ramp_epochs)))
    return float(lambda_max) * progress


@torch.no_grad()
def evaluate_loss(model, criterion, loader, device, physics_lambda: float) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n_batches = 0
    for xb, target, _, init_states, phys_seq in loader:
        xb = xb.to(device)
        target = target.to(device)
        init_states = init_states.to(device)
        phys_seq = phys_seq.to(device)
        pred = model(xb)
        _, parts = criterion(pred, target, init_states, phys_seq, model.graph(), physics_lambda)
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1
    return {k: v / max(1, n_batches) for k, v in totals.items()}


def train_model(
    model,
    physics_ode,
    train_loader,
    val_loader,
    args,
    state_scale: np.ndarray,
    delta_scale: np.ndarray,
    device,
) -> tuple[pd.DataFrame, float]:
    criterion = CombinedMultistateLoss(
        physics_ode=physics_ode,
        state_scale=state_scale,
        delta_scale=delta_scale,
        aux_weight=args.aux_weight,
        physics_loss_type=args.physics_loss_type,
        physics_state_weights=args.physics_state_weights,
        last_step_weight=args.last_step_weight,
        ode_coef_l2=args.ode_coef_l2,
    ).to(device)

    graph_params = list(model.graph.parameters())
    graph_param_ids = {id(p) for p in graph_params}
    base_params = [p for p in model.parameters() if id(p) not in graph_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": args.lr},
            {"params": graph_params, "lr": args.lr * args.graph_lr_mult},
            {"params": physics_ode.parameters(), "lr": args.lr * args.physics_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    rows = []

    for epoch in range(1, args.epochs + 1):
        current_lambda = physics_lambda_for_epoch(
            epoch, args.physics_lambda_max, args.physics_warmup_epochs, args.physics_ramp_epochs
        )
        model.train()
        physics_ode.train()
        train_totals: dict[str, float] = {}
        n_batches = 0

        for xb, target, _, init_states, phys_seq in train_loader:
            xb = xb.to(device)
            target = target.to(device)
            init_states = init_states.to(device)
            phys_seq = phys_seq.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss, parts = criterion(pred, target, init_states, phys_seq, model.graph(), current_lambda)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(physics_ode.parameters()), args.grad_clip)
            optimizer.step()
            for k, v in parts.items():
                train_totals[k] = train_totals.get(k, 0.0) + v
            n_batches += 1

        train_parts = {k: v / max(1, n_batches) for k, v in train_totals.items()}
        val_parts = evaluate_loss(model, criterion, val_loader, device, current_lambda)
        if args.selection_metric == "val_eta_data_loss":
            val_score = val_parts["eta_data_loss"]
        elif args.selection_metric == "val_last_loss":
            val_score = val_parts["last_loss"]
        elif args.selection_metric == "val_total_loss":
            val_score = val_parts["total_loss"]
        else:
            val_score = val_parts["data_loss"]
        scheduler.step(val_score)

        weights = model.graph.weight_dict()
        row = {
            "epoch": epoch,
            "selection_score": val_score,
            **{f"train_{k}": v for k, v in train_parts.items()},
            **{f"val_{k}": v for k, v in val_parts.items()},
            "w_identity": weights["identity"],
            "w_distance": weights["distance"],
            "w_corr": weights["corr"],
            **physics_ode.coefficients(),
        }
        rows.append(row)

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch={epoch:03d} lambda={current_lambda:.5f} "
                f"train_eta={train_parts['eta_data_loss']:.5f} train_phys={train_parts['physics_loss']:.5f} "
                f"val_eta={val_parts['eta_data_loss']:.5f} val_last={val_parts['last_loss']:.5f} "
                f"val_phys={val_parts['physics_loss']:.5f} select={val_score:.5f} "
                f"w=[{weights['identity']:.3f},{weights['distance']:.3f},{weights['corr']:.3f}]"
            )

        if val_score < best_val - args.min_delta:
            best_val = val_score
            bad_epochs = 0
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "physics_ode": {k: v.detach().cpu().clone() for k, v in physics_ode.state_dict().items()},
            }
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        physics_ode.load_state_dict(best_state["physics_ode"])

    return pd.DataFrame(rows), best_val


@torch.no_grad()
def predict(model, physics_ode, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    physics_ode.eval()
    preds, ys, tides = [], [], []
    phys_losses = []
    for xb, target, tide, init_states, phys_seq in loader:
        xb = xb.to(device)
        init_states_device = init_states.to(device)
        phys_seq_device = phys_seq.to(device)
        pred = model(xb)
        loss, _ = physics_ode(pred, init_states_device, phys_seq_device, model.graph())
        phys_losses.append(float(loss.detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        ys.append(target.numpy())
        tides.append(tide.numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(tides), float(np.mean(phys_losses))


def summarize_metrics(true_states: np.ndarray, pred_states: np.ndarray, tide: np.ndarray) -> dict[str, float]:
    true_residual = true_states[..., 0]
    pred_residual = pred_states[..., 0]
    true_level = true_residual + tide
    pred_level = pred_residual + tide

    out: dict[str, float] = {}
    for prefix, yt, yp in [
        ("seq_residual", true_residual, pred_residual),
        ("last_residual", true_residual[:, :, -1], pred_residual[:, :, -1]),
        ("seq_sea_level", true_level, pred_level),
        ("last_sea_level", true_level[:, :, -1], pred_level[:, :, -1]),
    ]:
        for key, value in regression_metrics(yt, yp).items():
            out[f"{prefix}_{key}"] = value

    for i, name in enumerate(STATE_NAMES[1:], start=1):
        for key, value in regression_metrics(true_states[..., i], pred_states[..., i]).items():
            out[f"seq_{name}_{key}"] = value
    return out


def plot_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    metrics = metrics.sort_values("horizon")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(metrics["horizon"], metrics["last_residual_R2"], marker="o", label="last-step residual R2")
    axes[0].plot(metrics["horizon"], metrics["seq_residual_R2"], marker="s", label="sequence residual R2")
    axes[0].axhline(0.9, color="crimson", linestyle="--", linewidth=1.2, label="R2=0.90 target")
    axes[0].set_xlabel("Forecast horizon (hours)")
    axes[0].set_ylabel("Residual R2")
    axes[0].set_ylim(min(-0.05, metrics[["last_residual_R2", "seq_residual_R2"]].min().min() - 0.05), 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(metrics["horizon"], metrics["last_residual_RMSE"], marker="o", label="last-step residual RMSE")
    axes[1].plot(metrics["horizon"], metrics["seq_residual_RMSE"], marker="s", label="sequence residual RMSE")
    axes[1].set_xlabel("Forecast horizon (hours)")
    axes[1].set_ylabel("RMSE (m)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Multistate physics-loss GNN-BiGRU v2")
    fig.tight_layout()
    fig.savefig(output_dir / "v2_residual_r2_rmse_summary.png", dpi=220)
    plt.close(fig)


def run_one_horizon(horizon: int, arrays: dict[str, np.ndarray], args, output_dir: Path, device) -> dict[str, float]:
    feature_cols = [
        "residual",
        "tide",
        "water_level",
        "u10",
        "v10",
        "wind_speed",
        "pressure_anom",
        "wind_stress_u_proxy",
        "wind_stress_v_proxy",
        "uo",
        "vo",
        "current_speed_mps",
        "wave_height",
        "wave_peak_period",
        "wave_mean_period",
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
    physics_cols = [
        "pressure_anom",
        "wind_stress_u_proxy",
        "wind_stress_v_proxy",
        "wind_speed",
        "current_speed_mps",
        "wave_energy",
        "wave_energy_flux",
        "wave_setup_proxy",
        "wave_stokes_drift_x",
        "wave_stokes_drift_y",
        "wave_dir_x_from_mps",
        "wave_dir_y_from_mps",
        "wind_wave_alignment",
        "inverse_depth",
    ]

    x_raw = np.stack([arrays[c] for c in feature_cols], axis=-1).astype(np.float32)
    physics_raw = np.stack([arrays[c] for c in physics_cols], axis=-1).astype(np.float32)
    states = np.stack(
        [
            arrays["residual"],
            arrays["uo"],
            arrays["vo"],
            arrays["wave_setup_proxy"],
        ],
        axis=-1,
    ).astype(np.float32)
    tide = arrays["tide"].astype(np.float32)

    n_time, nodes, feats = x_raw.shape
    train_end = int(n_time * args.train_ratio)
    val_end = int(n_time * (args.train_ratio + args.val_ratio))

    graph_priors = make_graph_priors(arrays, train_end)
    if args.graph_init == "identity":
        graph_init = [0.90, 0.05, 0.05]
    elif args.graph_init == "distance":
        graph_init = [0.10, 0.85, 0.05]
    elif args.graph_init == "corr":
        graph_init = [0.10, 0.05, 0.85]
    else:
        graph_init = [0.50, 0.35, 0.15]

    x_scaler = StandardScaler()
    x_scaler.fit(x_raw[:train_end].reshape(-1, feats))
    x_scaled = x_scaler.transform(x_raw.reshape(-1, feats)).reshape(n_time, nodes, feats).astype(np.float32)

    phys_scaler = StandardScaler()
    phys_scaler.fit(physics_raw[:train_end].reshape(-1, len(physics_cols)))
    phys_scaled = phys_scaler.transform(physics_raw.reshape(-1, len(physics_cols))).reshape(n_time, nodes, len(physics_cols)).astype(np.float32)

    state_scale = np.std(states[:train_end].reshape(-1, len(STATE_NAMES)), axis=0).astype(np.float32) + 1e-6
    delta_scale = np.std((states[1:train_end] - states[: train_end - 1]).reshape(-1, len(STATE_NAMES)), axis=0).astype(np.float32) + 1e-6

    train_ds = MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, 0, train_end, args.physics_forcing_mode, args.train_stride
    )
    val_ds = MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, train_end, val_end, args.physics_forcing_mode, 1
    )
    test_ds = MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, val_end, n_time, args.physics_forcing_mode, 1
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print("\n" + "=" * 80)
    print(f"horizon={horizon}h samples train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"input shape time={n_time} nodes={nodes} features={feats} device={device}")
    print(f"state_scale={dict(zip(STATE_NAMES, state_scale.round(6)))}")
    print(f"delta_scale={dict(zip(STATE_NAMES, delta_scale.round(6)))}")

    model = MultistateGNNBiGRU(
        input_dim=feats,
        graph_priors=graph_priors,
        graph_init_weights=graph_init,
        gnn_hidden=args.gnn_hidden,
        gru_hidden=args.gru_hidden,
        horizon=horizon,
        dropout=args.dropout,
        num_states=len(STATE_NAMES),
    ).to(device)
    physics_ode = MultistatePhysicsODE(nodes, len(physics_cols), len(STATE_NAMES)).to(device)

    history, best_val = train_model(
        model=model,
        physics_ode=physics_ode,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        state_scale=state_scale,
        delta_scale=delta_scale,
        device=device,
    )

    pred_states, true_states, target_tide, test_physics_loss = predict(model, physics_ode, test_loader, device)
    metrics = summarize_metrics(true_states, pred_states, target_tide)
    weights = model.graph.weight_dict()
    row = {
        "model": "multistate_physics_loss_gnn_bigru_v2",
        "horizon": horizon,
        "window": args.window,
        "physics_lambda_max": args.physics_lambda_max,
        "physics_forcing_mode": args.physics_forcing_mode,
        "aux_weight": args.aux_weight,
        "best_val_score": best_val,
        "test_physics_loss_unscaled": test_physics_loss,
        "learned_w_identity": weights["identity"],
        "learned_w_distance": weights["distance"],
        "learned_w_corr": weights["corr"],
        **{f"state_scale_{name}": float(state_scale[i]) for i, name in enumerate(STATE_NAMES)},
        **{f"delta_scale_{name}": float(delta_scale[i]) for i, name in enumerate(STATE_NAMES)},
        **physics_ode.coefficients(),
        **metrics,
    }

    horizon_dir = output_dir / f"horizon_{horizon}h"
    horizon_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(horizon_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(horizon_dir / "metrics.csv", index=False)
    np.savez_compressed(
        horizon_dir / "predictions.npz",
        pred_states=pred_states,
        true_states=true_states,
        target_tide=target_tide,
        target_start_times=np.array([str(arrays["time"][int(i)]) for i in test_ds.indices]),
        station_ids=np.array(STATION_IDS),
        state_names=np.array(STATE_NAMES),
        feature_cols=np.array(feature_cols),
        physics_cols=np.array(physics_cols),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "physics_ode_state_dict": physics_ode.state_dict(),
            "feature_cols": feature_cols,
            "physics_cols": physics_cols,
            "state_names": STATE_NAMES,
            "graph_init_weights": graph_init,
            "args": vars(args),
        },
        horizon_dir / "model.pt",
    )

    print("Test metrics:")
    for k in [
        "seq_residual_RMSE",
        "seq_residual_R2",
        "last_residual_RMSE",
        "last_residual_R2",
        "seq_sea_level_RMSE",
        "seq_sea_level_R2",
        "last_sea_level_RMSE",
        "last_sea_level_R2",
    ]:
        print(f"  {k}: {row[k]:.6f}")
    print(f"  graph weights: identity={weights['identity']:.4f}, distance={weights['distance']:.4f}, corr={weights['corr']:.4f}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Multistate physics-loss GNN-BiGRU v2 for 2023-2025 sea-level residuals")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 12, 24])
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=1)

    parser.add_argument("--gnn-hidden", type=int, default=48)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--graph-lr-mult", type=float, default=3.0)
    parser.add_argument("--physics-lr-mult", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=5)

    parser.add_argument("--physics-lambda-max", type=float, default=0.003)
    parser.add_argument("--physics-warmup-epochs", type=int, default=20)
    parser.add_argument("--physics-ramp-epochs", type=int, default=30)
    parser.add_argument("--physics-loss-type", default="huber", choices=["huber", "mse"])
    parser.add_argument("--physics-forcing-mode", default="last_input", choices=["last_input", "future"])
    parser.add_argument("--physics-state-weights", type=float, nargs=4, default=[1.0, 0.35, 0.35, 0.25])
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.10)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument("--selection-metric", default="val_eta_data_loss",
                        choices=["val_eta_data_loss", "val_data_loss", "val_last_loss", "val_total_loss"])
    parser.add_argument("--graph-init", default="mixed", choices=["mixed", "identity", "distance", "corr"])

    args = parser.parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading aligned 2023-2025 multistate data...")
    df, station_meta, distance_adj = make_hourly_feature_table()
    arrays = build_arrays(df, distance_adj)
    station_meta.to_csv(output_dir / "station_meta_used.csv", index=False)
    pd.DataFrame(distance_adj, index=STATION_IDS, columns=STATION_IDS).to_csv(output_dir / "distance_graph_adjacency.csv")
    print(f"Aligned hourly records: time={len(arrays['time'])}, stations={len(STATION_IDS)}")

    rows = []
    for horizon in args.horizons:
        rows.append(run_one_horizon(horizon, arrays, args, output_dir, device))

    summary = pd.DataFrame(rows).sort_values("horizon")
    summary_path = output_dir / "multistate_physics_loss_v2_metrics.csv"
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, output_dir)

    print("\n" + "=" * 80)
    print("Finished multistate physics-loss GNN-BiGRU v2.")
    print(f"Summary metrics: {summary_path}")
    print(summary[[
        "horizon",
        "seq_residual_RMSE",
        "seq_residual_R2",
        "last_residual_RMSE",
        "last_residual_R2",
        "seq_sea_level_R2",
        "last_sea_level_R2",
        "learned_w_identity",
        "learned_w_distance",
        "learned_w_corr",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
