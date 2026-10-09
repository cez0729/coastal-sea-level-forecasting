from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 2024, 2025, 3407]
BACKBONES = ["gwn", "bigru"]
VARIANTS = [
    "baseline",
    "physics_loss",
    "transport",
    "transport_physics_loss",
    "transport_coupled_physics_loss",
    "transport_shuffled",
    "transport_shuffled_physics_loss",
]
CONFIGS = [f"{backbone}_{variant}" for backbone in BACKBONES for variant in VARIANTS]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


confirm = load_module("transport_confirm", HERE / "134_confirmatory_hsdt_orc_chronological_refit.py")
p104 = confirm.p104
rolling = confirm.rolling
final4 = confirm.final4
priority1 = confirm.priority1
v2 = confirm.v2
v4 = final4.v4


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def row_normalize_torch(adjacency: torch.Tensor) -> torch.Tensor:
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def edge_direction(station_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lat_col = "station_lat" if "station_lat" in station_meta.columns else "lat"
    lon_col = "station_lon" if "station_lon" in station_meta.columns else "lon"
    lat = np.deg2rad(station_meta[lat_col].to_numpy(dtype=np.float32))
    lon = np.deg2rad(station_meta[lon_col].to_numpy(dtype=np.float32))
    # [destination i, source j]: direction from source j toward destination i.
    north = lat[:, None] - lat[None, :]
    east = (lon[:, None] - lon[None, :]) * np.cos((lat[:, None] + lat[None, :]) / 2.0)
    norm = np.sqrt(north**2 + east**2)
    norm[norm < 1e-8] = 1.0
    return (east / norm).astype(np.float32), (north / norm).astype(np.float32)


class TransportSupportBuilder(nn.Module):
    """Low-parameter causal support driven by forcing at the last input hour."""

    component_names = ["distance", "wind", "current", "wave", "pressure"]

    def __init__(
        self,
        distance_adjacency: np.ndarray,
        station_meta: pd.DataFrame,
        feature_cols: list[str],
        scaler_state: dict,
        shuffled: bool = False,
    ):
        super().__init__()
        east, north = edge_direction(station_meta)
        base = np.asarray(distance_adjacency, dtype=np.float32).copy()
        np.fill_diagonal(base, 0.0)
        empty = base.sum(axis=1) <= 1e-8
        if np.any(empty):
            base[empty] = np.asarray(distance_adjacency, dtype=np.float32)[empty]
        base = v2.row_normalize(base)
        self.register_buffer("distance", torch.tensor(base, dtype=torch.float32))
        self.register_buffer("edge_east", torch.tensor(east, dtype=torch.float32))
        self.register_buffer("edge_north", torch.tensor(north, dtype=torch.float32))
        self.register_buffer("feature_mean", torch.tensor(scaler_state["mean"], dtype=torch.float32))
        self.register_buffer("feature_scale", torch.tensor(scaler_state["scale"], dtype=torch.float32))
        self.feature_indices = {
            name: feature_cols.index(name)
            for name in [
                "wind_stress_u_proxy",
                "wind_stress_v_proxy",
                "uo",
                "vo",
                "wave_dir_x_from_mps",
                "wave_dir_y_from_mps",
                "pressure_anom",
            ]
        }
        self.shuffled = bool(shuffled)
        initial = torch.tensor([0.40, 0.20, 0.15, 0.15, 0.10], dtype=torch.float32)
        self.component_logits = nn.Parameter(torch.log(initial))
        self.raw_direction_scales = nn.Parameter(torch.zeros(4))
        self.sample_gate = nn.Linear(4, len(self.component_names), bias=False)
        nn.init.zeros_(self.sample_gate.weight)
        self.last_support: torch.Tensor | None = None
        self.last_gates: torch.Tensor | None = None

    def _raw_feature(self, x_last: torch.Tensor, name: str) -> torch.Tensor:
        index = self.feature_indices[name]
        return x_last[..., index] * self.feature_scale[index] + self.feature_mean[index]

    def _directional(self, vx: torch.Tensor, vy: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        magnitude = torch.sqrt(vx.square() + vy.square()).clamp_min(1e-8)
        vx = vx / magnitude
        vy = vy / magnitude
        projection = vx[:, None, :] * self.edge_east[None] + vy[:, None, :] * self.edge_north[None]
        weights = self.distance[None] * torch.exp(torch.clamp(scale * projection, min=-4.0, max=4.0))
        return row_normalize_torch(weights)

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        if x is None:
            return self.distance
        x_last = x[:, -1]
        wind_u = self._raw_feature(x_last, "wind_stress_u_proxy")
        wind_v = self._raw_feature(x_last, "wind_stress_v_proxy")
        current_u = self._raw_feature(x_last, "uo")
        current_v = self._raw_feature(x_last, "vo")
        wave_u = self._raw_feature(x_last, "wave_dir_x_from_mps")
        wave_v = self._raw_feature(x_last, "wave_dir_y_from_mps")
        pressure = self._raw_feature(x_last, "pressure_anom")
        if self.shuffled:
            wind_u, wind_v = torch.roll(wind_u, 1, 1), torch.roll(wind_v, 1, 1)
            current_u, current_v = torch.roll(current_u, 1, 1), torch.roll(current_v, 1, 1)
            wave_u, wave_v = torch.roll(wave_u, 1, 1), torch.roll(wave_v, 1, 1)
            pressure = torch.roll(pressure, 1, 1)

        scales = torch.nn.functional.softplus(self.raw_direction_scales) + 0.25
        wind = self._directional(wind_u, wind_v, scales[0])
        current = self._directional(current_u, current_v, scales[1])
        wave = self._directional(wave_u, wave_v, scales[2])
        pressure_std = pressure.std(dim=1, keepdim=True).clamp_min(1e-4)
        # Higher source pressure relative to the destination defines a directed gradient proxy.
        pressure_score = (pressure[:, None, :] - pressure[:, :, None]) / pressure_std[:, None, :]
        pressure_graph = self.distance[None] * torch.exp(
            torch.clamp(scales[3] * pressure_score, min=-4.0, max=4.0)
        )
        pressure_graph = row_normalize_torch(pressure_graph)
        distance = self.distance[None].expand(x.shape[0], -1, -1)
        components = torch.stack([distance, wind, current, wave, pressure_graph], dim=1)

        standardized = x_last
        intensity = torch.stack(
            [
                torch.sqrt(
                    standardized[..., self.feature_indices["wind_stress_u_proxy"]].square()
                    + standardized[..., self.feature_indices["wind_stress_v_proxy"]].square()
                ).mean(1),
                torch.sqrt(
                    standardized[..., self.feature_indices["uo"]].square()
                    + standardized[..., self.feature_indices["vo"]].square()
                ).mean(1),
                torch.sqrt(
                    standardized[..., self.feature_indices["wave_dir_x_from_mps"]].square()
                    + standardized[..., self.feature_indices["wave_dir_y_from_mps"]].square()
                ).mean(1),
                standardized[..., self.feature_indices["pressure_anom"]].std(1),
            ],
            dim=-1,
        )
        gates = torch.softmax(self.component_logits[None] + self.sample_gate(torch.tanh(intensity)), dim=-1)
        support = row_normalize_torch(torch.einsum("bk,bkij->bij", gates, components))
        self.last_support = support.detach()
        self.last_gates = gates.detach()
        return support

    def weight_dict(self) -> dict[str, float]:
        weights = torch.softmax(self.component_logits.detach(), dim=0).cpu().numpy()
        return {
            "identity": 0.0,
            "distance": float(weights[0]),
            "corr": float(weights[1:].sum()),
            **{f"transport_{name}": float(weights[index]) for index, name in enumerate(self.component_names)},
        }


class DynamicDiffusionGraphLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, diffusion_steps: int):
        super().__init__()
        self.diffusion_steps = int(diffusion_steps)
        self.linear = nn.Linear(in_dim * (self.diffusion_steps + 1), out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        values = [x]
        propagated = x
        for _ in range(self.diffusion_steps):
            propagated = torch.einsum("bij,bjf->bif", adjacency, propagated)
            values.append(propagated)
        return self.linear(torch.cat(values, dim=-1))


class DynamicGraphWaveNetBlock(nn.Module):
    def __init__(self, channels: int, diffusion_steps: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation
        self.filter_conv = nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation), padding=(0, pad))
        self.gate_conv = nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation), padding=(0, pad))
        self.graph = DynamicDiffusionGraphLinear(channels, channels, diffusion_steps)
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = x
        filt = torch.tanh(self.filter_conv(x)[..., : x.shape[-1]])
        gate = torch.sigmoid(self.gate_conv(x)[..., : x.shape[-1]])
        h = filt * gate
        batch, channels, nodes, steps = h.shape
        h_nodes = h.permute(0, 3, 2, 1).reshape(batch * steps, nodes, channels)
        repeated = adjacency[:, None].expand(-1, steps, -1, -1).reshape(batch * steps, nodes, nodes)
        h_nodes = self.graph(h_nodes, repeated).reshape(batch, steps, nodes, channels).permute(0, 3, 2, 1)
        return self.norm(self.dropout(h_nodes) + residual)


class TransportGraphWaveNetMultistate(nn.Module):
    def __init__(self, data: dict, args, shuffled: bool = False):
        super().__init__()
        self.horizon = int(args.horizon)
        self.num_states = len(v2.STATE_NAMES)
        distance = data["graph_priors"][args.fixed_graph_type]
        self.graph = TransportSupportBuilder(
            distance, data["station_meta"], data["feature_cols"], data["x_scaler_state"], shuffled
        )
        self.register_buffer("fixed_adjacency", torch.tensor(distance, dtype=torch.float32))
        self.input_proj = nn.Conv2d(data["feats"], args.hidden_dim, (1, 1))
        dilations = [2 ** (index % 4) for index in range(args.gwn_blocks)]
        self.blocks = nn.ModuleList(
            [DynamicGraphWaveNetBlock(args.hidden_dim, args.diffusion_steps, d, args.dropout) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Conv2d(args.hidden_dim, args.hidden_dim, (1, 1)),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Conv2d(args.hidden_dim, args.horizon * self.num_states, (1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = self.graph(x)
        h = self.input_proj(x.permute(0, 3, 2, 1))
        for block in self.blocks:
            h = block(h, adjacency)
        out = self.head(h)[..., -1].permute(0, 2, 1)
        return out.reshape(x.shape[0], x.shape[2], self.horizon, self.num_states)


class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("bij,btjf->btif", adjacency, x)
        return self.linear(propagated)


class TransportGNNBiGRU(nn.Module):
    def __init__(self, data: dict, args, shuffled: bool = False):
        super().__init__()
        self.horizon = int(args.horizon)
        self.num_states = len(v2.STATE_NAMES)
        distance = data["graph_priors"][args.fixed_graph_type]
        self.graph = TransportSupportBuilder(
            distance, data["station_meta"], data["feature_cols"], data["x_scaler_state"], shuffled
        )
        self.register_buffer("fixed_adjacency", torch.tensor(distance, dtype=torch.float32))
        self.gcn1 = DynamicGraphConvolution(data["feats"], args.gnn_hidden)
        self.gcn2 = DynamicGraphConvolution(args.gnn_hidden, args.gnn_hidden)
        self.norm = nn.LayerNorm(args.gnn_hidden)
        self.dropout = nn.Dropout(args.bigru_dropout)
        self.gru = nn.GRU(
            args.gnn_hidden, args.gru_hidden, num_layers=1, batch_first=True, bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(args.gru_hidden * 2, args.gru_hidden),
            nn.ReLU(),
            nn.Dropout(args.bigru_dropout),
            nn.Linear(args.gru_hidden, args.horizon * self.num_states),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = self.graph(x)
        h = torch.relu(self.gcn1(x, adjacency))
        h = torch.relu(self.gcn2(h, adjacency))
        h = self.dropout(self.norm(h))
        batch, steps, nodes, hidden = h.shape
        h = h.permute(0, 2, 1, 3).reshape(batch * nodes, steps, hidden)
        _, h_n = self.gru(h)
        pred = self.head(v2.bidirectional_final_hidden(h_n))
        return pred.reshape(batch, nodes, self.horizon, self.num_states)


class BatchedTransportPhysicsODE(v2.MultistatePhysicsODE):
    """The legacy residual equation with a sample-dependent transport support."""

    def forward(self, pred_states, init_states, phys_seq, adj):
        if adj.ndim == 2:
            return super().forward(pred_states, init_states, phys_seq, adj)
        prev = torch.cat([init_states.unsqueeze(2), pred_states[:, :, :-1, :]], dim=2)
        lhs = pred_states - prev
        spatial = torch.einsum("bij,bjhs->bihs", adj, prev) - prev
        u, v, wave = prev[..., 1], prev[..., 2], prev[..., 3]
        graph_u = torch.einsum("bij,bjh->bih", adj, u)
        graph_v = torch.einsum("bij,bjh->bih", adj, v)
        transport = (graph_u - u) + (graph_v - v)
        pressure_gradient = v2.GRAVITY * spatial[..., 0]
        extra = torch.stack([u, v, wave, transport, pressure_gradient], dim=-1)
        extra = torch.cat([extra, phys_seq], dim=-1)
        forcing = torch.einsum("sk,bnhk->bnhs", self.beta, extra)
        decay = torch.nn.functional.softplus(self.raw_decay).view(1, 1, 1, -1)
        kappa = torch.nn.functional.softplus(self.raw_kappa).view(1, 1, 1, -1)
        rhs = self.bias.T.view(1, self.num_nodes, 1, self.num_states) - decay * prev + kappa * spatial + forcing
        residual = lhs - rhs
        return torch.mean(residual.square()), residual


def make_model(config: str, data: dict, args, device: torch.device):
    backbone, variant = config.split("_", 1)
    dynamic = variant.startswith("transport")
    shuffled = variant.startswith("transport_shuffled")
    adjacency = data["graph_priors"][args.fixed_graph_type]
    if backbone == "gwn":
        if dynamic:
            model = TransportGraphWaveNetMultistate(data, args, shuffled)
        else:
            model = p104.GraphWaveNetMultistate(
                data["feats"], adjacency, args.hidden_dim, args.horizon, len(v2.STATE_NAMES),
                args.diffusion_steps, args.gwn_blocks, args.dropout,
            )
    elif dynamic:
        model = TransportGNNBiGRU(data, args, shuffled)
    else:
        model = v2.MultistateGNNBiGRU(
            data["feats"], data["graph_priors"], [0.50, 0.35, 0.15], args.gnn_hidden,
            args.gru_hidden, args.horizon, args.bigru_dropout, len(v2.STATE_NAMES),
        )
        model.register_buffer("fixed_adjacency", torch.tensor(adjacency, dtype=torch.float32))
    physics_class = BatchedTransportPhysicsODE if "coupled_physics_loss" in config else v2.MultistatePhysicsODE
    physics_ode = physics_class(data["nodes"], len(data["physics_cols"]), len(v2.STATE_NAMES))
    return model.to(device), physics_ode.to(device)


def uses_physics_loss(config: str) -> bool:
    return config.endswith("physics_loss")


def uses_coupled_physics_loss(config: str) -> bool:
    return "coupled_physics_loss" in config


@torch.no_grad()
def evaluate_coupled(model, criterion, loader, physics_lambda_value: float, device) -> dict[str, float]:
    model.eval()
    criterion.physics_ode.eval()
    totals: dict[str, float] = {}
    count = 0
    for xb, target, _, init_states, phys_seq in loader:
        xb = xb.to(device)
        pred = model(xb)
        adjacency = model.graph(xb)
        _, parts = criterion(
            pred, target.to(device), init_states.to(device), phys_seq.to(device), adjacency, physics_lambda_value
        )
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


def train_coupled(model, physics_ode, data, train_loader, val_loader, args, device):
    criterion = p104.make_multistate_criterion(physics_ode, data, args, True, args.horizon).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": physics_ode.parameters(), "lr": args.lr * args.physics_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    rows = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        lam = p104.physics_lambda(epoch, args)
        model.train()
        physics_ode.train()
        totals: dict[str, float] = {}
        count = 0
        for xb, target, _, init_states, phys_seq in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            adjacency = model.graph(xb)
            loss, parts = criterion(
                pred, target.to(device), init_states.to(device), phys_seq.to(device), adjacency, lam
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(physics_ode.parameters()), args.grad_clip)
            optimizer.step()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        train_parts = {key: value / max(1, count) for key, value in totals.items()}
        val_parts = evaluate_coupled(model, criterion, val_loader, lam, device)
        val_score = val_parts["eta_data_loss"]
        scheduler.step(val_score)
        row = {
            "epoch": epoch,
            "selection_score": val_score,
            **{f"train_{key}": value for key, value in train_parts.items()},
            **{f"val_{key}": value for key, value in val_parts.items()},
            **model.graph.weight_dict(),
            **physics_ode.coefficients(),
        }
        rows.append(row)
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch={epoch:03d} coupled lambda={lam:.6f} train_eta={train_parts['eta_data_loss']:.6f} "
                f"val_eta={val_score:.6f} val_last={val_parts['last_loss']:.6f}"
            )
        if val_score < best_val - args.min_delta:
            best_val = val_score
            bad_epochs = 0
            best_state = {
                "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "physics": {key: value.detach().cpu().clone() for key, value in physics_ode.state_dict().items()},
            }
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        physics_ode.load_state_dict(best_state["physics"])
    return pd.DataFrame(rows), best_val, {
        "training_seconds": time.perf_counter() - started,
        "epochs_completed": len(rows),
    }


def model_metadata(config: str, seed: int, args, data: dict, model: nn.Module) -> dict:
    return {
        "seed": int(seed),
        "config": config,
        "train_end_exclusive": args.fold_train_end,
        "validation_end_exclusive": args.fold_val_end,
        "evaluation_end_exclusive": args.fold_test_end if args.stage == "formal" else args.fold_val_end,
        "evaluation_split": "2025_h2_backtest" if args.stage == "formal" else "2025_h1_validation_screen",
        "strict_causal_preprocessing": True,
        "untouched_holdout": False,
        "future_residual_used_as_input": False,
        "transport_forcing_time": "last_observed_input_hour",
        "transport_components": TransportSupportBuilder.component_names,
        "physical_loss_formula": (
            "legacy_multistate_ode_residual_huber_with_dynamic_transport_support"
            if uses_coupled_physics_loss(config)
            else "legacy_multistate_ode_residual_huber"
        ),
        "physics_lambda_max": float(args.physics_lambda if uses_physics_loss(config) else 0.0),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "feature_cols": data["feature_cols"],
    }


def checkpoint_payload(model, physics_ode, metadata: dict, data: dict) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "physics_ode_state_dict": physics_ode.state_dict(),
        "metadata": metadata,
        "feature_cols": data["feature_cols"],
        "physics_cols": data["physics_cols"],
        "graph_priors": data["graph_priors"],
        "state_scale": data["state_scale"],
        "delta_scale": data["delta_scale"],
        "x_scaler_state": data["x_scaler_state"],
        "physics_scaler_state": data["physics_scaler_state"],
    }


def train_one(config: str, seed: int, data: dict, args, device: torch.device) -> dict:
    run_dir = project_path(args.output_dir) / args.stage / f"seed_{seed}" / config
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    prediction_path = run_dir / "predictions.npz"
    if args.resume and metrics_path.exists() and prediction_path.exists() and (run_dir / "COMPLETE.json").exists():
        print(f"Skipping complete run seed={seed} config={config}")
        return pd.read_csv(metrics_path).iloc[0].to_dict()

    p104.set_reproducible(seed, args.cpu_threads)
    model, physics_ode = make_model(config, data, args, device)
    train_loader = p104.make_loader(data["multi_train"], args, True, seed)
    val_loader = p104.make_loader(data["multi_val"], args, False, seed)
    evaluation_dataset = data["multi_test"] if args.stage == "formal" else data["multi_val"]
    evaluation_loader = p104.make_loader(evaluation_dataset, args, False, seed)
    started = time.perf_counter()
    if uses_coupled_physics_loss(config):
        history, best_val, timing = train_coupled(
            model, physics_ode, data, train_loader, val_loader, args, device
        )
        training_seconds = timing["training_seconds"]
    elif config.startswith("gwn_"):
        history, best_val, timing = p104.train_multistate(
            model, physics_ode, data, train_loader, val_loader, args, device,
            uses_physics_loss(config), run_dir,
        )
        training_seconds = timing["training_seconds"]
    else:
        train_args = copy.copy(args)
        train_args.physics_lambda_max = args.physics_lambda if uses_physics_loss(config) else 0.0
        train_args.physics_state_weights = [1.0, 0.35, 0.35, 0.25]
        history, best_val = v4.train_weighted_model(
            model, physics_ode, train_loader, val_loader, train_args,
            {
                "config_name": config,
                "physics_lambda_max": train_args.physics_lambda_max,
                "physics_state_weights": train_args.physics_state_weights,
                "physics_lead_gamma": 0.0,
                "data_lead_gamma": 0.0,
                "extreme_alpha": 0.0,
                "extreme_quantile": args.extreme_quantile,
                "last_step_weight": args.last_step_weight,
            },
            data["state_scale"], data["delta_scale"], data["train_abs_eta_threshold"],
            args.horizon, device,
        )
        training_seconds = time.perf_counter() - started

    pred_states, true_states, tide, inference_seconds = p104.predict_multistate(
        model, evaluation_loader, device
    )
    pred, true = pred_states[..., 0], true_states[..., 0]
    train_end = rolling.time_index(data["arrays"]["time"], args.fold_train_end)
    thresholds = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    row = {
        **model_metadata(config, seed, args, data, model),
        "best_val_eta_data_loss": float(best_val),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        **confirm.score(true, pred, tide, thresholds),
    }
    if hasattr(model, "graph") and hasattr(model.graph, "weight_dict"):
        row.update(model.graph.weight_dict())
    history.to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    indices = np.asarray(evaluation_dataset.indices, dtype=np.int64)
    target_times = pd.to_datetime(data["arrays"]["time"])[indices].to_numpy(dtype="datetime64[ns]")
    np.savez_compressed(
        prediction_path, pred_residual=pred, true_residual=true, target_tide=tide,
        target_origin_time=target_times, station_ids=np.asarray(v2.STATION_IDS),
    )
    metadata = model_metadata(config, seed, args, data, model)
    torch.save(checkpoint_payload(model, physics_ode, metadata, data), run_dir / "best_checkpoint.pt")
    (run_dir / "COMPLETE.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"seed={seed} config={config} seq={row['seq_residual_R2']:.6f} "
        f"lead24={row['last_residual_R2']:.6f} val={best_val:.6f}"
    )
    return row


def exact_wilcoxon_greater(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta) & (delta != 0)]
    if not len(delta):
        return np.nan
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(delta, alternative="greater", method="exact").pvalue)
    except Exception:
        positives = int(np.sum(delta > 0))
        return float(sum(math.comb(len(delta), k) for k in range(positives, len(delta) + 1)) / (2 ** len(delta)))


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    true_flat, pred_flat = np.asarray(true).reshape(-1), np.asarray(pred).reshape(-1)
    denominator = np.sum((true_flat - true_flat.mean()) ** 2)
    return float(1.0 - np.sum((true_flat - pred_flat) ** 2) / max(denominator, 1e-12))


def comparison_pairs() -> list[tuple[str, str, str]]:
    pairs = []
    for backbone in BACKBONES:
        pairs.extend(
            [
                (f"{backbone}_physics_loss", f"{backbone}_baseline", "old_loss_minus_baseline"),
                (f"{backbone}_transport", f"{backbone}_baseline", "transport_minus_baseline"),
                (f"{backbone}_transport_physics_loss", f"{backbone}_transport", "loss_on_transport"),
                (f"{backbone}_transport_coupled_physics_loss", f"{backbone}_transport", "coupled_loss_on_transport"),
                (f"{backbone}_transport_coupled_physics_loss", f"{backbone}_transport_physics_loss", "coupled_minus_fixed_graph_loss"),
                (f"{backbone}_transport_physics_loss", f"{backbone}_physics_loss", "combined_minus_old_loss"),
                (f"{backbone}_transport", f"{backbone}_transport_shuffled", "physical_alignment_minus_shuffled"),
                (
                    f"{backbone}_transport_physics_loss",
                    f"{backbone}_transport_shuffled_physics_loss",
                    "aligned_combined_minus_shuffled_combined",
                ),
            ]
        )
    return pairs


def merge_results(args) -> None:
    root = project_path(args.output_dir) / args.stage
    frames = [pd.read_csv(path) for path in sorted(root.glob("seed_*/*/metrics.csv"))]
    if not frames:
        raise RuntimeError(f"No completed metrics under {root}")
    all_runs = pd.concat(frames, ignore_index=True)
    all_runs.to_csv(root / "all_runs.csv", index=False)
    metrics = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2"]
    summary_rows = []
    for config, group in all_runs.groupby("config", sort=False):
        row = {"config": config, "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
        row["trainable_parameters"] = int(group["trainable_parameters"].iloc[0])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "mean_std.csv", index=False)

    paired_rows = []
    for candidate, reference, label in comparison_pairs():
        left = all_runs[all_runs["config"] == candidate].set_index("seed")
        right = all_runs[all_runs["config"] == reference].set_index("seed")
        seeds = left.index.intersection(right.index)
        for metric in metrics:
            delta = left.loc[seeds, metric].to_numpy() - right.loc[seeds, metric].to_numpy()
            paired_rows.append(
                {
                    "comparison": label,
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "n_seeds": len(seeds),
                    "mean_delta": float(np.mean(delta)) if len(delta) else np.nan,
                    "wins": int(np.sum(delta > 0)),
                    "wilcoxon_greater_p": exact_wilcoxon_greater(delta),
                    "seed_deltas": json.dumps(delta.tolist()),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(root / "paired_comparisons.csv", index=False)

    if args.stage == "formal":
        bootstrap_rows = []
        rng = np.random.default_rng(args.bootstrap_seed)
        for candidate, reference, label in comparison_pairs():
            deltas_by_metric = {"seq_residual_R2": [], "last_residual_R2": []}
            for seed in args.seeds:
                candidate_path = root / f"seed_{seed}" / candidate / "predictions.npz"
                reference_path = root / f"seed_{seed}" / reference / "predictions.npz"
                if not candidate_path.exists() or not reference_path.exists():
                    continue
                with np.load(candidate_path) as a, np.load(reference_path) as b:
                    true = a["true_residual"]
                    pred_a, pred_b = a["pred_residual"], b["pred_residual"]
                n_time = true.shape[0]
                n_blocks = int(np.ceil(n_time / args.bootstrap_block_hours))
                for _ in range(args.bootstrap_replicates):
                    starts = rng.integers(0, max(1, n_time - args.bootstrap_block_hours + 1), size=n_blocks)
                    indices = np.concatenate(
                        [np.arange(start, min(start + args.bootstrap_block_hours, n_time)) for start in starts]
                    )[:n_time]
                    deltas_by_metric["seq_residual_R2"].append(
                        r2_score(true[indices], pred_a[indices]) - r2_score(true[indices], pred_b[indices])
                    )
                    deltas_by_metric["last_residual_R2"].append(
                        r2_score(true[indices, :, -1], pred_a[indices, :, -1])
                        - r2_score(true[indices, :, -1], pred_b[indices, :, -1])
                    )
            for metric, values in deltas_by_metric.items():
                values = np.asarray(values, dtype=float)
                bootstrap_rows.append(
                    {
                        "comparison": label,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "n_seed_replicates": len(values),
                        "mean_delta": float(np.mean(values)) if len(values) else np.nan,
                        "ci_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                        "ci_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                        "probability_positive": float(np.mean(values > 0)) if len(values) else np.nan,
                    }
                )
        pd.DataFrame(bootstrap_rows).to_csv(root / "block_bootstrap_comparisons.csv", index=False)
    plot_summary(summary, paired, root)
    plot_example_predictions(root, args.seeds)
    write_report(summary, paired, root, args)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


def reuse_locked_formal_baselines(args) -> None:
    """Reuse exact strict-protocol baselines while preserving explicit provenance."""
    output_root = project_path(args.output_dir) / "formal"
    gwn_root = ROOT / "results" / "confirmatory_hsdt_orc_refit_2025_h2"
    bigru_root = ROOT / "results" / "strict_causal_physical_bigru_refit_2025_h2"
    for seed in args.seeds:
        with np.load(gwn_root / f"seed_{seed}" / "predictions.npz") as source:
            common = {
                "true_residual": source["true_residual"],
                "target_tide": source["target_tide"],
                "target_origin_time": source["target_origin_time"],
                "station_ids": source["station_ids"],
            }
            gwn_pred = source["gwn_multistate_no_physics"]
        gwn_metrics = pd.read_csv(gwn_root / f"seed_{seed}" / "metrics.csv")
        gwn_row = gwn_metrics[gwn_metrics["config"] == "gwn_multistate_no_physics"].iloc[0].to_dict()
        gwn_row.update(
            {
                "config": "gwn_baseline",
                "model": "gwn_baseline",
                "evaluation_end_exclusive": args.fold_test_end,
                "evaluation_split": "2025_h2_backtest",
                "untouched_holdout": False,
                "transport_forcing_time": "not_applicable_static_distance_graph",
                "transport_components": "[]",
                "physical_loss_formula": "none",
                "physics_lambda_max": 0.0,
                "trainable_parameters": 186592,
                "reused_locked_strict_baseline": True,
            }
        )
        destination = output_root / f"seed_{seed}" / "gwn_baseline"
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([gwn_row]).to_csv(destination / "metrics.csv", index=False)
        np.savez_compressed(destination / "predictions.npz", pred_residual=gwn_pred, **common)
        shutil.copy2(gwn_root / f"seed_{seed}" / "gwn_multistate_no_physics" / "best_checkpoint.pt", destination / "best_checkpoint.pt")
        shutil.copy2(gwn_root / f"seed_{seed}" / "gwn_multistate_no_physics" / "training_log.csv", destination / "training_log.csv")
        provenance = {
            "seed": seed,
            "destination_config": "gwn_baseline",
            "source": str((gwn_root / f"seed_{seed}").relative_to(ROOT)),
            "source_config": "gwn_multistate_no_physics",
            "reason": "Exact same strict chronological protocol and architecture; no retraining required.",
        }
        (destination / "REUSED_LOCKED_BASELINE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        (destination / "COMPLETE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

        reference_true = common["true_residual"]
        reference_times = common["target_origin_time"]
        for source_config, destination_config in [
            ("gnn_bigru_no_physics", "bigru_baseline"),
            ("gnn_bigru_physics", "bigru_physics_loss"),
        ]:
            source_dir = bigru_root / f"seed_{seed}" / source_config
            row = pd.read_csv(source_dir / "metrics.csv").iloc[0].to_dict()
            with np.load(source_dir / "predictions.npz") as source:
                if not np.array_equal(source["target_origin_time"], reference_times):
                    raise RuntimeError(f"Timestamp mismatch for seed={seed} config={source_config}")
                if not np.allclose(source["true_residual"], reference_true, atol=1e-7, rtol=0.0):
                    raise RuntimeError(f"Target mismatch for seed={seed} config={source_config}")
                prediction_payload = {key: source[key] for key in source.files}
            row.update(
                {
                    "config": destination_config,
                    "evaluation_end_exclusive": args.fold_test_end,
                    "evaluation_split": "2025_h2_backtest",
                    "untouched_holdout": False,
                    "transport_forcing_time": "not_applicable_static_graph",
                    "transport_components": "[]",
                    "physical_loss_formula": "legacy_multistate_ode_residual_huber" if "physics" in source_config else "none",
                    "trainable_parameters": 38403,
                    "reused_locked_strict_baseline": True,
                }
            )
            destination = output_root / f"seed_{seed}" / destination_config
            destination.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(destination / "metrics.csv", index=False)
            np.savez_compressed(destination / "predictions.npz", **prediction_payload)
            shutil.copy2(source_dir / "best_checkpoint.pt", destination / "best_checkpoint.pt")
            shutil.copy2(source_dir / "training_log.csv", destination / "training_log.csv")
            provenance = {
                "seed": seed,
                "destination_config": destination_config,
                "source": str(source_dir.relative_to(ROOT)),
                "source_config": source_config,
                "reason": "Exact same strict chronological protocol and architecture; no retraining required.",
            }
            (destination / "REUSED_LOCKED_BASELINE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            (destination / "COMPLETE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Reused and verified locked formal baselines for seeds={args.seeds}")


def plot_summary(summary: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> None:
    order = [config for config in CONFIGS if config in set(summary["config"])]
    table = summary.set_index("config").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    labels = [name.replace("_", "\n", 1) for name in order]
    colors = ["#4C78A8" if name.startswith("gwn") else "#F58518" for name in order]
    for axis, metric, title in [
        (axes[0], "seq_residual_R2", "24-hour sequence residual R2"),
        (axes[1], "last_residual_R2", "Lead-24 residual R2"),
    ]:
        means = table[f"{metric}_mean"].to_numpy()
        errors = table[f"{metric}_std"].fillna(0).to_numpy()
        axis.bar(np.arange(len(order)), means, yerr=errors, capsize=3, color=colors, alpha=0.88)
        axis.set_xticks(np.arange(len(order)), labels, rotation=35, ha="right")
        axis.set_ylabel("R2")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "transport_physics_model_comparison.png", dpi=220)
    plt.close(fig)

    selected = paired[paired["metric"].isin(["seq_residual_R2", "last_residual_R2"])].copy()
    if selected.empty:
        return
    fig, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    labels = selected["candidate"] + "\nvs " + selected["reference"] + "\n" + selected["metric"].str.replace("_residual_R2", "")
    colors = np.where(selected["mean_delta"] >= 0, "#2A9D8F", "#D1495B")
    axis.bar(np.arange(len(selected)), selected["mean_delta"], color=colors)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(np.arange(len(selected)), labels, rotation=55, ha="right")
    axis.set_ylabel("Paired mean R2 difference")
    axis.set_title("Attribution tests: architecture and legacy physical loss")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "transport_physics_paired_deltas.png", dpi=220)
    plt.close(fig)


def plot_example_predictions(output_dir: Path, seeds: list[int]) -> None:
    if not seeds:
        return
    seed = seeds[0]
    available = [config for config in CONFIGS if (output_dir / f"seed_{seed}" / config / "predictions.npz").exists()]
    if not available:
        return
    bundles = {}
    for config in available:
        with np.load(output_dir / f"seed_{seed}" / config / "predictions.npz") as data:
            bundles[config] = {key: data[key] for key in data.files}
    reference = bundles[available[0]]
    true = reference["true_residual"][:, :, -1]
    station_energy = np.max(np.abs(true), axis=0)
    station = int(np.argmax(station_energy))
    peak = int(np.argmax(np.abs(true[:, station])))
    start, end = max(0, peak - 72), min(len(true), peak + 73)
    times = pd.to_datetime(reference["target_origin_time"])[start:end]
    fig, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    axis.plot(times, true[start:end, station], color="black", linewidth=2.2, label="Observed residual")
    palette = plt.get_cmap("tab10")
    for index, config in enumerate(available):
        pred = bundles[config]["pred_residual"][:, station, -1]
        axis.plot(times, pred[start:end], linewidth=1.2, alpha=0.9, color=palette(index % 10), label=config)
    axis.axhline(0.0, color="grey", linewidth=0.8)
    station_id = str(reference["station_ids"][station])
    axis.set_title(f"Lead-24 prediction comparison around a high-residual period: station {station_id}, seed {seed}")
    axis.set_ylabel("Non-tidal residual (m)")
    axis.legend(ncol=3, fontsize=8)
    axis.grid(alpha=0.2)
    fig.savefig(output_dir / "transport_physics_example_predictions.png", dpi=220)
    plt.close(fig)


def write_report(summary: pd.DataFrame, paired: pd.DataFrame, output_dir: Path, args) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for column in display.columns:
            if pd.api.types.is_float_dtype(display[column]):
                display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
        headers = [str(column).replace("|", "\\|") for column in display.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in display.astype(str).itertuples(index=False, name=None):
            lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
        return "\n".join(lines)

    lines = [
        "# Transport-aware dynamic graph and physical-loss experiment",
        "",
        f"Stage: `{args.stage}`.",
        "",
        "All variants use strict causal preprocessing. The transport support uses only the final observed input hour. "
        "The physical-loss variants use the legacy multistate ODE-residual Huber loss with lambda=0.0002.",
        "",
        "## Mean results",
        "",
        markdown_table(summary),
        "",
        "## Paired attribution",
        "",
        markdown_table(paired),
        "",
        "The shuffled control cyclically shifts physical forcing among stations while retaining its marginal distribution. "
        "A publishable physical-attribution claim requires the aligned transport model to beat both the static baseline "
        "and this shuffled control across paired seeds, with a block-bootstrap interval above zero.",
    ]
    (output_dir / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def validate_implementation(data: dict, args) -> None:
    device = torch.device("cpu")
    loader = p104.make_loader(data["multi_train"], args, False, 42)
    batch = next(iter(loader))[0][: min(3, args.batch_size)].to(device)
    for config in ["gwn_transport", "gwn_transport_physics_loss", "bigru_transport"]:
        model, _ = make_model(config, data, args, device)
        model.eval()
        with torch.no_grad():
            output = model(batch)
        expected = (batch.shape[0], data["nodes"], args.horizon, len(v2.STATE_NAMES))
        if tuple(output.shape) != expected:
            raise AssertionError(f"{config}: output {tuple(output.shape)} != {expected}")
        adjacency = model.graph.last_support
        if adjacency is None or not torch.allclose(adjacency.sum(-1), torch.ones_like(adjacency.sum(-1)), atol=1e-5):
            raise AssertionError(f"{config}: dynamic adjacency is not row normalized")
        if not torch.isfinite(output).all():
            raise AssertionError(f"{config}: non-finite output")
    print("Implementation validation passed: shapes, finite outputs, and row-normalized dynamic supports.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transport-aware dynamic graph plus legacy physical-loss attribution.")
    parser.add_argument("--mode", choices=["run", "merge", "check", "reuse_locked"], default="run")
    parser.add_argument("--stage", choices=["screen", "formal"], default="screen")
    parser.add_argument("--output-dir", default="results/transport_aware_dynamic_graph_physics")
    parser.add_argument("--data-dir", default="data/processed_multiyear_2023_2025")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=CONFIGS)
    parser.add_argument("--fold-train-end", default="2025-01-01")
    parser.add_argument("--fold-val-end", default="2025-07-01")
    parser.add_argument("--fold-test-end", default="2026-01-01")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", choices=["identity", "distance", "corr"], default="distance")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--gwn-blocks", type=int, default=6)
    parser.add_argument("--gnn-hidden", type=int, default=40)
    parser.add_argument("--gru-hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--bigru-dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--graph-lr-mult", type=float, default=3.0)
    parser.add_argument("--physics-lr-mult", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--physics-lambda", type=float, default=0.0002)
    parser.add_argument("--physics-warmup-epochs", type=int, default=8)
    parser.add_argument("--physics-ramp-epochs", type=int, default=14)
    parser.add_argument("--physics-loss-type", choices=["huber", "mse"], default="huber")
    parser.add_argument("--physics-state-weights", type=float, nargs=4, default=[1.0, 0.35, 0.35, 0.25])
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.20)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument("--selection-metric", choices=["val_eta_data_loss", "val_data_loss", "val_last_loss", "val_total_loss"], default="val_eta_data_loss")
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--bootstrap-block-hours", type=int, default=168)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1372026)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--epoch-checkpoint-every", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-merge", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir) / args.stage
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = confirm.configure_data_dir(args.data_dir)
    config = {
        **vars(args),
        "data_dir": str(data_dir.relative_to(ROOT) if data_dir.is_relative_to(ROOT) else data_dir),
        "validation_label": "validation_screen_no_test_access" if args.stage == "screen" else "chronological_refit_backtest_not_untouched",
        "dynamic_support_rule": "distance_plus_last_input_wind_current_wave_pressure",
        "legacy_physical_loss_lambda_locked": 0.0002,
    }
    (output_dir / ("final_experiment_config.json" if args.mode == "merge" else "experiment_config.json")).write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    if args.mode == "merge":
        merge_results(args)
        return
    if args.mode == "reuse_locked":
        reuse_locked_formal_baselines(args)
        return
    data = rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    if args.mode == "check":
        validate_implementation(data, args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}; stage={args.stage}; seeds={args.seeds}; configs={args.configs}")
    rows = []
    for seed in args.seeds:
        for config_name in args.configs:
            rows.append(train_one(config_name, seed, data, args, device))
            pd.DataFrame(rows).to_csv(output_dir / "run_progress.csv", index=False)
    if not args.skip_merge:
        merge_results(args)


if __name__ == "__main__":
    main()
