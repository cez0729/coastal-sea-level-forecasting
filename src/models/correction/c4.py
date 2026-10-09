from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset


HERE = Path(__file__).resolve().parent
WORKSPACE = REPO_ROOT
sys.path.insert(0, str(HERE))

from shared.formal import (  # noqa: E402
    TierAContract,
    extract_checkpoint_state,
    load_formal_data,
    make_loader,
    set_reproducible,
    sha256_file,
)
from shared.metrics import (  # noqa: E402
    fit_global_sigma_scale,
    metrics_by_horizon,
    metrics_by_station,
    point_summary,
    probabilistic_metrics,
)


VARIANTS = {
    "C4_HSDT_FROZEN_NO_PHYS": {
        "freeze_experts": True, "physics": False, "train_gate": True, "train_correction": True,
    },
    "C4_HSDT_FROZEN_PHYS": {
        "freeze_experts": True, "physics": True, "train_gate": True, "train_correction": True,
    },
    "C4_HSDT_FROZEN_GATE_ONLY": {
        "freeze_experts": True, "physics": False, "train_gate": True, "train_correction": False,
    },
    "C4_HSDT_FROZEN_CORRECTION_ONLY": {
        "freeze_experts": True, "physics": False, "train_gate": False, "train_correction": True,
    },
    "C4_HSDT_INIT_JOINT_NO_PHYS": {
        "freeze_experts": False, "physics": False, "train_gate": True, "train_correction": True,
    },
    "C4_HSDT_INIT_JOINT_PHYS": {
        "freeze_experts": False, "physics": True, "train_gate": True, "train_correction": True,
    },
}


class EtaContext(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward_with_context(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone.input_proj(x.permute(0, 3, 2, 1))
        for block in self.backbone.blocks:
            hidden = block(hidden)
        context = hidden[..., -1]
        prediction = self.backbone.head(hidden)[..., -1].permute(0, 2, 1)
        return prediction, context


class MultiContext(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward_with_context(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone.input_proj(x.permute(0, 3, 2, 1))
        for block in self.backbone.blocks:
            hidden = block(hidden)
        context = hidden[..., -1]
        raw = self.backbone.head(hidden)[..., -1].permute(0, 2, 1)
        prediction = raw.reshape(x.shape[0], x.shape[2], self.backbone.horizon, self.backbone.num_states)
        return prediction, context


class HSDTResidualGate(nn.Module):
    """Context-dependent residual around the locked HS-DT horizon rule."""

    def __init__(self, channels: int, horizon: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2 * channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels, horizon, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        base = torch.full((horizon,), 0.5, dtype=torch.float32)
        base[-1] = 0.0
        self.register_buffer("base_eta_weight", base)

    def forward(self, eta_context: torch.Tensor, multi_context: torch.Tensor) -> torch.Tensor:
        delta = 0.25 * torch.tanh(self.net(torch.cat([eta_context, multi_context], dim=1)))
        base = self.base_eta_weight.view(1, -1, 1)
        return torch.clamp(base + delta, 0.0, 1.0).permute(0, 2, 1)


class PhysicsCorrectionHead(nn.Module):
    """Zero-initialized bounded correction driven by both formal GWN contexts."""

    def __init__(self, channels: int, horizon: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2 * channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels, horizon, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, eta_context: torch.Tensor, multi_context: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        correction_z = 0.25 * torch.tanh(self.net(torch.cat([eta_context, multi_context], dim=1)))
        return correction_z.permute(0, 2, 1) * scale


class GaussianHead(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -1.5)

    def forward(
        self,
        mean_z: torch.Tensor,
        eta_z: torch.Tensor,
        multi_z: torch.Tensor,
        eta_gate: torch.Tensor,
    ) -> torch.Tensor:
        horizon = mean_z.shape[-1]
        lead = torch.linspace(0.0, 1.0, horizon, device=mean_z.device, dtype=mean_z.dtype)
        lead = lead.view(1, 1, horizon).expand_as(mean_z)
        features = torch.stack((mean_z, eta_z, multi_z, eta_gate, lead), dim=-1)
        return torch.clamp(self.net(features).squeeze(-1), -6.0, 2.0)


@dataclass
class Output:
    mean: torch.Tensor
    eta: torch.Tensor
    multi_states: torch.Tensor
    multi_states_for_loss: torch.Tensor
    eta_gate: torch.Tensor
    correction: torch.Tensor
    log_sigma_z: torch.Tensor


class AlignedC4(nn.Module):
    def __init__(self, eta: nn.Module, multi: nn.Module, channels: int, horizon: int):
        super().__init__()
        self.eta_expert = eta
        self.multi_expert = multi
        self.gate = HSDTResidualGate(channels, horizon)
        self.correction = PhysicsCorrectionHead(channels, horizon)
        self.probability_head = GaussianHead()

    def forward(self, x: torch.Tensor, residual_scale: float) -> Output:
        eta, eta_context = self.eta_expert.forward_with_context(x)
        multi, multi_context = self.multi_expert.forward_with_context(x)
        return self.forward_from_cached(eta, eta_context, multi, multi_context, residual_scale)

    def forward_from_cached(
        self,
        eta: torch.Tensor,
        eta_context: torch.Tensor,
        multi: torch.Tensor,
        multi_context: torch.Tensor,
        residual_scale: float,
    ) -> Output:
        multi_eta = multi[..., 0]
        gate = self.gate(eta_context, multi_context)
        scale = torch.as_tensor(residual_scale, dtype=eta.dtype, device=eta.device).clamp_min(1e-8)
        correction = self.correction(eta_context, multi_context, scale)
        mean = gate * eta + (1.0 - gate) * multi_eta + correction
        multi_for_loss = multi.clone()
        multi_for_loss[..., 0] = mean
        log_sigma = self.probability_head(mean / scale, eta / scale, multi_eta / scale, gate)
        return Output(mean, eta, multi, multi_for_loss, gate, correction, log_sigma)


def gaussian_nll(target_z: torch.Tensor, mean_z: torch.Tensor, log_sigma_z: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(
        (target_z - mean_z).square() * torch.exp(-2.0 * log_sigma_z)
        + 2.0 * log_sigma_z
        + math.log(2.0 * math.pi)
    )


def checkpoint_paths(args: argparse.Namespace, seed: int) -> tuple[Path, Path]:
    eta_root = args.eta_results_root
    if seed == 42 and args.eta_seed42_checkpoint is not None:
        eta = args.eta_seed42_checkpoint
    else:
        eta = eta_root / f"seed_{seed}" / "horizon_24h" / "gwn_eta_only" / "best_checkpoint.pt"
    multi = args.multi_results_root / f"seed_{seed}" / "horizon_24h" / "gwn_multistate_no_physics" / "best_checkpoint.pt"
    # External-region runs use the same formal checkpoints without a horizon directory.
    if not eta.exists():
        eta = eta_root / f"seed_{seed}" / "gwn_eta_only" / "best_checkpoint.pt"
    if not multi.exists():
        multi = args.multi_results_root / f"seed_{seed}" / "gwn_multistate_no_physics" / "best_checkpoint.pt"
    missing = [path for path in (eta, multi) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing aligned HS-DT checkpoints: {missing}")
    return eta, multi


def build_model(
    official,
    data: dict,
    contract: TierAContract,
    eta_path: Path,
    multi_path: Path,
    freeze: bool,
    train_gate: bool = True,
    train_correction: bool = True,
):
    adjacency = data["graph_priors"][contract.fixed_graph_type]
    eta_backbone = official.priority1.GraphWaveNetForecaster(
        data["feats"], adjacency, contract.hidden_dim, contract.forecast_hours,
        contract.diffusion_steps, contract.gwn_blocks, contract.dropout,
    )
    multi_backbone = official.p104.GraphWaveNetMultistate(
        data["feats"], adjacency, contract.hidden_dim, contract.forecast_hours, 4,
        contract.diffusion_steps, contract.gwn_blocks, contract.dropout,
    )
    eta_backbone.load_state_dict(extract_checkpoint_state(eta_path), strict=True)
    multi_backbone.load_state_dict(extract_checkpoint_state(multi_path), strict=True)
    model = AlignedC4(EtaContext(eta_backbone), MultiContext(multi_backbone), contract.hidden_dim, contract.forecast_hours)
    if freeze:
        for expert in (model.eta_expert, model.multi_expert):
            for parameter in expert.parameters():
                parameter.requires_grad = False
    if not train_gate:
        for parameter in model.gate.parameters():
            parameter.requires_grad = False
    if not train_correction:
        for parameter in model.correction.parameters():
            parameter.requires_grad = False
    return model


def loss_for_batch(model, criterion, batch, scale, lam, joint: bool):
    x, target, _, initial, forcing = [item.to(next(model.parameters()).device) for item in batch]
    output = model(x, scale)
    target_eta = target[..., 0]
    z_scale = torch.as_tensor(scale, dtype=target.dtype, device=target.device).clamp_min(1e-8)
    target_z = target_eta / z_scale
    primary = gaussian_nll(target_z, output.mean / z_scale, output.log_sigma_z)
    eta_aux = torch.mean((output.eta / z_scale - target_z).square())
    multi_loss, parts = criterion(
        output.multi_states_for_loss, target, initial, forcing,
        model.multi_expert.backbone.fixed_adjacency, lam,
    )
    total = primary + 0.5 * multi_loss
    if joint:
        total = total + 0.5 * eta_aux
    return total, {
        "total": float(total.detach().cpu()),
        "primary_nll": float(primary.detach().cpu()),
        "eta_aux_z_mse": float(eta_aux.detach().cpu()),
        "multi_total": float(multi_loss.detach().cpu()),
        "physics_lambda": float(lam),
        **{f"multi_{key}": value for key, value in parts.items()},
    }


def loss_for_cached_batch(model, criterion, batch, scale, lam):
    eta, eta_context, multi, multi_context, target, _, initial, forcing = [
        item.to(next(model.parameters()).device) for item in batch
    ]
    output = model.forward_from_cached(eta, eta_context, multi, multi_context, scale)
    target_eta = target[..., 0]
    z_scale = torch.as_tensor(scale, dtype=target.dtype, device=target.device).clamp_min(1e-8)
    primary = gaussian_nll(target_eta / z_scale, output.mean / z_scale, output.log_sigma_z)
    multi_loss, parts = criterion(
        output.multi_states_for_loss, target, initial, forcing,
        model.multi_expert.backbone.fixed_adjacency, lam,
    )
    total = primary + 0.5 * multi_loss
    return total, {
        "total": float(total.detach().cpu()),
        "primary_nll": float(primary.detach().cpu()),
        "eta_aux_z_mse": float(torch.mean((output.eta / z_scale - target_eta / z_scale).square()).detach().cpu()),
        "multi_total": float(multi_loss.detach().cpu()),
        "physics_lambda": float(lam),
        **{f"multi_{key}": value for key, value in parts.items()},
    }


@torch.no_grad()
def cache_frozen_experts(model, loader):
    model.eval()
    cached = [[] for _ in range(8)]
    device = next(model.parameters()).device
    for x, target, tide, initial, forcing in loader:
        x_device = x.to(device)
        eta, eta_context = model.eta_expert.forward_with_context(x_device)
        multi, multi_context = model.multi_expert.forward_with_context(x_device)
        values = (eta, eta_context, multi, multi_context, target, tide, initial, forcing)
        for bucket, value in zip(cached, values):
            bucket.append(value.detach().cpu())
    return TensorDataset(*(torch.cat(bucket, dim=0) for bucket in cached))


@torch.no_grad()
def evaluate(model, criterion, loader, scale, lam, joint):
    model.eval()
    rows = [loss_for_batch(model, criterion, batch, scale, lam, joint)[1] for batch in loader]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


@torch.no_grad()
def evaluate_cached(model, criterion, loader, scale, lam):
    model.eval()
    rows = [loss_for_cached_batch(model, criterion, batch, scale, lam)[1] for batch in loader]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


@torch.no_grad()
def predict(model, loader, scale):
    model.eval()
    result = {key: [] for key in ("pred", "eta", "multi", "gate", "correction", "sigma", "true", "tide")}
    for x, target, tide, _, _ in loader:
        output = model(x.to(next(model.parameters()).device), scale)
        result["pred"].append(output.mean.cpu().numpy())
        result["eta"].append(output.eta.cpu().numpy())
        result["multi"].append(output.multi_states.cpu().numpy())
        result["gate"].append(output.eta_gate.cpu().numpy())
        result["correction"].append(output.correction.cpu().numpy())
        result["sigma"].append(np.exp(output.log_sigma_z.cpu().numpy()) * scale)
        result["true"].append(target[..., 0].numpy())
        result["tide"].append(tide.numpy())
    return {key: np.concatenate(value) for key, value in result.items()}


@torch.no_grad()
def predict_cached(model, loader, scale):
    model.eval()
    result = {key: [] for key in ("pred", "eta", "multi", "gate", "correction", "sigma", "true", "tide")}
    device = next(model.parameters()).device
    for eta, eta_context, multi, multi_context, target, tide, _, _ in loader:
        output = model.forward_from_cached(
            eta.to(device), eta_context.to(device), multi.to(device), multi_context.to(device), scale
        )
        result["pred"].append(output.mean.cpu().numpy())
        result["eta"].append(output.eta.cpu().numpy())
        result["multi"].append(output.multi_states.cpu().numpy())
        result["gate"].append(output.eta_gate.cpu().numpy())
        result["correction"].append(output.correction.cpu().numpy())
        result["sigma"].append(np.exp(output.log_sigma_z.cpu().numpy()) * scale)
        result["true"].append(target[..., 0].numpy())
        result["tide"].append(tide.numpy())
    return {key: np.concatenate(value) for key, value in result.items()}


def fixed_hsdt(eta: np.ndarray, multi_states: np.ndarray) -> np.ndarray:
    prediction = 0.5 * (eta + multi_states[..., 0])
    prediction[..., -1] = multi_states[..., 0][..., -1]
    return prediction


def keep_frozen_experts_eval(model: nn.Module, freeze_experts: bool) -> None:
    if freeze_experts:
        model.eta_expert.eval()
        model.multi_expert.eval()


def save_prediction(path: Path, prediction: dict, station_ids: list[str]):
    np.savez_compressed(
        path,
        pred_residual=prediction["pred"],
        true_residual=prediction["true"],
        target_tide=prediction["tide"],
        eta_pred=prediction["eta"],
        multi_pred_states=prediction["multi"],
        eta_gate=prediction["gate"],
        correction=prediction["correction"],
        sigma_residual=prediction["sigma"],
        station_ids=np.asarray(station_ids),
    )


def run_variant(args, variant_name, seed, official, formal_args, data, contract, device):
    settings = VARIANTS[variant_name]
    set_reproducible(seed, formal_args.cpu_threads)
    eta_path, multi_path = checkpoint_paths(args, seed)
    model = build_model(
        official,
        data,
        contract,
        eta_path,
        multi_path,
        settings["freeze_experts"],
        settings["train_gate"],
        settings["train_correction"],
    ).to(device)
    physics_ode = official.v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    criterion_args = deepcopy(formal_args)
    criterion = official.p104.make_multistate_criterion(
        physics_ode, data, criterion_args, settings["physics"], contract.forecast_hours
    ).to(device)
    loaders = {
        "train": make_loader(data["multi_train"], contract.batch_size, True, seed),
        "val": make_loader(data["multi_val"], contract.batch_size, False, seed),
        "test": make_loader(data["multi_test"], contract.batch_size, False, seed),
    }
    cached_loaders = None
    if settings["freeze_experts"]:
        print(f"{variant_name} seed={seed}: caching frozen expert outputs", flush=True)
        cached_datasets = {name: cache_frozen_experts(model, loader) for name, loader in loaders.items()}
        cached_loaders = {
            "train": make_loader(cached_datasets["train"], contract.batch_size, True, seed),
            "val": make_loader(cached_datasets["val"], contract.batch_size, False, seed),
            "test": make_loader(cached_datasets["test"], contract.batch_size, False, seed),
        }
    scale = float(data["state_scale"][0])
    output_dir = args.output_root / variant_name / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        active_loader = cached_loaders["train"] if cached_loaders is not None else loaders["train"]
        batch = tuple(item[:2] for item in next(iter(active_loader)))
        if cached_loaders is not None:
            loss, parts = loss_for_cached_batch(model, criterion, batch, scale, 0.0)
            eta, eta_context, multi, multi_context = [item.to(device) for item in batch[:4]]
            initial_output = model.forward_from_cached(eta, eta_context, multi, multi_context, scale)
        else:
            loss, parts = loss_for_batch(model, criterion, batch, scale, 0.0, True)
            initial_output = model(batch[0].to(device), scale)
        loss.backward()
        fixed_initial = 0.5 * (initial_output.eta + initial_output.multi_states[..., 0])
        fixed_initial[..., -1] = initial_output.multi_states[..., 0][..., -1]
        payload = {
            "variant": variant_name,
            "seed": seed,
            "loss": parts,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "initialization_max_abs_delta_from_fixed_hsdt": float(
                torch.max(torch.abs(initial_output.mean - fixed_initial)).detach().cpu()
            ),
            "train_gate": settings["train_gate"],
            "train_correction": settings["train_correction"],
            "eta_checkpoint": str(eta_path.resolve()),
            "multi_checkpoint": str(multi_path.resolve()),
        }
        (output_dir / "dry_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if settings["physics"]:
        trainable += list(physics_ode.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=contract.learning_rate, weight_decay=contract.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    best_score = float("inf")
    best_state = None
    bad = 0
    history = []
    start = time.perf_counter()
    max_epochs = contract.epochs if args.epochs is None else args.epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        keep_frozen_experts_eval(model, settings["freeze_experts"])
        physics_ode.train(settings["physics"])
        lam = official.v2.physics_lambda_for_epoch(
            epoch,
            contract.physics_lambda if settings["physics"] else 0.0,
            contract.physics_warmup_epochs,
            contract.physics_ramp_epochs,
        )
        rows = []
        active_train_loader = cached_loaders["train"] if cached_loaders is not None else loaders["train"]
        for batch in active_train_loader:
            optimizer.zero_grad(set_to_none=True)
            if cached_loaders is not None:
                loss, values = loss_for_cached_batch(model, criterion, batch, scale, lam)
            else:
                loss, values = loss_for_batch(model, criterion, batch, scale, lam, True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, contract.grad_clip)
            optimizer.step()
            rows.append(values)
        train_values = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        if cached_loaders is not None:
            val_values = evaluate_cached(model, criterion, cached_loaders["val"], scale, lam)
        else:
            val_values = evaluate(model, criterion, loaders["val"], scale, lam, True)
        score = val_values["primary_nll"]
        scheduler.step(score)
        history.append({
            "epoch": epoch,
            "selection_score": score,
            "selection_metric": "validation_gaussian_nll",
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_values.items()},
            **{f"val_{key}": value for key, value in val_values.items()},
        })
        if epoch == 1 or epoch % 5 == 0:
            print(f"{variant_name} seed={seed} epoch={epoch:03d} train_nll={train_values['primary_nll']:.6f} val_nll={score:.6f}", flush=True)
        if score < best_score - 1e-5:
            best_score = score
            bad = 0
            best_state = {
                "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "physics": {key: value.detach().cpu().clone() for key, value in physics_ode.state_dict().items()},
                "epoch": epoch,
            }
        else:
            bad += 1
        if bad >= contract.patience:
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint selected")
    model.load_state_dict(best_state["model"])
    physics_ode.load_state_dict(best_state["physics"])
    pd.DataFrame(history).to_csv(output_dir / "training_log.csv", index=False)

    if cached_loaders is not None:
        validation = predict_cached(model, cached_loaders["val"], scale)
        test = predict_cached(model, cached_loaders["test"], scale)
    else:
        validation = predict(model, loaders["val"], scale)
        test = predict(model, loaders["test"], scale)
    save_prediction(output_dir / "validation_predictions.npz", validation, official.v2.STATION_IDS)
    save_prediction(output_dir / "test_predictions.npz", test, official.v2.STATION_IDS)
    sigma_fit = fit_global_sigma_scale(validation["true"], validation["pred"], validation["sigma"])
    calibrated_sigma = test["sigma"] * sigma_fit["scale"]
    metrics = point_summary(official, test["true"], test["pred"], test["tide"])
    metrics.update({f"calibrated_{key}": value for key, value in probabilistic_metrics(test["true"], test["pred"], calibrated_sigma).items()})
    metrics.update({
        "variant": variant_name,
        "seed": seed,
        "best_epoch": best_state["epoch"],
        "best_validation_nll": best_score,
        "training_seconds": time.perf_counter() - start,
        "trainable_parameters": trainable_parameters,
        "device": str(device),
        "validation_sigma_scale": sigma_fit["scale"],
        "test_used_for_selection": False,
    })
    if settings["freeze_experts"]:
        baseline_source = test
    else:
        baseline_model = build_model(official, data, contract, eta_path, multi_path, True).to(device)
        baseline_source = predict(baseline_model, loaders["test"], scale)
        del baseline_model
    baseline_pred = fixed_hsdt(baseline_source["eta"], baseline_source["multi"])
    if not np.allclose(test["true"], baseline_source["true"], atol=1e-7, rtol=0.0):
        raise RuntimeError("Aligned C4 and formal HS-DT targets are not identical")
    baseline = point_summary(official, baseline_source["true"], baseline_pred, baseline_source["tide"])
    comparison = {
        "seed": seed,
        "variant": variant_name,
        "hsdt_seq_residual_R2": baseline["seq_residual_R2"],
        "aligned_c4_seq_residual_R2": metrics["seq_residual_R2"],
        "delta_seq_residual_R2": metrics["seq_residual_R2"] - baseline["seq_residual_R2"],
        "hsdt_last_residual_R2": baseline["last_residual_R2"],
        "aligned_c4_last_residual_R2": metrics["last_residual_R2"],
        "delta_last_residual_R2": metrics["last_residual_R2"] - baseline["last_residual_R2"],
        "hsdt_q95_residual_R2": baseline["extreme_abs_q95_residual_R2"],
        "aligned_c4_q95_residual_R2": metrics["extreme_abs_q95_residual_R2"],
        "delta_q95_residual_R2": metrics["extreme_abs_q95_residual_R2"] - baseline["extreme_abs_q95_residual_R2"],
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame([comparison]).to_csv(output_dir / "comparison_with_same_checkpoint_hsdt.csv", index=False)
    pd.DataFrame(metrics_by_horizon(test["true"], test["pred"], calibrated_sigma)).to_csv(output_dir / "metrics_by_horizon.csv", index=False)
    pd.DataFrame(metrics_by_station(test["true"], test["pred"], calibrated_sigma, official.v2.STATION_IDS)).to_csv(output_dir / "metrics_by_station.csv", index=False)
    torch.save({
        "model_state_dict": best_state["model"],
        "physics_ode_state_dict": best_state["physics"] if settings["physics"] else None,
        "metadata": {
            "variant": variant_name,
            "seed": seed,
            "best_epoch": best_state["epoch"],
            "train_gate": settings["train_gate"],
            "train_correction": settings["train_correction"],
            "test_used_for_selection": False,
        },
    }, output_dir / "best_checkpoint.pt")
    manifest = {
        "variant": variant_name,
        "seed": seed,
        "expert_mode": "frozen_formal_hsdt_checkpoints" if settings["freeze_experts"] else "initialized_from_formal_hsdt_checkpoints_then_jointly_finetuned",
        "physics": settings["physics"],
        "train_gate": settings["train_gate"],
        "train_correction": settings["train_correction"],
        "trainable_parameters": trainable_parameters,
        "device": str(device),
        "max_epochs": max_epochs,
        "spline": False,
        "spline_omission_reason": "Loading the frozen GWN checkpoints requires the original tanh/ReLU architecture.",
        "hsdt_rule_at_initialization": "eta/multistate mean at leads 1-23; multistate at lead 24",
        "eta_checkpoint": str(eta_path.resolve()),
        "eta_checkpoint_sha256": sha256_file(eta_path),
        "multi_checkpoint": str(multi_path.resolve()),
        "multi_checkpoint_sha256": sha256_file(multi_path),
        "selection_metric": "validation_gaussian_nll",
        "test_used_for_selection": False,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "COMPLETE.json").write_text(json.dumps({"complete": True, "variant": variant_name, "seed": seed}, indent=2), encoding="utf-8")
    return {**metrics, **comparison}


def main():
    parser = argparse.ArgumentParser(description="Train C4 on frozen HS-DT-GWN experts")
    parser.add_argument("--tiera-root", type=Path, default=WORKSPACE)
    parser.add_argument("--output-root", type=Path, default=WORKSPACE / "results" / "c4")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=["C4_HSDT_FROZEN_NO_PHYS"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eta-results-root", type=Path, default=WORKSPACE / "results" / "priority12_physics_graph_wavenet")
    parser.add_argument("--multi-results-root", type=Path, default=WORKSPACE / "results" / "priority12_physics_graph_wavenet")
    parser.add_argument(
        "--eta-seed42-checkpoint",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    contract = TierAContract()
    official, formal_args, data = load_formal_data(args.tiera_root, contract)
    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            rows.append(run_variant(args, variant, seed, official, formal_args, data, contract, torch.device(device_name)))
    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_root / ("dry_run_summary.csv" if args.dry_run else "all_runs.csv"), index=False)


if __name__ == "__main__":
    main()
