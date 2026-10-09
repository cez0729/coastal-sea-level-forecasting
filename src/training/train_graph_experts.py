from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader


ROOT = REPO_ROOT
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "priority12_physics_graph_wavenet"
CONFIGS = [
    "gwn_eta_only",
    "gwn_multistate_no_physics",
    "gwn_multistate_aux_only",
    "gwn_multistate_terminal_only",
    "gwn_multistate_physics",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


final4 = load_module("final4_p104", REPO_ROOT / 'src/data/feature_pipeline.py')
priority1 = load_module("priority1_p104", REPO_ROOT / 'src/models/graph/graph_wavenet.py')
v2 = final4.v2
v3 = final4.v3
v4 = final4.v4


class GraphWaveNetMultistate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        adj: np.ndarray,
        hidden_dim: int,
        horizon: int,
        num_states: int,
        diffusion_steps: int,
        blocks: int,
        dropout: float,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.num_states = int(num_states)
        supports = priority1.make_diffusion_supports(adj, diffusion_steps)
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
        dilations = [2 ** (index % 4) for index in range(blocks)]
        self.blocks = nn.ModuleList(
            [priority1.GraphWaveNetBlock(hidden_dim, supports, dilation=d, kernel_size=2, dropout=dropout) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, horizon * num_states, kernel_size=(1, 1)),
        )
        self.register_buffer("fixed_adjacency", torch.tensor(adj, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 3, 2, 1))
        for block in self.blocks:
            h = block(h)
        out = self.head(h)[..., -1].permute(0, 2, 1)
        return out.reshape(x.shape[0], x.shape[2], self.horizon, self.num_states)


def set_reproducible(seed: int, cpu_threads: int) -> None:
    v2.set_seed(seed)
    torch.set_num_threads(max(1, cpu_threads))
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(dataset, args, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def physics_lambda(epoch: int, args) -> float:
    return v2.physics_lambda_for_epoch(epoch, args.physics_lambda, args.physics_warmup_epochs, args.physics_ramp_epochs)


def save_checkpoint(path: Path, model: nn.Module, physics_ode: nn.Module | None, metadata: dict) -> None:
    payload = {"model_state_dict": model.state_dict(), "metadata": metadata}
    if physics_ode is not None:
        payload["physics_ode_state_dict"] = physics_ode.state_dict()
    torch.save(payload, path)


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def rng_state_payload(train_loader: DataLoader) -> dict:
    return {
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "train_loader_generator_state": train_loader.generator.get_state(),
    }


def restore_rng_state(payload: dict, train_loader: DataLoader) -> None:
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    train_loader.generator.set_state(payload["train_loader_generator_state"])


def train_eta_only(model, train_loader, val_loader, args, device, run_dir: Path) -> tuple[pd.DataFrame, float, dict]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    best = None
    best_val = float("inf")
    bad = 0
    rows = []
    start_epoch = 1
    elapsed_before = 0.0
    resume_path = run_dir / "last_epoch_checkpoint.pt"
    if args.resume and resume_path.exists():
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        best = payload["best_state_dict"]
        best_val = float(payload["best_val"])
        bad = int(payload["bad_epochs"])
        rows = payload["history"]
        start_epoch = int(payload["epoch"]) + 1
        elapsed_before = float(payload.get("elapsed_seconds", 0.0))
        restore_rng_state(payload, train_loader)
        print(f"Resuming eta-only run from epoch {start_epoch}; best={best_val:.6f}")
    total_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_losses = []
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                val_losses.append(float(criterion(model(xb.to(device)), yb.to(device)).detach().cpu()))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        row = {
            "epoch": epoch,
            "train_eta_data_loss": float(np.mean(train_losses)),
            "val_eta_data_loss": val_loss,
            "selection_score": val_loss,
            "physics_lambda": 0.0,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "lr": optimizer.param_groups[0]["lr"],
        }
        rows.append(row)
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"epoch={epoch:03d} train_eta={row['train_eta_data_loss']:.6f} val_eta={val_loss:.6f}")
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            bad = 0
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad += 1
        if epoch % args.epoch_checkpoint_every == 0 or bad >= args.patience:
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_state_dict": best,
                    "best_val": best_val,
                    "bad_epochs": bad,
                    "history": rows,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - total_start,
                    **rng_state_payload(train_loader),
                },
                resume_path,
            )
        if bad >= args.patience:
            print(f"Early stopping at epoch {epoch}; best={best_val:.6f}")
            break
    if best is not None:
        model.load_state_dict(best)
    timing = {"training_seconds": elapsed_before + time.perf_counter() - total_start, "epochs_completed": len(rows)}
    return pd.DataFrame(rows), best_val, timing


def make_multistate_criterion(physics_ode, data, args, use_physics: bool, horizon: int):
    return v4.WeightedMultistateLoss(
        physics_ode=physics_ode,
        state_scale=data["state_scale"],
        delta_scale=data["delta_scale"],
        aux_weight=args.aux_weight,
        physics_loss_type="huber",
        physics_state_weights=[1.0, 0.35, 0.35, 0.25],
        last_step_weight=args.last_step_weight,
        ode_coef_l2=args.ode_coef_l2 if use_physics else 0.0,
        horizon=horizon,
        physics_lead_gamma=0.0,
        data_lead_gamma=0.0,
        extreme_alpha=0.0,
        extreme_quantile=args.extreme_quantile,
        train_abs_eta_threshold=data["train_abs_eta_threshold"],
    )


@torch.no_grad()
def evaluate_multistate(model, criterion, loader, adj, physics_lambda_value: float, device) -> dict[str, float]:
    model.eval()
    criterion.physics_ode.eval()
    totals = {}
    count = 0
    for xb, target, _, init_states, phys_seq in loader:
        pred = model(xb.to(device))
        _, parts = criterion(
            pred,
            target.to(device),
            init_states.to(device),
            phys_seq.to(device),
            adj,
            physics_lambda_value,
        )
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


def train_multistate(model, physics_ode, data, train_loader, val_loader, args, device, use_physics: bool, run_dir: Path):
    horizon = args.horizon
    criterion = make_multistate_criterion(physics_ode, data, args, use_physics, horizon).to(device)
    parameter_groups = [{"params": model.parameters(), "lr": args.lr}]
    if use_physics:
        parameter_groups.append({"params": physics_ode.parameters(), "lr": args.lr * args.physics_lr_mult})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    adj = model.fixed_adjacency
    best = None
    best_val = float("inf")
    bad = 0
    rows = []
    start_epoch = 1
    elapsed_before = 0.0
    resume_path = run_dir / "last_epoch_checkpoint.pt"
    if args.resume and resume_path.exists():
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        physics_ode.load_state_dict(payload["physics_ode_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        best = payload["best_state_dict"]
        best_val = float(payload["best_val"])
        bad = int(payload["bad_epochs"])
        rows = payload["history"]
        start_epoch = int(payload["epoch"]) + 1
        elapsed_before = float(payload.get("elapsed_seconds", 0.0))
        restore_rng_state(payload, train_loader)
        print(f"Resuming multistate run from epoch {start_epoch}; best={best_val:.6f}")
    total_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        lam = physics_lambda(epoch, args) if use_physics else 0.0
        model.train()
        physics_ode.train(use_physics)
        totals = {}
        count = 0
        for xb, target, _, init_states, phys_seq in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb.to(device))
            loss, parts = criterion(
                pred,
                target.to(device),
                init_states.to(device),
                phys_seq.to(device),
                adj,
                lam,
            )
            loss.backward()
            params = list(model.parameters()) + (list(physics_ode.parameters()) if use_physics else [])
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        train_parts = {key: value / max(1, count) for key, value in totals.items()}
        val_parts = evaluate_multistate(model, criterion, val_loader, adj, lam, device)
        val_score = val_parts["eta_data_loss"]
        scheduler.step(val_score)
        row = {
            "epoch": epoch,
            "selection_score": val_score,
            **{f"train_{key}": value for key, value in train_parts.items()},
            **{f"val_{key}": value for key, value in val_parts.items()},
            "epoch_seconds": time.perf_counter() - epoch_start,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if use_physics:
            row.update(physics_ode.coefficients())
        rows.append(row)
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch={epoch:03d} lambda={lam:.6f} train_eta={train_parts['eta_data_loss']:.6f} "
                f"val_eta={val_parts['eta_data_loss']:.6f} val_last={val_parts['last_loss']:.6f}"
            )
        if val_score < best_val - args.min_delta:
            best_val = val_score
            bad = 0
            best = {
                "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "physics": {key: value.detach().cpu().clone() for key, value in physics_ode.state_dict().items()},
            }
        else:
            bad += 1
        if epoch % args.epoch_checkpoint_every == 0 or bad >= args.patience:
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "physics_ode_state_dict": physics_ode.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_state_dict": best,
                    "best_val": best_val,
                    "bad_epochs": bad,
                    "history": rows,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - total_start,
                    **rng_state_payload(train_loader),
                },
                resume_path,
            )
        if bad >= args.patience:
            print(f"Early stopping at epoch {epoch}; best={best_val:.6f}")
            break
    if best is not None:
        model.load_state_dict(best["model"])
        physics_ode.load_state_dict(best["physics"])
    timing = {"training_seconds": elapsed_before + time.perf_counter() - total_start, "epochs_completed": len(rows)}
    return pd.DataFrame(rows), best_val, timing


@torch.no_grad()
def predict_eta(model, loader, device):
    model.eval()
    preds, trues, tides = [], [], []
    start = time.perf_counter()
    for xb, target, tide in loader:
        preds.append(model(xb.to(device)).detach().cpu().numpy())
        trues.append(target.numpy())
        tides.append(tide.numpy())
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(tides), time.perf_counter() - start


@torch.no_grad()
def predict_multistate(model, loader, device):
    model.eval()
    preds, trues, tides = [], [], []
    start = time.perf_counter()
    for xb, target, tide, _, _ in loader:
        preds.append(model(xb.to(device)).detach().cpu().numpy())
        trues.append(target.numpy())
        tides.append(tide.numpy())
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(tides), time.perf_counter() - start


def operational_event_metrics(true: np.ndarray, pred: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    true_last = true[:, :, -1]
    pred_last = pred[:, :, -1]
    labels = true_last >= thresholds[None, :]
    calls = pred_last >= thresholds[None, :]
    tp = int(np.sum(labels & calls))
    fp = int(np.sum(~labels & calls))
    fn = int(np.sum(labels & ~calls))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    csi = tp / max(1, tp + fp + fn)
    flat_labels = labels.reshape(-1).astype(int)
    pr_auc = float(average_precision_score(flat_labels, pred_last.reshape(-1))) if len(np.unique(flat_labels)) > 1 else np.nan
    out = {
        "event_threshold_source": "training_stationwise_q95",
        "event_true_count": int(labels.sum()),
        "event_precision": precision,
        "event_recall": recall,
        "event_F1": f1,
        "event_CSI": csi,
        "event_PR_AUC": pr_auc,
    }
    out.update(priority1.tolerant_event_detection(true_last, pred_last, thresholds, tolerance=6))
    return out


def evaluate_and_save(config: str, seed: int, run_dir: Path, model, physics_ode, best_val: float, history: pd.DataFrame, timing: dict, data, test_loader, args, device):
    if config == "gwn_eta_only":
        pred, true, tide, inference_seconds = predict_eta(model, test_loader, device)
        metrics = final4.summarize_single(true, pred, tide)
        pred_states = None
    else:
        pred_states, true_states, tide, inference_seconds = predict_multistate(model, test_loader, device)
        pred = pred_states[..., 0]
        true = true_states[..., 0]
        metrics = v2.summarize_metrics(true_states, pred_states, tide)
        metrics.update(v3.summarize_extreme_metrics(true_states, pred_states))
    train_end = int(len(data["arrays"]["residual"]) * args.train_ratio)
    thresholds = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    metrics.update(operational_event_metrics(true, pred, thresholds))
    predictor_params = count_parameters(model)
    uses_physics = config == "gwn_multistate_physics"
    physics_params = count_parameters(physics_ode) if physics_ode is not None and uses_physics else 0
    row = {
        "seed": seed,
        "config": config,
        "horizon": args.horizon,
        "best_val_eta_loss": best_val,
        "predictor_trainable_parameters": predictor_params,
        "training_only_physics_parameters": physics_params,
        "effective_history_hours": 1 + sum(2 ** (index % 4) for index in range(args.gwn_blocks)),
        "physics_lambda": args.physics_lambda if uses_physics else 0.0,
        "aux_weight": args.aux_weight if config != "gwn_eta_only" else 0.0,
        "last_step_weight": args.last_step_weight if config != "gwn_eta_only" else 0.0,
        "training_seconds": timing["training_seconds"],
        "epochs_completed": timing["epochs_completed"],
        "inference_seconds_full_test": inference_seconds,
        "inference_ms_per_sample": inference_seconds * 1000.0 / max(1, pred.shape[0]),
        "num_test_samples": int(pred.shape[0]),
        **metrics,
    }
    history.to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
    if pred_states is None:
        np.savez_compressed(run_dir / "predictions.npz", pred_residual=pred, true_residual=true, target_tide=tide, station_ids=np.asarray(v2.STATION_IDS))
    else:
        np.savez_compressed(run_dir / "predictions.npz", pred_states=pred_states, true_states=true_states, target_tide=tide, station_ids=np.asarray(v2.STATION_IDS))
    metadata = {key: row[key] for key in ["seed", "config", "horizon", "best_val_eta_loss", "predictor_trainable_parameters", "training_only_physics_parameters"]}
    save_checkpoint(run_dir / "best_checkpoint.pt", model, physics_ode if uses_physics else None, metadata)
    (run_dir / "COMPLETE.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return row


def reuse_priority1_eta_baseline(seed: int, run_dir: Path, data, args) -> dict:
    source_dir = ROOT / "results" / "priority1_graph_baselines" / f"seed_{seed}" / f"horizon_{args.horizon}h" / "graph_wavenet"
    required = [source_dir / "metrics.csv", source_dir / "training_log.csv", source_dir / "predictions.npz"]
    if not all(path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        raise FileNotFoundError(f"Cannot reuse the published Priority-1 Graph WaveNet baseline; missing: {missing}")

    source_metrics = pd.read_csv(source_dir / "metrics.csv").iloc[0].to_dict()
    source_history = pd.read_csv(source_dir / "training_log.csv")
    with np.load(source_dir / "predictions.npz", allow_pickle=True) as prediction_file:
        pred = prediction_file["pred_residual"]
        true = prediction_file["true_residual"]
    train_end = int(len(data["arrays"]["residual"]) * args.train_ratio)
    thresholds = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    metric_values = {
        key: value
        for key, value in source_metrics.items()
        if key.startswith(("seq_", "last_", "extreme_"))
    }
    metric_values.update(operational_event_metrics(true, pred, thresholds))
    row = {
        "seed": seed,
        "config": "gwn_eta_only",
        "horizon": args.horizon,
        "best_val_eta_loss": float(source_metrics["best_val_loss"]),
        "predictor_trainable_parameters": 181912,
        "training_only_physics_parameters": 0,
        "effective_history_hours": 1 + sum(2 ** (index % 4) for index in range(args.gwn_blocks)),
        "physics_lambda": 0.0,
        "aux_weight": 0.0,
        "last_step_weight": 0.0,
        "training_seconds": np.nan,
        "epochs_completed": int(len(source_history)),
        "inference_seconds_full_test": np.nan,
        "inference_ms_per_sample": np.nan,
        "num_test_samples": int(pred.shape[0]),
        "reused_existing_priority1_baseline": True,
        "source_priority1_run": str(source_dir.relative_to(ROOT)),
        **metric_values,
    }
    source_history.to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
    shutil.copy2(source_dir / "predictions.npz", run_dir / "predictions.npz")
    provenance = {
        "seed": seed,
        "config": "gwn_eta_only",
        "horizon": args.horizon,
        "reused_existing_priority1_baseline": True,
        "source_priority1_run": str(source_dir.relative_to(ROOT)),
        "reason": "Identical published architecture/protocol; retained as a broad baseline, not a physics-only attribution pair.",
    }
    (run_dir / "REUSED_PRIORITY1_BASELINE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (run_dir / "COMPLETE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(
        f"Reused published Priority-1 Graph WaveNet seed={seed}: seq_R2={row['seq_residual_R2']:.4f} "
        f"terminal_R2={row['last_residual_R2']:.4f}"
    )
    return row


def run_config(config: str, seed: int, args, data, device) -> dict:
    run_dir = Path(args.output_dir) / f"seed_{seed}" / f"horizon_{args.horizon}h" / config
    metrics_path = run_dir / "metrics.csv"
    if args.resume and metrics_path.exists() and (run_dir / "predictions.npz").exists() and (run_dir / "COMPLETE.json").exists():
        print(f"Skipping complete run: seed={seed} config={config}")
        return pd.read_csv(metrics_path).iloc[0].to_dict()
    run_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible(seed, args.cpu_threads)
    if config == "gwn_eta_only" and args.reuse_priority1_eta:
        return reuse_priority1_eta_baseline(seed, run_dir, data, args)
    adj = data["graph_priors"][args.fixed_graph_type]
    print("\n" + "=" * 96)
    print(f"Running seed={seed} config={config} device={device} train_stride={args.train_stride}")
    if config == "gwn_eta_only":
        train_loader = make_loader(data["single_train"], args, True, seed)
        val_loader = make_loader(data["single_val"], args, False, seed)
        test_loader = make_loader(data["single_test"], args, False, seed)
        model = priority1.GraphWaveNetForecaster(
            data["feats"], adj, args.hidden_dim, args.horizon, args.diffusion_steps, args.gwn_blocks, args.dropout
        ).to(device)
        history, best_val, timing = train_eta_only(model, train_loader, val_loader, args, device, run_dir)
        physics_ode = None
    else:
        train_loader = make_loader(data["multi_train"], args, True, seed)
        val_loader = make_loader(data["multi_val"], args, False, seed)
        test_loader = make_loader(data["multi_test"], args, False, seed)
        model = GraphWaveNetMultistate(
            data["feats"], adj, args.hidden_dim, args.horizon, 4, args.diffusion_steps, args.gwn_blocks, args.dropout
        ).to(device)
        physics_ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
        use_physics = config == "gwn_multistate_physics"
        # Factorial attribution controls: preserve the same four-state head and
        # backbone, changing only one objective term at a time.
        objective_args = argparse.Namespace(**vars(args))
        if config == "gwn_multistate_aux_only":
            objective_args.last_step_weight = 0.0
        elif config == "gwn_multistate_terminal_only":
            objective_args.aux_weight = 0.0
        history, best_val, timing = train_multistate(
            model, physics_ode, data, train_loader, val_loader, objective_args, device, use_physics, run_dir
        )
    row = evaluate_and_save(config, seed, run_dir, model, physics_ode, best_val, history, timing, data, test_loader, args, device)
    print(
        f"Finished seed={seed} config={config}: seq_R2={row['seq_residual_R2']:.4f} "
        f"terminal_R2={row['last_residual_R2']:.4f} q95_R2={row['extreme_abs_q95_residual_R2']:.4f}"
    )
    return row


def exact_wilcoxon_greater(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if not len(values):
        return np.nan
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(values, alternative="greater", method="exact").pvalue)
    except Exception:
        positive = int(np.sum(values > 0))
        return float(sum(math.comb(len(values), k) for k in range(positive, len(values) + 1)) / (2 ** len(values)))


def load_all_completed(output_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(output_dir.glob("seed_*/horizon_24h/*/metrics.csv"))]
    if not frames:
        raise RuntimeError(f"No completed metrics found under {output_dir}")
    return pd.concat(frames, ignore_index=True)


def merge_results(args) -> None:
    output_dir = Path(args.output_dir)
    data = load_all_completed(output_dir)
    data = data[data["seed"].isin(args.seeds) & data["config"].isin(args.configs)].copy()
    data = data.sort_values(["seed", "config"]).drop_duplicates(["seed", "config"], keep="last")
    data.to_csv(output_dir / "priority2_physics_gwn_all_runs.csv", index=False)
    metrics = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "event_PR_AUC",
        "event_recall",
        "event_F1",
        "event_CSI",
        "event_detect_pm6h_rate",
        "event_detect_pm6h_peak_timing_MAE",
        "training_seconds",
        "epochs_completed",
    ]
    summary = data.groupby("config")[[c for c in metrics if c in data]].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x) for col in summary.columns.to_flat_index()]
    summary.to_csv(output_dir / "priority2_physics_gwn_mean_std.csv", index=False)

    pivot = data.pivot(index="seed", columns="config", values=[c for c in metrics if c in data])
    comparisons = [
        ("auxiliary_state_effect", "gwn_multistate_no_physics", "gwn_eta_only"),
        ("physics_net_effect", "gwn_multistate_physics", "gwn_multistate_no_physics"),
        ("broad_physics_system_effect", "gwn_multistate_physics", "gwn_eta_only"),
    ]
    paired_rows = []
    tests = []
    for label, left, right in comparisons:
        for metric in [c for c in metrics if c in data]:
            if (metric, left) not in pivot or (metric, right) not in pivot:
                continue
            delta = pivot[(metric, left)] - pivot[(metric, right)]
            for seed, value in delta.items():
                paired_rows.append({"comparison": label, "metric": metric, "seed": seed, "delta": value})
            tests.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "std_delta": float(delta.std()),
                    "wins": int((delta > 0).sum()) if not metric.endswith(("RMSE", "MAE", "seconds")) else int((delta < 0).sum()),
                    "count": int(delta.notna().sum()),
                    "wilcoxon_greater_p": exact_wilcoxon_greater(delta.to_numpy()) if not metric.endswith(("RMSE", "MAE", "seconds")) else exact_wilcoxon_greater((-delta).to_numpy()),
                }
            )
    paired_df = pd.DataFrame(paired_rows, columns=["comparison", "metric", "seed", "delta"])
    paired_df.to_csv(output_dir / "priority2_physics_gwn_paired_deltas.csv", index=False)
    tests_df = pd.DataFrame(
        tests,
        columns=["comparison", "metric", "mean_delta", "std_delta", "wins", "count", "wilcoxon_greater_p"],
    )
    tests_df.to_csv(output_dir / "priority2_physics_gwn_paired_tests.csv", index=False)

    station_rows = []
    horizon_rows = []
    for path in sorted(output_dir.glob("seed_*/horizon_24h/*/predictions.npz")):
        z = np.load(path, allow_pickle=True)
        pred = z["pred_states"][..., 0] if "pred_states" in z.files else z["pred_residual"]
        true = z["true_states"][..., 0] if "true_states" in z.files else z["true_residual"]
        seed = int(path.parts[-4].split("_")[-1])
        config = path.parent.name
        for station, station_id in enumerate(v2.STATION_IDS):
            values = priority1.r2_rmse_mae(true[:, station, -1], pred[:, station, -1])
            station_rows.append({"seed": seed, "config": config, "station_id": station_id, **{f"last_residual_{k}": v for k, v in values.items()}})
        for lead in range(pred.shape[-1]):
            values = priority1.r2_rmse_mae(true[:, :, lead], pred[:, :, lead])
            horizon_rows.append({"seed": seed, "config": config, "lead_hour": lead + 1, **{f"residual_{k}": v for k, v in values.items()}})
    pd.DataFrame(station_rows).to_csv(output_dir / "priority2_physics_gwn_per_station.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(output_dir / "priority2_physics_gwn_per_horizon.csv", index=False)

    plot_summary(output_dir, data, paired_df, pd.DataFrame(horizon_rows))
    physics_test = tests_df[(tests_df["comparison"] == "physics_net_effect") & (tests_df["metric"].isin(["seq_residual_R2", "last_residual_R2", "extreme_abs_q95_residual_R2", "event_recall"]))]
    lines = [
        "# Matched Physics-Graph-WaveNet Experiment",
        "",
        "Strict matched attribution compares `gwn_multistate_physics` with `gwn_multistate_no_physics`.",
        "The eta-only configuration is a broad architecture baseline, not the physics-only ablation.",
        "",
        "## Mean and standard deviation",
        "",
        summary.to_string(index=False),
        "",
        "## Net physics effect",
        "",
        physics_test.to_string(index=False),
    ]
    (output_dir / "PRIORITY2_PHYSICS_GWN_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nNet physics effect")
    print(physics_test.to_string(index=False))


def plot_summary(output_dir: Path, data: pd.DataFrame, paired: pd.DataFrame, horizon: pd.DataFrame) -> None:
    available = set(data["config"].unique())
    order = [key for key in CONFIGS if key in available]
    labels = {
        "gwn_eta_only": "GWN eta-only",
        "gwn_multistate_no_physics": "GWN multistate, no physics",
        "gwn_multistate_aux_only": "GWN multistate, auxiliary only",
        "gwn_multistate_terminal_only": "GWN multistate, terminal only",
        "gwn_multistate_physics": "GWN multistate + physics",
    }
    color_map = {
        "gwn_eta_only": "#D98C10",
        "gwn_multistate_no_physics": "#2A6FBB",
        "gwn_multistate_aux_only": "#5A9E6F",
        "gwn_multistate_terminal_only": "#8172B3",
        "gwn_multistate_physics": "#C44E52",
    }
    colors = [color_map[key] for key in order]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    for ax, metric, title in zip(
        axes,
        ["seq_residual_R2", "last_residual_R2", "extreme_abs_q95_residual_R2"],
        ["Trajectory R2", "24-h terminal R2", "Descriptive q95 R2"],
    ):
        grouped = data.groupby("config")[metric].agg(["mean", "std"]).reindex(order)
        grouped["std"] = grouped["std"].fillna(0.0)
        x = np.arange(len(order))
        ax.bar(x, grouped["mean"], yerr=grouped["std"], capsize=4, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[key] for key in order], rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Matched Graph WaveNet attribution: auxiliary states versus physics residual", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "priority2_physics_gwn_main_comparison.png", dpi=280, bbox_inches="tight")
    plt.close(fig)

    if not paired.empty:
        sub = paired[(paired["comparison"] == "physics_net_effect") & (paired["metric"] == "last_residual_R2")]
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        for _, row in sub.iterrows():
            seed = int(row["seed"])
            pair = data[data["seed"] == seed].set_index("config")["last_residual_R2"]
            ax.plot([0, 1], [pair["gwn_multistate_no_physics"], pair["gwn_multistate_physics"]], marker="o", alpha=0.75, label=f"seed {seed}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Multistate no physics", "Multistate + physics"])
        ax.set_ylabel("24-h terminal residual R2")
        ax.set_title("Paired net physics effect on the strongest backbone")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "priority2_physics_gwn_paired_terminal.png", dpi=280)
        plt.close(fig)

    if not horizon.empty:
        h = horizon.groupby(["config", "lead_hour"])["residual_R2"].agg(["mean", "std"]).reset_index()
        h["std"] = h["std"].fillna(0.0)
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        for key, color in zip(order, colors):
            sub = h[h["config"] == key]
            ax.plot(sub["lead_hour"], sub["mean"], color=color, label=labels[key], linewidth=2)
            ax.fill_between(sub["lead_hour"], sub["mean"] - sub["std"], sub["mean"] + sub["std"], color=color, alpha=0.10)
        ax.set_xlabel("Forecast lead (h)")
        ax.set_ylabel("Residual R2")
        ax.set_xlim(1, 24)
        ax.set_title("Lead-dependent skill under matched Graph WaveNet training")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.22)
        fig.tight_layout()
        fig.savefig(output_dir / "priority2_physics_gwn_per_horizon.png", dpi=280)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched Graph WaveNet auxiliary-state and physics-residual experiment.")
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024, 2025, 3407])
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=CONFIGS)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", choices=["identity", "distance", "corr"], default="distance")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--gwn-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--physics-lr-mult", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--physics-lambda", type=float, default=0.0002)
    parser.add_argument("--physics-warmup-epochs", type=int, default=8)
    parser.add_argument("--physics-ramp-epochs", type=int, default=14)
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.2)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--epoch-checkpoint-every", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--reuse-priority1-eta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_text = json.dumps(vars(args), indent=2)
    canonical_config = output_dir / "experiment_config.json"
    if not canonical_config.exists():
        canonical_config.write_text(config_text, encoding="utf-8")
    seed_label = "_".join(str(seed) for seed in args.seeds)
    (output_dir / f"experiment_config_seeds_{seed_label}.json").write_text(config_text, encoding="utf-8")
    if args.mode == "merge":
        merge_results(args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        set_reproducible(seed, args.cpu_threads)
        data = final4.build_enhanced_data(args, args.horizon, add_ode_prior=False)
        data["station_meta"].to_csv(output_dir / "station_meta_used.csv", index=False)
        pd.DataFrame({"feature": data["feature_cols"]}).to_csv(output_dir / "feature_cols_used.csv", index=False)
        pd.DataFrame({"physics_forcing_feature": data["physics_cols"]}).to_csv(output_dir / "physics_cols_used.csv", index=False)
        pd.DataFrame(
            {
                "state": v2.STATE_NAMES,
                "state_scale": data["state_scale"],
                "delta_scale": data["delta_scale"],
                "physics_state_weight": [1.0, 0.35, 0.35, 0.25],
            }
        ).to_csv(output_dir / "state_scaling_used.csv", index=False)
        for config in args.configs:
            rows.append(run_config(config, seed, args, data, device))
            pd.DataFrame(rows).to_csv(output_dir / "priority2_physics_gwn_partial.csv", index=False)
    merge_results(args)


if __name__ == "__main__":
    main()
