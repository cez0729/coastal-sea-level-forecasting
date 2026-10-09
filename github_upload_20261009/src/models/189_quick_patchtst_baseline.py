"""Validation-locked PatchTST-style baseline for the strict coastal task.

The model is deliberately non-graph: each station is encoded independently
with shared weights. A temporal patch contains all locally available features,
so the input schema and forecast origins match the Graph WaveNet experiments.
No test target is used for model selection or hyperparameter tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PRIMARY_DATA = ROOT / "data" / "processed_multiyear_2023_2025"
EXTERNAL_DATA = ROOT / "data" / "external_region_delaware_bay_2023_2025" / "processed"
DEFAULT_OUT = ROOT / "results" / "quick_patchtst_strict_20260813"
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p134 = load_module("patchtst_p134", HERE / "134_confirmatory_hsdt_orc_chronological_refit.py")
p187 = load_module("patchtst_p187", HERE / "187_run_delaware_bay_external_confirmation.py")


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(threads)))
    torch.use_deterministic_algorithms(True, warn_only=True)


class LocalPatchTransformer(nn.Module):
    """Shared station-wise patch transformer with a direct 24-lead head."""

    def __init__(
        self,
        input_dim: int,
        window: int,
        horizon: int,
        patch_len: int = 6,
        patch_stride: int = 3,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if patch_len > window or (window - patch_len) % patch_stride != 0:
            raise ValueError("Patch length/stride must tile the fixed input window")
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.patch_count = 1 + (window - patch_len) // patch_stride
        self.patch_projection = nn.Linear(input_dim * patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(self.patch_count * d_model, horizon),
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,N,F] -> station-wise [B*N,P,patch_len*F]
        batch, _, nodes, feats = x.shape
        local = x.permute(0, 2, 1, 3).reshape(batch * nodes, x.shape[1], feats)
        patches = local.unfold(1, self.patch_len, self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).reshape(batch * nodes, self.patch_count, -1)
        encoded = self.patch_projection(patches) + self.position
        encoded = self.norm(self.encoder(encoded))
        return self.head(encoded).reshape(batch, nodes, -1)


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(true, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    denom = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - p) ** 2) / max(denom, 1e-12))


def score(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error24 = pred[..., -1] - true[..., -1]
    return {
        "sequence_residual_R2": r2_score(true, pred),
        "lead24_residual_R2": r2_score(true[..., -1], pred[..., -1]),
        "lead24_residual_RMSE": float(np.sqrt(np.mean(error24**2))),
    }


@torch.no_grad()
def validation_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y, _ in loader:
        prediction = model(x.to(device))
        total += float(torch.sum((prediction - y.to(device)) ** 2).cpu())
        count += int(y.numel())
    return total / max(count, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    predictions, targets, tides = [], [], []
    for x, y, tide in loader:
        predictions.append(model(x.to(device)).cpu().numpy())
        targets.append(y.numpy())
        tides.append(tide.numpy())
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(tides)


def train_seed(seed: int, data: dict, args, region: str, station_ids: list[str]) -> dict:
    output = args.output_dir / region / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.csv"
    if args.resume and metrics_path.exists() and (output / "predictions.npz").exists():
        return pd.read_csv(metrics_path).iloc[0].to_dict()

    set_seed(seed, args.cpu_threads)
    generator = torch.Generator().manual_seed(seed)
    loader_args = dict(batch_size=args.batch_size, num_workers=0)
    train_loader = DataLoader(data["single_train"], shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(data["single_val"], shuffle=False, **loader_args)
    test_loader = DataLoader(data["single_test"], shuffle=False, **loader_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalPatchTransformer(
        input_dim=data["feats"], window=args.window, horizon=args.horizon,
        patch_len=args.patch_len, patch_stride=args.patch_stride,
        d_model=args.d_model, heads=args.heads, layers=args.layers,
        ff_dim=args.ff_dim, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    best_state, best_val, best_epoch, bad_epochs = None, float("inf"), 0, 0
    rows = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_sum, train_count = 0.0, 0
        for x, y, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x.to(device))
            loss = torch.mean((prediction - y.to(device)) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * int(y.numel())
            train_count += int(y.numel())
        val = validation_loss(model, val_loader, device)
        scheduler.step(val)
        rows.append({
            "epoch": epoch,
            "train_mse": train_sum / max(train_count, 1),
            "validation_mse": val,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        })
        print(
            f"region={region} seed={seed} epoch={epoch:03d} "
            f"train={rows[-1]['train_mse']:.6f} val={val:.6f} "
            f"seconds={rows[-1]['epoch_seconds']:.1f}", flush=True,
        )
        if val < best_val - args.min_delta:
            best_val, best_epoch, bad_epochs = val, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was selected")
    model.load_state_dict(best_state)
    prediction, true, tide = predict(model, test_loader, device)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Non-finite PatchTST prediction")
    metrics = {
        "period": "strict_2025_h2" if region == "primary_7_station" else "tier_d_2025_h2",
        "region": region,
        "model": "local_patchtst",
        "seed": seed,
        **score(true, prediction),
        "best_validation_mse": best_val,
        "best_epoch": best_epoch,
        "training_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "test_target_used_for_selection": False,
    }
    pd.DataFrame(rows).to_csv(output / "training_log.csv", index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    test_indices = np.asarray(data["single_test"].indices, dtype=np.int64)
    times = pd.to_datetime(data["arrays"]["time"])[test_indices].to_numpy(dtype="datetime64[ns]")
    np.savez_compressed(
        output / "predictions.npz", pred_residual=prediction.astype(np.float32),
        true_residual=true.astype(np.float32), target_tide=tide.astype(np.float32),
        target_origin_time=times, station_ids=np.asarray(station_ids),
    )
    torch.save({"model_state_dict": best_state, "metrics": metrics}, output / "best_checkpoint.pt")
    pd.DataFrame([
        {"seed": seed, "model": "local_patchtst", "lead_hour": lead + 1,
         "residual_R2": r2_score(true[..., lead], prediction[..., lead]),
         "RMSE": float(np.sqrt(np.mean((true[..., lead] - prediction[..., lead]) ** 2)))}
        for lead in range(args.horizon)
    ]).to_csv(output / "per_lead.csv", index=False)
    station_rows = []
    for index, station_id in enumerate(station_ids):
        station_rows.append({
            "seed": seed, "model": "local_patchtst", "station_id": station_id,
            "sequence_residual_R2": r2_score(true[:, index], prediction[:, index]),
            "lead24_residual_R2": r2_score(true[:, index, -1], prediction[:, index, -1]),
        })
    pd.DataFrame(station_rows).to_csv(output / "per_station.csv", index=False)
    return metrics


def training_args(args):
    return argparse.Namespace(
        fold_train_end="2025-01-01", fold_val_end="2025-07-01", fold_test_end="2026-01-01",
        horizon=args.horizon, window=args.window, train_stride=8,
        physics_forcing_mode="last_input", extreme_quantile=0.90,
    )


def build_region(region: str, args):
    fold_args = training_args(args)
    if region == "primary_7_station":
        p134.configure_data_dir(PRIMARY_DATA)
        data = p134.rolling.build_fold_data(fold_args, args.horizon, add_ode_prior=False)
        station_ids = [str(value) for value in p134.v2.STATION_IDS]
    elif region == "external_10_station":
        station_ids = pd.read_csv(EXTERNAL_DATA / "station_order.csv", dtype={"station_id": str})["station_id"].tolist()
        p187.patch_data_modules(station_ids)
        data = p187.p134.rolling.build_fold_data(fold_args, args.horizon, add_ode_prior=False)
        data = p187.filter_datasets(data, station_ids, args.window, args.horizon)
    else:
        raise ValueError(region)
    return data, station_ids


def config_payload(args) -> dict:
    return {
        "model": "local_patchtst",
        "model_role": "non_graph_modern_time_series_baseline",
        "window": args.window, "horizon": args.horizon,
        "patch_len": args.patch_len, "patch_stride": args.patch_stride,
        "d_model": args.d_model, "heads": args.heads, "layers": args.layers,
        "ff_dim": args.ff_dim, "dropout": args.dropout,
        "batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience,
        "lr": args.lr, "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
        "seeds": args.seeds, "regions": args.regions,
        "train": "2023-2024", "validation": "2025-H1", "test": "2025-H2",
        "selection": "validation MSE only; no hyperparameter sweep",
        "test_target_used_for_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--regions", nargs="+", choices=["primary_7_station", "external_10_station"], default=["primary_7_station"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--patch-len", type=int, default=6)
    parser.add_argument("--patch-stride", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = config_payload(args)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["config_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (args.output_dir / "frozen_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    results = []
    for region in args.regions:
        data, station_ids = build_region(region, args)
        print(
            f"region={region} train={len(data['single_train'])} val={len(data['single_val'])} "
            f"test={len(data['single_test'])} nodes={data['nodes']} features={data['feats']}", flush=True,
        )
        for seed in args.seeds:
            results.append(train_seed(seed, data, args, region, station_ids))
            pd.DataFrame(results).to_csv(args.output_dir / "QUICK_BASELINES.csv", index=False)
    frame = pd.DataFrame(results)
    summary = frame.groupby(["period", "region", "model"])[
        ["sequence_residual_R2", "lead24_residual_R2", "lead24_residual_RMSE"]
    ].agg(["mean", "std", "count"])
    summary.to_csv(args.output_dir / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
