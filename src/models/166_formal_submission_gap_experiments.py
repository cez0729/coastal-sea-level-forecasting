"""Formal submission-gap experiments for the seven-station residual benchmark.

This script is deliberately conservative.  It uses the same 2023-2024 train,
2025 H1 validation, and 2025 H2 test split as the FS-GWN benchmark.  The
attention and regional-anchor models are compact comparison models; VARX is a
direct ridge-regularized multi-output baseline.  No candidate is selected from
the 2025 H2 test results.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "results" / "formal_submission_gap_20260811"
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p104 = load_module("gap_p104", HERE / "104_priority2_physics_graph_wavenet.py")
final4 = p104.final4
priority1 = p104.priority1
v2 = p104.v2


class CompactSpatiotemporalAttention(nn.Module):
    """A small attention baseline: temporal self-attention then node attention."""

    def __init__(self, input_dim: int, horizon: int, d_model: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=layers)
        self.spatial = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F]. Temporal attention is node-wise; spatial attention
        # mixes the seven station tokens after the temporal encoder.
        batch, steps, nodes, _ = x.shape
        h = self.proj(x).permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        h = self.temporal(h)[:, -1].reshape(batch, nodes, -1)
        spatial, _ = self.spatial(h, h, h, need_weights=False)
        h = self.norm(h + spatial)
        return self.head(h)


class RegionalAnchorDataset(Dataset):
    """Add one deterministic regional-mean node to a frozen dataset."""

    def __init__(self, base):
        self.base = base
        self.indices = base.indices

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, y, tide = self.base[index]
        anchor_x = x.mean(dim=1, keepdim=True)
        anchor_y = y.mean(dim=0, keepdim=True)
        anchor_tide = tide.mean(dim=0, keepdim=True)
        return torch.cat([x, anchor_x], dim=1), torch.cat([y, anchor_y], dim=0), torch.cat([tide, anchor_tide], dim=0)


def set_seed(seed: int, cpu_threads: int) -> None:
    v2.set_seed(seed)
    torch.set_num_threads(max(1, cpu_threads))
    torch.use_deterministic_algorithms(True, warn_only=True)


def loader(dataset, args, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, generator=generator if shuffle else None, num_workers=0)


def training_args(args, out_dir: Path):
    values = vars(args).copy()
    values.update({"print_every": 10, "epoch_checkpoint_every": 1})
    return argparse.Namespace(**values)


def score(y_true: np.ndarray, pred: np.ndarray, tide: np.ndarray) -> dict[str, float]:
    return final4.summarize_single(y_true, pred, tide)


def train_neural(model: nn.Module, train_set, val_set, test_set, data, args, seed: int, model_name: str, anchor: bool = False):
    train_loader = loader(train_set, args, True, seed)
    val_loader = loader(val_set, args, False, seed)
    test_loader = loader(test_set, args, False, seed)
    run_dir = Path(args.output_dir) / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    history, best_val, timing = p104.train_eta_only(model, train_loader, val_loader, args, torch.device(args.device), run_dir)
    pred, true, tide, infer_seconds = p104.priority1.predict_single(model, test_loader, torch.device(args.device)) if hasattr(p104.priority1, "predict_single") else predict_single(model, test_loader, torch.device(args.device))
    if anchor:
        pred = pred[:, :7]
        true = true[:, :7]
        tide = tide[:, :7]
    metrics = score(true, pred, tide)
    row = {"model": model_name, "seed": seed, "best_val_loss": best_val, "training_seconds": timing["training_seconds"], "inference_seconds": infer_seconds, **metrics}
    pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
    np.savez_compressed(run_dir / "predictions.npz", pred_residual=pred, true_residual=true, target_tide=tide)
    history.to_csv(run_dir / "training_log.csv", index=False)
    row["wall_seconds"] = time.perf_counter() - start
    return row


@torch.no_grad()
def predict_single(model, data_loader, device):
    model.eval()
    predictions, truths, tides = [], [], []
    start = time.perf_counter()
    for xb, yb, tb in data_loader:
        predictions.append(model(xb.to(device)).cpu().numpy())
        truths.append(yb.numpy())
        tides.append(tb.numpy())
    return np.concatenate(predictions), np.concatenate(truths), np.concatenate(tides), time.perf_counter() - start


def design_matrix(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Direct VARX: 24-hour residual history plus current and 24-hour-mean
    # standardized exogenous features.  All predictors are available at origin.
    x = dataset.x_scaled
    residual = dataset.residual
    rows, targets, tides = [], [], []
    for t in dataset.indices:
        t = int(t)
        history = x[t - dataset.window:t, :, 0].reshape(-1)
        current = x[t - 1].reshape(-1)
        mean_exog = x[t - dataset.window:t].mean(axis=0).reshape(-1)
        rows.append(np.concatenate([history, current, mean_exog]))
        targets.append(residual[t:t + dataset.horizon].T.reshape(-1))
        tides.append(dataset.tide[t:t + dataset.horizon].T)
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64), np.asarray(tides, dtype=np.float64)


def run_varx(data, args, seed: int):
    train_x, train_y, _ = design_matrix(data["single_train"])
    val_x, val_y, _ = design_matrix(data["single_val"])
    test_x, test_y, test_tide = design_matrix(data["single_test"])
    best_alpha, best_val = None, float("inf")
    for alpha in (0.1, 1.0, 10.0, 100.0):
        candidate = Ridge(alpha=alpha, solver="lsqr", fit_intercept=True)
        candidate.fit(train_x, train_y)
        mse = float(np.mean((candidate.predict(val_x) - val_y) ** 2))
        if mse < best_val:
            best_alpha, best_val = alpha, mse
    start = time.perf_counter()
    model = Ridge(alpha=best_alpha, solver="lsqr", fit_intercept=True).fit(train_x, train_y)
    pred = model.predict(test_x).reshape(-1, 7, data["single_test"].horizon)
    metrics = score(test_y.reshape(-1, 7, data["single_test"].horizon), pred, test_tide)
    row = {"model": "VARX_ridge", "seed": seed, "alpha": best_alpha, "validation_mse": best_val, "training_seconds": time.perf_counter() - start, **metrics}
    out = Path(args.output_dir) / "VARX_ridge" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out / "metrics.csv", index=False)
    np.savez_compressed(out / "predictions.npz", pred_residual=pred, true_residual=test_y.reshape(-1, 7, data["single_test"].horizon), target_tide=test_tide)
    return row


def build_data(args):
    data_args = argparse.Namespace(window=args.window, train_ratio=args.train_ratio, val_ratio=args.val_ratio, train_stride=args.train_stride, physics_forcing_mode="last_input", extreme_quantile=0.90)
    return final4.build_enhanced_data(data_args, args.horizon, add_ode_prior=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--models", nargs="+", choices=["varx", "attention", "regional_anchor"], default=["varx", "attention"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    data = build_data(args)
    rows = []
    for seed in args.seeds:
        set_seed(seed, args.cpu_threads)
        if "varx" in args.models:
            rows.append(run_varx(data, args, seed))
        if "attention" in args.models:
            model = CompactSpatiotemporalAttention(data["feats"], args.horizon, args.hidden_dim, 4, 2, args.dropout).to(args.device)
            rows.append(train_neural(model, data["single_train"], data["single_val"], data["single_test"], data, training_args(args, Path(args.output_dir)), seed, "Compact_ST_Attention"))
        if "regional_anchor" in args.models:
            adj7 = data["graph_priors"]["distance"]
            adj8 = np.zeros((8, 8), dtype=np.float32)
            adj8[:7, :7] = 0.90 * adj7
            adj8[:7, 7] = 0.10
            adj8[7, :7] = 1.0 / 7.0
            adj8[7, 7] = 1e-3
            adj8 = adj8 / np.maximum(adj8.sum(axis=1, keepdims=True), 1e-8)
            anchor_model = priority1.GraphWaveNetForecaster(data["feats"], adj8, args.hidden_dim, args.horizon, 2, 6, args.dropout).to(args.device)
            rows.append(train_neural(anchor_model, RegionalAnchorDataset(data["single_train"]), RegionalAnchorDataset(data["single_val"]), RegionalAnchorDataset(data["single_test"]), data, training_args(args, Path(args.output_dir)), seed, "RegionalAnchor_GWN", anchor=True))
        pd.DataFrame(rows).to_csv(Path(args.output_dir) / "all_runs_partial.csv", index=False)
    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(Path(args.output_dir) / "all_runs.csv", index=False)
    summary = all_runs.groupby("model")[["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "training_seconds"]].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(x) for x in c if x) for c in summary.columns.to_flat_index()]
    summary.to_csv(Path(args.output_dir) / "mean_std.csv", index=False)
    (Path(args.output_dir) / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
