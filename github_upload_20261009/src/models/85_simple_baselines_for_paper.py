from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT78 = Path(__file__).resolve().parent / "78_final_four_models_enhanced_data.py"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "simple_baselines_for_paper"

spec78 = importlib.util.spec_from_file_location("final_impl", SCRIPT78)
final = importlib.util.module_from_spec(spec78)
assert spec78.loader is not None
spec78.loader.exec_module(final)
v2 = final.v2
v3 = final.v3


class BiGRUOnly(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, nodes, features]. Treat each node as an independent time series.
        b, t, n, f = x.shape
        h = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        out, _ = self.gru(h)
        pred = self.head(out[:, -1, :])
        return pred.reshape(b, n, -1)


class FlattenedStationDataset(Dataset):
    def __init__(self, single_dataset: Dataset):
        self.single_dataset = single_dataset
        self.n_nodes = single_dataset.residual.shape[1]

    def __len__(self) -> int:
        return len(self.single_dataset) * self.n_nodes

    def __getitem__(self, idx: int):
        sample_idx = idx // self.n_nodes
        node_idx = idx % self.n_nodes
        xb, yb, tb = self.single_dataset[sample_idx]
        return xb[:, node_idx, :], yb[node_idx, :], tb[node_idx, :]


def summarize_residual_and_level(true_residual: np.ndarray, pred_residual: np.ndarray, tide: np.ndarray) -> dict[str, float]:
    pred_level = pred_residual + tide
    true_level = true_residual + tide
    out: dict[str, float] = {}
    pairs = [
        ("seq_residual", true_residual, pred_residual),
        ("last_residual", true_residual[:, :, -1], pred_residual[:, :, -1]),
        ("seq_sea_level", true_level, pred_level),
        ("last_sea_level", true_level[:, :, -1], pred_level[:, :, -1]),
    ]
    for prefix, yt, yp in pairs:
        for k, val in v2.regression_metrics(yt, yp).items():
            out[f"{prefix}_{k}"] = val
    true_states = np.zeros((*true_residual.shape, 1), dtype=np.float32)
    pred_states = np.zeros((*pred_residual.shape, 1), dtype=np.float32)
    true_states[..., 0] = true_residual
    pred_states[..., 0] = pred_residual
    out.update(v3.summarize_extreme_metrics(true_states, pred_states))
    return out


def collect_single_dataset_arrays(dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, tides = [], [], []
    for i in range(len(dataset)):
        xb, yb, tb = dataset[i]
        xs.append(xb.numpy())
        ys.append(yb.numpy())
        tides.append(tb.numpy())
    return np.stack(xs), np.stack(ys), np.stack(tides)


def run_persistence(data: dict, horizon: int) -> dict[str, float]:
    _, true, tide = collect_single_dataset_arrays(data["single_test"])
    ds = data["single_test"]
    last_observed = []
    for idx in range(len(ds)):
        t = int(ds.indices[idx])
        last_observed.append(ds.residual[t - 1])
    last_observed_arr = np.stack(last_observed).astype(np.float32)
    pred = np.repeat(last_observed_arr[:, :, None], horizon, axis=2)
    row = {
        "model_key": "persistence",
        "model_name": "Persistence baseline",
        "horizon": horizon,
        "num_features": data["feats"],
        "training_mode": "no_training_last_observed_residual",
        **summarize_residual_and_level(true, pred, tide),
    }
    return row


def run_tide_only(data: dict, horizon: int) -> dict[str, float]:
    _, true, tide = collect_single_dataset_arrays(data["single_test"])
    pred = np.zeros_like(true, dtype=np.float32)
    row = {
        "model_key": "tide_only_total_water",
        "model_name": "Tide-only total-water baseline",
        "horizon": horizon,
        "num_features": 1,
        "training_mode": "no_training_residual_zero",
        **summarize_residual_and_level(true, pred, tide),
    }
    return row


def train_bigru_only(args, data: dict, horizon: int, device: torch.device) -> tuple[dict[str, float], pd.DataFrame]:
    train_loader = DataLoader(data["single_train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data["single_val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(data["single_test"], batch_size=args.batch_size, shuffle=False)
    model = BiGRUOnly(data["feats"], args.gru_hidden, horizon, args.dropout).to(device)
    history, best_val = final.train_single_model(model, train_loader, val_loader, args, device)
    pred, true, tide = final.predict_single(model, test_loader, device)
    row = {
        "model_key": "bigru_only",
        "model_name": "BiGRU-only baseline",
        "horizon": horizon,
        "best_val_loss": best_val,
        "num_features": data["feats"],
        "training_mode": f"train_stride{args.train_stride}_no_gnn",
        **summarize_residual_and_level(true, pred, tide),
    }
    return row, history


def make_rf_xy(dataset: Dataset, max_samples: int | None = None, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    flat = FlattenedStationDataset(dataset)
    indices = np.arange(len(flat))
    if max_samples is not None and len(indices) > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_samples, replace=False)
        indices.sort()
    xs, ys = [], []
    for idx in indices:
        xb, yb, _ = flat[int(idx)]
        xs.append(xb.numpy().reshape(-1))
        ys.append(yb.numpy())
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def make_rf_test(dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    x_win, true, tide = collect_single_dataset_arrays(dataset)
    samples, window, nodes, feats = x_win.shape
    x_flat = x_win.transpose(0, 2, 1, 3).reshape(samples * nodes, window * feats).astype(np.float32)
    return x_flat, true, tide, samples, nodes


def run_random_forest(args, data: dict, horizon: int) -> dict[str, float]:
    x_train, y_train = make_rf_xy(data["single_train"], args.rf_max_train_samples, args.seed)
    x_val, y_val = make_rf_xy(data["single_val"], args.rf_max_val_samples, args.seed)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    model = RandomForestRegressor(
        n_estimators=args.rf_trees,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        n_jobs=args.rf_n_jobs,
        random_state=args.seed,
    )
    model.fit(np.concatenate([x_train_s, x_val_s], axis=0), np.concatenate([y_train, y_val], axis=0))
    x_test, true, tide, samples, nodes = make_rf_test(data["single_test"])
    pred_flat = model.predict(scaler.transform(x_test)).astype(np.float32)
    pred = pred_flat.reshape(samples, nodes, horizon)
    row = {
        "model_key": "random_forest",
        "model_name": "Random Forest baseline",
        "horizon": horizon,
        "num_features": data["feats"],
        "training_mode": f"station_flattened_rf_trees{args.rf_trees}",
        **summarize_residual_and_level(true, pred, tide),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple paper baselines for sea-level residual forecasting")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--horizons", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--baselines", nargs="+", default=["persistence", "tide_only", "bigru_only", "random_forest"])
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", default="distance", choices=["identity", "distance", "corr"])
    parser.add_argument("--gnn-hidden", type=int, default=40)
    parser.add_argument("--gru-hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=9)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--graph-lr-mult", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--physics-forcing-mode", default="last_input", choices=["last_input", "future"])
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--rf-trees", type=int, default=40)
    parser.add_argument("--rf-max-depth", type=int, default=18)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=3)
    parser.add_argument("--rf-n-jobs", type=int, default=1)
    parser.add_argument("--rf-max-train-samples", type=int, default=20000)
    parser.add_argument("--rf-max-val-samples", type=int, default=8000)
    parser.add_argument("--resume", action="store_true", help="Resume from simple_baselines_metrics_partial.csv when present.")
    args = parser.parse_args()

    v2.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partial_path = output_dir / "simple_baselines_metrics_partial.csv"
    if args.resume and partial_path.exists():
        rows = pd.read_csv(partial_path).to_dict("records")
        done = {(int(r["horizon"]), str(r["model_key"])) for r in rows}
        print(f"Resuming from {partial_path}, existing rows={len(rows)}")
    else:
        rows = []
        done = set()
    for horizon in args.horizons:
        print("\n" + "=" * 90)
        print(f"Preparing baseline data for horizon={horizon}h")
        data = final.build_enhanced_data(args, horizon, add_ode_prior=False)
        if "persistence" in args.baselines and (horizon, "persistence") not in done:
            row = run_persistence(data, horizon)
            rows.append(row)
            done.add((horizon, "persistence"))
            pd.DataFrame(rows).to_csv(output_dir / "simple_baselines_metrics_partial.csv", index=False)
            print(f"Persistence {horizon}h: last_R2={row['last_residual_R2']:.4f}")
        if "tide_only" in args.baselines and (horizon, "tide_only_total_water") not in done:
            row = run_tide_only(data, horizon)
            rows.append(row)
            done.add((horizon, "tide_only_total_water"))
            pd.DataFrame(rows).to_csv(output_dir / "simple_baselines_metrics_partial.csv", index=False)
            print(f"Tide-only {horizon}h: last_residual_R2={row['last_residual_R2']:.4f}, last_level_R2={row['last_sea_level_R2']:.4f}")
        if "bigru_only" in args.baselines and (horizon, "bigru_only") not in done:
            print(f"Training BiGRU-only baseline horizon={horizon}h")
            row, history = train_bigru_only(args, data, horizon, device)
            rows.append(row)
            done.add((horizon, "bigru_only"))
            run_dir = output_dir / f"horizon_{horizon}h" / "bigru_only"
            run_dir.mkdir(parents=True, exist_ok=True)
            history.to_csv(run_dir / "training_log.csv", index=False)
            pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
            pd.DataFrame(rows).to_csv(output_dir / "simple_baselines_metrics_partial.csv", index=False)
            print(f"BiGRU-only {horizon}h: last_R2={row['last_residual_R2']:.4f}")
        if "random_forest" in args.baselines and (horizon, "random_forest") not in done:
            print(f"Training Random Forest baseline horizon={horizon}h")
            row = run_random_forest(args, data, horizon)
            rows.append(row)
            done.add((horizon, "random_forest"))
            run_dir = output_dir / f"horizon_{horizon}h" / "random_forest"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
            pd.DataFrame(rows).to_csv(output_dir / "simple_baselines_metrics_partial.csv", index=False)
            print(f"Random Forest {horizon}h: last_R2={row['last_residual_R2']:.4f}")
        pd.DataFrame(rows).to_csv(output_dir / "simple_baselines_metrics_partial.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "simple_baselines_metrics.csv", index=False)
    key_cols = [
        "horizon",
        "model_name",
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "seq_sea_level_R2",
        "last_sea_level_R2",
    ]
    existing = [c for c in key_cols if c in summary.columns]
    key = summary[existing].sort_values(["horizon", "last_residual_R2"], ascending=[True, False])
    key.to_csv(output_dir / "simple_baselines_key_metrics.csv", index=False)
    print("\nFinished simple baselines.")
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
