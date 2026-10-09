"""Zero-target spatial transfer audit on two additional NOAA stations.

The model receives causal 24-hour residual histories at nine nodes, but the
loss and validation score use only the original seven stations.  The two new
stations are therefore unseen as prediction targets during fitting.  This is a
supplementary node-transfer diagnostic, not a replacement for the 34-feature
main benchmark.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "results" / "external_node_transfer_20260811"
ORIGINAL_DIR = ROOT / "data" / "processed_multiyear_2023_2025"
EXTERNAL_FILE = ROOT / "data" / "external_nodes_noaa_2023_2025" / "processed" / "external_nodes_water_tide_residual_2023_2025.csv.gz"
ORIGINAL_STATIONS = ["8461490", "8510560", "8516945", "8518750", "8531680", "8534720", "8536110"]
EXTERNAL_STATIONS = ["8537121", "8551762"]
COORDINATES = {
    "8461490": (41.371666, -72.09556), "8510560": (41.048332, -71.95944),
    "8516945": (40.8103, -73.7649), "8518750": (40.700554, -74.01417),
    "8531680": (40.4669, -74.0094), "8534720": (39.356667, -74.41805),
    "8536110": (38.9683, -74.96), "8537121": (39.30539, -75.37668),
    "8551762": (39.582195, -75.588974),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


priority1 = load_module("external_priority1", HERE / "101_priority1_publication_experiments.py")


class TransferDataset(Dataset):
    def __init__(self, x_scaled, target, window, horizon, start, end, stride):
        self.x_scaled = x_scaled
        self.target = target
        self.window = int(window)
        self.horizon = int(horizon)
        self.indices = np.arange(start + window, end - horizon + 1, stride, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        t = int(self.indices[index])
        x = self.x_scaled[t - self.window:t, :, None]
        y = self.target[t:t + self.horizon].T
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


def load_matrix():
    original = pd.read_csv(ORIGINAL_DIR / "residual_matrix.csv", parse_dates=["datetime"]).set_index("datetime")
    original.columns = [str(c) for c in original.columns]
    external = pd.read_csv(EXTERNAL_FILE, parse_dates=["time"])
    external["station_id"] = external["station_id"].astype(str)
    external["time"] = pd.to_datetime(external["time"], utc=True).dt.tz_localize(None)
    external = external.pivot(index="time", columns="station_id", values="residual_m")
    index = pd.date_range("2023-01-01", "2025-12-31 23:00:00", freq="h")
    frame = original.reindex(index)[ORIGINAL_STATIONS].join(external.reindex(index)[EXTERNAL_STATIONS])
    missing = frame.isna().mean()
    if float(missing.max()) > 0.01:
        raise RuntimeError(f"Unexpected external-node missingness: {missing.to_dict()}")
    frame = frame.interpolate(limit=6).ffill().bfill()
    return frame


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    value = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(value))


def adjacency(stations):
    n = len(stations)
    distances = np.zeros((n, n), dtype=np.float32)
    for i, left in enumerate(stations):
        for j, right in enumerate(stations):
            distances[i, j] = haversine(*COORDINATES[left], *COORDINATES[right])
    sigma = float(np.median(distances[distances > 0]))
    graph = np.exp(-distances / sigma).astype(np.float32)
    np.fill_diagonal(graph, 0.0)
    return graph / np.maximum(graph.sum(axis=1, keepdims=True), 1e-8)


def set_seed(seed, threads):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads))
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    losses = []
    for x, y in loader:
        pred = model(x.to(device))[:, :7]
        losses.append(float(torch.mean((pred - y.to(device)[:, :7]) ** 2).cpu()))
    return float(np.mean(losses))


def train(model, train_loader, val_loader, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    best, best_val, bad, history = None, float("inf"), 0, []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x.to(device))[:, :7]
            loss = torch.mean((pred - y.to(device)[:, :7]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        value = validation_loss(model, val_loader, device)
        scheduler.step(value)
        history.append({"epoch": epoch, "train_loss": float(np.mean(batch_losses)), "val_original7_loss": value})
        if value < best_val - 1e-5:
            best_val, best, bad = value, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    if best is not None:
        model.load_state_dict(best)
    return pd.DataFrame(history), best_val, time.perf_counter() - start


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, truth = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        truth.append(y.numpy())
    return np.concatenate(preds), np.concatenate(truth)


def metrics(true, pred):
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    mse = float(np.mean((true - pred) ** 2))
    return {"R2": float(1 - np.sum((true - pred) ** 2) / max(denominator, 1e-12)), "RMSE": float(np.sqrt(mse)), "MAE": float(np.mean(np.abs(true - pred)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 2025, 3407])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_matrix()
    values = frame.to_numpy(dtype=np.float32)
    train_end, val_end = int(len(frame) * 0.70), int(len(frame) * 0.85)
    mean = float(np.mean(values[:train_end, :7]))
    std = float(np.std(values[:train_end, :7]) + 1e-6)
    x_scaled = (values - mean) / std
    train_set = TransferDataset(x_scaled, values, 24, 24, 0, train_end, 8)
    val_set = TransferDataset(x_scaled, values, 24, 24, train_end, val_end, 1)
    test_set = TransferDataset(x_scaled, values, 24, 24, val_end, len(frame), 1)
    stations = ORIGINAL_STATIONS + EXTERNAL_STATIONS
    graph = adjacency(stations)
    rows, station_rows = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        set_seed(seed, args.cpu_threads)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_set, args.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, args.batch_size, shuffle=False)
        model = priority1.GraphWaveNetForecaster(1, graph, 32, 24, 2, 4, 0.15).to(device)
        history, best_val, seconds = train(model, train_loader, val_loader, args, device)
        pred, true = predict(model, test_loader, device)
        persistence = np.repeat(values[test_set.indices - 1, :, None], 24, axis=2)
        row = {"seed": seed, "best_val_original7_mse": best_val, "training_seconds": seconds}
        for group, indices in (("original7", slice(0, 7)), ("external2", slice(7, 9))):
            row.update({f"{group}_{k}": v for k, v in metrics(true[:, indices], pred[:, indices]).items()})
            row.update({f"persistence_{group}_{k}": v for k, v in metrics(true[:, indices], persistence[:, indices]).items()})
        rows.append(row)
        for station_index, station in enumerate(stations):
            station_rows.append({"seed": seed, "station_id": station, "target_seen_in_training": station_index < 7, **metrics(true[:, station_index], pred[:, station_index]), **{f"persistence_{k}": v for k, v in metrics(true[:, station_index], persistence[:, station_index]).items()}})
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        history.to_csv(seed_dir / "training_log.csv", index=False)
        np.savez_compressed(seed_dir / "predictions.npz", pred=pred, true=true, persistence=persistence, station_ids=np.asarray(stations))
        pd.DataFrame(rows).to_csv(out / "all_runs_partial.csv", index=False)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "all_runs.csv", index=False)
    per_station = pd.DataFrame(station_rows)
    per_station.to_csv(out / "per_station.csv", index=False)
    summary = runs.drop(columns=["seed"]).agg(["mean", "std"]).T.reset_index(names="metric")
    summary.to_csv(out / "mean_std.csv", index=False)
    (out / "experiment_config.json").write_text(json.dumps({**vars(args), "stations": stations, "external_target_loss_weight": 0.0, "scaler_fit_nodes": ORIGINAL_STATIONS}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
