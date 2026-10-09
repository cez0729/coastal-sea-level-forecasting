from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.spatial import cKDTree
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "neuralcora_surge"
EVENT_ROOT = DATA_ROOT / "raw" / "cora" / "events"
DEFAULT_OUTPUT = ROOT / "results" / "neuralcora_surge_event_field_baselines"
UPSTREAM_ROOT = ROOT / "external" / "neuralcora_upstream"

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from src.networks import NeuralCoraCNN, NeuralCoraResNet, PeriodicConv2d, UNet  # noqa: E402


EVENT_SPLITS = {
    "train": ["irene_2011", "sandy_2012", "noreaster_2016_01"],
    "validation": ["isaias_2020"],
    "test": ["henri_2021", "ida_2021"],
}


def set_reproducible(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, cpu_threads))
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class FieldData:
    fields: dict[str, np.ndarray]
    times: dict[str, np.ndarray]
    lat: np.ndarray
    lon: np.ndarray
    sea_mask: np.ndarray
    train_mean: float
    train_std: float
    train_q95: float


def read_event(event_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parts: list[np.ndarray] = []
    times: list[np.ndarray] = []
    reference_lat: np.ndarray | None = None
    reference_lon: np.ndarray | None = None
    for path in sorted(event_dir.glob("*.nc")):
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            lat = ds["lat"].values.astype(np.float32)
            lon = ds["lon"].values.astype(np.float32)
            if reference_lat is None:
                reference_lat, reference_lon = lat, lon
            elif not (np.array_equal(reference_lat, lat) and np.array_equal(reference_lon, lon)):
                raise ValueError(f"CORA node coordinates changed in {path}")
            parts.append(ds["zeta"].transpose("time", "nodes").values.astype(np.float32))
            times.append(ds["time"].values.astype("datetime64[ns]"))
    if not parts or reference_lat is None or reference_lon is None:
        raise FileNotFoundError(f"No CORA NetCDF files found in {event_dir}")
    values = np.concatenate(parts, axis=0)
    event_times = np.concatenate(times)
    order = np.argsort(event_times)
    event_times = event_times[order]
    values = values[order]
    unique = np.concatenate(([True], event_times[1:] != event_times[:-1]))
    return values[unique], event_times[unique], reference_lat, reference_lon


def build_regular_grid_mapping(
    node_lat: np.ndarray,
    node_lon: np.ndarray,
    height: int,
    width: int,
    max_nearest_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = np.linspace(float(node_lat.min()), float(node_lat.max()), height, dtype=np.float32)
    lon = np.linspace(float(node_lon.min()), float(node_lon.max()), width, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_scale = np.cos(np.deg2rad(float(np.mean(lat))))
    tree = cKDTree(np.column_stack([node_lat, node_lon * lon_scale]))
    distances, indices = tree.query(
        np.column_stack([lat_grid.ravel(), lon_grid.ravel() * lon_scale]),
        k=1,
    )
    sea_mask = (distances.reshape(height, width) <= max_nearest_degrees).astype(np.float32)
    if sea_mask.mean() < 0.1:
        raise ValueError("Regular-grid sea mask is too sparse; increase --max-nearest-degrees")
    return lat, lon, indices.reshape(height, width), sea_mask


def build_or_load_field_data(args) -> FieldData:
    cache_path = args.cache
    if cache_path.exists() and not args.rebuild_cache:
        cached = np.load(cache_path, allow_pickle=False)
        fields = {event: cached[f"field__{event}"] for events in EVENT_SPLITS.values() for event in events}
        times = {event: cached[f"time__{event}"] for events in EVENT_SPLITS.values() for event in events}
        return FieldData(
            fields=fields,
            times=times,
            lat=cached["lat"],
            lon=cached["lon"],
            sea_mask=cached["sea_mask"],
            train_mean=float(cached["train_mean"]),
            train_std=float(cached["train_std"]),
            train_q95=float(cached["train_q95"]),
        )

    fields: dict[str, np.ndarray] = {}
    times: dict[str, np.ndarray] = {}
    mapping: np.ndarray | None = None
    sea_mask: np.ndarray | None = None
    grid_lat: np.ndarray | None = None
    grid_lon: np.ndarray | None = None
    reference_nodes: tuple[np.ndarray, np.ndarray] | None = None
    for events in EVENT_SPLITS.values():
        for event in events:
            values, event_times, node_lat, node_lon = read_event(EVENT_ROOT / event)
            if reference_nodes is None:
                reference_nodes = (node_lat, node_lon)
                grid_lat, grid_lon, mapping, sea_mask = build_regular_grid_mapping(
                    node_lat,
                    node_lon,
                    args.grid_height,
                    args.grid_width,
                    args.max_nearest_degrees,
                )
            elif not (np.array_equal(reference_nodes[0], node_lat) and np.array_equal(reference_nodes[1], node_lon)):
                raise ValueError(f"CORA nodes in {event} do not match the reference event")
            assert mapping is not None and sea_mask is not None
            field = values[:, mapping.ravel()].reshape(len(values), args.grid_height, args.grid_width)
            field = np.where(sea_mask[None] > 0, field, np.nan).astype(np.float32)
            fields[event] = field
            times[event] = event_times.astype("datetime64[ns]")

    assert grid_lat is not None and grid_lon is not None and sea_mask is not None
    train_values = np.concatenate([fields[event][:, sea_mask.astype(bool)] for event in EVENT_SPLITS["train"]])
    train_mean = float(np.nanmean(train_values))
    train_std = float(np.nanstd(train_values))
    train_std = train_std if train_std > 1e-6 else 1.0
    train_q95 = float(np.nanquantile(train_values, 0.95))
    payload: dict[str, np.ndarray] = {
        "lat": grid_lat,
        "lon": grid_lon,
        "sea_mask": sea_mask,
        "train_mean": np.asarray(train_mean, dtype=np.float32),
        "train_std": np.asarray(train_std, dtype=np.float32),
        "train_q95": np.asarray(train_q95, dtype=np.float32),
    }
    for event in fields:
        payload[f"field__{event}"] = fields[event]
        payload[f"time__{event}"] = times[event]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return FieldData(fields, times, grid_lat, grid_lon, sea_mask, train_mean, train_std, train_q95)


class EventFieldDataset(Dataset):
    def __init__(self, data: FieldData, split: str, input_hours: int, lead_hours: int, stride: int) -> None:
        self.data = data
        self.events = EVENT_SPLITS[split]
        self.input_hours = input_hours
        self.lead_hours = lead_hours
        self.samples: list[tuple[int, int]] = []
        for event_index, event in enumerate(self.events):
            count = len(data.fields[event]) - input_hours - lead_hours + 1
            self.samples.extend((event_index, start) for start in range(0, max(count, 0), stride))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        event_index, start = self.samples[index]
        event = self.events[event_index]
        target_index = start + self.input_hours + self.lead_hours - 1
        x = (self.data.fields[event][start : start + self.input_hours] - self.data.train_mean) / self.data.train_std
        y = (self.data.fields[event][target_index] - self.data.train_mean) / self.data.train_std
        x = np.nan_to_num(x, nan=0.0).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0).astype(np.float32)[None]
        mask = self.data.sea_mask.astype(np.float32)[None]
        target_time = self.data.times[event][target_index].astype("datetime64[ns]").astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), event_index, target_time


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = PeriodicConv2d(input_channels + hidden_channels, 4 * hidden_channels, 3)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        input_gate, forget_gate, output_gate, candidate = self.gates(torch.cat([x, hidden], dim=1)).chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class ConvLSTMForecaster(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(1, hidden_channels)
        self.head = PeriodicConv2d(hidden_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        hidden = x.new_zeros(batch, self.hidden_channels, height, width)
        cell = x.new_zeros(batch, self.hidden_channels, height, width)
        for step in range(x.shape[1]):
            hidden, cell = self.cell(x[:, step : step + 1], hidden, cell)
        return self.head(hidden)


def masked_latitude_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latitude_weights: torch.Tensor,
) -> torch.Tensor:
    weights = mask * latitude_weights
    return (((prediction - target) ** 2) * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, latitude_weights: torch.Tensor) -> float:
    model.eval()
    squared = 0.0
    weight_sum = 0.0
    for x, target, mask, _, _ in loader:
        prediction = model(x.to(device))
        weights = mask.to(device) * latitude_weights
        squared += float((((prediction - target.to(device)) ** 2) * weights).sum().cpu())
        weight_sum += float(weights.sum().cpu())
    return squared / max(weight_sum, 1.0)


def train_model(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, args, device, lat_weights):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    bad_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, target, mask, _, _ in train_loader:
            x, target, mask = x.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = masked_latitude_mse(prediction, target, mask, lat_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_loss = evaluate_loss(model, validation_loader, device, lat_weights)
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_masked_lat_mse": float(np.mean(losses)),
                "validation_masked_lat_mse": validation_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(f"epoch={epoch:03d} train={history[-1]['train_masked_lat_mse']:.6f} validation={validation_loss:.6f}")
        if validation_loss < best_validation - args.min_delta:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(history), best_validation


@torch.no_grad()
def collect_predictions(model: nn.Module | None, loader: DataLoader, device: torch.device, baseline: str | None = None):
    predictions, targets, masks, event_indices, times = [], [], [], [], []
    if model is not None:
        model.eval()
    for x, target, mask, event_index, target_time in loader:
        if baseline == "persistence":
            prediction = x[:, -1:, :, :]
        elif baseline == "climatology":
            prediction = torch.zeros_like(target)
        else:
            assert model is not None
            prediction = model(x.to(device)).cpu()
        predictions.append(prediction.numpy())
        targets.append(target.numpy())
        masks.append(mask.numpy())
        event_indices.append(event_index.numpy())
        times.append(target_time.numpy())
    return tuple(np.concatenate(items, axis=0) for items in (predictions, targets, masks, event_indices, times))


def metric_row(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float]:
    valid = np.broadcast_to(mask.astype(bool), target.shape) & np.isfinite(target) & np.isfinite(prediction)
    observed = target[valid]
    predicted = prediction[valid]
    error = predicted - observed
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    truth_event = observed >= threshold
    predicted_event = predicted >= threshold
    tp = int(np.sum(truth_event & predicted_event))
    fp = int(np.sum(~truth_event & predicted_event))
    fn = int(np.sum(truth_event & ~predicted_event))
    sample_mask = mask[:, 0].astype(bool)
    target_peaks = np.asarray([row[m].max() for row, m in zip(target[:, 0], sample_mask)])
    prediction_peaks = np.asarray([row[m].max() for row, m in zip(prediction[:, 0], sample_mask)])
    return {
        "count": int(valid.sum()),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "mae_m": float(np.mean(np.abs(error))),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
        "event_precision": tp / max(tp + fp, 1),
        "event_recall": tp / max(tp + fn, 1),
        "event_csi": tp / max(tp + fp + fn, 1),
        "field_peak_mae_m": float(np.mean(np.abs(prediction_peaks - target_peaks))),
    }


def evaluate_predictions(model_name: str, values, dataset: EventFieldDataset, data: FieldData, output_dir: Path):
    prediction_scaled, target_scaled, masks, event_indices, target_times = values
    prediction = prediction_scaled * data.train_std + data.train_mean
    target = target_scaled * data.train_std + data.train_mean
    summary = {"model": model_name, **metric_row(target, prediction, masks, data.train_q95)}
    rows = []
    for event_index, event in enumerate(dataset.events):
        selected = event_indices == event_index
        rows.append({"model": model_name, "event": event, **metric_row(target[selected], prediction[selected], masks[selected], data.train_q95)})
    pd.DataFrame(rows).to_csv(output_dir / f"{model_name}_per_event.csv", index=False)
    np.savez_compressed(
        output_dir / f"{model_name}_predictions.npz",
        prediction=prediction.astype(np.float32),
        target=target.astype(np.float32),
        mask=masks.astype(np.uint8),
        event_indices=event_indices,
        target_times=target_times,
        events=np.asarray(dataset.events),
        lat=data.lat,
        lon=data.lon,
    )
    return summary


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralCORA-Surge event field baselines")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DATA_ROOT / "processed" / "cora_event_grid_32x48.npz")
    parser.add_argument("--models", nargs="+", choices=["climatology", "persistence", "cnn", "resnet", "unet", "convlstm"], default=["climatology", "persistence", "cnn", "resnet", "unet", "convlstm"])
    parser.add_argument("--input-hours", type=int, default=24)
    parser.add_argument("--lead-hours", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--grid-height", type=int, default=32)
    parser.add_argument("--grid-width", type=int, default=48)
    parser.add_argument("--max-nearest-degrees", type=float, default=0.04)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--convlstm-hidden", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.stride = max(args.stride, 8)

    set_reproducible(args.seed, args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_or_load_field_data(args)
    train_dataset = EventFieldDataset(data, "train", args.input_hours, args.lead_hours, args.stride)
    validation_dataset = EventFieldDataset(data, "validation", args.input_hours, args.lead_hours, args.stride)
    test_dataset = EventFieldDataset(data, "test", args.input_hours, args.lead_hours, args.stride)
    if min(len(train_dataset), len(validation_dataset), len(test_dataset)) == 0:
        raise ValueError("At least one split has no valid windows")
    print(
        f"samples train={len(train_dataset)} validation={len(validation_dataset)} test={len(test_dataset)} "
        f"sea_fraction={data.sea_mask.mean():.3f} mean={data.train_mean:.4f} std={data.train_std:.4f} q95={data.train_q95:.4f}"
    )
    train_loader = make_loader(train_dataset, args.batch_size, True, args.seed)
    validation_loader = make_loader(validation_dataset, args.batch_size, False, args.seed)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lat_weights = torch.from_numpy(np.cos(np.deg2rad(data.lat)).astype(np.float32))[None, None, :, None].to(device)
    summaries: list[dict[str, float]] = []

    for baseline in [name for name in args.models if name in {"climatology", "persistence"}]:
        values = collect_predictions(None, test_loader, device, baseline)
        summaries.append(evaluate_predictions(baseline, values, test_dataset, data, args.output_dir))

    factories = {
        "cnn": lambda: NeuralCoraCNN(args.input_hours, [16, 32, 32, 16], 3, 1, dropout=args.dropout),
        "resnet": lambda: NeuralCoraResNet(args.input_hours, [16, 16, 16, 16, 1], [3, 3, 3, 3, 3], bn_position="post", dropout=args.dropout, long_skip=True),
        "unet": lambda: UNet(args.input_hours, 3, 8, 1, bn_position="post", dropout=args.dropout),
        "convlstm": lambda: ConvLSTMForecaster(args.convlstm_hidden),
    }
    for model_name in [name for name in args.models if name in factories]:
        set_reproducible(args.seed, args.cpu_threads)
        model = factories[model_name]().to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        start = time.perf_counter()
        history, best_validation = train_model(model, train_loader, validation_loader, args, device, lat_weights)
        elapsed = time.perf_counter() - start
        history.to_csv(args.output_dir / f"{model_name}_training_log.csv", index=False)
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "train_mean": data.train_mean, "train_std": data.train_std}, args.output_dir / f"{model_name}_checkpoint.pt")
        values = collect_predictions(model, test_loader, device)
        summary = evaluate_predictions(model_name, values, test_dataset, data, args.output_dir)
        summary.update({"parameters": parameter_count, "best_validation_masked_lat_mse": best_validation, "training_seconds": elapsed})
        summaries.append(summary)

    summary_path = args.output_dir / "model_summary.csv"
    new_summary = pd.DataFrame(summaries)
    if summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        existing_summary = existing_summary[~existing_summary["model"].isin(new_summary["model"])]
        new_summary = pd.concat([existing_summary, new_summary], ignore_index=True)
    new_summary.to_csv(summary_path, index=False)
    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "event_splits": EVENT_SPLITS,
            "samples": {"train": len(train_dataset), "validation": len(validation_dataset), "test": len(test_dataset)},
            "sea_fraction": float(data.sea_mask.mean()),
            "train_mean": data.train_mean,
            "train_std": data.train_std,
            "train_q95": data.train_q95,
            "scientific_status": "preliminary event-field benchmark; not a substitute for the full 1979-2022 NeuralCORA field benchmark",
            "models_completed_in_output": new_summary["model"].tolist(),
        }
    )
    (args.output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(new_summary.to_string(index=False))


if __name__ == "__main__":
    main()
