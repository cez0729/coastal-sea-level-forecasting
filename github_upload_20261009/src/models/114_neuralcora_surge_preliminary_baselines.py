from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "neuralcora_surge"
DEFAULT_INPUT = DATA_ROOT / "processed" / "residual_matrix_1999_2022.csv.gz"
DEFAULT_OUTPUT = ROOT / "results" / "neuralcora_surge_preliminary"

STATION_COORDS = {
    "New London": (41.371666, -72.09556),
    "Montauk": (41.048332, -71.95944),
    "Kings Point": (40.8103, -73.7649),
    "The Battery": (40.700554, -74.01417),
    "Sandy Hook": (40.4669, -74.0094),
    "Atlantic City": (39.356667, -74.41805),
    "Cape May": (38.9683, -74.96),
}

SPLITS = {
    "train": (pd.Timestamp("1999-01-01T00:00:00Z"), pd.Timestamp("2016-12-31T23:00:00Z")),
    "validation": (pd.Timestamp("2017-01-01T00:00:00Z"), pd.Timestamp("2019-12-31T23:00:00Z")),
    "test": (pd.Timestamp("2020-01-01T00:00:00Z"), pd.Timestamp("2022-12-31T23:00:00Z")),
}


def set_reproducible(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, cpu_threads))
    torch.use_deterministic_algorithms(True, warn_only=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def build_distance_adjacency(station_names: list[str], neighbors: int = 3) -> np.ndarray:
    count = len(station_names)
    distances = np.zeros((count, count), dtype=np.float32)
    for i, left in enumerate(station_names):
        for j, right in enumerate(station_names):
            distances[i, j] = haversine_km(*STATION_COORDS[left], *STATION_COORDS[right])
    positive = distances[distances > 0]
    scale = float(np.median(positive))
    weights = np.exp(-((distances / max(scale, 1e-6)) ** 2)).astype(np.float32)
    adjacency = np.zeros_like(weights)
    for row in range(count):
        selected = np.argsort(distances[row])[: neighbors + 1]
        adjacency[row, selected] = weights[row, selected]
    adjacency = np.maximum(adjacency, adjacency.T)
    adjacency += np.eye(count, dtype=np.float32)
    adjacency /= np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-6)
    return adjacency


@dataclass
class PreparedData:
    times: pd.DatetimeIndex
    station_names: list[str]
    raw_targets: np.ndarray
    target_mask: np.ndarray
    features: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    train_q95: np.ndarray
    adjacency: np.ndarray


def prepare_data(path: Path) -> PreparedData:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    station_names = list(frame.columns)
    unknown = sorted(set(station_names) - set(STATION_COORDS))
    if unknown:
        raise ValueError(f"Missing station coordinates: {unknown}")

    train_frame = frame.loc[SPLITS["train"][0] : SPLITS["train"][1]]
    mean = train_frame.mean(skipna=True).to_numpy(dtype=np.float32)
    std = train_frame.std(skipna=True).to_numpy(dtype=np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    train_q95 = train_frame.quantile(0.95).to_numpy(dtype=np.float32)

    mask = frame.notna().to_numpy(dtype=np.float32)
    # Forward fill is causal. Initial gaps use training means rather than future observations.
    filled = frame.ffill().fillna(pd.Series(mean, index=station_names))
    scaled = ((filled.to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
    raw_targets = frame.to_numpy(dtype=np.float32)

    hours = frame.index.hour.to_numpy(dtype=np.float32)
    day_of_year = frame.index.dayofyear.to_numpy(dtype=np.float32)
    temporal = np.stack(
        [
            np.sin(2 * np.pi * hours / 24.0),
            np.cos(2 * np.pi * hours / 24.0),
            np.sin(2 * np.pi * day_of_year / 365.25),
            np.cos(2 * np.pi * day_of_year / 365.25),
        ],
        axis=-1,
    ).astype(np.float32)
    temporal = np.repeat(temporal[:, None, :], len(station_names), axis=1)
    features = np.concatenate([scaled[:, :, None], mask[:, :, None], temporal], axis=-1)
    return PreparedData(
        times=frame.index,
        station_names=station_names,
        raw_targets=raw_targets,
        target_mask=mask,
        features=features,
        mean=mean,
        std=std,
        train_q95=train_q95,
        adjacency=build_distance_adjacency(station_names),
    )


class WindowDataset(Dataset):
    def __init__(
        self,
        data: PreparedData,
        split: str,
        input_hours: int,
        horizon: int,
        stride: int,
    ) -> None:
        self.data = data
        self.input_hours = input_hours
        self.horizon = horizon
        split_start, split_end = SPLITS[split]
        samples: list[int] = []
        for input_start in range(0, len(data.times) - input_hours - horizon + 1, stride):
            target_start = input_start + input_hours
            target_end = target_start + horizon - 1
            if data.times[target_start] >= split_start and data.times[target_end] <= split_end:
                samples.append(input_start)
        self.samples = np.asarray(samples, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        input_start = int(self.samples[index])
        target_start = input_start + self.input_hours
        target_end = target_start + self.horizon
        x = self.data.features[input_start:target_start]
        y_raw = self.data.raw_targets[target_start:target_end]
        mask = self.data.target_mask[target_start:target_end]
        y_filled = np.where(np.isfinite(y_raw), y_raw, self.data.mean[None, :])
        y_scaled = (y_filled - self.data.mean) / self.data.std
        target_times = self.data.times[target_start:target_end].view("int64")
        return (
            torch.from_numpy(x),
            torch.from_numpy(y_scaled.astype(np.float32).T),
            torch.from_numpy(mask.astype(np.float32).T),
            torch.from_numpy(np.asarray(target_times, dtype=np.int64)),
        )


class NodeGRUForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, features = x.shape
        node_series = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, features)
        _, hidden = self.gru(node_series)
        prediction = self.head(hidden[-1])
        return prediction.reshape(batch, nodes, -1)


def make_diffusion_supports(adjacency: np.ndarray, diffusion_steps: int) -> list[np.ndarray]:
    supports = [np.eye(adjacency.shape[0], dtype=np.float32)]
    power = adjacency.astype(np.float32)
    for _ in range(max(1, diffusion_steps)):
        supports.append(power)
        power = power @ adjacency
    return supports


class DiffusionGraphLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, supports: list[np.ndarray]):
        super().__init__()
        self.register_buffer("supports", torch.from_numpy(np.stack(supports)))
        self.linear = nn.Linear(in_dim * len(supports), out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("kij,bjf->bkif", self.supports, x)
        propagated = propagated.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.linear(propagated)


class GraphWaveNetBlock(nn.Module):
    def __init__(self, channels: int, supports: list[np.ndarray], dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.filter_conv = nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation), padding=(0, padding))
        self.gate_conv = nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation), padding=(0, padding))
        self.graph = DiffusionGraphLinear(channels, channels, supports)
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        filt = torch.tanh(self.filter_conv(x)[..., : x.shape[-1]])
        gate = torch.sigmoid(self.gate_conv(x)[..., : x.shape[-1]])
        hidden = filt * gate
        batch, channels, nodes, steps = hidden.shape
        node_hidden = hidden.permute(0, 3, 2, 1).reshape(batch * steps, nodes, channels)
        node_hidden = self.graph(node_hidden).reshape(batch, steps, nodes, channels).permute(0, 3, 2, 1)
        return self.norm(residual + self.dropout(node_hidden))


class GraphWaveNetForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        adjacency: np.ndarray,
        blocks: int,
        diffusion_steps: int,
        dropout: float,
    ) -> None:
        super().__init__()
        supports = make_diffusion_supports(adjacency, diffusion_steps)
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, (1, 1))
        self.blocks = nn.ModuleList(
            [GraphWaveNetBlock(hidden_dim, supports, 2 ** (index % 4), dropout) for index in range(blocks)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, horizon, (1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x.permute(0, 3, 2, 1))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)[..., -1].permute(0, 2, 1)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared = (prediction - target) ** 2 * mask
    return squared.sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def loader_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_squared = 0.0
    total_count = 0.0
    for x, target, mask, _ in loader:
        prediction = model(x.to(device))
        mask_device = mask.to(device)
        total_squared += float((((prediction - target.to(device)) ** 2) * mask_device).sum().cpu())
        total_count += float(mask_device.sum().cpu())
    return total_squared / max(total_count, 1.0)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args,
    device: torch.device,
) -> tuple[pd.DataFrame, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    bad_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for x, target, mask, _ in train_loader:
            x, target, mask = x.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = masked_mse(prediction, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        validation_loss = loader_loss(model, validation_loader, device)
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_masked_mse": float(np.mean(batch_losses)),
                "validation_masked_mse": validation_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_validation - args.min_delta:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"epoch={epoch:03d} train={history[-1]['train_masked_mse']:.6f} "
            f"validation={validation_loss:.6f}"
        )
        if bad_epochs >= args.patience:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(history), best_validation


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    predictions, targets, masks, times = [], [], [], []
    for x, target, mask, target_times in loader:
        predictions.append(model(x.to(device)).cpu().numpy())
        targets.append(target.numpy())
        masks.append(mask.numpy())
        times.append(target_times.numpy())
    return tuple(np.concatenate(items, axis=0) for items in (predictions, targets, masks, times))


def persistence_predictions(dataset: WindowDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions, targets, masks, times = [], [], [], []
    for index in range(len(dataset)):
        x, target, mask, target_times = dataset[index]
        last_scaled = x[-1, :, 0].numpy()
        predictions.append(np.repeat(last_scaled[:, None], dataset.horizon, axis=1))
        targets.append(target.numpy())
        masks.append(mask.numpy())
        times.append(target_times.numpy())
    return tuple(np.stack(items, axis=0) for items in (predictions, targets, masks, times))


def inverse_scale(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return values * std[None, :, None] + mean[None, :, None]


def masked_r2(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool) & np.isfinite(target) & np.isfinite(prediction)
    if valid.sum() < 2:
        return float("nan")
    observed = target[valid]
    predicted = prediction[valid]
    denominator = np.sum((observed - observed.mean()) ** 2)
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator) if denominator > 0 else float("nan")


def metric_row(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask.astype(bool) & np.isfinite(target) & np.isfinite(prediction)
    error = prediction[valid] - target[valid]
    return {
        "count": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": masked_r2(target, prediction, mask),
    }


def evaluate_predictions(
    model_name: str,
    prediction_scaled: np.ndarray,
    target_scaled: np.ndarray,
    mask: np.ndarray,
    target_times: np.ndarray,
    data: PreparedData,
    output_dir: Path,
) -> dict[str, float]:
    prediction = inverse_scale(prediction_scaled, data.mean, data.std)
    target = inverse_scale(target_scaled, data.mean, data.std)
    prediction = np.transpose(prediction, (0, 2, 1))
    target = np.transpose(target, (0, 2, 1))
    mask_time = np.transpose(mask, (0, 2, 1))

    sequence = metric_row(target, prediction, mask_time)
    terminal = metric_row(target[:, -1], prediction[:, -1], mask_time[:, -1])
    summary = {
        "model": model_name,
        "sequence_rmse": sequence["rmse"],
        "sequence_mae": sequence["mae"],
        "sequence_r2": sequence["r2"],
        "terminal_rmse": terminal["rmse"],
        "terminal_mae": terminal["mae"],
        "terminal_r2": terminal["r2"],
    }

    lead_rows = []
    for lead in range(target.shape[1]):
        row = metric_row(target[:, lead], prediction[:, lead], mask_time[:, lead])
        lead_rows.append({"model": model_name, "lead_hour": lead + 1, **row})
    pd.DataFrame(lead_rows).to_csv(output_dir / f"{model_name}_per_lead.csv", index=False)

    station_rows = []
    for station_index, station_name in enumerate(data.station_names):
        row = metric_row(
            target[:, :, station_index],
            prediction[:, :, station_index],
            mask_time[:, :, station_index],
        )
        station_rows.append({"model": model_name, "station_name": station_name, **row})
    pd.DataFrame(station_rows).to_csv(output_dir / f"{model_name}_per_station.csv", index=False)

    terminal_target = target[:, -1]
    terminal_prediction = prediction[:, -1]
    terminal_mask = mask_time[:, -1].astype(bool)
    event_truth = terminal_target >= data.train_q95[None, :]
    event_score = terminal_prediction - data.train_q95[None, :]
    valid = terminal_mask & np.isfinite(terminal_target) & np.isfinite(terminal_prediction)
    y_true = event_truth[valid].astype(int)
    y_score = event_score[valid]
    y_pred = y_score >= 0
    true_positive = int(np.sum((y_true == 1) & y_pred))
    false_positive = int(np.sum((y_true == 0) & y_pred))
    false_negative = int(np.sum((y_true == 1) & ~y_pred))
    summary.update(
        {
            "terminal_event_precision": true_positive / max(true_positive + false_positive, 1),
            "terminal_event_recall": true_positive / max(true_positive + false_negative, 1),
            "terminal_event_csi": true_positive / max(true_positive + false_positive + false_negative, 1),
            "terminal_event_pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        }
    )

    np.savez_compressed(
        output_dir / f"{model_name}_predictions.npz",
        prediction=prediction.astype(np.float32),
        target=target.astype(np.float32),
        mask=mask_time.astype(np.uint8),
        target_times=target_times,
        station_names=np.asarray(data.station_names),
    )
    return summary


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preliminary NeuralCORA-Surge station baselines")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=["persistence", "node_gru", "graph_wavenet"], default=["persistence", "node_gru", "graph_wavenet"])
    parser.add_argument("--input-hours", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--gwn-blocks", type=int, default=6)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.stride = max(args.stride, 168)
        args.hidden_dim = min(args.hidden_dim, 16)
        args.gwn_blocks = min(args.gwn_blocks, 2)

    set_reproducible(args.seed, args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(args.input)
    np.savetxt(args.output_dir / "distance_adjacency.csv", data.adjacency, delimiter=",")
    pd.DataFrame(
        {"station_name": data.station_names, "train_mean": data.mean, "train_std": data.std, "train_positive_q95": data.train_q95}
    ).to_csv(args.output_dir / "train_scaling_and_thresholds.csv", index=False)

    train_dataset = WindowDataset(data, "train", args.input_hours, args.horizon, args.stride)
    validation_dataset = WindowDataset(data, "validation", args.input_hours, args.horizon, args.stride)
    test_dataset = WindowDataset(data, "test", args.input_hours, args.horizon, args.stride)
    print(f"samples train={len(train_dataset)} validation={len(validation_dataset)} test={len(test_dataset)}")
    train_loader = make_loader(train_dataset, args.batch_size, True, args.seed)
    validation_loader = make_loader(validation_dataset, args.batch_size, False, args.seed)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries: list[dict[str, float]] = []

    if "persistence" in args.models:
        values = persistence_predictions(test_dataset)
        summaries.append(evaluate_predictions("persistence", *values, data, args.output_dir))

    factories = {
        "node_gru": lambda: NodeGRUForecaster(data.features.shape[-1], args.hidden_dim, args.horizon, args.dropout),
        "graph_wavenet": lambda: GraphWaveNetForecaster(
            data.features.shape[-1],
            args.hidden_dim,
            args.horizon,
            data.adjacency,
            args.gwn_blocks,
            args.diffusion_steps,
            args.dropout,
        ),
    }
    for model_name in (name for name in args.models if name != "persistence"):
        set_reproducible(args.seed, args.cpu_threads)
        model = factories[model_name]().to(device)
        start = time.perf_counter()
        history, best_validation = train_model(model, train_loader, validation_loader, args, device)
        elapsed = time.perf_counter() - start
        history.to_csv(args.output_dir / f"{model_name}_training_log.csv", index=False)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "station_names": data.station_names,
                "mean": data.mean,
                "std": data.std,
                "adjacency": data.adjacency,
                "args": vars(args),
            },
            args.output_dir / f"{model_name}_checkpoint.pt",
        )
        values = predict_model(model, test_loader, device)
        summary = evaluate_predictions(model_name, *values, data, args.output_dir)
        summary.update({"best_validation_masked_mse": best_validation, "training_seconds": elapsed})
        summaries.append(summary)

    pd.DataFrame(summaries).to_csv(args.output_dir / "model_summary.csv", index=False)
    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "station_names": data.station_names,
            "samples": {
                "train": len(train_dataset),
                "validation": len(validation_dataset),
                "test": len(test_dataset),
            },
            "split_boundaries": {key: [str(value[0]), str(value[1])] for key, value in SPLITS.items()},
        }
    )
    (args.output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
