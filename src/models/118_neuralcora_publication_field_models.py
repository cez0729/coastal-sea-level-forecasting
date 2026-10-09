from __future__ import annotations

import argparse
import copy
import json
import math
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "neuralcora_surge"
DEFAULT_CATALOG = DATA_ROOT / "processed" / "publication_event_catalog_1999_2022.csv"
DEFAULT_CORA_ROOT = DATA_ROOT / "raw" / "cora" / "catalog_days"
DEFAULT_CACHE = DATA_ROOT / "processed" / "publication_field_grid_32x48.npz"
DEFAULT_ERA5_ROOT = DATA_ROOT / "raw" / "era5"
DEFAULT_ERA5_CACHE = DATA_ROOT / "processed" / "publication_era5_grid_32x48.npz"
DEFAULT_NOAA = DATA_ROOT / "processed" / "noaa_station_water_tide_residual_1999_2022.csv.gz"
_BATHYMETRY_RELATIVE = Path("data") / "raw" / "GEBCO_gebco_unzip" / "gebco_2025_n42.5_s38.0_w-76.0_e-70.0.nc"
DEFAULT_BATHYMETRY = (Path.cwd() / _BATHYMETRY_RELATIVE) if (Path.cwd() / _BATHYMETRY_RELATIVE).exists() else ROOT / _BATHYMETRY_RELATIVE
DEFAULT_OUTPUT = ROOT / "results" / "neuralcora_surge_publication_field_models"
UPSTREAM_ROOT = ROOT / "external" / "neuralcora_upstream"

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from src.networks import NeuralCoraCNN, NeuralCoraResNet, PeriodicConv2d, UNet  # noqa: E402


STATIONS = {
    "8461490": ("New London", 41.371666, -72.09556),
    "8510560": ("Montauk", 41.048332, -71.95944),
    "8516945": ("Kings Point", 40.8103, -73.7649),
    "8518750": ("The Battery", 40.700554, -74.01417),
    "8531680": ("Sandy Hook", 40.4669, -74.0094),
    "8534720": ("Atlantic City", 39.356667, -74.41805),
    "8536110": ("Cape May", 38.9683, -74.96),
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
    split_by_event: dict[str, str]
    labels: dict[str, str]
    lat: np.ndarray
    lon: np.ndarray
    mapped_lat: np.ndarray
    mapped_lon: np.ndarray
    sea_mask: np.ndarray
    train_mean: float
    train_std: float
    train_q95: float
    train_q99: float
    completeness: str
    target_definition: str = "CORA posterior total-water-level field"

    def events(self, split: str) -> list[str]:
        return [event for event, value in self.split_by_event.items() if value == split]


@dataclass
class ForcingData:
    fields: dict[str, np.ndarray]
    mean: np.ndarray
    std: np.ndarray


def era5_file_rank(path: Path) -> int:
    name = path.stem
    if "_events_" in name:
        return 3
    if "Q" in name:
        return 1
    token = name.rsplit("_", 1)[-1]
    return 2 if len(token) == 6 else 0


def save_forcing_cache(path: Path, forcing: ForcingData) -> None:
    event_ids = list(forcing.fields)
    payload: dict[str, np.ndarray] = {
        "event_ids": np.asarray(event_ids),
        "mean": forcing.mean.astype(np.float32),
        "std": forcing.std.astype(np.float32),
    }
    for event in event_ids:
        payload[f"forcing__{event}"] = forcing.fields[event].astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_forcing_cache(path: Path, data: FieldData) -> ForcingData:
    cached = np.load(path, allow_pickle=False)
    event_ids = cached["event_ids"].astype(str).tolist()
    missing = sorted(set(data.fields) - set(event_ids))
    if missing:
        raise RuntimeError(f"ERA5 cache does not cover {len(missing)} CORA event groups; rebuild it")
    return ForcingData(
        fields={event: cached[f"forcing__{event}"] for event in data.fields},
        mean=cached["mean"],
        std=cached["std"],
    )


def build_or_load_forcing(args: argparse.Namespace, data: FieldData) -> ForcingData:
    if args.era5_cache.exists() and not args.rebuild_era5_cache:
        return load_forcing_cache(args.era5_cache, data)
    paths = sorted(args.era5_root.glob("*.nc"), key=lambda path: (era5_file_rank(path), path.name))
    if not paths:
        raise FileNotFoundError(f"No completed ERA5 NetCDF files found in {args.era5_root}")
    fields = {event: np.full((len(data.times[event]), 3, len(data.lat), len(data.lon)), np.nan, dtype=np.float32) for event in data.fields}
    lookup: dict[int, list[tuple[str, int]]] = {}
    for event, times in data.times.items():
        for index, value in enumerate(times.astype("datetime64[ns]").astype(np.int64)):
            lookup.setdefault(int(value), []).append((event, index))
    required = set(lookup)
    for path in paths:
        with xr.open_dataset(path) as ds:
            time_name = "valid_time" if "valid_time" in ds.coords else "time"
            available_times = ds[time_name].values.astype("datetime64[ns]").astype(np.int64)
            selected_positions = [index for index, value in enumerate(available_times) if int(value) in required]
            if not selected_positions:
                continue
            selected = ds[["u10", "v10", "msl"]].isel({time_name: selected_positions})
            selected = selected.interp(latitude=data.lat.astype(float), longitude=data.lon.astype(float))
            values = np.stack(
                [selected[name].transpose(time_name, "latitude", "longitude").values for name in ("u10", "v10", "msl")],
                axis=1,
            ).astype(np.float32)
            selected_times = available_times[selected_positions]
            for local_index, value in enumerate(selected_times):
                for event, event_index in lookup[int(value)]:
                    fields[event][event_index] = values[local_index]
    missing_hours = {event: int(np.any(~np.isfinite(values), axis=(1, 2, 3)).sum()) for event, values in fields.items()}
    missing_hours = {event: count for event, count in missing_hours.items() if count}
    if missing_hours:
        sample = dict(list(missing_hours.items())[:10])
        raise RuntimeError(f"ERA5 download/cache is incomplete for {len(missing_hours)} event groups; examples: {sample}")
    train = np.concatenate([fields[event] for event in data.events("train")], axis=0)
    mean = train.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = train.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    forcing = ForcingData(fields, mean, std)
    save_forcing_cache(args.era5_cache, forcing)
    return forcing


def merge_overlapping_events(catalog: pd.DataFrame) -> pd.DataFrame:
    catalog = catalog.copy()
    catalog["window_start"] = pd.to_datetime(catalog["window_start"], utc=True)
    catalog["window_end"] = pd.to_datetime(catalog["window_end"], utc=True)
    rows: list[dict[str, object]] = []
    for split, group in catalog.groupby("split", sort=False):
        group = group.sort_values("window_start")
        current: list[object] = []
        current_start: pd.Timestamp | None = None
        current_end: pd.Timestamp | None = None
        group_number = 0

        def emit() -> None:
            nonlocal group_number
            if not current or current_start is None or current_end is None:
                return
            group_number += 1
            rows.append(
                {
                    "event_group": f"{split}_group_{group_number:03d}",
                    "split": split,
                    "window_start": current_start,
                    "window_end": current_end,
                    "source_event_ids": ";".join(str(item.event_id) for item in current),
                    "source_event_count": len(current),
                }
            )

        for item in group.itertuples(index=False):
            start, end = item.window_start, item.window_end
            if current_end is None or start > current_end + pd.Timedelta(hours=1):
                emit()
                current = [item]
                current_start, current_end = start, end
            else:
                current.append(item)
                current_end = max(current_end, end)
        emit()
    return pd.DataFrame(rows).sort_values("window_start").reset_index(drop=True)


def completed_file_index(cora_root: Path) -> dict[str, Path]:
    paths = sorted(cora_root.glob("*/*.nc"))
    result: dict[str, Path] = {}
    for path in paths:
        token = path.stem.rsplit("_", 1)[-1]
        current = pd.to_datetime(token, format="%Y%m%d").date().isoformat()
        result[current] = path
    return result


def dates_for_window(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    return [value.date().isoformat() for value in pd.date_range(start.floor("D"), end.floor("D"), freq="D")]


def select_complete_groups(groups: pd.DataFrame, files: dict[str, Path], allow_incomplete: bool) -> tuple[pd.DataFrame, str]:
    complete = groups.apply(
        lambda row: all(current in files for current in dates_for_window(row.window_start, row.window_end)),
        axis=1,
    )
    missing = groups.loc[~complete]
    if not missing.empty and not allow_incomplete:
        counts = missing.groupby("split").size().to_dict()
        raise RuntimeError(
            f"CORA catalog download is incomplete; missing event groups by split: {counts}. "
            "Use --allow-incomplete only for explicitly preliminary runs."
        )
    selected = groups.loc[complete].copy().reset_index(drop=True)
    if any(selected.groupby("split").size().reindex(["train", "validation", "test"], fill_value=0) == 0):
        raise RuntimeError("At least one complete event group is required in every split")
    status = "complete" if len(selected) == len(groups) else f"partial_{len(selected)}_of_{len(groups)}_groups"
    return selected, status


def build_stable_grid_mapping(
    files: dict[str, Path],
    groups: pd.DataFrame,
    height: int,
    width: int,
    max_nearest_degrees: float,
    min_train_valid: float,
    wet_sample_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dates = sorted(
        {
            current
            for row in groups[groups["split"] == "train"].itertuples(index=False)
            for current in dates_for_window(row.window_start, row.window_end)
            if current in files
        }
    )
    if not train_dates:
        raise RuntimeError("No completed training dates are available for wet-node selection")
    if len(train_dates) > wet_sample_days:
        positions = np.linspace(0, len(train_dates) - 1, wet_sample_days).round().astype(int)
        train_dates = [train_dates[position] for position in np.unique(positions)]
    reference = files[train_dates[0]]
    with xr.open_dataset(reference, engine="h5netcdf") as ds:
        node_lat = ds["lat"].values.astype(np.float32)
        node_lon = ds["lon"].values.astype(np.float32)
    valid_count = np.zeros(len(node_lat), dtype=np.int64)
    hours = 0
    for current in train_dates:
        with xr.open_dataset(files[current], engine="h5netcdf") as ds:
            values = ds["zeta"].values
            if ds["zeta"].dims == ("nodes", "time"):
                valid_count += np.isfinite(values).sum(axis=1)
                hours += values.shape[1]
            else:
                values = ds["zeta"].transpose("nodes", "time").values
                valid_count += np.isfinite(values).sum(axis=1)
                hours += values.shape[1]
    stable = valid_count / max(hours, 1) >= min_train_valid
    if stable.sum() < 100:
        raise RuntimeError(f"Only {stable.sum()} nodes meet the train wetness threshold")

    lat = np.linspace(float(node_lat.min()), float(node_lat.max()), height, dtype=np.float32)
    lon = np.linspace(float(node_lon.min()), float(node_lon.max()), width, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_scale = np.cos(np.deg2rad(float(lat.mean())))
    stable_indices = np.flatnonzero(stable)
    tree = cKDTree(np.column_stack([node_lat[stable_indices], node_lon[stable_indices] * lon_scale]))
    distances, positions = tree.query(np.column_stack([lat_grid.ravel(), lon_grid.ravel() * lon_scale]), k=1)
    mapping = stable_indices[positions].reshape(height, width)
    sea_mask = (distances.reshape(height, width) <= max_nearest_degrees).astype(np.float32)
    if sea_mask.mean() < 0.05:
        raise RuntimeError("Stable wet-node grid is too sparse; inspect threshold or maximum distance")
    return lat, lon, mapping, sea_mask, node_lat[mapping], node_lon[mapping]


def read_group_field(
    row: object,
    files: dict[str, Path],
    mapping: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = mapping.ravel()
    values_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    for current in dates_for_window(row.window_start, row.window_end):
        with xr.open_dataset(files[current], engine="h5netcdf") as ds:
            values_parts.append(ds["zeta"].isel(nodes=indices).transpose("time", "nodes").values.astype(np.float32))
            time_parts.append(ds["time"].values.astype("datetime64[ns]"))
    values = np.concatenate(values_parts)
    times = np.concatenate(time_parts)
    order = np.argsort(times)
    values, times = values[order], times[order]
    unique = np.concatenate(([True], times[1:] != times[:-1]))
    values, times = values[unique], times[unique]
    start = np.datetime64(row.window_start.tz_convert("UTC").tz_localize(None), "ns")
    end = np.datetime64(row.window_end.tz_convert("UTC").tz_localize(None), "ns")
    selected = (times >= start) & (times <= end)
    values, times = values[selected], times[selected]
    expected = np.arange(start, end + np.timedelta64(1, "h"), np.timedelta64(1, "h"), dtype="datetime64[ns]")
    if len(times) != len(expected) or not np.array_equal(times, expected):
        raise RuntimeError(f"Hourly CORA coverage is incomplete in {row.event_group}")
    return values.reshape(len(values), *mapping.shape), times


def save_field_cache(path: Path, data: FieldData) -> None:
    event_ids = list(data.fields)
    payload: dict[str, np.ndarray] = {
        "event_ids": np.asarray(event_ids),
        "event_splits": np.asarray([data.split_by_event[event] for event in event_ids]),
        "event_labels": np.asarray([data.labels[event] for event in event_ids]),
        "lat": data.lat,
        "lon": data.lon,
        "mapped_lat": data.mapped_lat,
        "mapped_lon": data.mapped_lon,
        "sea_mask": data.sea_mask,
        "train_mean": np.asarray(data.train_mean, dtype=np.float32),
        "train_std": np.asarray(data.train_std, dtype=np.float32),
        "train_q95": np.asarray(data.train_q95, dtype=np.float32),
        "train_q99": np.asarray(data.train_q99, dtype=np.float32),
        "completeness": np.asarray(data.completeness),
    }
    for event in event_ids:
        payload[f"field__{event}"] = data.fields[event]
        payload[f"time__{event}"] = data.times[event]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_field_cache(path: Path) -> FieldData:
    cached = np.load(path, allow_pickle=False)
    event_ids = cached["event_ids"].astype(str).tolist()
    splits = cached["event_splits"].astype(str).tolist()
    labels = cached["event_labels"].astype(str).tolist()
    return FieldData(
        fields={event: cached[f"field__{event}"] for event in event_ids},
        times={event: cached[f"time__{event}"] for event in event_ids},
        split_by_event=dict(zip(event_ids, splits)),
        labels=dict(zip(event_ids, labels)),
        lat=cached["lat"],
        lon=cached["lon"],
        mapped_lat=cached["mapped_lat"],
        mapped_lon=cached["mapped_lon"],
        sea_mask=cached["sea_mask"],
        train_mean=float(cached["train_mean"]),
        train_std=float(cached["train_std"]),
        train_q95=float(cached["train_q95"]),
        train_q99=float(cached["train_q99"]),
        completeness=str(cached["completeness"]),
        target_definition=(str(cached["target_definition"]) if "target_definition" in cached.files else "CORA posterior total-water-level field"),
    )


def build_or_load_data(args: argparse.Namespace) -> FieldData:
    if args.cache.exists() and not args.rebuild_cache:
        return load_field_cache(args.cache)
    catalog = pd.read_csv(args.catalog)
    groups = merge_overlapping_events(catalog)
    files = completed_file_index(args.cora_root)
    groups, completeness = select_complete_groups(groups, files, args.allow_incomplete)
    lat, lon, mapping, sea_mask, mapped_lat, mapped_lon = build_stable_grid_mapping(
        files,
        groups,
        args.grid_height,
        args.grid_width,
        args.max_nearest_degrees,
        args.min_train_valid,
        args.wet_sample_days,
    )
    fields: dict[str, np.ndarray] = {}
    times: dict[str, np.ndarray] = {}
    split_by_event: dict[str, str] = {}
    labels: dict[str, str] = {}
    for row in groups.itertuples(index=False):
        field, event_times = read_group_field(row, files, mapping)
        field = np.where(sea_mask[None] > 0, field, np.nan).astype(np.float32)
        fields[row.event_group] = field
        times[row.event_group] = event_times
        split_by_event[row.event_group] = row.split
        labels[row.event_group] = row.source_event_ids
    train_values = np.concatenate(
        [fields[event][:, sea_mask.astype(bool)].ravel() for event in fields if split_by_event[event] == "train"]
    )
    train_values = train_values[np.isfinite(train_values)]
    if not len(train_values):
        raise RuntimeError("No finite training values found")
    train_mean = float(train_values.mean())
    train_std = max(float(train_values.std()), 1e-6)
    data = FieldData(
        fields,
        times,
        split_by_event,
        labels,
        lat,
        lon,
        mapped_lat,
        mapped_lon,
        sea_mask,
        train_mean,
        train_std,
        float(np.quantile(train_values, 0.95)),
        float(np.quantile(train_values, 0.99)),
        completeness,
    )
    save_field_cache(args.cache, data)
    return data


class EventFieldDataset(Dataset):
    def __init__(
        self,
        data: FieldData,
        split: str,
        input_hours: int,
        lead_hours: int,
        stride: int,
    ) -> None:
        self.data = data
        self.events = data.events(split)
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
        x_raw = self.data.fields[event][start : start + self.input_hours]
        target_raw = self.data.fields[event][target_index]
        x = np.nan_to_num((x_raw - self.data.train_mean) / self.data.train_std, nan=0.0).astype(np.float32)
        target = np.nan_to_num((target_raw - self.data.train_mean) / self.data.train_std, nan=0.0).astype(np.float32)[None]
        valid_target = self.data.sea_mask.astype(bool) & np.isfinite(target_raw)
        target_time = self.data.times[event][target_index].astype("datetime64[ns]").astype(np.int64)
        return (
            torch.from_numpy(x),
            torch.from_numpy(target),
            torch.from_numpy(valid_target.astype(np.float32)[None]),
            event_index,
            target_time,
        )


class ForcingEventFieldDataset(EventFieldDataset):
    def __init__(self, data: FieldData, forcing: ForcingData, split: str, input_hours: int, lead_hours: int, stride: int, mode: str) -> None:
        super().__init__(data, split, input_hours, lead_hours, stride)
        if mode not in {"past", "future"}:
            raise ValueError(mode)
        self.forcing = forcing
        self.mode = mode

    def __getitem__(self, index: int):
        x, target, mask, event_index, target_time = super().__getitem__(index)
        sample_event_index, start = self.samples[index]
        event = self.events[sample_event_index]
        forcing_start = start if self.mode == "past" else start + self.input_hours
        forcing_end = forcing_start + self.input_hours
        forcing = self.forcing.fields[event][forcing_start:forcing_end]
        forcing = (forcing - self.forcing.mean[None, :, None, None]) / self.forcing.std[None, :, None, None]
        forcing_channels = np.ascontiguousarray(forcing.reshape(-1, *forcing.shape[-2:]), dtype=np.float32)
        return torch.cat([x, torch.from_numpy(forcing_channels)], dim=0), target, mask, event_index, target_time


class BathymetryEventFieldDataset(EventFieldDataset):
    def __init__(self, data: FieldData, bathymetry: np.ndarray, split: str, input_hours: int, lead_hours: int, stride: int) -> None:
        super().__init__(data, split, input_hours, lead_hours, stride)
        self.bathymetry = torch.from_numpy(np.ascontiguousarray(bathymetry[None], dtype=np.float32))

    def __getitem__(self, index: int):
        x, target, mask, event_index, target_time = super().__getitem__(index)
        return torch.cat([x, self.bathymetry], dim=0), target, mask, event_index, target_time


def load_bathymetry(path: Path, data: FieldData) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        elevation = ds["elevation"].interp(lat=data.lat.astype(float), lon=data.lon.astype(float)).values.astype(np.float32)
    depth = np.maximum(-elevation, 0.0)
    transformed = np.log1p(depth)
    valid = data.sea_mask.astype(bool) & np.isfinite(transformed)
    mean = float(transformed[valid].mean())
    std = max(float(transformed[valid].std()), 1e-6)
    scaled = np.nan_to_num((transformed - mean) / std, nan=0.0)
    return np.where(data.sea_mask > 0, scaled, 0.0).astype(np.float32)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = PeriodicConv2d(input_channels + hidden_channels, 4 * hidden_channels, 3)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        input_gate, forget_gate, output_gate, candidate = self.gates(torch.cat([x, hidden], dim=1)).chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
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


class RegionalConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = 4 if output_channels % 4 == 0 else 1
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RegionalUNet(nn.Module):
    def __init__(self, input_channels: int, base_channels: int = 8) -> None:
        super().__init__()
        self.encoder1 = RegionalConvBlock(input_channels, base_channels)
        self.encoder2 = RegionalConvBlock(base_channels, base_channels * 2)
        self.bottleneck = RegionalConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.decoder2 = RegionalConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.decoder1 = RegionalConvBlock(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(self.pool(encoder1))
        hidden = self.bottleneck(self.pool(encoder2))
        hidden = self.decoder2(torch.cat([self.up2(hidden), encoder2], dim=1))
        hidden = self.decoder1(torch.cat([self.up1(hidden), encoder1], dim=1))
        return self.output(hidden)


class SpectralConv2d(nn.Module):
    def __init__(self, channels: int, modes_lat: int, modes_lon: int) -> None:
        super().__init__()
        self.modes_lat = modes_lat
        self.modes_lon = modes_lon
        scale = 1 / max(channels, 1)
        self.weight_top = nn.Parameter(scale * torch.randn(channels, channels, modes_lat, modes_lon, dtype=torch.cfloat))
        self.weight_bottom = nn.Parameter(scale * torch.randn(channels, channels, modes_lat, modes_lon, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        transformed = torch.fft.rfft2(x)
        output = torch.zeros(batch, channels, height, width // 2 + 1, dtype=torch.cfloat, device=x.device)
        modes_lat = min(self.modes_lat, height // 2)
        modes_lon = min(self.modes_lon, width // 2 + 1)
        output[:, :, :modes_lat, :modes_lon] = torch.einsum(
            "bixy,ioxy->boxy", transformed[:, :, :modes_lat, :modes_lon], self.weight_top[:, :, :modes_lat, :modes_lon]
        )
        output[:, :, -modes_lat:, :modes_lon] = torch.einsum(
            "bixy,ioxy->boxy", transformed[:, :, -modes_lat:, :modes_lon], self.weight_bottom[:, :, :modes_lat, :modes_lon]
        )
        return torch.fft.irfft2(output, s=(height, width))


class FNOForecaster(nn.Module):
    def __init__(self, input_hours: int, width: int, modes_lat: int, modes_lon: int, layers: int = 4) -> None:
        super().__init__()
        self.input = nn.Conv2d(input_hours + 2, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes_lat, modes_lon) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.GroupNorm(4 if width % 4 == 0 else 1, width) for _ in range(layers)])
        self.output = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        lat = torch.linspace(-1, 1, height, device=x.device).view(1, 1, height, 1).expand(batch, 1, height, width)
        lon = torch.linspace(-1, 1, width, device=x.device).view(1, 1, 1, width).expand(batch, 1, height, width)
        x = self.input(torch.cat([x, lat, lon], dim=1))
        for spectral, local, norm in zip(self.spectral, self.local, self.norm):
            x = torch.nn.functional.gelu(norm(spectral(x) + local(x)))
        return self.output(x)


def geographical_edges(mapped_lat: np.ndarray, mapped_lon: np.ndarray, mask: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.argwhere(mask.astype(bool))
    coordinates = np.column_stack([mapped_lat[mask.astype(bool)], mapped_lon[mask.astype(bool)] * np.cos(np.deg2rad(mapped_lat[mask.astype(bool)]))])
    tree = cKDTree(coordinates)
    _, nearest = tree.query(coordinates, k=min(neighbors + 1, len(coordinates)))
    source = np.repeat(np.arange(len(coordinates)), nearest.shape[1] - 1)
    target = nearest[:, 1:].reshape(-1)
    return positions, np.vstack([source, target]).astype(np.int64)


class GeoGraphBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(channels, channels)
        self.neighbor_linear = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        aggregate = torch.zeros_like(x)
        aggregate.index_add_(1, target, x[:, source])
        degree = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
        aggregate = aggregate / degree.clamp_min(1).view(1, -1, 1)
        return torch.nn.functional.gelu(self.norm(x + self.self_linear(x) + self.neighbor_linear(aggregate)))


class GeoGraphForecaster(nn.Module):
    def __init__(self, input_hours: int, hidden: int, positions: np.ndarray, edge_index: np.ndarray, shape: tuple[int, int]) -> None:
        super().__init__()
        self.height, self.width = shape
        self.register_buffer("rows", torch.from_numpy(positions[:, 0]).long())
        self.register_buffer("columns", torch.from_numpy(positions[:, 1]).long())
        self.register_buffer("edge_index", torch.from_numpy(edge_index).long())
        self.input = nn.Linear(input_hours + 2, hidden)
        self.blocks = nn.ModuleList([GeoGraphBlock(hidden) for _ in range(4)])
        self.output = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        nodes = x[:, :, self.rows, self.columns].transpose(1, 2)
        lat = self.rows.float() / max(self.height - 1, 1) * 2 - 1
        lon = self.columns.float() / max(self.width - 1, 1) * 2 - 1
        coordinates = torch.stack([lat, lon], dim=-1).unsqueeze(0).expand(batch, -1, -1)
        hidden = torch.nn.functional.gelu(self.input(torch.cat([nodes, coordinates], dim=-1)))
        for block in self.blocks:
            hidden = block(hidden, self.edge_index)
        values = self.output(hidden).squeeze(-1)
        output = x.new_zeros(batch, 1, self.height, self.width)
        output[:, 0, self.rows, self.columns] = values
        return output


def masked_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latitude_weights: torch.Tensor,
    extreme_threshold_scaled: float,
    extreme_alpha: float,
) -> torch.Tensor:
    extreme_weight = 1.0 + extreme_alpha * (target >= extreme_threshold_scaled).to(target.dtype)
    weights = mask * latitude_weights * extreme_weight
    return (((prediction - target) ** 2) * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def validation_loss(model: nn.Module, loader: DataLoader, device: torch.device, lat_weights: torch.Tensor, threshold: float, alpha: float) -> float:
    model.eval()
    numerator, denominator = 0.0, 0.0
    for x, target, mask, _, _ in loader:
        target, mask = target.to(device), mask.to(device)
        prediction = model(x.to(device))
        extreme = 1.0 + alpha * (target >= threshold).to(target.dtype)
        weights = mask * lat_weights * extreme
        numerator += float((((prediction - target) ** 2) * weights).sum().cpu())
        denominator += float(weights.sum().cpu())
    return numerator / max(denominator, 1.0)


def train_model(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, args: argparse.Namespace, device: torch.device, lat_weights: torch.Tensor, alpha: float):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    threshold = (args.train_q95 - args.train_mean) / args.train_std
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    bad_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for x, target, mask, _, _ in train_loader:
            x, target, mask = x.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = masked_weighted_mse(prediction, target, mask, lat_weights, threshold, alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        current_validation = validation_loss(model, validation_loader, device, lat_weights, threshold, alpha)
        scheduler.step(current_validation)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": current_validation, "lr": optimizer.param_groups[0]["lr"]})
        print(f"epoch={epoch:03d} train={history[-1]['train_loss']:.6f} validation={current_validation:.6f}", flush=True)
        if current_validation < best_validation - args.min_delta:
            best_validation = current_validation
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
    output = [[] for _ in range(5)]
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
        for bucket, value in zip(output, (prediction, target, mask, event_index, target_time)):
            bucket.append(value.numpy())
    return tuple(np.concatenate(bucket, axis=0) for bucket in output)


def field_metrics(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray, q95: float, q99: float) -> dict[str, float | int]:
    valid = np.broadcast_to(mask.astype(bool), target.shape) & np.isfinite(target) & np.isfinite(prediction)
    observed, predicted = target[valid], prediction[valid]
    error = predicted - observed
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    row: dict[str, float | int] = {
        "count": int(len(observed)),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "mae_m": float(np.mean(np.abs(error))),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else np.nan,
    }
    for name, threshold in (("q95", q95), ("q99", q99)):
        truth, forecast = observed >= threshold, predicted >= threshold
        tp = int(np.sum(truth & forecast))
        fp = int(np.sum(~truth & forecast))
        fn = int(np.sum(truth & ~forecast))
        row[f"{name}_precision"] = tp / max(tp + fp, 1)
        row[f"{name}_recall"] = tp / max(tp + fn, 1)
        row[f"{name}_csi"] = tp / max(tp + fp + fn, 1)
    sample_mask = mask[:, 0].astype(bool)
    target_peaks = np.asarray([item[current].max() for item, current in zip(target[:, 0], sample_mask)])
    prediction_peaks = np.asarray([item[current].max() for item, current in zip(prediction[:, 0], sample_mask)])
    row["field_peak_mae_m"] = float(np.mean(np.abs(prediction_peaks - target_peaks)))
    return row


def evaluate_predictions(model_name: str, seed: int, values, dataset: EventFieldDataset, data: FieldData, output_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    prediction_scaled, target_scaled, masks, event_indices, target_times = values
    prediction = prediction_scaled * data.train_std + data.train_mean
    target = target_scaled * data.train_std + data.train_mean
    summary: dict[str, object] = {"model": model_name, "seed": seed, **field_metrics(target, prediction, masks, data.train_q95, data.train_q99)}
    rows: list[dict[str, object]] = []
    for event_index, event in enumerate(dataset.events):
        selected = event_indices == event_index
        metrics = field_metrics(target[selected], prediction[selected], masks[selected], data.train_q95, data.train_q99)
        sample_mask = masks[selected, 0].astype(bool)
        target_peaks = np.asarray([item[current].max() for item, current in zip(target[selected, 0], sample_mask)])
        prediction_peaks = np.asarray([item[current].max() for item, current in zip(prediction[selected, 0], sample_mask)])
        target_peak_index = int(np.argmax(target_peaks))
        prediction_peak_index = int(np.argmax(prediction_peaks))
        selected_times = target_times[selected].astype("datetime64[ns]")
        metrics.update(
            {
                "model": model_name,
                "seed": seed,
                "event_group": event,
                "source_event_ids": data.labels[event],
                "peak_amplitude_error_m": float(prediction_peaks[prediction_peak_index] - target_peaks[target_peak_index]),
                "peak_timing_error_h": float((selected_times[prediction_peak_index] - selected_times[target_peak_index]) / np.timedelta64(1, "h")),
            }
        )
        rows.append(metrics)
    per_event = pd.DataFrame(rows)
    per_event.to_csv(output_dir / f"{model_name}_seed{seed}_per_event.csv", index=False)
    return summary, per_event


def station_grid_mapping(data: FieldData) -> pd.DataFrame:
    valid = data.sea_mask.astype(bool)
    rows, columns = np.where(valid)
    mapped_lat = data.mapped_lat[valid]
    mapped_lon = data.mapped_lon[valid]
    output: list[dict[str, object]] = []
    for station_id, (name, station_lat, station_lon) in STATIONS.items():
        scale = np.cos(np.deg2rad(station_lat))
        distance = np.sqrt((mapped_lat - station_lat) ** 2 + ((mapped_lon - station_lon) * scale) ** 2)
        selected = int(np.argmin(distance))
        output.append(
            {
                "station_id": station_id,
                "station_name": name,
                "grid_row": int(rows[selected]),
                "grid_column": int(columns[selected]),
                "grid_node_lat": float(mapped_lat[selected]),
                "grid_node_lon": float(mapped_lon[selected]),
                "distance_km": float(distance[selected] * 111.195),
            }
        )
    return pd.DataFrame(output)


def station_prediction_frame(values, data: FieldData, mapping: pd.DataFrame) -> pd.DataFrame:
    prediction_scaled, target_scaled, masks, _, target_times = values
    prediction = prediction_scaled * data.train_std + data.train_mean
    target = target_scaled * data.train_std + data.train_mean
    parts: list[pd.DataFrame] = []
    for row in mapping.itertuples(index=False):
        valid = masks[:, 0, row.grid_row, row.grid_column].astype(bool)
        predicted = prediction[:, 0, row.grid_row, row.grid_column].copy()
        cora_target = target[:, 0, row.grid_row, row.grid_column].copy()
        predicted[~valid] = np.nan
        cora_target[~valid] = np.nan
        parts.append(
            pd.DataFrame(
                {
                    "station_id": str(row.station_id),
                    "station_name": row.station_name,
                    "time": pd.to_datetime(target_times.astype("datetime64[ns]"), utc=True),
                    "model_prediction_m": predicted,
                    "cora_target_m": cora_target,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def simple_pair_metrics(observed: pd.Series, predicted: pd.Series, prefix: str) -> dict[str, float | int]:
    valid = observed.notna() & predicted.notna()
    y = observed[valid].to_numpy(dtype=float)
    p = predicted[valid].to_numpy(dtype=float)
    if not len(y):
        return {f"{prefix}_n": 0, f"{prefix}_rmse_m": np.nan, f"{prefix}_mae_m": np.nan, f"{prefix}_r2": np.nan, f"{prefix}_correlation": np.nan}
    error = p - y
    denominator = float(np.sum((y - y.mean()) ** 2))
    correlation = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0 else np.nan
    return {
        f"{prefix}_n": int(len(y)),
        f"{prefix}_rmse_m": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_mae_m": float(np.mean(np.abs(error))),
        f"{prefix}_r2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else np.nan,
        f"{prefix}_correlation": correlation,
    }


def evaluate_noaa_stations(
    model_name: str,
    seed: int,
    validation_values,
    test_values,
    data: FieldData,
    mapping: pd.DataFrame,
    noaa: pd.DataFrame,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    validation = station_prediction_frame(validation_values, data, mapping).merge(noaa, on=["station_id", "time"], how="left")
    test = station_prediction_frame(test_values, data, mapping).merge(noaa, on=["station_id", "time"], how="left")
    calibrated: list[pd.DataFrame] = []
    for station_id, group in test.groupby("station_id", sort=False):
        fit = validation[validation["station_id"] == station_id].dropna(subset=["model_prediction_m", "observed_msl_m"])
        group = group.copy()
        bias = float((fit["observed_msl_m"] - fit["model_prediction_m"]).mean()) if len(fit) else 0.0
        if len(fit) >= 2 and fit["model_prediction_m"].std() > 1e-8:
            slope, intercept = np.polyfit(fit["model_prediction_m"], fit["observed_msl_m"], deg=1)
        else:
            slope, intercept = 1.0, bias
        group["bias_corrected_m"] = group["model_prediction_m"] + bias
        group["affine_corrected_m"] = intercept + slope * group["model_prediction_m"]
        group["validation_bias_m"] = bias
        group["validation_affine_intercept_m"] = intercept
        group["validation_affine_slope"] = slope
        calibrated.append(group)
    test = pd.concat(calibrated, ignore_index=True)
    rows: list[dict[str, object]] = []
    for station_name, group in [("all", test), *list(test.groupby("station_name", sort=False))]:
        row: dict[str, object] = {"model": model_name, "seed": seed, "station_name": station_name}
        row.update(simple_pair_metrics(group["observed_msl_m"], group["model_prediction_m"], "raw"))
        row.update(simple_pair_metrics(group["observed_msl_m"], group["bias_corrected_m"], "bias"))
        row.update(simple_pair_metrics(group["observed_msl_m"], group["affine_corrected_m"], "affine"))
        row.update(simple_pair_metrics(group["observed_msl_m"], group["cora_target_m"], "cora_reference"))
        rows.append(row)
    details = pd.DataFrame(rows)
    overall = details.iloc[0].to_dict()
    summary = {f"noaa_{key}": value for key, value in overall.items() if key not in {"model", "seed", "station_name"}}
    return summary, details


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None)


def make_event_balanced_train_loader(dataset: EventFieldDataset, batch_size: int, seed: int, samples_per_epoch: int) -> DataLoader:
    event_counts = np.bincount([event_index for event_index, _ in dataset.samples], minlength=len(dataset.events))
    weights = np.asarray([1.0 / max(event_counts[event_index], 1) for event_index, _ in dataset.samples], dtype=np.float64)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=min(max(samples_per_epoch, 1), len(dataset)),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def model_factory(name: str, args: argparse.Namespace, data: FieldData) -> nn.Module:
    if name == "cnn":
        return NeuralCoraCNN(args.input_hours, [16, 32, 32, 16], 3, 1, dropout=args.dropout)
    if name == "resnet":
        return NeuralCoraResNet(args.input_hours, [16, 16, 16, 16, 1], [3] * 5, bn_position="post", dropout=args.dropout, long_skip=True)
    if name == "unet":
        return UNet(args.input_hours, 3, 8, 1, bn_position="post", dropout=args.dropout)
    if name in {"regional_unet", "extreme_unet"}:
        return RegionalUNet(args.input_hours)
    if name in {"era5_past_unet", "era5_future_unet"}:
        return RegionalUNet(args.input_hours * 4)
    if name == "bathymetry_unet":
        return RegionalUNet(args.input_hours + 1)
    if name == "convlstm":
        return ConvLSTMForecaster(args.convlstm_hidden)
    if name == "fno":
        return FNOForecaster(args.input_hours, args.fno_width, args.fno_modes_lat, args.fno_modes_lon)
    if name in {"era5_past_fno", "era5_future_fno"}:
        return FNOForecaster(args.input_hours * 4, args.fno_width, args.fno_modes_lat, args.fno_modes_lon)
    if name == "geognn":
        positions, edge_index = geographical_edges(data.mapped_lat, data.mapped_lon, data.sea_mask, args.graph_neighbors)
        return GeoGraphForecaster(args.input_hours, args.graph_hidden, positions, edge_index, data.sea_mask.shape)
    raise ValueError(name)


def mean_std_table(summary: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in summary.select_dtypes(include=[np.number]).columns if column not in {"seed", "parameters"}]
    rows: list[dict[str, object]] = []
    for model, group in summary.groupby("model", sort=False):
        row: dict[str, object] = {"model": model, "seeds": int(group["seed"].nunique())}
        for column in numeric:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publication event-field baselines for NeuralCORA-Surge")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--cora-root", type=Path, default=DEFAULT_CORA_ROOT)
    parser.add_argument("--era5-root", type=Path, default=DEFAULT_ERA5_ROOT)
    parser.add_argument("--noaa-csv", type=Path, default=DEFAULT_NOAA)
    parser.add_argument("--bathymetry", type=Path, default=DEFAULT_BATHYMETRY)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--era5-cache", type=Path, default=DEFAULT_ERA5_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=["climatology", "persistence", "cnn", "resnet", "unet", "regional_unet", "extreme_unet", "bathymetry_unet", "convlstm", "fno", "geognn", "era5_past_unet", "era5_future_unet", "era5_past_fno", "era5_future_fno"], default=["climatology", "persistence", "cnn", "resnet", "unet", "regional_unet", "extreme_unet", "bathymetry_unet", "convlstm", "fno", "geognn", "era5_past_unet", "era5_future_unet", "era5_past_fno", "era5_future_fno"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 2025, 3407])
    parser.add_argument("--input-hours", type=int, default=24)
    parser.add_argument("--lead-hours", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=3)
    parser.add_argument("--validation-stride", type=int, default=6)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--grid-height", type=int, default=32)
    parser.add_argument("--grid-width", type=int, default=48)
    parser.add_argument("--max-nearest-degrees", type=float, default=0.04)
    parser.add_argument("--min-train-valid", type=float, default=0.95)
    parser.add_argument("--wet-sample-days", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-samples-per-epoch", type=int, default=3072)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--extreme-alpha", type=float, default=4.0)
    parser.add_argument("--convlstm-hidden", type=int, default=8)
    parser.add_argument("--fno-width", type=int, default=24)
    parser.add_argument("--fno-modes-lat", type=int, default=8)
    parser.add_argument("--fno-modes-lon", type=int, default=12)
    parser.add_argument("--graph-hidden", type=int, default=32)
    parser.add_argument("--graph-neighbors", type=int, default=6)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-era5-cache", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.train_stride = max(args.train_stride, 12)
        args.validation_stride = max(args.validation_stride, 6)
        args.test_stride = max(args.test_stride, 6)
        args.train_samples_per_epoch = min(args.train_samples_per_epoch, 256)

    set_reproducible(args.seeds[0], args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_or_load_data(args)
    args.train_mean, args.train_std, args.train_q95 = data.train_mean, data.train_std, data.train_q95
    train_dataset = EventFieldDataset(data, "train", args.input_hours, args.lead_hours, args.train_stride)
    validation_dataset = EventFieldDataset(data, "validation", args.input_hours, args.lead_hours, args.validation_stride)
    test_dataset = EventFieldDataset(data, "test", args.input_hours, args.lead_hours, args.test_stride)
    if min(len(train_dataset), len(validation_dataset), len(test_dataset)) == 0:
        raise RuntimeError("One or more splits contain no valid forecast windows")
    print(
        f"data={data.completeness} events=train:{len(data.events('train'))},validation:{len(data.events('validation'))},test:{len(data.events('test'))} "
        f"samples=train:{len(train_dataset)},validation:{len(validation_dataset)},test:{len(test_dataset)} sea_fraction={data.sea_mask.mean():.3f}",
        flush=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lat_weights = torch.from_numpy(np.cos(np.deg2rad(data.lat)).astype(np.float32))[None, None, :, None].to(device)
    summary_rows: list[dict[str, object]] = []
    per_event_parts: list[pd.DataFrame] = []
    noaa_parts: list[pd.DataFrame] = []
    residual_target = data.target_definition.startswith("CORA zeta minus")
    if residual_target:
        print(f"Residual target detected: {data.target_definition}; NOAA total-water-level station scoring disabled", flush=True)
    station_mapping = station_grid_mapping(data)
    station_mapping.to_csv(args.output_dir / "noaa_station_grid_mapping.csv", index=False)
    noaa = pd.read_csv(args.noaa_csv, compression="infer", usecols=["station_id", "time", "observed_msl_m"])
    noaa["station_id"] = noaa["station_id"].astype(str)
    noaa["time"] = pd.to_datetime(noaa["time"], utc=True)

    test_loader = make_loader(test_dataset, args.batch_size, False, args.seeds[0])
    validation_loader = make_loader(validation_dataset, args.batch_size, False, args.seeds[0])
    for baseline in [name for name in args.models if name in {"climatology", "persistence"}]:
        validation_values = collect_predictions(None, validation_loader, device, baseline)
        values = collect_predictions(None, test_loader, device, baseline)
        summary, per_event = evaluate_predictions(baseline, -1, values, test_dataset, data, args.output_dir)
        if residual_target:
            noaa_details = pd.DataFrame()
            summary["noaa_evaluation_status"] = "disabled_for_residual_target"
        else:
            noaa_summary, noaa_details = evaluate_noaa_stations(baseline, -1, validation_values, values, data, station_mapping, noaa)
            summary.update(noaa_summary)
        summary_rows.append(summary)
        per_event_parts.append(per_event)
        noaa_parts.append(noaa_details)

    trainable = [name for name in args.models if name not in {"climatology", "persistence"}]
    forcing = build_or_load_forcing(args, data) if any(name.startswith("era5_") for name in trainable) else None
    bathymetry = load_bathymetry(args.bathymetry, data) if "bathymetry_unet" in trainable else None
    for model_name in trainable:
        for seed in args.seeds:
            set_reproducible(seed, args.cpu_threads)
            if model_name.startswith("era5_"):
                assert forcing is not None
                mode = "past" if "_past_" in model_name else "future"
                model_train_dataset = ForcingEventFieldDataset(data, forcing, "train", args.input_hours, args.lead_hours, args.train_stride, mode)
                model_validation_dataset = ForcingEventFieldDataset(data, forcing, "validation", args.input_hours, args.lead_hours, args.validation_stride, mode)
                model_test_dataset = ForcingEventFieldDataset(data, forcing, "test", args.input_hours, args.lead_hours, args.test_stride, mode)
            elif model_name == "bathymetry_unet":
                assert bathymetry is not None
                model_train_dataset = BathymetryEventFieldDataset(data, bathymetry, "train", args.input_hours, args.lead_hours, args.train_stride)
                model_validation_dataset = BathymetryEventFieldDataset(data, bathymetry, "validation", args.input_hours, args.lead_hours, args.validation_stride)
                model_test_dataset = BathymetryEventFieldDataset(data, bathymetry, "test", args.input_hours, args.lead_hours, args.test_stride)
            else:
                model_train_dataset, model_validation_dataset, model_test_dataset = train_dataset, validation_dataset, test_dataset
            train_loader = make_event_balanced_train_loader(model_train_dataset, args.batch_size, seed, args.train_samples_per_epoch)
            validation_loader = make_loader(model_validation_dataset, args.batch_size, False, seed)
            test_loader = make_loader(model_test_dataset, args.batch_size, False, seed)
            model = model_factory(model_name, args, data).to(device)
            parameters = sum(parameter.numel() for parameter in model.parameters())
            alpha = args.extreme_alpha if model_name == "extreme_unet" else 0.0
            started = time.perf_counter()
            history, best_validation = train_model(model, train_loader, validation_loader, args, device, lat_weights, alpha)
            elapsed = time.perf_counter() - started
            history.to_csv(args.output_dir / f"{model_name}_seed{seed}_training_log.csv", index=False)
            torch.save({"model_state_dict": model.state_dict(), "model": model_name, "seed": seed, "data_completeness": data.completeness, "args": vars(args)}, args.output_dir / f"{model_name}_seed{seed}_checkpoint.pt")
            validation_values = collect_predictions(model, validation_loader, device)
            values = collect_predictions(model, test_loader, device)
            summary, per_event = evaluate_predictions(model_name, seed, values, model_test_dataset, data, args.output_dir)
            if residual_target:
                noaa_details = pd.DataFrame()
                summary["noaa_evaluation_status"] = "disabled_for_residual_target"
            else:
                noaa_summary, noaa_details = evaluate_noaa_stations(model_name, seed, validation_values, values, data, station_mapping, noaa)
                summary.update(noaa_summary)
            summary.update({"parameters": parameters, "best_validation_loss": best_validation, "training_seconds": elapsed})
            summary_rows.append(summary)
            per_event_parts.append(per_event)
            noaa_parts.append(noaa_details)
            pd.DataFrame(summary_rows).to_csv(args.output_dir / "seed_summary.partial.csv", index=False)

    summary_frame = pd.DataFrame(summary_rows)
    per_event_frame = pd.concat(per_event_parts, ignore_index=True)
    noaa_frame = pd.concat(noaa_parts, ignore_index=True) if any(not frame.empty for frame in noaa_parts) else pd.DataFrame()
    summary_frame.to_csv(args.output_dir / "seed_summary.csv", index=False)
    per_event_frame.to_csv(args.output_dir / "per_event_all_models.csv", index=False)
    noaa_frame.to_csv(args.output_dir / "noaa_station_metrics.csv", index=False)
    mean_std = mean_std_table(summary_frame)
    mean_std.to_csv(args.output_dir / "mean_std.csv", index=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "device": str(device),
            "data_completeness": data.completeness,
            "events": {split: data.events(split) for split in ("train", "validation", "test")},
            "samples": {"train": len(train_dataset), "validation": len(validation_dataset), "test": len(test_dataset)},
            "target": f"{args.lead_hours}-hour-ahead {data.target_definition}",
            "forcing_status": "era5_past_* models use issue-time historical reanalysis; era5_future_* models use future ERA5 and are retrospective upper bounds only",
            "evidence_status": "formal only when data_completeness=complete and quick=false",
        }
    )
    (args.output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(mean_std.to_string(index=False))


if __name__ == "__main__":
    main()
