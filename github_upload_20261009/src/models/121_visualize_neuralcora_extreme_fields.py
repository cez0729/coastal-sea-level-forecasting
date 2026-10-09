from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm


_CWD_ROOT = Path.cwd()
_SCRIPT_ROOT = Path(__file__).absolute().parents[1]
ROOT = _CWD_ROOT if (_CWD_ROOT / "数据整理" / "118_neuralcora_publication_field_models.py").exists() else _SCRIPT_ROOT
TRAINING_SCRIPT = ROOT / "数据整理" / "118_neuralcora_publication_field_models.py"
FIELD_CACHE = ROOT / "data" / "neuralcora_surge" / "processed" / "publication_field_grid_32x48.npz"
ERA5_CACHE = ROOT / "data" / "neuralcora_surge" / "processed" / "publication_era5_grid_32x48.npz"
DEFAULT_OUTPUT = ROOT / "results" / "neuralcora_surge_publication_matched_primary" / "field_visualizations"
SEEDS = (42, 123, 2024, 2025, 3407)


def load_training_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("neuralcora_publication_models", TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_paths() -> OrderedDict[str, list[Path]]:
    matched = ROOT / "results" / "neuralcora_surge_matched_budget_multiseed"
    cnn_multi = ROOT / "results" / "neuralcora_surge_cnn_multiseed"
    cnn_42 = ROOT / "results" / "neuralcora_surge_cnn_seed42_matched_budget"
    era5_multi = ROOT / "results" / "neuralcora_surge_era5_past_multiseed"
    era5_unet_42 = ROOT / "results" / "neuralcora_surge_era5_screening_seed42"
    era5_fno_42 = ROOT / "results" / "neuralcora_surge_era5_past_fno_seed42_complete"

    def paths(directory: Path, model: str, seed_42_directory: Path | None = None) -> list[Path]:
        return [
            (seed_42_directory or directory) / f"{model}_seed42_checkpoint.pt",
            *[directory / f"{model}_seed{seed}_checkpoint.pt" for seed in SEEDS[1:]],
        ]

    result: OrderedDict[str, list[Path]] = OrderedDict(
        [
            ("CNN", paths(cnn_multi, "cnn", cnn_42)),
            ("U-Net", paths(matched, "unet")),
            ("Regional U-Net", paths(matched, "regional_unet")),
            ("FNO", paths(matched, "fno")),
            ("Past-ERA5 U-Net", paths(era5_multi, "era5_past_unet", era5_unet_42)),
            ("Past-ERA5 FNO", paths(era5_multi, "era5_past_fno", era5_fno_42)),
        ]
    )
    missing = [path for model_paths in result.values() for path in model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(str(path) for path in missing))
    return result


def select_top_event_samples(dataset, data, count: int) -> list[dict[str, object]]:
    sea_mask = data.sea_mask.astype(bool)
    best_by_event: dict[str, dict[str, object]] = {}
    for sample_index, (event_index, start) in enumerate(dataset.samples):
        event = dataset.events[event_index]
        target_index = start + dataset.input_hours + dataset.lead_hours - 1
        field = data.fields[event][target_index]
        valid = sea_mask & np.isfinite(field)
        if not np.any(valid):
            continue
        peak = float(np.max(field[valid]))
        current = best_by_event.get(event)
        if current is None or peak > float(current["true_field_peak_m"]):
            best_by_event[event] = {
                "sample_index": sample_index,
                "event": event,
                "event_label": data.labels.get(event, event),
                "target_time": str(pd.Timestamp(data.times[event][target_index])),
                "true_field_peak_m": peak,
            }
    ranked = sorted(best_by_event.values(), key=lambda row: float(row["true_field_peak_m"]), reverse=True)
    if len(ranked) < count:
        raise RuntimeError(f"Requested {count} events but only found {len(ranked)}")
    return ranked[:count]


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ensemble_prediction(module, data, inputs: torch.Tensor, paths: list[Path]) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for path in paths:
        checkpoint = load_checkpoint(path)
        args = SimpleNamespace(**checkpoint["args"])
        model_name = str(checkpoint["model"])
        model = module.model_factory(model_name, args, data)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.inference_mode():
            scaled = model(inputs).cpu().numpy()[:, 0]
        predictions.append(scaled * data.train_std + data.train_mean)
        del model, checkpoint
        gc.collect()
    return np.mean(np.stack(predictions, axis=0), axis=0)


def csi(observed: np.ndarray, predicted: np.ndarray, valid: np.ndarray, threshold: float) -> float:
    truth = observed[valid] >= threshold
    forecast = predicted[valid] >= threshold
    tp = int(np.sum(truth & forecast))
    fp = int(np.sum(~truth & forecast))
    fn = int(np.sum(truth & ~forecast))
    return tp / max(tp + fp + fn, 1)


def panel_metrics(observed: np.ndarray, predicted: np.ndarray, valid: np.ndarray, q95: float, q99: float) -> dict[str, float]:
    error = predicted[valid] - observed[valid]
    return {
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "mae_m": float(np.mean(np.abs(error))),
        "true_peak_m": float(np.max(observed[valid])),
        "predicted_peak_m": float(np.max(predicted[valid])),
        "peak_absolute_error_m": float(abs(np.max(predicted[valid]) - np.max(observed[valid]))),
        "q95_csi": csi(observed, predicted, valid, q95),
        "q99_csi": csi(observed, predicted, valid, q99),
    }


def masked(field: np.ndarray, sea_mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~sea_mask.astype(bool) | ~np.isfinite(field), field)


def add_threshold_contours(ax, lon: np.ndarray, lat: np.ndarray, field: np.ndarray, sea_mask: np.ndarray, q95: float, q99: float) -> None:
    valid_values = field[sea_mask.astype(bool) & np.isfinite(field)]
    if valid_values.size == 0:
        return
    for threshold, color, width in ((q95, "#facc15", 0.7), (q99, "#e11d48", 1.0)):
        if float(valid_values.min()) <= threshold <= float(valid_values.max()):
            ax.contour(lon, lat, masked(field, sea_mask), levels=[threshold], colors=[color], linewidths=width)


def plot_fields(
    output: Path,
    cases: list[dict[str, object]],
    data,
    truth: np.ndarray,
    predictions: OrderedDict[str, np.ndarray],
    metrics: pd.DataFrame,
) -> None:
    panels: OrderedDict[str, np.ndarray] = OrderedDict([("Truth", truth), *predictions.items()])
    rows, cols = len(cases), len(panels)
    fig = plt.figure(figsize=(3.35 * cols + 0.55, 3.45 * rows))
    grid = fig.add_gridspec(rows, cols + 1, width_ratios=[1] * cols + [0.045], wspace=0.06, hspace=0.17)
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(cols)] for row in range(rows)])
    colorbar_axes = [fig.add_subplot(grid[row, -1]) for row in range(rows)]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e8e5df")
    aspect = 1 / np.cos(np.deg2rad(float(np.mean(data.lat))))

    for row, case in enumerate(cases):
        row_values = np.concatenate([values[row][data.sea_mask.astype(bool)] for values in panels.values()])
        limit = max(float(np.nanmax(np.abs(row_values))), 1e-3)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        mappable = None
        for col, (name, values) in enumerate(panels.items()):
            ax = axes[row, col]
            field = values[row]
            mappable = ax.pcolormesh(data.lon, data.lat, masked(field, data.sea_mask), shading="auto", cmap=cmap, norm=norm)
            add_threshold_contours(ax, data.lon, data.lat, field, data.sea_mask, data.train_q95, data.train_q99)
            ax.set_aspect(aspect)
            ax.set_facecolor("#e8e5df")
            if row == 0:
                ax.set_title(name, fontsize=11, fontweight="bold")
            if name == "Truth":
                detail = f"peak={case['true_field_peak_m']:.3f} m"
            else:
                current = metrics[(metrics["case_rank"] == row + 1) & (metrics["model"] == name)].iloc[0]
                detail = f"RMSE={current.rmse_m:.3f} | peak err={current.peak_absolute_error_m:.3f} m"
            ax.text(
                0.02,
                0.02,
                detail,
                transform=ax.transAxes,
                fontsize=7.5,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.66, "edgecolor": "none", "pad": 2},
            )
            if col == 0:
                label = str(case["event_label"])
                ax.set_ylabel(f"#{row + 1} {label}\n{case['target_time']}\nLatitude", fontsize=8.5)
            else:
                ax.set_yticklabels([])
            if row == rows - 1:
                ax.set_xlabel("Longitude", fontsize=8.5)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=7)
        if mappable is not None:
            colorbar = fig.colorbar(mappable, cax=colorbar_axes[row])
            colorbar.set_label("Total water level (m)", fontsize=8)
            colorbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Top three independent test events by observed field peak: truth vs five-seed ensemble forecasts\n"
        f"Yellow contour: training q95={data.train_q95:.3f} m | Red contour: training q99={data.train_q99:.3f} m",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.045, right=0.975, bottom=0.055, top=0.91)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_errors(
    output: Path,
    cases: list[dict[str, object]],
    data,
    truth: np.ndarray,
    predictions: OrderedDict[str, np.ndarray],
    metrics: pd.DataFrame,
) -> None:
    rows, cols = len(cases), len(predictions)
    fig = plt.figure(figsize=(3.45 * cols + 0.55, 3.4 * rows))
    grid = fig.add_gridspec(rows, cols + 1, width_ratios=[1] * cols + [0.045], wspace=0.06, hspace=0.17)
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(cols)] for row in range(rows)])
    colorbar_axis = fig.add_subplot(grid[:, -1])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#e8e5df")
    sea_mask = data.sea_mask.astype(bool)
    errors = [np.abs(values - truth) for values in predictions.values()]
    vmax = max(float(np.nanmax(error[:, sea_mask])) for error in errors)
    aspect = 1 / np.cos(np.deg2rad(float(np.mean(data.lat))))
    mappable = None

    for row, case in enumerate(cases):
        for col, (name, values) in enumerate(predictions.items()):
            ax = axes[row, col]
            error = np.abs(values[row] - truth[row])
            mappable = ax.pcolormesh(data.lon, data.lat, masked(error, data.sea_mask), shading="auto", cmap=cmap, vmin=0, vmax=vmax)
            ax.set_aspect(aspect)
            ax.set_facecolor("#e8e5df")
            current = metrics[(metrics["case_rank"] == row + 1) & (metrics["model"] == name)].iloc[0]
            if row == 0:
                ax.set_title(name, fontsize=11, fontweight="bold")
            ax.text(
                0.02,
                0.02,
                f"MAE={current.mae_m:.3f} m | q95 CSI={current.q95_csi:.2f}",
                transform=ax.transAxes,
                fontsize=7.5,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.66, "edgecolor": "none", "pad": 2},
            )
            if col == 0:
                ax.set_ylabel(f"#{row + 1} {case['event_label']}\n{case['target_time']}\nLatitude", fontsize=8.5)
            else:
                ax.set_yticklabels([])
            if row == rows - 1:
                ax.set_xlabel("Longitude", fontsize=8.5)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=7)

    if mappable is not None:
        colorbar = fig.colorbar(mappable, cax=colorbar_axis)
        colorbar.set_label("Absolute error (m), common scale", fontsize=9)
    fig.suptitle("Absolute-error maps for the same three extreme test events", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.975, bottom=0.06, top=0.91)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize truth and matched-budget NeuralCORA model fields together")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-events", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.cpu_threads))
    module = load_training_module()
    data = module.load_field_cache(FIELD_CACHE)
    forcing = module.load_forcing_cache(ERA5_CACHE, data)
    base_dataset = module.EventFieldDataset(data, "test", input_hours=24, lead_hours=24, stride=1)
    forcing_dataset = module.ForcingEventFieldDataset(data, forcing, "test", input_hours=24, lead_hours=24, stride=1, mode="past")
    cases = select_top_event_samples(base_dataset, data, args.top_events)
    indices = [int(case["sample_index"]) for case in cases]

    base_items = [base_dataset[index] for index in indices]
    forcing_items = [forcing_dataset[index] for index in indices]
    base_inputs = torch.stack([item[0] for item in base_items])
    forcing_inputs = torch.stack([item[0] for item in forcing_items])
    truth = np.stack([item[1].numpy()[0] * data.train_std + data.train_mean for item in base_items])
    sea_mask = data.sea_mask.astype(bool)

    model_checkpoints = checkpoint_paths()
    predictions: OrderedDict[str, np.ndarray] = OrderedDict()
    predictions["Persistence"] = base_inputs[:, -1].numpy() * data.train_std + data.train_mean
    for display_name, paths in model_checkpoints.items():
        model_inputs = forcing_inputs if display_name.startswith("Past-ERA5") else base_inputs
        print(f"Predicting {display_name} from {len(paths)} checkpoints", flush=True)
        predictions[display_name] = ensemble_prediction(module, data, model_inputs, paths)

    metric_rows: list[dict[str, object]] = []
    for case_rank, case in enumerate(cases, start=1):
        valid = sea_mask & np.isfinite(truth[case_rank - 1])
        for model_name, values in predictions.items():
            metric_rows.append(
                {
                    "case_rank": case_rank,
                    **case,
                    "model": model_name,
                    **panel_metrics(truth[case_rank - 1], values[case_rank - 1], valid, data.train_q95, data.train_q99),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "top3_extreme_event_panel_metrics.csv", index=False)
    np.savez_compressed(
        args.output_dir / "top3_extreme_event_selected_fields.npz",
        truth=truth,
        sea_mask=data.sea_mask,
        lat=data.lat,
        lon=data.lon,
        **{f"prediction__{name.lower().replace('-', '_').replace(' ', '_')}": values for name, values in predictions.items()},
    )

    field_output = args.output_dir / "top3_extreme_event_field_comparison_5seed_ensemble.png"
    error_output = args.output_dir / "top3_extreme_event_absolute_error_5seed_ensemble.png"
    plot_fields(field_output, cases, data, truth, OrderedDict((key, value.copy()) for key, value in predictions.items()), metrics)
    plot_errors(error_output, cases, data, truth, predictions, metrics)

    metadata = {
        "selection": "The three independent test events with the largest observed total-water-level field peak; one peak target time per event.",
        "target": "CORA zeta total water level at a 24-hour lead, not tide-removed residual.",
        "extreme_definition": {
            "q95_m": data.train_q95,
            "q99_m": data.train_q99,
            "source": "Fixed percentiles of all valid training water-level grid-point values.",
        },
        "forecast_display": "Arithmetic mean of five independently trained matched-budget checkpoints for each neural architecture.",
        "cases": cases,
        "checkpoints": {name: [str(path) for path in paths] for name, paths in model_checkpoints.items()},
        "outputs": [str(field_output), str(error_output)],
    }
    (args.output_dir / "visualization_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
