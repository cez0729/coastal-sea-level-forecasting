from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT76 = Path(__file__).resolve().parent / "76_enhanced_forcing_physics_loss_ablation_v3.py"
DEFAULT_INPUT = Path(__file__).resolve().parent / "outputs" / "final_four_models_24h_5seed_matched"
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "publication_event_metrics"

spec = importlib.util.spec_from_file_location("v3_impl", SCRIPT76)
v3 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v3)


def load_residual_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "pred_states" in data.files:
        return data["pred_states"][..., 0], data["true_states"][..., 0]
    return data["pred_residual"], data["true_residual"]


def parse_run(path: Path) -> tuple[int | None, str]:
    text = str(path)
    matches = re.findall(r"seed[_]?(\d+)", text)
    seed = int(matches[-1]) if matches else None
    model = path.parent.name
    return seed, model


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    csi = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall_POD": recall,
        "F1": f1,
        "CSI": csi,
        "FAR": far,
    }


def event_groups(mask: np.ndarray, max_gap_hours: int) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    groups = [[int(indices[0])]]
    for idx in indices[1:]:
        if int(idx) - groups[-1][-1] <= max_gap_hours:
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])
    return [np.asarray(g, dtype=int) for g in groups]


def event_metrics(
    true_last: np.ndarray,
    pred_last: np.ndarray,
    thresholds: np.ndarray,
    max_gap_hours: int,
) -> dict[str, float]:
    detected = 0
    amplitude_errors = []
    timing_errors = []
    total_events = 0
    for station in range(true_last.shape[1]):
        threshold = float(thresholds[station])
        groups = event_groups(true_last[:, station] >= threshold, max_gap_hours)
        total_events += len(groups)
        for group in groups:
            start = max(0, int(group[0]) - max_gap_hours)
            end = min(true_last.shape[0], int(group[-1]) + max_gap_hours + 1)
            pred_window = pred_last[start:end, station]
            true_window = true_last[start:end, station]
            if np.any(pred_window >= threshold):
                detected += 1
            true_peak_idx = int(np.argmax(true_window))
            pred_peak_idx = int(np.argmax(pred_window))
            amplitude_errors.append(float(pred_window[pred_peak_idx] - true_window[true_peak_idx]))
            timing_errors.append(float(pred_peak_idx - true_peak_idx))
    return {
        "event_count": total_events,
        "event_detection_rate": detected / total_events if total_events else float("nan"),
        "peak_amplitude_bias": float(np.mean(amplitude_errors)) if amplitude_errors else float("nan"),
        "peak_amplitude_MAE": float(np.mean(np.abs(amplitude_errors))) if amplitude_errors else float("nan"),
        "peak_timing_MAE_hours": float(np.mean(np.abs(timing_errors))) if timing_errors else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-threshold event-based extreme evaluation.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--positive-quantile", type=float, default=0.95)
    parser.add_argument("--absolute-quantile", type=float, default=0.95)
    parser.add_argument("--max-event-gap-hours", type=int, default=6)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays, station_meta, _ = v3.build_enhanced_arrays()
    residual = arrays["residual"].astype(np.float64)
    train_end = int(len(residual) * args.train_ratio)
    positive_thresholds = np.quantile(residual[:train_end], args.positive_quantile, axis=0)
    absolute_thresholds = np.quantile(np.abs(residual[:train_end]), args.absolute_quantile, axis=0)

    rows = []
    station_rows = []
    for path in sorted(input_root.rglob("predictions.npz")):
        pred, true = load_residual_predictions(path)
        if pred.shape[-1] < 1:
            continue
        pred_last = pred[:, :, -1].astype(np.float64)
        true_last = true[:, :, -1].astype(np.float64)
        seed, model = parse_run(path)

        row = {
            "seed": seed,
            "model_key": model,
            "prediction_file": str(path),
            "threshold_scope": "training_period_per_station",
            **{
                f"positive_{k}": v
                for k, v in classification_metrics(
                    true_last >= positive_thresholds[None, :],
                    pred_last >= positive_thresholds[None, :],
                ).items()
            },
            **{
                f"absolute_{k}": v
                for k, v in classification_metrics(
                    np.abs(true_last) >= absolute_thresholds[None, :],
                    np.abs(pred_last) >= absolute_thresholds[None, :],
                ).items()
            },
            **event_metrics(true_last, pred_last, positive_thresholds, args.max_event_gap_hours),
        }
        rows.append(row)

        for station_idx, station in station_meta.reset_index(drop=True).iterrows():
            station_rows.append(
                {
                    "seed": seed,
                    "model_key": model,
                    "station_id": station.get("station_id", station_idx),
                    "station_index": station_idx,
                    "positive_q95_train_threshold": positive_thresholds[station_idx],
                    "absolute_q95_train_threshold": absolute_thresholds[station_idx],
                    **classification_metrics(
                        true_last[:, station_idx] >= positive_thresholds[station_idx],
                        pred_last[:, station_idx] >= positive_thresholds[station_idx],
                    ),
                }
            )

    if not rows:
        raise RuntimeError(f"No predictions.npz files found under {input_root}")
    all_df = pd.DataFrame(rows)
    all_df.to_csv(output_dir / "event_metrics_all_runs.csv", index=False)
    pd.DataFrame(station_rows).to_csv(output_dir / "event_metrics_by_station.csv", index=False)

    metric_cols = [
        "positive_precision",
        "positive_recall_POD",
        "positive_F1",
        "positive_CSI",
        "positive_FAR",
        "event_detection_rate",
        "peak_amplitude_MAE",
        "peak_timing_MAE_hours",
    ]
    agg = all_df.groupby("model_key")[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(x for x in col if x) for col in agg.columns.to_flat_index()]
    agg.to_csv(output_dir / "event_metrics_mean_std.csv", index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
