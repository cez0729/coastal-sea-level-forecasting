from __future__ import annotations

"""Evaluate the frozen Physics-Reliability HS-DT-GWN on 2026 H1 once."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


calibration = load_module("frozen_calibration", HERE / "156_validation_locked_physics_reliability_gwn.py")
pc = calibration.pc


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def subset(split: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[mask] for key, value in split.items()}


def collect_seed(seed: int, args: argparse.Namespace, device: torch.device):
    pc.p104.set_reproducible(seed, args.cpu_threads)
    data = pc.rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    eta_model = pc.load_expert(seed, data, args, 1, device)
    multi_model = pc.load_expert(seed, data, args, 4, device)
    ode = pc.load_ode(seed, data, args, device)
    adjacency = torch.tensor(data["graph_priors"][args.fixed_graph_type], dtype=torch.float32, device=device)
    val = pc.collect_split(data["multi_val"], eta_model, multi_model, ode, adjacency, args, device, seed)
    extended_test = pc.collect_split(data["multi_test"], eta_model, multi_model, ode, adjacency, args, device, seed)
    forecast_indices = np.asarray(data["multi_test"].indices, dtype=np.int64)
    forecast_times = pd.to_datetime(data["arrays"]["time"][forecast_indices])
    holdout_mask = np.asarray(
        (forecast_times >= pd.Timestamp(args.holdout_start)) & (forecast_times < pd.Timestamp(args.holdout_end))
    )
    for key in ("eta", "multi", "prior", "forcing", "target", "tide"):
        values = extended_test[key]
        holdout_mask &= np.isfinite(values).reshape(len(values), -1).all(axis=1)
    if not np.any(holdout_mask):
        raise RuntimeError("No 2026 holdout windows were constructed")
    holdout = subset(extended_test, holdout_mask)
    return data, val, extended_test, holdout, forecast_times[holdout_mask]


def run_seed(seed: int, args: argparse.Namespace, device: torch.device):
    data, val, _, holdout, times = collect_seed(seed, args, device)
    no_physics, no_physics_meta = calibration.fit_candidate(val, holdout, False, args)
    candidate, physics_meta = calibration.fit_candidate(val, holdout, True, args)
    hsdt = calibration.fixed_hsdt(holdout)

    out = project_path(args.output_dir)
    np.savez_compressed(
        out / f"seed_{seed}_test_frozen_features.npz",
        **holdout,
        forecast_start=times.to_numpy(dtype="datetime64[ns]"),
    )
    np.savez_compressed(
        out / f"seed_{seed}_validation_locked_ridge_no_physics_predictions.npz",
        pred_residual=no_physics,
        true_residual=holdout["target"],
        target_tide=holdout["tide"],
    )
    np.savez_compressed(
        out / f"seed_{seed}_physics_reliability_hsdt_gwn_predictions.npz",
        pred_residual=candidate,
        true_residual=holdout["target"],
        target_tide=holdout["tide"],
    )

    train_end = pc.rolling.time_index(data["arrays"]["time"], args.fold_train_end)
    thresholds = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    predictions = {
        "gwn_eta_only": holdout["eta"],
        "gwn_multistate_no_physics": holdout["multi"],
        "hs_dt_gwn": hsdt,
        "validation_locked_ridge_no_physics": no_physics,
        "physics_reliability_hsdt_gwn": candidate,
    }
    rows = [
        {"seed": seed, "model": name, **pc.score(holdout["target"], prediction, holdout["tide"], thresholds)}
        for name, prediction in predictions.items()
    ]
    metadata = [
        {"seed": seed, "model": "validation_locked_ridge_no_physics", **no_physics_meta},
        {"seed": seed, "model": "physics_reliability_hsdt_gwn", **physics_meta},
    ]
    return rows, metadata, {"seed": seed, "windows": int(len(times)), "start": str(times.min()), "end": str(times.max())}


def summarize(args: argparse.Namespace, rows: list[dict], metadata: list[dict], windows: list[dict]) -> None:
    out = project_path(args.output_dir)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "all_runs.csv", index=False)
    pd.DataFrame(metadata).to_csv(out / "selected_calibrators.csv", index=False)
    pd.DataFrame(windows).to_csv(out / "holdout_windows.csv", index=False)
    metrics = [
        "seq_residual_R2", "last_residual_R2", "last_residual_RMSE",
        "extreme_abs_q95_residual_R2", "event_PR_AUC", "event_CSI",
    ]
    summary = runs.groupby("model")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(value) for value in col if value) for col in summary.columns.to_flat_index()]
    summary.to_csv(out / "mean_std.csv", index=False)
    pivot = runs.pivot(index="seed", columns="model", values=metrics)
    comparisons = []
    for baseline in ("hs_dt_gwn", "validation_locked_ridge_no_physics"):
        for metric in metrics:
            delta = pivot[(metric, "physics_reliability_hsdt_gwn")] - pivot[(metric, baseline)]
            improvement = -delta if metric.endswith(("RMSE", "MAE")) else delta
            comparisons.append(
                {
                    "candidate": "physics_reliability_hsdt_gwn",
                    "baseline": baseline,
                    "metric": metric,
                    "mean_improvement": float(improvement.mean()),
                    "std_improvement": float(improvement.std(ddof=1)),
                    "wins": int((improvement > 0).sum()),
                    "count": int(len(improvement)),
                }
            )
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen 2026 H1 model evaluation")
    parser.add_argument("--data-dir", default="data/processed_multiyear_2023_2026_h1_frozen")
    parser.add_argument("--source-results", default="results/confirmatory_hsdt_orc_refit_2025_h2")
    parser.add_argument("--output-dir", default="results/frozen_2026_h1_physics_reliability_gwn")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--fold-train-end", default="2025-01-01")
    parser.add_argument("--fold-val-end", default="2025-07-01")
    parser.add_argument("--fold-test-end", default="2026-07-01")
    parser.add_argument("--holdout-start", default="2026-01-01")
    parser.add_argument("--holdout-end", default="2026-07-01")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", default="distance", choices=["identity", "distance", "corr"])
    parser.add_argument("--physics-forcing-mode", default="last_input", choices=["last_input"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--gwn-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--shrinks", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--event-weights", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--terminal-weight", type=float, default=2.0)
    parser.add_argument("--selection-terminal-weight", type=float, default=1.0)
    parser.add_argument("--selection-extreme-weight", type=float, default=2.0)
    parser.add_argument("--correction-clip-scale", type=float, default=2.0)
    parser.add_argument("--protection-quantiles", type=float, nargs="+", default=[0.80, 0.90, 0.95, 1.0])
    parser.add_argument("--protected-shrinks", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    args = parser.parse_args()
    args.resume = False
    args.gate_hidden = 64
    args.gate_epochs = args.gate_patience = 1
    args.gate_lr = 1e-3
    args.weight_decay = 1e-5
    args.gate_smooth_weight = args.correction_reg_weight = args.physics_correction_reg_weight = 0.0
    args.explicit_physics_candidate = False
    args.grad_clip = 1.0
    args.min_delta = 1e-5
    args.print_every = 1
    return args


def main() -> None:
    args = parse_args()
    out = project_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = pc.confirm.configure_data_dir(args.data_dir)
    (out / "experiment_config.json").write_text(
        json.dumps({**vars(args), "data_dir": str(data_dir), "test_used_for_selection": False, "frozen_before_2026_scoring": True}, indent=2),
        encoding="utf-8",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, metadata, windows = [], [], []
    for seed in args.seeds:
        print(f"seed={seed} device={device}", flush=True)
        seed_rows, seed_metadata, seed_windows = run_seed(seed, args, device)
        rows.extend(seed_rows)
        metadata.extend(seed_metadata)
        windows.append(seed_windows)
    summarize(args, rows, metadata, windows)


if __name__ == "__main__":
    main()
