from __future__ import annotations

"""Validation-locked physics-reliability calibration for HS-DT-GWN.

The frozen eta-only and multistate GWN experts are never fine-tuned here.
The chronological validation period is split into calibration and selection
halves. Hyperparameters are selected on the latter half, then the calibrator
is refit on the full validation period before one test-period evaluation.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pc = load_module("validation_locked_pc", HERE / "155_physics_conditioned_hsdt_gwn.py")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def fixed_hsdt(split: dict[str, np.ndarray]) -> np.ndarray:
    horizon = split["eta"].shape[-1]
    weight = np.full(horizon, 0.5, dtype=np.float32)
    weight[-1] = 1.0
    return split["eta"] + weight[None, None, :] * (split["multi"] - split["eta"])


def make_features(split: dict[str, np.ndarray], use_physics: bool) -> np.ndarray:
    eta = split["eta"]
    multi = split["multi"]
    prior = split["prior"]
    forcing = split["forcing"][..., None]
    baseline = fixed_hsdt(split)
    samples, nodes, horizon = baseline.shape

    diff = multi - eta
    prior_delta = prior - baseline
    force = np.broadcast_to(forcing, baseline.shape)
    lead = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[None, None, :]
    lead = np.broadcast_to(lead, baseline.shape)
    station = np.broadcast_to(np.arange(nodes, dtype=np.int64)[None, :, None], baseline.shape)
    hgroup = np.broadcast_to((np.arange(horizon) // 6).astype(np.int64)[None, None, :], baseline.shape)

    station_oh = np.eye(nodes, dtype=np.float32)[station]
    hgroup_oh = np.eye(4, dtype=np.float32)[hgroup]
    parts = [
        baseline[..., None],
        eta[..., None],
        multi[..., None],
        diff[..., None],
        np.abs(diff)[..., None],
        lead[..., None],
        (lead**2)[..., None],
        station_oh,
        hgroup_oh,
        diff[..., None] * hgroup_oh,
        diff[..., None] * station_oh,
    ]

    physics_parts = [
        prior[..., None],
        prior_delta[..., None],
        np.abs(prior_delta)[..., None],
        force[..., None],
        (force * diff)[..., None],
        (force * prior_delta)[..., None],
        prior_delta[..., None] * hgroup_oh,
        prior_delta[..., None] * station_oh,
        force[..., None] * hgroup_oh,
        force[..., None] * station_oh,
    ]
    if not use_physics:
        physics_parts = [np.zeros_like(value) for value in physics_parts]
    parts.extend(physics_parts)
    return np.concatenate(parts, axis=-1).reshape(samples * nodes * horizon, -1).astype(np.float32)


def flatten(values: np.ndarray) -> np.ndarray:
    return values.reshape(-1).astype(np.float32)


def row_mask(samples: int, nodes: int, horizon: int, sample_mask: np.ndarray) -> np.ndarray:
    return np.broadcast_to(sample_mask[:, None, None], (samples, nodes, horizon)).reshape(-1)


def fit_candidate(
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    use_physics: bool,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    val_base = fixed_hsdt(val)
    test_base = fixed_hsdt(test)
    val_x = make_features(val, use_physics)
    test_x = make_features(test, use_physics)
    val_y = flatten(val["target"] - val_base)
    samples, nodes, horizon = val_base.shape
    cut = samples // 2
    calibration_samples = np.arange(samples) < cut
    selection_samples = ~calibration_samples
    calibration = row_mask(samples, nodes, horizon, calibration_samples)
    selection = row_mask(samples, nodes, horizon, selection_samples)
    lead24 = np.tile(np.arange(horizon) == horizon - 1, samples * nodes)
    cal_target = flatten(val["target"])[calibration]
    extreme_threshold = float(np.quantile(np.abs(cal_target), args.extreme_quantile))

    scaler = StandardScaler().fit(val_x[calibration])
    x_cal = scaler.transform(val_x[calibration])
    x_select = scaler.transform(val_x[selection])
    y_scale = max(float(np.std(val_y[calibration])), 1e-6)
    y_cal = val_y[calibration] / y_scale
    select_true = flatten(val["target"])[selection]
    select_base = flatten(val_base)[selection]
    select_lead24 = lead24[selection]
    select_extreme = np.abs(select_true) >= extreme_threshold

    calibration_base = flatten(val_base)[calibration]
    protection_thresholds = {
        float(quantile): float(np.quantile(np.abs(calibration_base), quantile))
        for quantile in args.protection_quantiles
    }
    best = None
    for event_weight in args.event_weights:
        weights = np.ones_like(y_cal)
        weights[np.abs(cal_target) >= extreme_threshold] *= event_weight
        weights[lead24[calibration]] *= args.terminal_weight
        for alpha in args.alphas:
            model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", max_iter=3000)
            model.fit(x_cal, y_cal, sample_weight=weights)
            raw = model.predict(x_select) * y_scale
            for shrink in args.shrinks:
                for protection_quantile, protection_threshold in protection_thresholds.items():
                    protected = np.abs(select_base) >= protection_threshold
                    for protected_shrink in args.protected_shrinks:
                        multiplier = np.where(protected, protected_shrink, 1.0)
                        pred = select_base + shrink * raw * multiplier
                        error = pred - select_true
                        score = float(np.mean(error**2))
                        score += args.selection_terminal_weight * float(np.mean(error[select_lead24] ** 2))
                        if np.any(select_extreme):
                            score += args.selection_extreme_weight * float(np.mean(error[select_extreme] ** 2))
                        record = (
                            score,
                            float(alpha),
                            float(shrink),
                            float(event_weight),
                            protection_quantile,
                            float(protected_shrink),
                        )
                        if best is None or record[0] < best[0]:
                            best = record

    assert best is not None
    _, alpha, shrink, event_weight, protection_quantile, protected_shrink = best
    all_target = flatten(val["target"])
    all_extreme_threshold = float(np.quantile(np.abs(all_target), args.extreme_quantile))
    all_weights = np.ones_like(val_y)
    all_weights[np.abs(all_target) >= all_extreme_threshold] *= event_weight
    all_weights[lead24] *= args.terminal_weight
    scaler = StandardScaler().fit(val_x)
    x_val = scaler.transform(val_x)
    x_test = scaler.transform(test_x)
    y_scale = max(float(np.std(val_y)), 1e-6)
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", max_iter=3000)
    model.fit(x_val, val_y / y_scale, sample_weight=all_weights)
    correction = model.predict(x_test).reshape(test_base.shape) * y_scale * shrink
    protection_threshold = float(np.quantile(np.abs(val_base), protection_quantile))
    protection_multiplier = np.where(np.abs(test_base) >= protection_threshold, protected_shrink, 1.0)
    correction *= protection_multiplier
    clip = args.correction_clip_scale * float(np.std(val_y))
    correction = np.clip(correction, -clip, clip)
    prediction = (test_base + correction).astype(np.float32)
    metadata = {
        "use_physics": use_physics,
        "alpha": alpha,
        "shrink": shrink,
        "event_weight": event_weight,
        "protection_quantile": protection_quantile,
        "protected_shrink": protected_shrink,
        "protection_threshold": protection_threshold,
        "selection_score": best[0],
        "calibration_samples": int(cut),
        "selection_samples": int(samples - cut),
        "feature_count": int(val_x.shape[1]),
        "correction_abs_mean": float(np.mean(np.abs(correction))),
        "correction_clip": clip,
    }
    return prediction, metadata


def collect_seed(seed: int, args: argparse.Namespace, device: torch.device):
    pc.p104.set_reproducible(seed, args.cpu_threads)
    data = pc.rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    eta_model = pc.load_expert(seed, data, args, 1, device)
    multi_model = pc.load_expert(seed, data, args, 4, device)
    ode = pc.load_ode(seed, data, args, device)
    adjacency = torch.tensor(data["graph_priors"][args.fixed_graph_type], dtype=torch.float32, device=device)
    splits = {}
    for name in ("val", "test"):
        cache = project_path(args.output_dir) / f"seed_{seed}_{name}_frozen_features.npz"
        if args.resume and cache.exists():
            with np.load(cache) as payload:
                splits[name] = {key: payload[key] for key in payload.files}
        else:
            dataset = data[f"multi_{name}"]
            splits[name] = pc.collect_split(dataset, eta_model, multi_model, ode, adjacency, args, device, seed)
            np.savez_compressed(cache, **splits[name])
    return data, splits


def run_seed(seed: int, args: argparse.Namespace, device: torch.device):
    data, splits = collect_seed(seed, args, device)
    val, test = splits["val"], splits["test"]
    predictions = {
        "gwn_eta_only": test["eta"],
        "gwn_multistate_no_physics": test["multi"],
        "hs_dt_gwn": fixed_hsdt(test),
    }
    metadata_rows = []
    for name, use_physics in (
        ("validation_locked_ridge_no_physics", False),
        ("physics_reliability_hsdt_gwn", True),
    ):
        pred, metadata = fit_candidate(val, test, use_physics, args)
        predictions[name] = pred
        metadata_rows.append({"seed": seed, "model": name, **metadata})
        np.savez_compressed(
            project_path(args.output_dir) / f"seed_{seed}_{name}_predictions.npz",
            pred_residual=pred,
            true_residual=test["target"],
            target_tide=test["tide"],
        )

    train_end = pc.rolling.time_index(data["arrays"]["time"], args.fold_train_end)
    thresholds = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    rows = []
    for name, prediction in predictions.items():
        rows.append({"seed": seed, "model": name, **pc.score(test["target"], prediction, test["tide"], thresholds)})
    return rows, metadata_rows


def summarize(args: argparse.Namespace, rows: list[dict], metadata: list[dict]) -> None:
    out = project_path(args.output_dir)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "all_runs.csv", index=False)
    pd.DataFrame(metadata).to_csv(out / "selected_calibrators.csv", index=False)
    metrics = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "event_PR_AUC",
        "event_CSI",
    ]
    summary = runs.groupby("model")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(v) for v in col if v) for col in summary.columns.to_flat_index()]
    summary.to_csv(out / "mean_std.csv", index=False)
    pivot = runs.pivot(index="seed", columns="model", values=metrics)
    comparisons = []
    candidate = "physics_reliability_hsdt_gwn"
    for baseline in ("hs_dt_gwn", "validation_locked_ridge_no_physics"):
        for metric in metrics:
            delta = pivot[(metric, candidate)] - pivot[(metric, baseline)]
            improvement = -delta if metric.endswith(("RMSE", "MAE")) else delta
            comparisons.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "mean_improvement": float(improvement.mean()),
                    "std_improvement": float(improvement.std(ddof=1)),
                    "wins": int((improvement > 0).sum()),
                    "count": int(len(improvement)),
                }
            )
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-locked physics-reliability HS-DT-GWN")
    parser.add_argument("--output-dir", default="results/validation_locked_physics_reliability_gwn")
    parser.add_argument("--source-results", default="results/confirmatory_hsdt_orc_refit_2025_h2")
    parser.add_argument("--data-dir", default="data/processed_multiyear_2023_2025")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--fold-train-end", default="2025-01-01")
    parser.add_argument("--fold-val-end", default="2025-07-01")
    parser.add_argument("--fold-test-end", default="2026-01-01")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", choices=["identity", "distance", "corr"], default="distance")
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
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
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.gate_hidden = 64
    args.gate_epochs = 1
    args.gate_patience = 1
    args.gate_lr = 1e-3
    args.weight_decay = 1e-5
    args.gate_smooth_weight = 0.0
    args.correction_reg_weight = 0.0
    args.physics_correction_reg_weight = 0.0
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
        json.dumps(
            {
                **vars(args),
                "data_dir": str(data_dir),
                "calibrator_fit_period": "first half of validation",
                "hyperparameter_selection_period": "second half of validation",
                "test_used_for_selection": False,
                "future_residual_used_as_input": False,
                "untouched_holdout": False,
                "posthoc_after_unprotected_screen": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, metadata = [], []
    for seed in args.seeds:
        print(f"seed={seed} device={device}", flush=True)
        seed_rows, seed_metadata = run_seed(seed, args, device)
        rows.extend(seed_rows)
        metadata.extend(seed_metadata)
    summarize(args, rows, metadata)


if __name__ == "__main__":
    main()
