from __future__ import annotations

"""Evaluate the frozen safety-gated dual-head model on a later period once."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 2024, 2025, 3407]
CONTROL = "validation_locked_ridge_no_physics"
PHYSICS = "physics_reliability_hsdt_gwn"
CONTINUOUS = "safety_gated_continuous_head"
DUAL = "safety_gated_physics_dual_head"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


frozen = load_module("later_frozen_evaluation", HERE / "159_evaluate_frozen_2026_holdout.py")
diagnostics = load_module("later_frozen_diagnostics", HERE / "157_evaluate_physics_reliability_gwn.py")
calibration = frozen.calibration
pc = frozen.pc


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_gate(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = np.asarray(payload["gate_matrix_station_by_lead"], dtype=bool)
    if payload.get("h1_is_development_not_confirmation") is not True:
        raise ValueError("Gate metadata does not mark H1 as development data")
    return gate


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
    mask = np.asarray(
        (forecast_times >= pd.Timestamp(args.holdout_start)) & (forecast_times < pd.Timestamp(args.holdout_end))
    )
    for key in ("eta", "multi", "prior", "forcing", "target", "tide"):
        mask &= np.isfinite(extended_test[key]).reshape(len(extended_test[key]), -1).all(axis=1)
    if not np.any(mask):
        raise RuntimeError("No complete later-period holdout windows were constructed")
    return data, val, subset(extended_test, mask), forecast_times[mask]


def combine_dual(continuous_metrics: dict[str, float], event_metrics: dict[str, float]) -> dict[str, float]:
    combined = dict(continuous_metrics)
    for name in ("event_PR_AUC", "event_CSI", "event_POD", "event_FAR"):
        if name in event_metrics:
            combined[name] = event_metrics[name]
    return combined


def run_seed(seed: int, args: argparse.Namespace, gate: np.ndarray, device: torch.device):
    data, val, holdout, times = collect_seed(seed, args, device)
    control, control_meta = calibration.fit_candidate(val, holdout, False, args)
    physics, physics_meta = calibration.fit_candidate(val, holdout, True, args)
    if gate.shape != control.shape[1:]:
        raise ValueError(f"Frozen gate shape {gate.shape} does not match predictions {control.shape[1:]}")
    continuous = np.where(gate[None, :, :], physics, control).astype(np.float32)
    hsdt = calibration.fixed_hsdt(holdout)

    train_end = pc.rolling.time_index(data["arrays"]["time"], args.fold_train_end)
    thresholds = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    basic_predictions = {
        "gwn_eta_only": holdout["eta"],
        "gwn_multistate_no_physics": holdout["multi"],
        "hs_dt_gwn": hsdt,
        CONTROL: control,
        PHYSICS: physics,
        CONTINUOUS: continuous,
    }
    scored = {name: pc.score(holdout["target"], prediction, holdout["tide"], thresholds) for name, prediction in basic_predictions.items()}
    scored[DUAL] = combine_dual(scored[CONTINUOUS], scored[PHYSICS])
    rows = [
        {
            "seed": seed,
            "model": name,
            "metric_source": "continuous=gated; event=full_physics_head" if name == DUAL else "single_output",
            **metrics,
        }
        for name, metrics in scored.items()
    ]

    out = project_path(args.output_dir)
    np.savez_compressed(
        out / f"seed_{seed}_frozen_safety_gated_dual_head_predictions.npz",
        pred_residual_continuous=continuous,
        pred_residual_event_score=physics,
        pred_residual_control=control,
        true_residual=holdout["target"],
        target_tide=holdout["tide"],
        forecast_start=times.to_numpy(dtype="datetime64[ns]"),
    )
    metadata = [
        {"seed": seed, "model": CONTROL, **control_meta},
        {"seed": seed, "model": PHYSICS, **physics_meta},
    ]
    window = {"seed": seed, "windows": int(len(times)), "start": str(times.min()), "end": str(times.max())}
    bootstrap_payload = {"true": holdout["target"], "candidate": continuous, "baseline": control}
    return rows, metadata, window, bootstrap_payload


def summarize(args: argparse.Namespace, rows: list[dict], bootstrap_payloads: list[dict]) -> None:
    out = project_path(args.output_dir)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "all_runs.csv", index=False)
    metrics = [
        "seq_residual_R2", "last_residual_R2", "last_residual_RMSE",
        "extreme_abs_q95_residual_R2", "event_PR_AUC", "event_CSI",
    ]
    summary = runs.groupby("model")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(value) for value in col if value) for col in summary.columns.to_flat_index()]
    summary.to_csv(out / "mean_std.csv", index=False)

    wide = runs.pivot(index="seed", columns="model", values=metrics)
    comparison_rows = []
    for metric in metrics:
        delta = wide[(metric, DUAL)] - wide[(metric, CONTROL)]
        improvement = -delta if metric.endswith(("RMSE", "MAE")) else delta
        try:
            pvalue = float(wilcoxon(improvement, alternative="greater", zero_method="wilcox").pvalue)
        except ValueError:
            pvalue = 1.0
        comparison_rows.append(
            {
                "candidate": DUAL,
                "baseline": CONTROL,
                "metric": metric,
                "mean_improvement": float(improvement.mean()),
                "std_improvement": float(improvement.std(ddof=1)),
                "wins": int((improvement > 0).sum()),
                "count": int(len(improvement)),
                "wilcoxon_one_sided_p": pvalue,
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_rows = []
    for mode in ("sequence", "lead24", "q95"):
        seed_stats = [
            diagnostics.time_stats(payload["true"], payload["candidate"], payload["baseline"], mode)
            for payload in bootstrap_payloads
        ]
        values = diagnostics.hierarchical_bootstrap(
            seed_stats, args.block_hours, args.bootstrap_replicates, rng
        )
        bootstrap_rows.append(
            {
                "metric": mode,
                "mean_delta_r2": float(np.mean(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "probability_positive": float(np.mean(values > 0.0)),
                "replicates": args.bootstrap_replicates,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(out / "block_bootstrap_summary.csv", index=False)

    by_metric = comparisons.set_index("metric")
    control_rmse = float(wide[("last_residual_RMSE", CONTROL)].mean())
    dual_rmse = float(wide[("last_residual_RMSE", DUAL)].mean())
    checks = {
        "sequence_positive_and_4_of_5": bool(
            by_metric.loc["seq_residual_R2", "mean_improvement"] > 0
            and by_metric.loc["seq_residual_R2", "wins"] >= 4
        ),
        "lead24_noninferior_and_rmse_within_1pct": bool(
            by_metric.loc["last_residual_R2", "mean_improvement"] >= -0.002
            and dual_rmse <= control_rmse * 1.01
        ),
        "q95_noninferior": bool(by_metric.loc["extreme_abs_q95_residual_R2", "mean_improvement"] >= -0.01),
        "event_pr_auc_positive_and_4_of_5": bool(
            by_metric.loc["event_PR_AUC", "mean_improvement"] > 0
            and by_metric.loc["event_PR_AUC", "wins"] >= 4
        ),
        "sequence_bootstrap_ci_above_zero": bool(
            bootstrap.loc[bootstrap["metric"] == "sequence", "ci95_low"].iloc[0] > 0
        ),
    }
    acceptance = {
        "criteria": checks,
        "all_main_model_criteria_passed": all(checks.values()),
        "gate_retuning_on_this_holdout_allowed": False,
    }
    (out / "acceptance_decision.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nPaired improvements over no-physics control:")
    print(comparisons.to_string(index=False))
    print("\nBootstrap:")
    print(bootstrap.to_string(index=False))
    print("\nAcceptance:")
    print(json.dumps(acceptance, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen later-period evaluation of the safety-gated dual-head GWN")
    parser.add_argument("--data-dir", default="data/processed_multiyear_2023_2026_july_frozen")
    parser.add_argument("--source-results", default="results/confirmatory_hsdt_orc_refit_2025_h2")
    parser.add_argument("--gate-json", default="results/development_2026_h1_safety_gated_dual_head_gwn/development_gate.json")
    parser.add_argument("--output-dir", default="results/frozen_2026_july_safety_gated_dual_head_gwn")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--fold-train-end", default="2025-01-01")
    parser.add_argument("--fold-val-end", default="2025-07-01")
    parser.add_argument("--fold-test-end", default="2026-08-01")
    parser.add_argument("--holdout-start", default="2026-07-01")
    parser.add_argument("--holdout-end", default="2026-08-01")
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
    parser.add_argument("--block-hours", type=int, default=168)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
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
    gate = load_gate(project_path(args.gate_json))
    (out / "experiment_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "data_dir": str(data_dir),
                "gate_selected_on_2026_h1": True,
                "later_holdout_used_for_gate_selection": False,
                "continuous_and_event_heads_scored_separately": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, metadata, windows, bootstrap_payloads = [], [], [], []
    for seed in args.seeds:
        print(f"seed={seed} device={device}", flush=True)
        seed_rows, seed_metadata, seed_window, seed_bootstrap = run_seed(seed, args, gate, device)
        rows.extend(seed_rows)
        metadata.extend(seed_metadata)
        windows.append(seed_window)
        bootstrap_payloads.append(seed_bootstrap)
    pd.DataFrame(metadata).to_csv(out / "selected_calibrators.csv", index=False)
    pd.DataFrame(windows).to_csv(out / "holdout_windows.csv", index=False)
    summarize(args, rows, bootstrap_payloads)


if __name__ == "__main__":
    main()
