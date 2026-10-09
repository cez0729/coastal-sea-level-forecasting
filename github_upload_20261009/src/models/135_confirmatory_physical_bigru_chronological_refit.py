from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 2024, 2025, 3407]
CONFIGS = ["gnn_bigru_no_physics", "gnn_bigru_physics"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


confirm = load_module(
    "physical_refit_confirmatory_impl",
    HERE / "134_confirmatory_hsdt_orc_chronological_refit.py",
)
p104 = confirm.p104
rolling = confirm.rolling
final4 = confirm.final4
v2 = confirm.v2
v4 = final4.v4


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def metadata(seed: int, config: str, args) -> dict:
    return {
        "seed": int(seed),
        "config": config,
        "train_end_exclusive": args.fold_train_end,
        "validation_end_exclusive": args.fold_val_end,
        "test_end_exclusive": args.fold_test_end,
        "validation_label": "end_to_end_chronological_refit_not_untouched",
        "strict_causal_preprocessing": True,
        "future_residual_used_as_input": False,
        "physics_forcing_mode": "last_input",
        "physics_lambda_max": (
            float(args.physics_lambda) if config == "gnn_bigru_physics" else 0.0
        ),
        "bigru_state_extraction": "cat_top_layer_forward_backward_h_n",
        "paired_initialization_within_seed": True,
    }


def make_model(data: dict, args, device: torch.device):
    model = v2.MultistateGNNBiGRU(
        input_dim=data["feats"],
        graph_priors=data["graph_priors"],
        graph_init_weights=[0.50, 0.35, 0.15],
        gnn_hidden=args.gnn_hidden,
        gru_hidden=args.gru_hidden,
        horizon=args.horizon,
        dropout=args.dropout,
        num_states=len(v2.STATE_NAMES),
    ).to(device)
    physics_ode = v2.MultistatePhysicsODE(
        data["nodes"], len(data["physics_cols"]), len(v2.STATE_NAMES)
    ).to(device)
    return model, physics_ode


def checkpoint_payload(model, physics_ode, data: dict, run_metadata: dict) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "physics_ode_state_dict": physics_ode.state_dict(),
        "metadata": run_metadata,
        "feature_cols": data["feature_cols"],
        "physics_cols": data["physics_cols"],
        "graph_priors": data["graph_priors"],
        "state_scale": data["state_scale"],
        "delta_scale": data["delta_scale"],
        "x_scaler_state": data["x_scaler_state"],
        "physics_scaler_state": data["physics_scaler_state"],
    }


def train_one(seed: int, config_name: str, data: dict, args, device: torch.device):
    run_dir = project_path(args.output_dir) / f"seed_{seed}" / config_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    predictions_path = run_dir / "predictions.npz"
    if args.resume and metrics_path.exists() and predictions_path.exists():
        return pd.read_csv(metrics_path).iloc[0].to_dict()

    # Reset before both paired configurations so their stochastic initialization
    # and shuffled batch order are identical within each seed.
    p104.set_reproducible(seed, args.cpu_threads)
    model, physics_ode = make_model(data, args, device)
    train_loader = p104.make_loader(data["multi_train"], args, True, seed)
    val_loader = p104.make_loader(data["multi_val"], args, False, seed)
    test_loader = p104.make_loader(data["multi_test"], args, False, seed)
    lambda_max = args.physics_lambda if config_name == "gnn_bigru_physics" else 0.0
    train_config = {
        "config_name": config_name,
        "physics_lambda_max": float(lambda_max),
        "physics_state_weights": [1.0, 0.35, 0.35, 0.25],
        "physics_lead_gamma": 0.0,
        "data_lead_gamma": 0.0,
        "extreme_alpha": 0.0,
        "extreme_quantile": args.extreme_quantile,
        "last_step_weight": args.last_step_weight,
    }

    started = time.perf_counter()
    history, best_val = v4.train_weighted_model(
        model,
        physics_ode,
        train_loader,
        val_loader,
        args,
        train_config,
        data["state_scale"],
        data["delta_scale"],
        data["train_abs_eta_threshold"],
        args.horizon,
        device,
    )
    training_seconds = time.perf_counter() - started
    pred_states, true_states, tide, test_physics_loss = v2.predict(
        model, physics_ode, test_loader, device
    )
    pred = pred_states[..., 0]
    true = true_states[..., 0]
    thresholds = np.quantile(
        data["arrays"]["residual"][: rolling.time_index(data["arrays"]["time"], args.fold_train_end)],
        args.event_quantile,
        axis=0,
    )
    row = {
        **metadata(seed, config_name, args),
        "best_val_eta_data_loss": float(best_val),
        "training_seconds": float(training_seconds),
        "test_physics_loss_unscaled": float(test_physics_loss),
        **confirm.score(true, pred, tide, thresholds),
    }
    history.to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    target_indices = np.asarray(data["multi_test"].indices, dtype=np.int64)
    target_times = pd.to_datetime(data["arrays"]["time"])[target_indices].to_numpy(
        dtype="datetime64[ns]"
    )
    np.savez_compressed(
        predictions_path,
        pred_residual=pred,
        true_residual=true,
        target_tide=tide,
        target_origin_time=target_times,
        station_ids=np.asarray(v2.STATION_IDS),
    )
    torch.save(
        checkpoint_payload(model, physics_ode, data, metadata(seed, config_name, args)),
        run_dir / "best_checkpoint.pt",
    )
    print(
        f"seed={seed} config={config_name} seq={row['seq_residual_R2']:.6f} "
        f"lead24={row['last_residual_R2']:.6f} q95={row['extreme_abs_q95_residual_R2']:.6f}"
    )
    return row


def load_bundles(output_dir: Path, seeds: list[int]) -> list[dict]:
    bundles = []
    reference_true = None
    reference_times = None
    for seed in seeds:
        bundle = {"seed": seed}
        for config_name in CONFIGS:
            path = output_dir / f"seed_{seed}" / config_name / "predictions.npz"
            with np.load(path, allow_pickle=False) as payload:
                bundle[config_name] = payload["pred_residual"]
                if "true_residual" not in bundle:
                    bundle["true_residual"] = payload["true_residual"]
                    bundle["target_tide"] = payload["target_tide"]
                    bundle["target_origin_time"] = payload["target_origin_time"]
                    bundle["station_ids"] = payload["station_ids"]
                else:
                    paired_fields = (
                        "true_residual",
                        "target_tide",
                        "target_origin_time",
                        "station_ids",
                    )
                    mismatched = [
                        field
                        for field in paired_fields
                        if not np.array_equal(bundle[field], payload[field])
                    ]
                    if mismatched:
                        raise RuntimeError(
                            f"Paired prediction files differ at seed {seed}: {mismatched}"
                        )
        if reference_true is None:
            reference_true = bundle["true_residual"]
            reference_times = bundle["target_origin_time"]
        else:
            if not np.array_equal(reference_true, bundle["true_residual"]):
                raise RuntimeError(f"Targets differ across seeds at seed {seed}")
            if not np.array_equal(reference_times, bundle["target_origin_time"]):
                raise RuntimeError(f"Target times differ across seeds at seed {seed}")
        bundles.append(bundle)
    return bundles


def exact_paired(all_runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "event_CSI",
        "event_PR_AUC",
    ]
    pivot = all_runs.pivot(index="seed", columns="config", values=metrics)
    rows = []
    for metric in metrics:
        delta = pivot[(metric, "gnn_bigru_physics")] - pivot[
            (metric, "gnn_bigru_no_physics")
        ]
        improvement = -delta if metric.endswith(("RMSE", "MAE")) else delta
        rows.append(
            {
                "comparison": "physics_minus_no_physics",
                "metric": metric,
                "mean_improvement": float(improvement.mean()),
                "std_improvement": float(improvement.std(ddof=1)),
                "wins": int((improvement > 0).sum()),
                "count": int(improvement.notna().sum()),
                "wilcoxon_greater_p": p104.exact_wilcoxon_greater(improvement.to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def block_bootstrap(bundles: list[dict], args) -> pd.DataFrame:
    rng = np.random.default_rng(20260729)
    rows = []
    for metric, horizon_slice in [("sequence_R2", slice(None)), ("lead24_R2", -1)]:
        true = bundles[0]["true_residual"][..., horizon_slice].astype(np.float64)
        sample_count = true.shape[0]
        flattened = true.reshape(sample_count, -1)
        per_sample_sum = flattened.sum(axis=1)
        per_sample_sum_sq = (flattened**2).sum(axis=1)
        values_per_sample = flattened.shape[1]
        error_advantage = []
        for bundle in bundles:
            candidate_error = (
                bundle["gnn_bigru_physics"][..., horizon_slice].astype(np.float64) - true
            ).reshape(sample_count, -1)
            baseline_error = (
                bundle["gnn_bigru_no_physics"][..., horizon_slice].astype(np.float64) - true
            ).reshape(sample_count, -1)
            error_advantage.append(
                (baseline_error**2).sum(axis=1) - (candidate_error**2).sum(axis=1)
            )
        error_advantage = np.stack(error_advantage, axis=0)
        full_sst = float(
            per_sample_sum_sq.sum()
            - per_sample_sum.sum() ** 2 / (sample_count * values_per_sample)
        )
        point_deltas = error_advantage.sum(axis=1) / full_sst
        replicates = np.empty(args.bootstrap_replicates, dtype=np.float64)
        for replicate in range(args.bootstrap_replicates):
            indices = confirm.moving_block_indices(
                rng, sample_count, args.bootstrap_block_hours
            )
            sampled_seeds = rng.integers(0, len(bundles), size=len(bundles))
            total_sum = float(per_sample_sum[indices].sum())
            total_sum_sq = float(per_sample_sum_sq[indices].sum())
            sst = total_sum_sq - total_sum**2 / (len(indices) * values_per_sample)
            seed_deltas = error_advantage[sampled_seeds][:, indices].sum(axis=1) / sst
            replicates[replicate] = float(seed_deltas.mean())
        rows.append(
            {
                "comparison": "physics_minus_no_physics",
                "metric": metric,
                "point_mean_delta": float(point_deltas.mean()),
                "ci95_low": float(np.quantile(replicates, 0.025)),
                "ci95_high": float(np.quantile(replicates, 0.975)),
                "bootstrap_probability_positive": float(np.mean(replicates > 0.0)),
                "block_hours": int(args.bootstrap_block_hours),
                "replicates": int(args.bootstrap_replicates),
            }
        )
    return pd.DataFrame(rows)


def diagnostics(bundles: list[dict], output_dir: Path) -> None:
    lead_rows = []
    station_rows = []
    for bundle in bundles:
        true = bundle["true_residual"].astype(np.float64)
        for config_name in CONFIGS:
            pred = bundle[config_name].astype(np.float64)
            for lead in range(true.shape[-1]):
                error = pred[..., lead] - true[..., lead]
                lead_rows.append(
                    {
                        "seed": bundle["seed"],
                        "config": config_name,
                        "lead_hour": lead + 1,
                        "residual_R2": confirm.r2_score(true[..., lead], pred[..., lead]),
                        "residual_RMSE": float(np.sqrt(np.mean(error**2))),
                        "residual_MAE": float(np.mean(np.abs(error))),
                    }
                )
            for index, station_id in enumerate(bundle["station_ids"].tolist()):
                error = pred[:, index, -1] - true[:, index, -1]
                station_rows.append(
                    {
                        "seed": bundle["seed"],
                        "config": config_name,
                        "station_id": str(station_id),
                        "sequence_residual_R2": confirm.r2_score(
                            true[:, index, :], pred[:, index, :]
                        ),
                        "lead24_residual_R2": confirm.r2_score(
                            true[:, index, -1], pred[:, index, -1]
                        ),
                        "lead24_residual_RMSE": float(np.sqrt(np.mean(error**2))),
                    }
                )
    leads = pd.DataFrame(lead_rows)
    stations = pd.DataFrame(station_rows)
    leads.to_csv(output_dir / "per_lead_by_seed.csv", index=False)
    stations.to_csv(output_dir / "per_station_by_seed.csv", index=False)
    leads.groupby(["config", "lead_hour"])[
        ["residual_R2", "residual_RMSE", "residual_MAE"]
    ].agg(["mean", "std", "count"]).to_csv(output_dir / "per_lead_mean_std.csv")
    stations.groupby(["config", "station_id"])[
        ["sequence_residual_R2", "lead24_residual_R2", "lead24_residual_RMSE"]
    ].agg(["mean", "std", "count"]).to_csv(output_dir / "per_station_mean_std.csv")


def write_report(summary: pd.DataFrame, paired: pd.DataFrame, bootstrap: pd.DataFrame, output_dir: Path) -> None:
    lookup = summary.set_index("config")
    seq = paired.set_index("metric").loc["seq_residual_R2"]
    lead24 = paired.set_index("metric").loc["last_residual_R2"]
    seq_boot = bootstrap.set_index("metric").loc["sequence_R2"]
    lead_boot = bootstrap.set_index("metric").loc["lead24_R2"]
    lines = [
        "# Strict-causal Physical-loss GNN-BiGRU confirmatory refit",
        "",
        "This is an end-to-end chronological refit backtest, not an untouched holdout.",
        "Train: 2023-01-01 to 2024-12-31; validation: 2025 H1; test: 2025 H2.",
        "All preprocessing statistics and graph correlation priors are fit on training data only.",
        "",
        "| Model | Sequence residual R2 | Lead-24 residual R2 | Lead-24 RMSE (m) | Descriptive q95 R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "gnn_bigru_no_physics": "Multistate GNN-BiGRU, no physics",
        "gnn_bigru_physics": "Physical-loss GNN-BiGRU, lambda=0.0002",
    }
    for config_name in CONFIGS:
        row = lookup.loc[config_name]
        lines.append(
            f"| {labels[config_name]} | {row['seq_residual_R2_mean']:.6f} +/- {row['seq_residual_R2_std']:.6f} "
            f"| {row['last_residual_R2_mean']:.6f} +/- {row['last_residual_R2_std']:.6f} "
            f"| {row['last_residual_RMSE_mean']:.6f} +/- {row['last_residual_RMSE_std']:.6f} "
            f"| {row['extreme_abs_q95_residual_R2_mean']:.6f} +/- {row['extreme_abs_q95_residual_R2_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired interpretation",
            "",
            f"- Physics minus no physics, sequence R2: {seq['mean_improvement']:+.6f}; "
            f"wins {int(seq['wins'])}/5; one-sided exact Wilcoxon p={seq['wilcoxon_greater_p']:.5f}; "
            f"block-bootstrap 95% CI [{seq_boot['ci95_low']:.6f}, {seq_boot['ci95_high']:.6f}].",
            f"- Physics minus no physics, lead-24 R2: {lead24['mean_improvement']:+.6f}; "
            f"wins {int(lead24['wins'])}/5; one-sided exact Wilcoxon p={lead24['wilcoxon_greater_p']:.5f}; "
            f"block-bootstrap 95% CI [{lead_boot['ci95_low']:.6f}, {lead_boot['ci95_high']:.6f}].",
            "- A positive point estimate alone is not treated as evidence when seed consistency or the confidence interval does not support it.",
            "- The descriptive q95 metric uses a test-distribution subset and is not presented as an operational threshold; event metrics use training-defined station thresholds.",
        ]
    )
    (output_dir / "CONFIRMATORY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = ["No physics", "Physics loss"]
    ordered = summary.set_index("config").loc[CONFIGS]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    for axis, metric, title in [
        (axes[0], "seq_residual_R2", "Sequence R2"),
        (axes[1], "last_residual_R2", "Lead-24 R2"),
        (axes[2], "extreme_abs_q95_residual_R2", "Descriptive q95 R2"),
    ]:
        axis.bar(labels, ordered[f"{metric}_mean"], yerr=ordered[f"{metric}_std"], capsize=4)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=12)
    fig.suptitle("Strict-causal 2025 H2 chronological refit")
    fig.tight_layout()
    fig.savefig(output_dir / "physical_bigru_refit_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def merge(args) -> None:
    output_dir = project_path(args.output_dir)
    metric_files = [
        output_dir / f"seed_{seed}" / config_name / "metrics.csv"
        for seed in args.seeds
        for config_name in CONFIGS
    ]
    missing = [str(path) for path in metric_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing confirmatory runs:\n" + "\n".join(missing))
    all_runs = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    all_runs.to_csv(output_dir / "all_runs.csv", index=False)
    metric_cols = [
        "seq_residual_R2",
        "seq_residual_RMSE",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "event_CSI",
        "event_PR_AUC",
    ]
    summary = all_runs.groupby("config")[metric_cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "mean_std.csv", index=False)
    paired = exact_paired(all_runs)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    bundles = load_bundles(output_dir, args.seeds)
    bootstrap = block_bootstrap(bundles, args)
    bootstrap.to_csv(output_dir / "block_bootstrap_comparisons.csv", index=False)
    diagnostics(bundles, output_dir)
    write_report(summary, paired, bootstrap, output_dir)
    plot_summary(summary, output_dir)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(bootstrap.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired strict-causal refit of GNN-BiGRU with and without physics loss."
    )
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    parser.add_argument(
        "--output-dir", default="results/strict_causal_physical_bigru_refit_2025_h2"
    )
    parser.add_argument("--data-dir", default="data/processed_multiyear_2023_2025")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--fold-train-end", default="2025-01-01")
    parser.add_argument("--fold-val-end", default="2025-07-01")
    parser.add_argument("--fold-test-end", default="2026-01-01")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--gnn-hidden", type=int, default=40)
    parser.add_argument("--gru-hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--graph-lr-mult", type=float, default=3.0)
    parser.add_argument("--physics-lr-mult", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--physics-warmup-epochs", type=int, default=8)
    parser.add_argument("--physics-ramp-epochs", type=int, default=14)
    parser.add_argument("--physics-loss-type", choices=["huber", "mse"], default="huber")
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    parser.add_argument("--physics-lambda", type=float, default=0.0002)
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.2)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument(
        "--selection-metric",
        choices=["val_eta_data_loss", "val_data_loss", "val_last_loss", "val_total_loss"],
        default="val_eta_data_loss",
    )
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--bootstrap-block-hours", type=int, default=168)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-merge", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = confirm.configure_data_dir(args.data_dir)
    run_config = {
        **vars(args),
        "data_dir": str(data_dir.relative_to(ROOT) if data_dir.is_relative_to(ROOT) else data_dir),
        "validation_label": "end_to_end_chronological_refit_not_untouched",
        "bigru_state_extraction": "cat_top_layer_forward_backward_h_n",
        "paired_initialization_within_seed": True,
    }
    config_name = "final_experiment_config.json" if args.mode == "merge" else "experiment_config.json"
    (output_dir / config_name).write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    if args.mode == "merge":
        merge(args)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}; seeds={args.seeds}; data={data_dir}")
    for seed in args.seeds:
        data = rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
        for config_name in CONFIGS:
            train_one(seed, config_name, data, args, device)
    if not args.skip_merge:
        merge(args)


if __name__ == "__main__":
    main()
