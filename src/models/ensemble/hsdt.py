from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = REPO_ROOT
HERE = Path(__file__).resolve().parent
SOURCE_RESULTS = ROOT / "results" / "priority12_physics_graph_wavenet"
DEFAULT_OUT = ROOT / "results" / "horizon_specialized_dual_task_gwn"
SEEDS = [42, 123, 2024, 2025, 3407]
CONFIGS = [
    "gwn_eta_only",
    "gwn_multistate_no_physics",
    "dual_task_equal_ensemble",
    "horizon_specialized_dual_task_gwn",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p104 = load_module("priority2_gwn_p108", REPO_ROOT / 'src/training/train_graph_experts.py')
final4 = p104.final4
priority1 = p104.priority1
v2 = p104.v2


def fusion_weights(horizon: int, mode: str) -> np.ndarray:
    """Return the fixed horizon rule used by the public HS-DT comparison."""
    if mode == "dual_task_equal_ensemble":
        return np.full(horizon, 0.5, dtype=np.float64)
    if mode == "horizon_specialized_dual_task_gwn":
        weights = np.full(horizon, 0.5, dtype=np.float64)
        weights[-1] = 1.0
        return weights
    raise ValueError(f"No fusion weights for {mode}")


def load_seed_predictions(seed: int, args):
    root = Path(args.source_results) / f"seed_{seed}" / f"horizon_{args.horizon}h"
    eta_file = np.load(root / "gwn_eta_only" / "predictions.npz")
    multi_file = np.load(root / "gwn_multistate_no_physics" / "predictions.npz")
    eta = eta_file["pred_residual"].astype(np.float64)
    true = eta_file["true_residual"].astype(np.float64)
    tide = eta_file["target_tide"].astype(np.float64)
    multi = multi_file["pred_states"][..., 0].astype(np.float64)
    multi_true = multi_file["true_states"][..., 0].astype(np.float64)
    if not np.allclose(true, multi_true, atol=1e-7, rtol=0.0):
        raise RuntimeError(f"Target mismatch for seed {seed}")
    return eta, multi, true, tide


def train_thresholds(args) -> np.ndarray:
    data_args = argparse.Namespace(
        window=args.window,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        train_stride=8,
        physics_forcing_mode="last_input",
        extreme_quantile=0.90,
    )
    data = final4.build_enhanced_data(data_args, args.horizon, add_ode_prior=False)
    train_end = int(len(data["arrays"]["residual"]) * args.train_ratio)
    return np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)


def summarize(true: np.ndarray, pred: np.ndarray, tide: np.ndarray, thresholds: np.ndarray) -> dict:
    metrics = final4.summarize_single(true, pred, tide)
    metrics.update(p104.operational_event_metrics(true, pred, thresholds))
    return metrics


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = train_thresholds(args)
    rows = []
    lead_rows = []
    station_rows = []
    for seed in args.seeds:
        eta, multi, true, tide = load_seed_predictions(seed, args)
        predictions = {
            "gwn_eta_only": eta,
            "gwn_multistate_no_physics": multi,
        }
        for config in CONFIGS[2:]:
            weights = fusion_weights(args.horizon, config)
            predictions[config] = eta + weights[None, None, :] * (multi - eta)
        for config, pred in predictions.items():
            metrics = summarize(true, pred, tide, thresholds)
            rows.append({"seed": seed, "config": config, **metrics})
            for lead in range(args.horizon):
                lead_metric = priority1.r2_rmse_mae(true[..., lead], pred[..., lead])
                lead_rows.append(
                    {"seed": seed, "config": config, "lead_hour": lead + 1, **lead_metric}
                )
            for station, station_id in enumerate(v2.STATION_IDS):
                station_metric = priority1.r2_rmse_mae(true[:, station, -1], pred[:, station, -1])
                station_rows.append(
                    {"seed": seed, "config": config, "station_id": station_id, **station_metric}
                )
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for config in CONFIGS[2:]:
            np.savez_compressed(
                seed_dir / f"{config}_predictions.npz",
                pred_residual=predictions[config],
                true_residual=true,
                target_tide=tide,
                multistate_weights=fusion_weights(args.horizon, config),
                station_ids=np.asarray(v2.STATION_IDS),
            )
    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(output_dir / "all_runs.csv", index=False)
    pd.DataFrame(lead_rows).to_csv(output_dir / "per_lead_metrics.csv", index=False)
    pd.DataFrame(station_rows).to_csv(output_dir / "per_station_terminal_metrics.csv", index=False)
    summarize_and_compare(all_runs, pd.DataFrame(lead_rows), args)


def summarize_and_compare(all_runs: pd.DataFrame, lead_data: pd.DataFrame, args) -> None:
    output_dir = Path(args.output_dir)
    metrics = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "event_PR_AUC",
        "event_recall",
        "event_F1",
        "event_CSI",
        "event_detect_pm6h_rate",
    ]
    metrics = [metric for metric in metrics if metric in all_runs]
    summary = all_runs.groupby("config")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(part) for part in column if part) for column in summary.columns.to_flat_index()]
    summary.to_csv(output_dir / "mean_std.csv", index=False)
    winner = "horizon_specialized_dual_task_gwn"
    paired_rows = []
    pivot = all_runs.pivot(index="seed", columns="config", values=metrics)
    for baseline in ("gwn_eta_only", "gwn_multistate_no_physics", "dual_task_equal_ensemble"):
        for metric in metrics:
            delta = pivot[(metric, winner)] - pivot[(metric, baseline)]
            if metric.endswith(("RMSE", "MAE")):
                improvement = -delta
            else:
                improvement = delta
            paired_rows.append(
                {
                    "comparison": f"{winner}_minus_{baseline}",
                    "metric": metric,
                    "mean_improvement": float(improvement.mean()),
                    "std_improvement": float(improvement.std()),
                    "wins": int((improvement > 0).sum()),
                    "count": int(improvement.notna().sum()),
                    "wilcoxon_greater_p": p104.exact_wilcoxon_greater(improvement.to_numpy()),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    plot_results(summary, lead_data, output_dir)
    write_summary(summary, paired, output_dir)
    print(summary.to_string(index=False))
    print(
        paired[
            paired["metric"].isin(
                ["seq_residual_R2", "last_residual_R2", "extreme_abs_q95_residual_R2"]
            )
        ].to_string(index=False)
    )


def plot_results(summary: pd.DataFrame, lead_data: pd.DataFrame, output_dir: Path) -> None:
    labels = {
        "gwn_eta_only": "Eta-only GWN",
        "gwn_multistate_no_physics": "Multistate GWN",
        "dual_task_equal_ensemble": "Equal dual-task ensemble",
        "horizon_specialized_dual_task_gwn": "HS-DT-GWN",
    }
    colors = ["#5B6C8F", "#26857A", "#D68C45", "#B84A4A"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    display_metrics = ["seq_residual_R2_mean", "last_residual_R2_mean", "extreme_abs_q95_residual_R2_mean"]
    x = np.arange(len(display_metrics))
    width = 0.19
    for index, config in enumerate(CONFIGS):
        row = summary[summary["config"] == config].iloc[0]
        values = [row[metric] for metric in display_metrics]
        axes[0].bar(x + (index - 1.5) * width, values, width, color=colors[index], label=labels[config])
    axes[0].set_xticks(x, ["Trajectory R2", "24-h terminal R2", "Descriptive q95 R2"])
    axes[0].set_ylabel("Residual R2")
    axes[0].set_title("Five-seed dual-task fusion comparison")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7)
    mean_lead = lead_data.groupby(["config", "lead_hour"])["R2"].mean().reset_index()
    for color, config in zip(colors, CONFIGS):
        subset = mean_lead[mean_lead["config"] == config]
        axes[1].plot(subset["lead_hour"], subset["R2"], color=color, label=labels[config])
    axes[1].set_xlabel("Forecast lead (h)")
    axes[1].set_ylabel("Residual R2")
    axes[1].set_title("Lead-dependent skill")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "horizon_specialized_dual_task_gwn_results.png", dpi=200)
    plt.close(figure)


def write_summary(summary: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> None:
    winner = summary[summary["config"] == "horizon_specialized_dual_task_gwn"].iloc[0]
    eta = summary[summary["config"] == "gwn_eta_only"].iloc[0]
    multi = summary[summary["config"] == "gwn_multistate_no_physics"].iloc[0]
    lines = [
        "# Horizon-Specialized Dual-Task Graph WaveNet",
        "",
        "The eta-only and multistate Graph WaveNet experts are averaged for leads 1-23. "
        "Lead 24 is taken from the multistate expert because that expert was trained with explicit terminal supervision.",
        "",
        "## Five-seed result",
        "",
        f"- Sequence residual R2: {winner['seq_residual_R2_mean']:.6f} "
        f"(eta-only GWN {eta['seq_residual_R2_mean']:.6f}).",
        f"- 24-h terminal residual R2: {winner['last_residual_R2_mean']:.6f} "
        f"(eta-only GWN {eta['last_residual_R2_mean']:.6f}; multistate GWN {multi['last_residual_R2_mean']:.6f}).",
        f"- Descriptive q95 residual R2: {winner['extreme_abs_q95_residual_R2_mean']:.6f} "
        f"(eta-only GWN {eta['extreme_abs_q95_residual_R2_mean']:.6f}).",
        "",
        "## Interpretation",
        "",
        "This is a deterministic mixture-of-experts extension. It adds no test-fitted parameters. "
        "Because the horizon-specialized rule was proposed after inspecting the existing benchmark, "
        "a new chronological holdout is still required for a confirmatory publication claim.",
    ]
    (output_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Horizon-specialized dual-task Graph WaveNet fusion.")
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    parser.add_argument("--source-results", default=str(SOURCE_RESULTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if args.mode == "merge":
        all_runs = pd.read_csv(output_dir / "all_runs.csv")
        lead_data = pd.read_csv(output_dir / "per_lead_metrics.csv")
        summarize_and_compare(all_runs, lead_data, args)
    else:
        run(args)


if __name__ == "__main__":
    main()
