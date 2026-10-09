from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 2024, 2025, 3407]
CONFIGS = [
    "gnn_bigru_no_physics",
    "gnn_bigru_physics",
    "gwn_eta_only",
    "gwn_multistate_no_physics",
    "hs_dt_gwn",
]
COMPARISONS = [
    ("gwn_eta_minus_bigru_no_physics", "gwn_eta_only", "gnn_bigru_no_physics"),
    (
        "gwn_multistate_minus_bigru_no_physics",
        "gwn_multistate_no_physics",
        "gnn_bigru_no_physics",
    ),
    ("hsdt_minus_bigru_physics", "hs_dt_gwn", "gnn_bigru_physics"),
    (
        "physics_minus_no_physics_bigru",
        "gnn_bigru_physics",
        "gnn_bigru_no_physics",
    ),
    ("hsdt_minus_eta", "hs_dt_gwn", "gwn_eta_only"),
    ("hsdt_minus_multistate", "hs_dt_gwn", "gwn_multistate_no_physics"),
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


confirm = load_module(
    "submission_strict_confirmatory_impl",
    HERE / "134_confirmatory_hsdt_orc_chronological_refit.py",
)
p104 = confirm.p104


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_runs(gwn_dir: Path, physical_dir: Path) -> pd.DataFrame:
    gwn = pd.read_csv(gwn_dir / "all_runs.csv")
    gwn = gwn[gwn["config"].isin(CONFIGS)].copy()
    physical = pd.read_csv(physical_dir / "all_runs.csv")
    combined = pd.concat([physical, gwn], ignore_index=True, sort=False)
    counts = combined.groupby("config")["seed"].nunique()
    missing = [name for name in CONFIGS if int(counts.get(name, 0)) != len(SEEDS)]
    if missing:
        raise RuntimeError(f"Expected five seeds for each configuration; incomplete: {missing}")
    if combined.duplicated(["seed", "config"]).any():
        raise RuntimeError("Duplicate seed/config rows in combined evidence")
    return combined


def load_bundles(gwn_dir: Path, physical_dir: Path) -> list[dict]:
    bundles = []
    reference_true = None
    reference_times = None
    for seed in SEEDS:
        with np.load(gwn_dir / f"seed_{seed}" / "predictions.npz", allow_pickle=False) as payload:
            bundle = {
                "seed": seed,
                "gwn_eta_only": payload["gwn_eta_only"],
                "gwn_multistate_no_physics": payload["gwn_multistate_no_physics"],
                "hs_dt_gwn": payload["hs_dt_gwn"],
                "true_residual": payload["true_residual"],
                "target_origin_time": payload["target_origin_time"],
                "station_ids": payload["station_ids"],
            }
        for config_name in ("gnn_bigru_no_physics", "gnn_bigru_physics"):
            path = physical_dir / f"seed_{seed}" / config_name / "predictions.npz"
            with np.load(path, allow_pickle=False) as payload:
                if not np.array_equal(bundle["true_residual"], payload["true_residual"]):
                    raise RuntimeError(f"GWN/GNN-BiGRU targets differ at seed {seed}")
                if not np.array_equal(bundle["target_origin_time"], payload["target_origin_time"]):
                    raise RuntimeError(f"GWN/GNN-BiGRU target times differ at seed {seed}")
                bundle[config_name] = payload["pred_residual"]
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


def paired_metrics(all_runs: pd.DataFrame) -> pd.DataFrame:
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
    for comparison, candidate, baseline in COMPARISONS:
        for metric in metrics:
            delta = pivot[(metric, candidate)] - pivot[(metric, baseline)]
            improvement = -delta if metric.endswith(("RMSE", "MAE")) else delta
            rows.append(
                {
                    "comparison": comparison,
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "mean_improvement": float(improvement.mean()),
                    "std_improvement": float(improvement.std(ddof=1)),
                    "wins": int((improvement > 0).sum()),
                    "count": int(improvement.notna().sum()),
                    "wilcoxon_greater_p": p104.exact_wilcoxon_greater(
                        improvement.to_numpy()
                    ),
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
        for comparison, candidate, baseline in COMPARISONS:
            advantages = []
            for bundle in bundles:
                candidate_error = (
                    bundle[candidate][..., horizon_slice].astype(np.float64) - true
                ).reshape(sample_count, -1)
                baseline_error = (
                    bundle[baseline][..., horizon_slice].astype(np.float64) - true
                ).reshape(sample_count, -1)
                advantages.append(
                    (baseline_error**2).sum(axis=1) - (candidate_error**2).sum(axis=1)
                )
            advantages = np.stack(advantages, axis=0)
            full_sst = float(
                per_sample_sum_sq.sum()
                - per_sample_sum.sum() ** 2 / (sample_count * values_per_sample)
            )
            point_deltas = advantages.sum(axis=1) / full_sst
            replicates = np.empty(args.bootstrap_replicates, dtype=np.float64)
            for replicate in range(args.bootstrap_replicates):
                indices = confirm.moving_block_indices(
                    rng, sample_count, args.bootstrap_block_hours
                )
                sampled_seeds = rng.integers(0, len(bundles), size=len(bundles))
                total_sum = float(per_sample_sum[indices].sum())
                total_sum_sq = float(per_sample_sum_sq[indices].sum())
                sst = total_sum_sq - total_sum**2 / (len(indices) * values_per_sample)
                seed_deltas = advantages[sampled_seeds][:, indices].sum(axis=1) / sst
                replicates[replicate] = float(seed_deltas.mean())
            rows.append(
                {
                    "comparison": comparison,
                    "candidate": candidate,
                    "baseline": baseline,
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


def plot_submission(summary: pd.DataFrame, bootstrap: pd.DataFrame, output_dir: Path) -> None:
    labels = {
        "gnn_bigru_no_physics": "BiGRU\nno physics",
        "gnn_bigru_physics": "BiGRU\n+ physics",
        "gwn_eta_only": "FS-GWN\neta",
        "gwn_multistate_no_physics": "FS-GWN\nmultistate",
        "hs_dt_gwn": "HS-DT\nGWN",
    }
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F2CF5B", "#59A14F"]
    ordered = summary.set_index("config").loc[CONFIGS]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6))
    x = np.arange(len(CONFIGS))
    for axis, metric, title in [
        (axes[0], "seq_residual_R2", "Complete trajectory"),
        (axes[1], "last_residual_R2", "Lead 24"),
    ]:
        axis.bar(
            x,
            ordered[f"{metric}_mean"],
            yerr=ordered[f"{metric}_std"],
            color=colors,
            capsize=3,
            edgecolor="white",
            linewidth=0.7,
        )
        axis.set_xticks(x, [labels[name] for name in CONFIGS], fontsize=8.5)
        axis.set_ylabel(r"Residual $R^2$")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    selected = [
        ("hsdt_minus_eta", "sequence_R2", "HS-DT - eta\ntrajectory"),
        ("hsdt_minus_multistate", "sequence_R2", "HS-DT - multi\ntrajectory"),
        (
            "physics_minus_no_physics_bigru",
            "sequence_R2",
            "BiGRU phys. - none\ntrajectory",
        ),
        (
            "physics_minus_no_physics_bigru",
            "lead24_R2",
            "BiGRU phys. - none\nlead 24",
        ),
    ]
    lookup = bootstrap.set_index(["comparison", "metric"])
    values = np.asarray([lookup.loc[(name, metric), "point_mean_delta"] for name, metric, _ in selected])
    lows = np.asarray([lookup.loc[(name, metric), "ci95_low"] for name, metric, _ in selected])
    highs = np.asarray([lookup.loc[(name, metric), "ci95_high"] for name, metric, _ in selected])
    y = np.arange(len(selected))
    axes[2].errorbar(
        values,
        y,
        xerr=np.vstack([values - lows, highs - values]),
        fmt="o",
        color="#333333",
        ecolor="#777777",
        capsize=3,
    )
    axes[2].axvline(0.0, color="#B22222", linestyle="--", linewidth=1.0)
    axes[2].set_yticks(y, [label for _, _, label in selected])
    axes[2].invert_yaxis()
    axes[2].set_xlabel(r"Paired $\Delta R^2$ (95% block-bootstrap CI)")
    axes[2].set_title("Within-protocol effects")
    axes[2].grid(axis="x", alpha=0.25)
    axes[2].set_axisbelow(True)
    fig.suptitle("Strict-causal 2025 H2 chronological refit backtest", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "strict_submission_evidence.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge strict refit evidence for submission.")
    parser.add_argument(
        "--gwn-dir", default="results/confirmatory_hsdt_orc_refit_2025_h2"
    )
    parser.add_argument(
        "--physical-dir", default="results/strict_causal_physical_bigru_refit_2025_h2"
    )
    parser.add_argument(
        "--output-dir", default="results/submission_strict_combined_evidence"
    )
    parser.add_argument("--bootstrap-block-hours", type=int, default=168)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    gwn_dir = project_path(args.gwn_dir)
    physical_dir = project_path(args.physical_dir)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_runs = read_runs(gwn_dir, physical_dir)
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
    paired = paired_metrics(all_runs)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    bundles = load_bundles(gwn_dir, physical_dir)
    bootstrap = block_bootstrap(bundles, args)
    bootstrap.to_csv(output_dir / "block_bootstrap_comparisons.csv", index=False)
    plot_submission(summary, bootstrap, output_dir)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
