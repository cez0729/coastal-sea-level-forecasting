from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT78 = Path(__file__).resolve().parent / "78_final_four_models_enhanced_data.py"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "final_validation_tasks"


def run_command(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_experiment(output_dir: Path, horizons: list[int], models: list[str], seed: int, args) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT78),
        "--horizons",
        *[str(h) for h in horizons],
        "--models",
        *models,
        "--train-stride",
        str(args.train_stride),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--gnn-hidden",
        str(args.gnn_hidden),
        "--gru-hidden",
        str(args.gru_hidden),
        "--physics-lambda-max",
        str(args.physics_lambda_max),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    run_command(cmd, ROOT)
    return output_dir / "final_four_models_enhanced_metrics.csv"


def read_metrics(paths: list[Path], label_prefix: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            df["run_label"] = label_prefix
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_task_tables(all_metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(output_dir / "all_validation_metrics_long.csv", index=False)

    key_cols = [
        "seed",
        "horizon",
        "model_name",
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q90_residual_R2",
        "extreme_abs_q95_residual_R2",
        "extreme_abs_q95_residual_RMSE",
        "seq_sea_level_R2",
        "last_sea_level_R2",
        "learned_w_identity",
        "learned_w_distance",
        "learned_w_corr",
    ]
    existing = [c for c in key_cols if c in all_metrics.columns]
    all_metrics[existing].sort_values(["horizon", "model_name", "seed"]).to_csv(
        output_dir / "task_2_horizon_model_key_metrics.csv", index=False
    )

    agg_cols = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q90_residual_R2",
        "extreme_abs_q95_residual_R2",
        "extreme_abs_q95_residual_RMSE",
    ]
    agg_cols = [c for c in agg_cols if c in all_metrics.columns]
    if "seed" in all_metrics.columns and all_metrics["seed"].nunique() > 1:
        grouped = all_metrics.groupby(["horizon", "model_name"])[agg_cols].agg(["mean", "std"]).reset_index()
        grouped.columns = ["_".join([x for x in col if x]) for col in grouped.columns.to_flat_index()]
        grouped.to_csv(output_dir / "task_1_multiseed_mean_std.csv", index=False)
    else:
        all_metrics[existing].to_csv(output_dir / "task_1_single_seed_metrics.csv", index=False)

    extreme_cols = [
        "seed",
        "horizon",
        "model_name",
        "extreme_abs_q90_residual_RMSE",
        "extreme_abs_q90_residual_R2",
        "extreme_abs_q95_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "extreme_abs_q95_n",
    ]
    existing_extreme = [c for c in extreme_cols if c in all_metrics.columns]
    all_metrics[existing_extreme].sort_values(["horizon", "model_name", "seed"]).to_csv(
        output_dir / "task_3_extreme_event_metrics.csv", index=False
    )


def plot_horizon_summary(all_metrics: pd.DataFrame, output_dir: Path) -> None:
    for metric, filename, ylabel in [
        ("last_residual_R2", "task_2_last_residual_R2_by_horizon.png", "Last-step residual R2"),
        ("extreme_abs_q95_residual_R2", "task_3_extreme_q95_R2_by_horizon.png", "Extreme top 5% residual R2"),
    ]:
        if metric not in all_metrics.columns:
            continue
        plot_df = all_metrics.groupby(["horizon", "model_name"], as_index=False)[metric].mean()
        fig, ax = plt.subplots(figsize=(10, 5.2))
        for model, sub in plot_df.groupby("model_name"):
            sub = sub.sort_values("horizon")
            ax.plot(sub["horizon"], sub[metric], marker="o", label=model)
        ax.set_xlabel("Horizon (hours)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final validation tasks: multiseed, horizons, extremes")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--task2-horizons", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--task2-seed", type=int, default=42)
    parser.add_argument("--task1-horizon", type=int, default=24)
    parser.add_argument("--task1-seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--task1-models", nargs="+", default=["gnn_bigru", "physical_loss"])
    parser.add_argument("--models", nargs="+", default=["gnn_bigru", "learnable_graph", "ode_based_learnable", "physical_loss"])
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gnn-hidden", type=int, default=40)
    parser.add_argument("--gru-hidden", type=int, default=48)
    parser.add_argument("--physics-lambda-max", type=float, default=0.0003)
    parser.add_argument("--skip-task1", action="store_true")
    parser.add_argument("--skip-task2", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    if not args.skip_task2:
        task2_dir = output_dir / f"task2_all_models_horizons_seed{args.task2_seed}"
        path = run_experiment(task2_dir, args.task2_horizons, args.models, args.task2_seed, args)
        df = pd.read_csv(path)
        df["seed"] = args.task2_seed
        df["validation_task"] = "task2_all_models_6_12_24"
        df.to_csv(task2_dir / "metrics_with_seed.csv", index=False)
        paths.append(task2_dir / "metrics_with_seed.csv")

    if not args.skip_task1:
        for seed in args.task1_seeds:
            # Reuse task2 seed=42 physical/gnn rows if possible would complicate bookkeeping;
            # run explicitly for transparent multi-seed outputs.
            task1_dir = output_dir / f"task1_multiseed_h{args.task1_horizon}_seed{seed}"
            path = run_experiment(task1_dir, [args.task1_horizon], args.task1_models, seed, args)
            df = pd.read_csv(path)
            df["seed"] = seed
            df["validation_task"] = "task1_multiseed"
            df.to_csv(task1_dir / "metrics_with_seed.csv", index=False)
            paths.append(task1_dir / "metrics_with_seed.csv")

    frames = [pd.read_csv(path) for path in paths if path.exists()]
    all_metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if all_metrics.empty:
        raise RuntimeError("No metrics were produced.")
    make_task_tables(all_metrics, output_dir)
    plot_horizon_summary(all_metrics, output_dir)
    print("\nFinal validation tasks finished.")
    key = [
        "validation_task",
        "seed",
        "horizon",
        "model_name",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
    ]
    existing = [c for c in key if c in all_metrics.columns]
    print(all_metrics[existing].sort_values(["validation_task", "horizon", "model_name", "seed"]).to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
