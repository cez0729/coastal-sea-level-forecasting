from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


SCRIPT77 = Path(__file__).resolve().parent / "77_v4_physics_loss_weight_strategy_search.py"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "priority_top3_convincing_experiments"

spec77 = importlib.util.spec_from_file_location("v4_impl", SCRIPT77)
v4 = importlib.util.module_from_spec(spec77)
assert spec77.loader is not None
spec77.loader.exec_module(v4)
v3 = v4.v3
v2 = v4.v2


def make_config(
    family: str,
    config_name: str,
    physics_lambda_max: float,
    train_stride: int,
    extreme_alpha: float,
    last_step_weight: float,
    data_lead_gamma: float,
    physics_lead_gamma: float,
    extreme_quantile: float,
) -> dict:
    return {
        "experiment_family": family,
        "config_name": config_name,
        "train_stride_override": int(train_stride),
        "physics_lambda_max": float(physics_lambda_max),
        "physics_state_weights": [1.0, 0.35, 0.35, 0.25],
        "physics_lead_gamma": float(physics_lead_gamma),
        "data_lead_gamma": float(data_lead_gamma),
        "extreme_alpha": float(extreme_alpha),
        "extreme_quantile": float(extreme_quantile),
        "last_step_weight": float(last_step_weight),
    }


def build_experiment_plan(args) -> list[dict]:
    configs: list[dict] = []
    base = {
        "physics_lambda_max": args.base_lambda,
        "train_stride": args.base_train_stride,
        "extreme_alpha": args.base_extreme_alpha,
        "last_step_weight": args.last_step_weight,
        "data_lead_gamma": args.data_lead_gamma,
        "physics_lead_gamma": args.physics_lead_gamma,
        "extreme_quantile": args.extreme_quantile,
    }
    configs.append(make_config("baseline", "baseline_current_physics_loss", **base))

    for lam in args.lambda_grid:
        configs.append(
            make_config(
                "priority1_lambda_sweep",
                f"lambda_{lam:g}_stride{args.base_train_stride}_extreme{args.base_extreme_alpha:g}",
                lam,
                args.base_train_stride,
                args.base_extreme_alpha,
                args.last_step_weight,
                args.data_lead_gamma,
                args.physics_lead_gamma,
                args.extreme_quantile,
            )
        )

    for stride in args.stride_grid:
        configs.append(
            make_config(
                "priority2_stride_sweep",
                f"stride_{stride}_lambda{args.base_lambda:g}_extreme{args.base_extreme_alpha:g}",
                args.base_lambda,
                stride,
                args.base_extreme_alpha,
                args.last_step_weight,
                args.data_lead_gamma,
                args.physics_lead_gamma,
                args.extreme_quantile,
            )
        )

    for alpha in args.extreme_alpha_grid:
        configs.append(
            make_config(
                "priority3_extreme_weight_sweep",
                f"extreme_alpha_{alpha:g}_lambda{args.base_lambda:g}_stride{args.base_train_stride}",
                args.base_lambda,
                args.base_train_stride,
                alpha,
                args.last_step_weight,
                args.data_lead_gamma,
                args.physics_lead_gamma,
                args.extreme_quantile,
            )
        )

    deduped: list[dict] = []
    seen: set[tuple] = set()
    for cfg in configs:
        key = (
            cfg["physics_lambda_max"],
            cfg["train_stride_override"],
            cfg["extreme_alpha"],
            cfg["last_step_weight"],
            cfg["data_lead_gamma"],
            cfg["physics_lead_gamma"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cfg)
    return deduped


def run_one(args, arrays, coops_cols, horizon: int, config: dict, device) -> dict[str, float]:
    run_args = argparse.Namespace(**vars(args))
    run_args.train_stride = int(config["train_stride_override"])
    run_dir_name = config["config_name"].replace(".", "p")
    full_config = {k: v for k, v in config.items() if k not in {"experiment_family", "train_stride_override"}}
    row = v4.run_config(run_args, arrays, coops_cols, horizon, full_config, device)
    row["experiment_family"] = config["experiment_family"]
    row["train_stride"] = int(config["train_stride_override"])
    row["run_dir_name"] = run_dir_name
    return row


def add_gain_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    gain_cols = [
        "last_residual_R2",
        "seq_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "extreme_abs_q95_residual_RMSE",
        "last_sea_level_R2",
    ]
    for horizon, sub in out.groupby("horizon"):
        base_rows = sub[sub["config_name"] == "baseline_current_physics_loss"]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        idx = sub.index
        for col in gain_cols:
            if col not in out.columns:
                continue
            if "RMSE" in col:
                out.loc[idx, f"{col}_reduction_pct_vs_baseline"] = 100.0 * (base[col] - out.loc[idx, col]) / base[col]
            else:
                out.loc[idx, f"{col}_gain_vs_baseline"] = out.loc[idx, col] - base[col]
    return out


def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for horizon, sub in summary.groupby("horizon"):
        sub = sub.sort_values("last_residual_R2", ascending=False)
        labels = sub["config_name"].astype(str).tolist()
        y = np.arange(len(sub))
        fig, axes = plt.subplots(1, 3, figsize=(18, max(5.0, 0.42 * len(sub))))
        axes[0].barh(y, sub["last_residual_R2"], color="#3b6ea8")
        axes[0].set_title("24h last residual R2" if horizon == 24 else f"{horizon}h last residual R2")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].grid(axis="x", alpha=0.25)

        axes[1].barh(y, sub["seq_residual_R2"], color="#5a9b7a")
        axes[1].set_title("Sequence residual R2")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels([])
        axes[1].invert_yaxis()
        axes[1].grid(axis="x", alpha=0.25)

        metric = "extreme_abs_q95_residual_R2"
        axes[2].barh(y, sub[metric], color="#b46a55")
        axes[2].set_title("Extreme top5% residual R2")
        axes[2].set_yticks(y)
        axes[2].set_yticklabels([])
        axes[2].invert_yaxis()
        axes[2].grid(axis="x", alpha=0.25)

        fig.suptitle(f"Priority top-3 experiments, horizon={horizon}h")
        fig.tight_layout()
        fig.savefig(output_dir / f"priority_top3_horizon_{horizon}h_ranking.png", dpi=220)
        plt.close(fig)

    family = summary.groupby("experiment_family", as_index=False).agg(
        best_last_residual_R2=("last_residual_R2", "max"),
        best_seq_residual_R2=("seq_residual_R2", "max"),
        best_extreme_q95_R2=("extreme_abs_q95_residual_R2", "max"),
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(family))
    width = 0.25
    ax.bar(x - width, family["best_last_residual_R2"], width, label="Last residual R2")
    ax.bar(x, family["best_seq_residual_R2"], width, label="Seq residual R2")
    ax.bar(x + width, family["best_extreme_q95_R2"], width, label="Extreme q95 R2")
    ax.set_xticks(x)
    ax.set_xticklabels(family["experiment_family"], rotation=20, ha="right")
    ax.set_title("Best metric by experiment family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "priority_top3_best_by_family.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run priority top-3 experiments for paper credibility")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--horizons", type=int, nargs="+", default=[24])
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--base-train-stride", type=int, default=8)
    parser.add_argument("--gnn-hidden", type=int, default=40)
    parser.add_argument("--gru-hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--graph-lr-mult", type=float, default=3.0)
    parser.add_argument("--physics-lr-mult", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--physics-warmup-epochs", type=int, default=8)
    parser.add_argument("--physics-ramp-epochs", type=int, default=14)
    parser.add_argument("--physics-loss-type", default="huber", choices=["huber", "mse"])
    parser.add_argument("--physics-forcing-mode", default="last_input", choices=["last_input", "future"])
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument(
        "--selection-metric",
        default="val_eta_data_loss",
        choices=["val_eta_data_loss", "val_data_loss", "val_last_loss", "val_total_loss"],
    )
    parser.add_argument("--base-lambda", type=float, default=0.0003)
    parser.add_argument("--base-extreme-alpha", type=float, default=0.0)
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[0.0001, 0.0002, 0.0003, 0.0005, 0.001])
    parser.add_argument("--stride-grid", type=int, nargs="+", default=[4])
    parser.add_argument("--extreme-alpha-grid", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--physics-lead-gamma", type=float, default=0.0)
    parser.add_argument("--data-lead-gamma", type=float, default=0.0)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--last-step-weight", type=float, default=0.2)
    parser.add_argument("--max-configs", type=int, default=0)
    args = parser.parse_args()

    v2.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arrays, station_meta, coops_cols = v3.build_enhanced_arrays()
    station_meta.to_csv(output_dir / "station_meta_used.csv", index=False)
    pd.DataFrame({"coops_enhanced_feature": coops_cols}).to_csv(output_dir / "coops_features_used.csv", index=False)

    configs = build_experiment_plan(args)
    if args.max_configs > 0:
        configs = configs[: args.max_configs]
    pd.DataFrame(configs).to_csv(output_dir / "experiment_plan.csv", index=False)

    print(f"Device: {device}")
    print(f"Running {len(configs)} experiment configs: {[c['config_name'] for c in configs]}")

    rows = []
    for horizon in args.horizons:
        for cfg in configs:
            rows.append(run_one(args, arrays, coops_cols, horizon, cfg, device))
            pd.DataFrame(rows).to_csv(output_dir / "priority_top3_metrics_partial.csv", index=False)

    summary = pd.DataFrame(rows)
    summary = add_gain_columns(summary)
    summary = summary.sort_values(["horizon", "last_residual_R2"], ascending=[True, False])
    summary.to_csv(output_dir / "priority_top3_metrics.csv", index=False)

    key_cols = [
        "horizon",
        "experiment_family",
        "config_name",
        "train_stride",
        "physics_lambda_max",
        "extreme_alpha",
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_R2_gain_vs_baseline",
        "last_residual_RMSE",
        "last_residual_RMSE_reduction_pct_vs_baseline",
        "extreme_abs_q95_residual_R2",
        "extreme_abs_q95_residual_R2_gain_vs_baseline",
        "seq_sea_level_R2",
        "last_sea_level_R2",
        "best_val_score",
    ]
    existing = [c for c in key_cols if c in summary.columns]
    summary[existing].to_csv(output_dir / "priority_top3_key_metrics.csv", index=False)
    plot_results(summary, output_dir)

    print("\nFinished priority top-3 experiments.")
    print(summary[existing].to_string(index=False))


if __name__ == "__main__":
    main()
