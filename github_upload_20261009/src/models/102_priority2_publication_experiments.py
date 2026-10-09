from __future__ import annotations

import argparse
import copy
import importlib.util
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SCRIPT77 = Path(__file__).resolve().parent / "77_v4_physics_loss_weight_strategy_search.py"
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "priority2_publication_experiments"

v4 = None
v3 = None
v2 = None


def load_training_modules():
    global v4, v3, v2
    if v4 is not None:
        return
    spec77 = importlib.util.spec_from_file_location("v4_impl", SCRIPT77)
    module = importlib.util.module_from_spec(spec77)
    assert spec77.loader is not None
    spec77.loader.exec_module(module)
    v4 = module
    v3 = module.v3
    v2 = module.v2


def make_sensitivity_configs(args) -> list[dict]:
    configs = []
    for last_weight in args.no_physics_last_step_weights:
        configs.append(
            {
                "config_name": f"no_phys_a24_{last_weight:g}_aux_{args.default_aux_weight:g}",
                "physics_lambda_max": 0.0,
                "physics_state_weights": [1.0, 0.35, 0.35, 0.25],
                "physics_lead_gamma": 0.0,
                "data_lead_gamma": 0.0,
                "extreme_alpha": 0.0,
                "extreme_quantile": args.extreme_quantile,
                "last_step_weight": float(last_weight),
                "aux_weight": float(args.default_aux_weight),
                "group": "terminal_weight_no_physics",
            }
        )
    for last_weight in args.physics_last_step_weights:
        for aux_weight in args.aux_weights:
            configs.append(
                {
                    "config_name": f"phys_lam{args.physics_lambda_max:g}_a24_{last_weight:g}_aux_{aux_weight:g}",
                    "physics_lambda_max": float(args.physics_lambda_max),
                    "physics_state_weights": [1.0, 0.35, 0.35, 0.25],
                    "physics_lead_gamma": 0.0,
                    "data_lead_gamma": 0.0,
                    "extreme_alpha": 0.0,
                    "extreme_quantile": args.extreme_quantile,
                    "last_step_weight": float(last_weight),
                    "aux_weight": float(aux_weight),
                    "group": "terminal_aux_physics",
                }
            )
    return configs


def train_sensitivity(args) -> None:
    import torch

    load_training_modules()
    v2.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays, station_meta, coops_cols = v3.build_enhanced_arrays()
    station_meta.to_csv(output_dir / "station_meta_used.csv", index=False)
    pd.DataFrame({"coops_enhanced_feature": coops_cols}).to_csv(output_dir / "coops_features_used.csv", index=False)

    rows = []
    configs = make_sensitivity_configs(args)
    pd.DataFrame(configs).drop(columns=["physics_state_weights"]).to_csv(
        output_dir / f"priority2_sensitivity_plan_seed_{args.seed}.csv", index=False
    )
    for horizon in args.horizons:
        for config in configs:
            run_args = copy.copy(args)
            run_args.aux_weight = float(config["aux_weight"])
            row = v4.run_config(run_args, arrays, coops_cols, horizon, config, device)
            row["seed"] = args.seed
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / f"priority2_sensitivity_seed_{args.seed}_partial.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / f"priority2_sensitivity_seed_{args.seed}.csv", index=False)
    print(summary[important_sensitivity_cols(summary)].to_string(index=False))


def important_sensitivity_cols(df: pd.DataFrame) -> list[str]:
    cols = [
        "seed",
        "horizon",
        "config_name",
        "group",
        "physics_lambda_max",
        "last_step_weight",
        "aux_weight",
        "best_val_score",
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "learned_w_identity",
        "learned_w_distance",
        "learned_w_corr",
    ]
    return [c for c in cols if c in df.columns]


def parse_seed(path: Path) -> int | float:
    matches = re.findall(r"seed[_-]?(\d+)", str(path))
    return int(matches[-1]) if matches else float("nan")


def load_prediction_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    data = np.load(path, allow_pickle=True)
    if "pred_states" in data.files:
        pred = data["pred_states"][..., 0]
        true = data["true_states"][..., 0]
    else:
        pred = data["pred_residual"]
        true = data["true_residual"]
    times = data["target_start_times"] if "target_start_times" in data.files else None
    meta = {
        "seed": parse_seed(path),
        "model_key": path.parent.name,
        "prediction_file": str(path),
    }
    return pred.astype(float), true.astype(float), times, meta


def collect_csv(root_paths: list[str], patterns: list[str]) -> pd.DataFrame:
    frames = []
    seen_paths = set()
    for root in root_paths:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for pattern in patterns:
            for path in sorted(root_path.rglob(pattern)):
                resolved = path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
                df = df.copy()
                df["source_file"] = str(resolved)
                if "seed" not in df.columns:
                    df["seed"] = parse_seed(path)
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_sensitivity(args, out_dir: Path) -> None:
    df = collect_csv(args.analysis_roots, ["priority2_sensitivity_seed_*.csv"])
    if df.empty:
        return
    if "config_name" not in df.columns:
        return
    config_name = df["config_name"].fillna("").astype(str)
    sens = df[config_name.str.contains("phys_|no_phys_", regex=True)].copy()
    sens = sens[~sens["source_file"].str.contains("_partial.csv", regex=False)]
    sens = sens[np.isclose(sens["physics_lambda_max"].fillna(0.0), 0.0) | np.isclose(
        sens["physics_lambda_max"].fillna(0.0), args.locked_physics_lambda
    )]
    dedup_cols = [c for c in ["seed", "horizon", "config_name"] if c in sens.columns]
    sens = sens.drop_duplicates(subset=dedup_cols, keep="last").sort_values(dedup_cols)
    if sens.empty:
        return
    sens.to_csv(out_dir / "priority2_sensitivity_all_runs.csv", index=False)
    metrics = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "best_val_score",
    ]
    available = [c for c in metrics if c in sens.columns]
    agg = sens.groupby(["horizon", "config_name", "group", "physics_lambda_max", "last_step_weight", "aux_weight"])[available].agg(
        ["mean", "std", "count"]
    ).reset_index()
    agg.columns = ["_".join(str(x) for x in col if x) for col in agg.columns.to_flat_index()]
    agg.to_csv(out_dir / "priority2_sensitivity_mean_std.csv", index=False)
    plot_sensitivity(sens, out_dir)


def missing_data_audit(args, out_dir: Path) -> None:
    rows = []
    data_root = ROOT / "data"
    files = [
        "processed_multiyear_2023_2025/water_tide_residual_long.csv",
        "processed_multiyear_2023_2025/era5_station_hourly.csv",
        "processed_multiyear_2023_2025/surface_currents_station_daily.csv",
        "processed_multiyear_2023_2025/wave_direction_speed_station_3hourly.csv",
        "processed_multiyear_2023_2025/noaa_coops_met_station_hourly.csv",
        "processed/gebco_station_depth.csv",
    ]
    for rel in files:
        path = data_root / rel
        if not path.exists():
            rows.append({"source_file": rel, "column": "__file__", "status": "missing_file"})
            continue
        df = pd.read_csv(path)
        station_count = df["station_id"].astype(str).nunique() if "station_id" in df.columns else np.nan
        time_count = pd.to_datetime(df["datetime"]).nunique() if "datetime" in df.columns else np.nan
        for column in df.columns:
            if column in {"datetime", "station_id"}:
                continue
            miss = int(df[column].isna().sum())
            rows.append(
                {
                    "source_file": rel,
                    "column": column,
                    "rows": len(df),
                    "stations": station_count,
                    "unique_times": time_count,
                    "missing_count": miss,
                    "missing_rate": miss / max(1, len(df)),
                    "status": "ok",
                }
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "priority2_missing_data_by_source_column.csv", index=False)
    if "missing_rate" in raw.columns:
        summary = raw[raw["status"].eq("ok")].groupby("source_file")["missing_rate"].agg(["mean", "max"]).reset_index()
        summary.to_csv(out_dir / "priority2_missing_data_source_summary.csv", index=False)
        plot_missing(summary, out_dir)


def graph_weight_analysis(args, out_dir: Path) -> None:
    df = collect_csv(args.analysis_roots, ["*.csv", "metrics.csv"])
    if df.empty:
        return
    weight_cols = ["learned_w_identity", "learned_w_distance", "learned_w_corr"]
    if not all(c in df.columns for c in weight_cols):
        return
    name_col = "model_name" if "model_name" in df.columns else "config_name" if "config_name" in df.columns else "model"
    weights = df[df[weight_cols].notna().any(axis=1)].copy()
    weights["model_or_config"] = weights[name_col].astype(str)
    dedup_cols = [c for c in ["seed", "horizon", "model_or_config", *weight_cols] if c in weights.columns]
    weights = weights.drop_duplicates(subset=dedup_cols, keep="last")
    weights.to_csv(out_dir / "priority2_learned_graph_weights_all.csv", index=False)
    agg = weights.groupby("model_or_config")[weight_cols].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join(str(x) for x in col if x) for col in agg.columns.to_flat_index()]
    agg.to_csv(out_dir / "priority2_learned_graph_weights_mean_std.csv", index=False)
    plot_graph_weights(weights, out_dir)


def failure_case_analysis(args, out_dir: Path) -> None:
    paths = []
    for root in args.prediction_roots:
        root_path = Path(root)
        if root_path.exists():
            paths.extend(sorted(root_path.rglob("predictions.npz")))
    rows = []
    top_records = []
    for path in paths:
        if "phys_lam" in path.parent.name and f"phys_lam{args.locked_physics_lambda:g}_" not in path.parent.name:
            continue
        pred, true, times, meta = load_prediction_file(path)
        last_true = true[:, :, -1]
        last_pred = pred[:, :, -1]
        threshold = np.quantile(np.abs(last_true.reshape(-1)), args.failure_quantile)
        event_mask = np.abs(last_true) >= threshold
        error = np.abs(last_pred - last_true)
        score = error * event_mask
        if np.max(score) <= 0:
            score = error
        flat_idx = int(np.argmax(score))
        sample_idx, station_idx = np.unravel_index(flat_idx, error.shape)
        rows.append(
            {
                **meta,
                "sample_index": sample_idx,
                "station_index": station_idx,
                "target_start_time": str(times[sample_idx]) if times is not None else "",
                "terminal_true_residual": float(last_true[sample_idx, station_idx]),
                "terminal_pred_residual": float(last_pred[sample_idx, station_idx]),
                "terminal_abs_error": float(error[sample_idx, station_idx]),
                "event_abs_threshold": float(threshold),
                "is_event": bool(event_mask[sample_idx, station_idx]),
            }
        )
        top_records.append((float(error[sample_idx, station_idx]), path, sample_idx, station_idx, meta))
    cases = pd.DataFrame(rows).sort_values("terminal_abs_error", ascending=False)
    cases.to_csv(out_dir / "priority2_failure_cases_top_by_model.csv", index=False)
    top_records.sort(key=lambda item: item[0], reverse=True)
    for rank, (_, path, sample_idx, station_idx, meta) in enumerate(top_records[: args.max_failure_plots], start=1):
        pred, true, times, _ = load_prediction_file(path)
        lead = np.arange(1, pred.shape[-1] + 1)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(lead, true[sample_idx, station_idx], marker="o", label="true residual")
        ax.plot(lead, pred[sample_idx, station_idx], marker="s", label="predicted residual")
        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel("Residual sea level")
        title_time = str(times[sample_idx]) if times is not None else f"sample {sample_idx}"
        ax.set_title(f"{meta['model_key']} failure case, station {station_idx}, {title_time}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"priority2_failure_case_{rank:02d}_{meta['model_key']}.png", dpi=220)
        plt.close(fig)


def plot_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    if "last_residual_R2" not in df.columns:
        return
    agg = df.groupby(["config_name", "last_step_weight", "aux_weight", "physics_lambda_max"])["last_residual_R2"].mean().reset_index()
    agg = agg.sort_values("last_residual_R2", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4.2, 0.35 * len(agg))))
    ax.barh(np.arange(len(agg)), agg["last_residual_R2"])
    ax.set_yticks(np.arange(len(agg)))
    ax.set_yticklabels(agg["config_name"], fontsize=8)
    ax.set_xlabel("Mean terminal residual R2")
    ax.set_title("Priority-2 terminal/auxiliary loss sensitivity")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "priority2_sensitivity_terminal_r2.png", dpi=220)
    plt.close(fig)


def plot_missing(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(np.arange(len(summary)), summary["max"])
    ax.set_yticks(np.arange(len(summary)))
    ax.set_yticklabels(summary["source_file"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Maximum column missing rate")
    ax.set_title("Raw data missingness by source")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "priority2_missing_data_source_summary.png", dpi=220)
    plt.close(fig)


def plot_graph_weights(weights: pd.DataFrame, out_dir: Path) -> None:
    weight_cols = ["learned_w_identity", "learned_w_distance", "learned_w_corr"]
    means = weights.groupby("model_or_config")[weight_cols].mean().sort_index()
    fig, ax = plt.subplots(figsize=(9, max(4.2, 0.35 * len(means))))
    left = np.zeros(len(means))
    labels = ["identity", "distance", "corr"]
    for column, label in zip(weight_cols, labels):
        ax.barh(np.arange(len(means)), means[column], left=left, label=label)
        left += means[column].to_numpy()
    ax.set_yticks(np.arange(len(means)))
    ax.set_yticklabels(means.index, fontsize=8)
    ax.set_xlabel("Mean learned graph mixture weight")
    ax.set_title("Learned graph mixture interpretability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "priority2_learned_graph_weights.png", dpi=220)
    plt.close(fig)


def analyze_evidence(args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summarize_sensitivity(args, out_dir)
    missing_data_audit(args, out_dir)
    graph_weight_analysis(args, out_dir)
    failure_case_analysis(args, out_dir)
    print(f"Priority-2 evidence written to {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Priority-2 publication experiments and evidence diagnostics.")
    parser.add_argument("--mode", choices=["train_sensitivity", "analyze_evidence"], required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--analysis-roots", nargs="+", default=["results", "results/merged_publication", "results/priority2_sensitivity"])
    parser.add_argument("--prediction-roots", nargs="+", default=["results/final", "results/priority1_graph_baselines", "results/priority2_sensitivity"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[24])
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--gnn-hidden", type=int, default=40)
    parser.add_argument("--gru-hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
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
    parser.add_argument("--physics-loss-type", choices=["huber", "mse"], default="huber")
    parser.add_argument("--physics-forcing-mode", choices=["last_input", "future"], default="last_input")
    parser.add_argument("--physics-lambda-max", type=float, default=0.0002)
    parser.add_argument("--locked-physics-lambda", type=float, default=0.0002)
    parser.add_argument("--default-aux-weight", type=float, default=0.08)
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--aux-weights", type=float, nargs="+", default=[0.0, 0.08, 0.16])
    parser.add_argument("--no-physics-last-step-weights", type=float, nargs="+", default=[0.0, 0.2, 0.5])
    parser.add_argument("--physics-last-step-weights", type=float, nargs="+", default=[0.0, 0.2, 0.5])
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument("--selection-metric", choices=["val_eta_data_loss", "val_data_loss", "val_last_loss", "val_total_loss"], default="val_eta_data_loss")
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--failure-quantile", type=float, default=0.95)
    parser.add_argument("--max-failure-plots", type=int, default=12)
    args = parser.parse_args()
    if args.mode == "train_sensitivity":
        train_sensitivity(args)
    else:
        analyze_evidence(args)


if __name__ == "__main__":
    main()
