import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_DIRS = {
    "GNN-BiGRU": "gnn_bigru",
    "Learnable-graph GNN-BiGRU": "learnable_graph",
    "ODE-based learnable GNN-BiGRU": "ode_based_learnable",
    "Physical-loss GNN-BiGRU": "physical_loss",
}


def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def load_residual_predictions(pred_path):
    data = np.load(pred_path)
    if "pred_states" in data.files:
        pred = data["pred_states"][..., 0]
        true = data["true_states"][..., 0]
    else:
        pred = data["pred_residual"]
        true = data["true_residual"]
    return pred.astype(np.float64), true.astype(np.float64)


def load_sample_times(root, n_samples):
    candidates = [
        root / "data" / "dataset_multi_horizon" / "horizon_24h" / "sample_times.csv",
        root / "data" / "dataset_era5_wave_depth" / "horizon_24h" / "sample_times.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            col = df.columns[0]
            return pd.to_datetime(df[col]).tail(n_samples).reset_index(drop=True)
    return pd.Series(pd.RangeIndex(n_samples), name="sample_index")


def station_wise_metrics(root, output_dir):
    station_path = root / "data" / "processed_multiyear_2023_2025" / "station_order.csv"
    stations = pd.read_csv(station_path)
    pred_root = root / "数据整理" / "outputs" / "final_validation_tasks" / "task2_all_models_horizons_seed42" / "horizon_24h"

    rows = []
    for model_name, model_dir in MODEL_DIRS.items():
        pred_path = pred_root / model_dir / "predictions.npz"
        if not pred_path.exists():
            continue
        pred, true = load_residual_predictions(pred_path)
        last_pred = pred[:, :, -1]
        last_true = true[:, :, -1]
        q95 = np.quantile(np.abs(last_true.reshape(-1)), 0.95)
        for j, station in stations.iterrows():
            y_true = last_true[:, j]
            y_pred = last_pred[:, j]
            extreme_mask = np.abs(y_true) >= q95
            rows.append(
                {
                    "model_name": model_name,
                    "station_id": station["station_id"],
                    "station_name": station["station_name"],
                    "state": station["state"],
                    "last_residual_R2": r2_score_np(y_true, y_pred),
                    "last_residual_RMSE": rmse_np(y_true, y_pred),
                    "last_residual_MAE": mae_np(y_true, y_pred),
                    "extreme_q95_last_R2": r2_score_np(y_true[extreme_mask], y_pred[extreme_mask]),
                    "extreme_q95_n": int(extreme_mask.sum()),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "station_wise_24h_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def lead_wise_metrics(root, output_dir):
    pred_root = root / "数据整理" / "outputs" / "final_validation_tasks" / "task2_all_models_horizons_seed42" / "horizon_24h"
    rows = []
    for model_name, model_dir in MODEL_DIRS.items():
        pred_path = pred_root / model_dir / "predictions.npz"
        if not pred_path.exists():
            continue
        pred, true = load_residual_predictions(pred_path)
        for lead in range(pred.shape[2]):
            y_true = true[:, :, lead]
            y_pred = pred[:, :, lead]
            rows.append(
                {
                    "model_name": model_name,
                    "lead_hour": lead + 1,
                    "residual_R2": r2_score_np(y_true, y_pred),
                    "residual_RMSE": rmse_np(y_true, y_pred),
                    "residual_MAE": mae_np(y_true, y_pred),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "lead_wise_24h_metrics.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8.5, 5.2))
    for model_name, group in out.groupby("model_name"):
        plt.plot(group["lead_hour"], group["residual_R2"], marker="o", linewidth=2, label=model_name)
    plt.xlabel("Forecast lead time (hour)")
    plt.ylabel("Residual R2")
    plt.title("Lead-wise residual R2 for 24h forecasting")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "lead_wise_24h_r2.png", dpi=220)
    plt.close()
    return out


def multiseed_significance(root, output_dir, bootstrap_n=20000, seed=123):
    metrics_path = root / "数据整理" / "outputs" / "final_validation_tasks_multiseed" / "all_validation_metrics_long.csv"
    df = pd.read_csv(metrics_path)
    pivot_last = df.pivot_table(index="seed", columns="model_name", values="last_residual_R2", aggfunc="first")
    pivot_q95 = df.pivot_table(index="seed", columns="model_name", values="extreme_abs_q95_residual_R2", aggfunc="first")

    rng = np.random.default_rng(seed)
    rows = []
    for metric_name, pivot in [("last_residual_R2", pivot_last), ("extreme_abs_q95_residual_R2", pivot_q95)]:
        if "Physical-loss GNN-BiGRU" not in pivot.columns or "GNN-BiGRU" not in pivot.columns:
            continue
        paired = pivot[["Physical-loss GNN-BiGRU", "GNN-BiGRU"]].dropna()
        diffs = (paired["Physical-loss GNN-BiGRU"] - paired["GNN-BiGRU"]).to_numpy()
        boot = []
        for _ in range(bootstrap_n):
            sample = rng.choice(diffs, size=diffs.size, replace=True)
            boot.append(np.mean(sample))
        boot = np.asarray(boot)
        rows.append(
            {
                "comparison": "Physical-loss GNN-BiGRU minus GNN-BiGRU",
                "metric": metric_name,
                "n_seeds": int(diffs.size),
                "seed_level_differences": ";".join(f"{x:.6f}" for x in diffs),
                "mean_difference": float(np.mean(diffs)),
                "std_difference": float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0,
                "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
                "bootstrap_prob_improvement": float(np.mean(boot > 0)),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "multiseed_significance_summary.csv", index=False, encoding="utf-8-sig")
    return out


def extreme_event_case_plot(root, output_dir):
    station_path = root / "data" / "processed_multiyear_2023_2025" / "station_order.csv"
    stations = pd.read_csv(station_path)
    pred_root = root / "数据整理" / "outputs" / "final_validation_tasks" / "task2_all_models_horizons_seed42" / "horizon_24h"

    model_preds = {}
    true_ref = None
    for model_name, model_dir in MODEL_DIRS.items():
        pred_path = pred_root / model_dir / "predictions.npz"
        if not pred_path.exists():
            continue
        pred, true = load_residual_predictions(pred_path)
        model_preds[model_name] = pred
        true_ref = true

    if true_ref is None:
        return None

    last_true = true_ref[:, :, -1]
    idx_flat = int(np.argmax(np.abs(last_true.reshape(-1))))
    sample_idx, station_idx = np.unravel_index(idx_flat, last_true.shape)
    times = load_sample_times(root, true_ref.shape[0])
    target_start_time = times.iloc[sample_idx] if sample_idx < len(times) else sample_idx

    leads = np.arange(1, true_ref.shape[2] + 1)
    plt.figure(figsize=(9.5, 5.4))
    plt.plot(leads, true_ref[sample_idx, station_idx, :], color="black", marker="o", linewidth=2.4, label="Observed residual")
    colors = {
        "GNN-BiGRU": "#4C78A8",
        "Learnable-graph GNN-BiGRU": "#F58518",
        "ODE-based learnable GNN-BiGRU": "#54A24B",
        "Physical-loss GNN-BiGRU": "#E45756",
    }
    for model_name, pred in model_preds.items():
        plt.plot(
            leads,
            pred[sample_idx, station_idx, :],
            marker=".",
            linewidth=1.8,
            label=model_name,
            color=colors.get(model_name),
        )
    station_name = stations.iloc[station_idx]["station_name"]
    station_id = stations.iloc[station_idx]["station_id"]
    plt.xlabel("Forecast lead time (hour)")
    plt.ylabel("Residual sea level")
    plt.title(f"Extreme 24h residual case: {station_name} ({station_id}), start={target_start_time}")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "extreme_event_case_24h.png", dpi=240)
    plt.close()

    case = pd.DataFrame(
        {
            "sample_index": [sample_idx],
            "target_start_time": [target_start_time],
            "station_id": [station_id],
            "station_name": [station_name],
            "abs_last_true_residual": [float(abs(last_true[sample_idx, station_idx]))],
            "last_true_residual": [float(last_true[sample_idx, station_idx])],
        }
    )
    case.to_csv(output_dir / "extreme_event_case_24h_metadata.csv", index=False, encoding="utf-8-sig")
    return case


def write_summary(output_dir, station_df, lead_df, sig_df, case_df):
    best_station = station_df.sort_values("last_residual_R2", ascending=False).head(10)
    summary_path = output_dir / "publication_strengthening_summary.md"

    def markdown_table(df):
        if df is None or df.empty:
            return ""
        text_df = df.copy()
        for col in text_df.columns:
            text_df[col] = text_df[col].map(lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))
        header = "| " + " | ".join(text_df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in text_df.astype(str).to_numpy()]
        return "\n".join([header, sep] + rows)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Publication Strengthening Analysis\n\n")
        f.write("This analysis adds reviewer-facing evidence without retraining the models.\n\n")
        f.write("## Generated Files\n\n")
        f.write("- `station_wise_24h_metrics.csv`: station-level 24h last-step and q95 metrics.\n")
        f.write("- `lead_wise_24h_metrics.csv`: lead-time-wise 1h to 24h residual metrics.\n")
        f.write("- `lead_wise_24h_r2.png`: lead-wise residual R2 figure.\n")
        f.write("- `multiseed_significance_summary.csv`: seed-level paired improvement and bootstrap CI.\n")
        f.write("- `extreme_event_case_24h.png`: qualitative high-residual event trajectory.\n")
        f.write("- `extreme_event_case_24h_metadata.csv`: metadata for the selected case.\n\n")
        f.write("## Key Significance Summary\n\n")
        f.write(markdown_table(sig_df))
        f.write("\n\n## Selected Extreme Event\n\n")
        if case_df is not None:
            f.write(markdown_table(case_df))
        f.write("\n\n## Top Station-Level Rows By Last-Step R2\n\n")
        f.write(markdown_table(best_station))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument(
        "--output-dir",
        default="数据整理/outputs/publication_strengthening",
        help="Output directory for publication-strengthening analysis.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    station_df = station_wise_metrics(root, output_dir)
    lead_df = lead_wise_metrics(root, output_dir)
    sig_df = multiseed_significance(root, output_dir)
    case_df = extreme_event_case_plot(root, output_dir)
    write_summary(output_dir, station_df, lead_df, sig_df, case_df)

    print("Publication strengthening analysis completed.")
    print(f"Output directory: {output_dir}")
    print("Main files:")
    for name in [
        "station_wise_24h_metrics.csv",
        "lead_wise_24h_metrics.csv",
        "lead_wise_24h_r2.png",
        "multiseed_significance_summary.csv",
        "extreme_event_case_24h.png",
        "publication_strengthening_summary.md",
    ]:
        print(f"- {output_dir / name}")


if __name__ == "__main__":
    main()
