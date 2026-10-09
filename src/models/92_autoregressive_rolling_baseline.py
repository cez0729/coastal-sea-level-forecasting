from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCRIPT78 = Path(__file__).resolve().parent / "78_final_four_models_enhanced_data.py"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "autoregressive_rolling_baseline"

spec78 = importlib.util.spec_from_file_location("final_models_impl", SCRIPT78)
final_models = importlib.util.module_from_spec(spec78)
assert spec78.loader is not None
spec78.loader.exec_module(final_models)
v2 = final_models.v2
v3 = final_models.v3


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def summarize_residual(true_residual: np.ndarray, pred_residual: np.ndarray, tide: np.ndarray) -> dict[str, float]:
    out = {
        "seq_residual_R2": r2_score_np(true_residual, pred_residual),
        "seq_residual_RMSE": rmse_np(true_residual, pred_residual),
        "seq_residual_MAE": mae_np(true_residual, pred_residual),
        "last_residual_R2": r2_score_np(true_residual[:, :, -1], pred_residual[:, :, -1]),
        "last_residual_RMSE": rmse_np(true_residual[:, :, -1], pred_residual[:, :, -1]),
        "last_residual_MAE": mae_np(true_residual[:, :, -1], pred_residual[:, :, -1]),
    }
    pred_level = pred_residual + tide
    true_level = true_residual + tide
    out.update(
        {
            "seq_sea_level_R2": r2_score_np(true_level, pred_level),
            "last_sea_level_R2": r2_score_np(true_level[:, :, -1], pred_level[:, :, -1]),
        }
    )

    threshold = float(np.quantile(np.abs(true_residual.reshape(-1)), 0.95))
    mask = np.abs(true_residual) >= threshold
    out.update(
        {
            "extreme_abs_q95_threshold": threshold,
            "extreme_abs_q95_n": int(mask.sum()),
            "extreme_abs_q95_residual_R2": r2_score_np(true_residual[mask], pred_residual[mask]),
            "extreme_abs_q95_residual_RMSE": rmse_np(true_residual[mask], pred_residual[mask]),
        }
    )
    return out


def build_scaled_arrays(args):
    arrays, station_meta, coops_cols = v3.build_enhanced_arrays()
    feature_cols, _ = v3.make_feature_sets(coops_cols)
    x_raw = np.stack([arrays[c] for c in feature_cols], axis=-1).astype(np.float32)
    residual = arrays["residual"].astype(np.float32)
    tide = arrays["tide"].astype(np.float32)
    water_level = arrays["water_level"].astype(np.float32)

    n_time, nodes, feats = x_raw.shape
    train_end = int(n_time * args.train_ratio)
    val_end = int(n_time * (args.train_ratio + args.val_ratio))

    scaler = StandardScaler()
    scaler.fit(x_raw[:train_end].reshape(-1, feats))
    x_scaled = scaler.transform(x_raw.reshape(-1, feats)).reshape(n_time, nodes, feats).astype(np.float32)
    graph_priors = v2.make_graph_priors(arrays, train_end)

    return {
        "arrays": arrays,
        "station_meta": station_meta,
        "feature_cols": feature_cols,
        "x_raw": x_raw,
        "x_scaled": x_scaled,
        "scaler": scaler,
        "residual": residual,
        "tide": tide,
        "water_level": water_level,
        "graph_priors": graph_priors,
        "train_end": train_end,
        "val_end": val_end,
        "n_time": n_time,
        "nodes": nodes,
        "feats": feats,
    }


def make_one_step_datasets(args, data):
    train = final_models.SingleStateDataset(
        data["x_scaled"], data["residual"], data["tide"], args.window, 1, 0, data["train_end"], args.train_stride
    )
    val = final_models.SingleStateDataset(
        data["x_scaled"], data["residual"], data["tide"], args.window, 1, data["train_end"], data["val_end"], 1
    )
    return train, val


def make_model(args, data, model_key: str):
    if model_key == "gnn_bigru":
        return final_models.FixedGraphGNNBiGRU(
            data["feats"],
            data["graph_priors"][args.fixed_graph_type],
            args.gnn_hidden,
            args.gru_hidden,
            1,
            args.dropout,
        )
    if model_key == "learnable_graph":
        return final_models.LearnableGraphSingleStateGNNBiGRU(
            data["feats"],
            data["graph_priors"],
            [0.50, 0.35, 0.15],
            args.gnn_hidden,
            args.gru_hidden,
            1,
            args.dropout,
        )
    raise ValueError(f"Unsupported model_key: {model_key}")


def scaled_feature_from_raw(data, feature_name: str, raw_value: np.ndarray) -> np.ndarray:
    idx = data["feature_cols"].index(feature_name)
    mean = data["scaler"].mean_[idx]
    scale = data["scaler"].scale_[idx]
    return ((raw_value - mean) / max(scale, 1e-8)).astype(np.float32)


@torch.no_grad()
def autoregressive_rollout(model, data, args, mode: str, device):
    if mode not in {"causal_hold_exog", "oracle_future_exog"}:
        raise ValueError("mode must be causal_hold_exog or oracle_future_exog")

    start_indices = np.arange(data["val_end"] + args.window, data["n_time"] - args.rollout_horizon + 1, dtype=np.int64)
    if args.max_test_samples and args.max_test_samples > 0:
        start_indices = start_indices[: args.max_test_samples]

    residual_idx = data["feature_cols"].index("residual")
    tide_idx = data["feature_cols"].index("tide") if "tide" in data["feature_cols"] else None
    water_idx = data["feature_cols"].index("water_level") if "water_level" in data["feature_cols"] else None

    preds = []
    trues = []
    tides = []
    for t in start_indices:
        window = data["x_scaled"][t - args.window: t].copy()
        sample_pred = []
        for lead in range(args.rollout_horizon):
            xb = torch.from_numpy(window[None].astype(np.float32)).to(device)
            pred_next = model(xb).detach().cpu().numpy()[0, :, 0].astype(np.float32)
            sample_pred.append(pred_next)

            future_time = t + lead
            if lead < args.rollout_horizon - 1:
                if mode == "oracle_future_exog":
                    next_row = data["x_scaled"][future_time].copy()
                else:
                    next_row = window[-1].copy()

                next_row[:, residual_idx] = scaled_feature_from_raw(data, "residual", pred_next)
                if tide_idx is not None:
                    future_tide = data["tide"][future_time]
                    next_row[:, tide_idx] = scaled_feature_from_raw(data, "tide", future_tide)
                if water_idx is not None:
                    future_tide = data["tide"][future_time]
                    pred_water = pred_next + future_tide
                    next_row[:, water_idx] = scaled_feature_from_raw(data, "water_level", pred_water)

                window = np.concatenate([window[1:], next_row[None]], axis=0)

        preds.append(np.stack(sample_pred, axis=-1))
        trues.append(data["residual"][t: t + args.rollout_horizon].T)
        tides.append(data["tide"][t: t + args.rollout_horizon].T)

    return np.stack(preds), np.stack(trues), np.stack(tides), start_indices


def plot_rollout_case(pred: np.ndarray, true: np.ndarray, output_path: Path, title: str):
    last_abs = np.abs(true[:, :, -1])
    sample_idx, station_idx = np.unravel_index(int(np.argmax(last_abs.reshape(-1))), last_abs.shape)
    lead = np.arange(1, true.shape[-1] + 1)
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(lead, true[sample_idx, station_idx], color="black", marker="o", linewidth=2.2, label="Observed residual")
    plt.plot(lead, pred[sample_idx, station_idx], color="#E45756", marker="s", linewidth=1.8, label="Autoregressive rolling")
    plt.xlabel("Forecast lead time (hour)")
    plt.ylabel("Residual sea level")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def load_direct_24h_rows() -> pd.DataFrame:
    path = (
        Path(__file__).resolve().parent
        / "outputs"
        / "final_validation_tasks"
        / "task2_all_models_horizons_seed42"
        / "final_four_models_enhanced_key_metrics.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["horizon"] == 24].copy()
    df["experiment_type"] = "direct_multi_output_24h_existing"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 1-step GNN-BiGRU and evaluate autoregressive rolling 24h forecasts.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--models", nargs="+", default=["gnn_bigru", "learnable_graph"], choices=["gnn_bigru", "learnable_graph"])
    parser.add_argument("--rollout-modes", nargs="+", default=["causal_hold_exog", "oracle_future_exog"], choices=["causal_hold_exog", "oracle_future_exog"])
    parser.add_argument("--rollout-horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", default="distance", choices=["identity", "distance", "corr"])
    parser.add_argument("--gnn-hidden", type=int, default=48)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--graph-lr-mult", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="Backward-compatible single-seed option.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Run multiple paired seeds, for example: --seeds 42 123 2024 2025 3407.",
    )
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--max-test-samples", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = build_scaled_arrays(args)
    data["station_meta"].to_csv(output_dir / "station_meta_used.csv", index=False)
    pd.DataFrame({"feature_col": data["feature_cols"]}).to_csv(output_dir / "feature_cols_used.csv", index=False)
    train_ds, val_ds = make_one_step_datasets(args, data)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    rows = []
    seeds = args.seeds if args.seeds else [args.seed]
    for seed in seeds:
        v2.set_seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
        )
        for model_key in args.models:
            model_name = "Autoregressive rolling GNN-BiGRU" if model_key == "gnn_bigru" else "Autoregressive rolling learnable-graph GNN-BiGRU"
            model = make_model(args, data, model_key).to(device)
            print("\n" + "=" * 90)
            print(
                f"Training 1-step model: {model_name}; seed={seed}; "
                f"train={len(train_ds)} val={len(val_ds)} features={data['feats']}"
            )
            history, best_val = final_models.train_single_model(model, train_loader, val_loader, args, device)
            model_dir = output_dir / f"seed_{seed}" / model_key
            model_dir.mkdir(parents=True, exist_ok=True)
            history.to_csv(model_dir / "one_step_training_log.csv", index=False)

            for mode in args.rollout_modes:
                pred, true, tide, start_indices = autoregressive_rollout(model, data, args, mode, device)
                metrics = summarize_residual(true, pred, tide)
                weights = model.weight_dict() if hasattr(model, "weight_dict") else {}
                row = {
                    "experiment_type": "autoregressive_rolling_1step_to_24h",
                    "model_key": model_key,
                    "model_name": model_name,
                    "rollout_mode": mode,
                    "horizon": args.rollout_horizon,
                    "seed": seed,
                    "best_1step_val_loss": best_val,
                    "num_test_samples": int(pred.shape[0]),
                    "learned_w_identity": weights.get("identity", np.nan),
                    "learned_w_distance": weights.get("distance", np.nan),
                    "learned_w_corr": weights.get("corr", np.nan),
                    **metrics,
                }
                rows.append(row)
                out_dir = model_dir / mode
                out_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)
                np.savez_compressed(
                    out_dir / "predictions.npz",
                    pred_residual=pred,
                    true_residual=true,
                    target_tide=tide,
                    target_start_indices=start_indices,
                )
                plot_rollout_case(pred, true, out_dir / "extreme_case_rollout.png", f"{model_name}, mode={mode}, seed={seed}")
                print(
                    f"Done {model_name} seed={seed} mode={mode}: seq_R2={row['seq_residual_R2']:.4f}, "
                    f"last_R2={row['last_residual_R2']:.4f}, RMSE={row['last_residual_RMSE']:.4f}, "
                    f"q95_R2={row['extreme_abs_q95_residual_R2']:.4f}"
                )

    summary = pd.DataFrame(rows)
    direct = load_direct_24h_rows()
    if not direct.empty:
        keep = [
            "experiment_type",
            "model_name",
            "horizon",
            "seq_residual_R2",
            "last_residual_R2",
            "last_residual_RMSE",
            "extreme_abs_q95_residual_R2",
            "seq_sea_level_R2",
            "last_sea_level_R2",
        ]
        for col in keep:
            if col not in direct.columns:
                direct[col] = np.nan
        direct = direct[keep].copy()
        direct["rollout_mode"] = "not_applicable"
        summary_for_compare = pd.concat([summary, direct], ignore_index=True, sort=False)
    else:
        summary_for_compare = summary.copy()

    summary.to_csv(output_dir / "autoregressive_rolling_metrics.csv", index=False)
    summary_for_compare.to_csv(output_dir / "autoregressive_vs_direct_comparison.csv", index=False)

    metric_cols = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "seq_sea_level_R2",
        "last_sea_level_R2",
    ]
    aggregate = summary.groupby(["model_key", "model_name", "rollout_mode", "horizon"])[metric_cols].agg(["mean", "std", "count"]).reset_index()
    aggregate.columns = ["_".join(str(part) for part in col if part) for col in aggregate.columns.to_flat_index()]
    aggregate.to_csv(output_dir / "autoregressive_rolling_mean_std.csv", index=False)
    run_config = vars(args).copy()
    run_config["resolved_seeds"] = seeds
    run_config["device"] = str(device)
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Autoregressive Rolling Baseline Summary",
        "",
        "This experiment trains a one-step GNN-BiGRU and recursively rolls it forward to 24 hours.",
        "",
        "## Interpretation",
        "",
        "- `causal_hold_exog` is the strict rolling baseline: residual and water level are updated with predictions, tide is known, and other exogenous variables are held from the latest available window.",
        "- `oracle_future_exog` is a diagnostic upper-bound style setting: future exogenous rows are used, but future residual/water level are replaced by predictions. It should not be claimed as the operational result.",
        "- If autoregressive rolling underperforms direct multi-output prediction, this supports the paper's current choice of direct 24h prediction and the use of cycle consistency to reduce trajectory drift without full autoregression.",
        "",
        "## Rolling Results",
        "",
        summary.to_string(index=False),
        "",
        "## Across-seed summary",
        "",
        aggregate.to_string(index=False),
    ]
    (output_dir / "autoregressive_rolling_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nFinished autoregressive rolling baseline.")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
