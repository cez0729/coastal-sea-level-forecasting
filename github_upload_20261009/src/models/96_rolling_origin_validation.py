from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPT78 = Path(__file__).resolve().parent / "78_final_four_models_enhanced_data.py"
SCRIPT84 = Path(__file__).resolve().parent / "84_causal_preprocessing_robustness.py"
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "rolling_origin_causal_validation"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


final = load_module("final_models_impl", SCRIPT78)
causal = load_module("causal_impl", SCRIPT84)
v2 = final.v2
v3 = final.v3


FOLDS = {
    "test_2024_h2": ("2024-01-01", "2024-07-01", "2025-01-01"),
    "test_2025_h2": ("2025-01-01", "2025-07-01", "2026-01-01"),
}


def time_index(times: np.ndarray, timestamp: str) -> int:
    values = pd.to_datetime(times).to_numpy(dtype="datetime64[ns]")
    return int(np.searchsorted(values, np.datetime64(timestamp), side="left"))


def build_fold_data(args, horizon: int, add_ode_prior: bool = False):
    arrays, station_meta, coops_cols = causal.build_enhanced_arrays_causal()
    feature_cols, physics_cols = v3.make_feature_sets(coops_cols)
    if add_ode_prior:
        arrays["ode_persistence_residual"] = arrays["residual"].copy()
        arrays["ode_graph_blend_residual"] = 0.70 * arrays["residual"] + 0.30 * (
            arrays["adj_distance"] @ arrays["residual"].T
        ).T
        arrays["ode_local_trend_3h"] = np.zeros_like(arrays["residual"], dtype=np.float32)
        arrays["ode_local_trend_3h"][3:] = (arrays["residual"][3:] - arrays["residual"][:-3]) / 3.0
        feature_cols = feature_cols + ["ode_persistence_residual", "ode_graph_blend_residual", "ode_local_trend_3h"]

    x_raw = np.stack([arrays[c] for c in feature_cols], axis=-1).astype(np.float32)
    physics_raw = np.stack([arrays[c] for c in physics_cols], axis=-1).astype(np.float32)
    states = np.stack(
        [arrays["residual"], arrays["uo"], arrays["vo"], arrays["wave_setup_proxy"]], axis=-1
    ).astype(np.float32)
    residual = arrays["residual"].astype(np.float32)
    tide = arrays["tide"].astype(np.float32)

    train_end = time_index(arrays["time"], args.fold_train_end)
    val_end = time_index(arrays["time"], args.fold_val_end)
    test_end = min(time_index(arrays["time"], args.fold_test_end), len(arrays["time"]))
    if not (args.window + horizon < train_end < val_end < test_end):
        raise ValueError(f"Invalid fold boundaries: train={train_end}, val={val_end}, test={test_end}")

    n_time, nodes, feats = x_raw.shape
    x_scaler = StandardScaler().fit(x_raw[:train_end].reshape(-1, feats))
    x_scaled = x_scaler.transform(x_raw.reshape(-1, feats)).reshape(n_time, nodes, feats).astype(np.float32)
    phys_scaler = StandardScaler().fit(physics_raw[:train_end].reshape(-1, len(physics_cols)))
    phys_scaled = phys_scaler.transform(physics_raw.reshape(-1, len(physics_cols))).reshape(
        n_time, nodes, len(physics_cols)
    ).astype(np.float32)

    state_scale = np.std(states[:train_end].reshape(-1, len(v2.STATE_NAMES)), axis=0).astype(np.float32) + 1e-6
    delta_scale = np.std(
        (states[1:train_end] - states[: train_end - 1]).reshape(-1, len(v2.STATE_NAMES)), axis=0
    ).astype(np.float32) + 1e-6
    threshold = float(np.quantile(np.abs(states[:train_end, :, 0]).reshape(-1), args.extreme_quantile))
    graph_priors = v2.make_graph_priors(arrays, train_end)

    single_train = final.SingleStateDataset(x_scaled, residual, tide, args.window, horizon, 0, train_end, args.train_stride)
    single_val = final.SingleStateDataset(
        x_scaled, residual, tide, args.window, horizon, train_end - args.window, val_end, 1
    )
    single_test = final.SingleStateDataset(
        x_scaled, residual, tide, args.window, horizon, val_end - args.window, test_end, 1
    )
    multi_train = v2.MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, 0, train_end, args.physics_forcing_mode, args.train_stride
    )
    multi_val = v2.MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, train_end - args.window, val_end, args.physics_forcing_mode, 1
    )
    multi_test = v2.MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, val_end - args.window, test_end, args.physics_forcing_mode, 1
    )
    return {
        "arrays": arrays,
        "station_meta": station_meta,
        "coops_cols": coops_cols,
        "feature_cols": feature_cols,
        "physics_cols": physics_cols,
        "graph_priors": graph_priors,
        "state_scale": state_scale,
        "delta_scale": delta_scale,
        "x_scaler_state": {
            "mean": x_scaler.mean_.astype(np.float64),
            "scale": x_scaler.scale_.astype(np.float64),
            "var": x_scaler.var_.astype(np.float64),
            "n_features_in": int(x_scaler.n_features_in_),
        },
        "physics_scaler_state": {
            "mean": phys_scaler.mean_.astype(np.float64),
            "scale": phys_scaler.scale_.astype(np.float64),
            "var": phys_scaler.var_.astype(np.float64),
            "n_features_in": int(phys_scaler.n_features_in_),
        },
        "train_abs_eta_threshold": threshold,
        "nodes": nodes,
        "feats": feats,
        "single_train": single_train,
        "single_val": single_val,
        "single_test": single_test,
        "multi_train": multi_train,
        "multi_val": multi_val,
        "multi_test": multi_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict-causal rolling-origin validation for final models.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--folds", nargs="+", default=list(FOLDS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--models", nargs="+", default=["gnn_bigru", "physical_loss"])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", default="distance", choices=["identity", "distance", "corr"])
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
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--physics-warmup-epochs", type=int, default=8)
    parser.add_argument("--physics-ramp-epochs", type=int, default=14)
    parser.add_argument("--physics-loss-type", default="huber", choices=["huber", "mse"])
    parser.add_argument("--physics-forcing-mode", default="last_input", choices=["last_input"])
    parser.add_argument("--physics-lambda-max", type=float, default=0.0003)
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.2)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument("--selection-metric", default="val_eta_data_loss")
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    args = parser.parse_args()

    final.build_enhanced_data = build_fold_data
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for fold_name in args.folds:
        if fold_name not in FOLDS:
            raise ValueError(f"Unknown fold {fold_name}; choices={list(FOLDS)}")
        args.fold_train_end, args.fold_val_end, args.fold_test_end = FOLDS[fold_name]
        for seed in args.seeds:
            args.seed = seed
            v2.set_seed(seed)
            args.output_dir = str(output_root / fold_name / f"seed_{seed}")
            for model in args.models:
                if model == "gnn_bigru":
                    row = final.run_single(args, args.horizon, "gnn_bigru", "GNN-BiGRU", False, device)
                elif model == "learnable_graph":
                    row = final.run_single(args, args.horizon, "learnable_graph", "Learnable-graph GNN-BiGRU", False, device)
                elif model == "ode_based_learnable":
                    row = final.run_single(args, args.horizon, "ode_based_learnable", "ODE-based learnable GNN-BiGRU", True, device)
                elif model == "physical_loss":
                    row = final.run_physical_loss(args, args.horizon, device)
                else:
                    raise ValueError(f"Unsupported model: {model}")
                row.update(
                    {
                        "fold": fold_name,
                        "seed": seed,
                        "train_end": args.fold_train_end,
                        "val_end": args.fold_val_end,
                        "test_end": args.fold_test_end,
                        "strict_causal": True,
                    }
                )
                rows.append(row)
                pd.DataFrame(rows).to_csv(output_root / "rolling_origin_metrics_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "rolling_origin_metrics.csv", index=False)
    metric_cols = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2"]
    agg = df.groupby(["fold", "model_name"])[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(x for x in col if x) for col in agg.columns.to_flat_index()]
    agg.to_csv(output_root / "rolling_origin_mean_std.csv", index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
