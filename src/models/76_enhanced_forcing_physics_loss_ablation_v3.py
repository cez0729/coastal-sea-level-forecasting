from __future__ import annotations

import argparse
import importlib.util
import math
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
SCRIPT74 = Path(__file__).resolve().parent / "74_multistate_physics_loss_gnn_bigru_v2.py"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "enhanced_forcing_physics_loss_ablation_v3"
COOPS_PATH = ROOT / "data" / "processed_multiyear_2023_2025" / "noaa_coops_met_station_hourly.csv"

spec = importlib.util.spec_from_file_location("v2_impl", SCRIPT74)
v2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v2)


MODE_WEIGHTS = {
    "no_physics": [1.0, 0.0, 0.0, 0.0],
    "eta_only": [1.0, 0.0, 0.0, 0.0],
    "eta_uv": [1.0, 0.35, 0.35, 0.0],
    "eta_uvW": [1.0, 0.35, 0.35, 0.25],
}


def merge_coops_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if not COOPS_PATH.exists():
        print(f"CO-OPS enhanced forcing file not found: {COOPS_PATH}")
        return df, []
    coops = pd.read_csv(COOPS_PATH)
    coops["datetime"] = pd.to_datetime(coops["datetime"])
    coops["station_id"] = coops["station_id"].astype(str)
    coops_cols = [
        "coops_wind_speed",
        "coops_wind_gust",
        "coops_air_pressure",
        "coops_air_temperature",
        "coops_water_temperature",
        "coops_pressure_anom",
        "coops_pressure_tendency_3h",
        "coops_wind_speed_tendency_3h",
    ]
    keep = ["datetime", "station_id"] + [c for c in coops_cols if c in coops.columns]
    out = df.merge(coops[keep], on=["datetime", "station_id"], how="left")
    available = [c for c in coops_cols if c in out.columns]

    for col in available:
        out[col] = out.groupby("station_id")[col].transform(lambda s: s.interpolate(limit=6, limit_direction="both"))
        median = out[col].median(skipna=True)
        if np.isfinite(median):
            out[col] = out[col].fillna(median)
        else:
            out[col] = out[col].fillna(0.0)

    if "coops_air_pressure" in out.columns and "msl" in out.columns:
        out["coops_minus_era5_pressure_hpa"] = out["coops_air_pressure"] - out["msl"] / 100.0
        available.append("coops_minus_era5_pressure_hpa")
    return out, available


def add_arrays_from_df(arrays: dict[str, np.ndarray], df: pd.DataFrame, cols: list[str]) -> None:
    time_index = pd.to_datetime(arrays["time"])
    for col in cols:
        values = []
        for sid in v2.STATION_IDS:
            s = (
                df[df["station_id"] == sid]
                .set_index("datetime")
                .loc[time_index, col]
                .to_numpy(dtype=np.float32)
            )
            values.append(s)
        arr = np.stack(values, axis=1).astype(np.float32)
        arrays[col] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_enhanced_arrays() -> tuple[dict[str, np.ndarray], pd.DataFrame, list[str]]:
    df, station_meta, adj = v2.make_hourly_feature_table()
    df, coops_cols = merge_coops_features(df)
    arrays = v2.build_arrays(df, adj)
    add_arrays_from_df(arrays, df, coops_cols)
    return arrays, station_meta, coops_cols


def make_feature_sets(coops_cols: list[str]) -> tuple[list[str], list[str]]:
    base_features = [
        "residual",
        "tide",
        "water_level",
        "u10",
        "v10",
        "wind_speed",
        "pressure_anom",
        "wind_stress_u_proxy",
        "wind_stress_v_proxy",
        "uo",
        "vo",
        "current_speed_mps",
        "wave_height",
        "wave_peak_period",
        "wave_mean_period",
        "wave_energy",
        "wave_energy_flux",
        "wave_setup_proxy",
        "wave_stokes_drift_x",
        "wave_stokes_drift_y",
        "wave_dir_x_from_mps",
        "wave_dir_y_from_mps",
        "wind_wave_alignment",
        "depth",
        "inverse_depth",
    ]
    base_physics = [
        "pressure_anom",
        "wind_stress_u_proxy",
        "wind_stress_v_proxy",
        "wind_speed",
        "current_speed_mps",
        "wave_energy",
        "wave_energy_flux",
        "wave_setup_proxy",
        "wave_stokes_drift_x",
        "wave_stokes_drift_y",
        "wave_dir_x_from_mps",
        "wave_dir_y_from_mps",
        "wind_wave_alignment",
        "inverse_depth",
    ]
    coops_for_model = [
        c
        for c in [
            "coops_air_pressure",
            "coops_pressure_anom",
            "coops_pressure_tendency_3h",
            "coops_minus_era5_pressure_hpa",
            "coops_air_temperature",
            "coops_water_temperature",
            "coops_wind_speed",
            "coops_wind_gust",
            "coops_wind_speed_tendency_3h",
        ]
        if c in coops_cols
    ]
    return base_features + coops_for_model, base_physics + coops_for_model


def summarize_lead_metrics(true_states: np.ndarray, pred_states: np.ndarray, horizon: int) -> pd.DataFrame:
    rows = []
    true_eta = true_states[..., 0]
    pred_eta = pred_states[..., 0]
    for lead in range(horizon):
        metrics = v2.regression_metrics(true_eta[:, :, lead], pred_eta[:, :, lead])
        rows.append({"lead_hour": lead + 1, **{f"residual_{k}": val for k, val in metrics.items()}})
    return pd.DataFrame(rows)


def summarize_extreme_metrics(true_states: np.ndarray, pred_states: np.ndarray) -> dict[str, float]:
    true_eta = true_states[..., 0]
    pred_eta = pred_states[..., 0]
    flat_true = true_eta.reshape(-1)
    flat_pred = pred_eta.reshape(-1)
    out = {}
    for q in [0.90, 0.95]:
        thr = float(np.quantile(np.abs(flat_true), q))
        mask = np.abs(flat_true) >= thr
        if mask.sum() >= 10:
            metrics = v2.regression_metrics(flat_true[mask], flat_pred[mask])
            for k, val in metrics.items():
                out[f"extreme_abs_q{int(q*100)}_residual_{k}"] = val
            out[f"extreme_abs_q{int(q*100)}_threshold"] = thr
            out[f"extreme_abs_q{int(q*100)}_n"] = int(mask.sum())
    return out


def run_one(args, arrays: dict[str, np.ndarray], coops_cols: list[str], horizon: int, mode: str, device) -> dict[str, float]:
    feature_cols, physics_cols = make_feature_sets(coops_cols)
    x_raw = np.stack([arrays[c] for c in feature_cols], axis=-1).astype(np.float32)
    physics_raw = np.stack([arrays[c] for c in physics_cols], axis=-1).astype(np.float32)
    states = np.stack([arrays["residual"], arrays["uo"], arrays["vo"], arrays["wave_setup_proxy"]], axis=-1).astype(np.float32)
    tide = arrays["tide"].astype(np.float32)

    n_time, nodes, feats = x_raw.shape
    train_end = int(n_time * args.train_ratio)
    val_end = int(n_time * (args.train_ratio + args.val_ratio))
    graph_priors = v2.make_graph_priors(arrays, train_end)

    x_scaler = StandardScaler()
    x_scaler.fit(x_raw[:train_end].reshape(-1, feats))
    x_scaled = x_scaler.transform(x_raw.reshape(-1, feats)).reshape(n_time, nodes, feats).astype(np.float32)
    phys_scaler = StandardScaler()
    phys_scaler.fit(physics_raw[:train_end].reshape(-1, len(physics_cols)))
    phys_scaled = phys_scaler.transform(physics_raw.reshape(-1, len(physics_cols))).reshape(n_time, nodes, len(physics_cols)).astype(np.float32)

    state_scale = np.std(states[:train_end].reshape(-1, len(v2.STATE_NAMES)), axis=0).astype(np.float32) + 1e-6
    delta_scale = np.std((states[1:train_end] - states[: train_end - 1]).reshape(-1, len(v2.STATE_NAMES)), axis=0).astype(np.float32) + 1e-6

    train_ds = v2.MultistateWindowDataset(
        x_scaled,
        states,
        tide,
        phys_scaled,
        args.window,
        horizon,
        0,
        train_end,
        args.physics_forcing_mode,
        args.train_stride,
    )
    val_ds = v2.MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, train_end, val_end, args.physics_forcing_mode, 1
    )
    test_ds = v2.MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, val_end, n_time, args.physics_forcing_mode, 1
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    graph_init = [0.50, 0.35, 0.15]
    model = v2.MultistateGNNBiGRU(
        input_dim=feats,
        graph_priors=graph_priors,
        graph_init_weights=graph_init,
        gnn_hidden=args.gnn_hidden,
        gru_hidden=args.gru_hidden,
        horizon=horizon,
        dropout=args.dropout,
        num_states=len(v2.STATE_NAMES),
    ).to(device)
    physics_ode = v2.MultistatePhysicsODE(nodes, len(physics_cols), len(v2.STATE_NAMES)).to(device)

    train_args = argparse.Namespace(**vars(args))
    train_args.physics_lambda_max = 0.0 if mode == "no_physics" else args.physics_lambda_max
    train_args.physics_state_weights = MODE_WEIGHTS[mode]
    print("\n" + "=" * 80)
    print(
        f"V3 mode={mode} horizon={horizon} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"features={len(feature_cols)} physics={len(physics_cols)} lambda={train_args.physics_lambda_max}"
    )

    history, best_val = v2.train_model(
        model=model,
        physics_ode=physics_ode,
        train_loader=train_loader,
        val_loader=val_loader,
        args=train_args,
        state_scale=state_scale,
        delta_scale=delta_scale,
        device=device,
    )
    pred_states, true_states, target_tide, test_physics_loss = v2.predict(model, physics_ode, test_loader, device)
    metrics = v2.summarize_metrics(true_states, pred_states, target_tide)
    metrics.update(summarize_extreme_metrics(true_states, pred_states))
    weights = model.graph.weight_dict()

    row = {
        "model": "enhanced_forcing_physics_loss_ablation_v3",
        "mode": mode,
        "horizon": horizon,
        "training_mode": f"train_stride{args.train_stride}_full_val_test",
        "num_features": len(feature_cols),
        "num_physics_features": len(physics_cols),
        "physics_lambda_max": train_args.physics_lambda_max,
        "physics_state_weights": ",".join(str(x) for x in train_args.physics_state_weights),
        "best_val_score": best_val,
        "test_physics_loss_unscaled": test_physics_loss,
        "learned_w_identity": weights["identity"],
        "learned_w_distance": weights["distance"],
        "learned_w_corr": weights["corr"],
        **metrics,
    }

    run_dir = Path(args.output_dir) / f"horizon_{horizon}h" / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
    summarize_lead_metrics(true_states, pred_states, horizon).to_csv(run_dir / "per_lead_metrics.csv", index=False)
    np.savez_compressed(
        run_dir / "predictions.npz",
        pred_states=pred_states,
        true_states=true_states,
        target_tide=target_tide,
        target_start_times=np.array([str(arrays["time"][int(i)]) for i in test_ds.indices]),
        station_ids=np.array(v2.STATION_IDS),
        state_names=np.array(v2.STATE_NAMES),
        feature_cols=np.array(feature_cols),
        physics_cols=np.array(physics_cols),
    )
    print(
        f"Done mode={mode} horizon={horizon}: seq_R2={row['seq_residual_R2']:.4f}, "
        f"last_R2={row['last_residual_R2']:.4f}, last_RMSE={row['last_residual_RMSE']:.4f}"
    )
    return row


def plot_outputs(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summary.sort_values(["horizon", "mode"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for mode, sub in summary.groupby("mode"):
        sub = sub.sort_values("horizon")
        axes[0].plot(sub["horizon"], sub["last_residual_R2"], marker="o", label=mode)
        axes[1].plot(sub["horizon"], sub["last_residual_RMSE"], marker="o", label=mode)
    axes[0].axhline(0.9, color="crimson", linestyle="--", linewidth=1.2, label="R2=0.90 target")
    axes[0].set_xlabel("Forecast horizon (hours)")
    axes[0].set_ylabel("Last-step residual R2")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Forecast horizon (hours)")
    axes[1].set_ylabel("Last-step residual RMSE (m)")
    axes[1].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("V3 enhanced forcing physics-loss ablation")
    fig.tight_layout()
    fig.savefig(output_dir / "v3_ablation_last_step_comparison.png", dpi=220)
    plt.close(fig)

    if {"no_physics", "eta_uvW"}.issubset(set(summary["mode"])):
        base = summary[summary["mode"] == "no_physics"].set_index("horizon")
        phys = summary[summary["mode"] == "eta_uvW"].set_index("horizon")
        common = sorted(set(base.index) & set(phys.index))
        if common:
            imp = pd.DataFrame({
                "horizon": common,
                "last_R2_gain_eta_uvW_vs_no_physics": [phys.loc[h, "last_residual_R2"] - base.loc[h, "last_residual_R2"] for h in common],
                "last_RMSE_reduction_pct": [
                    100.0 * (base.loc[h, "last_residual_RMSE"] - phys.loc[h, "last_residual_RMSE"]) / base.loc[h, "last_residual_RMSE"]
                    for h in common
                ],
            })
            imp.to_csv(output_dir / "v3_physics_gain_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 enhanced forcing physics-loss ablation")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--horizons", type=int, nargs="+", default=[12, 24])
    parser.add_argument("--modes", nargs="+", default=["no_physics", "eta_only", "eta_uv", "eta_uvW"], choices=list(MODE_WEIGHTS))
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--gnn-hidden", type=int, default=40)
    parser.add_argument("--gru-hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--graph-lr-mult", type=float, default=3.0)
    parser.add_argument("--physics-lr-mult", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--physics-lambda-max", type=float, default=0.003)
    parser.add_argument("--physics-warmup-epochs", type=int, default=8)
    parser.add_argument("--physics-ramp-epochs", type=int, default=14)
    parser.add_argument("--physics-loss-type", default="huber", choices=["huber", "mse"])
    parser.add_argument("--physics-forcing-mode", default="last_input", choices=["last_input", "future"])
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.20)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument("--selection-metric", default="val_eta_data_loss",
                        choices=["val_eta_data_loss", "val_data_loss", "val_last_loss", "val_total_loss"])
    args = parser.parse_args()

    v2.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arrays, station_meta, coops_cols = build_enhanced_arrays()
    station_meta.to_csv(output_dir / "station_meta_used.csv", index=False)
    pd.DataFrame({"coops_enhanced_feature": coops_cols}).to_csv(output_dir / "coops_features_used.csv", index=False)
    print(f"Using CO-OPS enhanced features: {coops_cols}")
    print(f"Device: {device}")

    rows = []
    for horizon in args.horizons:
        for mode in args.modes:
            rows.append(run_one(args, arrays, coops_cols, horizon, mode, device))
    summary = pd.DataFrame(rows).sort_values(["horizon", "mode"])
    summary.to_csv(output_dir / "v3_ablation_metrics.csv", index=False)
    key_cols = [
        "horizon",
        "mode",
        "seq_residual_RMSE",
        "seq_residual_R2",
        "last_residual_RMSE",
        "last_residual_R2",
        "seq_sea_level_R2",
        "last_sea_level_R2",
        "extreme_abs_q95_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "learned_w_identity",
        "learned_w_distance",
        "learned_w_corr",
    ]
    existing = [c for c in key_cols if c in summary.columns]
    summary[existing].to_csv(output_dir / "v3_ablation_key_metrics.csv", index=False)
    plot_outputs(summary, output_dir)
    print("\nFinished V3 ablation.")
    print(summary[existing].to_string(index=False))


if __name__ == "__main__":
    main()
