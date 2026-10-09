from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT74 = Path(__file__).resolve().parent / "74_multistate_physics_loss_gnn_bigru_v2.py"
SCRIPT78 = Path(__file__).resolve().parent / "78_final_four_models_enhanced_data.py"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "causal_preprocessing_robustness"
COOPS_PATH = ROOT / "data" / "processed_multiyear_2023_2025" / "noaa_coops_met_station_hourly.csv"

spec74 = importlib.util.spec_from_file_location("v2_impl", SCRIPT74)
v2 = importlib.util.module_from_spec(spec74)
assert spec74.loader is not None
spec74.loader.exec_module(v2)

spec78 = importlib.util.spec_from_file_location("final_impl", SCRIPT78)
final = importlib.util.module_from_spec(spec78)
assert spec78.loader is not None
spec78.loader.exec_module(final)


def make_hourly_feature_table_causal() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build the hourly feature table with causal-only gap filling.

    Difference from the earlier offline table:
    - no time interpolation between past and future endpoints;
    - no backward fill;
    - lower-frequency current/wave data are carried forward only;
    - remaining leading missing values are filled with 0.0 instead of future values.
    """

    water = v2.load_long_csv("water_tide_residual_long.csv")
    era5 = v2.load_long_csv("era5_station_hourly.csv")
    currents = v2.load_long_csv("surface_currents_station_daily.csv")
    waves = v2.load_long_csv("wave_direction_speed_station_3hourly.csv")
    depth = pd.read_csv(v2.DEPTH_PATH)
    depth["station_id"] = depth["station_id"].astype(str)

    base = water[["datetime", "station_id", "water_level", "tide", "residual", "sigma"]].copy()
    base = base.sort_values(["station_id", "datetime"])

    era5_cols = [
        "datetime",
        "station_id",
        "u10",
        "v10",
        "msl",
        "wind_speed",
        "wind_stress_u_proxy",
        "wind_stress_v_proxy",
    ]
    base = base.merge(era5[era5_cols], on=["datetime", "station_id"], how="left")

    current_cols = ["datetime", "station_id", "uo", "vo", "current_speed_mps"]
    wave_cols = [
        "datetime",
        "station_id",
        "wave_height",
        "wave_peak_period",
        "wave_mean_period",
        "wave_direction",
        "wave_stokes_drift_x",
        "wave_stokes_drift_y",
        "wave_speed_from_peak_period_mps",
        "wave_dir_x_from_mps",
        "wave_dir_y_from_mps",
    ]

    frames = []
    for sid in v2.STATION_IDS:
        b = base[base["station_id"] == sid].set_index("datetime").sort_index()
        c = currents[currents["station_id"] == sid][current_cols].set_index("datetime").sort_index()
        w = waves[waves["station_id"] == sid][wave_cols].set_index("datetime").sort_index()
        b = b.join(c.drop(columns=["station_id"]), how="left")
        b = b.join(w.drop(columns=["station_id"]), how="left")

        causal_cols = [
            "uo",
            "vo",
            "current_speed_mps",
            "wave_height",
            "wave_peak_period",
            "wave_mean_period",
            "wave_direction",
            "wave_stokes_drift_x",
            "wave_stokes_drift_y",
            "wave_speed_from_peak_period_mps",
            "wave_dir_x_from_mps",
            "wave_dir_y_from_mps",
        ]
        b[causal_cols] = b[causal_cols].ffill().fillna(0.0)
        b["station_id"] = sid
        frames.append(b.reset_index())

    df = pd.concat(frames, ignore_index=True)
    df = df.merge(depth[["station_id", "depth", "station_lat", "station_lon"]], on="station_id", how="left")

    # Use only information available up to the current timestamp. The previous
    # implementation used the full-period mean, which leaked future climate
    # information into the pressure anomaly feature.
    causal_msl_mean = df.groupby("station_id")["msl"].transform(
        lambda series: series.expanding(min_periods=1).mean().shift(1)
    )
    causal_msl_mean = causal_msl_mean.fillna(df["msl"])
    df["pressure_anom"] = df["msl"] - causal_msl_mean
    df["inverse_depth"] = 1.0 / np.maximum(df["depth"], 0.5)
    df["wave_energy"] = df["wave_height"] ** 2
    df["wave_energy_flux"] = df["wave_height"] ** 2 * df["wave_speed_from_peak_period_mps"]
    df["wind_wave_alignment"] = (
        df["wind_stress_u_proxy"] * df["wave_dir_x_from_mps"]
        + df["wind_stress_v_proxy"] * df["wave_dir_y_from_mps"]
    )
    df["wave_setup_proxy"] = df["wave_energy"] * df["inverse_depth"]

    station_meta = (
        df[["station_id", "station_lat", "station_lon", "depth"]]
        .drop_duplicates("station_id")
        .set_index("station_id")
        .loc[v2.STATION_IDS]
        .reset_index()
    )
    adj = v2.build_distance_graph(station_meta)
    return df, station_meta, adj


def merge_coops_features_causal(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Merge local CO-OPS variables using causal-only forward filling."""

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
        out[col] = out.groupby("station_id", group_keys=False)[col].transform(lambda s: s.ffill(limit=6))
        out[col] = out[col].fillna(0.0)

    if "coops_air_pressure" in out.columns and "msl" in out.columns:
        out["coops_minus_era5_pressure_hpa"] = out["coops_air_pressure"] - out["msl"] / 100.0
        available.append("coops_minus_era5_pressure_hpa")
    return out, available


def build_enhanced_arrays_causal() -> tuple[dict[str, np.ndarray], pd.DataFrame, list[str]]:
    df, station_meta, adj = make_hourly_feature_table_causal()
    df, coops_cols = merge_coops_features_causal(df)
    arrays = v2.build_arrays(df, adj)
    final.v3.add_arrays_from_df(arrays, df, coops_cols)
    return arrays, station_meta, coops_cols


def write_preprocessing_audit(output_dir: Path, arrays: dict[str, np.ndarray], coops_cols: list[str]) -> None:
    rows = []
    for key, value in arrays.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in {"f", "i"} and value.ndim >= 1:
            rows.append(
                {
                    "array": key,
                    "shape": "x".join(map(str, value.shape)),
                    "nan_count": int(np.isnan(value).sum()) if value.dtype.kind == "f" else 0,
                    "zero_count": int((value == 0).sum()) if value.dtype.kind in {"f", "i"} else 0,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "causal_preprocessing_array_audit.csv", index=False)
    pd.DataFrame({"coops_feature": coops_cols}).to_csv(output_dir / "causal_coops_features_used.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final models with strict causal preprocessing/imputation")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--horizons", type=int, nargs="+", default=[24])
    parser.add_argument("--models", nargs="+", default=["gnn_bigru", "physical_loss"])
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", default="distance", choices=["identity", "distance", "corr"])
    parser.add_argument("--gnn-hidden", type=int, default=40)
    parser.add_argument("--gru-hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=9)
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
    parser.add_argument("--physics-lambda-max", type=float, default=0.0003)
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.2)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument(
        "--selection-metric",
        default="val_eta_data_loss",
        choices=["val_eta_data_loss", "val_data_loss", "val_last_loss", "val_total_loss"],
    )
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Monkey-patch the final runner so that all model variants use the same strict causal arrays.
    final.v3.build_enhanced_arrays = build_enhanced_arrays_causal
    final.OUT_DIR = output_dir
    args.output_dir = str(output_dir)

    arrays, _, coops_cols = build_enhanced_arrays_causal()
    write_preprocessing_audit(output_dir, arrays, coops_cols)

    final.v2.set_seed(args.seed)
    device = final.torch.device("cuda" if final.torch.cuda.is_available() else "cpu")
    rows = []
    for horizon in args.horizons:
        if "gnn_bigru" in args.models:
            rows.append(final.run_single(args, horizon, "gnn_bigru", "GNN-BiGRU", False, device))
        if "learnable_graph" in args.models:
            rows.append(final.run_single(args, horizon, "learnable_graph", "Learnable-graph GNN-BiGRU", False, device))
        if "ode_based_learnable" in args.models:
            rows.append(final.run_single(args, horizon, "ode_based_learnable", "ODE-based learnable GNN-BiGRU", True, device))
        if "physical_loss" in args.models:
            rows.append(final.run_physical_loss(args, horizon, device))
        pd.DataFrame(rows).to_csv(output_dir / "causal_final_metrics_partial.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "causal_final_metrics.csv", index=False)
    key_cols = [
        "horizon",
        "model_name",
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "seq_sea_level_R2",
        "last_sea_level_R2",
        "learned_w_identity",
        "learned_w_distance",
        "learned_w_corr",
    ]
    existing = [c for c in key_cols if c in summary.columns]
    summary[existing].sort_values(["horizon", "last_residual_R2"], ascending=[True, False]).to_csv(
        output_dir / "causal_final_key_metrics.csv", index=False
    )

    print("\nFinished strict causal preprocessing robustness run.")
    print(summary[existing].sort_values(["horizon", "last_residual_R2"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
