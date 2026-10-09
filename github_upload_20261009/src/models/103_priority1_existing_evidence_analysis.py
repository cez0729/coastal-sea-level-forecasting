from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "outputs" / "publication_priority1_complete"
FINAL_RESULTS = ROOT / "results" / "final"
GRAPH_RESULTS = ROOT / "results" / "priority1_graph_baselines"

STATION_NAMES = {
    "8461490": "New London",
    "8510560": "Montauk",
    "8516945": "Kings Point",
    "8518750": "The Battery",
    "8531680": "Sandy Hook",
    "8534720": "Atlantic City",
    "8536110": "Cape May",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


final4 = load_module("final4_p103", HERE / "78_final_four_models_enhanced_data.py")
priority1 = load_module("priority1_p103", HERE / "101_priority1_publication_experiments.py")
v2 = final4.v2


def r2_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if not len(y):
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan}
    mse = float(np.mean((y - p) ** 2))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "R2": float(1.0 - np.sum((y - p) ** 2) / denom) if denom > 1e-12 else np.nan,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(y - p))),
    }


def parse_seed(path: Path) -> int:
    matches = re.findall(r"seed[_-](\d+)", str(path).replace("\\", "/"))
    if not matches:
        raise ValueError(f"Cannot parse seed from {path}")
    return int(matches[-1])


def load_residual_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "pred_states" in data.files:
        return data["pred_states"][..., 0], data["true_states"][..., 0]
    return data["pred_residual"], data["true_residual"]


def station_analysis(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = list(FINAL_RESULTS.glob("seed_*/horizon_24h/*/predictions.npz"))
    paths += list(GRAPH_RESULTS.glob("seed_*/horizon_24h/*/predictions.npz"))
    rows = []
    station_ids = list(v2.STATION_IDS)
    for path in sorted(paths):
        pred, true = load_residual_prediction(path)
        if pred.shape[-1] != 24:
            continue
        for station_index, station_id in enumerate(station_ids[: pred.shape[1]]):
            metrics = r2_rmse_mae(true[:, station_index, -1], pred[:, station_index, -1])
            rows.append(
                {
                    "seed": parse_seed(path),
                    "model_key": path.parent.name,
                    "station_index": station_index,
                    "station_id": str(station_id),
                    "station_name": STATION_NAMES.get(str(station_id), str(station_id)),
                    "last_residual_R2": metrics["R2"],
                    "last_residual_RMSE": metrics["RMSE"],
                    "last_residual_MAE": metrics["MAE"],
                }
            )
    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(out_dir / "station_metrics_all_models_all_seeds.csv", index=False)
    metrics = ["last_residual_R2", "last_residual_RMSE", "last_residual_MAE"]
    summary = all_runs.groupby(["model_key", "station_id", "station_name"])[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x) for col in summary.columns.to_flat_index()]
    summary.to_csv(out_dir / "station_metrics_all_models_mean_std.csv", index=False)

    pivot = all_runs.pivot_table(index=["seed", "station_id", "station_name"], columns="model_key", values="last_residual_R2").reset_index()
    if {"physical_loss", "gnn_bigru"}.issubset(pivot.columns):
        pivot["delta_physics_minus_gnn"] = pivot["physical_loss"] - pivot["gnn_bigru"]
    if {"physical_loss", "ode_based_learnable"}.issubset(pivot.columns):
        pivot["delta_physics_minus_ode"] = pivot["physical_loss"] - pivot["ode_based_learnable"]
    pivot.to_csv(out_dir / "station_paired_physics_deltas_all_seeds.csv", index=False)
    delta_cols = [c for c in pivot.columns if c.startswith("delta_")]
    delta_summary = pivot.groupby(["station_id", "station_name"])[delta_cols].agg(["mean", "std", "count"]).reset_index()
    delta_summary.columns = ["_".join(str(x) for x in col if x) for col in delta_summary.columns.to_flat_index()]
    delta_summary.to_csv(out_dir / "station_paired_physics_deltas_mean_std.csv", index=False)

    if not delta_summary.empty and "delta_physics_minus_gnn_mean" in delta_summary:
        plot = delta_summary.sort_values("delta_physics_minus_gnn_mean")
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        colors = np.where(plot["delta_physics_minus_gnn_mean"] >= 0, "#17806D", "#C44E52")
        ax.barh(
            plot["station_name"],
            plot["delta_physics_minus_gnn_mean"],
            xerr=plot["delta_physics_minus_gnn_std"],
            color=colors,
            capsize=3,
        )
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_xlabel("Paired 24-h terminal R2 difference: physical loss - plain GNN")
        ax.set_title("Physics benefit is station dependent")
        ax.grid(axis="x", alpha=0.22)
        fig.tight_layout()
        fig.savefig(out_dir / "station_paired_physics_delta.png", dpi=260)
        plt.close(fig)
    return all_runs, delta_summary


def coefficient_analysis(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted(FINAL_RESULTS.glob("seed_*/horizon_24h/physical_loss/training_log.csv")):
        data = pd.read_csv(path)
        best = data.loc[data["selection_score"].idxmin()].copy()
        row = {"seed": parse_seed(path), "best_epoch": int(best["epoch"]), "selection_score": float(best["selection_score"])}
        for col in data.columns:
            if col.endswith(("_decay", "_kappa", "_mean_bias")) or "_beta_" in col:
                row[col] = float(best[col])
        rows.append(row)
    best_rows = pd.DataFrame(rows).sort_values("seed")
    best_rows.to_csv(out_dir / "physics_coefficients_best_validation_epoch_all_seeds.csv", index=False)

    scalar_cols = [c for c in best_rows.columns if c.endswith(("_decay", "_kappa", "_mean_bias"))]
    scalar_summary = best_rows[scalar_cols].agg(["mean", "std", "min", "max"]).T.reset_index(names="coefficient")
    scalar_summary["coefficient_of_variation_abs"] = scalar_summary["std"] / scalar_summary["mean"].abs().clip(lower=1e-12)
    scalar_summary.to_csv(out_dir / "physics_coefficient_stability_summary.csv", index=False)

    beta_rows = []
    for seed_row in rows:
        for state in v2.STATE_NAMES:
            values = np.asarray([value for key, value in seed_row.items() if key.startswith(f"{state}_beta_")], dtype=float)
            beta_rows.append(
                {
                    "seed": seed_row["seed"],
                    "state": state,
                    "beta_l2": float(np.linalg.norm(values)),
                    "beta_mean": float(np.mean(values)),
                    "beta_abs_mean": float(np.mean(np.abs(values))),
                    "beta_min": float(np.min(values)),
                    "beta_max": float(np.max(values)),
                }
            )
    beta_df = pd.DataFrame(beta_rows)
    beta_df.to_csv(out_dir / "physics_beta_block_summary_all_seeds.csv", index=False)
    beta_summary = beta_df.groupby("state")[["beta_l2", "beta_mean", "beta_abs_mean", "beta_min", "beta_max"]].agg(["mean", "std"]).reset_index()
    beta_summary.columns = ["_".join(str(x) for x in col if x) for col in beta_summary.columns.to_flat_index()]
    beta_summary.to_csv(out_dir / "physics_beta_block_summary_mean_std.csv", index=False)

    if not scalar_summary.empty:
        plot = scalar_summary[scalar_summary["coefficient"].str.endswith(("_decay", "_kappa"))].copy()
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        x = np.arange(len(plot))
        ax.bar(x, plot["mean"], yerr=plot["std"], capsize=4, color="#2A6FBB")
        ax.set_xticks(x)
        ax.set_xticklabels(plot["coefficient"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Learned coefficient in standardized-state dynamics")
        ax.set_title("Learned damping and graph coupling are stable across five seeds")
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(out_dir / "physics_coefficient_stability.png", dpi=260)
        plt.close(fig)
    return best_rows, scalar_summary


def build_data_args() -> SimpleNamespace:
    return SimpleNamespace(
        train_ratio=0.70,
        val_ratio=0.15,
        window=24,
        train_stride=8,
        physics_forcing_mode="last_input",
        extreme_quantile=0.90,
    )


def parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


@torch.inference_mode()
def profile_forward(model: torch.nn.Module, shape: tuple[int, ...], repeats: int) -> tuple[float, float]:
    model.eval().cpu()
    x = torch.zeros(shape, dtype=torch.float32)
    for _ in range(3):
        model(x)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(x)
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples)), float(np.quantile(samples, 0.95))


def scale_and_complexity_analysis(out_dir: Path, repeats: int, cpu_threads: int) -> pd.DataFrame:
    torch.set_num_threads(max(1, cpu_threads))
    args = build_data_args()
    data = final4.build_enhanced_data(args, 24, add_ode_prior=False)
    data_ode = final4.build_enhanced_data(args, 24, add_ode_prior=True)
    adj = data["graph_priors"]["distance"]
    models = {
        "gnn_bigru": final4.FixedGraphGNNBiGRU(data["feats"], adj, 40, 48, 24, 0.12),
        "learnable_graph": final4.LearnableGraphSingleStateGNNBiGRU(data["feats"], data["graph_priors"], [0.50, 0.35, 0.15], 40, 48, 24, 0.12),
        "ode_based_learnable": final4.LearnableGraphSingleStateGNNBiGRU(data_ode["feats"], data_ode["graph_priors"], [0.50, 0.35, 0.15], 40, 48, 24, 0.12),
        "physical_loss": v2.MultistateGNNBiGRU(data["feats"], data["graph_priors"], [0.50, 0.35, 0.15], 40, 48, 24, 0.12, 4),
        "dcrnn": priority1.DCRNNForecaster(data["feats"], adj, 64, 24, 2, 0.15),
        "graph_wavenet": priority1.GraphWaveNetForecaster(data["feats"], adj, 64, 24, 2, 6, 0.15),
    }
    ode_loss_module = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4)
    rows = []
    for key, model in models.items():
        feats = data_ode["feats"] if key == "ode_based_learnable" else data["feats"]
        total, trainable = parameter_count(model)
        median, p95 = profile_forward(model, (1, 24, data["nodes"], feats), repeats)
        receptive = 19 if key == "graph_wavenet" else 24
        rows.append(
            {
                "model_key": key,
                "predictor_parameters_total": total,
                "predictor_parameters_trainable": trainable,
                "training_only_physics_parameters": parameter_count(ode_loss_module)[1] if key == "physical_loss" else 0,
                "effective_history_hours": receptive,
                "cpu_batch1_forward_median_ms": median,
                "cpu_batch1_forward_p95_ms": p95,
                "cpu_threads": cpu_threads,
                "input_features": feats,
            }
        )
    complexity = pd.DataFrame(rows)
    complexity.to_csv(out_dir / "model_complexity_and_cpu_inference.csv", index=False)

    scale_rows = []
    units = {"eta": "m", "u": "m s^-1", "v": "m s^-1", "W": "m (wave-setup proxy)"}
    for index, state in enumerate(v2.STATE_NAMES):
        scale_rows.append(
            {
                "state": state,
                "native_unit": units[state],
                "data_loss_scale_train_std": float(data["state_scale"][index]),
                "physics_residual_scale_train_delta_std": float(data["delta_scale"][index]),
                "physics_state_weight": [1.0, 0.35, 0.35, 0.25][index],
                "interpretation_boundary": "loss is dimensionless after train-only scaling; learned coefficients are not calibrated physical constants",
            }
        )
    pd.DataFrame(scale_rows).to_csv(out_dir / "physics_state_scale_and_units.csv", index=False)
    pd.DataFrame({"feature": data["feature_cols"]}).to_csv(out_dir / "model_input_features_exact.csv", index=False)
    pd.DataFrame({"physics_forcing_feature": data["physics_cols"]}).to_csv(out_dir / "physics_forcing_features_exact.csv", index=False)
    return complexity


def write_summary(out_dir: Path, delta_summary: pd.DataFrame, coeff_summary: pd.DataFrame, complexity: pd.DataFrame) -> None:
    lines = [
        "# Publication Priority-1 Existing-Evidence Audit",
        "",
        "## Interpretation lock",
        "",
        "- Coefficients are learned in standardized-state dynamics. Cross-seed stability is evidence of optimization stability, not proof of dimensional physical validity.",
        "- Per-station deltas are paired across the same seeds and test period. They reveal heterogeneity but do not identify a causal geographic mechanism.",
        "- CPU timing is a reproducible local forward-pass profile, not a substitute for GPU training-time reporting.",
        "",
        "## Station-level paired physics effects",
        "",
        delta_summary.to_string(index=False),
        "",
        "## Learned coefficient stability",
        "",
        coeff_summary.to_string(index=False),
        "",
        "## Model complexity",
        "",
        complexity.to_string(index=False),
    ]
    (out_dir / "PRIORITY1_EXISTING_EVIDENCE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete the no-retraining Priority-1 publication evidence audit.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--profile-repeats", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, delta_summary = station_analysis(out_dir)
    _, coeff_summary = coefficient_analysis(out_dir)
    complexity = scale_and_complexity_analysis(out_dir, args.profile_repeats, args.cpu_threads)
    write_summary(out_dir, delta_summary, coeff_summary, complexity)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Priority-1 existing-evidence analysis saved to {out_dir}")


if __name__ == "__main__":
    main()
