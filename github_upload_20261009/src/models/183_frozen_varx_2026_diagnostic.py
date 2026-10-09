"""Diagnostic-only frozen VARX evaluation on already-viewed 2026 periods."""
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
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "results" / "frozen_varx_2026_diagnostic_20260811"
SEEDS = [42, 123, 2024, 2025, 3407]
DEEP_MODELS = ["gwn_eta_only", "gwn_multistate_no_physics", "hs_dt_gwn"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p180 = load_module("frozen_varx_p180", HERE / "180_strict_varx_chronological_validation.py")


def fold_args(data_dir: str, test_end: str) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=data_dir,
        fold_train_end="2025-01-01",
        fold_val_end="2025-07-01",
        fold_test_end=test_end,
        window=24,
        horizon=24,
        train_stride=8,
        physics_forcing_mode="last_input",
        extreme_quantile=0.90,
    )


def build_data(data_dir: str, test_end: str):
    args = fold_args(data_dir, test_end)
    p180.p134.configure_data_dir(args.data_dir)
    return p180.p134.rolling.build_fold_data(args, args.horizon, add_ode_prior=False)


def fit_frozen_varx(data, alpha_grid: list[float]):
    train_x, train_y, _ = p180.p170.design_matrix(data["single_train"])
    val_x, val_y, _ = p180.p170.design_matrix(data["single_val"])
    rows = []
    best_alpha, best_mse = None, float("inf")
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, solver="lsqr").fit(train_x, train_y)
        mse = float(np.mean((model.predict(val_x) - val_y) ** 2))
        rows.append({"alpha": alpha, "validation_MSE": mse})
        if mse < best_mse:
            best_alpha, best_mse = float(alpha), mse
    model = Ridge(alpha=best_alpha, solver="lsqr").fit(train_x, train_y)
    return model, best_alpha, pd.DataFrame(rows), train_x, train_y, val_x, val_y


def period_prediction(model, data, start: str, end: str):
    x, y, tide = p180.p170.design_matrix(data["single_test"])
    prediction = model.predict(x).reshape(-1, 7, 24).astype(np.float32)
    true = y.reshape(-1, 7, 24).astype(np.float32)
    tide = tide.astype(np.float32)
    indices = np.asarray(data["single_test"].indices, dtype=np.int64)
    times = pd.to_datetime(data["arrays"]["time"])[indices].to_numpy(dtype="datetime64[ns]")
    mask = (times >= np.datetime64(start)) & (times < np.datetime64(end))
    return {"pred": prediction[mask], "true": true[mask], "tide": tide[mask], "time": times[mask]}


def load_deep(period: str, seed: int) -> dict[str, np.ndarray]:
    if period == "2026_h1":
        path = ROOT / "results" / "frozen_2026_h1_physics_reliability_gwn" / f"seed_{seed}_test_frozen_features.npz"
        with np.load(path) as payload:
            return {
                "gwn_eta_only": payload["eta"],
                "gwn_multistate_no_physics": payload["multi"],
                "hs_dt_gwn": np.concatenate([0.5 * (payload["eta"][..., :23] + payload["multi"][..., :23]), payload["multi"][..., 23:]], axis=-1),
                "true": payload["target"],
                "tide": payload["tide"],
                "time": payload["forecast_start"],
            }
    path = ROOT / "results" / "formal_hsdt_independent_validation_2026" / f"seed_{seed}_july_base_predictions.npz"
    with np.load(path) as payload:
        return {
            "gwn_eta_only": payload["pred_eta"],
            "gwn_multistate_no_physics": payload["pred_multi"],
            "hs_dt_gwn": payload["pred_hsdt"],
            "true": payload["true_residual"],
            "tide": payload["target_tide"],
            "time": payload["forecast_start"],
        }


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    denominator = float(np.sum((true - true.mean()) ** 2))
    return float(1.0 - np.sum((true - pred) ** 2) / max(denominator, 1e-12))


def block_ci(values: np.ndarray, block: int, replicates: int, rng: np.random.Generator):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    starts = np.arange(max(1, n - block + 1))
    blocks_needed = int(np.ceil(n / block))
    offsets = np.arange(block)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = (chosen[:, None] + offsets[None, :]).reshape(-1)[:n]
        estimates[index] = values[sample].mean()
    return float(values.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--block-hours", type=int, default=168)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    h1_data = build_data("data/processed_multiyear_2023_2026_h1_frozen", "2026-07-01")
    july_data = build_data("data/processed_multiyear_2023_2026_july_frozen", "2026-08-01")
    model, alpha, alpha_table, train_x, train_y, val_x, val_y = fit_frozen_varx(h1_data, args.alpha_grid)
    alpha_table.to_csv(output / "validation_alpha_selection.csv", index=False)

    july_train_x, july_train_y, _ = p180.p170.design_matrix(july_data["single_train"])
    july_val_x, july_val_y, _ = p180.p170.design_matrix(july_data["single_val"])
    if not (
        np.allclose(train_x, july_train_x, atol=1e-7, rtol=0)
        and np.allclose(train_y, july_train_y, atol=1e-7, rtol=0)
        and np.allclose(val_x, july_val_x, atol=1e-7, rtol=0)
        and np.allclose(val_y, july_val_y, atol=1e-7, rtol=0)
    ):
        raise RuntimeError("The frozen training or validation design changed in the July data extension")

    varx_periods = {
        "2026_h1": period_prediction(model, h1_data, "2026-01-01", "2026-07-01"),
        "2026_july": period_prediction(model, july_data, "2026-07-01", "2026-08-01"),
    }
    metric_rows, paired_rows, lead_rows, bootstrap_rows = [], [], [], []
    rng = np.random.default_rng(20260811)
    for period, varx_payload in varx_periods.items():
        metric_rows.append({"period": period, "seed": 0, "model": "varx_ridge", **p180.p170.summarize(varx_payload["true"], varx_payload["pred"], varx_payload["tide"])})
        for lead in range(24):
            lead_rows.append({"period": period, "seed": 0, "model": "varx_ridge", "lead": lead + 1, "residual_R2": r2(varx_payload["true"][..., lead], varx_payload["pred"][..., lead])})
        for seed in SEEDS:
            deep = load_deep(period, seed)
            if not np.array_equal(varx_payload["time"], deep["time"]):
                raise RuntimeError(f"Timestamp mismatch for {period}, seed {seed}")
            if not np.allclose(varx_payload["true"], deep["true"], atol=1e-6, rtol=0):
                raise RuntimeError(f"Target mismatch for {period}, seed {seed}")
            for name in DEEP_MODELS:
                prediction = deep[name].astype(np.float32)
                metrics = p180.p170.summarize(deep["true"], prediction, deep["tide"])
                metric_rows.append({"period": period, "seed": seed, "model": name, **metrics})
                varx_metrics = p180.p170.summarize(varx_payload["true"], varx_payload["pred"], varx_payload["tide"])
                paired_rows.append({
                    "period": period,
                    "seed": seed,
                    "deep_model": name,
                    "varx_minus_deep_sequence_R2": varx_metrics["seq_residual_R2"] - metrics["seq_residual_R2"],
                    "varx_minus_deep_lead24_R2": varx_metrics["last_residual_R2"] - metrics["last_residual_R2"],
                    "varx_minus_deep_q95_R2": varx_metrics["extreme_abs_q95_residual_R2"] - metrics["extreme_abs_q95_residual_R2"],
                })
                for lead in range(24):
                    lead_rows.append({"period": period, "seed": seed, "model": name, "lead": lead + 1, "residual_R2": r2(deep["true"][..., lead], prediction[..., lead])})
                sequence_reduction = np.mean((prediction - deep["true"]) ** 2 - (varx_payload["pred"] - deep["true"]) ** 2, axis=(1, 2))
                lead24_reduction = np.mean((prediction[..., -1] - deep["true"][..., -1]) ** 2 - (varx_payload["pred"][..., -1] - deep["true"][..., -1]) ** 2, axis=1)
                for metric, values in (("sequence_MSE_reduction_by_VARX", sequence_reduction), ("lead24_MSE_reduction_by_VARX", lead24_reduction)):
                    mean, low, high = block_ci(values, args.block_hours, args.bootstrap_replicates, rng)
                    bootstrap_rows.append({"period": period, "seed": seed, "deep_model": name, "metric": metric, "mean": mean, "ci_low": low, "ci_high": high})

    metrics = pd.DataFrame(metric_rows)
    paired = pd.DataFrame(paired_rows)
    leads = pd.DataFrame(lead_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    metrics.to_csv(output / "model_metrics.csv", index=False)
    paired.to_csv(output / "paired_varx_deep_comparison.csv", index=False)
    leads.to_csv(output / "per_lead_metrics.csv", index=False)
    bootstrap.to_csv(output / "block_bootstrap_by_seed.csv", index=False)

    deep_mean = metrics[metrics["model"] != "varx_ridge"].groupby(["period", "model"], as_index=False).mean(numeric_only=True)
    varx_only = metrics[metrics["model"] == "varx_ridge"].copy()
    summary = pd.concat([deep_mean, varx_only], ignore_index=True).sort_values(["period", "model"])
    summary.to_csv(output / "model_summary.csv", index=False)
    lead_summary = leads.groupby(["period", "model", "lead"], as_index=False)["residual_R2"].agg(["mean", "std"]).reset_index()
    lead_summary.to_csv(output / "per_lead_mean_std.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), sharey=False)
    colors = {"varx_ridge": "#111111", "gwn_eta_only": "#2A6FBB", "gwn_multistate_no_physics": "#D98C10", "hs_dt_gwn": "#388E3C"}
    labels = {"varx_ridge": "VARX-Ridge", "gwn_eta_only": "Eta-only GWN", "gwn_multistate_no_physics": "Multistate GWN", "hs_dt_gwn": "HS-DT-GWN"}
    for axis, period in zip(axes, ("2026_h1", "2026_july")):
        for name in ("varx_ridge", *DEEP_MODELS):
            part = lead_summary[(lead_summary["period"] == period) & (lead_summary["model"] == name)]
            axis.plot(part["lead"], part["mean"], color=colors[name], linewidth=2.4 if name == "varx_ridge" else 1.9, label=labels[name])
        axis.set_title(period.replace("_", " ").upper())
        axis.set_xlabel("Forecast lead (h)")
        axis.set_xlim(1, 24)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Residual R2")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Diagnostic-only frozen VARX comparison on already-viewed 2026 periods", weight="bold")
    fig.tight_layout()
    fig.savefig(output / "frozen_varx_2026_per_lead.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    decisions = {}
    for period in ("2026_h1", "2026_july"):
        decisions[period] = {}
        for name in DEEP_MODELS:
            part = paired[(paired["period"] == period) & (paired["deep_model"] == name)]
            lead_part = lead_summary[(lead_summary["period"] == period) & (lead_summary["model"] == name)].set_index("lead")
            varx_lead = lead_summary[(lead_summary["period"] == period) & (lead_summary["model"] == "varx_ridge")].set_index("lead")
            boot = bootstrap[(bootstrap["period"] == period) & (bootstrap["deep_model"] == name)]
            decisions[period][name] = {
                "mean_sequence_R2_advantage_VARX": float(part["varx_minus_deep_sequence_R2"].mean()),
                "sequence_seed_wins_VARX": int((part["varx_minus_deep_sequence_R2"] > 0).sum()),
                "mean_lead24_R2_advantage_VARX": float(part["varx_minus_deep_lead24_R2"].mean()),
                "lead24_seed_wins_VARX": int((part["varx_minus_deep_lead24_R2"] > 0).sum()),
                "leads_VARX_above_deep_mean": int((varx_lead["mean"] > lead_part["mean"]).sum()),
                "sequence_bootstrap_CI_above_zero_seeds": int(((boot["metric"] == "sequence_MSE_reduction_by_VARX") & (boot["ci_low"] > 0)).sum()),
            }
    output_json = {
        "status": "diagnostic_only_already_viewed_not_independent_confirmation",
        "frozen_alpha": alpha,
        "training_design_identical_across_data_extensions": True,
        "period_comparisons": decisions,
    }
    (output / "FROZEN_VARX_2026_DECISION.json").write_text(json.dumps(output_json, indent=2), encoding="utf-8")
    (output / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps(output_json, indent=2))


if __name__ == "__main__":
    main()
