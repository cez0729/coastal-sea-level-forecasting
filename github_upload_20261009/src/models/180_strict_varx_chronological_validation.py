"""Place VARX in the matched 2025 H2 strict chronological protocol."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "results" / "strict_varx_chronological_validation_2025_h2"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p134 = load_module("p134_strict_varx", HERE / "134_confirmatory_hsdt_orc_chronological_refit.py")
p170 = load_module("p170_strict_varx", HERE / "170_physics_reliability_gated_residual_moe.py")


def fit_and_score(data, alpha_grid):
    train_x, train_y, train_tide = p170.design_matrix(data["single_train"])
    val_x, val_y, val_tide = p170.design_matrix(data["single_val"])
    test_x, test_y, test_tide = p170.design_matrix(data["single_test"])
    alpha_rows = []
    best_alpha, best_mse = None, float("inf")
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, solver="lsqr").fit(train_x, train_y)
        mse = float(np.mean((model.predict(val_x) - val_y) ** 2))
        alpha_rows.append({"alpha": alpha, "validation_MSE": mse})
        if mse < best_mse:
            best_alpha, best_mse = float(alpha), mse
    model = Ridge(alpha=best_alpha, solver="lsqr").fit(train_x, train_y)
    outputs = {}
    for split, x, y, tide in (
        ("train", train_x, train_y, train_tide),
        ("validation", val_x, val_y, val_tide),
        ("test", test_x, test_y, test_tide),
    ):
        true = y.reshape(-1, 7, data["single_test"].horizon).astype(np.float32)
        pred = model.predict(x).reshape(true.shape).astype(np.float32)
        outputs[split] = {"true": true, "pred": pred, "tide": tide.astype(np.float32)}
    return best_alpha, pd.DataFrame(alpha_rows), outputs


def metric_rows(protocol, outputs):
    return [
        {"protocol": protocol, "split": split, "model": "VARX-Ridge", **p170.summarize(payload["true"], payload["pred"], payload["tide"])}
        for split, payload in outputs.items()
    ]


def existing_model_means():
    retro = pd.read_csv(ROOT / "results" / "horizon_specialized_dual_task_gwn" / "all_runs.csv")
    strict = pd.read_csv(ROOT / "results" / "confirmatory_hsdt_orc_refit_2025_h2" / "all_runs.csv")
    keep = ["gwn_eta_only", "gwn_multistate_no_physics", "horizon_specialized_dual_task_gwn", "hs_dt_gwn"]
    rows = []
    for protocol, frame in (("retrospective", retro), ("strict_2025_h2", strict)):
        frame = frame[frame["config"].isin(keep)]
        for config, group in frame.groupby("config"):
            canonical = "hs_dt_gwn" if config in ("horizon_specialized_dual_task_gwn", "hs_dt_gwn") else config
            rows.append({
                "protocol": protocol,
                "model": canonical,
                "sequence_R2": float(group["seq_residual_R2"].mean()),
                "lead24_R2": float(group["last_residual_R2"].mean()),
                "q95_R2": float(group["extreme_abs_q95_residual_R2"].mean()),
                "seed_count": int(group["seed"].nunique()),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--data-dir", default="data/processed_multiyear_2023_2025")
    parser.add_argument("--fold-train-end", default="2025-01-01")
    parser.add_argument("--fold-val-end", default="2025-07-01")
    parser.add_argument("--fold-test-end", default="2026-01-01")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--physics-forcing-mode", default="last_input")
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    p134.configure_data_dir(args.data_dir)

    strict_data = p134.rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    strict_alpha, strict_alpha_table, strict_outputs = fit_and_score(strict_data, args.alpha_grid)
    strict_alpha_table.assign(protocol="strict_2025_h2").to_csv(out / "strict_alpha_selection.csv", index=False)

    retrospective_args = argparse.Namespace(
        window=args.window, train_ratio=0.70, val_ratio=0.15, train_stride=args.train_stride,
        physics_forcing_mode="last_input", extreme_quantile=0.90,
    )
    retrospective_data = p170.final4.build_enhanced_data(retrospective_args, args.horizon, add_ode_prior=False)
    retro_alpha, retro_alpha_table, retro_outputs = fit_and_score(retrospective_data, args.alpha_grid)
    retro_alpha_table.assign(protocol="retrospective").to_csv(out / "retrospective_alpha_selection.csv", index=False)

    metrics = pd.DataFrame(metric_rows("strict_2025_h2", strict_outputs) + metric_rows("retrospective", retro_outputs))
    metrics.to_csv(out / "varx_metrics_by_protocol_split.csv", index=False)
    strict_test = metrics[(metrics["protocol"] == "strict_2025_h2") & (metrics["split"] == "test")].iloc[0]
    retro_test = metrics[(metrics["protocol"] == "retrospective") & (metrics["split"] == "test")].iloc[0]

    comparison = existing_model_means()
    comparison = pd.concat([comparison, pd.DataFrame([
        {"protocol": "retrospective", "model": "varx_ridge", "sequence_R2": retro_test["seq_residual_R2"], "lead24_R2": retro_test["last_residual_R2"], "q95_R2": retro_test["extreme_abs_q95_residual_R2"], "seed_count": 1},
        {"protocol": "strict_2025_h2", "model": "varx_ridge", "sequence_R2": strict_test["seq_residual_R2"], "lead24_R2": strict_test["last_residual_R2"], "q95_R2": strict_test["extreme_abs_q95_residual_R2"], "seed_count": 1},
    ])], ignore_index=True)
    wide = comparison.pivot_table(index="model", columns="protocol", values=["sequence_R2", "lead24_R2", "q95_R2"], aggfunc="first")
    degradation_rows = []
    for model in wide.index:
        row = {"model": model}
        for metric in ("sequence_R2", "lead24_R2", "q95_R2"):
            retro = wide.loc[model, (metric, "retrospective")] if (metric, "retrospective") in wide.columns else np.nan
            strict = wide.loc[model, (metric, "strict_2025_h2")] if (metric, "strict_2025_h2") in wide.columns else np.nan
            row[f"{metric}_retrospective"] = retro
            row[f"{metric}_strict"] = strict
            row[f"{metric}_degradation"] = (retro - strict) / max(abs(retro), 1e-8) if np.isfinite(retro) and np.isfinite(strict) else np.nan
        degradation_rows.append(row)
    degradation = pd.DataFrame(degradation_rows)
    comparison.to_csv(out / "matched_protocol_model_means.csv", index=False)
    degradation.to_csv(out / "retrospective_to_strict_degradation.csv", index=False)

    target_indices = np.asarray(strict_data["single_test"].indices, dtype=np.int64)
    target_times = pd.to_datetime(strict_data["arrays"]["time"])[target_indices].to_numpy(dtype="datetime64[ns]")
    np.savez_compressed(
        out / "strict_varx_predictions.npz",
        pred_residual=strict_outputs["test"]["pred"], true_residual=strict_outputs["test"]["true"],
        target_tide=strict_outputs["test"]["tide"], target_origin_time=target_times,
    )
    decision = {
        "strict_alpha": strict_alpha,
        "retrospective_alpha": retro_alpha,
        "strict_sequence_R2": float(strict_test["seq_residual_R2"]),
        "strict_lead24_R2": float(strict_test["last_residual_R2"]),
        "strict_q95_R2": float(strict_test["extreme_abs_q95_residual_R2"]),
        "retrospective_sequence_R2": float(retro_test["seq_residual_R2"]),
        "retrospective_lead24_R2": float(retro_test["last_residual_R2"]),
        "retrospective_q95_R2": float(retro_test["extreme_abs_q95_residual_R2"]),
        "strict_causal_preprocessing": True,
        "future_residual_used_as_input": False,
        "holdout_status": "chronological_refit_backtest_already_viewed_not_untouched",
    }
    (out / "VARX_STRICT_DECISION.json").write_text(json.dumps(decision, ensure_ascii=True, indent=2), encoding="utf-8")
    (out / "experiment_config.json").write_text(json.dumps(vars(args), ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=True, indent=2))
    print(degradation.to_string(index=False))


if __name__ == "__main__":
    main()
