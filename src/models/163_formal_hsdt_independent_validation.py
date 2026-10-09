from __future__ import annotations

"""Formal diagnostics for fixed HS-DT-GWN on H1 and July 2026."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


later = load_module("formal_hsdt_later", HERE / "161_evaluate_frozen_safety_gated_dual_head_gwn.py")
diagnostics = later.diagnostics
calibration = later.calibration
pc = later.pc


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    return float(1.0 - np.sum((true - pred) ** 2) / max(denominator, 1e-12))


def score_continuous(true: np.ndarray, pred: np.ndarray, tide: np.ndarray) -> dict[str, float]:
    return pc.confirm.final4.summarize_single(true, pred, tide)


def load_h1(seed: int) -> dict[str, np.ndarray]:
    path = ROOT / "results" / "frozen_2026_h1_physics_reliability_gwn" / f"seed_{seed}_test_frozen_features.npz"
    with np.load(path) as payload:
        values = {key: payload[key] for key in payload.files}
    return {
        "eta": values["eta"],
        "multi": values["multi"],
        "true": values["target"],
        "tide": values["tide"],
        "time": values["forecast_start"],
    }


def collect_july(seed: int, args, device: torch.device) -> dict[str, np.ndarray]:
    _, _, holdout, times = later.collect_seed(seed, args, device)
    return {
        "eta": holdout["eta"],
        "multi": holdout["multi"],
        "true": holdout["target"],
        "tide": holdout["tide"],
        "time": times.to_numpy(dtype="datetime64[ns]"),
    }


def fixed_hsdt(payload: dict[str, np.ndarray]) -> np.ndarray:
    split = {"eta": payload["eta"], "multi": payload["multi"]}
    return calibration.fixed_hsdt(split)


def main() -> None:
    out = ROOT / "results" / "formal_hsdt_independent_validation_2026"
    out.mkdir(parents=True, exist_ok=True)
    args = later.parse_args()
    args.output_dir = str(out)
    pc.confirm.configure_data_dir(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payloads: dict[str, dict[int, dict[str, np.ndarray]]] = {"2026_h1": {}, "2026_july": {}}
    rows: list[dict] = []
    for seed in SEEDS:
        print(f"seed={seed} device={device}", flush=True)
        payloads["2026_h1"][seed] = load_h1(seed)
        payloads["2026_july"][seed] = collect_july(seed, args, device)
        for period in payloads:
            payload = payloads[period][seed]
            hsdt = fixed_hsdt(payload)
            for model, prediction in (
                ("gwn_eta_only", payload["eta"]),
                ("gwn_multistate_no_physics", payload["multi"]),
                ("hs_dt_gwn", hsdt),
            ):
                rows.append({"period": period, "seed": seed, "model": model, **score_continuous(payload["true"], prediction, payload["tide"])})
            if period == "2026_july":
                np.savez_compressed(
                    out / f"seed_{seed}_july_base_predictions.npz",
                    pred_eta=payload["eta"], pred_multi=payload["multi"], pred_hsdt=hsdt,
                    true_residual=payload["true"], target_tide=payload["tide"], forecast_start=payload["time"],
                )

    runs = pd.DataFrame(rows)
    runs.to_csv(out / "all_runs.csv", index=False)
    metrics = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2"]
    mean_std = runs.groupby(["period", "model"])[metrics].agg(["mean", "std", "count"]).reset_index()
    mean_std.columns = ["_".join(str(value) for value in col if value) for col in mean_std.columns.to_flat_index()]
    mean_std.to_csv(out / "mean_std.csv", index=False)

    comparison_rows = []
    for period in payloads:
        period_runs = runs[runs["period"] == period]
        wide = period_runs.pivot(index="seed", columns="model", values=metrics)
        for baseline in ("gwn_multistate_no_physics", "gwn_eta_only"):
            for metric in metrics:
                delta = wide[(metric, "hs_dt_gwn")] - wide[(metric, baseline)]
                improvement = -delta if metric.endswith(("RMSE", "MAE")) else delta
                try:
                    pvalue = float(wilcoxon(improvement, alternative="greater", zero_method="wilcox").pvalue)
                except ValueError:
                    pvalue = 1.0
                comparison_rows.append(
                    {
                        "period": period, "candidate": "hs_dt_gwn", "baseline": baseline, "metric": metric,
                        "mean_improvement": float(improvement.mean()), "std_improvement": float(improvement.std(ddof=1)),
                        "wins": int((improvement > 0).sum()), "count": int(len(improvement)),
                        "wilcoxon_one_sided_p": pvalue,
                    }
                )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)

    rng = np.random.default_rng(20260810)
    bootstrap_rows = []
    for period, period_payloads in payloads.items():
        for baseline_name in ("multi", "eta"):
            for mode in ("sequence", "lead24", "q95"):
                seed_stats = []
                for seed in SEEDS:
                    payload = period_payloads[seed]
                    seed_stats.append(
                        diagnostics.time_stats(
                            payload["true"], fixed_hsdt(payload), payload[baseline_name], mode
                        )
                    )
                values = diagnostics.hierarchical_bootstrap(seed_stats, 168, 2000, rng)
                bootstrap_rows.append(
                    {
                        "period": period, "baseline": baseline_name, "metric": mode,
                        "mean_delta_r2": float(np.mean(values)),
                        "ci95_low": float(np.quantile(values, 0.025)),
                        "ci95_high": float(np.quantile(values, 0.975)),
                        "probability_positive": float(np.mean(values > 0.0)), "replicates": 2000,
                    }
                )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(out / "block_bootstrap_summary.csv", index=False)

    primary = comparisons[
        (comparisons["baseline"] == "gwn_multistate_no_physics")
        & (comparisons["metric"] == "seq_residual_R2")
    ].set_index("period")
    primary_boot = bootstrap[(bootstrap["baseline"] == "multi") & (bootstrap["metric"] == "sequence")].set_index("period")
    decision = {
        "primary_endpoint": "sequence residual R2 relative to multistate GWN",
        "h1_positive_4_of_5": bool(primary.loc["2026_h1", "mean_improvement"] > 0 and primary.loc["2026_h1", "wins"] >= 4),
        "july_positive_4_of_5": bool(primary.loc["2026_july", "mean_improvement"] > 0 and primary.loc["2026_july", "wins"] >= 4),
        "h1_bootstrap_ci_above_zero": bool(primary_boot.loc["2026_h1", "ci95_low"] > 0),
        "july_bootstrap_ci_above_zero": bool(primary_boot.loc["2026_july", "ci95_low"] > 0),
        "main_sequence_model_supported": False,
        "extreme_q95_requires_separate_caveat": True,
    }
    decision["main_sequence_model_supported"] = all(
        decision[key]
        for key in ("h1_positive_4_of_5", "july_positive_4_of_5", "h1_bootstrap_ci_above_zero", "july_bootstrap_ci_above_zero")
    )
    (out / "main_model_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(mean_std.to_string(index=False))
    print("\nPaired comparisons:")
    print(comparisons.to_string(index=False))
    print("\nBootstrap:")
    print(bootstrap.to_string(index=False))
    print("\nDecision:")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
