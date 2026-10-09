from __future__ import annotations

"""Matched differentiable physical-loss control for the multistate HS-DT expert."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


m140 = load_module("hsdt_multistate_physics_142", HERE / "140_hsdt_expert_physics_conditioned.py")


def run_seed(seed, args, device):
    data = m140.rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    eta_base = m140.load_base(data, args, seed, 1, device)
    multi_base = m140.load_base(data, args, seed, 4, device)
    enhanced, expert_row = m140.train_expert(
        "multistate", copy.deepcopy(multi_base), data, args, seed, device, args.stage
    )
    eta_pred, true, tide = m140.score_predictions(eta_base, data, args, seed, device, args.stage, 1)
    multi_pred = m140.score_predictions(multi_base, data, args, seed, device, args.stage, 4)[0]
    enhanced_pred = m140.score_predictions(enhanced, data, args, seed, device, args.stage, 4)[0]
    weights = np.full(args.horizon, 0.5, dtype=np.float64)
    weights[-1] = 1.0
    baseline = eta_pred + weights[None, None, :] * (multi_pred - eta_pred)
    candidate = eta_pred + weights[None, None, :] * (enhanced_pred - eta_pred)
    if np.max(np.abs(candidate[..., -1] - enhanced_pred[..., -1])) > 1e-6:
        raise RuntimeError("Candidate lead-24 does not equal enhanced multistate expert")
    train_end = m140.rolling.time_index(data["arrays"]["time"], args.fold_train_end)
    threshold = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    label = "2025_h1_validation_screen" if args.stage == "screen" else "2025_h2_backtest"
    common = {
        "seed": seed,
        "evaluation_split": label,
        "strict_causal_preprocessing": True,
        "future_residual_used_as_input": False,
    }
    rows = [
        {**common, "config": "hsdt_baseline", **m140.confirm.score(true, baseline, tide, threshold)},
        {
            **common,
            "config": "hsdt_multistate_prior_plus_physics_loss",
            "physical_loss_gradient_active": True,
            "physics_lambda": args.physics_lambda,
            **m140.confirm.score(true, candidate, tide, threshold),
        },
    ]
    output = ROOT / args.output_dir / args.stage
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"seed_{seed}_predictions.npz",
        true_residual=true.astype(np.float32),
        tide=tide.astype(np.float32),
        hsdt_baseline=baseline.astype(np.float32),
        hsdt_multistate_prior_plus_physics_loss=candidate.astype(np.float32),
    )
    pd.DataFrame(rows).to_csv(output / f"seed_{seed}_metrics.csv", index=False)
    return rows


def main():
    args = m140.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, args, device))
    output = ROOT / args.output_dir / args.stage
    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(output / "all_runs.csv", index=False)
    metrics = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2", "event_PR_AUC"]
    all_runs.groupby("config")[metrics].agg(["mean", "std", "count"]).to_csv(output / "mean_std.csv")
    pivot = all_runs.pivot(index="seed", columns="config", values=metrics)
    paired = []
    for metric in metrics:
        delta = pivot[(metric, "hsdt_multistate_prior_plus_physics_loss")] - pivot[(metric, "hsdt_baseline")]
        improvement = -delta if metric.endswith(("RMSE", "MAE")) else delta
        paired.append({"metric": metric, "mean_improvement": float(improvement.mean()),
                       "std_improvement": float(improvement.std(ddof=1)), "wins": int((improvement > 0).sum()),
                       "count": int(len(improvement)), "wilcoxon_greater_p": m140.p104.exact_wilcoxon_greater(improvement.to_numpy())})
    pd.DataFrame(paired).to_csv(output / "paired_tests.csv", index=False)


if __name__ == "__main__":
    main()
