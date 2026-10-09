"""Reproduce reviewer-requested HS-DT weight and bootstrap sensitivity audits.

This script only reads frozen prediction artifacts. It does not train models,
select a new weight, or change the primary HS-DT rule.
"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_sensitivity_20260810"
SEEDS = [42, 123, 2024, 2025, 3407]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
BLOCKS = [24, 72, 168, 336]


def r2(y_true: np.ndarray, y_pred: np.ndarray, idx=None) -> float:
    if idx is None:
        yt, yp = y_true.reshape(-1), y_pred.reshape(-1)
    else:
        yt, yp = y_true[idx].reshape(-1), y_pred[idx].reshape(-1)
    den = np.sum((yt - np.mean(yt)) ** 2)
    return float(1.0 - np.sum((yt - yp) ** 2) / den) if den > 0 else float("nan")


def load_h2(seed: int):
    base = ROOT / "results" / "priority12_physics_graph_wavenet" / f"seed_{seed}" / "horizon_24h"
    eta = np.load(base / "gwn_eta_only" / "predictions.npz", allow_pickle=True)
    multi = np.load(base / "gwn_multistate_no_physics" / "predictions.npz", allow_pickle=True)
    return eta["pred_residual"], multi["pred_states"][..., 0], eta["true_residual"]


def load_july(seed: int):
    p = ROOT / "results" / "formal_hsdt_independent_validation_2026" / f"seed_{seed}_july_base_predictions.npz"
    z = np.load(p, allow_pickle=True)
    return z["pred_eta"], z["pred_multi"], z["true_residual"]


def load_h1(seed: int):
    p = ROOT / "results" / "frozen_2026_h1_physics_reliability_gwn" / f"seed_{seed}_test_frozen_features.npz"
    z = np.load(p, allow_pickle=True)
    return z["eta"], z["multi"], z["target"]


def fused(eta, multi, alpha):
    out = eta.copy()
    out[..., :23] = (1.0 - alpha) * eta[..., :23] + alpha * multi[..., :23]
    out[..., 23] = multi[..., 23]
    return out


def bootstrap_delta(y, eta, multi, alpha, block, reps=500, seed=17):
    rng = np.random.default_rng(seed + block)
    n = y.shape[0]
    starts = np.arange(0, n, block)
    blocks = [np.arange(s, min(s + block, n)) for s in starts]
    hs = fused(eta, multi, alpha)
    # Aggregate over station and horizon once; bootstrap then only sums origins.
    y_flat = y.reshape(n, -1)
    y_sum = np.sum(y_flat, axis=1)
    y_sq = np.sum(y_flat * y_flat, axis=1)
    se_hs = np.sum((y_flat - hs.reshape(n, -1)) ** 2, axis=1)
    se_multi = np.sum((y_flat - multi.reshape(n, -1)) ** 2, axis=1)
    deltas = np.empty(reps)
    for b in range(reps):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[j] for j in chosen])
        den = np.sum(y_sq[idx]) - np.sum(y_sum[idx]) ** 2 / len(idx) / y.shape[1] / y.shape[2]
        deltas[b] = (np.sum(se_multi[idx]) - np.sum(se_hs[idx])) / den
    return float(np.mean(deltas)), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def origin_stats(y, eta, multi, alpha):
    n = y.shape[0]
    hs = fused(eta, multi, alpha)
    y_flat = y.reshape(n, -1)
    return {
        "n_scalar": y.shape[1] * y.shape[2],
        "y_sum": np.sum(y_flat, axis=1),
        "y_sq": np.sum(y_flat * y_flat, axis=1),
        "se_hs": np.sum((y_flat - hs.reshape(n, -1)) ** 2, axis=1),
        "se_multi": np.sum((y_flat - multi.reshape(n, -1)) ** 2, axis=1),
    }


def hierarchical_bootstrap(loaders, block, reps=1000, seed=29):
    rng = np.random.default_rng(seed + block)
    stats = [origin_stats(loader[2], loader[0], loader[1], 0.5) for loader in loaders]
    deltas = np.empty(reps)
    for b in range(reps):
        total_count = 0
        total_y_sum = total_y_sq = total_se_hs = total_se_multi = 0.0
        for seed_idx in rng.integers(0, len(stats), size=len(stats)):
            s = stats[seed_idx]
            n = len(s["y_sum"])
            blocks = [np.arange(start, min(start + block, n)) for start in range(0, n, block)]
            chosen = rng.integers(0, len(blocks), size=len(blocks))
            idx = np.concatenate([blocks[j] for j in chosen])
            total_count += len(idx) * s["n_scalar"]
            total_y_sum += np.sum(s["y_sum"][idx])
            total_y_sq += np.sum(s["y_sq"][idx])
            total_se_hs += np.sum(s["se_hs"][idx])
            total_se_multi += np.sum(s["se_multi"][idx])
        den = total_y_sq - total_y_sum ** 2 / total_count
        deltas[b] = (total_se_multi - total_se_hs) / den
    return float(np.mean(deltas)), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    weight_rows = []
    periods = [
        ("2025_H2_retrospective", load_h2),
        ("2026_H1_frozen", load_h1),
        ("2026_July_frozen", load_july),
    ]
    for period, loader in periods:
        for seed in SEEDS:
            eta, multi, y = loader(seed)
            for alpha in WEIGHTS:
                weight_rows.append({
                    "period": period,
                    "seed": seed,
                    "alpha_multistate_leads_1_23": alpha,
                    "lead24_alpha": 1.0,
                    "sequence_r2": r2(y, fused(eta, multi, alpha)),
                    "lead24_r2": r2(y[..., 23], fused(eta, multi, alpha)[..., 23]),
                })
    weight = pd.DataFrame(weight_rows)
    weight.to_csv(OUT / "hsdt_weight_sensitivity.csv", index=False)
    summary = (
        weight.groupby(["period", "alpha_multistate_leads_1_23"], as_index=False)
        .agg(sequence_r2_mean=("sequence_r2", "mean"), sequence_r2_sd=("sequence_r2", "std"),
             lead24_r2_mean=("lead24_r2", "mean"), lead24_r2_sd=("lead24_r2", "std"))
    )
    summary.to_csv(OUT / "hsdt_weight_sensitivity_summary.csv", index=False)

    block_rows = []
    for period, loader in periods:
        for seed in SEEDS:
            eta, multi, y = loader(seed)
            for block in BLOCKS:
                mean, lo, hi = bootstrap_delta(y, eta, multi, 0.5, block, seed=seed)
                block_rows.append({"period": period, "seed": seed, "block_hours": block,
                                   "delta_sequence_r2_mean": mean, "delta_ci_low": lo, "delta_ci_high": hi})
    block_df = pd.DataFrame(block_rows)
    block_df.to_csv(OUT / "bootstrap_block_sensitivity.csv", index=False)
    block_summary = (
        block_df.groupby(["period", "block_hours"], as_index=False)
        .agg(delta_mean=("delta_sequence_r2_mean", "mean"),
             seed_sd=("delta_sequence_r2_mean", "std"),
             ci_low_min=("delta_ci_low", "min"), ci_high_max=("delta_ci_high", "max"))
    )
    block_summary.to_csv(OUT / "bootstrap_block_sensitivity_summary.csv", index=False)
    hierarchical_rows = []
    for period, loader in periods:
        loaded = [loader(seed) for seed in SEEDS]
        for block in BLOCKS:
            mean, lo, hi = hierarchical_bootstrap(loaded, block)
            hierarchical_rows.append({"period": period, "block_hours": block,
                                      "delta_sequence_r2_mean": mean,
                                      "ci_low": lo, "ci_high": hi,
                                      "resamples": 1000})
    pd.DataFrame(hierarchical_rows).to_csv(
        OUT / "bootstrap_block_sensitivity_hierarchical.csv", index=False
    )
    (OUT / "README.md").write_text(
        "Reviewer sensitivity audit. All calculations read frozen predictions only. "
        "Alpha is the multistate weight at leads 1-23; lead 24 is always multistate. "
        "Candidate alphas and bootstrap block lengths were specified before scoring; "
        "no candidate was selected for the primary paper claim.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
