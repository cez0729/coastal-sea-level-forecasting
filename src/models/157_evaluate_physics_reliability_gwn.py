from __future__ import annotations

"""Statistical diagnostics for validation-locked physics-reliability GWN."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 2024, 2025, 3407]
METRICS = {
    "sequence": lambda values: values,
    "lead24": lambda values: values[..., -1:],
}


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    return float(1.0 - np.sum((true - pred) ** 2) / max(denominator, 1e-12))


def load_seed(root: Path, seed: int) -> dict[str, np.ndarray]:
    with np.load(root / f"seed_{seed}_test_frozen_features.npz") as payload:
        test = {key: payload[key] for key in payload.files}
    with np.load(root / f"seed_{seed}_physics_reliability_hsdt_gwn_predictions.npz") as payload:
        candidate = payload["pred_residual"]
    with np.load(root / f"seed_{seed}_validation_locked_ridge_no_physics_predictions.npz") as payload:
        control = payload["pred_residual"]
    weight = np.full(test["eta"].shape[-1], 0.5, dtype=np.float32)
    weight[-1] = 1.0
    hsdt = test["eta"] + weight[None, None, :] * (test["multi"] - test["eta"])
    return {"true": test["target"], "candidate": candidate, "no_physics": control, "hsdt": hsdt}


def time_stats(true: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    if mode == "lead24":
        true, candidate, baseline = true[..., -1:], candidate[..., -1:], baseline[..., -1:]
        mask = np.ones_like(true, dtype=bool)
    elif mode == "q95":
        threshold = float(np.quantile(np.abs(true), 0.95))
        mask = np.abs(true) >= threshold
    else:
        mask = np.ones_like(true, dtype=bool)
    axes = tuple(range(1, true.ndim))
    masked_true = np.where(mask, true, 0.0)
    return {
        "count": np.sum(mask, axis=axes).astype(np.float64),
        "sum_y": np.sum(masked_true, axis=axes, dtype=np.float64),
        "sum_y2": np.sum(masked_true**2, axis=axes, dtype=np.float64),
        "sse_candidate": np.sum(np.where(mask, (true - candidate) ** 2, 0.0), axis=axes, dtype=np.float64),
        "sse_baseline": np.sum(np.where(mask, (true - baseline) ** 2, 0.0), axis=axes, dtype=np.float64),
    }


def sample_block_indices(rng: np.random.Generator, length: int, block: int) -> np.ndarray:
    if length <= block:
        return np.arange(length)
    starts = rng.integers(0, length - block + 1, size=int(np.ceil(length / block)))
    return np.concatenate([np.arange(start, start + block) for start in starts])[:length]


def hierarchical_bootstrap(
    seed_stats: list[dict[str, np.ndarray]], block: int, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    improvements = np.empty(replicates, dtype=np.float64)
    seed_count = len(seed_stats)
    for rep in range(replicates):
        totals = {key: 0.0 for key in seed_stats[0]}
        for seed_idx in rng.integers(0, seed_count, size=seed_count):
            stats = seed_stats[int(seed_idx)]
            indices = sample_block_indices(rng, len(stats["count"]), block)
            for key, values in stats.items():
                totals[key] += float(np.sum(values[indices]))
        denominator = totals["sum_y2"] - totals["sum_y"] ** 2 / max(totals["count"], 1.0)
        improvements[rep] = (totals["sse_baseline"] - totals["sse_candidate"]) / max(denominator, 1e-12)
    return improvements


def paired_rows(all_runs: pd.DataFrame, candidate: str, baseline: str) -> list[dict]:
    rows = []
    wide = all_runs.pivot(index="seed", columns="model")
    for metric in ("seq_residual_R2", "last_residual_R2", "extreme_abs_q95_residual_R2", "event_PR_AUC", "event_CSI"):
        delta = wide[(metric, candidate)] - wide[(metric, baseline)]
        try:
            pvalue = float(wilcoxon(delta, alternative="greater", zero_method="wilcox").pvalue)
        except ValueError:
            pvalue = 1.0
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "metric": metric,
                "mean_delta": float(delta.mean()),
                "std_delta": float(delta.std(ddof=1)),
                "wins": int((delta > 0).sum()),
                "exact_wilcoxon_one_sided_p": pvalue,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the physics-reliability HS-DT-GWN candidate")
    parser.add_argument("--input-dir", default="results/validation_locked_physics_reliability_gwn_tail_protected")
    parser.add_argument("--output-dir", default="results/validation_locked_physics_reliability_gwn_tail_protected/diagnostics")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--block-hours", type=int, default=168)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260809)
    args = parser.parse_args()
    source = ROOT / args.input_dir
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    payloads = {seed: load_seed(source, seed) for seed in args.seeds}

    per_lead, per_station = [], []
    for seed, payload in payloads.items():
        true, candidate = payload["true"], payload["candidate"]
        for baseline_name in ("hsdt", "no_physics"):
            baseline = payload[baseline_name]
            for lead in range(true.shape[-1]):
                per_lead.append(
                    {
                        "seed": seed,
                        "baseline": baseline_name,
                        "lead_hour": lead + 1,
                        "candidate_r2": r2(true[..., lead], candidate[..., lead]),
                        "baseline_r2": r2(true[..., lead], baseline[..., lead]),
                        "delta_r2": r2(true[..., lead], candidate[..., lead]) - r2(true[..., lead], baseline[..., lead]),
                    }
                )
            for station in range(true.shape[1]):
                per_station.append(
                    {
                        "seed": seed,
                        "baseline": baseline_name,
                        "station_index": station,
                        "candidate_r2": r2(true[:, station], candidate[:, station]),
                        "baseline_r2": r2(true[:, station], baseline[:, station]),
                        "delta_r2": r2(true[:, station], candidate[:, station]) - r2(true[:, station], baseline[:, station]),
                    }
                )
    pd.DataFrame(per_lead).to_csv(out / "per_lead_delta.csv", index=False)
    pd.DataFrame(per_station).to_csv(out / "per_station_delta.csv", index=False)

    rng = np.random.default_rng(args.random_seed)
    bootstrap_rows = []
    for baseline_name in ("hsdt", "no_physics"):
        for mode in ("sequence", "lead24", "q95"):
            stats = [
                time_stats(payloads[seed]["true"], payloads[seed]["candidate"], payloads[seed][baseline_name], mode)
                for seed in args.seeds
            ]
            values = hierarchical_bootstrap(stats, args.block_hours, args.bootstrap_replicates, rng)
            bootstrap_rows.append(
                {
                    "baseline": baseline_name,
                    "metric": mode,
                    "mean_delta_r2": float(np.mean(values)),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "probability_positive": float(np.mean(values > 0)),
                    "replicates": args.bootstrap_replicates,
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(out / "block_bootstrap_summary.csv", index=False)

    all_runs = pd.read_csv(source / "all_runs.csv")
    paired = []
    for baseline in ("hs_dt_gwn", "validation_locked_ridge_no_physics"):
        paired.extend(paired_rows(all_runs, "physics_reliability_hsdt_gwn", baseline))
    pd.DataFrame(paired).to_csv(out / "paired_exact_tests.csv", index=False)

    lead_summary = pd.DataFrame(per_lead).groupby(["baseline", "lead_hour"])["delta_r2"].agg(["mean", "std", "min", "max"])
    station_summary = pd.DataFrame(per_station).groupby(["baseline", "station_index"])["delta_r2"].agg(["mean", "std", "min", "max"])
    report = [
        "# Physics-reliability HS-DT-GWN diagnostics",
        "",
        "The model was selected on a validation split; 2025 H2 was not used by this script for tuning. However, the tail-protection rule was proposed after inspecting an earlier unprotected 2025 H2 screen, so the result remains post-hoc exploratory rather than untouched confirmation.",
        "",
        "## 168-hour hierarchical block bootstrap",
        "",
        bootstrap.to_string(index=False),
        "",
        "## Per-lead delta R2",
        "",
        lead_summary.to_string(),
        "",
        "## Per-station delta R2",
        "",
        station_summary.to_string(),
        "",
        "## Evidence boundary",
        "",
        "A new chronological period or external-station confirmation is still required before treating this candidate as the confirmed submission model. Event CSI is a known weakness and must be reported rather than omitted.",
    ]
    (out / "DIAGNOSTIC_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (out / "diagnostic_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(bootstrap.to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))


if __name__ == "__main__":
    main()
