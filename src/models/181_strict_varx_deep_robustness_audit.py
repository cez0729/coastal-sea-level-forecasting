"""Matched strict-period comparison of VARX against GWN and HS-DT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "results" / "strict_varx_deep_robustness_audit_2025_h2"
SEEDS = [42, 123, 2024, 2025, 3407]
MODELS = ["gwn_eta_only", "gwn_multistate_no_physics", "hs_dt_gwn"]


def r2(true, pred):
    denominator = float(np.sum((true - true.mean()) ** 2))
    return float(1.0 - np.sum((true - pred) ** 2) / max(denominator, 1e-12))


def bootstrap_mean_ci(values, block, replicates, rng):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    starts = np.arange(max(1, n - block + 1))
    estimates = np.empty(replicates, dtype=np.float64)
    blocks_needed = int(np.ceil(n / block))
    offsets = np.arange(block)
    for i in range(replicates):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        indices = (chosen[:, None] + offsets[None, :]).reshape(-1)[:n]
        estimates[i] = values[indices].mean()
    return float(values.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--block-hours", type=int, default=168)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    varx_file = np.load(ROOT / "results" / "strict_varx_chronological_validation_2025_h2" / "strict_varx_predictions.npz")
    varx = varx_file["pred_residual"].astype(np.float64)
    true = varx_file["true_residual"].astype(np.float64)
    times = varx_file["target_origin_time"]
    tail_threshold = float(np.quantile(np.abs(true), 0.95))
    tail_mask = np.abs(true) >= tail_threshold
    paired_rows, lead_rows, station_rows, bootstrap_rows = [], [], [], []
    rng = np.random.default_rng(20260811)
    for seed in args.seeds:
        saved = np.load(ROOT / "results" / "confirmatory_hsdt_orc_refit_2025_h2" / f"seed_{seed}" / "predictions.npz")
        if not np.allclose(true, saved["true_residual"], atol=1e-7, rtol=0):
            raise RuntimeError(f"Target mismatch for seed {seed}")
        if not np.array_equal(times, saved["target_origin_time"]):
            raise RuntimeError(f"Timestamp mismatch for seed {seed}")
        for model in MODELS:
            pred = saved[model].astype(np.float64)
            paired_rows.append({
                "seed": seed, "model": model,
                "varx_sequence_R2": r2(true, varx), "deep_sequence_R2": r2(true, pred),
                "varx_minus_deep_sequence_R2": r2(true, varx) - r2(true, pred),
                "varx_lead24_R2": r2(true[..., -1], varx[..., -1]), "deep_lead24_R2": r2(true[..., -1], pred[..., -1]),
                "varx_minus_deep_lead24_R2": r2(true[..., -1], varx[..., -1]) - r2(true[..., -1], pred[..., -1]),
                "varx_q95_R2": r2(true[tail_mask], varx[tail_mask]), "deep_q95_R2": r2(true[tail_mask], pred[tail_mask]),
                "varx_minus_deep_q95_R2": r2(true[tail_mask], varx[tail_mask]) - r2(true[tail_mask], pred[tail_mask]),
            })
            for lead in range(true.shape[-1]):
                lead_rows.append({
                    "seed": seed, "model": model, "lead": lead + 1,
                    "varx_R2": r2(true[..., lead], varx[..., lead]),
                    "deep_R2": r2(true[..., lead], pred[..., lead]),
                    "varx_minus_deep_R2": r2(true[..., lead], varx[..., lead]) - r2(true[..., lead], pred[..., lead]),
                    "varx_MSE": float(np.mean((varx[..., lead] - true[..., lead]) ** 2)),
                    "deep_MSE": float(np.mean((pred[..., lead] - true[..., lead]) ** 2)),
                })
            for station, station_id in enumerate(saved["station_ids"]):
                station_rows.append({
                    "seed": seed, "model": model, "station_id": str(station_id),
                    "varx_sequence_R2": r2(true[:, station], varx[:, station]),
                    "deep_sequence_R2": r2(true[:, station], pred[:, station]),
                    "varx_minus_deep_sequence_R2": r2(true[:, station], varx[:, station]) - r2(true[:, station], pred[:, station]),
                    "varx_minus_deep_lead24_R2": r2(true[:, station, -1], varx[:, station, -1]) - r2(true[:, station, -1], pred[:, station, -1]),
                })
            seq_reduction = np.mean((pred - true) ** 2 - (varx - true) ** 2, axis=(1, 2))
            terminal_reduction = np.mean((pred[..., -1] - true[..., -1]) ** 2 - (varx[..., -1] - true[..., -1]) ** 2, axis=1)
            for metric, values in (("sequence_MSE_reduction_by_VARX", seq_reduction), ("lead24_MSE_reduction_by_VARX", terminal_reduction)):
                mean, low, high = bootstrap_mean_ci(values, args.block_hours, args.bootstrap_replicates, rng)
                bootstrap_rows.append({"seed": seed, "model": model, "metric": metric, "mean": mean, "ci_low": low, "ci_high": high})

    paired = pd.DataFrame(paired_rows); leads = pd.DataFrame(lead_rows); stations = pd.DataFrame(station_rows); bootstrap = pd.DataFrame(bootstrap_rows)
    paired.to_csv(out / "paired_seed_comparison.csv", index=False)
    leads.to_csv(out / "per_lead_by_seed.csv", index=False)
    stations.to_csv(out / "per_station_by_seed.csv", index=False)
    bootstrap.to_csv(out / "block_bootstrap_by_seed.csv", index=False)
    summary = paired.groupby("model").agg(["mean", "std"]); summary.columns = ["_".join(c) for c in summary.columns]; summary.reset_index().to_csv(out / "paired_summary.csv", index=False)
    lead_summary = leads.groupby(["model", "lead"])[["varx_R2", "deep_R2", "varx_minus_deep_R2"]].agg(["mean", "std"]).reset_index()
    lead_summary.columns = ["_".join(c for c in col if c) for col in lead_summary.columns.to_flat_index()]
    lead_summary.to_csv(out / "per_lead_mean_std.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    varx_curve = lead_summary.groupby("lead")["varx_R2_mean"].mean()
    ax.plot(varx_curve.index, varx_curve.values, color="black", linewidth=2.4, label="VARX-Ridge")
    colors = {"gwn_eta_only": "#2A6FBB", "gwn_multistate_no_physics": "#D98C10", "hs_dt_gwn": "#388E3C"}
    for model in MODELS:
        sub = lead_summary[lead_summary["model"] == model]
        ax.plot(sub["lead"], sub["deep_R2_mean"], color=colors[model], linewidth=2, label=model)
    ax.set_xlabel("Forecast lead (h)"); ax.set_ylabel("Residual R2"); ax.set_xlim(1, 24); ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Strict 2025 H2: VARX versus graph-temporal models")
    fig.tight_layout(); fig.savefig(out / "strict_varx_deep_per_lead.png", dpi=260); plt.close(fig)

    decisions = {}
    for model in MODELS:
        sub = paired[paired["model"] == model]
        boot = bootstrap[bootstrap["model"] == model]
        decisions[model] = {
            "sequence_VARX_wins": int((sub["varx_minus_deep_sequence_R2"] > 0).sum()),
            "lead24_VARX_wins": int((sub["varx_minus_deep_lead24_R2"] > 0).sum()),
            "q95_VARX_wins": int((sub["varx_minus_deep_q95_R2"] > 0).sum()),
            "mean_sequence_R2_advantage": float(sub["varx_minus_deep_sequence_R2"].mean()),
            "mean_lead24_R2_advantage": float(sub["varx_minus_deep_lead24_R2"].mean()),
            "mean_q95_R2_advantage": float(sub["varx_minus_deep_q95_R2"].mean()),
            "sequence_bootstrap_CI_above_zero_seeds": int(((boot["metric"] == "sequence_MSE_reduction_by_VARX") & (boot["ci_low"] > 0)).sum()),
            "lead24_bootstrap_CI_above_zero_seeds": int(((boot["metric"] == "lead24_MSE_reduction_by_VARX") & (boot["ci_low"] > 0)).sum()),
        }
    output = {"interpretation": "linear_deep_complementary_robustness_required", "descriptive_q95_threshold_source": "strict_test_abs_q95", "comparisons": decisions}
    (out / "ROBUSTNESS_DECISION.json").write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
