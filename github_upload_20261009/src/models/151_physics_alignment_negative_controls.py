from __future__ import annotations

"""Pre-specified alignment and negative-control audit for the physics gate."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def exact_greater(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    # One-sided sign/permutation test around zero, with ties split evenly.
    nonzero = values[values != 0]
    if nonzero.size == 0:
        return 0.5
    observed = float(nonzero.mean())
    count = 0
    total = 2 ** nonzero.size if nonzero.size <= 20 else 100000
    rng = np.random.default_rng(20260801)
    for _ in range(total):
        signs = rng.choice([-1.0, 1.0], size=nonzero.size)
        if float((signs * nonzero).mean()) >= observed:
            count += 1
    return float((count + 1) / (total + 1))


def score_effect(true: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    reduction = (true - baseline) ** 2 - (true - candidate) ** 2
    selected = reduction[mask]
    baseline_mse = ((true - baseline) ** 2)[mask]
    if selected.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(selected.mean())
    lead = float(reduction[..., -1][mask].mean())
    pct = float(100.0 * mean / max(float(baseline_mse.mean()), 1e-12))
    return mean, lead, pct


def run(args: argparse.Namespace) -> None:
    m149 = load_module("m149_alignment", ROOT / "数据整理" / "149_physics_regime_switched_hsdt.py")
    data_args = SimpleNamespace(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        window=args.window,
        train_stride=args.train_stride,
        physics_forcing_mode="last_input",
        extreme_quantile=args.extreme_quantile,
    )
    data = m149.final4.build_enhanced_data(data_args, args.horizon, add_ode_prior=False)
    forcing = m149.build_forcing_intensity(
        data,
        SimpleNamespace(
            train_ratio=args.train_ratio,
            candidate_quantiles=[args.quantile],
            regime_quantiles=[args.quantile],
        ),
    )
    indices = data["multi_test"].indices
    intensity = forcing["combined"][indices - 1]
    threshold = forcing["component_thresholds"]["combined"][float(args.quantile)]
    aligned = intensity >= threshold[None, :]
    rows = []
    paired = []
    shifts = [1, 6, 12, 24, 72, 168]
    prediction_args = SimpleNamespace(orc_results=args.orc_results, horizon=args.horizon)
    for seed in args.seeds:
        candidate, true, _ = m149.load_prediction(seed, "orc_hsdt_gwn", prediction_args)
        baseline, true_base, _ = m149.load_prediction(seed, "hsdt_gwn", prediction_args)
        if not np.allclose(true, true_base, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"Target mismatch for seed {seed}")
        masks = {"aligned": aligned, "inverse": ~aligned}
        for shift in shifts:
            masks[f"shift_{shift}h"] = np.roll(aligned, shift, axis=0)
        for name, mask in masks.items():
            seq, lead, pct = score_effect(true, candidate, baseline, mask)
            rows.append(
                {
                    "seed": seed,
                    "mask": name,
                    "quantile": args.quantile,
                    "coverage": float(mask.mean()),
                    "sequence_MSE_reduction": seq,
                    "lead24_MSE_reduction": lead,
                    "relative_sequence_MSE_reduction_pct": pct,
                }
            )
        aligned_seq, aligned_lead, _ = score_effect(true, candidate, baseline, aligned)
        shifted_seq, shifted_lead, _ = score_effect(true, candidate, baseline, np.roll(aligned, args.reference_shift_hours, axis=0))
        paired.append(
            {
                "seed": seed,
                "aligned_minus_reference_shift_sequence": aligned_seq - shifted_seq,
                "aligned_minus_reference_shift_lead24": aligned_lead - shifted_lead,
            }
        )

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    pair = pd.DataFrame(paired)
    frame.to_csv(out / "alignment_negative_controls.csv", index=False)
    pair.to_csv(out / "alignment_paired_differences.csv", index=False)
    summary = frame.groupby("mask", as_index=False).agg(
        coverage=("coverage", "mean"),
        sequence_MSE_reduction_mean=("sequence_MSE_reduction", "mean"),
        sequence_MSE_reduction_std=("sequence_MSE_reduction", "std"),
        lead24_MSE_reduction_mean=("lead24_MSE_reduction", "mean"),
        lead24_MSE_reduction_std=("lead24_MSE_reduction", "std"),
        positive_sequence_seeds=("sequence_MSE_reduction", lambda x: int((x > 0).sum())),
        positive_lead24_seeds=("lead24_MSE_reduction", lambda x: int((x > 0).sum())),
    )
    summary.to_csv(out / "alignment_negative_control_summary.csv", index=False)
    p_seq = exact_greater(pair["aligned_minus_reference_shift_sequence"].to_numpy())
    p_lead = exact_greater(pair["aligned_minus_reference_shift_lead24"].to_numpy())
    report = f"""# Physics-gate alignment negative controls

This audit reuses the ordinary 70/15/15 benchmark predictions. The combined
forcing q{args.quantile:.2f} threshold is computed station-wise from the
training period. The ORC correction is evaluated under the aligned mask, an
inverse mask, and fixed circular time shifts; no test result is used to select
the mask.

## Result

| Mask | Sequence MSE reduction | Lead-24 MSE reduction | Positive Lead-24 seeds |
|---|---:|---:|---:|
""" + "\n".join(
        f"| {r['mask']} | {r['sequence_MSE_reduction_mean']:.8f} | {r['lead24_MSE_reduction_mean']:.8f} | {int(r['positive_lead24_seeds'])}/5 |"
        for _, r in summary.iterrows()
    ) + f"""

The pre-specified aligned-minus-{args.reference_shift_hours}h paired sign test
gives p={p_seq:.5f} for sequence MSE reduction and p={p_lead:.5f} for Lead-24.
With only five seeds these are mechanism checks, not population-level
confirmatory tests. The aligned mask should be interpreted as evidence that
the physical state and correction timing are related; it is not evidence that
physics improves every sample.
"""
    (out / "EXPERIMENT_REPORT_CN.md").write_text(report, encoding="utf-8")

    order = ["aligned", "inverse", *[f"shift_{shift}h" for shift in shifts]]
    lookup = summary.set_index("mask")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(order))
    ax.bar(x, [lookup.loc[name, "lead24_MSE_reduction_mean"] for name in order], color=["#2F7D6D", "#9A6A6A", *(["#7F8C8D"] * len(shifts))])
    ax.axhline(0, color="#222", linewidth=0.8)
    ax.set_xticks(x, [name.replace("_", " ") for name in order], rotation=25, ha="right")
    ax.set_ylabel("Lead-24 MSE reduction")
    ax.set_title("Physical alignment versus negative-control masks")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "alignment_negative_controls.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Physics alignment negative controls")
    parser.add_argument("--output-dir", default="results/physics_alignment_negative_controls")
    parser.add_argument("--orc-results", default="results/ode_residual_corrected_hsdt_gwn")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--reference-shift-hours", type=int, default=168)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
