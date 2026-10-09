from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parent / "outputs" / "final_four_models_24h_5seed_matched"
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "block_bootstrap_significance"
SCRIPT76 = Path(__file__).resolve().parent / "76_enhanced_forcing_physics_loss_ablation_v3.py"

spec = importlib.util.spec_from_file_location("v3_impl", SCRIPT76)
v3 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v3)


def load_residual_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "pred_states" in data.files:
        return data["pred_states"][..., 0], data["true_states"][..., 0]
    return data["pred_residual"], data["true_residual"]


def parse_seed(path: Path) -> int | None:
    matches = re.findall(r"seed[_]?(\d+)", str(path))
    return int(matches[-1]) if matches else None


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=blocks)
    pieces = [(start + np.arange(block)) % n for start in starts]
    return np.concatenate(pieces)[:n]


def bootstrap_comparison(
    true: np.ndarray,
    physical: np.ndarray,
    baseline: np.ndarray,
    block_hours: int,
    bootstrap_n: int,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    n = min(len(true), len(physical), len(baseline))
    true = true[-n:]
    physical = physical[-n:]
    baseline = baseline[-n:]
    if mask is not None:
        mask = mask[-n:]

    def loss_difference(indices: np.ndarray) -> float:
        yt = true[indices]
        pp = physical[indices]
        pb = baseline[indices]
        if mask is not None:
            m = mask[indices]
            if not np.any(m):
                return float("nan")
            yt, pp, pb = yt[m], pp[m], pb[m]
        physical_mse = np.mean((yt - pp) ** 2)
        baseline_mse = np.mean((yt - pb) ** 2)
        return float(baseline_mse - physical_mse)

    observed = loss_difference(np.arange(n))
    boot = np.asarray(
        [loss_difference(circular_block_indices(n, block_hours, rng)) for _ in range(bootstrap_n)],
        dtype=float,
    )
    boot = boot[np.isfinite(boot)]
    return {
        "observed_MSE_reduction": observed,
        "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_probability_physical_better": float(np.mean(boot > 0)),
        "bootstrap_valid_replicates": int(len(boot)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired moving-block bootstrap for forecast error differences.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--physical-key", default="physical_loss")
    parser.add_argument("--baseline-keys", nargs="+", default=["gnn_bigru", "learnable_graph", "ode_based_learnable"])
    parser.add_argument("--block-hours", type=int, default=168)
    parser.add_argument("--bootstrap-n", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260711)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--extreme-quantile", type=float, default=0.95)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays, _, _ = v3.build_enhanced_arrays()
    residual = arrays["residual"].astype(np.float64)
    train_end = int(len(residual) * args.train_ratio)
    train_abs_thresholds = np.quantile(
        np.abs(residual[:train_end]), args.extreme_quantile, axis=0
    )
    runs: dict[tuple[int | None, str], tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(input_root.rglob("predictions.npz")):
        model = path.parent.name
        if model not in {args.physical_key, *args.baseline_keys}:
            continue
        runs[(parse_seed(path), model)] = load_residual_predictions(path)

    rng = np.random.default_rng(args.random_seed)
    rows = []
    seeds = sorted({seed for seed, model in runs if model == args.physical_key and seed is not None})
    for seed in seeds:
        physical_pred, physical_true = runs[(seed, args.physical_key)]
        physical_last = physical_pred[:, :, -1]
        true_last = physical_true[:, :, -1]
        if true_last.shape[1] != len(train_abs_thresholds):
            raise ValueError(
                f"Station count mismatch: predictions={true_last.shape[1]}, "
                f"thresholds={len(train_abs_thresholds)}"
            )
        extreme_mask = np.abs(true_last) >= train_abs_thresholds[None, :]
        for baseline_key in args.baseline_keys:
            key = (seed, baseline_key)
            if key not in runs:
                continue
            baseline_pred, baseline_true = runs[key]
            n = min(len(true_last), len(baseline_true))
            if not np.allclose(true_last[-n:], baseline_true[-n:, :, -1], atol=1e-6):
                raise ValueError(f"True targets do not align for seed={seed}, baseline={baseline_key}")
            for subset, mask in [("all_last_step", None), ("train_threshold_abs_q95", extreme_mask)]:
                result = bootstrap_comparison(
                    true_last,
                    physical_last,
                    baseline_pred[:, :, -1],
                    args.block_hours,
                    args.bootstrap_n,
                    rng,
                    mask,
                )
                rows.append(
                    {
                        "seed": seed,
                        "physical_model": args.physical_key,
                        "baseline_model": baseline_key,
                        "subset": subset,
                        "block_hours": args.block_hours,
                        "threshold_scope": "training_period_per_station",
                        **result,
                    }
                )

    if not rows:
        raise RuntimeError("No matched seed/model prediction pairs found.")
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "block_bootstrap_by_seed.csv", index=False)
    summary = out.groupby(["baseline_model", "subset"])[
        ["observed_MSE_reduction", "bootstrap_probability_physical_better"]
    ].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(x for x in col if x) for col in summary.columns.to_flat_index()]
    summary.to_csv(output_dir / "block_bootstrap_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
