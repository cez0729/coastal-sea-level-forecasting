from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "neuralcora_surge_publication_field_models"

METRIC_DIRECTIONS = {
    "rmse_m": "lower",
    "mae_m": "lower",
    "r2": "higher",
    "q95_csi": "higher",
    "q95_recall": "higher",
    "q99_csi": "higher",
    "q99_recall": "higher",
    "field_peak_mae_m": "lower",
}


def paired_event_bootstrap(
    frame: pd.DataFrame,
    baseline: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    required = {"model", "event_group", *METRIC_DIRECTIONS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Per-event table is missing columns: {missing}")
    event_means = frame.groupby(["model", "event_group"], as_index=False)[list(METRIC_DIRECTIONS)].mean()
    if baseline not in set(event_means["model"]):
        raise ValueError(f"Baseline {baseline!r} is not present")
    baseline_frame = event_means[event_means["model"] == baseline].set_index("event_group")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for model in sorted(set(event_means["model"]) - {baseline}):
        model_frame = event_means[event_means["model"] == model].set_index("event_group")
        common = baseline_frame.index.intersection(model_frame.index)
        if len(common) < 2:
            continue
        for metric, direction in METRIC_DIRECTIONS.items():
            baseline_values = baseline_frame.loc[common, metric].to_numpy(dtype=float)
            model_values = model_frame.loc[common, metric].to_numpy(dtype=float)
            valid = np.isfinite(baseline_values) & np.isfinite(model_values)
            baseline_values, model_values = baseline_values[valid], model_values[valid]
            if len(model_values) < 2:
                continue
            difference = model_values - baseline_values
            improvement = difference if direction == "higher" else -difference
            samples = rng.integers(0, len(improvement), size=(replicates, len(improvement)))
            bootstrap = improvement[samples].mean(axis=1)
            probability_positive = float(np.mean(bootstrap > 0))
            rows.append(
                {
                    "model": model,
                    "baseline": baseline,
                    "metric": metric,
                    "direction": direction,
                    "events": int(len(improvement)),
                    "model_event_mean": float(model_values.mean()),
                    "baseline_event_mean": float(baseline_values.mean()),
                    "improvement_mean": float(improvement.mean()),
                    "improvement_ci_lower": float(np.quantile(bootstrap, 0.025)),
                    "improvement_ci_upper": float(np.quantile(bootstrap, 0.975)),
                    "probability_improvement": probability_positive,
                    "two_sided_p": float(min(1.0, 2 * min(probability_positive, 1 - probability_positive))),
                    "replicates": replicates,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired event-block bootstrap for NeuralCORA-Surge field models")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--baseline", default="persistence")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    source = args.results_dir / "per_event_all_models.csv"
    frame = pd.read_csv(source)
    result = paired_event_bootstrap(frame, args.baseline, args.replicates, args.seed)
    target = args.results_dir / "event_block_bootstrap.csv"
    result.to_csv(target, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()
