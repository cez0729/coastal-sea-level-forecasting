from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parent / "outputs" / "final_four_models_24h_5seed_matched"
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "ensemble_uncertainty_calibration"


def load_residual_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "pred_states" in data.files:
        return data["pred_states"][..., 0], data["true_states"][..., 0]
    return data["pred_residual"], data["true_residual"]


def parse_seed(path: Path) -> int | None:
    matches = re.findall(r"seed[_]?(\d+)", str(path))
    return int(matches[-1]) if matches else None


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.reshape(-1).astype(float)
    yp = y_pred.reshape(-1).astype(float)
    return float(1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-seed ensemble uncertainty and interval calibration.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--models", nargs="+", default=["gnn_bigru", "learnable_graph", "ode_based_learnable", "physical_loss"])
    parser.add_argument("--calibration-fraction", type=float, default=0.30)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {model: [] for model in args.models}
    for path in sorted(input_root.rglob("predictions.npz")):
        model = path.parent.name
        seed = parse_seed(path)
        if model in runs and seed is not None:
            pred, true = load_residual_predictions(path)
            runs[model].append((seed, pred, true))

    rows = []
    for model, model_runs in runs.items():
        if len(model_runs) < 3:
            print(f"Skip {model}: need at least 3 seeds, found {len(model_runs)}")
            continue
        model_runs.sort(key=lambda x: x[0])
        n = min(len(x[1]) for x in model_runs)
        preds = np.stack([x[1][-n:] for x in model_runs], axis=0)
        true = model_runs[0][2][-n:]
        for _, _, other_true in model_runs[1:]:
            if not np.allclose(true, other_true[-n:], atol=1e-6):
                raise ValueError(f"True targets do not align across seeds for {model}")

        ensemble_mean = np.mean(preds, axis=0)
        ensemble_std = np.std(preds, axis=0, ddof=1)
        true_last = true[:, :, -1]
        mean_last = ensemble_mean[:, :, -1]
        std_last = ensemble_std[:, :, -1]
        threshold = float(np.quantile(np.abs(true_last), 0.95))
        extreme = np.abs(true_last) >= threshold

        base = {
            "model_key": model,
            "n_members": len(model_runs),
            "seeds": ";".join(str(x[0]) for x in model_runs),
            "ensemble_last_R2": r2_score(true_last, mean_last),
            "ensemble_last_RMSE": float(np.sqrt(np.mean((true_last - mean_last) ** 2))),
            "ensemble_extreme_q95_R2": r2_score(true_last[extreme], mean_last[extreme]),
        }
        for nominal, z in [(0.80, 1.2815515655), (0.90, 1.6448536270), (0.95, 1.9599639845)]:
            lower = mean_last - z * std_last
            upper = mean_last + z * std_last
            covered = (true_last >= lower) & (true_last <= upper)
            base[f"coverage_{int(nominal * 100)}"] = float(np.mean(covered))
            base[f"mean_width_{int(nominal * 100)}"] = float(np.mean(upper - lower))
            base[f"extreme_coverage_{int(nominal * 100)}"] = float(np.mean(covered[extreme]))

            # Chronological split-conformal calibration. The first portion of
            # the test timeline is used only for interval calibration; coverage
            # is reported on the remaining, later evaluation period.
            calibration_end = max(1, min(len(true_last) - 1, int(len(true_last) * args.calibration_fraction)))
            eps = 1e-6
            calibration_scores = np.abs(true_last[:calibration_end] - mean_last[:calibration_end]) / (
                std_last[:calibration_end] + eps
            )
            alpha = 1.0 - nominal
            q_level = min(1.0, np.ceil((calibration_scores.size + 1) * (1.0 - alpha)) / calibration_scores.size)
            q_hat = float(np.quantile(calibration_scores.reshape(-1), q_level, method="higher"))
            eval_true = true_last[calibration_end:]
            eval_mean = mean_last[calibration_end:]
            eval_std = std_last[calibration_end:]
            conformal_lower = eval_mean - q_hat * (eval_std + eps)
            conformal_upper = eval_mean + q_hat * (eval_std + eps)
            conformal_covered = (eval_true >= conformal_lower) & (eval_true <= conformal_upper)
            eval_threshold = float(np.quantile(np.abs(eval_true), 0.95))
            eval_extreme = np.abs(eval_true) >= eval_threshold
            base[f"conformal_qhat_{int(nominal * 100)}"] = q_hat
            base[f"conformal_coverage_{int(nominal * 100)}"] = float(np.mean(conformal_covered))
            base[f"conformal_mean_width_{int(nominal * 100)}"] = float(
                np.mean(conformal_upper - conformal_lower)
            )
            base[f"conformal_extreme_coverage_{int(nominal * 100)}"] = float(
                np.mean(conformal_covered[eval_extreme])
            )
        rows.append(base)

        np.savez_compressed(
            output_dir / f"{model}_ensemble_predictions.npz",
            ensemble_mean=ensemble_mean,
            ensemble_std=ensemble_std,
            true_residual=true,
            seeds=np.asarray([x[0] for x in model_runs]),
        )

    if not rows:
        raise RuntimeError("No model had at least three aligned seed predictions.")
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "ensemble_uncertainty_metrics.csv", index=False)
    print(df.to_string(index=False))
    print("\nNote: seed ensembles estimate epistemic/model uncertainty only; they are not full predictive intervals.")


if __name__ == "__main__":
    main()
