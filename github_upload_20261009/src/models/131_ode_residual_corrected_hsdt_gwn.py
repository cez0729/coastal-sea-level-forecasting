from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "ode_residual_corrected_hsdt_gwn"
SEEDS = [42, 123, 2024, 2025, 3407]
MODEL_NAMES = {
    "hsdt_gwn": "HS-DT-GWN",
    "hsdt_physics_correction": "HS-DT + physics correction",
    "hsdt_zero_adapter": "HS-DT + zero-prior correction",
    "hsdt_persistence_adapter": "HS-DT + persistence correction",
    "orc_hsdt_gwn": "ORC-HS-DT-GWN",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


hsdt = load_module("hsdt_impl_131", HERE / "108_horizon_specialized_dual_task_gwn.py")
p104 = hsdt.p104
priority1 = hsdt.priority1
v2 = hsdt.v2


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_hsdt(seed: int, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = (
        project_path(args.hsdt_results)
        / f"seed_{seed}"
        / "horizon_specialized_dual_task_gwn_predictions.npz"
    )
    with np.load(path) as payload:
        return (
            payload["pred_residual"].astype(np.float64),
            payload["true_residual"].astype(np.float64),
            payload["target_tide"].astype(np.float64),
        )


def load_correction(
    seed: int,
    source_root: Path,
    reference_true: np.ndarray,
    reference_tide: np.ndarray,
    reference_multistate: np.ndarray,
) -> tuple[np.ndarray, dict]:
    path = source_root / f"seed_{seed}" / "predictions.npz"
    with np.load(path) as payload:
        source_true = payload["true"][..., 0].astype(np.float64)
        source_tide = payload["tide"].astype(np.float64)
        if not np.allclose(source_true, reference_true, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"Target mismatch between HS-DT and correction source: {path}")
        if not np.allclose(source_tide, reference_tide, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"Tide mismatch between HS-DT and correction source: {path}")
        base = payload["base"][..., 0].astype(np.float64)
        corrected = payload["gated"][..., 0].astype(np.float64)
        correction = corrected - base
    audit = {
        "seed": seed,
        "source": str(source_root.relative_to(ROOT) if source_root.is_relative_to(ROOT) else source_root),
        "correction_mean": float(correction.mean()),
        "correction_abs_mean": float(np.abs(correction).mean()),
        "correction_abs_max": float(np.abs(correction).max()),
        "source_base_vs_locked_multistate_abs_mean": float(
            np.abs(base - reference_multistate).mean()
        ),
        "source_base_vs_locked_multistate_abs_max": float(
            np.abs(base - reference_multistate).max()
        ),
    }
    return correction, audit


def load_physics_pair(
    seed: int, args, reference_true: np.ndarray, reference_tide: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    root = project_path(args.physics_results) / f"seed_{seed}" / f"horizon_{args.horizon}h"
    with np.load(root / "gwn_multistate_no_physics" / "predictions.npz") as no_physics:
        locked_multistate = no_physics["pred_states"][..., 0].astype(np.float64)
        no_physics_true = no_physics["true_states"][..., 0].astype(np.float64)
        no_physics_tide = no_physics["target_tide"].astype(np.float64)
    with np.load(root / "gwn_multistate_physics" / "predictions.npz") as physics:
        physics_eta = physics["pred_states"][..., 0].astype(np.float64)
        physics_true = physics["true_states"][..., 0].astype(np.float64)
    if not np.allclose(no_physics_true, reference_true, atol=1e-6, rtol=0.0):
        raise RuntimeError(f"Target mismatch in locked multistate predictions for seed {seed}")
    if not np.allclose(physics_true, reference_true, atol=1e-6, rtol=0.0):
        raise RuntimeError(f"Target mismatch in physics predictions for seed {seed}")
    if not np.allclose(no_physics_tide, reference_tide, atol=1e-6, rtol=0.0):
        raise RuntimeError(f"Tide mismatch in locked multistate predictions for seed {seed}")
    return locked_multistate, physics_eta - locked_multistate


def summarize(true: np.ndarray, pred: np.ndarray, tide: np.ndarray, thresholds: np.ndarray) -> dict:
    return hsdt.summarize(true, pred, tide, thresholds)


def configured_sources(args) -> dict[str, Path]:
    sources = {
        "orc_hsdt_gwn": project_path(args.learned_ode_results),
        "hsdt_zero_adapter": project_path(args.zero_results),
        "hsdt_persistence_adapter": project_path(args.persistence_results),
    }
    if not args.require_controls:
        sources = {
            name: path
            for name, path in sources.items()
            if all((path / f"seed_{seed}" / "predictions.npz").exists() for seed in args.seeds)
        }
    missing = [
        str(path / f"seed_{seed}" / "predictions.npz")
        for path in sources.values()
        for seed in args.seeds
        if not (path / f"seed_{seed}" / "predictions.npz").exists()
    ]
    if missing:
        raise FileNotFoundError("Missing correction predictions:\n" + "\n".join(missing))
    return sources


def run(args) -> None:
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = hsdt.train_thresholds(args)
    sources = configured_sources(args)
    if "orc_hsdt_gwn" not in sources:
        raise FileNotFoundError("The learned-ODE correction source is required")

    rows = []
    lead_rows = []
    station_rows = []
    audit_rows = []
    for seed in args.seeds:
        base, true, tide = load_hsdt(seed, args)
        locked_multistate, physics_correction = load_physics_pair(seed, args, true, tide)
        predictions = {
            "hsdt_gwn": base,
            "hsdt_physics_correction": base + physics_correction,
        }
        corrections = {"hsdt_physics_correction": physics_correction}
        audit_rows.append(
            {
                "seed": seed,
                "source": str(Path(args.physics_results)),
                "model": "hsdt_physics_correction",
                "correction_scale": 1.0,
                "protect_terminal": False,
                "correction_mean": float(physics_correction.mean()),
                "correction_abs_mean": float(np.abs(physics_correction).mean()),
                "correction_abs_max": float(np.abs(physics_correction).max()),
                "source_base_vs_locked_multistate_abs_mean": 0.0,
                "source_base_vs_locked_multistate_abs_max": 0.0,
            }
        )
        for model_name, source_root in sources.items():
            correction, audit = load_correction(
                seed, source_root, true, tide, locked_multistate
            )
            if args.protect_terminal:
                correction[..., -1] = 0.0
            corrections[model_name] = correction
            predictions[model_name] = base + float(args.correction_scale) * correction
            audit.update(
                {
                    "model": model_name,
                    "correction_scale": float(args.correction_scale),
                    "protect_terminal": bool(args.protect_terminal),
                }
            )
            audit_rows.append(audit)

        for model_name, pred in predictions.items():
            metrics = summarize(true, pred, tide, thresholds)
            rows.append({"seed": seed, "model": model_name, **metrics})
            for lead in range(args.horizon):
                lead_metric = priority1.r2_rmse_mae(true[..., lead], pred[..., lead])
                lead_rows.append(
                    {"seed": seed, "model": model_name, "lead_hour": lead + 1, **lead_metric}
                )
            for station, station_id in enumerate(v2.STATION_IDS):
                station_metric = priority1.r2_rmse_mae(
                    true[:, station, -1], pred[:, station, -1]
                )
                station_rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "station_id": station_id,
                        **station_metric,
                    }
                )

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for model_name, pred in predictions.items():
            np.savez_compressed(
                seed_dir / f"{model_name}_predictions.npz",
                pred_residual=pred,
                true_residual=true,
                target_tide=tide,
                correction=corrections.get(model_name, np.zeros_like(base)),
                correction_scale=np.asarray(args.correction_scale),
                protect_terminal=np.asarray(args.protect_terminal),
                station_ids=np.asarray(v2.STATION_IDS),
            )

    all_runs = pd.DataFrame(rows)
    lead_data = pd.DataFrame(lead_rows)
    all_runs.to_csv(output_dir / "all_runs.csv", index=False)
    lead_data.to_csv(output_dir / "per_lead_metrics.csv", index=False)
    pd.DataFrame(station_rows).to_csv(
        output_dir / "per_station_terminal_metrics.csv", index=False
    )
    pd.DataFrame(audit_rows).to_csv(output_dir / "correction_transfer_audit.csv", index=False)
    summarize_results(all_runs, lead_data, output_dir)


def summarize_results(all_runs: pd.DataFrame, lead_data: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "event_CSI",
        "event_PR_AUC",
        "event_recall",
        "event_F1",
    ]
    metrics = [metric for metric in metrics if metric in all_runs]
    summary = all_runs.groupby("model")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part)
        for column in summary.columns.to_flat_index()
    ]
    summary.to_csv(output_dir / "mean_std.csv", index=False)

    pivot = all_runs.pivot(index="seed", columns="model", values=metrics)
    comparisons = []
    winner = "orc_hsdt_gwn"
    for baseline in [name for name in all_runs["model"].unique() if name != winner]:
        for metric in metrics:
            delta = pivot[(metric, winner)] - pivot[(metric, baseline)]
            improvement = -delta if metric.endswith(("RMSE", "MAE")) else delta
            comparisons.append(
                {
                    "comparison": f"{winner}_minus_{baseline}",
                    "metric": metric,
                    "mean_improvement": float(improvement.mean()),
                    "std_improvement": float(improvement.std()),
                    "wins": int((improvement > 0).sum()),
                    "count": int(improvement.notna().sum()),
                    "wilcoxon_greater_p": p104.exact_wilcoxon_greater(
                        improvement.to_numpy()
                    ),
                }
            )
    paired = pd.DataFrame(comparisons)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    plot_results(summary, lead_data, output_dir)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


def plot_results(summary: pd.DataFrame, lead_data: pd.DataFrame, output_dir: Path) -> None:
    order = [
        name
        for name in [
            "hsdt_gwn",
            "hsdt_physics_correction",
            "hsdt_zero_adapter",
            "hsdt_persistence_adapter",
            "orc_hsdt_gwn",
        ]
        if name in set(summary["model"])
    ]
    colors = ["#596780", "#A35D5D", "#9A9A9A", "#C98743", "#2B7A68"]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    display = [
        ("seq_residual_R2_mean", "Sequence R2"),
        ("last_residual_R2_mean", "24-h R2"),
        ("extreme_abs_q95_residual_R2_mean", "Descriptive q95 R2"),
    ]
    x = np.arange(len(display))
    correction_order = [name for name in order if name != "hsdt_gwn"]
    width = 0.8 / len(correction_order)
    lookup = summary.set_index("model")
    baseline = np.asarray(
        [float(lookup.loc["hsdt_gwn", metric]) for metric, _ in display]
    )
    for index, model_name in enumerate(correction_order):
        values = np.asarray(
            [float(lookup.loc[model_name, metric]) for metric, _ in display]
        ) - baseline
        axes[0].bar(
            x + (index - (len(correction_order) - 1) / 2) * width,
            values,
            width,
            color=colors[order.index(model_name)],
            label=MODEL_NAMES[model_name],
        )
    axes[0].axhline(0.0, color="#303030", linewidth=0.8)
    axes[0].set_xticks(x, [label for _, label in display])
    axes[0].set_ylabel("R2 change relative to HS-DT")
    axes[0].set_title("Five-seed correction gains")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7)

    mean_lead = lead_data.groupby(["model", "lead_hour"])["R2"].mean().reset_index()
    for color, model_name in zip(colors, order):
        subset = mean_lead[mean_lead["model"] == model_name]
        axes[1].plot(
            subset["lead_hour"],
            subset["R2"],
            color=color,
            label=MODEL_NAMES[model_name],
        )
    axes[1].set_xlabel("Forecast lead (h)")
    axes[1].set_ylabel("Residual R2")
    axes[1].set_title("Lead-dependent skill")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "orc_hsdt_gwn_results.png", dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer ODE-conditioned residual corrections onto locked HS-DT-GWN predictions."
    )
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    parser.add_argument("--output-dir", default="results/ode_residual_corrected_hsdt_gwn")
    parser.add_argument("--hsdt-results", default="results/horizon_specialized_dual_task_gwn")
    parser.add_argument(
        "--physics-results", default="results/priority12_physics_graph_wavenet"
    )
    parser.add_argument(
        "--learned-ode-results",
        default="results/ode_prior_gated_multistate_gwn_learned_repro_stride8",
    )
    parser.add_argument(
        "--zero-results", default="results/ode_prior_gated_multistate_gwn_zero_stride8"
    )
    parser.add_argument(
        "--persistence-results",
        default="results/ode_prior_gated_multistate_gwn_persistence_stride8",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--correction-scale", type=float, default=1.0)
    parser.add_argument("--protect-terminal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-controls", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "merge":
        all_runs = pd.read_csv(output_dir / "all_runs.csv")
        lead_data = pd.read_csv(output_dir / "per_lead_metrics.csv")
        summarize_results(all_runs, lead_data, output_dir)
        return
    (output_dir / "experiment_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    run(args)


if __name__ == "__main__":
    main()
