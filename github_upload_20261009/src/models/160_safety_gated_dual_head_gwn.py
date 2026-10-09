from __future__ import annotations

"""Build a shared safety gate and a physics-aware event head on 2026 H1.

This script treats 2026 H1 as development data.  It never selects a gate per
seed: one station-by-horizon-group gate is learned from all five seeds and is
then frozen for evaluation on a later chronological period.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 2024, 2025, 3407]
CONTROL = "validation_locked_ridge_no_physics"
PHYSICS = "physics_reliability_hsdt_gwn"
CONTINUOUS_HEAD = "safety_gated_continuous_head"
DUAL_HEAD = "safety_gated_physics_dual_head"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


calibration = load_module("safety_gate_calibration", HERE / "156_validation_locked_physics_reliability_gwn.py")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_seed(source: Path, seed: int) -> dict[str, np.ndarray]:
    with np.load(source / f"seed_{seed}_{CONTROL}_predictions.npz") as payload:
        control = payload["pred_residual"].astype(np.float32)
        true = payload["true_residual"].astype(np.float32)
        tide = payload["target_tide"].astype(np.float32)
    with np.load(source / f"seed_{seed}_{PHYSICS}_predictions.npz") as payload:
        physics = payload["pred_residual"].astype(np.float32)
        if not np.array_equal(true, payload["true_residual"]):
            raise ValueError(f"Target mismatch between prediction files for seed {seed}")
    if control.shape != physics.shape or control.shape != true.shape:
        raise ValueError(f"Shape mismatch for seed {seed}: {control.shape}, {physics.shape}, {true.shape}")
    return {"control": control, "physics": physics, "true": true, "tide": tide}


def build_gate(
    payloads: dict[int, dict[str, np.ndarray]], group_hours: int, minimum_seed_wins: int
) -> tuple[np.ndarray, pd.DataFrame]:
    first = payloads[next(iter(payloads))]["true"]
    _, nodes, horizon = first.shape
    gate = np.zeros((nodes, horizon), dtype=bool)
    rows: list[dict] = []
    for station in range(nodes):
        for start in range(0, horizon, group_hours):
            end = min(start + group_hours, horizon)
            improvements = []
            control_mses = []
            physics_mses = []
            for seed, payload in payloads.items():
                target = payload["true"][:, station, start:end]
                control_mse = float(np.mean((target - payload["control"][:, station, start:end]) ** 2))
                physics_mse = float(np.mean((target - payload["physics"][:, station, start:end]) ** 2))
                control_mses.append(control_mse)
                physics_mses.append(physics_mse)
                improvements.append(control_mse - physics_mse)
            mean_improvement = float(np.mean(improvements))
            wins = int(np.sum(np.asarray(improvements) > 0.0))
            enabled = mean_improvement > 0.0 and wins >= minimum_seed_wins
            gate[station, start:end] = enabled
            rows.append(
                {
                    "station_index": station,
                    "lead_start_hour": start + 1,
                    "lead_end_hour": end,
                    "control_mse_mean": float(np.mean(control_mses)),
                    "physics_mse_mean": float(np.mean(physics_mses)),
                    "mse_improvement_mean": mean_improvement,
                    "mse_improvement_relative": mean_improvement / max(float(np.mean(control_mses)), 1e-12),
                    "seed_wins": wins,
                    "seed_count": len(improvements),
                    "physics_enabled": enabled,
                }
            )
    return gate, pd.DataFrame(rows)


def continuous_metrics(true: np.ndarray, pred: np.ndarray, tide: np.ndarray) -> dict[str, float]:
    return calibration.pc.confirm.final4.summarize_single(true, pred, tide)


def combine_dual_head_metrics(continuous: dict[str, float], physics_row: pd.Series) -> dict[str, float]:
    combined = dict(continuous)
    for metric in ("event_PR_AUC", "event_CSI", "event_POD", "event_FAR"):
        if metric in physics_row.index:
            combined[metric] = float(physics_row[metric])
    return combined


def write_protocol(out: Path, args: argparse.Namespace, gate: np.ndarray, gate_csv: Path) -> None:
    code_path = Path(__file__).resolve()
    protocol = f"""# Frozen safety-gated dual-head protocol

Freeze date: {pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}

## Evidence boundary

- The 2026 H1 period is development data for this model because it was used to learn the shared safety gate.
- The gate is shared by all seeds and has shape station x lead hour; it is not fitted separately per seed.
- Continuous residual forecasts use the physics prediction only in enabled cells and otherwise fall back to the matched no-physics prediction.
- Event ranking uses the full physics-reliability prediction as a separate event head.
- Continuous and event metrics must be attributed to their respective heads; the two outputs must not be presented as one numerical forecast.
- The frozen gate must not be changed after inspecting the later holdout targets.

## Gate rule

- Horizon group size: {args.group_hours} hours.
- Enable physics only when mean MSE improvement is positive and at least {args.minimum_seed_wins}/{len(args.seeds)} seeds improve on 2026 H1.
- Enabled station-group cells: {int(gate[:, ::args.group_hours].sum())}/{gate.shape[0] * int(np.ceil(gate.shape[1] / args.group_hours))}.

## Frozen hashes

- Model/gate code SHA-256: `{sha256(code_path)}`.
- Development gate CSV SHA-256: `{sha256(gate_csv)}`.
- Source H1 all-runs SHA-256: `{sha256(project_path(args.source_dir) / 'all_runs.csv')}`.

## Later-period acceptance criteria

Relative to the matched validation-locked no-physics control, evaluated over all five seeds:

1. Mean sequence residual R2 improvement is positive and at least 4/5 seeds win.
2. Mean Lead-24 residual R2 change is at least -0.002 and Lead-24 RMSE does not worsen by more than 1%.
3. Mean q95 residual R2 change is at least -0.01.
4. The physics event head improves mean event PR-AUC and at least 4/5 seeds win; CSI changes are reported regardless of sign.
5. A 168-hour hierarchical block bootstrap is reported for sequence, Lead-24 and q95 changes. Confirmed main-model language requires the sequence 95% interval lower bound to exceed zero.

If these criteria fail, the model remains exploratory and the gate is not retuned on the same holdout.
"""
    (out / "FROZEN_LATER_HOLDOUT_PROTOCOL.md").write_text(protocol, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a shared safety-gated physics dual-head GWN")
    parser.add_argument("--source-dir", default="results/frozen_2026_h1_physics_reliability_gwn")
    parser.add_argument("--output-dir", default="results/development_2026_h1_safety_gated_dual_head_gwn")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--group-hours", type=int, default=4)
    parser.add_argument("--minimum-seed-wins", type=int, default=4)
    args = parser.parse_args()

    source = project_path(args.source_dir)
    out = project_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payloads = {seed: load_seed(source, seed) for seed in args.seeds}
    shapes = {payload["true"].shape for payload in payloads.values()}
    if len(shapes) != 1:
        raise ValueError(f"Seeds do not share one prediction shape: {shapes}")

    gate, gate_table = build_gate(payloads, args.group_hours, args.minimum_seed_wins)
    gate_csv = out / "development_gate.csv"
    gate_table.to_csv(gate_csv, index=False)
    source_runs = pd.read_csv(source / "all_runs.csv")
    rows: list[dict] = []
    comparisons: list[dict] = []
    for seed, payload in payloads.items():
        gated = np.where(gate[None, :, :], payload["physics"], payload["control"]).astype(np.float32)
        continuous = continuous_metrics(payload["true"], gated, payload["tide"])
        control_row = source_runs[(source_runs["seed"] == seed) & (source_runs["model"] == CONTROL)].iloc[0]
        physics_row = source_runs[(source_runs["seed"] == seed) & (source_runs["model"] == PHYSICS)].iloc[0]
        rows.append({"seed": seed, "model": CONTINUOUS_HEAD, "metric_source": "gated_continuous_output", **continuous})
        rows.append(
            {
                "seed": seed,
                "model": DUAL_HEAD,
                "metric_source": "continuous=gated; event=full_physics_head",
                **combine_dual_head_metrics(continuous, physics_row),
            }
        )
        for metric in ("seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2"):
            delta = float(continuous[metric] - control_row[metric])
            improvement = -delta if metric.endswith(("RMSE", "MAE")) else delta
            comparisons.append({"seed": seed, "metric": metric, "improvement_over_no_physics": improvement})
        comparisons.append(
            {
                "seed": seed,
                "metric": "event_PR_AUC",
                "improvement_over_no_physics": float(physics_row["event_PR_AUC"] - control_row["event_PR_AUC"]),
            }
        )
        np.savez_compressed(
            out / f"seed_{seed}_safety_gated_dual_head_predictions.npz",
            pred_residual_continuous=gated,
            pred_residual_event_score=payload["physics"],
            pred_residual_control=payload["control"],
            true_residual=payload["true"],
            target_tide=payload["tide"],
        )

    runs = pd.DataFrame(rows)
    runs.to_csv(out / "development_all_runs.csv", index=False)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out / "development_paired_improvements.csv", index=False)
    summary = comparison.groupby("metric")["improvement_over_no_physics"].agg(["mean", "std", "min", "max"])
    summary["wins"] = comparison.groupby("metric")["improvement_over_no_physics"].apply(lambda x: int((x > 0).sum()))
    summary.to_csv(out / "development_improvement_summary.csv")

    gate_payload = {
        "development_period": "2026-01-01 through 2026-06-30 complete-case windows",
        "source_dir": str(source),
        "seeds": args.seeds,
        "group_hours": args.group_hours,
        "minimum_seed_wins": args.minimum_seed_wins,
        "gate_matrix_station_by_lead": gate.astype(int).tolist(),
        "enabled_lead_cells": int(gate.sum()),
        "total_lead_cells": int(gate.size),
        "event_head": PHYSICS,
        "continuous_control": CONTROL,
        "h1_is_development_not_confirmation": True,
    }
    (out / "development_gate.json").write_text(json.dumps(gate_payload, indent=2), encoding="utf-8")
    (out / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    write_protocol(out, args, gate, gate_csv)
    print(gate_table.to_string(index=False))
    print("\nDevelopment improvements over matched no-physics control:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
