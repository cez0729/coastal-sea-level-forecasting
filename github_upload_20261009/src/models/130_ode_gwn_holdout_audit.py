from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "ode_gwn_holdout_audit_2025_h2"
SOURCE_RESULTS = ROOT / "results" / "priority12_physics_graph_wavenet"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


impl = load_module("ode_gwn_impl_130", HERE / "109_ode_prior_gated_multistate_gwn.py")
rolling = load_module("rolling_impl_130", HERE / "96_rolling_origin_validation.py")
v2 = impl.v2


def load_base(seed: int, data: dict, args, device: torch.device):
    return impl.load_base_model(seed, data, args, device)


def backbone_audit_metadata(data: dict, args) -> dict:
    """Describe the chronological coverage of the reused formal checkpoint."""
    times = pd.to_datetime(data["arrays"]["time"])
    formal_train_end = int(len(times) * 0.70)
    formal_val_end = int(len(times) * 0.85)
    adapter_val_start = int(
        np.searchsorted(times.to_numpy(), np.datetime64(args.fold_train_end))
    )
    adapter_test_start = int(
        np.searchsorted(times.to_numpy(), np.datetime64(args.fold_val_end))
    )
    adapter_test_end = int(
        np.searchsorted(times.to_numpy(), np.datetime64(args.fold_test_end))
    )
    val_train_overlap = max(
        0, min(adapter_test_start, formal_train_end) - adapter_val_start
    )
    val_validation_overlap = max(
        0,
        min(adapter_test_start, formal_val_end)
        - max(adapter_val_start, formal_train_end),
    )
    test_validation_overlap = max(
        0,
        min(adapter_test_end, formal_val_end)
        - max(adapter_test_start, formal_train_end),
    )
    return {
        "backbone_source_split": "full_2023_2025_chronological_70_15_15",
        "backbone_training_preprocessing": "historical_retrospective_aligned_forcing",
        "backbone_train_end_exclusive": str(times[formal_train_end]),
        "backbone_validation_end_exclusive": str(times[formal_val_end]),
        "adapter_validation_backbone_train_overlap_hours": int(val_train_overlap),
        "adapter_validation_backbone_validation_overlap_hours": int(val_validation_overlap),
        "adapter_test_backbone_validation_overlap_hours": int(test_validation_overlap),
        "adapter_validation_independent_of_backbone": False,
        "independent_backbone_holdout": False,
        "full_model_strict_causal_training": False,
    }


def run_seed(seed: int, prior_mode: str, args, device: torch.device) -> dict:
    out_dir = Path(args.output_dir) / f"{prior_mode}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    if args.resume and metrics_path.exists():
        return pd.read_csv(metrics_path).iloc[0].to_dict()

    impl.set_seed(seed, args.cpu_threads)
    data = rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    base = load_base(seed, data, args, device)
    ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    model = impl.ODEPriorGatedModel(
        base, ode, args.horizon, args.initial_gate, args.refiner_hidden, prior_mode
    ).to(device)
    model.ode_adj = torch.tensor(
        data["graph_priors"][args.fixed_graph_type], dtype=torch.float32, device=device
    )
    train_loader = impl.make_loader(data["multi_train"], args, True, seed)
    val_loader = impl.make_loader(data["multi_val"], args, False, seed)
    test_loader = impl.make_loader(data["multi_test"], args, False, seed)
    history, best_val, seconds = impl.train_one(
        model, data, train_loader, val_loader, args, device, out_dir
    )
    outputs = impl.predict_all(model, test_loader, device)
    metrics = impl.score_prediction(outputs["final"], outputs["true"], outputs["tide"], data, args)
    backbone_audit = backbone_audit_metadata(data, args)
    row = {
        "seed": seed,
        "prior_mode": prior_mode,
        "fold": "test_2025_h2",
        "train_end": args.fold_train_end,
        "val_end": args.fold_val_end,
        "test_end": args.fold_test_end,
        "strict_causal_preprocessing": True,
        "strict_causal_adapter_inputs": True,
        "frozen_formal_gwn_checkpoint": True,
        **backbone_audit,
        "best_val_eta_loss": best_val,
        "training_seconds": seconds,
        "gate_values": json.dumps(model.gate_values().tolist()),
        **metrics,
    }
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    np.savez_compressed(
        out_dir / "predictions.npz",
        final=outputs["final"],
        base=outputs["base"],
        prior=outputs["prior"],
        true=outputs["true"],
        tide=outputs["tide"],
        station_ids=np.asarray(v2.STATION_IDS),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "seed": seed,
                "prior_mode": prior_mode,
                "fold": "test_2025_h2",
                "train_end": args.fold_train_end,
                "val_end": args.fold_val_end,
                "test_end": args.fold_test_end,
                "strict_causal_preprocessing": True,
                "strict_causal_adapter_inputs": True,
                "frozen_formal_gwn_checkpoint": True,
                **backbone_audit,
                "best_val_eta_loss": best_val,
            },
            "feature_cols": data["feature_cols"],
            "physics_cols": data["physics_cols"],
            "graph_priors": data["graph_priors"],
            "state_scale": data["state_scale"],
            "delta_scale": data["delta_scale"],
            "x_scaler_state": data["x_scaler_state"],
            "physics_scaler_state": data["physics_scaler_state"],
        },
        out_dir / "best_checkpoint.pt",
    )
    print(
        f"holdout seed={seed} prior={prior_mode} seq={row['seq_residual_R2']:.4f} "
        f"terminal={row['last_residual_R2']:.4f} q95={row['extreme_abs_q95_residual_R2']:.4f}"
    )
    return row


def summarize(args):
    root = Path(args.output_dir)
    files = sorted(root.glob("*/seed_*/metrics.csv"))
    if not files:
        raise RuntimeError("No holdout metrics found")
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    df.to_csv(root / "holdout_all_runs.csv", index=False)
    metrics = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "event_CSI",
        "event_PR_AUC",
    ]
    summary = df.groupby("prior_mode")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(x) for x in c if x) for c in summary.columns.to_flat_index()]
    summary.to_csv(root / "holdout_mean_std.csv", index=False)
    comparisons = []
    piv = df.pivot(index="seed", columns="prior_mode", values=metrics)
    for mode in ["learned_ode", "persistence"]:
        for metric in metrics:
            if (metric, mode) not in piv or (metric, "zero") not in piv:
                continue
            delta = piv[(metric, mode)] - piv[(metric, "zero")]
            if metric.endswith("RMSE"):
                delta = -delta
            comparisons.append(
                {
                    "comparison": f"{mode}_minus_zero",
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "std_delta": float(delta.std()),
                    "wins": int((delta > 0).sum()),
                    "count": int(delta.notna().sum()),
                    "deltas": json.dumps(delta.to_numpy().tolist()),
                }
            )
    pd.DataFrame(comparisons).to_csv(root / "holdout_paired_deltas.csv", index=False)
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoints(args, device: torch.device) -> None:
    root = Path(args.output_dir)
    data = rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    backbone_audit = backbone_audit_metadata(data, args)
    preprocessing_payload = {
        "metadata": {
            "fold": "test_2025_h2",
            "train_end": args.fold_train_end,
            "val_end": args.fold_val_end,
            "test_end": args.fold_test_end,
            "strict_causal_preprocessing": True,
            "strict_causal_adapter_inputs": True,
            **backbone_audit,
        },
        "feature_cols": data["feature_cols"],
        "physics_cols": data["physics_cols"],
        "graph_priors": data["graph_priors"],
        "state_scale": data["state_scale"],
        "delta_scale": data["delta_scale"],
        "x_scaler_state": data["x_scaler_state"],
        "physics_scaler_state": data["physics_scaler_state"],
    }
    torch.save(preprocessing_payload, root / "preprocessing_state.pt")

    required = {
        "model_state_dict",
        "metadata",
        "feature_cols",
        "physics_cols",
        "graph_priors",
        "state_scale",
        "delta_scale",
    }
    rows = []
    for prior_mode in args.prior_modes:
        for seed in args.seeds:
            path = root / prior_mode / f"seed_{seed}" / "best_checkpoint.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            missing = sorted(required.difference(payload))
            if missing:
                raise KeyError(f"Checkpoint {path} is missing keys: {missing}")
            if list(payload["feature_cols"]) != list(data["feature_cols"]):
                raise ValueError(f"Feature order mismatch in {path}")
            if list(payload["physics_cols"]) != list(data["physics_cols"]):
                raise ValueError(f"Physics feature order mismatch in {path}")
            graph_shapes = {name: tuple(value.shape) for name, value in payload["graph_priors"].items()}
            if graph_shapes != {"identity": (7, 7), "distance": (7, 7), "corr": (7, 7)}:
                raise ValueError(f"Unexpected graph shapes in {path}: {graph_shapes}")
            if tuple(payload["state_scale"].shape) != (4,) or tuple(payload["delta_scale"].shape) != (4,):
                raise ValueError(f"Unexpected state scaling shape in {path}")

            base = load_base(seed, data, args, device)
            ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
            model = impl.ODEPriorGatedModel(
                base, ode, args.horizon, args.initial_gate, args.refiner_hidden, prior_mode
            ).to(device)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            rows.append(
                {
                    "seed": seed,
                    "prior_mode": prior_mode,
                    "checkpoint": str(path),
                    "load_strict_passed": True,
                    "feature_count": len(payload["feature_cols"]),
                    "physics_feature_count": len(payload["physics_cols"]),
                    "state_dict_key_count": len(payload["model_state_dict"]),
                    "checkpoint_has_embedded_scalers": all(
                        key in payload for key in ["x_scaler_state", "physics_scaler_state"]
                    ),
                    "preprocessing_sidecar": str(root / "preprocessing_state.pt"),
                    "sha256": file_sha256(path),
                    **backbone_audit,
                }
            )
    validation = pd.DataFrame(rows)
    validation.to_csv(root / "checkpoint_validation.csv", index=False)
    metric_names = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "event_CSI",
        "event_PR_AUC",
    ]
    comparison_rows = []
    for prior_mode in args.prior_modes:
        for seed in args.seeds:
            current_path = root / prior_mode / f"seed_{seed}" / "metrics.csv"
            reference_path = DEFAULT_OUT / prior_mode / f"seed_{seed}" / "metrics.csv"
            if not current_path.exists() or not reference_path.exists():
                continue
            current = pd.read_csv(current_path).iloc[0]
            reference = pd.read_csv(reference_path).iloc[0]
            for metric in metric_names:
                comparison_rows.append(
                    {
                        "seed": seed,
                        "prior_mode": prior_mode,
                        "metric": metric,
                        "rerun_value": float(current[metric]),
                        "reference_value": float(reference[metric]),
                        "delta": float(current[metric] - reference[metric]),
                    }
                )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(root / "rerun_metric_comparison.csv", index=False)
    max_abs_metric_delta = (
        float(comparison["delta"].abs().max()) if not comparison.empty else None
    )
    (root / "CHECKPOINT_AUDIT.json").write_text(
        json.dumps(
            {
                "all_passed": bool(validation["load_strict_passed"].all()),
                "checkpoint_count": int(len(validation)),
                "feature_count": int(len(data["feature_cols"])),
                "physics_feature_count": int(len(data["physics_cols"])),
                "preprocessing_sidecar": str(root / "preprocessing_state.pt"),
                "metric_comparison_count": int(len(comparison)),
                "max_abs_metric_delta": max_abs_metric_delta,
                "exact_metric_reproduction": max_abs_metric_delta == 0.0,
                **backbone_audit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(validation.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Strict-causal 2025 H2 holdout audit for ODE-GWN residual adapters.")
    parser.add_argument("--mode", choices=["run", "merge", "validate"], default="run")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--source-results", default="results/priority12_physics_graph_wavenet")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--prior-modes", nargs="+", choices=["learned_ode", "zero", "persistence"], default=["learned_ode", "zero", "persistence"])
    parser.add_argument("--fold-train-end", default="2025-01-01")
    parser.add_argument("--fold-val-end", default="2025-07-01")
    parser.add_argument("--fold-test-end", default="2026-01-01")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", choices=["identity", "distance", "corr"], default="distance")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--gwn-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--prior-loss-weight", type=float, default=0.05)
    parser.add_argument("--gate-smooth-weight", type=float, default=0.01)
    parser.add_argument("--ode-reg-weight", type=float, default=1e-5)
    parser.add_argument("--initial-gate", type=float, default=0.8)
    parser.add_argument("--refiner-hidden", type=int, default=128)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    parser.add_argument("--print-every", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "merge":
        summarize(args)
        return
    if args.mode == "validate":
        (Path(args.output_dir) / "validation_config.json").write_text(
            json.dumps(vars(args), indent=2), encoding="utf-8"
        )
        validate_checkpoints(args, device)
        return
    (Path(args.output_dir) / "experiment_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    rows = []
    for prior_mode in args.prior_modes:
        for seed in args.seeds:
            rows.append(run_seed(seed, prior_mode, args, device))
    summarize(args)


if __name__ == "__main__":
    main()
