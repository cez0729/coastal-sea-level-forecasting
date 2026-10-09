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
import torch
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results" / "smooth_horizon_gated_gwn"
SOURCE_RESULTS = ROOT / "results" / "priority12_physics_graph_wavenet"
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p104 = load_module("priority2_gwn_p106", HERE / "104_priority2_physics_graph_wavenet.py")
final4 = p104.final4
priority1 = p104.priority1
v2 = p104.v2
v3 = p104.v3


def load_best_state(path: Path, model_kind: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if model_kind == "eta":
        return payload["best_state_dict"]
    return payload["model_state_dict"]


def load_experts(seed: int, data: dict, args, device: torch.device):
    adjacency = data["graph_priors"][args.fixed_graph_type]
    eta = priority1.GraphWaveNetForecaster(
        data["feats"],
        adjacency,
        args.hidden_dim,
        args.horizon,
        args.diffusion_steps,
        args.gwn_blocks,
        args.dropout,
    ).to(device)
    multistate = p104.GraphWaveNetMultistate(
        data["feats"],
        adjacency,
        args.hidden_dim,
        args.horizon,
        4,
        args.diffusion_steps,
        args.gwn_blocks,
        args.dropout,
    ).to(device)
    root = Path(args.source_results) / f"seed_{seed}" / f"horizon_{args.horizon}h"
    eta_state = load_best_state(root / "gwn_eta_only" / "last_epoch_checkpoint.pt", "eta")
    multi_state = load_best_state(root / "gwn_multistate_no_physics" / "best_checkpoint.pt", "multistate")
    eta.load_state_dict(eta_state)
    multistate.load_state_dict(multi_state)
    eta.eval()
    multistate.eval()
    return eta, multistate


@torch.no_grad()
def predict_experts(eta_model, multi_model, loader, device: torch.device):
    eta_predictions = []
    multi_predictions = []
    targets = []
    tides = []
    for xb, target, tide, _, _ in loader:
        xb = xb.to(device)
        eta_predictions.append(eta_model(xb).detach().cpu().numpy())
        multi_predictions.append(multi_model(xb).detach().cpu().numpy()[..., 0])
        targets.append(target.numpy()[..., 0])
        tides.append(tide.numpy())
    return (
        np.concatenate(eta_predictions),
        np.concatenate(multi_predictions),
        np.concatenate(targets),
        np.concatenate(tides),
    )


def gate_objective(
    weights: np.ndarray,
    eta: np.ndarray,
    multi: np.ndarray,
    target: np.ndarray,
    smoothness: float,
) -> float:
    fused = eta + weights[None, None, :] * (multi - eta)
    data_loss = float(np.mean((target - fused) ** 2))
    smooth_loss = float(np.mean(np.diff(weights) ** 2)) if len(weights) > 1 else 0.0
    return data_loss + smoothness * smooth_loss


def fit_gate(
    eta: np.ndarray,
    multi: np.ndarray,
    target: np.ndarray,
    smoothness: float,
) -> np.ndarray:
    horizon = eta.shape[-1]
    result = minimize(
        gate_objective,
        x0=np.full(horizon, 0.5, dtype=np.float64),
        args=(eta, multi, target, smoothness),
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * horizon,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Gate optimization failed: {result.message}")
    return np.clip(result.x, 0.0, 1.0)


def choose_smoothness(
    eta: np.ndarray,
    multi: np.ndarray,
    target: np.ndarray,
    candidates: list[float],
    fit_fraction: float,
) -> tuple[float, pd.DataFrame]:
    split = max(1, min(len(target) - 1, int(len(target) * fit_fraction)))
    rows = []
    for value in candidates:
        weights = fit_gate(eta[:split], multi[:split], target[:split], value)
        fused = eta[split:] + weights[None, None, :] * (multi[split:] - eta[split:])
        mse = float(np.mean((target[split:] - fused) ** 2))
        rows.append(
            {
                "smoothness": value,
                "validation_tail_mse": mse,
                "mean_multistate_weight": float(weights.mean()),
                "terminal_multistate_weight": float(weights[-1]),
            }
        )
    table = pd.DataFrame(rows).sort_values(["validation_tail_mse", "smoothness"])
    return float(table.iloc[0]["smoothness"]), table


def residual_metrics(true: np.ndarray, pred: np.ndarray, tide: np.ndarray, data: dict, args) -> dict:
    states_true = np.zeros((*true.shape, 4), dtype=np.float32)
    states_pred = np.zeros((*pred.shape, 4), dtype=np.float32)
    states_true[..., 0] = true
    states_pred[..., 0] = pred
    metrics = final4.summarize_single(true, pred, tide)
    train_end = int(len(data["arrays"]["residual"]) * args.train_ratio)
    thresholds = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    metrics.update(p104.operational_event_metrics(true, pred, thresholds))
    return metrics


def verify_source_predictions(seed: int, eta: np.ndarray, multi: np.ndarray, args) -> dict[str, float]:
    root = Path(args.source_results) / f"seed_{seed}" / f"horizon_{args.horizon}h"
    eta_saved = np.load(root / "gwn_eta_only" / "predictions.npz")["pred_residual"]
    multi_saved = np.load(root / "gwn_multistate_no_physics" / "predictions.npz")["pred_states"][..., 0]
    return {
        "eta_source_max_abs_difference": float(np.max(np.abs(eta_saved - eta))),
        "multistate_source_max_abs_difference": float(np.max(np.abs(multi_saved - multi))),
    }


def run_seed(seed: int, args, device: torch.device) -> dict:
    output_dir = Path(args.output_dir) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    if args.resume and metrics_path.exists() and (output_dir / "predictions.npz").exists():
        print(f"Skipping completed seed {seed}")
        return pd.read_csv(metrics_path).iloc[0].to_dict()
    p104.set_reproducible(seed, args.cpu_threads)
    data = final4.build_enhanced_data(args, args.horizon, add_ode_prior=False)
    eta_model, multi_model = load_experts(seed, data, args, device)
    val_loader = p104.make_loader(data["multi_val"], args, False, seed)
    test_loader = p104.make_loader(data["multi_test"], args, False, seed)
    val_eta, val_multi, val_true, val_tide = predict_experts(eta_model, multi_model, val_loader, device)
    selected_smoothness, tuning = choose_smoothness(
        val_eta,
        val_multi,
        val_true,
        args.smoothness_candidates,
        args.gate_fit_fraction,
    )
    gate = fit_gate(val_eta, val_multi, val_true, selected_smoothness)
    test_eta, test_multi, test_true, test_tide = predict_experts(
        eta_model, multi_model, test_loader, device
    )
    provenance = verify_source_predictions(seed, test_eta, test_multi, args)
    fused = test_eta + gate[None, None, :] * (test_multi - test_eta)
    metrics = residual_metrics(test_true, fused, test_tide, data, args)
    eta_metrics = residual_metrics(test_true, test_eta, test_tide, data, args)
    multi_metrics = residual_metrics(test_true, test_multi, test_tide, data, args)
    row = {
        "seed": seed,
        "model": "smooth_horizon_gated_dual_task_gwn",
        "selected_smoothness": selected_smoothness,
        "mean_multistate_weight": float(gate.mean()),
        "terminal_multistate_weight": float(gate[-1]),
        **provenance,
        **metrics,
        **{f"eta_expert_{key}": value for key, value in eta_metrics.items()},
        **{f"multistate_expert_{key}": value for key, value in multi_metrics.items()},
    }
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    tuning.to_csv(output_dir / "gate_smoothness_selection.csv", index=False)
    pd.DataFrame({"lead_hour": np.arange(1, args.horizon + 1), "multistate_weight": gate}).to_csv(
        output_dir / "horizon_gate_weights.csv", index=False
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        pred_residual=fused,
        true_residual=test_true,
        target_tide=test_tide,
        eta_expert_prediction=test_eta,
        multistate_expert_prediction=test_multi,
        horizon_gate=gate,
        station_ids=np.asarray(v2.STATION_IDS),
    )
    print(
        f"seed={seed} smooth={selected_smoothness:g} seq={row['seq_residual_R2']:.4f} "
        f"last={row['last_residual_R2']:.4f} eta_seq={eta_metrics['seq_residual_R2']:.4f} "
        f"multi_last={multi_metrics['last_residual_R2']:.4f}"
    )
    return row


def merge_results(args) -> None:
    output_dir = Path(args.output_dir)
    files = sorted(output_dir.glob("seed_*/metrics.csv"))
    if not files:
        raise RuntimeError(f"No completed metrics under {output_dir}")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    data = data[data["seed"].isin(args.seeds)].sort_values("seed").drop_duplicates("seed", keep="last")
    data.to_csv(output_dir / "all_runs.csv", index=False)
    metrics = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "event_PR_AUC",
        "event_recall",
        "eta_expert_seq_residual_R2",
        "eta_expert_last_residual_R2",
        "multistate_expert_seq_residual_R2",
        "multistate_expert_last_residual_R2",
        "mean_multistate_weight",
        "terminal_multistate_weight",
    ]
    metrics = [metric for metric in metrics if metric in data]
    summary = data[metrics].agg(["mean", "std", "count"]).T.reset_index(names="metric")
    summary.to_csv(output_dir / "mean_std.csv", index=False)
    paired = []
    for expert in ("eta_expert", "multistate_expert"):
        for metric in ("seq_residual_R2", "last_residual_R2", "last_residual_RMSE"):
            reference = f"{expert}_{metric}"
            if reference not in data:
                continue
            delta = data[metric] - data[reference]
            if metric.endswith("RMSE"):
                delta = -delta
            paired.append(
                {
                    "comparison": f"gated_minus_{expert}",
                    "metric": metric,
                    "mean_improvement": float(delta.mean()),
                    "std_improvement": float(delta.std()),
                    "wins": int((delta > 0).sum()),
                    "count": int(delta.notna().sum()),
                    "wilcoxon_greater_p": p104.exact_wilcoxon_greater(delta.to_numpy()),
                }
            )
    pd.DataFrame(paired).to_csv(output_dir / "paired_comparisons.csv", index=False)
    plot_results(data, output_dir)
    print(summary.to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))


def plot_results(data: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    model_values = {
        "Eta GWN": [data["eta_expert_seq_residual_R2"].mean(), data["eta_expert_last_residual_R2"].mean()],
        "Multistate GWN": [
            data["multistate_expert_seq_residual_R2"].mean(),
            data["multistate_expert_last_residual_R2"].mean(),
        ],
        "Smooth gated GWN": [data["seq_residual_R2"].mean(), data["last_residual_R2"].mean()],
    }
    x = np.arange(2)
    width = 0.24
    for index, (label, values) in enumerate(model_values.items()):
        axes[0].bar(x + (index - 1) * width, values, width, label=label)
    axes[0].set_xticks(x, ["Trajectory R2", "24-h terminal R2"])
    axes[0].set_ylabel("Residual R2")
    axes[0].set_title("Five-seed matched comparison")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    gates = []
    for seed in sorted(data["seed"]):
        gate_path = output_dir / f"seed_{int(seed)}" / "horizon_gate_weights.csv"
        gate = pd.read_csv(gate_path)
        axes[1].plot(gate["lead_hour"], gate["multistate_weight"], alpha=0.65, label=f"seed {int(seed)}")
        gates.append(gate["multistate_weight"].to_numpy())
    if gates:
        axes[1].plot(np.arange(1, len(gates[0]) + 1), np.mean(gates, axis=0), color="black", linewidth=2.4, label="mean")
    axes[1].set_xlabel("Forecast lead (h)")
    axes[1].set_ylabel("Multistate expert weight")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Validation-learned smooth horizon gate")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "smooth_horizon_gated_gwn_results.png", dpi=190)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only smooth horizon gating of two GWN experts.")
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--source-results", default=str(SOURCE_RESULTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--smoothness-candidates", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--gate-fit-fraction", type=float, default=0.70)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", choices=["identity", "distance", "corr"], default="distance")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--gwn-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if args.mode == "merge":
        merge_results(args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; seeds={args.seeds}")
    for seed in args.seeds:
        run_seed(seed, args, device)
    merge_results(args)


if __name__ == "__main__":
    main()
