from __future__ import annotations

"""Validation-only selection of a causal forcing lag for the physics gate."""

import argparse
import importlib.util
import json
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


def causal_lag(mask: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    if lag == 0:
        return mask.copy()
    out[lag:] = mask[:-lag]
    return out


def mse_pair(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    return float(np.mean((true - pred) ** 2)), float(np.mean((true[..., -1] - pred[..., -1]) ** 2))


def model_args() -> SimpleNamespace:
    return SimpleNamespace(
        train_ratio=0.70, val_ratio=0.15, window=24, train_stride=8,
        physics_forcing_mode="last_input", extreme_quantile=0.90,
        event_quantile=0.95, batch_size=512, horizon=24,
        fixed_graph_type="distance", hidden_dim=64, diffusion_steps=2,
        gwn_blocks=6, dropout=0.15, source_results="results/priority12_physics_graph_wavenet",
        initial_gate=0.80, refiner_hidden=128, prior_loss_weight=0.05,
        gate_smooth_weight=0.01, ode_reg_weight=1e-4, aux_weight=0.08,
        lr=5e-4, weight_decay=1e-5, grad_clip=1.0, patience=9,
        min_delta=1e-5, print_every=5, cpu_threads=4, resume=True,
        output_dir="results/physics_regime_switched_hsdt_benchmark",
        learned_ode_results="results/ode_prior_gated_multistate_gwn_learned_repro_stride8",
        persistence_results="results/ode_prior_gated_multistate_gwn_persistence_stride8",
    )


def validation_pair(m149, seed: int, data, args, device):
    loader = m149.adapter.make_loader(data["multi_val"], args, False, seed)
    learned_model = m149.load_trained_adapter(seed, "learned_ode", data, args, device)
    persistence_model = m149.load_trained_adapter(seed, "persistence", data, args, device)
    learned = m149.adapter.predict_all(learned_model, loader, device)
    persistence = m149.adapter.predict_all(persistence_model, loader, device)
    if not np.allclose(learned["true"], persistence["true"], atol=1e-6, rtol=0.0):
        raise RuntimeError(f"Validation target mismatch for seed {seed}")
    return learned["final"][..., 0], persistence["final"][..., 0], learned["true"][..., 0]


def test_pair(seed: int):
    learned_path = ROOT / "results/ode_prior_gated_multistate_gwn_learned_repro_stride8" / f"seed_{seed}" / "predictions.npz"
    persistence_path = ROOT / "results/ode_prior_gated_multistate_gwn_persistence_stride8" / f"seed_{seed}" / "predictions.npz"
    with np.load(learned_path) as z:
        learned = z["gated"][..., 0].astype(np.float64)
        true = z["true"][..., 0].astype(np.float64)
    with np.load(persistence_path) as z:
        persistence = z["gated"][..., 0].astype(np.float64)
        if not np.allclose(true, z["true"][..., 0], atol=1e-6, rtol=0.0):
            raise RuntimeError(f"Test target mismatch for seed {seed}")
    return learned, persistence, true


def run(args: argparse.Namespace) -> None:
    m149 = load_module("m149_lagged_gate", ROOT / "数据整理" / "149_physics_regime_switched_hsdt.py")
    cfg = model_args()
    data = m149.final4.build_enhanced_data(cfg, args.horizon, add_ode_prior=False)
    forcing = m149.build_forcing_intensity(
        data, SimpleNamespace(train_ratio=cfg.train_ratio, candidate_quantiles=args.quantiles, regime_quantiles=args.quantiles)
    )
    val_intensity = forcing["combined"][data["multi_val"].indices - 1]
    test_intensity = forcing["combined"][data["multi_test"].indices - 1]
    thresholds = {q: forcing["thresholds"][q] for q in args.quantiles}
    device = m149.torch.device("cuda" if m149.torch.cuda.is_available() else "cpu")

    val_rows = []
    for seed in args.seeds:
        learned, persistence, true = validation_pair(m149, seed, data, cfg, device)
        for q in args.quantiles:
            high = val_intensity >= thresholds[q][None, :]
            for lag in args.lags:
                mask = causal_lag(high, lag)
                hybrid = np.where(mask[..., None], learned, persistence)
                seq_mse, lead_mse = mse_pair(true, hybrid)
                val_rows.append({"seed": seed, "quantile": q, "lag_hours": lag, "coverage": float(mask.mean()), "val_sequence_MSE": seq_mse, "val_lead24_MSE": lead_mse})
    val = pd.DataFrame(val_rows)
    grid = val.groupby(["quantile", "lag_hours"], as_index=False).agg(val_sequence_MSE=("val_sequence_MSE", "mean"), val_lead24_MSE=("val_lead24_MSE", "mean"), coverage=("coverage", "mean"))
    selected = grid.sort_values(["val_lead24_MSE", "quantile", "lag_hours"]).iloc[0]
    selected_q = float(selected["quantile"])
    selected_lag = int(selected["lag_hours"])

    test_rows = []
    for seed in args.seeds:
        learned, persistence, true = test_pair(seed)
        high = test_intensity >= thresholds[selected_q][None, :]
        mask = causal_lag(high, selected_lag)
        hybrid = np.where(mask[..., None], learned, persistence)
        for name, pred in [("learned_ode_adapter", learned), ("persistence_adapter", persistence), ("lagged_physics_gate", hybrid)]:
            seq_mse, lead_mse = mse_pair(true, pred)
            test_rows.append({"seed": seed, "model": name, "selected_quantile": selected_q, "selected_lag_hours": selected_lag, "physics_usage_fraction": float(mask.mean()) if name == "lagged_physics_gate" else np.nan, "seq_MSE": seq_mse, "lead24_MSE": lead_mse})
    test = pd.DataFrame(test_rows)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    val.to_csv(out / "lag_validation_grid.csv", index=False)
    grid.to_csv(out / "lag_validation_grid_summary.csv", index=False)
    test.to_csv(out / "lagged_gate_test_metrics.csv", index=False)
    summary = test.groupby("model", as_index=False).agg(seq_MSE_mean=("seq_MSE", "mean"), lead24_MSE_mean=("lead24_MSE", "mean"))
    summary.to_csv(out / "lagged_gate_test_summary.csv", index=False)
    persistence = test[test.model == "persistence_adapter"].set_index("seed")["lead24_MSE"]
    lagged = test[test.model == "lagged_physics_gate"].set_index("seed")["lead24_MSE"]
    delta = float((persistence - lagged).mean())
    selection = {"selected_quantile": selected_q, "selected_lag_hours": selected_lag, "candidate_quantiles": args.quantiles, "candidate_lags": args.lags, "test_lead24_MSE_reduction_vs_persistence": delta}
    (out / "lagged_gate_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    lines = [
        "# Validation-locked lagged physics gate",
        "",
        f"Candidate lags {args.lags} and forcing quantiles {args.quantiles} were fixed before evaluation. Validation selected q{selected_q:.2f} and a causal lag of {selected_lag} h. Test targets were not used for selection.",
        "",
        "| Model | Sequence MSE | Lead-24 MSE |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {r['model']} | {r['seq_MSE_mean']:.8f} | {r['lead24_MSE_mean']:.8f} |" for _, r in summary.iterrows())
    lines.extend(["", f"Lagged gate minus persistence Lead-24 MSE reduction: `{delta:.8f}`.", "", "This is an ordinary benchmark mechanism candidate, not an independent chronological holdout. If the gain is small or inconsistent, it must remain exploratory."])
    (out / "LAGGED_GATE_REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grid["lag_hours"], grid["val_lead24_MSE"], marker="o")
    ax.axvline(selected_lag, color="#2F7D6D", linestyle="--", label=f"selected {selected_lag} h")
    ax.set_xlabel("Candidate forcing lag (h)")
    ax.set_ylabel("Validation Lead-24 MSE")
    ax.set_title("Validation-locked lag selection")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "lag_validation_selection.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-locked lagged physics gate")
    parser.add_argument("--output-dir", default="results/lagged_physics_gate_benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.75, 0.90, 0.95])
    parser.add_argument("--lags", nargs="+", type=int, default=[0, 6, 12, 24, 72, 168])
    parser.add_argument("--horizon", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
