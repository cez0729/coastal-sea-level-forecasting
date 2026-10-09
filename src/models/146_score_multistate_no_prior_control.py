from __future__ import annotations

"""Score saved multistate no-prior fine-tuning checkpoints on 2025 H2."""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


confirm = load_module("confirm_146", HERE / "134_confirmatory_hsdt_orc_chronological_refit.py")
hsdt = load_module("hsdt_146", HERE / "140_hsdt_expert_physics_conditioned.py")
p104 = confirm.p104
rolling = confirm.rolling


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/multistate_no_prior_control_scored_2025_h2")
    parser.add_argument("--data-dir", default="data/processed_multiyear_2023_2025")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--fold-train-end", default="2025-01-01")
    parser.add_argument("--fold-val-end", default="2025-07-01")
    parser.add_argument("--fold-test-end", default="2026-01-01")
    parser.add_argument("--horizon", type=int, default=24); parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=8); parser.add_argument("--fixed-graph-type", default="distance")
    parser.add_argument("--hidden-dim", type=int, default=64); parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--gwn-blocks", type=int, default=6); parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--extreme-quantile", type=float, default=0.90); parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        data = rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
        model = confirm.make_gwn(data, args, 4).to(device)
        checkpoint = (
            ROOT / "results" / "hsdt_no_prior_finetune_control" / "formal"
            / f"seed_{seed}" / "multistate_no_prior_finetune" / "best_checkpoint.pt"
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=False)
        loader = p104.make_loader(data["multi_test"], args, False, seed)
        metrics = hsdt.score_model(model, loader, device, 4, data, args)
        rows.append({
            "seed": seed,
            "config": "multistate_no_prior_finetune",
            "evaluation_split": "2025_h2_backtest",
            "strict_causal_preprocessing": True,
            "future_residual_used_as_input": False,
            **metrics,
        })
        print(f"seed={seed} seq={metrics['seq_residual_R2']:.6f} lead24={metrics['last_residual_R2']:.6f}", flush=True)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "all_runs.csv", index=False)
    metrics = ["seq_residual_R2", "last_residual_R2", "extreme_abs_q95_residual_R2", "event_PR_AUC"]
    frame.groupby("config")[metrics].agg(["mean", "std", "count"]).to_csv(out / "mean_std.csv")


if __name__ == "__main__":
    main()
