from __future__ import annotations

"""Formal single-multistate GWN ODE/physics factorial on the H2 backtest.

This experiment starts from the same strict-causal multistate GWN checkpoints
used by the HS-DT refit and separates three effects: a frozen ODE prior,
joint ODE/GWN fine-tuning, and the additional differentiable physics loss.
"""

import argparse
import copy
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
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


confirm = load_module("confirm_144", HERE / "134_confirmatory_hsdt_orc_chronological_refit.py")
p104 = confirm.p104
rolling = confirm.rolling
v2 = confirm.v2
hsdt = load_module("hsdt_144", HERE / "140_hsdt_expert_physics_conditioned.py")


def load_base(data, args, seed: int, device: torch.device):
    model = confirm.make_gwn(data, args, 4).to(device)
    root = ROOT / "results" / "confirmatory_hsdt_orc_refit_2025_h2" / f"seed_{seed}" / "gwn_multistate_no_physics"
    payload = torch.load(root / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    return model


def train_one(base, data, args, seed: int, device: torch.device, mode: str):
    joint = mode in {"ode_joint", "ode_joint_physics"}
    use_physics = mode == "ode_joint_physics"
    out = ROOT / args.output_dir / f"seed_{seed}" / mode
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.csv"
    if metrics_path.exists() and (out / "COMPLETE.json").exists() and args.resume:
        return pd.read_csv(metrics_path).iloc[0].to_dict()

    p104.set_reproducible(seed, args.cpu_threads)
    ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    adjacency = torch.tensor(data["graph_priors"][args.fixed_graph_type], dtype=torch.float32, device=device)
    model = hsdt.PhysicsConditionedExpert(
        copy.deepcopy(base), ode, adjacency, args.horizon, 4, args.initial_gate
    ).to(device)
    model.base.requires_grad_(joint)
    train_loader = p104.make_loader(data["multi_train"], args, True, seed)
    val_loader = p104.make_loader(data["multi_val"], args, False, seed)
    test_loader = p104.make_loader(data["multi_test"], args, False, seed)
    scale = torch.tensor(data["state_scale"].reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    criterion = p104.make_multistate_criterion(model.ode, data, args, True, args.horizon)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    best_state, best_val, bad, history = None, float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, target, _, init_states, phys_seq in train_loader:
            xb, target = xb.to(device), target.to(device)
            init_states, phys_seq = init_states.to(device), phys_seq.to(device)
            optimizer.zero_grad(set_to_none=True)
            final, _, prior = model(xb, init_states, phys_seq)
            data_loss = torch.mean(((final - target) / scale) ** 2)
            prior_loss = torch.mean(((prior - target) / scale) ** 2)
            gate = torch.sigmoid(model.gate_logits)
            smooth = torch.mean((gate[1:] - gate[:-1]) ** 2)
            ode_reg = torch.mean(model.ode.beta ** 2) + torch.mean(model.ode.bias ** 2)
            loss = data_loss + args.prior_loss_weight * prior_loss
            if use_physics:
                loss = loss + p104.physics_lambda(epoch, args) * criterion.physics_loss(
                    final, init_states, phys_seq, model.adjacency
                )
            loss = loss + args.gate_smooth_weight * smooth + args.ode_reg_weight * ode_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, target, _, init_states, phys_seq in val_loader:
                final = model(xb.to(device), init_states.to(device), phys_seq.to(device))[0]
                vals.append(float(torch.mean(((final - target.to(device)) / scale) ** 2).cpu()))
        val = float(np.mean(vals))
        scheduler.step(val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val})
        if val < best_val - args.min_delta:
            best_val, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"seed={seed} mode={mode} epoch={epoch:03d} val={val:.6f}", flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    metrics = hsdt.score_model(model, test_loader, device, 4, data, args)
    row = {
        "seed": seed,
        "config": mode,
        "evaluation_split": "2025_h2_backtest",
        "strict_causal_preprocessing": True,
        "future_residual_used_as_input": False,
        "base_frozen": not joint,
        "physics_loss_active": use_physics,
        "physics_lambda": args.physics_lambda if use_physics else 0.0,
        "best_val_loss": best_val,
        "training_seconds": time.perf_counter() - started,
        **metrics,
    }
    pd.DataFrame(history).to_csv(out / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    torch.save({"model_state_dict": model.state_dict(), "metadata": row}, out / "best_checkpoint.pt")
    (out / "COMPLETE.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def train_no_prior(base, data, args, seed: int, device: torch.device):
    mode = "no_prior_joint"
    out = ROOT / args.output_dir / f"seed_{seed}" / mode
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.csv"
    if metrics_path.exists() and (out / "COMPLETE.json").exists() and args.resume:
        return pd.read_csv(metrics_path).iloc[0].to_dict()
    p104.set_reproducible(seed, args.cpu_threads)
    model = copy.deepcopy(base).to(device)
    train_loader = p104.make_loader(data["multi_train"], args, True, seed)
    val_loader = p104.make_loader(data["multi_val"], args, False, seed)
    test_loader = p104.make_loader(data["multi_test"], args, False, seed)
    scale = torch.tensor(data["state_scale"].reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    best_state, best_val, bad, history = None, float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, target, _, _, _ in train_loader:
            xb, target = xb.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.mean(((pred - target) / scale) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, target, _, _, _ in val_loader:
                pred = model(xb.to(device))
                vals.append(float(torch.mean(((pred - target.to(device)) / scale) ** 2).cpu()))
        val = float(np.mean(vals))
        scheduler.step(val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val})
        if val < best_val - args.min_delta:
            best_val, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"seed={seed} mode={mode} epoch={epoch:03d} val={val:.6f}", flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    metrics = hsdt.score_model(model, test_loader, device, 4, data, args)
    row = {
        "seed": seed,
        "config": mode,
        "evaluation_split": "2025_h2_backtest",
        "strict_causal_preprocessing": True,
        "future_residual_used_as_input": False,
        "base_frozen": False,
        "physics_loss_active": False,
        "physics_lambda": 0.0,
        "best_val_loss": best_val,
        "training_seconds": time.perf_counter() - started,
        **metrics,
    }
    pd.DataFrame(history).to_csv(out / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    torch.save({"model_state_dict": model.state_dict(), "metadata": row}, out / "best_checkpoint.pt")
    (out / "COMPLETE.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def run_seed(seed: int, args, device: torch.device):
    data = rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    base = load_base(data, args, seed, device)
    loader = p104.make_loader(data["multi_test"], args, False, seed)
    base_metrics = hsdt.score_model(base, loader, device, 4, data, args)
    rows = [{
        "seed": seed,
        "config": "baseline",
        "evaluation_split": "2025_h2_backtest",
        "strict_causal_preprocessing": True,
        "future_residual_used_as_input": False,
        "base_frozen": True,
        "physics_loss_active": False,
        "physics_lambda": 0.0,
        **base_metrics,
    }]
    if "no_prior_joint" in args.modes:
        rows.append(train_no_prior(base, data, args, seed, device))
    for mode in ("ode_frozen", "ode_joint", "ode_joint_physics"):
        if mode in args.modes:
            rows.append(train_one(base, data, args, seed, device, mode))
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/formal_gwn_ode_physics_factorial_2025_h2")
    parser.add_argument("--data-dir", default="data/processed_multiyear_2023_2025")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["no_prior_joint", "ode_frozen", "ode_joint", "ode_joint_physics"],
        default=["no_prior_joint", "ode_frozen", "ode_joint", "ode_joint_physics"],
    )
    parser.add_argument("--fold-train-end", default="2025-01-01")
    parser.add_argument("--fold-val-end", default="2025-07-01")
    parser.add_argument("--fold-test-end", default="2026-01-01")
    parser.add_argument("--horizon", type=int, default=24); parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=8); parser.add_argument("--fixed-graph-type", default="distance")
    parser.add_argument("--hidden-dim", type=int, default=64); parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--gwn-blocks", type=int, default=6); parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10); parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0); parser.add_argument("--initial-gate", type=float, default=0.20)
    parser.add_argument("--prior-loss-weight", type=float, default=0.05); parser.add_argument("--gate-smooth-weight", type=float, default=0.01)
    parser.add_argument("--ode-reg-weight", type=float, default=1e-5); parser.add_argument("--physics-lambda", type=float, default=0.0002)
    parser.add_argument("--physics-warmup-epochs", type=int, default=0); parser.add_argument("--physics-ramp-epochs", type=int, default=4)
    parser.add_argument("--physics-lr-mult", type=float, default=0.5); parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.20); parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument("--extreme-quantile", type=float, default=0.90); parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    parser.add_argument("--print-every", type=int, default=5); parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / args.output_dir; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, args, device))
    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(out / "all_runs.csv", index=False)
    metrics = ["seq_residual_R2", "last_residual_R2", "extreme_abs_q95_residual_R2", "event_PR_AUC"]
    summary = all_runs.groupby("config")[metrics].agg(["mean", "std", "count"])
    summary.to_csv(out / "mean_std.csv")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
