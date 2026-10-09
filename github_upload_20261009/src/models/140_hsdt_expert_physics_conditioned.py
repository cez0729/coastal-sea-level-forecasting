from __future__ import annotations

"""Causal physics-conditioned HS-DT expert screen.

The locked strict-causal GWN experts are used as initialization.  A causal
ODE prior is injected into each expert output through a learned horizon gate,
and the GWN weights are fine-tuned jointly.  No future residual is used.
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


confirm = load_module("hsdt_confirm_140", HERE / "134_confirmatory_hsdt_orc_chronological_refit.py")
p104 = confirm.p104
rolling = confirm.rolling
v2 = confirm.v2


def causal_ode_prior(ode, init_states, phys_seq, adjacency):
    prev = init_states
    outputs = []
    decay = torch.nn.functional.softplus(ode.raw_decay).clamp(max=0.5).view(1, 1, -1)
    kappa = torch.nn.functional.softplus(ode.raw_kappa).clamp(max=0.5).view(1, 1, -1)
    beta = torch.tanh(ode.beta)
    bias = torch.tanh(ode.bias).T.unsqueeze(0)
    for lead in range(phys_seq.shape[2]):
        spatial = torch.einsum("ij,bjs->bis", adjacency, prev) - prev
        u, v, wave = prev[..., 1], prev[..., 2], prev[..., 3]
        graph_u = torch.einsum("ij,bj->bi", adjacency, u)
        graph_v = torch.einsum("ij,bj->bi", adjacency, v)
        transport = (graph_u - u) + (graph_v - v)
        pressure_gradient = v2.GRAVITY * spatial[..., 0]
        extra = torch.stack([u, v, wave, transport, pressure_gradient], dim=-1)
        forcing = torch.einsum("sk,bnk->bns", beta, torch.cat([extra, phys_seq[:, :, lead, :]], dim=-1))
        prev = prev + bias - decay * prev + kappa * spatial + forcing
        outputs.append(prev)
    return torch.stack(outputs, dim=2)


class PhysicsConditionedExpert(nn.Module):
    def __init__(self, base, ode, adjacency, horizon: int, num_states: int, initial_gate: float):
        super().__init__()
        self.base = base
        self.ode = ode
        self.register_buffer("adjacency", adjacency)
        self.horizon = int(horizon)
        self.num_states = int(num_states)
        init = min(max(float(initial_gate), 1e-4), 1.0 - 1e-4)
        self.gate_logits = nn.Parameter(torch.full((horizon, num_states), math.log(init / (1.0 - init))))

    def forward(self, x, init_states, phys_seq):
        base = self.base(x)
        prior = causal_ode_prior(self.ode, init_states, phys_seq, self.adjacency)
        if self.num_states == 1:
            base = base.unsqueeze(-1)
        prior = prior[..., : self.num_states]
        gate = torch.sigmoid(self.gate_logits).view(1, 1, self.horizon, self.num_states)
        final = base + gate * (prior - base)
        return final, base, prior


def norm_eta_loss(pred, target, scale):
    return torch.mean(((pred - target[..., 0]) / scale[..., 0]) ** 2)


def norm_multi_loss(pred, target, scale):
    return torch.mean(((pred - target) / scale) ** 2)


def load_base(data, args, seed, num_states, device):
    model = confirm.make_gwn(data, args, num_states).to(device)
    name = "gwn_eta_only" if num_states == 1 else "gwn_multistate_no_physics"
    root = ROOT / "results" / "confirmatory_hsdt_orc_refit_2025_h2" / f"seed_{seed}" / name
    payload = torch.load(root / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    return model


def score_model(model, loader, device, num_states, data, args):
    model.eval()
    preds, true, tide = [], [], []
    with torch.no_grad():
        for batch in loader:
            xb, target, td, init_states, phys_seq = batch
            if isinstance(model, PhysicsConditionedExpert):
                out = model(xb.to(device), init_states.to(device), phys_seq.to(device))[0]
            else:
                out = model(xb.to(device))
            preds.append((out[..., 0] if out.ndim == 4 else out).cpu().numpy())
            true.append(target[..., 0].numpy())
            tide.append(td.numpy())
    pred = np.concatenate(preds)
    target = np.concatenate(true)
    tides = np.concatenate(tide)
    train_end = rolling.time_index(data["arrays"]["time"], args.fold_train_end)
    threshold = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    return confirm.score(target, pred, tides, threshold)


def train_expert(kind, base, data, args, seed, device, stage):
    num_states = 1 if kind == "eta" else 4
    out = ROOT / args.output_dir / stage / f"seed_{seed}" / f"{kind}_causal_ode_expert"
    out.mkdir(parents=True, exist_ok=True)
    p104.set_reproducible(seed, args.cpu_threads)
    ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    adjacency = torch.tensor(data["graph_priors"][args.fixed_graph_type], dtype=torch.float32, device=device)
    model = PhysicsConditionedExpert(base, ode, adjacency, args.horizon, num_states, args.initial_gate).to(device)
    loader_train = p104.make_loader(data["multi_train"], args, True, seed)
    loader_val = p104.make_loader(data["multi_val"], args, False, seed)
    scale = torch.tensor(data["state_scale"].reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    physics_criterion = None
    if num_states == 4:
        physics_criterion = p104.make_multistate_criterion(
            model.ode, data, args, True, args.horizon
        ).to(device)
    best, best_val, bad, history = None, float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, target, _, init_states, phys_seq in loader_train:
            xb, target = xb.to(device), target.to(device)
            init_states, phys_seq = init_states.to(device), phys_seq.to(device)
            optimizer.zero_grad(set_to_none=True)
            final, _, prior = model(xb, init_states, phys_seq)
            data_loss = norm_eta_loss(final[..., 0], target, scale)
            prior_loss = norm_eta_loss(prior[..., 0], target, scale)
            smooth = torch.mean((torch.sigmoid(model.gate_logits)[1:] - torch.sigmoid(model.gate_logits)[:-1]) ** 2)
            ode_reg = torch.mean(model.ode.beta ** 2) + torch.mean(model.ode.bias ** 2)
            if num_states == 4:
                aux_loss = norm_multi_loss(final, target, scale)
                physics_value = physics_criterion.physics_loss(
                    final, init_states, phys_seq, model.adjacency
                )
                loss = data_loss + args.aux_weight * aux_loss + args.prior_loss_weight * prior_loss
                loss = loss + p104.physics_lambda(epoch, args) * physics_value
            else:
                loss = data_loss + args.prior_loss_weight * prior_loss
            loss = loss + args.gate_smooth_weight * smooth + args.ode_reg_weight * ode_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, target, _, init_states, phys_seq in loader_val:
                final = model(xb.to(device), init_states.to(device), phys_seq.to(device))[0]
                vals.append(float(norm_eta_loss(final[..., 0], target.to(device), scale).cpu()))
        val = float(np.mean(vals))
        scheduler.step(val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_eta_loss": val})
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"{kind}_causal_ode_expert epoch={epoch:03d} val={val:.6f}", flush=True)
        if val < best_val - args.min_delta:
            best_val, bad = val, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            break
    model.load_state_dict(best)
    row = {
        "seed": seed,
        "config": f"{kind}_causal_ode_expert",
        "evaluation_split": "2025_h1_validation_screen" if stage == "screen" else "2025_h2_backtest",
        "strict_causal_preprocessing": True,
        "future_residual_used_as_input": False,
        "joint_base_finetuning": True,
        "best_val_eta_loss": best_val,
        "training_seconds": time.perf_counter() - started,
        **score_model(model, p104.make_loader(data["multi_val"] if stage == "screen" else data["multi_test"], args, False, seed), device, num_states, data, args),
    }
    pd.DataFrame(history).to_csv(out / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(out / "metrics.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "metadata": row}, out / "best_checkpoint.pt")
    (out / "COMPLETE.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return model, row


def run_seed(seed, args, device, stage):
    data = rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    eta_base = load_base(data, args, seed, 1, device)
    multi_base = load_base(data, args, seed, 4, device)
    loader = p104.make_loader(data["multi_val"] if stage == "screen" else data["multi_test"], args, False, seed)
    rows = []
    eta_stats = score_model(eta_base, loader, device, 1, data, args)
    multi_stats = score_model(multi_base, loader, device, 4, data, args)
    for name, stats in [("eta_only_baseline", eta_stats), ("multistate_baseline", multi_stats)]:
        rows.append({"seed": seed, "config": name, "evaluation_split": "2025_h1_validation_screen" if stage == "screen" else "2025_h2_backtest", "strict_causal_preprocessing": True, "future_residual_used_as_input": False, **stats})
    enhanced = {}
    for kind, base, num_states in (("eta", eta_base, 1), ("multistate", multi_base, 4)):
        if args.recompute_only:
            ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
            adjacency = torch.tensor(
                data["graph_priors"][args.fixed_graph_type], dtype=torch.float32, device=device
            )
            enhanced[kind] = PhysicsConditionedExpert(
                copy.deepcopy(base), ode, adjacency, args.horizon, num_states, args.initial_gate
            ).to(device)
            checkpoint = (
                ROOT
                / args.output_dir
                / stage
                / f"seed_{seed}"
                / f"{kind}_causal_ode_expert"
                / "best_checkpoint.pt"
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            enhanced[kind].load_state_dict(payload["model_state_dict"])
            stats = score_model(enhanced[kind], loader, device, num_states, data, args)
            rows.append({
                "seed": seed,
                "config": f"{kind}_causal_ode_expert",
                "evaluation_split": "2025_h1_validation_screen" if stage == "screen" else "2025_h2_backtest",
                "strict_causal_preprocessing": True,
                "future_residual_used_as_input": False,
                "joint_base_finetuning": True,
                **stats,
            })
        else:
            enhanced[kind], row = train_expert(
                kind, copy.deepcopy(base), data, args, seed, device, stage
            )
            rows.append(row)
    with torch.no_grad():
        pred_eta = score_predictions(enhanced["eta"], data, args, seed, device, stage, 1)
        pred_multi = score_predictions(enhanced["multistate"], data, args, seed, device, stage, 4)
        pred_eta_base = score_predictions(eta_base, data, args, seed, device, stage, 1)
        pred_multi_base = score_predictions(multi_base, data, args, seed, device, stage, 4)
    weights = np.full(args.horizon, 0.5, dtype=np.float64); weights[-1] = 1.0
    true = pred_eta_base[1]; tide = pred_eta_base[2]
    combos = {
        "hsdt_baseline": pred_eta_base[0] + weights[None, None, :] * (pred_multi_base[0] - pred_eta_base[0]),
        "hsdt_eta_causal_ode_expert": pred_eta[0] + weights[None, None, :] * (pred_multi_base[0] - pred_eta[0]),
        "hsdt_multistate_causal_ode_expert": pred_eta_base[0] + weights[None, None, :] * (pred_multi[0] - pred_eta_base[0]),
        "hsdt_both_causal_ode_experts": pred_eta[0] + weights[None, None, :] * (pred_multi[0] - pred_eta[0]),
    }
    train_end = rolling.time_index(data["arrays"]["time"], args.fold_train_end)
    threshold = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    for name, pred in combos.items():
        rows.append({"seed": seed, "config": name, "evaluation_split": "2025_h1_validation_screen" if stage == "screen" else "2025_h2_backtest", "strict_causal_preprocessing": True, "future_residual_used_as_input": False, **confirm.score(true, pred, tide, threshold)})
    out = ROOT / args.output_dir / stage
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"seed_{seed}_all_metrics.csv", index=False)
    return rows


def score_predictions(model, data, args, seed, device, stage, num_states):
    model.eval()
    loader = p104.make_loader(data["multi_val"] if stage == "screen" else data["multi_test"], args, False, seed)
    pred, true, tide = [], [], []
    with torch.no_grad():
        for xb, target, td, init_states, phys_seq in loader:
            if isinstance(model, PhysicsConditionedExpert):
                out = model(xb.to(device), init_states.to(device), phys_seq.to(device))[0]
            else:
                out = model(xb.to(device))
            pred.append((out[..., 0] if out.ndim == 4 else out).cpu().numpy())
            true.append(target[..., 0].numpy()); tide.append(td.numpy())
    return np.concatenate(pred), np.concatenate(true), np.concatenate(tide)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen", "formal"], default="screen")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output-dir", default="results/hsdt_expert_physics_conditioned")
    parser.add_argument("--fold-train-end", default="2025-01-01")
    parser.add_argument("--fold-val-end", default="2025-07-01")
    parser.add_argument("--fold-test-end", default="2026-01-01")
    parser.add_argument("--horizon", type=int, default=24); parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=8); parser.add_argument("--fixed-graph-type", default="distance")
    parser.add_argument("--hidden-dim", type=int, default=64); parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--gwn-blocks", type=int, default=6); parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10); parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4); parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0); parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.20)
    parser.add_argument("--physics-lambda", type=float, default=0.0002); parser.add_argument("--physics-warmup-epochs", type=int, default=8)
    parser.add_argument("--physics-ramp-epochs", type=int, default=14); parser.add_argument("--physics-lr-mult", type=float, default=0.5)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5); parser.add_argument("--prior-loss-weight", type=float, default=0.05)
    parser.add_argument("--gate-smooth-weight", type=float, default=0.01); parser.add_argument("--ode-reg-weight", type=float, default=1e-5)
    parser.add_argument("--initial-gate", type=float, default=0.20); parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.95); parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--recompute-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, args, device, args.stage))
    out = ROOT / args.output_dir / args.stage
    all_runs = pd.DataFrame(rows); all_runs.to_csv(out / "all_runs.csv", index=False)
    summary = all_runs.groupby("config").agg({"seq_residual_R2": ["mean", "std"], "last_residual_R2": ["mean", "std"], "extreme_abs_q95_residual_R2": ["mean", "std"]})
    summary.to_csv(out / "summary.csv")
    (out / "README_CN.md").write_text("严格因果物理条件专家筛选：物理先验由 t-1 状态和 last-input forcing 递推，不使用未来 residual。正式解释必须称 2025 H2 chronological refit backtest。", encoding="utf-8")


if __name__ == "__main__":
    main()
