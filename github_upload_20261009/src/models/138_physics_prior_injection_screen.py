from __future__ import annotations

"""Screen direct physics-prior injection for the locked GNN and GWN backbones.

This is a validation-only exploration.  The backbone is loaded from the locked
formal baseline checkpoint and frozen; only the ODE prior and a small adapter
are trained on the existing train/2025-H1 validation split.  No 2025-H2 data
is read by this script.
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


exp137 = load_module("physics_injection_exp137", HERE / "137_transport_aware_physics_graph_experiment.py")
v2 = exp137.v2
p104 = exp137.p104
rolling = exp137.rolling
confirm = exp137.confirm

CONFIGS = [
    "gwn_ode_prior_gate",
    "gwn_ode_prior_blend",
    "bigru_ode_prior_gate",
    "bigru_ode_prior_blend",
]


def integrate_ode_prior(
    ode: nn.Module,
    init_states: torch.Tensor,
    phys_seq: torch.Tensor,
    adj: torch.Tensor,
) -> torch.Tensor:
    """Causal Euler rollout using only the last observed state and future forcings."""
    prev = init_states
    outputs = []
    decay = torch.nn.functional.softplus(ode.raw_decay).clamp(max=0.5).view(1, 1, -1)
    kappa = torch.nn.functional.softplus(ode.raw_kappa).clamp(max=0.5).view(1, 1, -1)
    beta = torch.tanh(ode.beta)
    bias = torch.tanh(ode.bias).T.unsqueeze(0)
    for lead in range(phys_seq.shape[2]):
        spatial = torch.einsum("ij,bjs->bis", adj, prev) - prev
        eta, u, v, wave = prev[..., 0], prev[..., 1], prev[..., 2], prev[..., 3]
        graph_u = torch.einsum("ij,bj->bi", adj, u)
        graph_v = torch.einsum("ij,bj->bi", adj, v)
        transport = (graph_u - u) + (graph_v - v)
        pressure_gradient = v2.GRAVITY * spatial[..., 0]
        extra = torch.stack([u, v, wave, transport, pressure_gradient], dim=-1)
        forcing_input = torch.cat([extra, phys_seq[:, :, lead, :]], dim=-1)
        forcing = torch.einsum("sk,bnk->bns", beta, forcing_input)
        rhs = bias - decay * prev + kappa * spatial + forcing
        prev = prev + rhs
        outputs.append(prev)
    return torch.stack(outputs, dim=2)


class PriorRefiner(nn.Module):
    """Small zero-initialized residual adapter conditioned on the ODE prior."""

    def __init__(self, horizon: int, states: int, hidden: int):
        super().__init__()
        input_dim = 2 * horizon * states
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon * states),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.horizon = horizon
        self.states = states

    def forward(self, base: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        bsz, nodes, horizon, states = base.shape
        features = torch.cat([base, prior], dim=-1).reshape(bsz * nodes, -1)
        return self.net(features).reshape(bsz, nodes, horizon, states)


class PhysicsPriorAdapter(nn.Module):
    def __init__(self, base: nn.Module, ode: nn.Module, adjacency: torch.Tensor, args, mode: str):
        super().__init__()
        self.base = base
        self.ode = ode
        self.register_buffer("adjacency", adjacency)
        self.horizon = int(args.horizon)
        self.mode = mode
        self.gate_logits = nn.Parameter(torch.full((self.horizon,), self._logit(args.initial_gate)))
        if mode == "gate":
            self.refiner = PriorRefiner(self.horizon, len(v2.STATE_NAMES), args.refiner_hidden)
        else:
            self.refiner = None
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    @staticmethod
    def _logit(value: float) -> float:
        value = min(max(float(value), 1e-4), 1.0 - 1e-4)
        return math.log(value / (1.0 - value))

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x, init_states, phys_seq):
        with torch.no_grad():
            base = self.base(x)
        prior = integrate_ode_prior(self.ode, init_states, phys_seq, self.adjacency)
        gate = torch.sigmoid(self.gate_logits).view(1, 1, self.horizon, 1)
        if self.mode == "gate":
            final = base + gate * self.refiner(base, prior)
        elif self.mode == "blend":
            final = base + gate * (prior - base)
        else:
            raise ValueError(self.mode)
        return final, base, prior, gate


def normalized_loss(pred, target, state_scale, aux_weight: float):
    scaled = (pred - target) / state_scale
    return torch.mean(scaled[..., 0] ** 2) + aux_weight * torch.mean(scaled[..., 1:] ** 2)


def load_base(config: str, data: dict, args, device: torch.device) -> nn.Module:
    backbone = config.split("_", 1)[0]
    base, _ = exp137.make_model(f"{backbone}_baseline", data, args, device)
    source_root = ROOT / "results" / "transport_aware_dynamic_graph_physics" / "formal" / f"seed_{args.seed}"
    checkpoint = source_root / ("gwn_baseline" if backbone == "gwn" else "bigru_baseline") / "best_checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing locked baseline checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = base.load_state_dict(payload["model_state_dict"], strict=False)
    allowed_missing = {"fixed_adjacency"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"Unexpected baseline checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return base.to(device)


def score_validation(model, loader, data, device):
    model.eval()
    pred, true, tide = [], [], []
    with torch.no_grad():
        for xb, target, tide_batch, init_states, phys_seq in loader:
            final, _, _, _ = model(xb.to(device), init_states.to(device), phys_seq.to(device))
            pred.append(final[..., 0].cpu().numpy())
            true.append(target[..., 0].numpy())
            tide.append(tide_batch.numpy())
    pred = np.concatenate(pred, axis=0)
    true = np.concatenate(true, axis=0)
    tide = np.concatenate(tide, axis=0)
    train_end = rolling.time_index(data["arrays"]["time"], "2025-01-01")
    thresholds = np.quantile(data["arrays"]["residual"][:train_end], 0.95, axis=0)
    return confirm.score(true, pred, tide, thresholds)


def train_one(config: str, data: dict, args, device: torch.device):
    run_dir = ROOT / args.output_dir / "screen" / f"seed_{args.seed}" / config
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists() and (run_dir / "COMPLETE.json").exists() and args.resume:
        return pd.read_csv(metrics_path).iloc[0].to_dict()
    p104.set_reproducible(args.seed, args.cpu_threads)
    base = load_base(config, data, args, device)
    ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), len(v2.STATE_NAMES)).to(device)
    adjacency = torch.tensor(data["graph_priors"][args.fixed_graph_type], dtype=torch.float32, device=device)
    mode = "gate" if config.endswith("gate") else "blend"
    model = PhysicsPriorAdapter(base, ode, adjacency, args, mode).to(device)
    train_loader = p104.make_loader(data["multi_train"], args, True, args.seed)
    val_loader = p104.make_loader(data["multi_val"], args, False, args.seed)
    scale = torch.tensor(data["state_scale"].reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    trainable = list(model.ode.parameters()) + [model.gate_logits]
    if model.refiner is not None:
        trainable += list(model.refiner.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    bad = 0
    rows = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = []
        for xb, target, _, init_states, phys_seq in train_loader:
            optimizer.zero_grad(set_to_none=True)
            final, _, prior, gate = model(xb.to(device), init_states.to(device), phys_seq.to(device))
            final_loss = normalized_loss(final, target.to(device), scale, args.aux_weight)
            prior_loss = normalized_loss(prior, target.to(device), scale, args.aux_weight)
            smooth = torch.mean((gate[:, :, 1:, :] - gate[:, :, :-1, :]) ** 2)
            ode_reg = torch.mean(ode.beta ** 2) + torch.mean(ode.bias ** 2)
            loss = final_loss + args.prior_loss_weight * prior_loss + args.gate_smooth_weight * smooth + args.ode_reg_weight * ode_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            total.append([float(loss.detach().cpu()), float(final_loss.detach().cpu()), float(prior_loss.detach().cpu())])
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, target, _, init_states, phys_seq in val_loader:
                final, _, _, _ = model(xb.to(device), init_states.to(device), phys_seq.to(device))
                val_losses.append(float(normalized_loss(final, target.to(device), scale, 0.0).cpu()))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_loss": float(np.mean(np.asarray(total)[:, 0])), "val_eta_loss": val_loss, "prior_loss": float(np.mean(np.asarray(total)[:, 2])), "gate_mean": float(model.gate_logits.sigmoid().mean().detach().cpu()), "gate_24h": float(model.gate_logits.sigmoid()[-1].detach().cpu())}
        rows.append(row)
        if val_loss < best_val - args.min_delta:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"{config} epoch={epoch:03d} val={val_loss:.6f} gate24={row['gate_24h']:.3f}", flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    metrics = score_validation(model, val_loader, data, device)
    row = {
        "seed": args.seed,
        "config": config,
        "evaluation_split": "2025_h1_validation_screen",
        "strict_causal_preprocessing": True,
        "untouched_holdout": False,
        "future_residual_used_as_input": False,
        "physics_injection": "causal_ode_prior_gated_residual" if mode == "gate" else "causal_ode_prior_horizon_blend",
        "base_frozen": True,
        "best_val_eta_loss": best_val,
        "training_seconds": time.perf_counter() - started,
        **metrics,
    }
    pd.DataFrame(rows).to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    torch.save({"model_state_dict": model.state_dict(), "metadata": row}, run_dir / "best_checkpoint.pt")
    (run_dir / "COMPLETE.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/physics_prior_injection_screen")
    parser.add_argument("--data-dir", default="data/processed_multiyear_2023_2025")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=CONFIGS)
    parser.add_argument("--fold-train-end", default="2025-01-01")
    parser.add_argument("--fold-val-end", default="2025-07-01")
    parser.add_argument("--fold-test-end", default="2026-01-01")
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", choices=["identity", "distance", "corr"], default="distance")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--gwn-blocks", type=int, default=6)
    parser.add_argument("--gnn-hidden", type=int, default=40)
    parser.add_argument("--gru-hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--bigru-dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--initial-gate", type=float, default=0.08)
    parser.add_argument("--refiner-hidden", type=int, default=64)
    parser.add_argument("--prior-loss-weight", type=float, default=0.05)
    parser.add_argument("--gate-smooth-weight", type=float, default=0.01)
    parser.add_argument("--ode-reg-weight", type=float, default=1e-5)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = ["" if value is None or (isinstance(value, float) and not np.isfinite(value)) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    root_out = ROOT / args.output_dir / "screen"
    root_out.mkdir(parents=True, exist_ok=True)
    data_dir = exp137.confirm.configure_data_dir(args.data_dir)
    data = rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}; configs={args.configs}; evaluation=2025_h1_validation_screen", flush=True)
    rows = []
    for config in args.configs:
        rows.append(train_one(config, data, args, device))
    # Rebuild the combined screen table so separate backbone runs do not
    # overwrite each other when executed in different processes.
    frames = []
    for metrics_path in sorted((root_out / f"seed_{args.seed}").glob("*/metrics.csv")):
        frames.append(pd.read_csv(metrics_path))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(rows)
    result.to_csv(root_out / "all_runs.csv", index=False)
    summary = result[["config", "seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2", "gate_24h"] if "gate_24h" in result.columns else ["config", "seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2"]]
    summary.to_csv(root_out / "summary.csv", index=False)
    write_markdown_table(summary, root_out / "SUMMARY.md")


if __name__ == "__main__":
    main()
