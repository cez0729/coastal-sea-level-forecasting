from __future__ import annotations

"""Screen physics-prior injection into the frozen HS-DT-GWN experts.

The eta-only and multistate GWN experts are loaded from the locked formal
chronological-refit checkpoints.  Only the causal ODE prior and horizon gates
are trained on the train/2025-H1 validation split; 2025-H2 is never read.
"""

import argparse
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


confirm = load_module("hsdt_confirm", HERE / "134_confirmatory_hsdt_orc_chronological_refit.py")
p104 = confirm.p104
rolling = confirm.rolling
v2 = confirm.v2

CONFIGS = [
    "hsdt_baseline",
    "hsdt_multistate_ode_blend",
    "hsdt_dual_expert_ode_blend",
    "hsdt_post_fusion_ode_blend",
    "hsdt_post_fusion_persistence_control",
]


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
        forcing = torch.einsum(
            "sk,bnk->bns", beta, torch.cat([extra, phys_seq[:, :, lead, :]], dim=-1)
        )
        prev = prev + bias - decay * prev + kappa * spatial + forcing
        outputs.append(prev)
    return torch.stack(outputs, dim=2)


class HSDTPhysicsAdapter(nn.Module):
    def __init__(self, eta_model, multi_model, ode, adjacency, args, mode: str):
        super().__init__()
        self.eta_model = eta_model
        self.multi_model = multi_model
        self.ode = ode
        self.register_buffer("adjacency", adjacency)
        self.horizon = int(args.horizon)
        self.mode = mode
        self.weights = torch.tensor([0.5] * (self.horizon - 1) + [1.0], dtype=torch.float32)
        if mode == "dual":
            self.gate_eta_logits = nn.Parameter(torch.full((self.horizon,), self._logit(args.initial_gate)))
            self.gate_multi_logits = nn.Parameter(torch.full((self.horizon,), self._logit(args.initial_gate)))
        else:
            self.gate_logits = nn.Parameter(torch.full((self.horizon,), self._logit(args.initial_gate)))
        for model in (self.eta_model, self.multi_model):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.eval()

    @staticmethod
    def _logit(value: float) -> float:
        value = min(max(float(value), 1e-4), 1.0 - 1e-4)
        return math.log(value / (1.0 - value))

    def train(self, mode: bool = True):
        super().train(mode)
        self.eta_model.eval()
        self.multi_model.eval()
        return self

    def forward(self, x, init_states, phys_seq):
        with torch.no_grad():
            eta = self.eta_model(x)
            multi = self.multi_model(x)
        hsdt = eta + self.weights.view(1, 1, self.horizon) * (multi[..., 0] - eta)
        if self.mode == "baseline":
            return hsdt, eta, multi, None
        if self.mode == "persistence":
            prior = init_states.unsqueeze(2).expand(-1, -1, self.horizon, -1)
            gate = torch.sigmoid(self.gate_logits).view(1, 1, self.horizon)
            final = hsdt + gate * (prior[..., 0] - hsdt)
            return final, eta, multi, prior
        prior = causal_ode_prior(self.ode, init_states, phys_seq, self.adjacency)
        if self.mode == "multistate":
            gate = torch.sigmoid(self.gate_logits).view(1, 1, self.horizon, 1)
            enhanced_multi = multi + gate * (prior - multi)
            final = eta + self.weights.view(1, 1, self.horizon) * (enhanced_multi[..., 0] - eta)
        elif self.mode == "dual":
            gate_eta = torch.sigmoid(self.gate_eta_logits).view(1, 1, self.horizon)
            gate_multi = torch.sigmoid(self.gate_multi_logits).view(1, 1, self.horizon, 1)
            enhanced_eta = eta + gate_eta * (prior[..., 0] - eta)
            enhanced_multi = multi + gate_multi * (prior - multi)
            final = enhanced_eta + self.weights.view(1, 1, self.horizon) * (enhanced_multi[..., 0] - enhanced_eta)
        elif self.mode == "post":
            gate = torch.sigmoid(self.gate_logits).view(1, 1, self.horizon)
            final = hsdt + gate * (prior[..., 0] - hsdt)
        else:
            raise ValueError(self.mode)
        return final, eta, multi, prior


def normalized_eta_loss(pred, target, scale):
    return torch.mean(((pred - target[..., 0]) / scale[..., 0]) ** 2)


def normalized_multi_loss(pred, target, scale):
    return torch.mean(((pred - target) / scale) ** 2)


def load_experts(data, args, device):
    eta_model = confirm.make_gwn(data, args, 1).to(device)
    multi_model = confirm.make_gwn(data, args, 4).to(device)
    root = ROOT / "results" / "confirmatory_hsdt_orc_refit_2025_h2" / f"seed_{args.seed}"
    eta_payload = torch.load(root / "gwn_eta_only" / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    multi_payload = torch.load(root / "gwn_multistate_no_physics" / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    eta_model.load_state_dict(eta_payload["model_state_dict"], strict=False)
    multi_model.load_state_dict(multi_payload["model_state_dict"], strict=False)
    return eta_model, multi_model


def make_adapter(config, data, args, device):
    eta_model, multi_model = load_experts(data, args, device)
    adjacency = torch.tensor(data["graph_priors"][args.fixed_graph_type], dtype=torch.float32, device=device)
    ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), len(v2.STATE_NAMES)).to(device)
    mode = {"hsdt_multistate_ode_blend": "multistate", "hsdt_dual_expert_ode_blend": "dual", "hsdt_post_fusion_ode_blend": "post", "hsdt_post_fusion_persistence_control": "persistence", "hsdt_baseline": "baseline"}[config]
    return HSDTPhysicsAdapter(eta_model, multi_model, ode, adjacency, args, mode)


def score(model, loader, data, args, device):
    model.eval()
    preds, trues, tides = [], [], []
    with torch.no_grad():
        for xb, target, tide, init_states, phys_seq in loader:
            final, _, _, _ = model(xb.to(device), init_states.to(device), phys_seq.to(device))
            preds.append(final.cpu().numpy())
            trues.append(target[..., 0].numpy())
            tides.append(tide.numpy())
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    tide = np.concatenate(tides, axis=0)
    train_end = rolling.time_index(data["arrays"]["time"], args.fold_train_end)
    threshold = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    return confirm.score(true, pred, tide, threshold)


def train_one(config, data, args, device):
    out = ROOT / args.output_dir / args.stage / f"seed_{args.seed}" / config
    out.mkdir(parents=True, exist_ok=True)
    if args.resume and (out / "COMPLETE.json").exists():
        return pd.read_csv(out / "metrics.csv").iloc[0].to_dict()
    p104.set_reproducible(args.seed, args.cpu_threads)
    model = make_adapter(config, data, args, device)
    loader_train = p104.make_loader(data["multi_train"], args, True, args.seed)
    loader_val = p104.make_loader(data["multi_val"], args, False, args.seed)
    scale = torch.tensor(data["state_scale"].reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    trainable = []
    if model.mode != "baseline":
        trainable += list(model.ode.parameters())
        if model.mode == "dual":
            trainable += [model.gate_eta_logits, model.gate_multi_logits]
        else:
            trainable += [model.gate_logits]
    if not trainable:
        eval_loader = loader_val if args.stage == "screen" else p104.make_loader(data["multi_test"], args, False, args.seed)
        row = {"seed": args.seed, "config": config, "evaluation_split": "2025_h1_validation_screen" if args.stage == "screen" else "2025_h2_backtest", "strict_causal_preprocessing": True, "future_residual_used_as_input": False, **score(model, eval_loader, data, args, device)}
        pd.DataFrame([row]).to_csv(out / "metrics.csv", index=False)
        (out / "COMPLETE.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        return row
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val, bad, history = float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, target, _, init_states, phys_seq in loader_train:
            optimizer.zero_grad(set_to_none=True)
            final, eta, multi, prior = model(xb.to(device), init_states.to(device), phys_seq.to(device))
            final_loss = normalized_eta_loss(final, target.to(device), scale)
            prior_loss = torch.zeros((), device=device)
            multi_loss = torch.zeros((), device=device)
            if prior is not None:
                prior_loss = normalized_eta_loss(prior[..., 0], target.to(device), scale)
                multi_loss = normalized_multi_loss(prior, target.to(device), scale)
            smooth = torch.zeros((), device=device)
            if model.mode == "dual":
                gates = [torch.sigmoid(model.gate_eta_logits), torch.sigmoid(model.gate_multi_logits)]
            elif model.mode != "baseline":
                gates = [torch.sigmoid(model.gate_logits)]
            else:
                gates = []
            for gate in gates:
                smooth = smooth + torch.mean((gate[1:] - gate[:-1]) ** 2)
            ode_reg = torch.mean(model.ode.beta ** 2) + torch.mean(model.ode.bias ** 2)
            loss = final_loss + args.prior_loss_weight * prior_loss + args.multi_aux_weight * multi_loss + args.gate_smooth_weight * smooth + args.ode_reg_weight * ode_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, target, _, init_states, phys_seq in loader_val:
                final, _, _, _ = model(xb.to(device), init_states.to(device), phys_seq.to(device))
                val_losses.append(float(normalized_eta_loss(final, target.to(device), scale).cpu()))
        val = float(np.mean(val_losses))
        scheduler.step(val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_eta_loss": val})
        if val < best_val - args.min_delta:
            best_val, bad = val, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"{config} epoch={epoch:03d} val={val:.6f}", flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best)
    eval_loader = loader_val if args.stage == "screen" else p104.make_loader(data["multi_test"], args, False, args.seed)
    row = {"seed": args.seed, "config": config, "evaluation_split": "2025_h1_validation_screen" if args.stage == "screen" else "2025_h2_backtest", "strict_causal_preprocessing": True, "future_residual_used_as_input": False, "base_experts_frozen": True, "training_seconds": time.perf_counter() - started, "best_val_eta_loss": best_val, **score(model, eval_loader, data, args, device)}
    pd.DataFrame(history).to_csv(out / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(out / "metrics.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "metadata": row}, out / "best_checkpoint.pt")
    (out / "COMPLETE.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/hsdt_physics_prior_screen")
    parser.add_argument("--stage", choices=["screen", "formal"], default="screen")
    parser.add_argument("--data-dir", default="data/processed_multiyear_2023_2025")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=CONFIGS)
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
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--initial-gate", type=float, default=0.08)
    parser.add_argument("--prior-loss-weight", type=float, default=0.05)
    parser.add_argument("--multi-aux-weight", type=float, default=0.08)
    parser.add_argument("--gate-smooth-weight", type=float, default=0.01)
    parser.add_argument("--ode-reg-weight", type=float, default=1e-5)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def write_table(frame, path):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    confirm.configure_data_dir(args.data_dir)
    data = rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = ROOT / args.output_dir / args.stage
    root.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    print(f"Device={device}; seeds={seeds}; configs={args.configs}; stage={args.stage}", flush=True)
    for seed in seeds:
        args.seed = seed
        for config in args.configs:
            train_one(config, data, args, device)
    frames = [pd.read_csv(p) for p in sorted(root.glob("seed_*/*/metrics.csv"))]
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(root / "all_runs.csv", index=False)
    summary = result[["config", "seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2"]]
    summary.to_csv(root / "summary.csv", index=False)
    write_table(summary, root / "SUMMARY.md")


if __name__ == "__main__":
    main()
