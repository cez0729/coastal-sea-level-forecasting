from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "ode_prior_gated_multistate_gwn"
SOURCE_RESULTS = ROOT / "results" / "priority12_physics_graph_wavenet"
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p104 = load_module("priority2_ode_prior", HERE / "104_priority2_physics_graph_wavenet.py")
final4 = p104.final4
priority1 = p104.priority1
v2 = p104.v2
v3 = p104.v3


def set_seed(seed: int, cpu_threads: int) -> None:
    v2.set_seed(seed)
    torch.set_num_threads(max(1, int(cpu_threads)))
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(dataset, args, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def integrate_ode_prior(
    ode: nn.Module,
    init_states: torch.Tensor,
    phys_seq: torch.Tensor,
    adj: torch.Tensor,
    step_size: float = 1.0,
) -> torch.Tensor:
    """Causal Euler rollout of the existing learnable multistate ODE."""
    # init_states [B,N,S], phys_seq [B,N,H,F], output [B,N,H,S].
    prev = init_states
    outputs = []
    decay = torch.nn.functional.softplus(ode.raw_decay).clamp(max=0.5).view(1, 1, -1)
    kappa = torch.nn.functional.softplus(ode.raw_kappa).clamp(max=0.5).view(1, 1, -1)
    beta = torch.tanh(ode.beta)
    bias = torch.tanh(ode.bias).T.unsqueeze(0)
    for lead in range(phys_seq.shape[2]):
        spatial = torch.einsum("ij,bjs->bis", adj, prev) - prev
        eta = prev[..., 0]
        u = prev[..., 1]
        v = prev[..., 2]
        wave_setup = prev[..., 3]
        graph_u = torch.einsum("ij,bj->bi", adj, u)
        graph_v = torch.einsum("ij,bj->bi", adj, v)
        transport = (graph_u - u) + (graph_v - v)
        pressure_gradient = v2.GRAVITY * spatial[..., 0]
        extra = torch.stack([u, v, wave_setup, transport, pressure_gradient], dim=-1)
        forcing_input = torch.cat([extra, phys_seq[:, :, lead, :]], dim=-1)
        forcing = torch.einsum("sk,bnk->bns", beta, forcing_input)
        rhs = bias - decay * prev + kappa * spatial + forcing
        prev = prev + float(step_size) * rhs
        outputs.append(prev)
    return torch.stack(outputs, dim=2)


class ResidualRefiner(nn.Module):
    """Zero-initialized adapter that learns an ODE-conditioned correction."""

    def __init__(self, horizon: int, num_states: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = 2 * horizon * num_states
        output_dim = horizon * num_states
        self.horizon = int(horizon)
        self.num_states = int(num_states)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        bsz, nodes, horizon, states = base.shape
        features = torch.cat([base, prior], dim=-1).reshape(bsz * nodes, -1)
        return self.net(features).reshape(bsz, nodes, horizon, states)


class ODEPriorGatedModel(nn.Module):
    """Frozen learned GWN trend plus an ODE-conditioned residual adapter."""

    def __init__(self, base_model: nn.Module, ode: nn.Module, horizon: int, initial_gate: float, refiner_hidden: int, prior_mode: str):
        super().__init__()
        self.base_model = base_model
        self.ode = ode
        self.refiner = ResidualRefiner(horizon, 4, refiner_hidden)
        if prior_mode not in {"learned_ode", "zero", "persistence"}:
            raise ValueError("prior_mode must be learned_ode, zero, or persistence")
        self.prior_mode = prior_mode
        initial_logit = math.log(initial_gate / (1.0 - initial_gate))
        self.gate_logits = nn.Parameter(torch.full((horizon,), initial_logit, dtype=torch.float32))
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()
        self.horizon = int(horizon)

    def train(self, mode: bool = True):
        """Train the adapter while keeping the frozen backbone deterministic."""
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, x: torch.Tensor, init_states: torch.Tensor, phys_seq: torch.Tensor, adj: torch.Tensor):
        with torch.no_grad():
            base = self.base_model(x)
        if self.prior_mode == "learned_ode":
            prior = integrate_ode_prior(self.ode, init_states, phys_seq, adj)
        elif self.prior_mode == "persistence":
            prior = init_states.unsqueeze(2).expand(-1, -1, self.horizon, -1)
        else:
            prior = torch.zeros_like(base)
        gate = torch.sigmoid(self.gate_logits).view(1, 1, self.horizon, 1)
        correction = self.refiner(base, prior)
        final = base + gate * correction
        return final, base, prior, gate

    def gate_values(self) -> np.ndarray:
        return torch.sigmoid(self.gate_logits.detach()).cpu().numpy()


def load_base_model(seed: int, data: dict, args, device: torch.device) -> nn.Module:
    adj = data["graph_priors"][args.fixed_graph_type]
    base = p104.GraphWaveNetMultistate(
        data["feats"], adj, args.hidden_dim, args.horizon, 4,
        args.diffusion_steps, args.gwn_blocks, args.dropout,
    ).to(device)
    source_results = Path(args.source_results)
    if not source_results.is_absolute():
        source_results = ROOT / source_results
    checkpoint = (
        source_results / f"seed_{seed}" / f"horizon_{args.horizon}h"
        / "gwn_multistate_no_physics" / "best_checkpoint.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing formal Multistate GWN checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base.load_state_dict(payload["model_state_dict"])
    base.eval()
    return base


def normalized_eta_loss(pred: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor) -> torch.Tensor:
    scaled = (pred - target) / state_scale
    return torch.mean(scaled[..., 0] ** 2)


def normalized_multistate_loss(pred: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, aux_weight: float) -> torch.Tensor:
    scaled = (pred - target) / state_scale
    return torch.mean(scaled[..., 0] ** 2) + float(aux_weight) * torch.mean(scaled[..., 1:] ** 2)


def train_one(model, data, train_loader, val_loader, args, device, run_dir: Path):
    state_scale = torch.tensor(data["state_scale"].reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    trainable = list(model.refiner.parameters()) + [model.gate_logits]
    if model.prior_mode == "learned_ode":
        trainable = list(model.ode.parameters()) + trainable
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    bad_epochs = 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = []
        for xb, target, _, init_states, phys_seq in train_loader:
            xb = xb.to(device)
            target = target.to(device)
            init_states = init_states.to(device)
            phys_seq = phys_seq.to(device)
            optimizer.zero_grad(set_to_none=True)
            final, _, prior, gate = model(xb, init_states, phys_seq, model.ode_adj)
            final_loss = normalized_multistate_loss(final, target, state_scale, args.aux_weight)
            prior_loss = normalized_eta_loss(prior, target, state_scale) if model.prior_mode == "learned_ode" else torch.zeros((), device=device)
            smooth_gate = torch.mean((gate[:, :, 1:, :] - gate[:, :, :-1, :]) ** 2)
            ode_reg = torch.mean(model.ode.beta ** 2) + torch.mean(model.ode.bias ** 2) if model.prior_mode == "learned_ode" else torch.zeros((), device=device)
            loss = final_loss + args.prior_loss_weight * prior_loss + args.gate_smooth_weight * smooth_gate + args.ode_reg_weight * ode_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            totals.append([float(loss.detach().cpu()), float(final_loss.detach().cpu()), float(prior_loss.detach().cpu())])
        model.eval()
        val_final, val_prior = [], []
        with torch.no_grad():
            for xb, target, _, init_states, phys_seq in val_loader:
                final, _, prior, _ = model(xb.to(device), init_states.to(device), phys_seq.to(device), model.ode_adj)
                val_final.append(float(normalized_eta_loss(final, target.to(device), state_scale).cpu()))
                val_prior.append(float(normalized_eta_loss(prior, target.to(device), state_scale).cpu()))
        val_loss = float(np.mean(val_final))
        scheduler.step(val_loss)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(np.asarray(totals)[:, 0])),
            "train_final_loss": float(np.mean(np.asarray(totals)[:, 1])),
            "train_prior_loss": float(np.mean(np.asarray(totals)[:, 2])),
            "val_eta_loss": val_loss,
            "val_prior_eta_loss": float(np.mean(val_prior)),
            "lr": optimizer.param_groups[0]["lr"],
            **{f"gate_{i+1:02d}h": float(value) for i, value in enumerate(model.gate_values())},
        }
        history.append(row)
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"epoch={epoch:03d} train={row['train_loss']:.6f} val={val_loss:.6f} prior_val={row['val_prior_eta_loss']:.6f} gate24={model.gate_values()[-1]:.3f}")
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "training_log.csv", index=False)
    return history_df, best_val, time.perf_counter() - start


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    outputs = {"final": [], "base": [], "prior": [], "true": [], "tide": []}
    for xb, target, tide, init_states, phys_seq in loader:
        final, base, prior, _ = model(xb.to(device), init_states.to(device), phys_seq.to(device), model.ode_adj)
        outputs["final"].append(final.cpu().numpy())
        outputs["base"].append(base.cpu().numpy())
        outputs["prior"].append(prior.cpu().numpy())
        outputs["true"].append(target.numpy())
        outputs["tide"].append(tide.numpy())
    return {key: np.concatenate(value) for key, value in outputs.items()}


def score_prediction(pred_states: np.ndarray, true_states: np.ndarray, tide: np.ndarray, data: dict, args) -> dict:
    metrics = v2.summarize_metrics(true_states, pred_states, tide)
    metrics.update(v3.summarize_extreme_metrics(true_states, pred_states))
    if hasattr(args, "train_ratio"):
        train_end = int(len(data["arrays"]["residual"]) * args.train_ratio)
    else:
        train_end = int(np.searchsorted(pd.to_datetime(data["arrays"]["time"]).to_numpy(), pd.Timestamp(args.fold_train_end).to_datetime64(), side="left"))
    thresholds = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    metrics.update(p104.operational_event_metrics(true_states[..., 0], pred_states[..., 0], thresholds))
    return metrics


def run_seed(seed: int, args, device: torch.device) -> list[dict]:
    run_dir = Path(args.output_dir) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metric_path = run_dir / "metrics.csv"
    if args.resume and metric_path.exists() and (run_dir / "predictions.npz").exists():
        return pd.read_csv(metric_path).to_dict("records")
    set_seed(seed, args.cpu_threads)
    data = final4.build_enhanced_data(args, args.horizon, add_ode_prior=False)
    base = load_base_model(seed, data, args, device)
    ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    model = ODEPriorGatedModel(base, ode, args.horizon, args.initial_gate, args.refiner_hidden, args.prior_mode).to(device)
    model.ode_adj = torch.tensor(data["graph_priors"][args.fixed_graph_type], dtype=torch.float32, device=device)
    train_loader = make_loader(data["multi_train"], args, True, seed)
    val_loader = make_loader(data["multi_val"], args, False, seed)
    test_loader = make_loader(data["multi_test"], args, False, seed)
    history, best_val, seconds = train_one(model, data, train_loader, val_loader, args, device, run_dir)
    outputs = predict_all(model, test_loader, device)
    rows = []
    for key, pred in [("multistate_gwn", outputs["base"]), ("ode_only", outputs["prior"]), ("ode_gwn_equal", 0.5 * outputs["base"] + 0.5 * outputs["prior"]), ("ode_gwn_gated", outputs["final"])]:
        metrics = score_prediction(pred, outputs["true"], outputs["tide"], data, args)
        rows.append({"seed": seed, "model": key, "best_val_eta_loss": best_val, "training_seconds": seconds, "gate_values": json.dumps(model.gate_values().tolist()), **metrics})
    pd.DataFrame(rows).to_csv(metric_path, index=False)
    np.savez_compressed(
        run_dir / "predictions.npz", base=outputs["base"], prior=outputs["prior"], equal=0.5 * outputs["base"] + 0.5 * outputs["prior"], gated=outputs["final"], true=outputs["true"], tide=outputs["tide"], gate=model.gate_values(), station_ids=np.asarray(v2.STATION_IDS)
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "seed": seed,
                "model": "ode_prior_gated_multistate_gwn",
                "prior_mode": args.prior_mode,
                "horizon": args.horizon,
                "window": args.window,
                "train_stride": args.train_stride,
                "best_val_eta_loss": best_val,
            },
        },
        run_dir / "best_checkpoint.pt",
    )
    (run_dir / "ODE_COEFFICIENTS.json").write_text(json.dumps(ode.coefficients(), indent=2), encoding="utf-8")
    print(f"seed={seed} base24={rows[0]['last_residual_R2']:.4f} prior24={rows[1]['last_residual_R2']:.4f} equal24={rows[2]['last_residual_R2']:.4f} gated24={rows[3]['last_residual_R2']:.4f}")
    return rows


def merge_results(args) -> None:
    output_dir = Path(args.output_dir)
    files = sorted(output_dir.glob("seed_*/metrics.csv"))
    if not files:
        raise RuntimeError("No completed seed metrics found")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    data = data[data["seed"].isin(args.seeds)].copy()
    data.to_csv(output_dir / "all_runs.csv", index=False)
    metrics = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2", "event_CSI", "event_PR_AUC"]
    summary = data.groupby("model")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x) for col in summary.columns.to_flat_index()]
    summary.to_csv(output_dir / "mean_std.csv", index=False)
    piv = data.pivot(index="seed", columns="model", values=metrics)
    comparisons = []
    for model in ["ode_only", "ode_gwn_equal", "ode_gwn_gated"]:
        for metric in metrics:
            if (metric, model) not in piv or (metric, "multistate_gwn") not in piv:
                continue
            delta = piv[(metric, model)] - piv[(metric, "multistate_gwn")]
            comparisons.append({"comparison": f"{model}_minus_multistate_gwn", "metric": metric, "mean_delta": float(delta.mean()), "std_delta": float(delta.std()), "wins": int((delta > 0).sum()), "count": int(delta.notna().sum()), "deltas": json.dumps(delta.to_numpy().tolist())})
    pd.DataFrame(comparisons).to_csv(output_dir / "paired_deltas.csv", index=False)
    plot_summary(summary, output_dir / "comparison.png")
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    models = ["multistate_gwn", "ode_only", "ode_gwn_equal", "ode_gwn_gated"]
    labels = ["Multistate GWN", "ODE-only", "50/50 fusion", "Learned gate"]
    metrics = [("last_residual_R2_mean", "24-h terminal R²"), ("seq_residual_R2_mean", "Sequence R²"), ("extreme_abs_q95_residual_R2_mean", "q95 R²")]
    lookup = summary.set_index("model_") if "model_" in summary.columns else summary.set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for axis, (metric, title) in zip(axes, metrics):
        values = [float(lookup.loc[m, metric]) if m in lookup.index else np.nan for m in models]
        errors = [float(lookup.loc[m, metric.replace("_mean", "_std")]) if m in lookup.index else 0.0 for m in models]
        axis.bar(np.arange(len(models)), values, yerr=errors, capsize=3, color=["#4472C4", "#A5A5A5", "#70AD47", "#ED7D31"])
        axis.set_xticks(np.arange(len(models)), labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ODE-prior gated residual refinement of formal Multistate Graph WaveNet.")
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--source-results", default="results/priority12_physics_graph_wavenet")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=16)
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
    parser.add_argument("--initial-gate", type=float, default=0.80)
    parser.add_argument("--refiner-hidden", type=int, default=128)
    parser.add_argument("--prior-mode", choices=["learned_ode", "zero", "persistence"], default="learned_ode")
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    parser.add_argument("--print-every", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; seeds={args.seeds}; formal base checkpoints are frozen")
    if args.mode == "merge":
        merge_results(args)
        return
    rows = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, args, device))
    if rows:
        merge_results(args)


if __name__ == "__main__":
    main()
