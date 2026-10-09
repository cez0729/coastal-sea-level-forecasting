from __future__ import annotations

"""Tier-A adapter for the user-supplied ``16_A_Bayes_output.py`` model.

This script evaluates the proposed dual-expert, spline-activation,
heteroscedastic Gaussian Graph WaveNet on exactly the retrospective protocol
used by Table 1 of the manuscript.  It deliberately calls the output head
"probabilistic" rather than a full Bayesian neural network: the weights have
point estimates and only the conditional Gaussian scale is learned.
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
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "tier_a_probabilistic_spline_dual_expert_gwn"
SEEDS = [42, 123, 2024, 2025, 3407]
MODEL_KEY = "probabilistic_spline_dual_expert_gwn"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p104 = load_module("tier_a_p104", HERE / "104_priority2_physics_graph_wavenet.py")
final4 = p104.final4
priority1 = p104.priority1
v2 = p104.v2
v3 = p104.v3
v4 = p104.v4


class ChannelwiseLinearSpline(nn.Module):
    """Trainable channel-wise piecewise-linear activation."""

    def __init__(self, channels: int, knots: int, initialization: str):
        super().__init__()
        if knots < 3:
            raise ValueError("knots must be at least 3")
        grid = torch.linspace(-5.0, 5.0, knots)
        initial = torch.tanh(grid) if initialization == "tanh" else torch.relu(grid)
        self.register_buffer("grid", grid)
        self.values = nn.Parameter(initial.repeat(channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,N,T]. Linear extrapolation is intentionally replaced by
        # boundary clipping so extreme activations cannot explode.
        clipped = x.clamp(float(self.grid[0]), float(self.grid[-1]))
        step = (self.grid[-1] - self.grid[0]) / (len(self.grid) - 1)
        position = (clipped - self.grid[0]) / step
        left = position.floor().long().clamp(0, len(self.grid) - 2)
        fraction = position - left.to(position.dtype)

        flat_left = left.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        channel = torch.arange(x.shape[1], device=x.device).view(1, -1).expand_as(flat_left)
        y0 = self.values[channel, flat_left]
        y1 = self.values[channel, flat_left + 1]
        flat_fraction = fraction.permute(0, 2, 3, 1).reshape_as(y0)
        out = y0 + flat_fraction * (y1 - y0)
        return out.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[1]).permute(0, 3, 1, 2)

    def smoothness(self) -> torch.Tensor:
        second_difference = self.values[:, 2:] - 2.0 * self.values[:, 1:-1] + self.values[:, :-2]
        return torch.mean(second_difference.square())


class SplineGraphWaveBlock(nn.Module):
    def __init__(self, channels: int, supports: list[np.ndarray], dilation: int, dropout: float, knots: int):
        super().__init__()
        pad = dilation
        self.filter_conv = nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation), padding=(0, pad))
        self.gate_conv = nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation), padding=(0, pad))
        self.filter_activation = ChannelwiseLinearSpline(channels, knots, "tanh")
        self.graph = priority1.DiffusionGraphLinear(channels, channels, supports)
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        filt = self.filter_activation(self.filter_conv(x)[..., : x.shape[-1]])
        gate = torch.sigmoid(self.gate_conv(x)[..., : x.shape[-1]])
        h = filt * gate
        batch, channels, nodes, steps = h.shape
        h = h.permute(0, 3, 2, 1).reshape(batch * steps, nodes, channels)
        h = self.graph(h).reshape(batch, steps, nodes, channels).permute(0, 3, 2, 1)
        return self.norm(residual + self.dropout(h))


class SplineGraphWaveExpert(nn.Module):
    def __init__(self, input_dim: int, adjacency: np.ndarray, hidden: int, horizon: int, states: int,
                 diffusion_steps: int, blocks: int, dropout: float, knots: int):
        super().__init__()
        supports = priority1.make_diffusion_supports(adjacency, diffusion_steps)
        self.horizon = int(horizon)
        self.states = int(states)
        self.input_proj = nn.Conv2d(input_dim, hidden, (1, 1))
        self.blocks = nn.ModuleList([
            SplineGraphWaveBlock(hidden, supports, 2 ** (index % 4), dropout, knots)
            for index in range(blocks)
        ])
        self.context_activation = ChannelwiseLinearSpline(hidden, knots, "relu")
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, (1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, horizon * states, (1, 1)),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x.permute(0, 3, 2, 1))
        for block in self.blocks:
            h = block(h)
        context = self.context_activation(h)[..., -1]
        out = self.head(h)[..., -1].permute(0, 2, 1)
        return out.reshape(x.shape[0], x.shape[2], self.horizon, self.states), context

    def spline_smoothness(self) -> torch.Tensor:
        terms = [block.filter_activation.smoothness() for block in self.blocks]
        terms.append(self.context_activation.smoothness())
        return torch.stack(terms).mean()


class ProbabilisticSplineDualExpertGWN(nn.Module):
    def __init__(self, input_dim: int, adjacency: np.ndarray, hidden: int, horizon: int,
                 diffusion_steps: int, blocks: int, dropout: float, knots: int, uncertainty_hidden: int):
        super().__init__()
        self.horizon = int(horizon)
        # Both experts receive the same 34-feature history. "Eta-only" refers
        # to its supervision/output, matching the manuscript's expert pair.
        self.eta_expert = SplineGraphWaveExpert(
            input_dim, adjacency, hidden, horizon, 1, diffusion_steps, blocks, dropout, knots
        )
        self.multistate_expert = SplineGraphWaveExpert(
            input_dim, adjacency, hidden, horizon, 4, diffusion_steps, blocks, dropout, knots
        )
        self.gate = nn.Sequential(
            nn.Conv1d(2 * hidden, hidden, 1), nn.ReLU(), nn.Conv1d(hidden, horizon, 1)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.scale_head = nn.Sequential(nn.Linear(5, uncertainty_hidden), nn.ReLU(), nn.Linear(uncertainty_hidden, 1))
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.constant_(self.scale_head[-1].bias, -1.5)

    def forward(self, x: torch.Tensor):
        eta, eta_context = self.eta_expert(x)
        multi, multi_context = self.multistate_expert(x)
        eta = eta[..., 0]
        multi_eta = multi[..., 0]
        weight = torch.sigmoid(self.gate(torch.cat([eta_context, multi_context], dim=1))).permute(0, 2, 1)
        mean = weight * eta + (1.0 - weight) * multi_eta
        lead = torch.linspace(0.0, 1.0, self.horizon, device=x.device, dtype=x.dtype).view(1, 1, -1)
        lead = lead.expand_as(mean)
        scale_features = torch.stack([mean, eta, multi_eta, weight, lead], dim=-1)
        log_sigma_z = self.scale_head(scale_features).squeeze(-1).clamp(-6.0, 2.0)
        return mean, log_sigma_z, eta, multi, weight

    def spline_smoothness(self) -> torch.Tensor:
        return 0.5 * (self.eta_expert.spline_smoothness() + self.multistate_expert.spline_smoothness())


def gaussian_nll(target_z: torch.Tensor, mean_z: torch.Tensor, log_sigma_z: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean((target_z - mean_z).square() * torch.exp(-2.0 * log_sigma_z)
                            + 2.0 * log_sigma_z + math.log(2.0 * math.pi))


class CandidateLoss(nn.Module):
    def __init__(self, physics_ode: nn.Module, data: dict, args):
        super().__init__()
        self.physics_ode = physics_ode
        self.register_buffer("state_scale", torch.tensor(data["state_scale"], dtype=torch.float32).view(1, 1, 1, -1))
        self.register_buffer("delta_scale", torch.tensor(data["delta_scale"], dtype=torch.float32).view(1, 1, 1, -1))
        self.register_buffer("state_weights", torch.tensor([1.0, 0.35, 0.35, 0.25]).view(1, 1, 1, -1))
        self.aux_weight = float(args.aux_weight)
        self.eta_aux_weight = float(args.eta_aux_weight)
        self.last_step_weight = float(args.last_step_weight)
        self.spline_lambda = float(args.spline_lambda)
        self.huber = nn.SmoothL1Loss(reduction="none")

    def physics_loss(self, pred: torch.Tensor, init_states: torch.Tensor, phys_seq: torch.Tensor,
                     adjacency: torch.Tensor) -> torch.Tensor:
        _, residual = self.physics_ode(pred, init_states, phys_seq, adjacency)
        scaled = residual / self.delta_scale
        return torch.mean(self.huber(scaled, torch.zeros_like(scaled)) * self.state_weights)

    def forward(self, outputs, target, init_states, phys_seq, adjacency, physics_lambda: float):
        mean, log_sigma_z, eta, multi, _ = outputs
        target_eta = target[..., 0]
        eta_scale = self.state_scale[..., 0]
        nll = gaussian_nll(target_eta / eta_scale, mean / eta_scale, log_sigma_z)
        eta_aux = torch.mean(((eta - target_eta) / eta_scale).square())
        multi_mse = torch.mean(((multi - target) / self.state_scale).square())
        terminal = torch.mean(((mean[:, :, -1] - target_eta[:, :, -1]) / eta_scale.squeeze(-1)).square())
        physics = self.physics_loss(multi, init_states, phys_seq, adjacency) if physics_lambda > 0 else mean.new_zeros(())
        spline = outputs[0].new_zeros(())
        total = nll + self.eta_aux_weight * eta_aux + self.aux_weight * multi_mse
        total = total + self.last_step_weight * terminal + physics_lambda * physics
        return total, {"nll": nll, "eta_mse_z": torch.mean(((mean - target_eta) / eta_scale).square()),
                       "eta_aux_mse_z": eta_aux, "multi_mse_z": multi_mse, "terminal_mse_z": terminal,
                       "physics_loss": physics, "spline_loss": spline}


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator if shuffle else None, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def physics_weight(epoch: int, args) -> float:
    return v2.physics_lambda_for_epoch(epoch, args.physics_lambda, args.physics_warmup_epochs, args.physics_ramp_epochs)


def epoch_pass(model, criterion, loader, adjacency, device, args, epoch: int, optimizer=None):
    training = optimizer is not None
    model.train(training)
    criterion.physics_ode.train(training and args.physics_lambda > 0)
    totals: dict[str, float] = {}
    count = 0
    lam = physics_weight(epoch, args) if args.physics_lambda > 0 else 0.0
    for xb, target, _, init_states, phys_seq in loader:
        xb, target = xb.to(device), target.to(device)
        init_states, phys_seq = init_states.to(device), phys_seq.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(xb)
            loss, parts = criterion(outputs, target, init_states, phys_seq, adjacency, lam)
            spline = model.spline_smoothness()
            loss = loss + args.spline_lambda * spline
            parts["spline_loss"] = spline
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.physics_ode.parameters()), args.grad_clip)
                optimizer.step()
        values = {"total_loss": loss, **parts}
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    means, logs, truths, tides, gates = [], [], [], [], []
    for xb, target, tide, _, _ in loader:
        mean, log_sigma, _, _, gate = model(xb.to(device))
        means.append(mean.cpu().numpy())
        logs.append(log_sigma.cpu().numpy())
        truths.append(target[..., 0].numpy())
        tides.append(tide.numpy())
        gates.append(gate.cpu().numpy())
    return tuple(np.concatenate(items) for items in [means, logs, truths, tides, gates])


def uncertainty_metrics(true: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> pd.DataFrame:
    rows = []
    for lead in range(true.shape[-1]):
        error = true[..., lead] - mean[..., lead]
        s = np.maximum(sigma[..., lead], 1e-8)
        rows.append({
            "lead_hour": lead + 1,
            "gaussian_nll": float(np.mean(0.5 * (error / s) ** 2 + np.log(s) + 0.5 * np.log(2 * np.pi))),
            "coverage_50": float(np.mean(np.abs(error) <= 0.67448975 * s)),
            "coverage_80": float(np.mean(np.abs(error) <= 1.28155157 * s)),
            "coverage_95": float(np.mean(np.abs(error) <= 1.95996398 * s)),
            "mean_interval_width_95_m": float(np.mean(2 * 1.95996398 * s)),
        })
    return pd.DataFrame(rows)


def train_seed(seed: int, args, data: dict, device: torch.device) -> dict:
    p104.set_reproducible(seed, args.cpu_threads)
    run_dir = Path(args.output_dir) / f"seed_{seed}"
    metrics_path = run_dir / "metrics.csv"
    if args.resume and metrics_path.exists() and (run_dir / "COMPLETE.json").exists():
        return pd.read_csv(metrics_path).iloc[0].to_dict()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(data["multi_train"], args.batch_size, True, seed)
    val_loader = make_loader(data["multi_val"], args.batch_size, False, seed)
    test_loader = make_loader(data["multi_test"], args.batch_size, False, seed)
    adjacency_np = data["graph_priors"][args.fixed_graph_type]
    adjacency = torch.tensor(adjacency_np, dtype=torch.float32, device=device)
    model = ProbabilisticSplineDualExpertGWN(
        data["feats"], adjacency_np, args.hidden_dim, args.horizon,
        args.diffusion_steps, args.gwn_blocks, args.dropout, args.spline_knots,
        args.uncertainty_hidden,
    ).to(device)
    physics_ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    criterion = CandidateLoss(physics_ode, data, args).to(device)
    parameters = list(model.parameters()) + (list(physics_ode.parameters()) if args.physics_lambda > 0 else [])
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    best_state = None
    best_physics_state = None
    best_val = float("inf")
    bad = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_values = epoch_pass(model, criterion, train_loader, adjacency, device, args, epoch, optimizer)
        val_values = epoch_pass(model, criterion, val_loader, adjacency, device, args, epoch)
        selection = val_values[args.selection_metric]
        scheduler.step(selection)
        row = {"epoch": epoch, "physics_lambda": physics_weight(epoch, args), "selection_score": selection,
               **{f"train_{k}": v for k, v in train_values.items()},
               **{f"val_{k}": v for k, v in val_values.items()}}
        history.append(row)
        if selection < best_val - args.min_delta:
            best_val, bad = selection, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_physics_state = {
                key: value.detach().cpu().clone() for key, value in physics_ode.state_dict().items()
            }
        else:
            bad += 1
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"seed={seed} epoch={epoch:03d} train_nll={train_values['nll']:.5f} "
                  f"val_nll={val_values['nll']:.5f} val_mse_z={val_values['eta_mse_z']:.5f}", flush=True)
        if bad >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was produced")
    model.load_state_dict(best_state)
    if best_physics_state is not None:
        physics_ode.load_state_dict(best_physics_state)
    mean, log_sigma_z, true, tide, gate = predict(model, test_loader, device)
    residual_scale = float(data["state_scale"][0])
    sigma = np.exp(log_sigma_z) * residual_scale
    metrics = final4.summarize_single(true, mean, tide)
    uncertainty = uncertainty_metrics(true, mean, sigma)
    row = {
        "seed": seed, "model": MODEL_KEY, "evidence_status": "exploratory_tier_a_candidate",
        "best_val_score": best_val, "selection_metric": args.selection_metric,
        "physics_lambda": args.physics_lambda, "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "training_seconds": time.perf_counter() - started,
        "test_gaussian_nll_mean": float(uncertainty["gaussian_nll"].mean()),
        "test_coverage_95_mean": float(uncertainty["coverage_95"].mean()),
        "test_interval_width_95_m_mean": float(uncertainty["mean_interval_width_95_m"].mean()),
        **metrics,
    }
    pd.DataFrame(history).to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    uncertainty.to_csv(run_dir / "uncertainty_by_lead.csv", index=False)
    np.savez_compressed(run_dir / "predictions.npz", pred_residual=mean, true_residual=true, target_tide=tide,
                        sigma_residual=sigma, gate_eta_weight=gate, station_ids=np.asarray(v2.STATION_IDS))
    torch.save(
        {
            "model_state_dict": best_state,
            "physics_ode_state_dict": best_physics_state,
            "args": vars(args),
            "seed": seed,
        },
        run_dir / "best_checkpoint.pt",
    )
    (run_dir / "COMPLETE.json").write_text(json.dumps({"seed": seed, "model": MODEL_KEY}, indent=2), encoding="utf-8")
    return row


def smoke_test(args, data: dict, device: torch.device) -> None:
    loader = make_loader(data["multi_train"], min(2, args.batch_size), False, args.seeds[0])
    xb, target, _, init_states, phys_seq = next(iter(loader))
    adjacency_np = data["graph_priors"][args.fixed_graph_type]
    model = ProbabilisticSplineDualExpertGWN(
        data["feats"], adjacency_np, args.hidden_dim,
        args.horizon, args.diffusion_steps, args.gwn_blocks, args.dropout, args.spline_knots,
        args.uncertainty_hidden,
    ).to(device)
    physics_ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    criterion = CandidateLoss(physics_ode, data, args).to(device)
    outputs = model(xb.to(device))
    loss, _ = criterion(outputs, target.to(device), init_states.to(device), phys_seq.to(device),
                        torch.tensor(adjacency_np, device=device), 0.0)
    (loss + args.spline_lambda * model.spline_smoothness()).backward()
    expected = (len(xb), data["nodes"], args.horizon)
    if tuple(outputs[0].shape) != expected or not torch.isfinite(loss):
        raise RuntimeError(f"Smoke test failed: mean={tuple(outputs[0].shape)}, expected={expected}, loss={loss}")
    print(
        f"PASS smoke test: mean={expected}, multistate={tuple(outputs[3].shape)}, "
        f"loss={float(loss.detach().cpu()):.6f}"
    )


def baseline_rows() -> pd.DataFrame:
    rows = []
    bigru = pd.read_csv(ROOT / "results/corrected_bigru_ladder/merged/corrected_bigru_ladder_all_runs.csv")
    sub = bigru[bigru["model_key"] == "physical_loss"]
    for _, r in sub.iterrows():
        rows.append({"seed": int(r["seed"]), "model": "Physical-loss GNN-BiGRU",
                     "seq_residual_R2": r["seq_residual_R2"], "last_residual_R2": r["last_residual_R2"],
                     "extreme_abs_q95_residual_R2": r["extreme_abs_q95_residual_R2"]})
    p2 = pd.read_csv(ROOT / "results/priority12_physics_graph_wavenet/priority2_physics_gwn_all_runs.csv")
    labels = {"gwn_eta_only": "Eta-only GWN", "gwn_multistate_no_physics": "Multistate GWN, no physics"}
    for key, label in labels.items():
        for _, r in p2[p2["config"] == key].iterrows():
            rows.append({"seed": int(r["seed"]), "model": label, "seq_residual_R2": r["seq_residual_R2"],
                         "last_residual_R2": r["last_residual_R2"],
                         "extreme_abs_q95_residual_R2": r["extreme_abs_q95_residual_R2"]})
    hsdt = pd.read_csv(ROOT / "results/horizon_specialized_dual_task_gwn/all_runs.csv")
    for _, r in hsdt[hsdt["config"] == "horizon_specialized_dual_task_gwn"].iterrows():
        rows.append({"seed": int(r["seed"]), "model": "HS-DT-GWN", "seq_residual_R2": r["seq_residual_R2"],
                     "last_residual_R2": r["last_residual_R2"],
                     "extreme_abs_q95_residual_R2": r["extreme_abs_q95_residual_R2"]})
    # Ridge is deterministic. The source stores five identical bookkeeping
    # rows, which must not be presented as five stochastic replications.
    r = pd.read_csv(ROOT / "results/formal_varx_ridge_20260811/all_runs.csv").iloc[0]
    rows.append({"seed": np.nan, "model": "VARX-Ridge", "seq_residual_R2": r["seq_residual_R2"],
                 "last_residual_R2": r["last_residual_R2"],
                 "extreme_abs_q95_residual_R2": r["extreme_abs_q95_residual_R2"]})
    return pd.DataFrame(rows)


def merge(args) -> None:
    output = Path(args.output_dir)
    candidate_paths = sorted(output.glob("seed_*/metrics.csv"))
    candidate = pd.concat([pd.read_csv(path) for path in candidate_paths], ignore_index=True) if candidate_paths else pd.DataFrame()
    if not candidate.empty:
        candidate.to_csv(output / "candidate_all_runs.csv", index=False)
    combined = baseline_rows()
    if not candidate.empty:
        add = candidate.rename(columns={"model": "model"})
        add["model"] = "Probabilistic Spline Dual-Expert GWN"
        combined = pd.concat([combined, add[combined.columns]], ignore_index=True)
    combined.to_csv(output / "tier_a_comparison_all_runs.csv", index=False)
    metrics = ["seq_residual_R2", "last_residual_R2", "extreme_abs_q95_residual_R2"]
    summary = combined.groupby("model")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(x) for x in column if x) for column in summary.columns.to_flat_index()]
    summary.to_csv(output / "tier_a_comparison_mean_std.csv", index=False)
    order = ["Physical-loss GNN-BiGRU", "Eta-only GWN", "Multistate GWN, no physics", "HS-DT-GWN", "VARX-Ridge",
             "Probabilistic Spline Dual-Expert GWN"]
    lookup = summary.set_index("model")
    lines = ["\\begin{tabular}{lccc}", "\\toprule", "Model & Sequence $R^2$ & Lead-24 $R^2$ & $q95$ $R^2$ \\\\", "\\midrule"]
    for model in order:
        if model not in lookup.index:
            continue
        r = lookup.loc[model]
        lines.append(f"{model} & {r['seq_residual_R2_mean']:.4f} & {r['last_residual_R2_mean']:.4f} & "
                     f"{r['extreme_abs_q95_residual_R2_mean']:.4f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "tier_a_comparison_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Tier-A probabilistic spline dual-expert GWN adapter")
    parser.add_argument("--mode", choices=["smoke", "run", "merge"], default="smoke")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    parser.add_argument("--fixed-graph-type", choices=["identity", "distance", "corr"], default="distance")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--gwn-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--spline-knots", type=int, default=17)
    parser.add_argument("--uncertainty-hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eta-aux-weight", type=float, default=0.25)
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.20)
    parser.add_argument("--physics-lambda", type=float, default=0.0002)
    parser.add_argument("--physics-warmup-epochs", type=int, default=8)
    parser.add_argument("--physics-ramp-epochs", type=int, default=14)
    parser.add_argument("--spline-lambda", type=float, default=1e-4)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    # Table 1 compares point-forecast R2, so the default checkpoint criterion
    # matches the deterministic baselines. NLL remains available for a
    # probability-first sensitivity run.
    parser.add_argument("--selection-metric", choices=["nll", "eta_mse_z"], default="eta_mse_z")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if args.mode == "merge":
        merge(args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = final4.build_enhanced_data(args, args.horizon, add_ode_prior=False)
    if args.mode == "smoke":
        smoke_test(args, data, device)
        return
    rows = []
    for seed in args.seeds:
        rows.append(train_seed(seed, args, data, device))
        pd.DataFrame(rows).to_csv(output / "candidate_partial.csv", index=False)
    merge(args)


if __name__ == "__main__":
    main()
