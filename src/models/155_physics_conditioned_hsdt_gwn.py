from __future__ import annotations

"""Physics-conditioned horizon-specialized dual-expert GWN prototype.

The two GWN experts are frozen chronological-refit checkpoints.  A small
causal gate learns when to trust the eta-only or multistate expert and when
to apply an ODE-conditioned residual correction.  A same-capacity no-physics
gate is trained with the physical features masked, so this screen separates
gate/refiner capacity from the physical signal.
"""

import argparse
import copy
import importlib.util
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "pc_hsdt_gwn_screen"
DEFAULT_SOURCE = ROOT / "results" / "confirmatory_hsdt_orc_refit_2025_h2"
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


confirm = load_module("pc_confirm", HERE / "134_confirmatory_hsdt_orc_chronological_refit.py")
p104 = confirm.p104
rolling = confirm.rolling
v2 = confirm.v2
adapter_impl = load_module("pc_adapter", HERE / "109_ode_prior_gated_multistate_gwn.py")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def load_expert(seed: int, data: dict, args, states: int, device: torch.device) -> nn.Module:
    model = confirm.make_gwn(data, args, states).to(device)
    name = "gwn_eta_only" if states == 1 else "gwn_multistate_no_physics"
    path = project_path(args.source_results) / f"seed_{seed}" / name / "best_checkpoint.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_ode(seed: int, data: dict, args, device: torch.device) -> nn.Module:
    ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    path = (
        project_path(args.source_results)
        / f"seed_{seed}"
        / "adapter_learned_ode"
        / "best_checkpoint.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = {key[4:]: value for key, value in payload["model_state_dict"].items() if key.startswith("ode.")}
    if state:
        ode.load_state_dict(state)
    ode.eval()
    for parameter in ode.parameters():
        parameter.requires_grad_(False)
    return ode


def causal_prior(ode, init_states, phys_seq, adjacency):
    return adapter_impl.integrate_ode_prior(ode, init_states, phys_seq, adjacency)


@torch.no_grad()
def collect_split(dataset, eta_model, multi_model, ode, adjacency, args, device, seed):
    loader = make_loader(dataset, args, False, seed)
    eta_rows, multi_rows, prior_rows, target_rows, tide_rows, force_rows = [], [], [], [], [], []
    for xb, target, tide, init_states, phys_seq in loader:
        xb = xb.to(device)
        init_states = init_states.to(device)
        phys_seq = phys_seq.to(device)
        eta = eta_model(xb)
        multi = multi_model(xb)
        prior = causal_prior(ode, init_states, phys_seq, adjacency)
        forcing = torch.linalg.vector_norm(phys_seq[:, :, 0, :], dim=-1)
        eta_rows.append(eta.cpu().numpy())
        multi_rows.append(multi[..., 0].cpu().numpy())
        prior_rows.append(prior[..., 0].cpu().numpy())
        target_rows.append(target[..., 0].numpy())
        tide_rows.append(tide.numpy())
        force_rows.append(forcing.cpu().numpy())
    return {
        "eta": np.concatenate(eta_rows).astype(np.float32),
        "multi": np.concatenate(multi_rows).astype(np.float32),
        "prior": np.concatenate(prior_rows).astype(np.float32),
        "target": np.concatenate(target_rows).astype(np.float32),
        "tide": np.concatenate(tide_rows).astype(np.float32),
        "forcing": np.concatenate(force_rows).astype(np.float32),
    }


class PhysicsConditionedFusion(nn.Module):
    """Per-station, per-horizon expert gate plus physics residual correction."""

    def __init__(self, horizon: int, nodes: int, hidden: int, eta_scale: float, use_physics: bool, explicit_candidate: bool):
        super().__init__()
        self.horizon = int(horizon)
        self.nodes = int(nodes)
        self.use_physics = bool(use_physics)
        self.explicit_candidate = bool(explicit_candidate)
        self.register_buffer("eta_scale", torch.tensor(float(eta_scale)))
        self.horizon_bias = nn.Parameter(torch.zeros(horizon))
        self.station_embedding = nn.Parameter(torch.zeros(nodes, 2))
        in_dim = 8
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def features(self, eta, multi, prior, forcing):
        bsz, nodes, horizon = eta.shape
        prior_delta = multi - prior
        violation = prior_delta.abs()
        force = forcing.unsqueeze(-1).expand(-1, -1, horizon)
        h = torch.linspace(0.0, 1.0, horizon, device=eta.device).view(1, 1, horizon).expand(bsz, nodes, -1)
        station = self.station_embedding[:, 0].view(1, nodes, 1).expand(bsz, nodes, horizon)
        features = torch.stack([eta, multi, prior, prior_delta.abs(), force, violation, h, station], dim=-1)
        if not self.use_physics:
            features[..., 2:6] = 0.0
        return features

    def forward(self, eta, multi, prior, forcing):
        features = self.features(eta, multi, prior, forcing)
        raw = self.net(features)
        gate = torch.sigmoid(raw[..., 0] + self.horizon_bias.view(1, 1, -1))
        correction = 0.25 * self.eta_scale * torch.tanh(raw[..., 1]) * gate
        physics_gate = torch.sigmoid(raw[..., 2] + self.horizon_bias.view(1, 1, -1))
        if self.use_physics and self.explicit_candidate:
            normalized_delta = torch.clamp((prior - multi) / self.eta_scale, -3.0, 3.0)
            physics_correction = 0.25 * self.eta_scale * physics_gate * torch.tanh(normalized_delta)
        else:
            physics_gate = torch.zeros_like(physics_gate)
            physics_correction = torch.zeros_like(correction)
        fused = eta + gate * (multi - eta)
        final = fused + correction + physics_correction
        return final, gate, correction, physics_gate, physics_correction


def gate_loss(final, target, gate, correction, physics_correction, args, eta_scale):
    normalized = (final - target) / eta_scale
    data_loss = torch.mean(normalized**2)
    smooth = torch.mean((gate[..., 1:] - gate[..., :-1]) ** 2)
    correction_reg = torch.mean((correction / eta_scale) ** 2)
    physics_reg = torch.mean((physics_correction / eta_scale) ** 2)
    return data_loss + args.gate_smooth_weight * smooth + args.correction_reg_weight * correction_reg + args.physics_correction_reg_weight * physics_reg


def train_gate(train, val, data, args, seed, device, use_physics):
    eta_scale = float(data["state_scale"][0])
    model = PhysicsConditionedFusion(args.horizon, data["nodes"], args.gate_hidden, eta_scale, use_physics, args.explicit_physics_candidate).to(device)
    def tensors(split):
        return TensorDataset(
            torch.from_numpy(split["eta"]), torch.from_numpy(split["multi"]),
            torch.from_numpy(split["prior"]), torch.from_numpy(split["forcing"]),
            torch.from_numpy(split["target"]),
        )
    train_loader = make_loader(tensors(train), args, True, seed)
    val_loader = make_loader(tensors(val), args, False, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.gate_lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    bad = 0
    history = []
    for epoch in range(1, args.gate_epochs + 1):
        model.train()
        train_losses = []
        for eta, multi, prior, forcing, target in train_loader:
            eta, multi, prior, forcing, target = [v.to(device) for v in (eta, multi, prior, forcing, target)]
            optimizer.zero_grad(set_to_none=True)
            final, gate, correction, _, physics_correction = model(eta, multi, prior, forcing)
            loss = gate_loss(final, target, gate, correction, physics_correction, args, model.eta_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for eta, multi, prior, forcing, target in val_loader:
                eta, multi, prior, forcing, target = [v.to(device) for v in (eta, multi, prior, forcing, target)]
                final, gate, correction, _, physics_correction = model(eta, multi, prior, forcing)
                val_losses.append(float(gate_loss(final, target, gate, correction, physics_correction, args, model.eta_scale).cpu()))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"seed={seed} physics={use_physics} epoch={epoch:03d} val={val_loss:.6f}", flush=True)
        if bad >= args.gate_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_val


def score(true, pred, tide, thresholds):
    metrics = confirm.final4.summarize_single(true, pred, tide)
    metrics.update(p104.operational_event_metrics(true, pred, thresholds))
    return metrics


def evaluate_model(model, split, device):
    model.eval()
    with torch.no_grad():
        tensors = [torch.from_numpy(split[key]).to(device) for key in ("eta", "multi", "prior", "forcing")]
        final, gate, correction, physics_gate, physics_correction = model(*tensors)
    return final.cpu().numpy(), gate.cpu().numpy(), correction.cpu().numpy(), physics_gate.cpu().numpy(), physics_correction.cpu().numpy()


def run_seed(seed, args, device):
    p104.set_reproducible(seed, args.cpu_threads)
    data = rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    eta_model = load_expert(seed, data, args, 1, device)
    multi_model = load_expert(seed, data, args, 4, device)
    ode = load_ode(seed, data, args, device)
    adj = torch.tensor(data["graph_priors"][args.fixed_graph_type], dtype=torch.float32, device=device)
    splits = {
        "train": collect_split(data["multi_train"], eta_model, multi_model, ode, adj, args, device, seed),
        "val": collect_split(data["multi_val"], eta_model, multi_model, ode, adj, args, device, seed),
        "test": collect_split(data["multi_test"], eta_model, multi_model, ode, adj, args, device, seed),
    }
    models = {}
    for use_physics in (False, True):
        model, history, best_val = train_gate(splits["train"], splits["val"], data, args, seed, device, use_physics)
        name = "adaptive_hsdt_no_physics" if not use_physics else "pc_hsdt_gwn"
        models[name] = model
        history.to_csv(project_path(args.output_dir) / f"seed_{seed}_{name}_training.csv", index=False)
        torch.save({"model_state_dict": model.state_dict(), "best_val_loss": best_val, "use_physics": use_physics}, project_path(args.output_dir) / f"seed_{seed}_{name}.pt")
    test = splits["test"]
    eta, multi, true, tide = test["eta"], test["multi"], test["target"], test["tide"]
    fixed_weights = np.full(args.horizon, 0.5, dtype=np.float32); fixed_weights[-1] = 1.0
    predictions = {
        "gwn_eta_only": eta,
        "gwn_multistate_no_physics": multi,
        "hs_dt_gwn": eta + fixed_weights[None, None, :] * (multi - eta),
    }
    gate_rows = []
    for name, model in models.items():
        pred, gate, correction, physics_gate, physics_correction = evaluate_model(model, test, device)
        predictions[name] = pred
        for lead in range(args.horizon):
            gate_rows.append({"seed": seed, "model": name, "lead_hour": lead + 1, "gate_mean": float(gate[..., lead].mean()), "gate_std": float(gate[..., lead].std()), "physics_gate_mean": float(physics_gate[..., lead].mean()), "physics_gate_std": float(physics_gate[..., lead].std()), "correction_abs_mean": float(np.abs(correction[..., lead]).mean()), "physics_correction_abs_mean": float(np.abs(physics_correction[..., lead]).mean())})
        np.savez_compressed(project_path(args.output_dir) / f"seed_{seed}_{name}_predictions.npz", pred_residual=pred, gate=gate, physics_gate=physics_gate, correction=correction, physics_correction=physics_correction, true_residual=true, target_tide=tide)
    train_end = rolling.time_index(data["arrays"]["time"], args.fold_train_end)
    thresholds = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    rows = []
    for name, pred in predictions.items():
        rows.append({"seed": seed, "model": name, "use_physics": name == "pc_hsdt_gwn", **score(true, pred, tide, thresholds)})
    return rows, gate_rows


def summarize(args, rows, gate_rows):
    out = project_path(args.output_dir)
    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(out / "all_runs.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(out / "gate_diagnostics.csv", index=False)
    metrics = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2", "event_PR_AUC", "event_CSI"]
    summary = all_runs.groupby("model")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(v) for v in c if v) for c in summary.columns.to_flat_index()]
    summary.to_csv(out / "mean_std.csv", index=False)
    pivot = all_runs.pivot(index="seed", columns="model", values=metrics)
    comparisons = []
    for candidate in ("adaptive_hsdt_no_physics", "pc_hsdt_gwn"):
        for baseline in ("hs_dt_gwn", "adaptive_hsdt_no_physics") if candidate == "pc_hsdt_gwn" else ("hs_dt_gwn",):
            if candidate == baseline or (candidate == "adaptive_hsdt_no_physics" and baseline != "hs_dt_gwn"):
                continue
            for metric in metrics:
                delta = pivot[(metric, candidate)] - pivot[(metric, baseline)]
                improvement = -delta if metric.endswith(("RMSE", "MAE")) else delta
                comparisons.append({"candidate": candidate, "baseline": baseline, "metric": metric, "mean_improvement": float(improvement.mean()), "std_improvement": float(improvement.std(ddof=1)), "wins": int((improvement > 0).sum()), "count": int(len(improvement))})
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)
    labels = {"gwn_eta_only":"Eta-only GWN", "gwn_multistate_no_physics":"Multistate GWN", "hs_dt_gwn":"Fixed HS-DT", "adaptive_hsdt_no_physics":"Adaptive gate (no physics)", "pc_hsdt_gwn":"PC-HS-DT-GWN"}
    order = list(labels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    lookup = summary.set_index("model")
    x = np.arange(3); width = 0.16
    for idx, name in enumerate(order):
        vals = [lookup.loc[name, f"{metric}_mean"] for metric in ["seq_residual_R2", "last_residual_R2", "extreme_abs_q95_residual_R2"]]
        axes[0].bar(x + (idx - 2) * width, vals, width, label=labels[name])
    axes[0].set_xticks(x, ["Trajectory R2", "Lead-24 R2", "q95 R2"]); axes[0].set_ylabel("Residual R2"); axes[0].grid(axis="y", alpha=0.25); axes[0].legend(fontsize=7)
    gates = pd.DataFrame(gate_rows)
    for name, color in zip(("adaptive_hsdt_no_physics", "pc_hsdt_gwn"), ("#607D8B", "#2E7D6F")):
        g = gates[gates["model"] == name].groupby("lead_hour")["gate_mean"].mean()
        axes[1].plot(g.index, g.values, marker="o", ms=3, label=labels[name], color=color)
    axes[1].set_xlabel("Forecast lead (h)"); axes[1].set_ylabel("Mean multistate/physics gate"); axes[1].set_ylim(0, 1); axes[1].grid(alpha=0.25); axes[1].legend(fontsize=8)
    fig.suptitle("Physics-conditioned horizon-specialized dual-expert screen")
    fig.tight_layout(); fig.savefig(out / "pc_hsdt_gwn_comparison.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    report = ["# PC-HS-DT-GWN prototype", "", "This is a strict-causal chronological-refit screen using frozen eta-only and multistate GWN experts.", "The gate and residual adapter are trained on the training period and selected by validation eta loss. Physics features use only last-input forcing, causal ODE prior, and expert-prior discrepancy.", "", "## Evidence boundary", "", "2025 H2 was previously viewed during project development; this is not an untouched independent holdout. The no-physics adaptive gate has the same trainable gate/refiner capacity as PC-HS-DT-GWN.", "", "## Results", "", summary.to_string(index=False)]
    (out / "RESULTS_SUMMARY_CN.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False)); print(pd.DataFrame(comparisons).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Physics-conditioned HS-DT-GWN prototype")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT)); parser.add_argument("--source-results", default=str(DEFAULT_SOURCE)); parser.add_argument("--data-dir", default="data/processed_multiyear_2023_2025")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42]); parser.add_argument("--fold-train-end", default="2025-01-01"); parser.add_argument("--fold-val-end", default="2025-07-01"); parser.add_argument("--fold-test-end", default="2026-01-01")
    parser.add_argument("--horizon", type=int, default=24); parser.add_argument("--window", type=int, default=24); parser.add_argument("--train-stride", type=int, default=8); parser.add_argument("--fixed-graph-type", choices=["identity", "distance", "corr"], default="distance"); parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input"); parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--hidden-dim", type=int, default=64); parser.add_argument("--diffusion-steps", type=int, default=2); parser.add_argument("--gwn-blocks", type=int, default=6); parser.add_argument("--dropout", type=float, default=0.15); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gate-hidden", type=int, default=64); parser.add_argument("--gate-epochs", type=int, default=30); parser.add_argument("--gate-patience", type=int, default=7); parser.add_argument("--gate-lr", type=float, default=0.001); parser.add_argument("--weight-decay", type=float, default=1e-5); parser.add_argument("--gate-smooth-weight", type=float, default=0.02); parser.add_argument("--correction-reg-weight", type=float, default=0.01); parser.add_argument("--physics-correction-reg-weight", type=float, default=0.01); parser.add_argument("--explicit-physics-candidate", action=argparse.BooleanOptionalAction, default=False); parser.add_argument("--grad-clip", type=float, default=1.0); parser.add_argument("--min-delta", type=float, default=1e-5); parser.add_argument("--event-quantile", type=float, default=0.95); parser.add_argument("--cpu-threads", type=int, default=8); parser.add_argument("--print-every", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args(); out = project_path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    data_dir = confirm.configure_data_dir(args.data_dir)
    config = {**vars(args), "data_dir": str(data_dir), "strict_causal_preprocessing": True, "future_residual_used_as_input": False, "validation_locked_gate": True, "untouched_holdout": False}
    (out / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows=[]; gate_rows=[]
    for seed in args.seeds:
        seed_rows, seed_gates = run_seed(seed, args, device); rows.extend(seed_rows); gate_rows.extend(seed_gates)
    summarize(args, rows, gate_rows)


if __name__ == "__main__":
    main()
