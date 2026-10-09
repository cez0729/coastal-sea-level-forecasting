from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results" / "adaptive_multiscale_gwn"
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p104 = load_module("priority2_gwn_p105", HERE / "104_priority2_physics_graph_wavenet.py")
final4 = p104.final4
priority1 = p104.priority1
v2 = p104.v2
v3 = p104.v3


CANDIDATES = {
    "fixed_multiscale_ms": {"adaptive": False, "kernels": (2, 3, 6)},
    "adaptive_single_ms": {"adaptive": True, "kernels": (2,)},
    "adaptive_multiscale_ms": {"adaptive": True, "kernels": (2, 3, 6)},
}


def normalize_rows(adjacency: torch.Tensor) -> torch.Tensor:
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class AdaptiveDirectedGraph(nn.Module):
    """MTGNN-style sparse directed adjacency learned from node embeddings."""

    def __init__(self, nodes: int, embedding_dim: int, top_k: int, alpha: float = 3.0):
        super().__init__()
        self.nodes = int(nodes)
        self.top_k = min(int(top_k), self.nodes)
        self.alpha = float(alpha)
        self.source = nn.Parameter(torch.empty(nodes, embedding_dim))
        self.target = nn.Parameter(torch.empty(nodes, embedding_dim))
        nn.init.xavier_uniform_(self.source)
        nn.init.xavier_uniform_(self.target)

    def forward(self) -> torch.Tensor:
        forward_score = torch.tanh(self.alpha * (self.source @ self.target.T))
        reverse_score = torch.tanh(self.alpha * (self.target @ self.source.T))
        score = torch.relu(forward_score - reverse_score)
        score = score + torch.eye(self.nodes, dtype=score.dtype, device=score.device) * 1e-3
        if self.top_k < self.nodes:
            indices = torch.topk(score, self.top_k, dim=-1).indices
            mask = torch.zeros_like(score).scatter_(1, indices, 1.0)
            score = score * mask
        return normalize_rows(score)


class MixHopPropagation(nn.Module):
    def __init__(self, channels: int, depth: int, retain: float, dropout: float):
        super().__init__()
        self.depth = int(depth)
        self.retain = float(retain)
        self.project = nn.Conv2d((self.depth + 1) * channels, channels, kernel_size=(1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = normalize_rows(adjacency)
        states = [x]
        propagated = x
        for _ in range(self.depth):
            neighbor = torch.einsum("nm,bcmt->bcnt", adjacency, propagated)
            propagated = self.retain * x + (1.0 - self.retain) * neighbor
            states.append(propagated)
        return self.dropout(self.project(torch.cat(states, dim=1)))


class CausalGatedInception(nn.Module):
    def __init__(self, channels: int, kernels: tuple[int, ...], dilation: int):
        super().__init__()
        self.kernels = tuple(int(k) for k in kernels)
        self.dilation = int(dilation)
        self.filters = nn.ModuleList()
        self.gates = nn.ModuleList()
        for kernel in self.kernels:
            padding = (kernel - 1) * self.dilation
            kwargs = dict(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, kernel),
                dilation=(1, self.dilation),
                padding=(0, padding),
            )
            self.filters.append(nn.Conv2d(**kwargs))
            self.gates.append(nn.Conv2d(**kwargs))
        self.branch_logits = nn.Parameter(torch.zeros(len(self.kernels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = []
        for filter_conv, gate_conv in zip(self.filters, self.gates):
            filtered = torch.tanh(filter_conv(x)[..., : x.shape[-1]])
            gated = torch.sigmoid(gate_conv(x)[..., : x.shape[-1]])
            branches.append(filtered * gated)
        weights = torch.softmax(self.branch_logits, dim=0)
        return sum(weight * branch for weight, branch in zip(weights, branches))


class AdaptiveMultiScaleBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        skip_channels: int,
        kernels: tuple[int, ...],
        dilation: int,
        mix_hops: int,
        mix_retain: float,
        dropout: float,
        use_adaptive: bool,
    ):
        super().__init__()
        self.use_adaptive = bool(use_adaptive)
        self.temporal = CausalGatedInception(channels, kernels, dilation)
        self.fixed_graph = MixHopPropagation(channels, mix_hops, mix_retain, dropout)
        self.adaptive_graph = MixHopPropagation(channels, mix_hops, mix_retain, dropout) if use_adaptive else None
        support_count = 2 if use_adaptive else 1
        self.support_logits = nn.Parameter(torch.zeros(support_count))
        self.residual = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.skip = nn.Conv2d(channels, skip_channels, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        fixed_adjacency: torch.Tensor,
        adaptive_adjacency: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = self.temporal(x)
        graph_states = [self.fixed_graph(temporal, fixed_adjacency)]
        if self.use_adaptive:
            assert self.adaptive_graph is not None and adaptive_adjacency is not None
            graph_states.append(self.adaptive_graph(temporal, adaptive_adjacency))
        weights = torch.softmax(self.support_logits, dim=0)
        graph = sum(weight * state for weight, state in zip(weights, graph_states))
        graph = self.dropout(graph)
        output = self.norm(self.residual(x) + graph)
        return output, self.skip(graph)


class AdaptiveMultiScaleGraphWaveNet(nn.Module):
    """Static coastal prior plus adaptive graph and multi-scale gated temporal blocks."""

    def __init__(
        self,
        input_dim: int,
        adjacency: np.ndarray,
        hidden_dim: int,
        skip_dim: int,
        horizon: int,
        num_states: int,
        blocks: int,
        kernels: tuple[int, ...],
        adaptive: bool,
        node_embedding_dim: int,
        adaptive_top_k: int,
        mix_hops: int,
        mix_retain: float,
        dropout: float,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.num_states = int(num_states)
        self.use_adaptive = bool(adaptive)
        self.register_buffer("fixed_adjacency", torch.tensor(adjacency, dtype=torch.float32))
        self.graph_learner = (
            AdaptiveDirectedGraph(adjacency.shape[0], node_embedding_dim, adaptive_top_k)
            if self.use_adaptive
            else None
        )
        self.input_projection = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
        dilations = [2 ** (index % 4) for index in range(blocks)]
        self.blocks = nn.ModuleList(
            [
                AdaptiveMultiScaleBlock(
                    hidden_dim,
                    skip_dim,
                    kernels,
                    dilation,
                    mix_hops,
                    mix_retain,
                    dropout,
                    self.use_adaptive,
                )
                for dilation in dilations
            ]
        )
        self.end = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(skip_dim, skip_dim, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(skip_dim, horizon * num_states, kernel_size=(1, 1)),
        )

    def adaptive_adjacency(self) -> torch.Tensor | None:
        return self.graph_learner() if self.graph_learner is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x.permute(0, 3, 2, 1))
        adaptive = self.adaptive_adjacency()
        skip_total = None
        for block in self.blocks:
            hidden, skip = block(hidden, self.fixed_adjacency, adaptive)
            skip_total = skip if skip_total is None else skip_total + skip
        assert skip_total is not None
        output = self.end(skip_total)[..., -1].permute(0, 2, 1)
        return output.reshape(x.shape[0], x.shape[2], self.horizon, self.num_states)

    def graph_diagnostics(self) -> dict[str, np.ndarray | list[float]]:
        adaptive = self.adaptive_adjacency()
        support_weights = [
            torch.softmax(block.support_logits.detach(), dim=0).cpu().numpy().tolist() for block in self.blocks
        ]
        branch_weights = [
            torch.softmax(block.temporal.branch_logits.detach(), dim=0).cpu().numpy().tolist()
            for block in self.blocks
        ]
        return {
            "adaptive_adjacency": adaptive.detach().cpu().numpy() if adaptive is not None else np.empty((0, 0)),
            "support_weights": support_weights,
            "temporal_branch_weights": branch_weights,
        }


def candidate_model(name: str, data: dict, args, device: torch.device) -> AdaptiveMultiScaleGraphWaveNet:
    if name not in CANDIDATES:
        raise ValueError(f"Unknown candidate {name}; choices are {sorted(CANDIDATES)}")
    config = CANDIDATES[name]
    adjacency = data["graph_priors"][args.fixed_graph_type]
    return AdaptiveMultiScaleGraphWaveNet(
        input_dim=data["feats"],
        adjacency=adjacency,
        hidden_dim=args.hidden_dim,
        skip_dim=args.skip_dim,
        horizon=args.horizon,
        num_states=4,
        blocks=args.blocks,
        kernels=config["kernels"],
        adaptive=config["adaptive"],
        node_embedding_dim=args.node_embedding_dim,
        adaptive_top_k=args.adaptive_top_k,
        mix_hops=args.mix_hops,
        mix_retain=args.mix_retain,
        dropout=args.dropout,
    ).to(device)


def evaluate_predictions(true_states: np.ndarray, pred_states: np.ndarray, tide: np.ndarray, data: dict, args) -> dict:
    metrics = v2.summarize_metrics(true_states, pred_states, tide)
    metrics.update(v3.summarize_extreme_metrics(true_states, pred_states))
    train_end = int(len(data["arrays"]["residual"]) * args.train_ratio)
    thresholds = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    metrics.update(p104.operational_event_metrics(true_states[..., 0], pred_states[..., 0], thresholds))
    return metrics


def save_diagnostics(model: AdaptiveMultiScaleGraphWaveNet, run_dir: Path) -> None:
    diagnostics = model.graph_diagnostics()
    adaptive = diagnostics.pop("adaptive_adjacency")
    if isinstance(adaptive, np.ndarray) and adaptive.size:
        pd.DataFrame(adaptive, index=v2.STATION_IDS, columns=v2.STATION_IDS).to_csv(
            run_dir / "learned_adaptive_adjacency.csv"
        )
    (run_dir / "learned_fusion_weights.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )


def run_candidate(name: str, seed: int, args, data: dict, device: torch.device) -> dict:
    run_dir = Path(args.output_dir) / args.stage / f"seed_{seed}" / name
    metrics_name = "validation_metrics.csv" if args.stage == "screen" else "metrics.csv"
    metrics_path = run_dir / metrics_name
    if args.resume and metrics_path.exists():
        print(f"Skipping complete candidate={name} seed={seed}")
        return pd.read_csv(metrics_path).iloc[0].to_dict()
    run_dir.mkdir(parents=True, exist_ok=True)
    p104.set_reproducible(seed, args.cpu_threads)
    model = candidate_model(name, data, args, device)
    physics_ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    train_loader = p104.make_loader(data["multi_train"], args, True, seed)
    val_loader = p104.make_loader(data["multi_val"], args, False, seed)
    history, best_val, timing = p104.train_multistate(
        model,
        physics_ode,
        data,
        train_loader,
        val_loader,
        args,
        device,
        use_physics=False,
        run_dir=run_dir,
    )
    val_pred, val_true, val_tide, val_seconds = p104.predict_multistate(model, val_loader, device)
    val_metrics = evaluate_predictions(val_true, val_pred, val_tide, data, args)
    row = {
        "stage": args.stage,
        "seed": seed,
        "candidate": name,
        "best_val_eta_loss": best_val,
        "predictor_trainable_parameters": p104.count_parameters(model),
        "training_seconds": timing["training_seconds"],
        "epochs_completed": timing["epochs_completed"],
        "validation_inference_seconds": val_seconds,
        **{f"val_{key}": value for key, value in val_metrics.items()},
    }
    row["val_composite_R2"] = (
        row["val_seq_residual_R2"] + 0.5 * row["val_last_residual_R2"]
    )
    history.to_csv(run_dir / "training_log.csv", index=False)
    save_diagnostics(model, run_dir)
    if args.stage == "screen":
        pd.DataFrame([row]).to_csv(metrics_path, index=False)
        print(
            f"Screen {name}: val_seq={row['val_seq_residual_R2']:.4f} "
            f"val_last={row['val_last_residual_R2']:.4f} composite={row['val_composite_R2']:.4f}"
        )
        return row

    test_loader = p104.make_loader(data["multi_test"], args, False, seed)
    pred, true, tide, test_seconds = p104.predict_multistate(model, test_loader, device)
    metrics = evaluate_predictions(true, pred, tide, data, args)
    row.update({"test_inference_seconds": test_seconds, **metrics})
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    np.savez_compressed(
        run_dir / "predictions.npz",
        pred_states=pred,
        true_states=true,
        target_tide=tide,
        station_ids=np.asarray(v2.STATION_IDS),
    )
    (run_dir / "COMPLETE.json").write_text(
        json.dumps({"candidate": name, "seed": seed, "completed": True}, indent=2), encoding="utf-8"
    )
    print(
        f"Final {name} seed={seed}: seq={row['seq_residual_R2']:.4f} "
        f"last={row['last_residual_R2']:.4f} q95={row['extreme_abs_q95_residual_R2']:.4f}"
    )
    return row


def merge_results(args) -> None:
    output_dir = Path(args.output_dir)
    pattern = "screen/seed_*/**/validation_metrics.csv" if args.stage == "screen" else "final/seed_*/**/metrics.csv"
    files = sorted(output_dir.glob(pattern))
    if not files:
        raise RuntimeError(f"No result files match {pattern} under {output_dir}")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    data = data[data["candidate"].isin(args.candidates) & data["seed"].isin(args.seeds)].copy()
    data = data.sort_values(["candidate", "seed"]).drop_duplicates(["candidate", "seed"], keep="last")
    data.to_csv(output_dir / f"{args.stage}_all_runs.csv", index=False)
    metric_prefix = "val_" if args.stage == "screen" else ""
    metrics = [
        f"{metric_prefix}seq_residual_R2",
        f"{metric_prefix}last_residual_R2",
        f"{metric_prefix}last_residual_RMSE",
        f"{metric_prefix}extreme_abs_q95_residual_R2",
        f"{metric_prefix}event_PR_AUC",
        f"{metric_prefix}event_recall",
        "val_composite_R2",
        "training_seconds",
        "epochs_completed",
    ]
    metrics = [metric for metric in metrics if metric in data]
    summary = data.groupby("candidate")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(part) for part in col if part) for col in summary.columns.to_flat_index()]
    summary.to_csv(output_dir / f"{args.stage}_mean_std.csv", index=False)
    print(summary.to_string(index=False))
    if args.stage == "final":
        add_baseline_comparison(data, output_dir)


def add_baseline_comparison(candidate_data: pd.DataFrame, output_dir: Path) -> None:
    baseline_rows = []
    for seed in SEEDS:
        path = ROOT / "results" / "priority1_graph_baselines" / f"seed_{seed}" / "horizon_24h" / "graph_wavenet" / "metrics.csv"
        if path.exists():
            row = pd.read_csv(path).iloc[0].to_dict()
            row.update({"seed": seed, "candidate": "fixed_support_graph_wavenet"})
            baseline_rows.append(row)
    if not baseline_rows:
        return
    combined = pd.concat([candidate_data, pd.DataFrame(baseline_rows)], ignore_index=True, sort=False)
    comparison_metrics = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
    ]
    summary = combined.groupby("candidate")[comparison_metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(part) for part in col if part) for col in summary.columns.to_flat_index()]
    summary.to_csv(output_dir / "final_vs_graph_wavenet_mean_std.csv", index=False)
    plot_comparison(summary, output_dir / "final_vs_graph_wavenet.png")


def plot_comparison(summary: pd.DataFrame, path: Path) -> None:
    metrics = ["seq_residual_R2_mean", "last_residual_R2_mean", "extreme_abs_q95_residual_R2_mean"]
    labels = ["Trajectory R2", "24-h terminal R2", "Descriptive q95 R2"]
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(summary))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, row in summary.iterrows():
        values = [row[metric] for metric in metrics]
        offset = (index - (len(summary) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=row["candidate"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Residual R2")
    ax.set_title("Adaptive multi-scale candidates versus fixed-support Graph WaveNet")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive multi-scale Graph WaveNet research experiment.")
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    parser.add_argument("--stage", choices=["screen", "final"], default="screen")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--candidates", nargs="+", choices=sorted(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", choices=["identity", "distance", "corr"], default="distance")
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--skip-dim", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--node-embedding-dim", type=int, default=12)
    parser.add_argument("--adaptive-top-k", type=int, default=4)
    parser.add_argument("--mix-hops", type=int, default=2)
    parser.add_argument("--mix-retain", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--physics-lr-mult", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--physics-lambda", type=float, default=0.0)
    parser.add_argument("--physics-warmup-epochs", type=int, default=8)
    parser.add_argument("--physics-ramp-epochs", type=int, default=14)
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.20)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--epoch-checkpoint-every", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"config_{args.stage}.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if args.mode == "merge":
        merge_results(args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; stage={args.stage}; candidates={args.candidates}; seeds={args.seeds}")
    rows = []
    for seed in args.seeds:
        data = final4.build_enhanced_data(args, args.horizon, add_ode_prior=False)
        for name in args.candidates:
            start = time.perf_counter()
            row = run_candidate(name, seed, args, data, device)
            row["wall_seconds"] = time.perf_counter() - start
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / f"{args.stage}_partial.csv", index=False)
    merge_results(args)


if __name__ == "__main__":
    main()
