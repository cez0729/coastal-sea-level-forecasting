from __future__ import annotations

import argparse
import copy
import importlib.util
import json
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
DEFAULT_OUT = ROOT / "results" / "adaptive_residual_gwn_adapter"
SOURCE_RESULTS = ROOT / "results" / "priority12_physics_graph_wavenet"
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p104 = load_module("priority2_gwn_p107", HERE / "104_priority2_physics_graph_wavenet.py")
final4 = p104.final4
priority1 = p104.priority1
v2 = p104.v2
v3 = p104.v3


def row_normalize(adjacency: torch.Tensor) -> torch.Tensor:
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class AdaptiveDirectedGraph(nn.Module):
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
        score_forward = torch.tanh(self.alpha * (self.source @ self.target.T))
        score_reverse = torch.tanh(self.alpha * (self.target @ self.source.T))
        score = torch.relu(score_forward - score_reverse)
        score = score + torch.eye(self.nodes, dtype=score.dtype, device=score.device) * 1e-3
        if self.top_k < self.nodes:
            indices = torch.topk(score, self.top_k, dim=-1).indices
            mask = torch.zeros_like(score).scatter_(1, indices, 1.0)
            score = score * mask
        return row_normalize(score)


class AdaptiveDiffusionAdapter(nn.Module):
    def __init__(self, channels: int, depth: int, dropout: float):
        super().__init__()
        self.depth = int(depth)
        self.project = nn.Conv2d((depth + 1) * channels, channels, kernel_size=(1, 1))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.project.weight, gain=0.1)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        states = [x]
        propagated = x
        for _ in range(self.depth):
            propagated = torch.einsum("nm,bcmt->bcnt", adjacency, propagated)
            states.append(propagated)
        return self.dropout(self.project(torch.cat(states, dim=1)))


class AdaptiveResidualBlock(nn.Module):
    def __init__(self, base_block: nn.Module, channels: int, adapter_depth: int, dropout: float):
        super().__init__()
        self.dilation = base_block.dilation
        self.kernel_size = base_block.kernel_size
        self.filter_conv = copy.deepcopy(base_block.filter_conv)
        self.gate_conv = copy.deepcopy(base_block.gate_conv)
        self.base_graph = copy.deepcopy(base_block.graph)
        self.norm = copy.deepcopy(base_block.norm)
        self.dropout = copy.deepcopy(base_block.dropout)
        self.adapter = AdaptiveDiffusionAdapter(channels, adapter_depth, dropout)
        self.adapter_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, adaptive_adjacency: torch.Tensor) -> torch.Tensor:
        residual = x
        filtered = torch.tanh(self.filter_conv(x)[..., : x.shape[-1]])
        gated = torch.sigmoid(self.gate_conv(x)[..., : x.shape[-1]])
        temporal = filtered * gated
        batch, channels, nodes, steps = temporal.shape
        temporal_nodes = temporal.permute(0, 3, 2, 1).reshape(batch * steps, nodes, channels)
        base = self.base_graph(temporal_nodes).reshape(batch, steps, nodes, channels).permute(0, 3, 2, 1)
        adaptive = self.adapter(temporal, adaptive_adjacency)
        graph = base + torch.tanh(self.adapter_scale) * adaptive
        return self.norm(self.dropout(graph) + residual)


class AdaptiveResidualGraphWaveNet(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        nodes: int,
        channels: int,
        embedding_dim: int,
        top_k: int,
        adapter_depth: int,
        dropout: float,
    ):
        super().__init__()
        self.horizon = base_model.horizon
        self.num_states = base_model.num_states
        self.input_proj = copy.deepcopy(base_model.input_proj)
        self.blocks = nn.ModuleList(
            [
                AdaptiveResidualBlock(block, channels, adapter_depth, dropout)
                for block in base_model.blocks
            ]
        )
        self.head = copy.deepcopy(base_model.head)
        self.register_buffer("fixed_adjacency", base_model.fixed_adjacency.detach().clone())
        self.graph_learner = AdaptiveDirectedGraph(nodes, embedding_dim, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x.permute(0, 3, 2, 1))
        adaptive_adjacency = self.graph_learner()
        for block in self.blocks:
            hidden = block(hidden, adaptive_adjacency)
        output = self.head(hidden)[..., -1].permute(0, 2, 1)
        return output.reshape(x.shape[0], x.shape[2], self.horizon, self.num_states)

    def adapter_parameters(self):
        modules = [self.graph_learner]
        modules.extend(block.adapter for block in self.blocks)
        parameters = [parameter for module in modules for parameter in module.parameters()]
        parameters.extend(block.adapter_scale for block in self.blocks)
        return parameters

    def adapter_scales(self) -> list[float]:
        return [float(torch.tanh(block.adapter_scale.detach()).cpu()) for block in self.blocks]

    def adaptive_adjacency(self) -> np.ndarray:
        return self.graph_learner().detach().cpu().numpy()


def build_model(seed: int, data: dict, args, device: torch.device) -> AdaptiveResidualGraphWaveNet:
    adjacency = data["graph_priors"][args.fixed_graph_type]
    base = p104.GraphWaveNetMultistate(
        data["feats"],
        adjacency,
        args.hidden_dim,
        args.horizon,
        4,
        args.diffusion_steps,
        args.gwn_blocks,
        args.dropout,
    )
    checkpoint = (
        Path(args.source_results)
        / f"seed_{seed}"
        / f"horizon_{args.horizon}h"
        / "gwn_multistate_no_physics"
        / "best_checkpoint.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base.load_state_dict(payload["model_state_dict"])
    model = AdaptiveResidualGraphWaveNet(
        base,
        data["nodes"],
        args.hidden_dim,
        args.node_embedding_dim,
        args.adaptive_top_k,
        args.adapter_depth,
        args.dropout,
    )
    return model.to(device)


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def set_trainable_parameters(
    model: AdaptiveResidualGraphWaveNet, adapter_only: bool, train_head: bool
) -> None:
    if not adapter_only:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.adapter_parameters():
        parameter.requires_grad_(True)
    if train_head:
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)


def train_adapter(model, physics_ode, data, train_loader, val_loader, args, device, run_dir: Path):
    criterion = p104.make_multistate_criterion(physics_ode, data, args, False, args.horizon).to(device)
    set_trainable_parameters(model, args.adapter_only, args.train_head)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    initial = p104.evaluate_multistate(model, criterion, val_loader, model.fixed_adjacency, 0.0, device)
    best_val = float(initial["eta_data_loss"])
    best_state = clone_state(model)
    bad_epochs = 0
    rows = [
        {
            "epoch": 0,
            "selection_score": best_val,
            **{f"val_{key}": value for key, value in initial.items()},
            "lr": args.lr,
        }
    ]
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        if args.adapter_only:
            for block in model.blocks:
                block.norm.eval()
        totals = {}
        batches = 0
        for xb, target, _, init_states, phys_seq in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb.to(device))
            loss, parts = criterion(
                prediction,
                target.to(device),
                init_states.to(device),
                phys_seq.to(device),
                model.fixed_adjacency,
                0.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
        train_parts = {key: value / max(1, batches) for key, value in totals.items()}
        validation = p104.evaluate_multistate(
            model, criterion, val_loader, model.fixed_adjacency, 0.0, device
        )
        score = float(validation["eta_data_loss"])
        scheduler.step(score)
        rows.append(
            {
                "epoch": epoch,
                "selection_score": score,
                **{f"train_{key}": value for key, value in train_parts.items()},
                **{f"val_{key}": value for key, value in validation.items()},
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_start,
                **{f"adapter_scale_{index}": value for index, value in enumerate(model.adapter_scales())},
            }
        )
        if score < best_val - args.min_delta:
            best_val = score
            best_state = clone_state(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_state_dict": best_state,
                "best_val": best_val,
                "optimizer_state_dict": optimizer.state_dict(),
                "history": rows,
            },
            run_dir / "last_epoch_checkpoint.pt",
        )
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch={epoch:03d} train_eta={train_parts['eta_data_loss']:.6f} "
                f"val_eta={score:.6f} best={best_val:.6f} scales={model.adapter_scales()}"
            )
        if bad_epochs >= args.patience:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows), best_val, time.perf_counter() - start


def metrics_for_predictions(true_states, pred_states, tide, data, args) -> dict:
    metrics = v2.summarize_metrics(true_states, pred_states, tide)
    metrics.update(v3.summarize_extreme_metrics(true_states, pred_states))
    train_end = int(len(data["arrays"]["residual"]) * args.train_ratio)
    thresholds = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    metrics.update(p104.operational_event_metrics(true_states[..., 0], pred_states[..., 0], thresholds))
    return metrics


def run_seed(seed: int, args, device: torch.device) -> dict:
    run_dir = Path(args.output_dir) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    if args.resume and metrics_path.exists() and (run_dir / "predictions.npz").exists():
        return pd.read_csv(metrics_path).iloc[0].to_dict()
    p104.set_reproducible(seed, args.cpu_threads)
    data = final4.build_enhanced_data(args, args.horizon, add_ode_prior=False)
    model = build_model(seed, data, args, device)
    physics_ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    train_loader = p104.make_loader(data["multi_train"], args, True, seed)
    val_loader = p104.make_loader(data["multi_val"], args, False, seed)
    test_loader = p104.make_loader(data["multi_test"], args, False, seed)
    history, best_val, training_seconds = train_adapter(
        model, physics_ode, data, train_loader, val_loader, args, device, run_dir
    )
    pred, true, tide, inference_seconds = p104.predict_multistate(model, test_loader, device)
    metrics = metrics_for_predictions(true, pred, tide, data, args)
    source_path = (
        Path(args.source_results)
        / f"seed_{seed}"
        / f"horizon_{args.horizon}h"
        / "gwn_multistate_no_physics"
        / "metrics.csv"
    )
    source = pd.read_csv(source_path).iloc[0].to_dict()
    row = {
        "seed": seed,
        "model": "adaptive_residual_graph_wavenet_adapter",
        "best_val_eta_loss": best_val,
        "training_seconds": training_seconds,
        "epochs_completed": int(history["epoch"].max()),
        "inference_seconds": inference_seconds,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": p104.count_parameters(model),
        "adapter_scales": json.dumps(model.adapter_scales()),
        **metrics,
        **{f"source_{key}": source[key] for key in [
            "seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2"
        ]},
    }
    history.to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    pd.DataFrame(model.adaptive_adjacency(), index=v2.STATION_IDS, columns=v2.STATION_IDS).to_csv(
        run_dir / "learned_adaptive_adjacency.csv"
    )
    np.savez_compressed(
        run_dir / "predictions.npz",
        pred_states=pred,
        true_states=true,
        target_tide=tide,
        station_ids=np.asarray(v2.STATION_IDS),
    )
    print(
        f"seed={seed} seq={row['seq_residual_R2']:.4f} last={row['last_residual_R2']:.4f} "
        f"source_seq={row['source_seq_residual_R2']:.4f} source_last={row['source_last_residual_R2']:.4f}"
    )
    return row


def merge_results(args) -> None:
    output_dir = Path(args.output_dir)
    files = sorted(output_dir.glob("seed_*/metrics.csv"))
    if not files:
        raise RuntimeError(f"No completed metrics under {output_dir}")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    data = data[data["seed"].isin(args.seeds)].sort_values("seed").drop_duplicates("seed", keep="last")
    data.to_csv(output_dir / "all_runs.csv", index=False)
    metrics = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2"]
    summary_rows = []
    paired_rows = []
    for metric in metrics:
        source = f"source_{metric}"
        improvement = data[metric] - data[source]
        if metric.endswith("RMSE"):
            improvement = -improvement
        summary_rows.append(
            {
                "metric": metric,
                "adapter_mean": float(data[metric].mean()),
                "adapter_std": float(data[metric].std()),
                "source_mean": float(data[source].mean()),
                "source_std": float(data[source].std()),
                "mean_improvement": float(improvement.mean()),
                "std_improvement": float(improvement.std()),
                "wins": int((improvement > 0).sum()),
                "count": int(improvement.notna().sum()),
                "wilcoxon_greater_p": p104.exact_wilcoxon_greater(improvement.to_numpy()),
            }
        )
        for seed, value in zip(data["seed"], improvement):
            paired_rows.append({"seed": seed, "metric": metric, "improvement": value})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "mean_std_and_paired_tests.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(output_dir / "paired_seed_deltas.csv", index=False)
    plot_summary(summary, output_dir / "adaptive_residual_gwn_comparison.png")
    print(summary.to_string(index=False))


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    selected = summary[summary["metric"].isin(["seq_residual_R2", "last_residual_R2", "extreme_abs_q95_residual_R2"])]
    x = np.arange(len(selected))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(x - width / 2, selected["source_mean"], width, label="Multistate GWN")
    axis.bar(x + width / 2, selected["adapter_mean"], width, label="Adaptive residual GWN")
    axis.set_xticks(x, ["Trajectory R2", "24-h terminal R2", "Descriptive q95 R2"])
    axis.set_ylabel("Residual R2")
    axis.set_title("Function-preserving adaptive graph adapter")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Function-preserving adaptive graph residual adapter for GWN.")
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--source-results", default=str(SOURCE_RESULTS))
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
    parser.add_argument("--node-embedding-dim", type=int, default=8)
    parser.add_argument("--adaptive-top-k", type=int, default=4)
    parser.add_argument("--adapter-depth", type=int, default=2)
    parser.add_argument("--adapter-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--physics-lambda", type=float, default=0.0)
    parser.add_argument("--physics-lr-mult", type=float, default=0.5)
    parser.add_argument("--physics-warmup-epochs", type=int, default=8)
    parser.add_argument("--physics-ramp-epochs", type=int, default=14)
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.20)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
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
    if args.mode == "merge":
        merge_results(args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; seeds={args.seeds}; adapter_only={args.adapter_only}")
    for seed in args.seeds:
        run_seed(seed, args, device)
    merge_results(args)


if __name__ == "__main__":
    main()
