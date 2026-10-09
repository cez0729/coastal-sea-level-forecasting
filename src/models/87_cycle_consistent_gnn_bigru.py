from __future__ import annotations

import argparse
import importlib.util
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
SCRIPT78 = Path(__file__).resolve().parent / "78_final_four_models_enhanced_data.py"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "cycle_consistent_gnn_bigru"

spec78 = importlib.util.spec_from_file_location("final_models_impl", SCRIPT78)
final_models = importlib.util.module_from_spec(spec78)
assert spec78.loader is not None
spec78.loader.exec_module(final_models)
v2 = final_models.v2


class CycleConsistentGNNBiGRU(nn.Module):
    """Data-driven forward-backward consistency model.

    Inference uses only the forward forecast. The backward branch is a training
    constraint that maps the predicted future residual trajectory back to the
    latent initial state.
    """

    def __init__(
        self,
        input_dim: int,
        graph_priors: dict[str, np.ndarray],
        graph_init_weights: list[float],
        gnn_hidden: int,
        gru_hidden: int,
        horizon: int,
        dropout: float,
        graph_mode: str,
        fixed_adj: np.ndarray | None = None,
    ):
        super().__init__()
        self.graph_mode = graph_mode
        if graph_mode == "learnable":
            self.graph = v2.LearnableGraphFusion(graph_priors, graph_init_weights)
        elif graph_mode == "fixed":
            if fixed_adj is None:
                raise ValueError("fixed_adj is required when graph_mode='fixed'")
            self.register_buffer("fixed_adj", torch.tensor(fixed_adj, dtype=torch.float32))
        else:
            raise ValueError("graph_mode must be 'fixed' or 'learnable'")

        self.gcn1 = v2.GraphConvolution(input_dim, gnn_hidden)
        self.gcn2 = v2.GraphConvolution(gnn_hidden, gnn_hidden)
        self.norm = nn.LayerNorm(gnn_hidden)
        self.dropout = nn.Dropout(dropout)
        self.forward_gru = nn.GRU(gnn_hidden, gru_hidden, batch_first=True, bidirectional=True)
        latent_dim = gru_hidden * 2
        self.forecast_head = nn.Sequential(
            nn.Linear(latent_dim, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, horizon),
        )

        self.backward_gru = nn.GRU(1, gru_hidden, batch_first=True, bidirectional=True)
        self.backward_head = nn.Sequential(
            nn.Linear(latent_dim, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, latent_dim),
        )

    def adjacency(self):
        if self.graph_mode == "learnable":
            return self.graph()
        return self.fixed_adj

    def forward(self, x):
        adj = self.adjacency()
        h = torch.relu(self.gcn1(x, adj))
        h = torch.relu(self.gcn2(h, adj))
        h = self.dropout(self.norm(h))
        b, t, n, f = h.shape
        h = h.permute(0, 2, 1, 3).reshape(b * n, t, f)
        out, _ = self.forward_gru(h)
        latent = out[:, -1, :]
        pred = self.forecast_head(latent).reshape(b, n, -1)
        return pred, latent.reshape(b, n, -1)

    def backward_reconstruct_latent(self, pred_residual):
        b, n, h = pred_residual.shape
        # Reverse the predicted future trajectory: future -> current latent state.
        seq = torch.flip(pred_residual, dims=[-1]).reshape(b * n, h, 1)
        out, _ = self.backward_gru(seq)
        recon = self.backward_head(out[:, -1, :])
        return recon.reshape(b, n, -1)

    def predict(self, x):
        pred, _ = self.forward(x)
        return pred

    def weight_dict(self):
        if self.graph_mode == "learnable":
            return self.graph.weight_dict()
        return {}


def lead_weighted_mse(pred, target, gamma: float):
    horizon = pred.shape[-1]
    lead = torch.linspace(1.0 / horizon, 1.0, horizon, device=pred.device, dtype=pred.dtype)
    weights = lead.pow(float(gamma)).view(1, 1, horizon)
    return torch.mean((pred - target).pow(2) * weights)


def trend_loss(pred, target):
    if pred.shape[-1] < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return torch.mean((torch.diff(pred, dim=-1) - torch.diff(target, dim=-1)).pow(2))


def cycle_lambda_for_epoch(epoch: int, max_lambda: float, warmup: int, ramp: int) -> float:
    if epoch <= warmup:
        return 0.0
    if ramp <= 0:
        return float(max_lambda)
    frac = min(1.0, max(0.0, (epoch - warmup) / ramp))
    return float(max_lambda) * frac


def evaluate_cycle_loss(model, loader, args, device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred, latent = model(xb)
            recon = model.backward_reconstruct_latent(pred)
            forecast = lead_weighted_mse(pred, yb, args.data_lead_gamma)
            last = torch.mean((pred[:, :, -1] - yb[:, :, -1]).pow(2))
            cycle = torch.mean((recon - latent).pow(2))
            trend = trend_loss(pred, yb)
            parts = {
                "forecast_loss": float(forecast.detach().cpu()),
                "last_loss": float(last.detach().cpu()),
                "cycle_loss": float(cycle.detach().cpu()),
                "trend_loss": float(trend.detach().cpu()),
            }
            for k, val in parts.items():
                totals[k] = totals.get(k, 0.0) + val
            n += 1
    return {k: val / max(1, n) for k, val in totals.items()}


def train_cycle_model(model, train_loader, val_loader, args, device):
    if args.graph_mode == "learnable":
        graph_params = list(model.graph.parameters())
        graph_ids = {id(p) for p in graph_params}
        base_params = [p for p in model.parameters() if id(p) not in graph_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": args.lr},
                {"params": graph_params, "lr": args.lr * args.graph_lr_mult},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    best_state = None
    best_val = float("inf")
    bad = 0
    rows = []
    for epoch in range(1, args.epochs + 1):
        current_cycle = cycle_lambda_for_epoch(epoch, args.cycle_lambda, args.cycle_warmup_epochs, args.cycle_ramp_epochs)
        model.train()
        totals: dict[str, float] = {}
        n = 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred, latent = model(xb)
            recon = model.backward_reconstruct_latent(pred)
            forecast = lead_weighted_mse(pred, yb, args.data_lead_gamma)
            last = torch.mean((pred[:, :, -1] - yb[:, :, -1]).pow(2))
            cycle = torch.mean((recon - latent.detach()).pow(2))
            trend = trend_loss(pred, yb)
            total = forecast + args.last_step_weight * last + current_cycle * cycle + args.trend_lambda * trend
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            parts = {
                "total_loss": float(total.detach().cpu()),
                "forecast_loss": float(forecast.detach().cpu()),
                "last_loss": float(last.detach().cpu()),
                "cycle_loss": float(cycle.detach().cpu()),
                "trend_loss": float(trend.detach().cpu()),
            }
            for k, val in parts.items():
                totals[k] = totals.get(k, 0.0) + val
            n += 1

        train_parts = {k: val / max(1, n) for k, val in totals.items()}
        val_parts = evaluate_cycle_loss(model, val_loader, args, device)
        if args.selection_metric == "val_last_loss":
            val_score = val_parts["last_loss"]
        else:
            val_score = val_parts["forecast_loss"]
        scheduler.step(val_score)

        weights = model.weight_dict()
        row = {
            "epoch": epoch,
            "selection_score": val_score,
            "cycle_lambda": current_cycle,
            **{f"train_{k}": v for k, v in train_parts.items()},
            **{f"val_{k}": v for k, v in val_parts.items()},
            "w_identity": weights.get("identity", np.nan),
            "w_distance": weights.get("distance", np.nan),
            "w_corr": weights.get("corr", np.nan),
        }
        rows.append(row)
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch={epoch:03d} cycle={current_cycle:.4f} "
                f"train_forecast={train_parts['forecast_loss']:.5f} val_forecast={val_parts['forecast_loss']:.5f} "
                f"val_last={val_parts['last_loss']:.5f} val_cycle={val_parts['cycle_loss']:.5f} select={val_score:.5f}"
            )

        if val_score < best_val - args.min_delta:
            best_val = val_score
            bad = 0
            best_state = {k: val.detach().cpu().clone() for k, val in model.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            print(f"Early stopping at epoch {epoch}; best={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(rows), best_val


@torch.no_grad()
def predict_cycle(model, loader, device):
    model.eval()
    preds, ys, tides = [], [], []
    for xb, yb, tb in loader:
        preds.append(model.predict(xb.to(device)).detach().cpu().numpy())
        ys.append(yb.numpy())
        tides.append(tb.numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(tides)


def plot_training_log(history: pd.DataFrame, output_path: Path) -> None:
    if history.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in ["train_forecast_loss", "val_forecast_loss", "val_last_loss", "val_cycle_loss"]:
        if col in history.columns:
            ax.plot(history["epoch"], history[col], label=col)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Cycle-consistent GNN-BiGRU training curves")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_cycle(args, horizon: int, seed: int, device) -> dict[str, float]:
    v2.set_seed(seed)
    data = final_models.build_enhanced_data(args, horizon, add_ode_prior=False)
    train_loader = DataLoader(data["single_train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data["single_val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(data["single_test"], batch_size=args.batch_size, shuffle=False)

    fixed_adj = data["graph_priors"][args.fixed_graph_type]
    model = CycleConsistentGNNBiGRU(
        input_dim=data["feats"],
        graph_priors=data["graph_priors"],
        graph_init_weights=[0.50, 0.35, 0.15],
        gnn_hidden=args.gnn_hidden,
        gru_hidden=args.gru_hidden,
        horizon=horizon,
        dropout=args.dropout,
        graph_mode=args.graph_mode,
        fixed_adj=fixed_adj,
    ).to(device)

    print("\n" + "=" * 90)
    print(
        f"Cycle-consistent run: horizon={horizon} seed={seed} graph={args.graph_mode} "
        f"train={len(data['single_train'])} val={len(data['single_val'])} test={len(data['single_test'])} "
        f"features={data['feats']}"
    )
    history, best_val = train_cycle_model(model, train_loader, val_loader, args, device)
    pred, true, tide = predict_cycle(model, test_loader, device)
    metrics = final_models.summarize_single(true, pred, tide)
    weights = model.weight_dict()
    row = {
        "model_key": f"cycle_consistent_{args.graph_mode}_gnn_bigru",
        "model_name": f"Cycle-consistent {args.graph_mode} GNN-BiGRU",
        "horizon": horizon,
        "seed": seed,
        "best_val_loss": best_val,
        "num_features": data["feats"],
        "cycle_lambda": args.cycle_lambda,
        "trend_lambda": args.trend_lambda,
        "last_step_weight": args.last_step_weight,
        "selection_metric": args.selection_metric,
        "learned_w_identity": weights.get("identity", np.nan),
        "learned_w_distance": weights.get("distance", np.nan),
        "learned_w_corr": weights.get("corr", np.nan),
        **metrics,
    }

    run_dir = Path(args.output_dir) / f"horizon_{horizon}h" / f"seed_{seed}" / f"cycle_{args.graph_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
    np.savez_compressed(run_dir / "predictions.npz", pred_residual=pred, true_residual=true, target_tide=tide)
    plot_training_log(history, run_dir / "training_curves.png")
    print(
        f"Done Cycle {horizon}h seed={seed}: seq_R2={row['seq_residual_R2']:.4f}, "
        f"last_R2={row['last_residual_R2']:.4f}, last_RMSE={row['last_residual_RMSE']:.4f}, "
        f"q95_R2={row['extreme_abs_q95_residual_R2']:.4f}"
    )
    return row


def write_summary(rows: list[dict[str, float]], output_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df.to_csv(output_dir / "cycle_consistent_all_metrics.csv", index=False)
    key_cols = [
        "horizon",
        "seed",
        "model_name",
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "last_sea_level_R2",
    ]
    existing = [c for c in key_cols if c in df.columns]
    df[existing].sort_values(["horizon", "seed"]).to_csv(output_dir / "cycle_consistent_key_metrics.csv", index=False)

    agg_cols = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "last_sea_level_R2",
    ]
    grouped = df.groupby(["horizon", "model_name"])[agg_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join([x for x in col if x]) for col in grouped.columns.to_flat_index()]
    grouped.to_csv(output_dir / "cycle_consistent_mean_std.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = grouped.sort_values("horizon")
    ax.errorbar(
        plot_df["horizon"],
        plot_df["last_residual_R2_mean"],
        yerr=plot_df["last_residual_R2_std"].fillna(0.0),
        marker="o",
        capsize=4,
        label="Last residual R2",
    )
    ax.errorbar(
        plot_df["horizon"],
        plot_df["extreme_abs_q95_residual_R2_mean"],
        yerr=plot_df["extreme_abs_q95_residual_R2_std"].fillna(0.0),
        marker="s",
        capsize=4,
        label="Extreme q95 residual R2",
    )
    ax.set_xlabel("Horizon (hours)")
    ax.set_ylabel("R2")
    ax.set_title("Cycle-consistent GNN-BiGRU multi-seed summary")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cycle_consistent_r2_summary.png", dpi=220)
    plt.close(fig)

    lines = [
        "# Cycle-Consistent GNN-BiGRU Summary",
        "",
        "This experiment evaluates a pure data-driven forward-backward consistency model.",
        "The backward branch is used only during training; inference uses the forward forecast only.",
        "",
        "## Mean/Std Results",
        "",
        grouped.to_string(index=False),
        "",
        "## Interpretation",
        "",
        "- Compare this model mainly against GNN-BiGRU and Physical-loss GNN-BiGRU at 24h.",
        "- If it improves 24h last-step R2, it supports the idea that trajectory consistency helps reduce long-horizon drift.",
        "- If it does not outperform Physical-loss, it is still useful as a pure data-driven consistency baseline.",
        "- Because validation selection uses forecast/last-step loss, early stopping helps control overfitting from the cycle branch.",
    ]
    (output_dir / "cycle_consistent_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cycle-consistent GNN-BiGRU experiments")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--horizons", type=int, nargs="+", default=[24])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--graph-mode", choices=["fixed", "learnable"], default="learnable")
    parser.add_argument("--fixed-graph-type", default="distance", choices=["identity", "distance", "corr"])
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gnn-hidden", type=int, default=48)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--graph-lr-mult", type=float, default=0.35)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--cycle-lambda", type=float, default=0.04)
    parser.add_argument("--cycle-warmup-epochs", type=int, default=8)
    parser.add_argument("--cycle-ramp-epochs", type=int, default=14)
    parser.add_argument("--trend-lambda", type=float, default=0.15)
    parser.add_argument("--last-step-weight", type=float, default=0.35)
    parser.add_argument("--data-lead-gamma", type=float, default=1.3)
    parser.add_argument("--selection-metric", choices=["val_forecast_loss", "val_last_loss"], default="val_forecast_loss")
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--physics-forcing-mode", default="future")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows = []
    for horizon in args.horizons:
        for seed in args.seeds:
            row = run_cycle(args, horizon, seed, device)
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / "cycle_consistent_metrics_partial.csv", index=False)
    write_summary(rows, output_dir)
    key = [
        "horizon",
        "seed",
        "model_name",
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "last_sea_level_R2",
    ]
    existing = [c for c in key if c in pd.DataFrame(rows).columns]
    print("\nFinished Cycle-consistent experiments.")
    print(pd.DataFrame(rows)[existing].sort_values(["horizon", "seed"]).to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
