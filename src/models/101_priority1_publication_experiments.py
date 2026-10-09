from __future__ import annotations

import argparse
import importlib.util
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCRIPT78 = Path(__file__).resolve().parent / "78_final_four_models_enhanced_data.py"
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "priority1_publication_experiments"

spec78 = importlib.util.spec_from_file_location("final4_impl", SCRIPT78)
final4 = importlib.util.module_from_spec(spec78)
assert spec78.loader is not None
spec78.loader.exec_module(final4)
v2 = final4.v2
v3 = final4.v3


def make_diffusion_supports(adj: np.ndarray, k_steps: int) -> list[np.ndarray]:
    base = adj.astype(np.float32)
    eye = np.eye(base.shape[0], dtype=np.float32)
    supports = [eye]
    power = base.copy()
    for _ in range(max(1, k_steps)):
        supports.append(power.astype(np.float32))
        power = power @ base
    return supports


class DiffusionGraphLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, supports: list[np.ndarray], bias: bool = True):
        super().__init__()
        stack = np.stack(supports, axis=0).astype(np.float32)
        self.register_buffer("supports", torch.tensor(stack, dtype=torch.float32))
        self.linear = nn.Linear(in_dim * len(supports), out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F]
        propagated = torch.einsum("kij,bjf->bkif", self.supports, x)
        propagated = propagated.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.linear(propagated)


class DCRNNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, supports: list[np.ndarray]):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gate = DiffusionGraphLinear(input_dim + hidden_dim, 2 * hidden_dim, supports)
        self.update = DiffusionGraphLinear(input_dim + hidden_dim, hidden_dim, supports)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=-1)
        z_r = torch.sigmoid(self.gate(combined))
        z, r = torch.chunk(z_r, 2, dim=-1)
        candidate = torch.tanh(self.update(torch.cat([x, r * h], dim=-1)))
        return (1.0 - z) * h + z * candidate


class DCRNNForecaster(nn.Module):
    def __init__(self, input_dim: int, adj: np.ndarray, hidden_dim: int, horizon: int, diffusion_steps: int, dropout: float):
        super().__init__()
        supports = make_diffusion_supports(adj, diffusion_steps)
        self.cell = DCRNNCell(input_dim, hidden_dim, supports)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F]
        batch, _, nodes, _ = x.shape
        h = torch.zeros(batch, nodes, self.cell.hidden_dim, dtype=x.dtype, device=x.device)
        for step in range(x.shape[1]):
            h = self.cell(x[:, step], h)
        h = self.dropout(self.norm(h))
        return self.head(h)


class GraphWaveNetBlock(nn.Module):
    def __init__(self, channels: int, supports: list[np.ndarray], dilation: int, kernel_size: int, dropout: float):
        super().__init__()
        self.dilation = int(dilation)
        self.kernel_size = int(kernel_size)
        pad = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), dilation=(1, dilation), padding=(0, pad))
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), dilation=(1, dilation), padding=(0, pad))
        self.graph = DiffusionGraphLinear(channels, channels, supports)
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, T]
        residual = x
        filt = torch.tanh(self.filter_conv(x)[..., : x.shape[-1]])
        gate = torch.sigmoid(self.gate_conv(x)[..., : x.shape[-1]])
        h = filt * gate
        b, c, n, t = h.shape
        h_nodes = h.permute(0, 3, 2, 1).reshape(b * t, n, c)
        h_nodes = self.graph(h_nodes).reshape(b, t, n, c).permute(0, 3, 2, 1)
        h = self.dropout(h_nodes)
        return self.norm(h + residual)


class GraphWaveNetForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        adj: np.ndarray,
        hidden_dim: int,
        horizon: int,
        diffusion_steps: int,
        blocks: int,
        dropout: float,
    ):
        super().__init__()
        supports = make_diffusion_supports(adj, diffusion_steps)
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
        dilations = [2 ** (i % 4) for i in range(blocks)]
        self.blocks = nn.ModuleList(
            [GraphWaveNetBlock(hidden_dim, supports, dilation=d, kernel_size=2, dropout=dropout) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, horizon, kernel_size=(1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, N, F] -> [B, H, N, T] -> last temporal state -> [B, N, H]
        h = x.permute(0, 3, 2, 1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)[..., -1]
        return out.permute(0, 2, 1)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, args, device) -> tuple[pd.DataFrame, float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = evaluate_loss(model, val_loader, device)
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss}
        rows.append(row)
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"epoch={epoch:03d} train={row['train_loss']:.6f} val={val_loss:.6f}")
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            bad_epochs = 0
            best_state = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best_val={best_val:.6f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(rows), best_val


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device) -> float:
    criterion = nn.MSELoss()
    model.eval()
    losses = []
    for xb, yb, _ in loader:
        pred = model(xb.to(device))
        losses.append(float(criterion(pred, yb.to(device)).detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, ys, tides = [], [], []
    for xb, yb, tb in loader:
        preds.append(model(xb.to(device)).detach().cpu().numpy())
        ys.append(yb.numpy())
        tides.append(tb.numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(tides)


def run_graph_baselines(args) -> None:
    v2.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for horizon in args.horizons:
        data = final4.build_enhanced_data(args, horizon, add_ode_prior=False)
        train_loader = DataLoader(data["single_train"], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(data["single_val"], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(data["single_test"], batch_size=args.batch_size, shuffle=False)
        adj = data["graph_priors"][args.fixed_graph_type]
        model_factories = {
            "dcrnn": lambda: DCRNNForecaster(
                input_dim=data["feats"],
                adj=adj,
                hidden_dim=args.hidden_dim,
                horizon=horizon,
                diffusion_steps=args.diffusion_steps,
                dropout=args.dropout,
            ),
            "graph_wavenet": lambda: GraphWaveNetForecaster(
                input_dim=data["feats"],
                adj=adj,
                hidden_dim=args.hidden_dim,
                horizon=horizon,
                diffusion_steps=args.diffusion_steps,
                blocks=args.gwn_blocks,
                dropout=args.dropout,
            ),
        }
        for model_key in args.models:
            if model_key not in model_factories:
                raise ValueError(f"Unknown graph baseline: {model_key}")
            model = model_factories[model_key]().to(device)
            run_dir = out_dir / f"seed_{args.seed}" / f"horizon_{horizon}h" / model_key
            run_dir.mkdir(parents=True, exist_ok=True)
            print("\n" + "=" * 90)
            print(
                f"Priority-1 graph baseline={model_key} seed={args.seed} horizon={horizon} "
                f"train={len(data['single_train'])} val={len(data['single_val'])} test={len(data['single_test'])} "
                f"features={data['feats']} device={device}"
            )
            history, best_val = train_model(model, train_loader, val_loader, args, device)
            pred, true, tide = predict(model, test_loader, device)
            metrics = final4.summarize_single(true, pred, tide)
            row = {
                "seed": args.seed,
                "model_key": model_key,
                "model_name": "DCRNN" if model_key == "dcrnn" else "Graph WaveNet",
                "horizon": horizon,
                "best_val_loss": best_val,
                "num_features": data["feats"],
                "fixed_graph_type": args.fixed_graph_type,
                "diffusion_steps": args.diffusion_steps,
                "hidden_dim": args.hidden_dim,
                **metrics,
            }
            rows.append(row)
            history.to_csv(run_dir / "training_log.csv", index=False)
            pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
            np.savez_compressed(
                run_dir / "predictions.npz",
                pred_residual=pred,
                true_residual=true,
                target_tide=tide,
                station_ids=np.array(v2.STATION_IDS),
            )
            print(
                f"Done {model_key}: seq_R2={row['seq_residual_R2']:.4f}, "
                f"last_R2={row['last_residual_R2']:.4f}, last_RMSE={row['last_residual_RMSE']:.4f}"
            )
            pd.DataFrame(rows).to_csv(out_dir / f"priority1_graph_baselines_seed_{args.seed}_partial.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"priority1_graph_baselines_seed_{args.seed}.csv", index=False)
    key_cols = [
        "seed",
        "horizon",
        "model_name",
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
    ]
    print(summary[[c for c in key_cols if c in summary.columns]].to_string(index=False))


def load_prediction_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(path, allow_pickle=True)
    if "pred_states" in data.files:
        pred = data["pred_states"][..., 0]
        true = data["true_states"][..., 0]
    else:
        pred = data["pred_residual"]
        true = data["true_residual"]
    tide = data["target_tide"] if "target_tide" in data.files else None
    return pred.astype(np.float64), true.astype(np.float64), tide


def parse_prediction_metadata(path: Path) -> dict[str, object]:
    text = str(path).replace("\\", "/")
    seed_match = re.findall(r"seed[_-]?(\d+)", text)
    horizon_match = re.findall(r"horizon[_-](\d+)h", text)
    return {
        "seed": int(seed_match[-1]) if seed_match else np.nan,
        "horizon": int(horizon_match[-1]) if horizon_match else np.nan,
        "model_key": path.parent.name,
        "prediction_file": str(path),
    }


def r2_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    mse = float(np.mean((yt - yp) ** 2))
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - np.sum((yt - yp) ** 2) / denom) if denom > 0 else float("nan")
    return {"R2": r2, "RMSE": math.sqrt(mse), "MAE": float(np.mean(np.abs(yt - yp)))}


def event_groups(mask: np.ndarray, max_gap: int) -> list[np.ndarray]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    groups = [[int(idx[0])]]
    for value in idx[1:]:
        if int(value) - groups[-1][-1] <= max_gap:
            groups[-1].append(int(value))
        else:
            groups.append([int(value)])
    return [np.asarray(g, dtype=int) for g in groups]


def tolerant_event_detection(true_last: np.ndarray, pred_last: np.ndarray, thresholds: np.ndarray, tolerance: int) -> dict[str, float]:
    total = 0
    detected = 0
    timing_errors = []
    amplitude_errors = []
    for station in range(true_last.shape[1]):
        threshold = float(thresholds[station])
        for group in event_groups(true_last[:, station] >= threshold, max_gap=tolerance):
            total += 1
            start = max(0, int(group[0]) - tolerance)
            end = min(true_last.shape[0], int(group[-1]) + tolerance + 1)
            pred_window = pred_last[start:end, station]
            true_window = true_last[start:end, station]
            if np.any(pred_window >= threshold):
                detected += 1
            pred_peak = int(np.argmax(pred_window))
            true_peak = int(np.argmax(true_window))
            timing_errors.append(float(pred_peak - true_peak))
            amplitude_errors.append(float(pred_window[pred_peak] - true_window[true_peak]))
    return {
        f"event_detect_pm{tolerance}h_count": total,
        f"event_detect_pm{tolerance}h_rate": detected / total if total else float("nan"),
        f"event_detect_pm{tolerance}h_peak_timing_MAE": float(np.mean(np.abs(timing_errors))) if timing_errors else float("nan"),
        f"event_detect_pm{tolerance}h_peak_amp_MAE": float(np.mean(np.abs(amplitude_errors))) if amplitude_errors else float("nan"),
    }


def evaluate_predictions(args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, station_meta, _ = v3.build_enhanced_arrays()
    residual = arrays["residual"].astype(np.float64)
    train_end = int(len(residual) * args.train_ratio)
    q_threshold = np.quantile(residual[:train_end], args.event_quantile, axis=0)
    abs_threshold = np.quantile(np.abs(residual[:train_end]), args.event_quantile, axis=0)
    station_ids = [str(x) for x in station_meta["station_id"].tolist()] if "station_id" in station_meta else [str(i) for i in range(residual.shape[1])]

    prediction_paths: list[Path] = []
    for root in args.prediction_roots:
        root_path = Path(root)
        if root_path.exists():
            prediction_paths.extend(sorted(root_path.rglob("predictions.npz")))
    if not prediction_paths:
        raise RuntimeError(f"No predictions.npz files found under: {args.prediction_roots}")

    summary_rows = []
    horizon_rows = []
    station_rows = []
    event_rows = []
    pr_curve_rows = []

    for path in prediction_paths:
        pred, true, _ = load_prediction_file(path)
        meta = parse_prediction_metadata(path)
        horizon = pred.shape[-1]
        meta["horizon"] = int(meta["horizon"]) if not pd.isna(meta["horizon"]) else horizon
        summary = {
            **meta,
            **{f"seq_residual_{k}": v for k, v in r2_rmse_mae(true, pred).items()},
            **{f"last_residual_{k}": v for k, v in r2_rmse_mae(true[:, :, -1], pred[:, :, -1]).items()},
        }
        summary_rows.append(summary)

        for lead in range(horizon):
            row = {**meta, "lead_hour": lead + 1}
            row.update({f"residual_{k}": v for k, v in r2_rmse_mae(true[:, :, lead], pred[:, :, lead]).items()})
            horizon_rows.append(row)

        for station_idx, station_id in enumerate(station_ids[: pred.shape[1]]):
            row = {**meta, "station_index": station_idx, "station_id": station_id}
            row.update({f"last_residual_{k}": v for k, v in r2_rmse_mae(true[:, station_idx, -1], pred[:, station_idx, -1]).items()})
            station_rows.append(row)

        true_last = true[:, :, -1]
        pred_last = pred[:, :, -1]
        labels_positive = (true_last >= q_threshold[None, :]).reshape(-1).astype(int)
        scores_positive = pred_last.reshape(-1)
        labels_abs = (np.abs(true_last) >= abs_threshold[None, :]).reshape(-1).astype(int)
        scores_abs = np.abs(pred_last).reshape(-1)
        event_row = dict(meta)
        if labels_positive.sum() > 0 and len(np.unique(labels_positive)) > 1:
            event_row["positive_q95_PR_AUC"] = float(average_precision_score(labels_positive, scores_positive))
            precision, recall, _ = precision_recall_curve(labels_positive, scores_positive)
            for p, r in zip(precision, recall):
                pr_curve_rows.append({**meta, "event_type": "positive_q95", "precision": float(p), "recall": float(r)})
        else:
            event_row["positive_q95_PR_AUC"] = float("nan")
        if labels_abs.sum() > 0 and len(np.unique(labels_abs)) > 1:
            event_row["absolute_q95_PR_AUC"] = float(average_precision_score(labels_abs, scores_abs))
        else:
            event_row["absolute_q95_PR_AUC"] = float("nan")
        for tolerance in args.tolerances:
            event_row.update(tolerant_event_detection(true_last, pred_last, q_threshold, tolerance))
        event_rows.append(event_row)

    summary_df = pd.DataFrame(summary_rows)
    horizon_df = pd.DataFrame(horizon_rows)
    station_df = pd.DataFrame(station_rows)
    event_df = pd.DataFrame(event_rows)
    summary_df.to_csv(out_dir / "priority1_prediction_summary.csv", index=False)
    horizon_df.to_csv(out_dir / "priority1_per_horizon_metrics.csv", index=False)
    station_df.to_csv(out_dir / "priority1_per_station_metrics.csv", index=False)
    event_df.to_csv(out_dir / "priority1_event_pr_auc_tolerant.csv", index=False)
    pd.DataFrame(pr_curve_rows).to_csv(out_dir / "priority1_pr_curve_points.csv", index=False)

    group_cols = ["model_key"]
    agg_cols = [c for c in ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE"] if c in summary_df.columns]
    if agg_cols:
        agg = summary_df.groupby(group_cols)[agg_cols].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["_".join(str(x) for x in col if x) for col in agg.columns.to_flat_index()]
        agg.to_csv(out_dir / "priority1_model_mean_std.csv", index=False)
        print(agg.to_string(index=False))
    make_diagnostic_plots(horizon_df, station_df, event_df, out_dir)


def make_diagnostic_plots(horizon_df: pd.DataFrame, station_df: pd.DataFrame, event_df: pd.DataFrame, out_dir: Path) -> None:
    if not horizon_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_key, sub in horizon_df.groupby("model_key"):
            curve = sub.groupby("lead_hour")["residual_R2"].mean().reset_index()
            ax.plot(curve["lead_hour"], curve["residual_R2"], marker="o", linewidth=1.8, label=model_key)
        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel("Residual R2")
        ax.set_title("Per-horizon forecast skill")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "priority1_per_horizon_r2.png", dpi=220)
        plt.close(fig)

    if not station_df.empty:
        pivot = station_df.groupby(["model_key", "station_id"])["last_residual_R2"].mean().unstack("station_id")
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45 * len(pivot))))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_title("Station-wise terminal residual R2")
        fig.colorbar(image, ax=ax, label="R2")
        fig.tight_layout()
        fig.savefig(out_dir / "priority1_per_station_terminal_r2.png", dpi=220)
        plt.close(fig)

    if not event_df.empty and "positive_q95_PR_AUC" in event_df:
        plot_df = event_df.groupby("model_key")["positive_q95_PR_AUC"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(np.arange(len(plot_df)), plot_df.values)
        ax.set_yticks(np.arange(len(plot_df)))
        ax.set_yticklabels(plot_df.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("PR AUC")
        ax.set_title("Threshold-independent positive q95 event skill")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "priority1_event_pr_auc.png", dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Priority-1 publication experiments: DCRNN/GraphWaveNet + diagnostics.")
    parser.add_argument("--mode", choices=["train_graph_baselines", "evaluate_predictions"], required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--horizons", type=int, nargs="+", default=[24])
    parser.add_argument("--models", nargs="+", default=["dcrnn", "graph_wavenet"])
    parser.add_argument("--prediction-roots", nargs="+", default=["results/final", "results/priority1_graph_baselines"])
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", choices=["identity", "distance", "corr"], default="distance")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--gwn-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--physics-forcing-mode", choices=["last_input", "future"], default="last_input")
    parser.add_argument("--tolerances", type=int, nargs="+", default=[3, 6])
    args = parser.parse_args()
    if args.mode == "train_graph_baselines":
        run_graph_baselines(args)
    else:
        evaluate_predictions(args)


if __name__ == "__main__":
    main()
