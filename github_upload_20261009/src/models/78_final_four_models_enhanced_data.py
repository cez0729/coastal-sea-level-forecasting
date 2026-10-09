from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPT76 = Path(__file__).resolve().parent / "76_enhanced_forcing_physics_loss_ablation_v3.py"
SCRIPT77 = Path(__file__).resolve().parent / "77_v4_physics_loss_weight_strategy_search.py"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "final_four_models_enhanced_data"

spec76 = importlib.util.spec_from_file_location("v3_impl", SCRIPT76)
v3 = importlib.util.module_from_spec(spec76)
assert spec76.loader is not None
spec76.loader.exec_module(v3)
v2 = v3.v2

spec77 = importlib.util.spec_from_file_location("v4_impl", SCRIPT77)
v4 = importlib.util.module_from_spec(spec77)
assert spec77.loader is not None
spec77.loader.exec_module(v4)


class SingleStateDataset(Dataset):
    def __init__(
        self,
        x_scaled: np.ndarray,
        residual: np.ndarray,
        tide: np.ndarray,
        window: int,
        horizon: int,
        start: int,
        end: int,
        stride: int = 1,
    ):
        self.x_scaled = x_scaled
        self.residual = residual
        self.tide = tide
        self.window = int(window)
        self.horizon = int(horizon)
        self.indices = np.arange(start + window, end - horizon + 1, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = int(self.indices[idx])
        xb = self.x_scaled[t - self.window: t]
        y = self.residual[t: t + self.horizon].T
        tide = self.tide[t: t + self.horizon].T
        return (
            torch.from_numpy(xb.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
            torch.from_numpy(tide.astype(np.float32)),
        )


class FixedGraphGNNBiGRU(nn.Module):
    def __init__(self, input_dim: int, adj: np.ndarray, gnn_hidden: int, gru_hidden: int, horizon: int, dropout: float):
        super().__init__()
        self.register_buffer("adj", torch.tensor(adj, dtype=torch.float32))
        self.gcn1 = v2.GraphConvolution(input_dim, gnn_hidden)
        self.gcn2 = v2.GraphConvolution(gnn_hidden, gnn_hidden)
        self.norm = nn.LayerNorm(gnn_hidden)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(gnn_hidden, gru_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, horizon),
        )

    def forward(self, x):
        h = torch.relu(self.gcn1(x, self.adj))
        h = torch.relu(self.gcn2(h, self.adj))
        h = self.dropout(self.norm(h))
        b, t, n, f = h.shape
        h = h.permute(0, 2, 1, 3).reshape(b * n, t, f)
        _, h_n = self.gru(h)
        pred = self.head(v2.bidirectional_final_hidden(h_n))
        return pred.reshape(b, n, -1)


class LearnableGraphSingleStateGNNBiGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        graph_priors: dict[str, np.ndarray],
        graph_init_weights: list[float],
        gnn_hidden: int,
        gru_hidden: int,
        horizon: int,
        dropout: float,
    ):
        super().__init__()
        self.graph = v2.LearnableGraphFusion(graph_priors, graph_init_weights)
        self.gcn1 = v2.GraphConvolution(input_dim, gnn_hidden)
        self.gcn2 = v2.GraphConvolution(gnn_hidden, gnn_hidden)
        self.norm = nn.LayerNorm(gnn_hidden)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(gnn_hidden, gru_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, horizon),
        )

    def forward(self, x):
        adj = self.graph()
        h = torch.relu(self.gcn1(x, adj))
        h = torch.relu(self.gcn2(h, adj))
        h = self.dropout(self.norm(h))
        b, t, n, f = h.shape
        h = h.permute(0, 2, 1, 3).reshape(b * n, t, f)
        _, h_n = self.gru(h)
        pred = self.head(v2.bidirectional_final_hidden(h_n))
        return pred.reshape(b, n, -1)

    def weight_dict(self):
        return self.graph.weight_dict()


def build_enhanced_data(args, horizon: int, add_ode_prior: bool = False):
    arrays, station_meta, coops_cols = v3.build_enhanced_arrays()
    feature_cols, physics_cols = v3.make_feature_sets(coops_cols)
    if add_ode_prior:
        arrays["ode_persistence_residual"] = arrays["residual"].copy()
        arrays["ode_graph_blend_residual"] = 0.70 * arrays["residual"] + 0.30 * (arrays["adj_distance"] @ arrays["residual"].T).T
        arrays["ode_local_trend_3h"] = np.zeros_like(arrays["residual"], dtype=np.float32)
        arrays["ode_local_trend_3h"][3:] = (arrays["residual"][3:] - arrays["residual"][:-3]) / 3.0
        feature_cols = feature_cols + ["ode_persistence_residual", "ode_graph_blend_residual", "ode_local_trend_3h"]

    x_raw = np.stack([arrays[c] for c in feature_cols], axis=-1).astype(np.float32)
    physics_raw = np.stack([arrays[c] for c in physics_cols], axis=-1).astype(np.float32)
    states = np.stack([arrays["residual"], arrays["uo"], arrays["vo"], arrays["wave_setup_proxy"]], axis=-1).astype(np.float32)
    residual = arrays["residual"].astype(np.float32)
    tide = arrays["tide"].astype(np.float32)

    n_time, nodes, feats = x_raw.shape
    train_end = int(n_time * args.train_ratio)
    val_end = int(n_time * (args.train_ratio + args.val_ratio))

    x_scaler = StandardScaler()
    x_scaler.fit(x_raw[:train_end].reshape(-1, feats))
    x_scaled = x_scaler.transform(x_raw.reshape(-1, feats)).reshape(n_time, nodes, feats).astype(np.float32)

    phys_scaler = StandardScaler()
    phys_scaler.fit(physics_raw[:train_end].reshape(-1, len(physics_cols)))
    phys_scaled = phys_scaler.transform(physics_raw.reshape(-1, len(physics_cols))).reshape(n_time, nodes, len(physics_cols)).astype(np.float32)

    state_scale = np.std(states[:train_end].reshape(-1, len(v2.STATE_NAMES)), axis=0).astype(np.float32) + 1e-6
    delta_scale = np.std((states[1:train_end] - states[: train_end - 1]).reshape(-1, len(v2.STATE_NAMES)), axis=0).astype(np.float32) + 1e-6
    train_abs_eta_threshold = float(np.quantile(np.abs(states[:train_end, :, 0]).reshape(-1), args.extreme_quantile))
    graph_priors = v2.make_graph_priors(arrays, train_end)

    single_train = SingleStateDataset(x_scaled, residual, tide, args.window, horizon, 0, train_end, args.train_stride)
    single_val = SingleStateDataset(x_scaled, residual, tide, args.window, horizon, train_end, val_end, 1)
    single_test = SingleStateDataset(x_scaled, residual, tide, args.window, horizon, val_end, n_time, 1)

    multi_train = v2.MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, 0, train_end, args.physics_forcing_mode, args.train_stride
    )
    multi_val = v2.MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, train_end, val_end, args.physics_forcing_mode, 1
    )
    multi_test = v2.MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, val_end, n_time, args.physics_forcing_mode, 1
    )

    return {
        "arrays": arrays,
        "station_meta": station_meta,
        "coops_cols": coops_cols,
        "feature_cols": feature_cols,
        "physics_cols": physics_cols,
        "graph_priors": graph_priors,
        "state_scale": state_scale,
        "delta_scale": delta_scale,
        "train_abs_eta_threshold": train_abs_eta_threshold,
        "nodes": nodes,
        "feats": feats,
        "single_train": single_train,
        "single_val": single_val,
        "single_test": single_test,
        "multi_train": multi_train,
        "multi_val": multi_val,
        "multi_test": multi_test,
    }


def summarize_single(true_residual, pred_residual, tide) -> dict[str, float]:
    pred_level = pred_residual + tide
    true_level = true_residual + tide
    out = {}
    for prefix, yt, yp in [
        ("seq_residual", true_residual, pred_residual),
        ("last_residual", true_residual[:, :, -1], pred_residual[:, :, -1]),
        ("seq_sea_level", true_level, pred_level),
        ("last_sea_level", true_level[:, :, -1], pred_level[:, :, -1]),
    ]:
        for k, val in v2.regression_metrics(yt, yp).items():
            out[f"{prefix}_{k}"] = val
    true_states = np.zeros((*true_residual.shape, 1), dtype=np.float32)
    pred_states = np.zeros((*pred_residual.shape, 1), dtype=np.float32)
    true_states[..., 0] = true_residual
    pred_states[..., 0] = pred_residual
    out.update(v3.summarize_extreme_metrics(true_states, pred_states))
    return out


def train_single_model(model, train_loader, val_loader, args, device):
    criterion = nn.MSELoss()
    if hasattr(model, "graph"):
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
        model.train()
        train_loss = 0.0
        n = 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            n += 1
        train_loss /= max(1, n)
        val_loss = evaluate_single_loss(model, val_loader, device)
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if hasattr(model, "weight_dict"):
            row.update({f"w_{k}": v for k, v in model.weight_dict().items()})
        rows.append(row)
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"epoch={epoch:03d} train={train_loss:.5f} val={val_loss:.5f}")
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
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
def evaluate_single_loss(model, loader, device) -> float:
    model.eval()
    losses = []
    criterion = nn.MSELoss()
    for xb, yb, _ in loader:
        pred = model(xb.to(device))
        losses.append(float(criterion(pred, yb.to(device)).detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def predict_single(model, loader, device):
    model.eval()
    preds, ys, tides = [], [], []
    for xb, yb, tb in loader:
        preds.append(model(xb.to(device)).detach().cpu().numpy())
        ys.append(yb.numpy())
        tides.append(tb.numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(tides)


def run_single(args, horizon: int, model_key: str, model_name: str, add_ode_prior: bool, device):
    data = build_enhanced_data(args, horizon, add_ode_prior=add_ode_prior)
    train_loader = DataLoader(data["single_train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data["single_val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(data["single_test"], batch_size=args.batch_size, shuffle=False)
    if model_key == "gnn_bigru":
        adj = data["graph_priors"][args.fixed_graph_type]
        model = FixedGraphGNNBiGRU(data["feats"], adj, args.gnn_hidden, args.gru_hidden, horizon, args.dropout).to(device)
    else:
        model = LearnableGraphSingleStateGNNBiGRU(
            data["feats"],
            data["graph_priors"],
            [0.50, 0.35, 0.15],
            args.gnn_hidden,
            args.gru_hidden,
            horizon,
            args.dropout,
        ).to(device)
    print("\n" + "=" * 90)
    print(f"Final enhanced run: {model_name} horizon={horizon} train={len(data['single_train'])} val={len(data['single_val'])} test={len(data['single_test'])} features={data['feats']}")
    history, best_val = train_single_model(model, train_loader, val_loader, args, device)
    pred, true, tide = predict_single(model, test_loader, device)
    metrics = summarize_single(true, pred, tide)
    weights = model.weight_dict() if hasattr(model, "weight_dict") else {}
    row = {
        "model_key": model_key,
        "model_name": model_name,
        "horizon": horizon,
        "best_val_loss": best_val,
        "num_features": data["feats"],
        "training_mode": f"train_stride{args.train_stride}_full_val_test",
        "bigru_state_extraction": "cat_top_layer_forward_backward_h_n",
        "learned_w_identity": weights.get("identity", np.nan),
        "learned_w_distance": weights.get("distance", np.nan),
        "learned_w_corr": weights.get("corr", np.nan),
        **metrics,
    }
    run_dir = Path(args.output_dir) / f"horizon_{horizon}h" / model_key
    run_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
    np.savez_compressed(run_dir / "predictions.npz", pred_residual=pred, true_residual=true, target_tide=tide)
    print(f"Done {model_name} {horizon}h: seq_R2={row['seq_residual_R2']:.4f}, last_R2={row['last_residual_R2']:.4f}, last_RMSE={row['last_residual_RMSE']:.4f}")
    return row


def run_physical_loss(args, horizon: int, device):
    data = build_enhanced_data(args, horizon, add_ode_prior=False)
    train_loader = DataLoader(data["multi_train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data["multi_val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(data["multi_test"], batch_size=args.batch_size, shuffle=False)
    model = v2.MultistateGNNBiGRU(
        input_dim=data["feats"],
        graph_priors=data["graph_priors"],
        graph_init_weights=[0.50, 0.35, 0.15],
        gnn_hidden=args.gnn_hidden,
        gru_hidden=args.gru_hidden,
        horizon=horizon,
        dropout=args.dropout,
        num_states=len(v2.STATE_NAMES),
    ).to(device)
    physics_ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), len(v2.STATE_NAMES)).to(device)
    config = {
        "config_name": f"eta_uvW_lam{args.physics_lambda_max:g}_pgamma0_corrected_bigru",
        "physics_lambda_max": args.physics_lambda_max,
        "physics_state_weights": [1.0, 0.35, 0.35, 0.25],
        "physics_lead_gamma": 0.0,
        "data_lead_gamma": 0.0,
        "extreme_alpha": 0.0,
        "extreme_quantile": args.extreme_quantile,
        "last_step_weight": args.last_step_weight,
    }
    print("\n" + "=" * 90)
    print(f"Final enhanced run: Physical-loss GNN-BiGRU horizon={horizon} train={len(data['multi_train'])} val={len(data['multi_val'])} test={len(data['multi_test'])} features={data['feats']}")
    history, best_val = v4.train_weighted_model(
        model,
        physics_ode,
        train_loader,
        val_loader,
        args,
        config,
        data["state_scale"],
        data["delta_scale"],
        data["train_abs_eta_threshold"],
        horizon,
        device,
    )
    pred_states, true_states, tide, test_phys = v2.predict(model, physics_ode, test_loader, device)
    metrics = v2.summarize_metrics(true_states, pred_states, tide)
    metrics.update(v3.summarize_extreme_metrics(true_states, pred_states))
    weights = model.graph.weight_dict()
    row = {
        "model_key": "physical_loss",
        "model_name": "Physical-loss GNN-BiGRU",
        "horizon": horizon,
        "best_val_loss": best_val,
        "num_features": data["feats"],
        "training_mode": f"train_stride{args.train_stride}_full_val_test",
        "bigru_state_extraction": "cat_top_layer_forward_backward_h_n",
        "learned_w_identity": weights["identity"],
        "learned_w_distance": weights["distance"],
        "learned_w_corr": weights["corr"],
        "test_physics_loss_unscaled": test_phys,
        **metrics,
    }
    run_dir = Path(args.output_dir) / f"horizon_{horizon}h" / "physical_loss"
    run_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
    np.savez_compressed(run_dir / "predictions.npz", pred_states=pred_states, true_states=true_states, target_tide=tide)
    print(f"Done Physical-loss {horizon}h: seq_R2={row['seq_residual_R2']:.4f}, last_R2={row['last_residual_R2']:.4f}, last_RMSE={row['last_residual_RMSE']:.4f}")
    return row


def plot_summary(summary: pd.DataFrame, output_dir: Path):
    for horizon, sub in summary.groupby("horizon"):
        sub = sub.sort_values("last_residual_R2", ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        y = np.arange(len(sub))
        axes[0].barh(y, sub["last_residual_R2"])
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(sub["model_name"], fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Last-step residual R2")
        axes[0].grid(axis="x", alpha=0.25)
        axes[1].barh(y, sub["last_residual_RMSE"])
        axes[1].set_yticks(y)
        axes[1].set_yticklabels([])
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Last-step residual RMSE (m)")
        axes[1].grid(axis="x", alpha=0.25)
        fig.suptitle(f"Final four models with enhanced CO-OPS data, horizon={horizon}h")
        fig.tight_layout()
        fig.savefig(output_dir / f"final_four_models_enhanced_{horizon}h.png", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run final four models with enhanced CO-OPS data")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--horizons", type=int, nargs="+", default=[24])
    parser.add_argument("--models", nargs="+", default=["gnn_bigru", "learnable_graph", "ode_based_learnable", "physical_loss"])
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", default="distance", choices=["identity", "distance", "corr"])
    parser.add_argument("--gnn-hidden", type=int, default=40)
    parser.add_argument("--gru-hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=9)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--graph-lr-mult", type=float, default=3.0)
    parser.add_argument("--physics-lr-mult", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--physics-warmup-epochs", type=int, default=8)
    parser.add_argument("--physics-ramp-epochs", type=int, default=14)
    parser.add_argument("--physics-loss-type", default="huber", choices=["huber", "mse"])
    parser.add_argument("--physics-forcing-mode", default="last_input", choices=["last_input", "future"])
    parser.add_argument("--physics-lambda-max", type=float, default=0.0002)
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.2)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument("--selection-metric", default="val_eta_data_loss",
                        choices=["val_eta_data_loss", "val_data_loss", "val_last_loss", "val_total_loss"])
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    v2.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(
        json.dumps({**vars(args), "bigru_state_extraction": "cat_top_layer_forward_backward_h_n"}, indent=2),
        encoding="utf-8",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for horizon in args.horizons:
        if "gnn_bigru" in args.models:
            rows.append(run_single(args, horizon, "gnn_bigru", "GNN-BiGRU", False, device))
        if "learnable_graph" in args.models:
            rows.append(run_single(args, horizon, "learnable_graph", "Learnable-graph GNN-BiGRU", False, device))
        if "ode_based_learnable" in args.models:
            rows.append(run_single(args, horizon, "ode_based_learnable", "ODE-based learnable GNN-BiGRU", True, device))
        if "physical_loss" in args.models:
            rows.append(run_physical_loss(args, horizon, device))
        summary = pd.DataFrame(rows)
        summary.to_csv(output_dir / "final_four_models_enhanced_metrics_partial.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "final_four_models_enhanced_metrics.csv", index=False)
    key_cols = [
        "horizon",
        "model_name",
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "seq_sea_level_R2",
        "last_sea_level_R2",
        "learned_w_identity",
        "learned_w_distance",
        "learned_w_corr",
    ]
    existing = [c for c in key_cols if c in summary.columns]
    summary[existing].sort_values(["horizon", "last_residual_R2"], ascending=[True, False]).to_csv(
        output_dir / "final_four_models_enhanced_key_metrics.csv", index=False
    )
    plot_summary(summary, output_dir)
    print("\nFinished final four enhanced-data models.")
    print(summary[existing].sort_values(["horizon", "last_residual_R2"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
