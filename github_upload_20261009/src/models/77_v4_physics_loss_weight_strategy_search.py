from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCRIPT76 = Path(__file__).resolve().parent / "76_enhanced_forcing_physics_loss_ablation_v3.py"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "v4_physics_loss_weight_strategy_search"

spec = importlib.util.spec_from_file_location("v3_impl", SCRIPT76)
v3 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v3)
v2 = v3.v2


class WeightedMultistateLoss(nn.Module):
    def __init__(
        self,
        physics_ode,
        state_scale: np.ndarray,
        delta_scale: np.ndarray,
        aux_weight: float,
        physics_loss_type: str,
        physics_state_weights: list[float],
        last_step_weight: float,
        ode_coef_l2: float,
        horizon: int,
        physics_lead_gamma: float,
        data_lead_gamma: float,
        extreme_alpha: float,
        extreme_quantile: float,
        train_abs_eta_threshold: float,
    ):
        super().__init__()
        self.physics_ode = physics_ode
        self.register_buffer("state_scale", torch.tensor(state_scale.reshape(1, 1, 1, -1), dtype=torch.float32))
        self.register_buffer("delta_scale", torch.tensor(delta_scale.reshape(1, 1, 1, -1), dtype=torch.float32))
        self.register_buffer("state_weights", torch.tensor(physics_state_weights, dtype=torch.float32).view(1, 1, 1, -1))
        lead = torch.linspace(1.0 / horizon, 1.0, horizon, dtype=torch.float32)
        self.register_buffer("physics_lead_weights", (lead ** float(physics_lead_gamma)).view(1, 1, horizon, 1))
        self.register_buffer("data_lead_weights", (lead ** float(data_lead_gamma)).view(1, 1, horizon))
        self.aux_weight = float(aux_weight)
        self.physics_loss_type = physics_loss_type
        self.last_step_weight = float(last_step_weight)
        self.ode_coef_l2 = float(ode_coef_l2)
        self.extreme_alpha = float(extreme_alpha)
        self.extreme_quantile = float(extreme_quantile)
        self.train_abs_eta_threshold = float(train_abs_eta_threshold)
        self.huber = nn.SmoothL1Loss(beta=1.0, reduction="none")

    def data_loss(self, pred, target):
        scaled = (pred - target) / self.state_scale
        eta_sq = scaled[..., 0] ** 2
        target_abs = torch.abs(target[..., 0])
        extreme_weight = 1.0 + self.extreme_alpha * (target_abs >= self.train_abs_eta_threshold).float()
        eta_weight = self.data_lead_weights * extreme_weight
        eta_loss = torch.sum(eta_sq * eta_weight) / torch.clamp(torch.sum(eta_weight), min=1.0)

        aux_sq = scaled[..., 1:] ** 2
        aux_loss = torch.mean(aux_sq)
        last_loss = torch.mean(scaled[:, :, -1, 0] ** 2)
        return eta_loss + self.aux_weight * aux_loss, eta_loss, aux_loss, last_loss

    def physics_loss(self, pred, init_states, phys_seq, adj):
        _, residual = self.physics_ode(pred, init_states, phys_seq, adj)
        scaled = residual / self.delta_scale
        if self.physics_loss_type == "huber":
            loss_by_item = self.huber(scaled, torch.zeros_like(scaled))
        elif self.physics_loss_type == "mse":
            loss_by_item = scaled ** 2
        else:
            raise ValueError("physics_loss_type must be huber or mse")
        weighted = loss_by_item * self.state_weights * self.physics_lead_weights
        return torch.mean(weighted)

    def ode_regularization(self):
        reg = torch.mean(self.physics_ode.beta ** 2)
        reg = reg + torch.mean(self.physics_ode.bias ** 2)
        reg = reg + torch.mean(torch.nn.functional.softplus(self.physics_ode.raw_decay) ** 2)
        reg = reg + torch.mean(torch.nn.functional.softplus(self.physics_ode.raw_kappa) ** 2)
        return reg

    def forward(self, pred, target, init_states, phys_seq, adj, physics_lambda: float):
        data_loss, eta_loss, aux_loss, last_loss = self.data_loss(pred, target)
        physics_loss = self.physics_loss(pred, init_states, phys_seq, adj)
        ode_reg = self.ode_regularization()
        total = data_loss + self.last_step_weight * last_loss + float(physics_lambda) * physics_loss + self.ode_coef_l2 * ode_reg
        return total, {
            "data_loss": float(data_loss.detach().cpu()),
            "eta_data_loss": float(eta_loss.detach().cpu()),
            "aux_data_loss": float(aux_loss.detach().cpu()),
            "last_loss": float(last_loss.detach().cpu()),
            "physics_loss": float(physics_loss.detach().cpu()),
            "ode_reg": float(ode_reg.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
            "physics_lambda": float(physics_lambda),
        }


def evaluate_loss(model, criterion, loader, device, physics_lambda: float) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n_batches = 0
    with torch.no_grad():
        for xb, target, _, init_states, phys_seq in loader:
            xb = xb.to(device)
            target = target.to(device)
            init_states = init_states.to(device)
            phys_seq = phys_seq.to(device)
            pred = model(xb)
            _, parts = criterion(pred, target, init_states, phys_seq, model.graph(), physics_lambda)
            for k, val in parts.items():
                totals[k] = totals.get(k, 0.0) + val
            n_batches += 1
    return {k: val / max(1, n_batches) for k, val in totals.items()}


def train_weighted_model(
    model,
    physics_ode,
    train_loader,
    val_loader,
    args,
    config: dict,
    state_scale: np.ndarray,
    delta_scale: np.ndarray,
    train_abs_eta_threshold: float,
    horizon: int,
    device,
) -> tuple[pd.DataFrame, float]:
    criterion = WeightedMultistateLoss(
        physics_ode=physics_ode,
        state_scale=state_scale,
        delta_scale=delta_scale,
        aux_weight=args.aux_weight,
        physics_loss_type=args.physics_loss_type,
        physics_state_weights=config["physics_state_weights"],
        last_step_weight=config["last_step_weight"],
        ode_coef_l2=args.ode_coef_l2,
        horizon=horizon,
        physics_lead_gamma=config["physics_lead_gamma"],
        data_lead_gamma=config["data_lead_gamma"],
        extreme_alpha=config["extreme_alpha"],
        extreme_quantile=config["extreme_quantile"],
        train_abs_eta_threshold=train_abs_eta_threshold,
    ).to(device)

    graph_params = list(model.graph.parameters())
    graph_param_ids = {id(p) for p in graph_params}
    base_params = [p for p in model.parameters() if id(p) not in graph_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": args.lr},
            {"params": graph_params, "lr": args.lr * args.graph_lr_mult},
            {"params": physics_ode.parameters(), "lr": args.lr * args.physics_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    rows = []
    for epoch in range(1, args.epochs + 1):
        current_lambda = v2.physics_lambda_for_epoch(
            epoch, config["physics_lambda_max"], args.physics_warmup_epochs, args.physics_ramp_epochs
        )
        model.train()
        physics_ode.train()
        totals: dict[str, float] = {}
        n_batches = 0
        for xb, target, _, init_states, phys_seq in train_loader:
            xb = xb.to(device)
            target = target.to(device)
            init_states = init_states.to(device)
            phys_seq = phys_seq.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss, parts = criterion(pred, target, init_states, phys_seq, model.graph(), current_lambda)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(physics_ode.parameters()), args.grad_clip)
            optimizer.step()
            for k, val in parts.items():
                totals[k] = totals.get(k, 0.0) + val
            n_batches += 1
        train_parts = {k: val / max(1, n_batches) for k, val in totals.items()}
        val_parts = evaluate_loss(model, criterion, val_loader, device, current_lambda)

        if args.selection_metric == "val_last_loss":
            val_score = val_parts["last_loss"]
        elif args.selection_metric == "val_data_loss":
            val_score = val_parts["data_loss"]
        elif args.selection_metric == "val_total_loss":
            val_score = val_parts["total_loss"]
        else:
            val_score = val_parts["eta_data_loss"]
        scheduler.step(val_score)

        weights = model.graph.weight_dict()
        row = {
            "epoch": epoch,
            "selection_score": val_score,
            **{f"train_{k}": val for k, val in train_parts.items()},
            **{f"val_{k}": val for k, val in val_parts.items()},
            "w_identity": weights["identity"],
            "w_distance": weights["distance"],
            "w_corr": weights["corr"],
            **physics_ode.coefficients(),
        }
        rows.append(row)

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch={epoch:03d} lambda={current_lambda:.5f} "
                f"train_eta={train_parts['eta_data_loss']:.5f} val_eta={val_parts['eta_data_loss']:.5f} "
                f"val_last={val_parts['last_loss']:.5f} val_phys={val_parts['physics_loss']:.5f} "
                f"select={val_score:.5f} w=[{weights['identity']:.3f},{weights['distance']:.3f},{weights['corr']:.3f}]"
            )

        if val_score < best_val - args.min_delta:
            best_val = val_score
            bad_epochs = 0
            best_state = {
                "model": {k: val.detach().cpu().clone() for k, val in model.state_dict().items()},
                "physics_ode": {k: val.detach().cpu().clone() for k, val in physics_ode.state_dict().items()},
            }
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        physics_ode.load_state_dict(best_state["physics_ode"])
    return pd.DataFrame(rows), best_val


def build_datasets(arrays: dict[str, np.ndarray], coops_cols: list[str], args, horizon: int):
    feature_cols, physics_cols = v3.make_feature_sets(coops_cols)
    x_raw = np.stack([arrays[c] for c in feature_cols], axis=-1).astype(np.float32)
    physics_raw = np.stack([arrays[c] for c in physics_cols], axis=-1).astype(np.float32)
    states = np.stack([arrays["residual"], arrays["uo"], arrays["vo"], arrays["wave_setup_proxy"]], axis=-1).astype(np.float32)
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

    train_ds = v2.MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, 0, train_end, args.physics_forcing_mode, args.train_stride
    )
    val_ds = v2.MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, train_end, val_end, args.physics_forcing_mode, 1
    )
    test_ds = v2.MultistateWindowDataset(
        x_scaled, states, tide, phys_scaled, args.window, horizon, val_end, n_time, args.physics_forcing_mode, 1
    )
    graph_priors = v2.make_graph_priors(arrays, train_end)
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "state_scale": state_scale,
        "delta_scale": delta_scale,
        "train_abs_eta_threshold": train_abs_eta_threshold,
        "graph_priors": graph_priors,
        "feature_cols": feature_cols,
        "physics_cols": physics_cols,
        "nodes": nodes,
        "feats": feats,
    }


def make_configs(args) -> list[dict]:
    configs = [
        {
            "config_name": "no_physics_same_weighted_data",
            "physics_lambda_max": 0.0,
            "physics_state_weights": [1.0, 0.35, 0.35, 0.25],
            "physics_lead_gamma": args.physics_lead_gamma,
            "data_lead_gamma": args.data_lead_gamma,
            "extreme_alpha": args.extreme_alpha,
            "extreme_quantile": args.extreme_quantile,
            "last_step_weight": args.last_step_weight,
        }
    ]
    for lam in args.lambda_grid:
        for gamma in args.physics_lead_gamma_grid:
            configs.append({
                "config_name": f"eta_uvW_lam{lam:g}_pgamma{gamma:g}",
                "physics_lambda_max": lam,
                "physics_state_weights": [1.0, 0.35, 0.35, 0.25],
                "physics_lead_gamma": gamma,
                "data_lead_gamma": args.data_lead_gamma,
                "extreme_alpha": args.extreme_alpha,
                "extreme_quantile": args.extreme_quantile,
                "last_step_weight": args.last_step_weight,
            })
    return configs


def run_config(args, arrays, coops_cols, horizon: int, config: dict, device) -> dict[str, float]:
    data = build_datasets(arrays, coops_cols, args, horizon)
    train_loader = DataLoader(data["train_ds"], batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(data["val_ds"], batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(data["test_ds"], batch_size=args.batch_size, shuffle=False, drop_last=False)

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

    print("\n" + "=" * 90)
    print(
        f"V4 horizon={horizon} config={config['config_name']} lambda={config['physics_lambda_max']} "
        f"phys_gamma={config['physics_lead_gamma']} data_gamma={config['data_lead_gamma']} "
        f"extreme_alpha={config['extreme_alpha']} train={len(data['train_ds'])} val={len(data['val_ds'])} test={len(data['test_ds'])}"
    )
    history, best_val = train_weighted_model(
        model=model,
        physics_ode=physics_ode,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        config=config,
        state_scale=data["state_scale"],
        delta_scale=data["delta_scale"],
        train_abs_eta_threshold=data["train_abs_eta_threshold"],
        horizon=horizon,
        device=device,
    )
    pred_states, true_states, target_tide, test_physics_loss = v2.predict(model, physics_ode, test_loader, device)
    metrics = v2.summarize_metrics(true_states, pred_states, target_tide)
    metrics.update(v3.summarize_extreme_metrics(true_states, pred_states))
    weights = model.graph.weight_dict()
    row = {
        "model": "v4_physics_loss_weight_strategy_search",
        "horizon": horizon,
        **config,
        "training_mode": f"train_stride{args.train_stride}_full_val_test",
        "num_features": len(data["feature_cols"]),
        "num_physics_features": len(data["physics_cols"]),
        "best_val_score": best_val,
        "test_physics_loss_unscaled": test_physics_loss,
        "train_abs_eta_threshold": data["train_abs_eta_threshold"],
        "learned_w_identity": weights["identity"],
        "learned_w_distance": weights["distance"],
        "learned_w_corr": weights["corr"],
        **metrics,
    }
    run_dir = Path(args.output_dir) / f"horizon_{horizon}h" / config["config_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
    v3.summarize_lead_metrics(true_states, pred_states, horizon).to_csv(run_dir / "per_lead_metrics.csv", index=False)
    np.savez_compressed(
        run_dir / "predictions.npz",
        pred_states=pred_states,
        true_states=true_states,
        target_tide=target_tide,
        target_start_times=np.array([str(arrays["time"][int(i)]) for i in data["test_ds"].indices]),
        station_ids=np.array(v2.STATION_IDS),
        state_names=np.array(v2.STATE_NAMES),
        feature_cols=np.array(data["feature_cols"]),
        physics_cols=np.array(data["physics_cols"]),
    )
    print(
        f"Done {config['config_name']}: seq_R2={row['seq_residual_R2']:.4f}, "
        f"last_R2={row['last_residual_R2']:.4f}, last_RMSE={row['last_residual_RMSE']:.4f}, "
        f"ext95_R2={row.get('extreme_abs_q95_residual_R2', np.nan):.4f}"
    )
    return row


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for horizon, sub in summary.groupby("horizon"):
        sub = sub.sort_values("last_residual_R2", ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4.8, 0.42 * len(sub))))
        y = np.arange(len(sub))
        axes[0].barh(y, sub["last_residual_R2"])
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(sub["config_name"], fontsize=8)
        axes[0].invert_yaxis()
        axes[0].axvline(float(sub.loc[sub["config_name"].eq("no_physics_same_weighted_data"), "last_residual_R2"].iloc[0]), color="gray", linestyle="--", label="no physics")
        axes[0].set_xlabel("Last-step residual R2")
        axes[0].legend()
        metric = "extreme_abs_q95_residual_R2" if "extreme_abs_q95_residual_R2" in sub.columns else "last_residual_RMSE"
        axes[1].barh(y, sub[metric])
        axes[1].set_yticks(y)
        axes[1].set_yticklabels([])
        axes[1].invert_yaxis()
        axes[1].set_xlabel(metric)
        fig.suptitle(f"V4 physics-loss weighting search, horizon={horizon}h")
        fig.tight_layout()
        fig.savefig(output_dir / f"v4_horizon_{horizon}h_strategy_ranking.png", dpi=220)
        plt.close(fig)

    rows = []
    for horizon, sub in summary.groupby("horizon"):
        base = sub[sub["config_name"] == "no_physics_same_weighted_data"].iloc[0]
        for _, row in sub.iterrows():
            rows.append({
                "horizon": horizon,
                "config_name": row["config_name"],
                "last_R2_gain_vs_no_physics": row["last_residual_R2"] - base["last_residual_R2"],
                "last_RMSE_reduction_pct_vs_no_physics": 100.0 * (base["last_residual_RMSE"] - row["last_residual_RMSE"]) / base["last_residual_RMSE"],
                "extreme_q95_R2_gain_vs_no_physics": row.get("extreme_abs_q95_residual_R2", np.nan) - base.get("extreme_abs_q95_residual_R2", np.nan),
            })
    pd.DataFrame(rows).to_csv(output_dir / "v4_gain_vs_no_physics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 physics-loss weighting strategy search")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--horizons", type=int, nargs="+", default=[24])
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--gnn-hidden", type=int, default=40)
    parser.add_argument("--gru-hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
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
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument("--selection-metric", default="val_last_loss",
                        choices=["val_eta_data_loss", "val_data_loss", "val_last_loss", "val_total_loss"])
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[0.0003, 0.0005, 0.001, 0.002])
    parser.add_argument("--physics-lead-gamma-grid", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    parser.add_argument("--physics-lead-gamma", type=float, default=1.0)
    parser.add_argument("--data-lead-gamma", type=float, default=1.5)
    parser.add_argument("--extreme-alpha", type=float, default=1.0)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--last-step-weight", type=float, default=0.50)
    parser.add_argument("--max-configs", type=int, default=0)
    args = parser.parse_args()

    v2.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arrays, station_meta, coops_cols = v3.build_enhanced_arrays()
    station_meta.to_csv(output_dir / "station_meta_used.csv", index=False)
    pd.DataFrame({"coops_enhanced_feature": coops_cols}).to_csv(output_dir / "coops_features_used.csv", index=False)
    print(f"Using CO-OPS enhanced features: {coops_cols}")
    print(f"Device: {device}")

    configs = make_configs(args)
    if args.max_configs and args.max_configs > 0:
        configs = configs[: args.max_configs]

    rows = []
    for horizon in args.horizons:
        for config in configs:
            rows.append(run_config(args, arrays, coops_cols, horizon, config, device))
            summary = pd.DataFrame(rows)
            summary.to_csv(output_dir / "v4_strategy_search_metrics_partial.csv", index=False)

    summary = pd.DataFrame(rows).sort_values(["horizon", "last_residual_R2"], ascending=[True, False])
    summary.to_csv(output_dir / "v4_strategy_search_metrics.csv", index=False)
    key_cols = [
        "horizon",
        "config_name",
        "physics_lambda_max",
        "physics_lead_gamma",
        "data_lead_gamma",
        "extreme_alpha",
        "last_step_weight",
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "extreme_abs_q95_residual_RMSE",
        "learned_w_identity",
        "learned_w_distance",
        "learned_w_corr",
    ]
    existing = [c for c in key_cols if c in summary.columns]
    summary[existing].to_csv(output_dir / "v4_strategy_search_key_metrics.csv", index=False)
    plot_summary(summary, output_dir)
    print("\nFinished V4 strategy search.")
    print(summary[existing].to_string(index=False))


if __name__ == "__main__":
    main()
