"""Reliability-supervised PRG-RM audit.

This experiment implements the P0 route from the PRG-RM design memo:
continuous expert-reliability supervision, gate-input ablations, simple
fusion baselines, reliability metrics, and mechanism visualizations. Frozen
VARX and Adaptive-GWN predictions are used; no test target is used for fitting.
"""
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "results" / "reliability_supervised_gate_audit_20260811"
SOURCE = ROOT / "results" / "physics_reliability_gated_residual_moe_20260811_final"
SEEDS = [42, 123, 2024, 2025, 3407]
NODES, HORIZON = 7, 24
STATION_LABELS = ["New London", "Montauk", "Kings Point", "The Battery", "Sandy Hook", "Atlantic City", "Cape May"]
MODES = ["G0_station_horizon", "G1_expert_disagreement", "G2_weather", "G3_ocean", "Full_physical_context"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p170 = load_module("p170_reliability", HERE / "170_physics_reliability_gated_residual_moe.py")
final4 = p170.final4


def set_seed(seed: int) -> None:
    p170.set_seed(seed)
    torch.set_num_threads(4)


def build_data(args):
    data_args = argparse.Namespace(
        window=args.window, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        train_stride=8, physics_forcing_mode="last_input", extreme_quantile=.90,
    )
    return final4.build_enhanced_data(data_args, args.horizon, add_ode_prior=False)


def load_experts(data, seed: int):
    varx = np.load(ROOT / "results" / "formal_varx_ridge_20260811" / "VARX_ridge" / f"seed_{seed}" / "predictions.npz")
    gwn = np.load(SOURCE / f"frozen_gwn_predictions_seed_{seed}.npz")
    out = {}
    for period in ("single_val", "single_test"):
        suffix = "val" if period.endswith("val") else "test"
        out[period] = {
            "varx": varx["pred_residual"].astype(np.float32) if suffix == "test" else None,
            "gwn": gwn[f"{suffix}_pred"].astype(np.float32),
            "true": gwn[f"{suffix}_true"].astype(np.float32),
            "tide": gwn[f"{suffix}_tide"].astype(np.float32),
        }
    # VARX files store the test predictions. Validation predictions are
    # reconstructed from the deterministic fitted Ridge model below.
    tx, ty, _ = p170.design_matrix(data["single_train"])
    vx, vy, _ = p170.design_matrix(data["single_val"])
    from sklearn.linear_model import Ridge
    best_alpha, best_loss = None, np.inf
    for alpha in (.1, 1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha, solver="lsqr").fit(tx, ty)
        loss = np.mean((model.predict(vx) - vy) ** 2)
        if loss < best_loss:
            best_alpha, best_loss = alpha, loss
    model = Ridge(alpha=best_alpha, solver="lsqr").fit(tx, ty)
    vp = model.predict(vx).reshape(-1, NODES, HORIZON).astype(np.float32)
    out["single_val"]["varx"] = vp
    # Assert the two frozen sources use exactly the same target windows.
    if not np.allclose(out["single_val"]["true"], vy.reshape(-1, NODES, HORIZON), atol=1e-6):
        raise RuntimeError(f"Validation target mismatch for seed {seed}")
    if not np.allclose(out["single_test"]["true"], varx["true_residual"], atol=1e-6):
        raise RuntimeError(f"Test target mismatch for seed {seed}")
    return out


def window_stats(data, period: str):
    ds = data[period]
    windows = np.stack([ds.x_scaled[int(t) - ds.window:int(t)] for t in ds.indices]).astype(np.float32)
    last = windows[:, -1]
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    trend = windows[:, -1] - windows[:, 0]
    return np.concatenate([last, mean, std, trend], axis=-1)


def numeric_inputs(stats: np.ndarray, varx: np.ndarray, gwn: np.ndarray, mode: str) -> np.ndarray:
    expert = np.stack([varx, gwn, gwn - varx, np.abs(gwn - varx)], axis=-1)
    b, n, h, _ = expert.shape
    stats_parts = []
    if mode != "G0_station_horizon":
        # Last, mean, std, and trend are concatenated for selected forcing
        # variables. Feature indices match 34-column preprocessing in 04_DATA.
        selected = []
        if mode in ("G2_weather", "G3_ocean", "Full_physical_context"):
            selected += [5, 6, 27]  # wind speed, pressure anomaly/tendency
        if mode in ("G3_ocean", "Full_physical_context"):
            selected += [11, 12, 16, 17]  # current and wave state
        if mode == "Full_physical_context":
            selected += [7, 8, 9, 18, 19, 20, 21, 22, 26, 28, 31, 32, 33]
        selected = sorted(set(selected))
        stats_parts.append(stats[..., [4 * idx + offset for idx in selected for offset in range(4)]])
    if stats_parts:
        context = np.concatenate(stats_parts, axis=-1)
        context = np.repeat(context[:, :, None, :], h, axis=2)
        return np.concatenate([expert, context], axis=-1)
    return np.zeros((b, n, h, 1), dtype=np.float32)


class ReliabilityGate(nn.Module):
    def __init__(self, numeric_dim: int, nodes: int = NODES, horizon: int = HORIZON, hidden: int = 48):
        super().__init__()
        self.station = nn.Embedding(nodes, 12)
        self.horizon = nn.Embedding(horizon, 12)
        self.numeric = nn.Sequential(nn.Linear(numeric_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.head = nn.Sequential(nn.Linear(hidden + 24, hidden), nn.GELU(), nn.Dropout(.10), nn.Linear(hidden, 1))

    def forward(self, numeric):
        b, n, h, _ = numeric.shape
        z = self.numeric(numeric)
        station = self.station(torch.arange(n, device=numeric.device))[None, :, None].expand(b, n, h, -1)
        horizon = self.horizon(torch.arange(h, device=numeric.device))[None, None].expand(b, n, h, -1)
        return self.head(torch.cat([z, station, horizon], dim=-1)).squeeze(-1)


def reliability_target(y, varx, gwn, tau):
    ev = np.abs(y - varx)
    eg = np.abs(y - gwn)
    # soft target is P(GWN is the more reliable expert), with a neutral value
    # near 0.5 when the two experts have indistinguishable errors.
    a = np.exp(-eg / tau)
    b = np.exp(-ev / tau)
    return (a / np.maximum(a + b, 1e-8)).astype(np.float32)


def train_one(model, train_loader, cal_loader, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_loss, bad = None, np.inf, 0
    history = []
    for epoch in range(args.epochs):
        model.train(); train_loss = 0.0
        for numeric, varx, gwn, y, rel in train_loader:
            numeric, varx, gwn, y, rel = [x.to(args.device) for x in (numeric, varx, gwn, y, rel)]
            logits = model(numeric); gate = torch.sigmoid(logits); pred = varx + gate * (gwn - varx)
            mse = (pred - y).pow(2)
            loss = mse.mean() + args.terminal_weight * mse[..., -1].mean()
            loss = loss + args.reliability_weight * nn.functional.binary_cross_entropy_with_logits(logits, rel)
            if args.smooth_weight:
                loss = loss + args.smooth_weight * (gate[..., 1:] - gate[..., :-1]).pow(2).mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss += float(loss.detach()) * len(y)
        model.eval(); cal_losses = []
        with torch.no_grad():
            for numeric, varx, gwn, y, rel in cal_loader:
                numeric, varx, gwn, y, rel = [x.to(args.device) for x in (numeric, varx, gwn, y, rel)]
                logits = model(numeric); gate = torch.sigmoid(logits); pred = varx + gate * (gwn - varx)
                mse = (pred - y).pow(2)
                cal_losses.append(float(mse.mean() + args.terminal_weight * mse[..., -1].mean() + args.reliability_weight * nn.functional.binary_cross_entropy_with_logits(logits, rel)))
        cal_loss = float(np.mean(cal_losses)); history.append({"epoch": epoch + 1, "train_loss": train_loss / len(train_loader.dataset), "cal_loss": cal_loss})
        if cal_loss < best_loss - args.min_delta:
            best_loss = cal_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= args.patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return pd.DataFrame(history), best_loss


@torch.no_grad()
def predict_gate(model, numeric, device):
    model.eval(); out=[]
    for start in range(0, len(numeric), 512):
        out.append(torch.sigmoid(model(torch.from_numpy(numeric[start:start + 512]).to(device))).cpu().numpy())
    return np.concatenate(out)


def reliability_metrics(gate, y, varx, gwn):
    label = (np.abs(y - gwn) < np.abs(y - varx)).astype(np.int8).reshape(-1)
    score = gate.reshape(-1)
    pred = (score >= .5).astype(np.int8)
    result = {
        "selection_accuracy": float(np.mean(pred == label)),
        "brier": float(brier_score_loss(label, score)),
        "mean_gate": float(score.mean()),
        "mean_target": float(label.mean()),
    }
    if np.unique(label).size > 1:
        result["reliability_AUROC"] = float(roc_auc_score(label, score))
        result["reliability_AUPRC"] = float(average_precision_score(label, score))
    else:
        result["reliability_AUROC"] = np.nan; result["reliability_AUPRC"] = np.nan
    bins = []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (score >= lo) & (score < hi if hi < 1 else score <= hi)
        if mask.any(): bins.append(abs(score[mask].mean() - label[mask].mean()) * mask.mean())
    result["ECE_10bin"] = float(np.sum(bins))
    return result


def fit_simple_baselines(y, varx, gwn, cut):
    d = gwn - varx
    def fit_weight(a, target, diff):
        den = np.sum(diff * diff, axis=0)
        num = np.sum(diff * (target - a), axis=0)
        return np.clip(num / np.maximum(den, 1e-8), 0, 1)
    weights = {
        "Average_0.5": np.full((NODES, HORIZON), .5, dtype=np.float32),
        "Global_weight": np.full((NODES, HORIZON), float(fit_weight(varx[:cut], y[:cut], d[:cut]).mean()), dtype=np.float32),
        "Station_only": np.repeat(fit_weight(varx[:cut], y[:cut], d[:cut]).mean(axis=1, keepdims=True), HORIZON, axis=1).astype(np.float32),
        "Horizon_only": np.repeat(fit_weight(varx[:cut], y[:cut], d[:cut]).mean(axis=0, keepdims=True), NODES, axis=0).astype(np.float32),
    }
    return weights


def run_seed(seed, args, data, thresholds):
    exp = load_experts(data, seed)
    val, test = exp["single_val"], exp["single_test"]
    vstats, tstats = window_stats(data, "single_val"), window_stats(data, "single_test")
    cut = len(val["true"]) // 2
    tau = max(float(np.median(np.abs(np.abs(val["true"][:cut] - val["gwn"][:cut]) - np.abs(val["true"][:cut] - val["varx"][:cut])))), .01)
    rows, rel_rows, gate_outputs = [], [], {}
    simple = fit_simple_baselines(val["true"], val["varx"], val["gwn"], cut)
    for name, w in simple.items():
        pred = test["varx"] + w[None] * (test["gwn"] - test["varx"])
        rows.append({"seed": seed, "model": name, **p170.summarize(test["true"], pred, test["tide"])})
    for mode in MODES:
        vnum = numeric_inputs(vstats, val["varx"], val["gwn"], mode)
        tnum = numeric_inputs(tstats, test["varx"], test["gwn"], mode)
        vrel = reliability_target(val["true"], val["varx"], val["gwn"], tau)
        train = TensorDataset(*(torch.from_numpy(x[:cut]) for x in (vnum, val["varx"], val["gwn"], val["true"], vrel)))
        cal = TensorDataset(*(torch.from_numpy(x[cut:]) for x in (vnum, val["varx"], val["gwn"], val["true"], vrel)))
        set_seed(seed); model = ReliabilityGate(vnum.shape[-1]).to(args.device)
        history, best = train_one(model, DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)), DataLoader(cal, batch_size=args.batch_size, shuffle=False), args)
        gate = predict_gate(model, tnum, args.device)
        pred = test["varx"] + gate * (test["gwn"] - test["varx"])
        row = {"seed": seed, "model": mode, "best_cal_loss": best, "tau": tau, **p170.summarize(test["true"], pred, test["tide"])}
        rows.append(row)
        rel_rows.append({"seed": seed, "model": mode, **reliability_metrics(gate, test["true"], test["varx"], test["gwn"])})
        gate_outputs[mode] = gate
        run_dir = Path(args.output_dir) / mode / f"seed_{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(run_dir / "training_log.csv", index=False)
        np.savez_compressed(run_dir / "predictions.npz", pred=pred, gate=gate, true=test["true"], tide=test["tide"], varx=test["varx"], gwn=test["gwn"])
    return rows, rel_rows, gate_outputs


def make_visuals(out, gates, forcing):
    full = np.mean(np.stack(gates), axis=0).mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 4.4)); im = ax.imshow(full, aspect="auto", vmin=0, vmax=1, cmap="viridis"); ax.set_xlabel("Forecast lead (h)"); ax.set_ylabel("Station"); ax.set_xticks(range(0, HORIZON, 2), range(1, HORIZON + 1, 2)); ax.set_yticks(range(NODES), STATION_LABELS); fig.colorbar(im, ax=ax, label="P(GWN is more reliable)"); fig.tight_layout(); fig.savefig(out / "full_gate_station_horizon_heatmap.png", dpi=220); plt.close(fig)
    forcing = np.asarray(forcing); q = np.quantile(forcing, .75); strong = forcing >= q
    all_gate = np.concatenate(gates, axis=0)
    values = [float(all_gate[strong].mean()), float(all_gate[~strong].mean())]
    fig, ax = plt.subplots(figsize=(5.5, 4)); ax.bar(["Strong forcing\n(top 25%)", "Normal forcing"], values, color=["#D95F02", "#1B9E77"]); ax.set_ylim(0, 1); ax.set_ylabel("Mean gate weight to GWN"); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out / "gate_by_forcing_regime.png", dpi=220); plt.close(fig)


def summarize(out):
    all_runs = pd.read_csv(out / "all_runs.csv"); rel = pd.read_csv(out / "reliability_metrics.csv")
    metrics = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2"]
    s = all_runs.groupby("model")[metrics].agg(["mean", "std", "count"]).reset_index(); s.columns = ["_".join(str(x) for x in c if x) for c in s.columns.to_flat_index()]; s.to_csv(out / "mean_std.csv", index=False)
    rs = rel.groupby("model")[['selection_accuracy','reliability_AUROC','reliability_AUPRC','brier','ECE_10bin']].agg(['mean','std','count']).reset_index(); rs.columns = ['_'.join(str(x) for x in c if x) for c in rs.columns.to_flat_index()]; rs.to_csv(out / "reliability_mean_std.csv", index=False)
    print(s.to_string(index=False)); print(rs.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", default=str(OUT_DEFAULT)); ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--patience", type=int, default=5); ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--hidden-dim", type=int, default=48); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--weight-decay", type=float, default=1e-4); ap.add_argument("--terminal-weight", type=float, default=.5); ap.add_argument("--reliability-weight", type=float, default=.25); ap.add_argument("--smooth-weight", type=float, default=.01); ap.add_argument("--min-delta", type=float, default=1e-5); ap.add_argument("--window", type=int, default=24); ap.add_argument("--horizon", type=int, default=24); ap.add_argument("--train-ratio", type=float, default=.70); ap.add_argument("--val-ratio", type=float, default=.15); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--merge-only", action="store_true"); args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if args.merge_only: summarize(out); return
    data = build_data(args); thresholds = np.quantile(data["arrays"]["residual"][:int(len(data["arrays"]["residual"]) * args.train_ratio)], .95, axis=0)
    rows, rel_rows = [], []; gate_store = []; forcing_store = []
    for seed in args.seeds:
        r, rr, gates = run_seed(seed, args, data, thresholds); rows.extend(r); rel_rows.extend(rr); gate_store.append(gates["Full_physical_context"]); stats = window_stats(data, "single_test"); forcing_store.append(np.mean(np.abs(stats[..., [4 * i for i in [5, 6, 11, 16, 17]]]), axis=(1, 2)))
        pd.DataFrame(rows).to_csv(out / "all_runs_partial.csv", index=False); pd.DataFrame(rel_rows).to_csv(out / "reliability_metrics_partial.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "all_runs.csv", index=False); pd.DataFrame(rel_rows).to_csv(out / "reliability_metrics.csv", index=False); make_visuals(out, gate_store, np.concatenate(forcing_store)); (out / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8"); summarize(out)


if __name__ == "__main__": main()
