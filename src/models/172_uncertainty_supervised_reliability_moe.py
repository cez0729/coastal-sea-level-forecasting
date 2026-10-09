"""Focused PRG-RM prototype with explicit reliability supervision.

This exploratory script keeps the existing VARX-Ridge and Adaptive-support
Graph WaveNet experts frozen.  It adds seed-ensemble uncertainty and trains a
station-horizon router against a soft target derived from the two experts'
training errors.  The physics context is an optional routing input, so the
physics-specific increment can be separated from routing capacity.

The script is intentionally a screening tool, not a formal paper result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "results" / "uncertainty_supervised_reliability_moe_20260811"
DEFAULT_ENSEMBLE_SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p170 = load_module("p170_screen", HERE / "170_physics_reliability_gated_residual_moe.py")
p105 = p170.p105
final4 = p170.final4


def set_seed(seed: int) -> None:
    p170.set_seed(seed)


class ReliabilityDataset(Dataset):
    def __init__(self, x, varx, gwn, varx_unc, gwn_unc, target, tide):
        arrays = (x, varx, gwn, varx_unc, gwn_unc, target, tide)
        self.arrays = tuple(torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in arrays)

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, index):
        return tuple(a[index] for a in self.arrays)


def windows(data, period: str) -> np.ndarray:
    ds = data[period]
    return np.stack([ds.x_scaled[int(t) - ds.window:int(t)] for t in ds.indices]).astype(np.float32)


def reliability_target(y: torch.Tensor, varx: torch.Tensor, gwn: torch.Tensor, tau: float) -> torch.Tensor:
    """Soft probability that GWN has lower absolute error than VARX."""
    ev = torch.abs(y - varx)
    eg = torch.abs(y - gwn)
    return torch.sigmoid((ev - eg) / max(float(tau), 1e-3))


def ece_score(score: np.ndarray, label: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (score >= lo) & ((score < hi) if hi < 1 else (score <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(score[mask].mean()) - float(label[mask].mean()))
    return ece


def reliability_metrics(gate: np.ndarray, y: np.ndarray, varx: np.ndarray, gwn: np.ndarray) -> dict[str, float]:
    label = (np.abs(y - gwn) < np.abs(y - varx)).astype(np.int8).reshape(-1)
    score = gate.reshape(-1)
    result = {
        "selection_accuracy": float(np.mean((score >= 0.5) == label)),
        "brier": float(brier_score_loss(label, score)),
        "ece_10bin": ece_score(score, label),
    }
    if np.unique(label).size > 1:
        result["reliability_auroc"] = float(roc_auc_score(label, score))
        result["reliability_auprc"] = float(average_precision_score(label, score))
    else:
        result["reliability_auroc"] = float("nan")
        result["reliability_auprc"] = float("nan")
    return result


class UncertaintyReliabilityMoE(nn.Module):
    """Causal graph-temporal router over frozen VARX and GWN forecasts."""

    def __init__(self, features: int, nodes: int, horizon: int, hidden: int, mode: str, horizon_init: np.ndarray, gate_residual_scale: float, delta_scale: float):
        super().__init__()
        self.nodes, self.horizon, self.mode = nodes, horizon, mode
        self.temporal = nn.Sequential(
            nn.Conv1d(features, hidden, 3, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, dilation=2, padding=4), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, dilation=4, padding=8), nn.GELU(),
        )
        self.transport = p170.DynamicTransport(nodes, hidden)
        self.node_attn = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden)
        self.horizon_emb = nn.Parameter(torch.randn(horizon, hidden) * 0.03)
        init = np.clip(np.asarray(horizon_init, dtype=np.float32), 0.05, 0.95)
        self.horizon_bias = nn.Parameter(torch.from_numpy(np.log(init / (1.0 - init))))
        self.gate_residual_scale = float(gate_residual_scale)
        self.expert_proj = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.LayerNorm(hidden))
        # context + expert embedding + horizon token + six diagnostic channels
        gate_in = hidden * 3 + 6
        self.gate = nn.Sequential(nn.Linear(gate_in, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
        self.delta = nn.Sequential(nn.Linear(gate_in, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.delta_scale = nn.Parameter(torch.tensor(float(delta_scale)))

    def forward(self, x, varx, gwn, varx_unc, gwn_unc):
        b, t, n, f = x.shape
        h = x.permute(0, 2, 3, 1).reshape(b * n, f, t)
        h = self.temporal(h)[..., -1].reshape(b, n, -1)
        physics = x[..., [5, 6, 11, 16, 17]].mean(dim=1)
        if self.mode == "physics_uncertainty":
            h = h + torch.bmm(self.transport(physics), h)
        h, _ = self.node_attn(h, h, h, need_weights=False)
        h = self.norm(h)
        expert = torch.stack([varx, gwn, gwn - varx, (gwn - varx).abs(), varx_unc, gwn_unc], dim=-1)
        e = self.expert_proj(expert)
        context = h[:, :, None, :].expand(-1, -1, self.horizon, -1)
        horizon = self.horizon_emb[None, None].expand(b, n, -1, -1)
        force_norm = physics.norm(dim=-1)[:, :, None].expand(-1, -1, self.horizon)
        pressure = x[:, -1, :, 6][:, :, None].expand(-1, -1, self.horizon)
        lead = torch.linspace(0, 1, self.horizon, device=x.device)[None, None].expand(b, n, -1)
        aux = torch.stack([(gwn - varx).abs(), lead, force_norm, pressure, varx_unc, gwn_unc], dim=-1)
        z = torch.cat([context, e, horizon, aux], dim=-1)
        logits = self.horizon_bias[None, None, :] + self.gate_residual_scale * self.gate(z).squeeze(-1)
        gate = torch.sigmoid(logits)
        correction = torch.tanh(self.delta(z)).squeeze(-1) * self.delta_scale.abs().clamp(0.002, 0.06)
        pred = varx + gate * (gwn - varx) + correction
        return pred, gate, correction


def train_model(model, train_ds, cal_ds, args, tau):
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best, bad = None, float("inf"), 0
    for _ in range(args.epochs):
        model.train()
        for x, v, g, vu, gu, y, _ in loader:
            x, v, g, vu, gu, y = [q.to(args.device) for q in (x, v, g, vu, gu, y)]
            pred, gate, corr = model(x, v, g, vu, gu)
            rel = reliability_target(y, v, g, tau)
            mse = (pred - y).pow(2)
            loss = mse.mean() + args.terminal_weight * mse[..., -1].mean()
            loss = loss + args.reliability_weight * nn.functional.binary_cross_entropy(gate, rel)
            loss = loss + args.correction_weight * corr.pow(2).mean()
            loss = loss + args.smooth_weight * (gate[..., 1:] - gate[..., :-1]).pow(2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); losses = []
        with torch.no_grad():
            for x, v, g, vu, gu, y, _ in cal_loader:
                x, v, g, vu, gu, y = [q.to(args.device) for q in (x, v, g, vu, gu, y)]
                pred, gate, corr = model(x, v, g, vu, gu)
                rel = reliability_target(y, v, g, tau)
                mse = (pred - y).pow(2)
                # Select checkpoints by the primary continuous forecast
                # objective. Reliability supervision is an auxiliary task and
                # must not trade away the quantity the model is meant to predict.
                losses.append(float(mse.mean() + args.terminal_weight * mse[..., -1].mean()))
        score = float(np.mean(losses))
        if score < best - args.min_delta:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


@torch.no_grad()
def predict_model(model, ds, args):
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pred, gate, corr, true, tide = [], [], [], [], []
    model.eval()
    for x, v, g, vu, gu, y, t in loader:
        out = model(x.to(args.device), v.to(args.device), g.to(args.device), vu.to(args.device), gu.to(args.device))
        pred.append(out[0].cpu().numpy()); gate.append(out[1].cpu().numpy()); corr.append(out[2].cpu().numpy())
        true.append(y.numpy()); tide.append(t.numpy())
    return {"pred": np.concatenate(pred), "gate": np.concatenate(gate), "correction": np.concatenate(corr),
            "true": np.concatenate(true), "tide": np.concatenate(tide)}


def make_period_dataset(period, data, varx, gwn, varx_unc, gwn_unc):
    return ReliabilityDataset(windows(data, period), varx[period]["pred"], gwn[period]["pred"],
                              varx_unc, gwn_unc, varx[period]["true"], varx[period]["tide"])


def run_seed(seed, args, data, varx, gwn, uncertainty):
    val, test = make_period_dataset("single_val", data, varx, gwn, uncertainty["varx"]["single_val"], uncertainty["gwn"]["single_val"]), make_period_dataset("single_test", data, varx, gwn, uncertainty["varx"]["single_test"], uncertainty["gwn"]["single_test"])
    cut = len(val) // 2
    train, cal = Subset(val, range(cut)), Subset(val, range(cut, len(val)))
    tau = max(float(np.median(np.abs(np.abs(val.arrays[5][:cut].numpy() - val.arrays[1][:cut].numpy()) - np.abs(val.arrays[5][:cut].numpy() - val.arrays[2][:cut].numpy())))), .01)
    y_fit, v_fit, g_fit = (val.arrays[i][:cut].numpy() for i in (5, 1, 2))
    diff = g_fit - v_fit
    horizon_init = np.clip(np.sum(diff * (y_fit - v_fit), axis=(0, 1)) / np.maximum(np.sum(diff * diff, axis=(0, 1)), 1e-8), 0.05, 0.95)
    rows, rel_rows = [], []
    for mode in args.models:
        set_seed(seed); args.seed = seed
        model = UncertaintyReliabilityMoE(data["feats"], 7, args.horizon, args.hidden_dim, mode, horizon_init, args.gate_residual_scale, args.delta_scale).to(args.device)
        best = train_model(model, train, cal, args, tau)
        out = predict_model(model, test, args)
        rows.append({"seed": seed, "model": mode, "best_cal_loss": best, **p170.summarize(out["true"], out["pred"], out["tide"]), "parameters": sum(p.numel() for p in model.parameters()), "mean_gate": float(out["gate"].mean()), "mean_abs_correction": float(np.abs(out["correction"]).mean())})
        rel_rows.append({"seed": seed, "model": mode, **reliability_metrics(out["gate"], out["true"], out["varx"] if "varx" in out else test.arrays[1].numpy(), out["gwn"] if "gwn" in out else test.arrays[2].numpy())})
    return rows, rel_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(OUT_DEFAULT)); ap.add_argument("--expert-cache-dir", default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--ensemble-seeds", nargs="+", type=int, default=DEFAULT_ENSEMBLE_SEEDS)
    ap.add_argument("--models", nargs="+", choices=["uncertainty", "physics_uncertainty"], default=["uncertainty", "physics_uncertainty"])
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--patience", type=int, default=5); ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--hidden-dim", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--weight-decay", type=float, default=1e-4); ap.add_argument("--terminal-weight", type=float, default=.5); ap.add_argument("--reliability-weight", type=float, default=.25); ap.add_argument("--correction-weight", type=float, default=.1); ap.add_argument("--smooth-weight", type=float, default=.01); ap.add_argument("--min-delta", type=float, default=1e-5); ap.add_argument("--window", type=int, default=24); ap.add_argument("--horizon", type=int, default=24); ap.add_argument("--train-ratio", type=float, default=.70); ap.add_argument("--val-ratio", type=float, default=.15); ap.add_argument("--gate-residual-scale", type=float, default=.25); ap.add_argument("--delta-scale", type=float, default=.012); ap.add_argument("--device", default="cpu"); ap.add_argument("--cpu-threads", type=int, default=4)
    args = ap.parse_args(); torch.set_num_threads(args.cpu_threads)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    data_args = argparse.Namespace(window=args.window, train_ratio=args.train_ratio, val_ratio=args.val_ratio, train_stride=8, physics_forcing_mode="last_input", extreme_quantile=.90)
    data = final4.build_enhanced_data(data_args, args.horizon, add_ode_prior=False)
    thresholds = np.quantile(data["arrays"]["residual"][:int(len(data["arrays"]["residual"]) * args.train_ratio)], .95, axis=0).astype(np.float32)
    varx, _ = p170.fit_varx(data)
    all_gwn = {}
    expert_args = argparse.Namespace(**vars(args))
    expert_args.output_dir = args.expert_cache_dir or args.output_dir
    for s in args.ensemble_seeds:
        all_gwn[s] = p170.load_adaptive_predictions(data, s, expert_args)
    rows, rel_rows = [], []
    for seed in args.seeds:
        gwn = all_gwn[seed]
        train_sigma = np.std(varx["single_train"]["true"] - varx["single_train"]["pred"], axis=0).astype(np.float32)
        varx_unc = {p: np.broadcast_to(train_sigma, varx[p]["pred"].shape).copy() for p in ("single_val", "single_test")}
        gwn_unc = {p: np.std(np.stack([all_gwn[s][p]["pred"] for s in args.ensemble_seeds]), axis=0).astype(np.float32) for p in ("single_val", "single_test")}
        uncertainty = {"varx": varx_unc, "gwn": gwn_unc}
        r, rr = run_seed(seed, args, data, varx, gwn, uncertainty); rows.extend(r); rel_rows.extend(rr)
        pd.DataFrame(rows).to_csv(out / "all_runs_partial.csv", index=False); pd.DataFrame(rel_rows).to_csv(out / "reliability_metrics_partial.csv", index=False)
    all_runs = pd.DataFrame(rows); all_runs.to_csv(out / "all_runs.csv", index=False); pd.DataFrame(rel_rows).to_csv(out / "reliability_metrics.csv", index=False)
    metric_columns = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2", "parameters", "mean_gate", "mean_abs_correction"]
    summary = all_runs.groupby("model")[metric_columns].agg(["mean", "std", "count"])
    summary.to_csv(out / "mean_std.csv")
    (out / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
