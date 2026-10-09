"""Physics-guided reliability-gated residual mixture for the sea-level benchmark.

The two forecasting experts (VARX-Ridge and Adaptive-support GWN) are frozen.
Only the fusion/refinement module is trained on a chronological split of the
original validation period.  The final test period is never used for model or
threshold selection.  This script is intentionally separate from the existing
formal results so the candidate can be audited before entering a manuscript.
"""
from __future__ import annotations

import argparse
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
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "results" / "physics_reliability_gated_residual_moe_20260811"
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p105 = load_module("p105_moe", HERE / "105_adaptive_multiscale_graph_wavenet.py")
p104 = p105.p104
final4 = p105.final4
priority1 = p105.priority1
STATIONS = list(p105.v2.STATION_IDS)


def set_seed(seed: int) -> None:
    p104.set_reproducible(seed, 4)
    torch.use_deterministic_algorithms(True, warn_only=True)


def design_matrix(dataset):
    x = dataset.x_scaled
    residual = dataset.residual
    rows, targets, tides = [], [], []
    for t in dataset.indices:
        t = int(t)
        rows.append(np.concatenate([
            x[t - dataset.window:t, :, 0].reshape(-1),
            x[t - 1].reshape(-1),
            x[t - dataset.window:t].mean(axis=0).reshape(-1),
        ]))
        targets.append(residual[t:t + dataset.horizon].T.reshape(-1))
        tides.append(dataset.tide[t:t + dataset.horizon].T)
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64), np.asarray(tides, dtype=np.float64)


def fit_varx(data, alpha_grid=(0.1, 1.0, 10.0, 100.0)):
    tx, ty, _ = design_matrix(data["single_train"])
    vx, vy, _ = design_matrix(data["single_val"])
    best_alpha, best_loss = None, np.inf
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, solver="lsqr").fit(tx, ty)
        loss = np.mean((model.predict(vx) - vy) ** 2)
        if loss < best_loss:
            best_alpha, best_loss = alpha, loss
    model = Ridge(alpha=best_alpha, solver="lsqr").fit(tx, ty)
    outputs = {}
    for name in ("single_train", "single_val", "single_test"):
        x, y, tide = design_matrix(data[name])
        outputs[name] = {
            "pred": model.predict(x).reshape(-1, 7, data[name].horizon).astype(np.float32),
            "true": y.reshape(-1, 7, data[name].horizon).astype(np.float32),
            "tide": tide.astype(np.float32),
        }
    return outputs, best_alpha


def make_loader(dataset, batch_size=256, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None, num_workers=0)


def load_adaptive_predictions(data, seed: int, args):
    """Load a frozen best Adaptive GWN and infer all three periods."""
    cache = Path(args.output_dir) / f"frozen_gwn_predictions_seed_{seed}.npz"
    if cache.exists():
        saved = np.load(cache)
        return {
            "single_val": {"pred": saved["val_pred"], "true": saved["val_true"], "tide": saved["val_tide"]},
            "single_test": {"pred": saved["test_pred"], "true": saved["test_true"], "tide": saved["test_tide"]},
        }
    ckpt = ROOT / "results" / "formal_adaptive_gwn_20260811" / "final" / f"seed_{seed}" / "adaptive_single_ms" / "last_epoch_checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing frozen GWN checkpoint: {ckpt}")
    cfg = json.loads((ROOT / "results" / "formal_adaptive_gwn_20260811" / "config_final.json").read_text(encoding="utf-8"))
    model = p105.AdaptiveMultiScaleGraphWaveNet(
        input_dim=data["feats"], adjacency=data["graph_priors"]["distance"],
        hidden_dim=int(cfg["hidden_dim"]), skip_dim=int(cfg["skip_dim"]), horizon=args.horizon,
        num_states=4, blocks=int(cfg["blocks"]), kernels=(2,), adaptive=True,
        node_embedding_dim=int(cfg["node_embedding_dim"]), adaptive_top_k=int(cfg["adaptive_top_k"]),
        mix_hops=int(cfg["mix_hops"]), mix_retain=float(cfg["mix_retain"]), dropout=float(cfg["dropout"]),
    ).to(args.device)
    payload = torch.load(ckpt, map_location=args.device, weights_only=False)
    state = payload.get("best_state_dict", payload.get("model_state_dict"))
    # Multistate training stores the predictor and ODE auxiliary state under
    # separate keys; only the frozen predictor is needed for this meta-model.
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    outputs = {}
    # Gate training uses validation predictions. The formal test predictions
    # were already saved by the frozen expert run, so re-use that exact file.
    with torch.no_grad():
        loader = make_loader(data["multi_val"], args.batch_size, False, seed)
        pred, true, tide, _ = p104.predict_multistate(model, loader, torch.device(args.device))
        outputs["single_val"] = {"pred": pred[..., 0].astype(np.float32), "true": true[..., 0].astype(np.float32), "tide": tide.astype(np.float32)}
    formal_test = np.load(ckpt.parent / "predictions.npz")
    outputs["single_test"] = {
        "pred": formal_test["pred_states"][..., 0].astype(np.float32),
        "true": formal_test["true_states"][..., 0].astype(np.float32),
        "tide": formal_test["target_tide"].astype(np.float32),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        val_pred=outputs["single_val"]["pred"], val_true=outputs["single_val"]["true"], val_tide=outputs["single_val"]["tide"],
        test_pred=outputs["single_test"]["pred"], test_true=outputs["single_test"]["true"], test_tide=outputs["single_test"]["tide"],
    )
    return outputs


class FusionDataset(Dataset):
    def __init__(self, features, varx, gwn, target, tide, event_thresholds):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.varx = torch.from_numpy(varx.astype(np.float32))
        self.gwn = torch.from_numpy(gwn.astype(np.float32))
        self.target = torch.from_numpy(target.astype(np.float32))
        self.tide = torch.from_numpy(tide.astype(np.float32))
        self.event_thresholds = torch.from_numpy(event_thresholds.astype(np.float32))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):
        return self.features[i], self.varx[i], self.gwn[i], self.target[i], self.tide[i], self.event_thresholds


class DynamicTransport(nn.Module):
    """Forcing-conditioned directed graph over station tokens."""
    def __init__(self, nodes: int, dim: int):
        super().__init__()
        self.source = nn.Parameter(torch.randn(nodes, dim) * 0.08)
        self.target = nn.Parameter(torch.randn(nodes, dim) * 0.08)
        self.forcing = nn.Sequential(nn.Linear(5, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, forcing):
        # forcing: [B,N,5]; returns row-normalized [B,N,N]
        b, n, _ = forcing.shape
        q = self.source[None] + self.forcing(forcing)
        k = self.target[None] + self.forcing(forcing)
        score = torch.einsum("bnd,bmd->bnm", q, k) / self.temperature.abs().clamp_min(0.2)
        eye = torch.eye(n, device=forcing.device)[None]
        return torch.softmax(score.masked_fill(eye.bool(), -1e4), dim=-1) * 0.9 + eye * 0.1


class ReliabilityGatedResidualMoE(nn.Module):
    """VARX anchor + GWN residual with horizon/station reliability gating."""
    def __init__(self, features: int, nodes: int, horizon: int, hidden: int = 64, mode: str = "physics_gated"):
        super().__init__()
        self.nodes, self.horizon, self.mode = nodes, horizon, mode
        self.temporal = nn.Sequential(
            nn.Conv1d(features, hidden, 3, padding=2, dilation=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=8, dilation=4), nn.GELU(),
        )
        self.transport = DynamicTransport(nodes, hidden)
        self.node_attn = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden)
        self.horizon_emb = nn.Parameter(torch.randn(horizon, hidden) * 0.03)
        self.expert_proj = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.LayerNorm(hidden))
        gate_in = hidden * 3 + 4  # context + expert embedding + horizon token + 4 diagnostics
        self.gate = nn.Sequential(nn.Linear(gate_in, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
        self.delta = nn.Sequential(nn.Linear(gate_in, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.event_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.delta_scale = nn.Parameter(torch.tensor(0.015))

    def forward(self, x, varx, gwn):
        # x [B,T,N,F], experts [B,N,H]
        b, t, n, f = x.shape
        h = x.permute(0, 2, 3, 1).reshape(b * n, f, t)
        h = self.temporal(h)[..., -1].reshape(b, n, -1)
        physics = x[..., [5, 6, 11, 16, 17]].mean(dim=1)
        if self.mode == "physics_gated":
            adj = self.transport(physics)
            h = h + torch.bmm(adj, h)
        h, _ = self.node_attn(h, h, h, need_weights=False)
        h = self.norm(h)
        expert = torch.stack([varx, gwn, gwn - varx, (gwn - varx).abs()], dim=-1)
        e = self.expert_proj(expert)
        context = h[:, :, None, :].expand(-1, -1, self.horizon, -1)
        token = torch.cat([context, e, self.horizon_emb[None, None].expand(b, n, -1, -1)], dim=-1)
        # The extra four channels are expert disagreement, horizon fraction,
        # and two physical reliability summaries.
        aux = torch.stack([
            (gwn - varx).abs(),
            torch.linspace(0, 1, self.horizon, device=x.device)[None, None].expand(b, n, -1),
            physics.norm(dim=-1)[:, :, None].expand(-1, -1, self.horizon),
            x[:, -1, :, 6][:, :, None].expand(-1, -1, self.horizon),
        ], dim=-1)
        z = torch.cat([token, aux], dim=-1)
        gate = torch.sigmoid(self.gate(z)).squeeze(-1)
        correction = torch.tanh(self.delta(z)).squeeze(-1) * self.delta_scale.abs().clamp(0.002, 0.08)
        if self.mode == "convex_gate":
            correction = correction * 0.0
        pred = varx + gate * (gwn - varx) + correction
        event_logit = self.event_head(h).squeeze(-1)
        return pred, gate, correction, event_logit


def event_targets(target, thresholds):
    return (target.max(dim=-1).values > thresholds[None]).float()


def train_gate(model, train_ds, cal_ds, args, seed):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    best_state, best = None, np.inf
    bad = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x, v, g, y, _, th in train_loader:
            x, v, g, y, th = x.to(args.device), v.to(args.device), g.to(args.device), y.to(args.device), th.to(args.device)
            pred, gate, corr, event = model(x, v, g)
            mse = (pred - y).pow(2)
            loss = mse.mean() + args.terminal_weight * mse[..., -1].mean() + args.anchor_weight * (corr.pow(2).mean())
            loss = loss + args.event_weight * nn.functional.binary_cross_entropy_with_logits(event, event_targets(y, th[0]))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            train_loss += float(loss.detach()) * len(y)
        model.eval(); cal_losses = []
        with torch.no_grad():
            for x, v, g, y, _, th in cal_loader:
                x, v, g, y, th = x.to(args.device), v.to(args.device), g.to(args.device), y.to(args.device), th.to(args.device)
                pred, _, corr, event = model(x, v, g)
                loss = (pred - y).pow(2).mean() + args.terminal_weight * (pred[..., -1] - y[..., -1]).pow(2).mean()
                loss = loss + args.anchor_weight * corr.pow(2).mean() + args.event_weight * nn.functional.binary_cross_entropy_with_logits(event, event_targets(y, th[0]))
                cal_losses.append(float(loss))
        cal_loss = float(np.mean(cal_losses)); history.append({"epoch": epoch + 1, "train_loss": train_loss / len(train_ds), "cal_loss": cal_loss})
        if cal_loss < best - args.min_delta:
            best = cal_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= args.patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return pd.DataFrame(history), best


@torch.no_grad()
def predict_gate(model, ds, args):
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.eval(); preds=[]; true=[]; tide=[]; gates=[]; corr=[]; event=[]
    for x,v,g,y,t,th in loader:
        out = model(x.to(args.device), v.to(args.device), g.to(args.device))
        preds.append(out[0].cpu().numpy()); gates.append(out[1].cpu().numpy()); corr.append(out[2].cpu().numpy()); event.append(out[3].cpu().numpy()); true.append(y.numpy()); tide.append(t.numpy())
    return {"pred":np.concatenate(preds),"true":np.concatenate(true),"tide":np.concatenate(tide),"gate":np.concatenate(gates),"correction":np.concatenate(corr),"event_logit":np.concatenate(event)}


def make_fusion_ds(period, varx, gwn, data, thresholds):
    ds = data[period]
    x = np.stack([ds.x_scaled[int(t)-ds.window:int(t)] for t in ds.indices]).astype(np.float32)
    return FusionDataset(x, varx[period]["pred"], gwn[period]["pred"], varx[period]["true"], varx[period]["tide"], thresholds)


def summarize(true, pred, tide):
    return final4.summarize_single(true, pred, tide)


def run_seed(seed, args, data, varx, gwn, thresholds):
    # Split validation chronologically: first half trains the gate, second half calibrates it.
    n = len(varx["single_val"]["pred"]); cut = n // 2
    full = make_fusion_ds("single_val", varx, gwn, data, thresholds)
    train_ds = torch.utils.data.Subset(full, range(cut)); cal_ds = torch.utils.data.Subset(full, range(cut, n))
    test_ds = make_fusion_ds("single_test", varx, gwn, data, thresholds)
    rows=[]; diagnostics={}
    for mode in args.models:
        set_seed(seed)
        model = ReliabilityGatedResidualMoE(data["feats"], 7, args.horizon, args.hidden_dim, mode).to(args.device)
        hist, best = train_gate(model, train_ds, cal_ds, args, seed)
        out = predict_gate(model, test_ds, args)
        metrics = summarize(out["true"], out["pred"], out["tide"])
        row = {"seed":seed,"model":mode,"best_cal_loss":best,"parameters":sum(p.numel() for p in model.parameters()),**metrics,"mean_gate":float(out["gate"].mean()),"mean_abs_correction":float(np.abs(out["correction"]).mean())}
        rows.append(row); diagnostics[mode] = out
        (Path(args.output_dir)/mode/f"seed_{seed}").mkdir(parents=True, exist_ok=True)
        hist.to_csv(Path(args.output_dir)/mode/f"seed_{seed}"/"training_log.csv", index=False)
        np.savez_compressed(Path(args.output_dir)/mode/f"seed_{seed}"/"predictions.npz", **out)
    return rows, diagnostics


def plot_results(summary, out):
    metrics=["seq_residual_R2_mean","last_residual_R2_mean","extreme_abs_q95_residual_R2_mean"]
    labels=["Trajectory R2","24-h terminal R2","Descriptive q95 R2"]
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(3); width=.18
    for i,(_,r) in enumerate(summary.iterrows()): ax.bar(x+(i-(len(summary)-1)/2)*width,[r[m] for m in metrics],width,label=r["model"])
    ax.set_xticks(x,labels); ax.set_ylabel("Residual R2"); ax.grid(axis="y",alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out,dpi=200); plt.close(fig)


def summarize_output(out: Path) -> None:
    all_runs = pd.read_csv(out / "all_runs.csv")
    summary = all_runs.groupby("model")[["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2", "parameters", "mean_gate", "mean_abs_correction"]].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(c) for c in x if c) for x in summary.columns.to_flat_index()]
    summary.to_csv(out / "mean_std.csv", index=False)
    plot_results(summary, out / "candidate_comparison.png")
    print(summary.to_string(index=False))
    compare_frozen_baselines(out)


def compare_frozen_baselines(out: Path) -> None:
    rows = []
    for seed in SEEDS:
        gate_path = out / "convex_gate" / f"seed_{seed}" / "predictions.npz"
        varx_path = ROOT / "results" / "formal_varx_ridge_20260811" / "VARX_ridge" / f"seed_{seed}" / "predictions.npz"
        gwn_path = out / f"frozen_gwn_predictions_seed_{seed}.npz"
        if not (gate_path.exists() and varx_path.exists() and gwn_path.exists()):
            continue
        gate = np.load(gate_path); varx = np.load(varx_path); gwn = np.load(gwn_path)
        items = [
            ("ReliabilityGate", gate["pred"], gate["true"], gate["tide"]),
            ("VARX-Ridge", varx["pred_residual"], varx["true_residual"], varx["target_tide"]),
            ("Adaptive-GWN", gwn["test_pred"], gwn["test_true"], gwn["test_tide"]),
        ]
        for model_name, pred, true, tide in items:
            rows.append({"seed": seed, "model": model_name, **summarize(true, pred, tide)})
    if not rows:
        return
    data = pd.DataFrame(rows)
    data.to_csv(out / "baseline_comparison_all_runs.csv", index=False)
    metrics = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2"]
    summary = data.groupby("model")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(c) for c in x if c) for x in summary.columns.to_flat_index()]
    summary.to_csv(out / "baseline_comparison_mean_std.csv", index=False)
    pivot = data.pivot(index="seed", columns="model", values=metrics)
    paired = []
    for baseline in ("VARX-Ridge", "Adaptive-GWN"):
        for metric in metrics:
            delta = pivot[(metric, "ReliabilityGate")] - pivot[(metric, baseline)]
            improvement = -delta if metric.endswith("RMSE") else delta
            paired.append({"comparison": f"ReliabilityGate - {baseline}", "metric": metric, "mean_improvement": improvement.mean(), "std_improvement": improvement.std(), "wins": int((improvement > 0).sum()), "n": len(improvement)})
    pd.DataFrame(paired).to_csv(out / "paired_comparisons.csv", index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",default=str(OUT_DEFAULT)); ap.add_argument("--seeds",nargs="+",type=int,default=SEEDS); ap.add_argument("--models",nargs="+",choices=["convex_gate","residual_gate","physics_gated"],default=["convex_gate","residual_gate","physics_gated"]); ap.add_argument("--merge-only",action="store_true"); ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--patience",type=int,default=6); ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--hidden-dim",type=int,default=64); ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--weight-decay",type=float,default=1e-4); ap.add_argument("--terminal-weight",type=float,default=.5); ap.add_argument("--anchor-weight",type=float,default=.10); ap.add_argument("--event-weight",type=float,default=.03); ap.add_argument("--min-delta",type=float,default=1e-5); ap.add_argument("--window",type=int,default=24); ap.add_argument("--horizon",type=int,default=24); ap.add_argument("--train-ratio",type=float,default=.70); ap.add_argument("--val-ratio",type=float,default=.15); ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--cpu-threads",type=int,default=4); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); (out/"experiment_config.json").write_text(json.dumps(vars(args),indent=2),encoding="utf-8")
    if args.merge_only:
        summarize_output(out)
        return
    data_args=argparse.Namespace(window=args.window,train_ratio=args.train_ratio,val_ratio=args.val_ratio,train_stride=8,physics_forcing_mode="last_input",extreme_quantile=.90)
    data=final4.build_enhanced_data(data_args,args.horizon,add_ode_prior=False)
    thresholds=np.quantile(data["arrays"]["residual"][:int(len(data["arrays"]["residual"])*args.train_ratio)],.95,axis=0).astype(np.float32)
    rows=[]
    for seed in args.seeds:
        print(f"Preparing frozen experts for seed {seed} ...",flush=True); varx,_=fit_varx(data); gwn=load_adaptive_predictions(data,seed,args)
        r,_=run_seed(seed,args,data,varx,gwn,thresholds); rows.extend(r); pd.DataFrame(rows).to_csv(out/"all_runs_partial.csv",index=False)
    all_runs=pd.DataFrame(rows); all_runs.to_csv(out/"all_runs.csv",index=False)
    summarize_output(out)


if __name__ == "__main__": main()
