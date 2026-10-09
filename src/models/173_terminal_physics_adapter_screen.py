"""Screen a terminal-only, physics-conditioned residual adapter.

The fixed horizon-only VARX/GWN mixture is the backbone.  A bounded adapter
can change only lead 24, so the trajectory score remains protected.  The
physics and no-physics variants have identical capacity.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "results" / "terminal_physics_adapter_screen_20260811"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p170 = load_module("p170_terminal", HERE / "170_physics_reliability_gated_residual_moe.py")
final4 = p170.final4


class AdapterDataset(Dataset):
    def __init__(self, x, base, target, tide):
        self.arrays = tuple(torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in (x, base, target, tide))

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, i):
        return tuple(a[i] for a in self.arrays)


def windows(data, period):
    ds = data[period]
    return np.stack([ds.x_scaled[int(t) - ds.window:int(t)] for t in ds.indices]).astype(np.float32)


class TerminalAdapter(nn.Module):
    def __init__(self, features, nodes, hidden, mode, scale):
        super().__init__()
        self.mode = mode
        self.temporal = nn.Sequential(
            nn.Conv1d(features, hidden, 3, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, dilation=2, padding=4), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, dilation=4, padding=8), nn.GELU(),
        )
        self.transport = p170.DynamicTransport(nodes, hidden)
        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=.1)
        self.norm = nn.LayerNorm(hidden)
        in_dim = hidden + 10
        self.head = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(.1), nn.Linear(hidden, 1))
        self.scale = float(scale)

    def forward(self, x, base):
        b, t, n, f = x.shape
        h = x.permute(0, 2, 3, 1).reshape(b * n, f, t)
        h = self.temporal(h)[..., -1].reshape(b, n, -1)
        forcing = x[..., [5, 6, 11, 16, 17]].mean(dim=1)
        if self.mode == "physics":
            h = h + torch.bmm(self.transport(forcing), h)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm(h)
        last = x[:, -1, :, [5, 6, 11, 16, 17]]
        trend = x[:, -1, :, [5, 6, 11, 16, 17]] - x[:, 0, :, [5, 6, 11, 16, 17]]
        context = torch.cat([h, last, trend], dim=-1)
        correction = torch.tanh(self.head(context).squeeze(-1)) * self.scale
        pred = base.clone()
        pred[..., -1] = pred[..., -1] + correction
        return pred, correction


def horizon_weight(y, v, g, cut):
    y, v, g = y[:cut], v[:cut], g[:cut]
    d = g - v
    return np.clip(np.sum(d * (y - v), axis=(0, 1)) / np.maximum(np.sum(d * d, axis=(0, 1)), 1e-8), 0.0, 1.0).astype(np.float32)


def train(model, train, cal, args):
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    cal_loader = DataLoader(cal, batch_size=args.batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best, bad = None, float("inf"), 0
    for _ in range(args.epochs):
        model.train()
        for x, base, y, _ in loader:
            x, base, y = [z.to(args.device) for z in (x, base, y)]
            pred, corr = model(x, base)
            loss = (pred - y).pow(2).mean() + args.terminal_weight * (pred[..., -1] - y[..., -1]).pow(2).mean() + args.reg_weight * corr.pow(2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vals = []
        with torch.no_grad():
            for x, base, y, _ in cal_loader:
                x, base, y = [z.to(args.device) for z in (x, base, y)]
                pred, corr = model(x, base)
                vals.append(float((pred - y).pow(2).mean() + args.terminal_weight * (pred[..., -1] - y[..., -1]).pow(2).mean()))
        score = float(np.mean(vals))
        if score < best - args.min_delta:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return best


@torch.no_grad()
def predict(model, ds, args):
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    pred, true, tide = [], [], []
    model.eval()
    for x, base, y, t in loader:
        p, _ = model(x.to(args.device), base.to(args.device))
        pred.append(p.cpu().numpy()); true.append(y.numpy()); tide.append(t.numpy())
    return np.concatenate(pred), np.concatenate(true), np.concatenate(tide)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(OUT_DEFAULT)); ap.add_argument("--expert-cache-dir", default="")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--patience", type=int, default=2); ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--hidden-dim", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--weight-decay", type=float, default=1e-4); ap.add_argument("--terminal-weight", type=float, default=.8); ap.add_argument("--reg-weight", type=float, default=.1); ap.add_argument("--min-delta", type=float, default=1e-5); ap.add_argument("--scale", type=float, default=.01); ap.add_argument("--device", default="cpu"); ap.add_argument("--cpu-threads", type=int, default=8)
    args = ap.parse_args(); torch.set_num_threads(args.cpu_threads); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    data_args = argparse.Namespace(window=24, train_ratio=.70, val_ratio=.15, train_stride=8, physics_forcing_mode="last_input", extreme_quantile=.90)
    data = final4.build_enhanced_data(data_args, 24, add_ode_prior=False)
    expert_args = argparse.Namespace(output_dir=args.expert_cache_dir or str(out), horizon=24, device=args.device, batch_size=args.batch_size)
    varx, _ = p170.fit_varx(data); gwn = p170.load_adaptive_predictions(data, args.seed, expert_args)
    val = AdapterDataset(windows(data, "single_val"), varx["single_val"]["pred"], varx["single_val"]["true"], varx["single_val"]["tide"])
    test = AdapterDataset(windows(data, "single_test"), varx["single_test"]["pred"], varx["single_test"]["true"], varx["single_test"]["tide"])
    # Replace the VARX base by the horizon-only VARX/GWN mixture.
    cut = len(val) // 2; w = horizon_weight(varx["single_val"]["true"], varx["single_val"]["pred"], gwn["single_val"]["pred"], cut)
    val_base = varx["single_val"]["pred"] + w[None, None] * (gwn["single_val"]["pred"] - varx["single_val"]["pred"])
    test_base = varx["single_test"]["pred"] + w[None, None] * (gwn["single_test"]["pred"] - varx["single_test"]["pred"])
    val = AdapterDataset(windows(data, "single_val"), val_base, varx["single_val"]["true"], varx["single_val"]["tide"])
    test = AdapterDataset(windows(data, "single_test"), test_base, varx["single_test"]["true"], varx["single_test"]["tide"])
    rows = [{"model": "horizon_only", **p170.summarize(test.arrays[2].numpy(), test.arrays[1].numpy(), test.arrays[3].numpy())}]
    for mode in ("no_physics", "physics"):
        p170.set_seed(args.seed); args.seed = args.seed
        model = TerminalAdapter(data["feats"], 7, args.hidden_dim, mode, args.scale).to(args.device)
        train(model, Subset(val, range(cut)), Subset(val, range(cut, len(val))), args)
        pred, true, tide = predict(model, test, args)
        rows.append({"model": mode, **p170.summarize(true, pred, tide), "parameters": sum(p.numel() for p in model.parameters())})
    pd.DataFrame(rows).to_csv(out / "screen_metrics.csv", index=False); (out / "experiment_config.json").write_text(str(vars(args)), encoding="utf-8"); print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
