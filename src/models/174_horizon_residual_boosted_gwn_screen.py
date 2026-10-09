"""Screen an Adaptive GWN that learns residuals on top of a strong mixture.

The horizon-only VARX/GWN mixture is frozen as an anchor.  A zero-initialized
Adaptive Graph WaveNet learns only its remaining residual error.  This avoids
the failure mode where a free gate overwrites a strong simple baseline.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "results" / "horizon_residual_boosted_gwn_screen_20260811"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p170 = load_module("p170_boost", HERE / "170_physics_reliability_gated_residual_moe.py")
p105 = p170.p105
final4 = p170.final4


class ResidualDataset(Dataset):
    def __init__(self, x, base, target, tide):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.base = torch.from_numpy(np.asarray(base, dtype=np.float32))
        self.target = torch.from_numpy(np.asarray(target, dtype=np.float32))
        self.tide = torch.from_numpy(np.asarray(tide, dtype=np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.base[i], self.target[i], self.tide[i]


def windows(data, period):
    ds = data[period]
    return np.stack([ds.x_scaled[int(t) - ds.window:int(t)] for t in ds.indices]).astype(np.float32)


def horizon_weight(y, v, g, cut):
    d = g[:cut] - v[:cut]
    return np.clip(np.sum(d * (y[:cut] - v[:cut]), axis=(0, 1)) / np.maximum(np.sum(d * d, axis=(0, 1)), 1e-8), 0, 1).astype(np.float32)


def load_frozen_gwn_train(data, seed, args):
    cache_dir = Path(args.expert_cache_dir or args.output_dir)
    cache = cache_dir / f"frozen_gwn_train_predictions_seed_{seed}.npz"
    if cache.exists():
        return np.load(cache)["pred"]
    ckpt = ROOT / "results" / "formal_adaptive_gwn_20260811" / "final" / f"seed_{seed}" / "adaptive_single_ms" / "last_epoch_checkpoint.pt"
    config = __import__("json").loads((ROOT / "results" / "formal_adaptive_gwn_20260811" / "config_final.json").read_text(encoding="utf-8"))
    model = p105.AdaptiveMultiScaleGraphWaveNet(
        input_dim=data["feats"], adjacency=data["graph_priors"]["distance"], hidden_dim=int(config["hidden_dim"]),
        skip_dim=int(config["skip_dim"]), horizon=24, num_states=4, blocks=int(config["blocks"]), kernels=(2,),
        adaptive=True, node_embedding_dim=int(config["node_embedding_dim"]), adaptive_top_k=int(config["adaptive_top_k"]),
        mix_hops=int(config["mix_hops"]), mix_retain=float(config["mix_retain"]), dropout=float(config["dropout"]),
    ).to(args.device)
    payload = torch.load(ckpt, map_location=args.device, weights_only=False)
    state = payload.get("best_state_dict", payload.get("model_state_dict"))
    if isinstance(state, dict) and "model" in state: state = state["model"]
    model.load_state_dict(state); model.eval()
    loader = p170.make_loader(data["multi_train"], args.batch_size, False, seed)
    pred, _, _, _ = p170.p104.predict_multistate(model, loader, torch.device(args.device))
    cache_dir.mkdir(parents=True, exist_ok=True); np.savez_compressed(cache, pred=pred[..., 0].astype(np.float32))
    return pred[..., 0].astype(np.float32)


def make_model(data, args, device):
    model = p105.AdaptiveMultiScaleGraphWaveNet(
        input_dim=data["feats"], adjacency=data["graph_priors"]["distance"], hidden_dim=args.hidden_dim,
        skip_dim=args.skip_dim, horizon=args.horizon, num_states=1, blocks=args.blocks, kernels=(2,),
        adaptive=True, node_embedding_dim=args.node_embedding_dim, adaptive_top_k=args.adaptive_top_k,
        mix_hops=args.mix_hops, mix_retain=args.mix_retain, dropout=args.dropout,
    ).to(device)
    # Start exactly at the strong anchor; the learned network must earn every
    # residual correction during validation.
    last = model.end[-1]
    nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    if args.physics_modulation:
        model.physics_beta = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=device))
    return model


def correction_output(model, x):
    correction = model(x)[..., 0]
    if hasattr(model, "physics_beta"):
        forcing = x[..., [5, 6, 11, 16, 17]]
        intensity = torch.tanh((forcing[:, -1] - forcing[:, 0]).abs().mean(dim=(1, 2)))
        correction = correction * (1.0 + model.physics_beta * intensity[:, None, None])
    return correction


def train(model, train, cal, args, device):
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    cal_loader = DataLoader(cal, batch_size=args.batch_size, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def validation_score():
        model.eval(); vals = []
        with torch.no_grad():
            for x, base, y, _ in cal_loader:
                x, base, y = x.to(device), base.to(device), y.to(device)
                pred = base + correction_output(model, x)
                mse = (pred - y).pow(2)
                vals.append(float(mse.mean() + args.terminal_weight * mse[..., -1].mean()))
        return float(np.mean(vals))

    # Epoch zero is the frozen anchor because the correction head is exactly
    # zero-initialized. Include it in early stopping so training cannot force a
    # residual correction that is already worse on validation.
    best = validation_score()
    best_epoch, bad = 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, base, y, _ in loader:
            x, base, y = x.to(device), base.to(device), y.to(device)
            correction = correction_output(model, x)
            pred = base + correction
            mse = (pred - y).pow(2)
            loss = mse.mean() + args.terminal_weight * mse[..., -1].mean() + args.correction_reg * correction.pow(2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step()
        score = validation_score()
        if score < best - args.min_delta:
            best, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience: break
    model.load_state_dict(best_state)
    return best, best_epoch


@torch.no_grad()
def predict(model, ds, args, device):
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    preds, trues, tides = [], [], []
    model.eval()
    for x, base, y, tide in loader:
        preds.append((base.to(device) + correction_output(model, x.to(device))).cpu().numpy()); trues.append(y.numpy()); tides.append(tide.numpy())
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(tides)


def protect_terminal(anchor: np.ndarray, boosted: np.ndarray) -> np.ndarray:
    """Keep residual corrections for leads 1--23 and freeze lead 24."""
    if anchor.shape != boosted.shape or anchor.shape[-1] != 24:
        raise ValueError(f"Expected matching 24-lead arrays, got {anchor.shape} and {boosted.shape}")
    protected = boosted.copy()
    protected[..., -1] = anchor[..., -1]
    return protected


def evaluation_rows(split, true, tide, anchor, boosted, protected, parameters):
    rows = []
    for model_name, pred in (
        ("horizon_anchor", anchor),
        ("residual_boosted_adaptive_gwn", boosted),
        ("lead24_protected_residual_boosted_gwn", protected),
    ):
        rows.append({
            "split": split,
            "model": model_name,
            **p170.summarize(true, pred, tide),
            "parameters": parameters if model_name != "horizon_anchor" else np.nan,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(OUT_DEFAULT)); ap.add_argument("--expert-cache-dir", default=""); ap.add_argument("--anchor", choices=["varx", "horizon"], default="varx"); ap.add_argument("--physics-modulation", action="store_true"); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--horizon", type=int, default=24); ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--patience", type=int, default=2); ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--hidden-dim", type=int, default=64); ap.add_argument("--skip-dim", type=int, default=64); ap.add_argument("--blocks", type=int, default=6); ap.add_argument("--node-embedding-dim", type=int, default=12); ap.add_argument("--adaptive-top-k", type=int, default=4); ap.add_argument("--mix-hops", type=int, default=2); ap.add_argument("--mix-retain", type=float, default=.05); ap.add_argument("--dropout", type=float, default=.15); ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--weight-decay", type=float, default=1e-5); ap.add_argument("--terminal-weight", type=float, default=.5); ap.add_argument("--correction-reg", type=float, default=1e-4); ap.add_argument("--grad-clip", type=float, default=1.0); ap.add_argument("--min-delta", type=float, default=1e-5); ap.add_argument("--device", default="cpu"); ap.add_argument("--cpu-threads", type=int, default=8)
    args = ap.parse_args(); torch.set_num_threads(args.cpu_threads); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    data_args = argparse.Namespace(window=24, train_ratio=.70, val_ratio=.15, train_stride=8, physics_forcing_mode="last_input", extreme_quantile=.90)
    data = final4.build_enhanced_data(data_args, 24, add_ode_prior=False)
    varx, _ = p170.fit_varx(data)
    bases = {}
    if args.anchor == "varx":
        bases = {period: varx[period]["pred"] for period in ("single_train", "single_val", "single_test")}
    else:
        yv, vv = varx["single_val"]["true"], varx["single_val"]["pred"]
        expert_args = argparse.Namespace(output_dir=args.expert_cache_dir or str(out), horizon=24, device=args.device, batch_size=args.batch_size)
        gwn = p170.load_adaptive_predictions(data, args.seed, expert_args)
        cut = len(yv) // 2; w = horizon_weight(yv, vv, gwn["single_val"]["pred"], cut)
        gwn_train = load_frozen_gwn_train(data, args.seed, args)
        for period in ("single_train", "single_val", "single_test"):
            g = gwn_train if period == "single_train" else gwn[period]["pred"]
            bases[period] = varx[period]["pred"] + w[None, None] * (g - varx[period]["pred"])
    train_ds = ResidualDataset(windows(data, "single_train"), bases["single_train"], varx["single_train"]["true"], varx["single_train"]["tide"])
    val = ResidualDataset(windows(data, "single_val"), bases["single_val"], varx["single_val"]["true"], varx["single_val"]["tide"])
    test = ResidualDataset(windows(data, "single_test"), bases["single_test"], varx["single_test"]["true"], varx["single_test"]["tide"])
    p170.set_seed(args.seed)
    device = torch.device(args.device)
    model = make_model(data, args, device)
    best_val_loss, best_epoch = train(model, train_ds, val, args, device)
    parameters = sum(p.numel() for p in model.parameters())

    val_boosted, val_true, val_tide = predict(model, val, args, device)
    test_boosted, test_true, test_tide = predict(model, test, args, device)
    val_anchor = val.base.numpy()
    test_anchor = test.base.numpy()
    val_protected = protect_terminal(val_anchor, val_boosted)
    test_protected = protect_terminal(test_anchor, test_boosted)

    rows = evaluation_rows("validation", val_true, val_tide, val_anchor, val_boosted, val_protected, parameters)
    rows.extend(evaluation_rows("test", test_true, test_tide, test_anchor, test_boosted, test_protected, parameters))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "evaluation_metrics.csv", index=False)
    metrics[metrics["split"] == "test"].drop(columns="split").to_csv(out / "screen_metrics.csv", index=False)
    np.savez_compressed(
        out / "predictions.npz",
        val_true=val_true,
        val_tide=val_tide,
        val_anchor=val_anchor,
        val_boosted=val_boosted,
        val_protected=val_protected,
        test_true=test_true,
        test_tide=test_tide,
        test_anchor=test_anchor,
        test_boosted=test_boosted,
        test_protected=test_protected,
    )
    torch.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "best_validation_loss": best_val_loss,
        "best_epoch": best_epoch,
        "parameters": parameters,
        "config": vars(args),
    }, out / "best_checkpoint.pt")
    (out / "experiment_config.json").write_text(json.dumps(vars(args), ensure_ascii=True, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
