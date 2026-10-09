from __future__ import annotations

"""Matched no-prior fine-tuning control for both locked HS-DT experts."""

import copy
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


m140 = load_module("hsdt_no_prior_143", HERE / "140_hsdt_expert_physics_conditioned.py")


def train_control(kind, model, data, args, seed, device):
    states = 1 if kind == "eta" else 4
    out = ROOT / args.output_dir / args.stage / f"seed_{seed}" / f"{kind}_no_prior_finetune"
    out.mkdir(parents=True, exist_ok=True)
    m140.p104.set_reproducible(seed, args.cpu_threads)
    train_loader = m140.p104.make_loader(data["multi_train"], args, True, seed)
    val_loader = m140.p104.make_loader(data["multi_val"], args, False, seed)
    scale = torch.tensor(data["state_scale"].reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    best, best_val, bad, history = None, float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for xb, target, _, _, _ in train_loader:
            xb, target = xb.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw = model(xb)
            pred = raw.unsqueeze(-1) if states == 1 else raw
            loss = m140.norm_eta_loss(pred[..., 0], target, scale)
            if states == 4:
                loss = loss + args.aux_weight * m140.norm_multi_loss(pred, target, scale)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval(); vals = []
        with torch.no_grad():
            for xb, target, _, _, _ in val_loader:
                raw = model(xb.to(device)); pred = raw.unsqueeze(-1) if states == 1 else raw
                vals.append(float(m140.norm_eta_loss(pred[..., 0], target.to(device), scale).cpu()))
        val = float(np.mean(vals)); scheduler.step(val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_eta_loss": val})
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"{kind}_no_prior_finetune epoch={epoch:03d} val={val:.6f}", flush=True)
        if val < best_val - args.min_delta:
            best_val, bad = val, 0; best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience: break
    model.load_state_dict(best); model.eval()
    metadata = {"seed": seed, "kind": kind, "best_val_eta_loss": best_val,
                "training_seconds": time.perf_counter() - started, "no_prior": True,
                "strict_causal_preprocessing": True, "future_residual_used_as_input": False}
    pd.DataFrame(history).to_csv(out / "training_log.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, out / "best_checkpoint.pt")
    (out / "COMPLETE.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model


def run_seed(seed, args, device):
    data = m140.rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    eta = m140.load_base(data, args, seed, 1, device)
    multi = m140.load_base(data, args, seed, 4, device)
    eta_ft = train_control("eta", copy.deepcopy(eta), data, args, seed, device)
    multi_ft = train_control("multistate", copy.deepcopy(multi), data, args, seed, device)
    eta_pred, true, tide = m140.score_predictions(eta, data, args, seed, device, args.stage, 1)
    multi_pred = m140.score_predictions(multi, data, args, seed, device, args.stage, 4)[0]
    eta_ft_pred = m140.score_predictions(eta_ft, data, args, seed, device, args.stage, 1)[0]
    multi_ft_pred = m140.score_predictions(multi_ft, data, args, seed, device, args.stage, 4)[0]
    weights = np.full(args.horizon, 0.5, dtype=np.float64); weights[-1] = 1.0
    baseline = eta_pred + weights[None, None, :] * (multi_pred - eta_pred)
    control = eta_ft_pred + weights[None, None, :] * (multi_ft_pred - eta_ft_pred)
    train_end = m140.rolling.time_index(data["arrays"]["time"], args.fold_train_end)
    threshold = np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)
    split = "2025_h1_validation_screen" if args.stage == "screen" else "2025_h2_backtest"
    common = {"seed": seed, "evaluation_split": split, "strict_causal_preprocessing": True,
              "future_residual_used_as_input": False}
    rows = [
        {**common, "config": "hsdt_baseline", **m140.confirm.score(true, baseline, tide, threshold)},
        {**common, "config": "hsdt_no_prior_finetune_control", **m140.confirm.score(true, control, tide, threshold)},
    ]
    out = ROOT / args.output_dir / args.stage; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"seed_{seed}_metrics.csv", index=False)
    np.savez_compressed(out / f"seed_{seed}_predictions.npz", true_residual=true.astype(np.float32),
                        hsdt_baseline=baseline.astype(np.float32), hsdt_no_prior_finetune_control=control.astype(np.float32))
    return rows


def main():
    args = m140.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds: rows.extend(run_seed(seed, args, device))
    out = ROOT / args.output_dir / args.stage; data = pd.DataFrame(rows); data.to_csv(out / "all_runs.csv", index=False)
    metrics = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2", "event_PR_AUC"]
    data.groupby("config")[metrics].agg(["mean", "std", "count"]).to_csv(out / "mean_std.csv")
    pivot = data.pivot(index="seed", columns="config", values=metrics); paired = []
    for metric in metrics:
        delta = pivot[(metric, "hsdt_no_prior_finetune_control")] - pivot[(metric, "hsdt_baseline")]
        improvement = -delta if metric.endswith(("RMSE", "MAE")) else delta
        paired.append({"metric": metric, "mean_improvement": float(improvement.mean()),
                       "std_improvement": float(improvement.std(ddof=1)), "wins": int((improvement > 0).sum()),
                       "count": len(improvement), "wilcoxon_greater_p": m140.p104.exact_wilcoxon_greater(improvement.to_numpy())})
    pd.DataFrame(paired).to_csv(out / "paired_tests.csv", index=False)


if __name__ == "__main__": main()
