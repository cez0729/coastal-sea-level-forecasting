"""Screen a full-train residual GWN with a separate terminal head."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module); return module


p174 = load_module("p174_joint", HERE / "174_horizon_residual_boosted_gwn_screen.py")
p170 = p174.p170
final4 = p174.final4
p105 = p174.p105


class JointTerminalModel(nn.Module):
    def __init__(self, data, args, device):
        super().__init__()
        self.base = p174.make_model(data, args, device)
        self.head = nn.Sequential(nn.Linear(10, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.scale = float(args.terminal_head_scale)

    def forward(self, x):
        correction = self.base(x)[..., 0]
        forcing = x[..., [5, 6, 11, 16, 17]]
        context = torch.cat([forcing[:, -1], forcing[:, -1] - forcing[:, 0]], dim=-1)
        terminal = torch.tanh(self.head(context).squeeze(-1)) * self.scale
        correction = correction.clone(); correction[..., -1] = correction[..., -1] + terminal
        return correction.unsqueeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(ROOT / "results" / "joint_terminal_head_residual_gwn_20260811")); ap.add_argument("--expert-cache-dir", default="results\\prg_rm_candidate_screen_20260811"); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--horizon", type=int, default=24); ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--patience", type=int, default=2); ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--hidden-dim", type=int, default=64); ap.add_argument("--skip-dim", type=int, default=64); ap.add_argument("--blocks", type=int, default=6); ap.add_argument("--node-embedding-dim", type=int, default=12); ap.add_argument("--adaptive-top-k", type=int, default=4); ap.add_argument("--mix-hops", type=int, default=2); ap.add_argument("--mix-retain", type=float, default=.05); ap.add_argument("--dropout", type=float, default=.15); ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--weight-decay", type=float, default=1e-5); ap.add_argument("--terminal-weight", type=float, default=.8); ap.add_argument("--correction-reg", type=float, default=1e-4); ap.add_argument("--terminal-head-scale", type=float, default=.01); ap.add_argument("--grad-clip", type=float, default=1.0); ap.add_argument("--min-delta", type=float, default=1e-5); ap.add_argument("--device", default="cpu"); ap.add_argument("--cpu-threads", type=int, default=8)
    args = ap.parse_args(); args.physics_modulation = False; torch.set_num_threads(args.cpu_threads); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    data_args = argparse.Namespace(window=24, train_ratio=.70, val_ratio=.15, train_stride=8, physics_forcing_mode="last_input", extreme_quantile=.90)
    data = final4.build_enhanced_data(data_args, 24, add_ode_prior=False); varx, _ = p170.fit_varx(data)
    expert_args = argparse.Namespace(output_dir=args.expert_cache_dir or str(out), horizon=24, device=args.device, batch_size=args.batch_size)
    gwn = p170.load_adaptive_predictions(data, args.seed, expert_args); gwn_train = p174.load_frozen_gwn_train(data, args.seed, args)
    w = p174.horizon_weight(varx["single_val"]["true"], varx["single_val"]["pred"], gwn["single_val"]["pred"], len(varx["single_val"]["pred"]) // 2)
    gwn_by_period = {"single_train": gwn_train, "single_val": gwn["single_val"]["pred"], "single_test": gwn["single_test"]["pred"]}
    bases = {p: varx[p]["pred"] + w[None, None] * (gwn_by_period[p] - varx[p]["pred"]) for p in gwn_by_period}
    train = p174.ResidualDataset(p174.windows(data, "single_train"), bases["single_train"], varx["single_train"]["true"], varx["single_train"]["tide"])
    val = p174.ResidualDataset(p174.windows(data, "single_val"), bases["single_val"], varx["single_val"]["true"], varx["single_val"]["tide"])
    test = p174.ResidualDataset(p174.windows(data, "single_test"), bases["single_test"], varx["single_test"]["true"], varx["single_test"]["tide"])
    rows = [{"model": "horizon_only", **p170.summarize(test.target.numpy(), test.base.numpy(), test.tide.numpy())}]
    p170.set_seed(args.seed); model = JointTerminalModel(data, args, torch.device(args.device)); p174.train(model, train, val, args, torch.device(args.device))
    pred, true, tide = p174.predict(model, test, args, torch.device(args.device)); rows.append({"model": "joint_terminal_head", **p170.summarize(true, pred, tide), "parameters": sum(p.numel() for p in model.parameters())})
    pd.DataFrame(rows).to_csv(out / "screen_metrics.csv", index=False); (out / "experiment_config.json").write_text(str(vars(args)), encoding="utf-8"); print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__": main()
