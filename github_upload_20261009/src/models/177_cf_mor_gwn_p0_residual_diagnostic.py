"""P0 residual-predictability diagnostic for CF-MOR-GWN.

This low-cost gate follows the CF-MOR-GWN design before any expensive
cross-fitted Graph WaveNet training. A VARX anchor is cross-fitted over purged
chronological blocks, and simple residual learners are trained only on OOF
anchor errors. The first half of the formal validation interval selects a
candidate; the second half confirms it. Test predictions are never loaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "results" / "cf_mor_gwn_p0_residual_diagnostic_20260811"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p170 = load_module("p170_cf_mor_p0", HERE / "170_physics_reliability_gated_residual_moe.py")


def purged_folds(origins: np.ndarray, folds: int, window: int, horizon: int):
    """Yield held rows and complement rows whose full windows do not overlap."""
    all_rows = np.arange(len(origins))
    for fold, held in enumerate(np.array_split(all_rows, folds)):
        held_start = int(origins[held].min()) - window
        held_end = int(origins[held].max()) + horizon - 1
        candidate_start = origins - window
        candidate_end = origins + horizon - 1
        separated = (candidate_end < held_start) | (candidate_start > held_end)
        train = all_rows[separated]
        if not len(train) or not len(held):
            raise RuntimeError(f"Fold {fold} is empty after purge")
        yield fold, train, held, held_start, held_end


def cross_fitted_varx(x, y, origins, alpha_grid, folds, window, horizon):
    fold_specs = list(purged_folds(origins, folds, window, horizon))
    alpha_rows = []
    best_alpha, best_pred, best_mse = None, None, float("inf")
    for alpha in alpha_grid:
        pred = np.empty_like(y, dtype=np.float32)
        fold_mse = []
        for fold, train, held, _, _ in fold_specs:
            model = Ridge(alpha=alpha, solver="lsqr").fit(x[train], y[train])
            pred[held] = model.predict(x[held]).astype(np.float32)
            fold_mse.append(float(np.mean((pred[held] - y[held]) ** 2)))
        mse = float(np.mean((pred - y) ** 2))
        alpha_rows.append({"alpha": alpha, "oof_mse": mse, **{f"fold_{i}_mse": v for i, v in enumerate(fold_mse)}})
        if mse < best_mse:
            best_alpha, best_pred, best_mse = float(alpha), pred.copy(), mse
    fold_rows = [
        {
            "fold": fold,
            "train_rows_after_purge": len(train),
            "held_rows": len(held),
            "held_information_start": held_start,
            "held_information_end": held_end,
        }
        for fold, train, held, held_start, held_end in fold_specs
    ]
    return best_alpha, best_pred, pd.DataFrame(alpha_rows), pd.DataFrame(fold_rows)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def train_mlp(x, target_z, args):
    split = int(len(x) * 0.85)
    train_ds = TensorDataset(torch.from_numpy(x[:split]), torch.from_numpy(target_z[:split]))
    cal_ds = TensorDataset(torch.from_numpy(x[split:]), torch.from_numpy(target_z[split:]))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    torch.manual_seed(args.seed)
    model = ResidualMLP(x.shape[1], target_z.shape[1], args.mlp_hidden_dim, args.mlp_dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)
    loss_fn = nn.HuberLoss(delta=1.0)

    def score():
        model.eval(); values = []
        with torch.no_grad():
            for xb, yb in cal_loader:
                values.append(float(loss_fn(model(xb), yb)))
        return float(np.mean(values))

    best_score = score()
    best_epoch, bad = 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [{"epoch": 0, "cal_huber": best_score}]
    for epoch in range(1, args.mlp_epochs + 1):
        model.train(); losses = []
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb) + args.mlp_shrink * pred.abs().mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); losses.append(float(loss.detach()))
        cal_score = score()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "cal_huber": cal_score})
        if cal_score < best_score - args.min_delta:
            best_score, best_epoch, bad = cal_score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.mlp_patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch, best_score


def predict_mlp(model, x, batch_size):
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval(); pred = []
    with torch.no_grad():
        for (xb,) in loader:
            pred.append(model(xb).numpy())
    return np.concatenate(pred)


def bounded_correction(raw, sigma, kappa):
    if kappa == 0:
        return np.zeros_like(raw, dtype=np.float32)
    scale = np.maximum(kappa * sigma, 1e-6)
    return (scale * np.tanh(raw / scale)).astype(np.float32)


def r2(true, pred):
    denominator = float(np.sum((true - true.mean()) ** 2))
    return float(1.0 - np.sum((true - pred) ** 2) / max(denominator, 1e-12))


def diagnostic_metrics(true, anchor, correction, tail_thresholds):
    pred = anchor + correction
    error = true - anchor
    tail_mask = np.abs(true) >= tail_thresholds[None, :, None]
    return {
        "residual_predictability_R2": r2(error, correction),
        "seq_MSE": float(np.mean((pred - true) ** 2)),
        "seq_R2": r2(true, pred),
        "lead24_MSE": float(np.mean((pred[..., -1] - true[..., -1]) ** 2)),
        "lead24_R2": r2(true[..., -1], pred[..., -1]),
        "tail_MSE_train_q95": float(np.mean((pred[tail_mask] - true[tail_mask]) ** 2)),
        "mean_abs_correction": float(np.mean(np.abs(correction))),
        "max_abs_correction": float(np.max(np.abs(correction))),
    }


def objective(metrics, anchor_metrics):
    return (
        metrics["seq_MSE"] / anchor_metrics["seq_MSE"]
        + 0.5 * metrics["lead24_MSE"] / anchor_metrics["lead24_MSE"]
        + 0.2 * metrics["tail_MSE_train_q95"] / anchor_metrics["tail_MSE_train_q95"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--anchor-alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--residual-alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--mlp-patience", type=int, default=5)
    parser.add_argument("--mlp-lr", type=float, default=5e-4)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-shrink", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    np.random.seed(args.seed)
    started = time.perf_counter()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "experiment_config.json").write_text(json.dumps(vars(args), ensure_ascii=True, indent=2), encoding="utf-8")

    data_args = argparse.Namespace(
        window=args.window, train_ratio=0.70, val_ratio=0.15, train_stride=args.train_stride,
        physics_forcing_mode="last_input", extreme_quantile=0.90,
    )
    data = p170.final4.build_enhanced_data(data_args, args.horizon, add_ode_prior=False)
    train_x, train_y, train_tide = p170.design_matrix(data["single_train"])
    val_x, val_y, val_tide = p170.design_matrix(data["single_val"])
    origins = np.asarray(data["single_train"].indices, dtype=int)

    best_alpha, oof_anchor, alpha_table, fold_table = cross_fitted_varx(
        train_x, train_y, origins, args.anchor_alphas, args.folds, args.window, args.horizon
    )
    alpha_table.to_csv(out / "anchor_alpha_oof_selection.csv", index=False)
    fold_table.to_csv(out / "purged_fold_audit.csv", index=False)
    final_anchor_model = Ridge(alpha=best_alpha, solver="lsqr").fit(train_x, train_y)
    val_anchor = final_anchor_model.predict(val_x).astype(np.float32)
    oof_error = (train_y - oof_anchor).astype(np.float32)
    sigma = np.maximum(oof_error.std(axis=0, keepdims=True), 1e-4).astype(np.float32)

    scaler = StandardScaler().fit(np.concatenate([train_x, oof_anchor], axis=1))
    oof_features = scaler.transform(np.concatenate([train_x, oof_anchor], axis=1)).astype(np.float32)
    val_features = scaler.transform(np.concatenate([val_x, val_anchor], axis=1)).astype(np.float32)
    candidates = {"no_correction": np.zeros_like(val_anchor, dtype=np.float32)}
    for alpha in args.residual_alphas:
        residual_model = Ridge(alpha=alpha, solver="lsqr").fit(oof_features, oof_error)
        candidates[f"ridge_alpha_{alpha:g}"] = residual_model.predict(val_features).astype(np.float32)

    mlp, history, best_epoch, best_mlp_score = train_mlp(
        oof_features, (oof_error / sigma).astype(np.float32), args
    )
    history.to_csv(out / "mlp_training_log.csv", index=False)
    candidates["mlp"] = (predict_mlp(mlp, val_features, args.batch_size) * sigma).astype(np.float32)

    train_y_3d = train_y.reshape(-1, 7, args.horizon).astype(np.float32)
    val_y_3d = val_y.reshape(-1, 7, args.horizon).astype(np.float32)
    val_anchor_3d = val_anchor.reshape(-1, 7, args.horizon).astype(np.float32)
    tail_thresholds = np.quantile(np.abs(train_y_3d), 0.95, axis=(0, 2)).astype(np.float32)
    cut = len(val_y_3d) // 2
    splits = {"calibration": slice(0, cut), "confirmation": slice(cut, len(val_y_3d))}
    rows = []
    for model_name, raw_flat in candidates.items():
        raw = raw_flat.reshape(-1, 7, args.horizon)
        kappa_values = [0.0] if model_name == "no_correction" else args.kappas
        for kappa in kappa_values:
            correction = bounded_correction(raw, sigma.reshape(1, 7, args.horizon), kappa)
            for split_name, split in splits.items():
                metrics = diagnostic_metrics(
                    val_y_3d[split], val_anchor_3d[split], correction[split], tail_thresholds
                )
                rows.append({"split": split_name, "residual_model": model_name, "kappa": kappa, **metrics})
    metrics = pd.DataFrame(rows)
    anchor_by_split = {
        split: metrics[(metrics["split"] == split) & (metrics["residual_model"] == "no_correction")].iloc[0]
        for split in splits
    }
    metrics["multiobjective_score"] = [objective(row, anchor_by_split[row["split"]]) for _, row in metrics.iterrows()]
    metrics.to_csv(out / "candidate_metrics.csv", index=False)
    cal = metrics[metrics["split"] == "calibration"].sort_values("multiobjective_score")
    selected = cal.iloc[0]
    confirm = metrics[
        (metrics["split"] == "confirmation")
        & (metrics["residual_model"] == selected["residual_model"])
        & np.isclose(metrics["kappa"], float(selected["kappa"]))
    ].iloc[0]
    confirm_anchor = anchor_by_split["confirmation"]
    go = bool(
        selected["residual_model"] != "no_correction"
        and selected["residual_predictability_R2"] > 0
        and confirm["residual_predictability_R2"] > 0
        and confirm["seq_MSE"] < confirm_anchor["seq_MSE"]
        and confirm["lead24_MSE"] <= confirm_anchor["lead24_MSE"] * 1.005
        and confirm["tail_MSE_train_q95"] <= confirm_anchor["tail_MSE_train_q95"] * 1.01
    )
    decision = {
        "status": "GO_TO_FS_GWN_OOF" if go else "STOP_BEFORE_FS_GWN_OOF",
        "proxy_scope": "purged blocked cross-fitted VARX residual diagnostic; not the final CF-MOR-GWN",
        "test_predictions_generated": False,
        "test_metrics_computed": False,
        "selected_anchor_alpha": best_alpha,
        "selected_residual_model": str(selected["residual_model"]),
        "selected_kappa": float(selected["kappa"]),
        "calibration_residual_R2": float(selected["residual_predictability_R2"]),
        "confirmation_residual_R2": float(confirm["residual_predictability_R2"]),
        "confirmation_sequence_MSE_delta": float(confirm["seq_MSE"] - confirm_anchor["seq_MSE"]),
        "confirmation_lead24_MSE_delta": float(confirm["lead24_MSE"] - confirm_anchor["lead24_MSE"]),
        "confirmation_tail_MSE_delta": float(confirm["tail_MSE_train_q95"] - confirm_anchor["tail_MSE_train_q95"]),
        "mlp_best_epoch": best_epoch,
        "mlp_best_internal_score": best_mlp_score,
        "runtime_seconds": time.perf_counter() - started,
    }
    (out / "P0_DECISION.json").write_text(json.dumps(decision, ensure_ascii=True, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "oof_diagnostic_predictions.npz",
        train_origins=origins,
        train_true=train_y_3d,
        oof_anchor=oof_anchor.reshape(-1, 7, args.horizon),
        oof_error=oof_error.reshape(-1, 7, args.horizon),
        val_true=val_y_3d,
        val_anchor=val_anchor_3d,
        tail_thresholds=tail_thresholds,
    )
    print(json.dumps(decision, ensure_ascii=True, indent=2))
    print(cal.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
