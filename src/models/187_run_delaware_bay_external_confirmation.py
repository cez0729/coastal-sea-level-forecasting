"""Run the preregistered Delaware Bay external confirmation.

The script reuses the exact fixed-support Graph WaveNet implementations from
experiment 134, changes only the frozen station/data configuration, and adds a
dynamic-node VARX implementation. Missing target windows are excluded by an
availability mask before any fitting or scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "delaware_bay_external_confirmation_20260812.json"
DATA_DIR = ROOT / "data" / "external_region_delaware_bay_2023_2025" / "processed"
OUT_DIR = ROOT / "results" / "delaware_bay_external_confirmation_20260812"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p134 = load_module("delaware_p134", HERE / "134_confirmatory_hsdt_orc_chronological_refit.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_data_modules(station_ids: list[str]) -> None:
    """Patch all independently imported v2/casual module instances."""
    seen: set[int] = set()

    def visit(obj, depth: int = 0):
        if obj is None or id(obj) in seen or depth > 5:
            return
        seen.add(id(obj))
        if hasattr(obj, "STATION_IDS"):
            obj.STATION_IDS = list(station_ids)
        if hasattr(obj, "DATA_DIR"):
            obj.DATA_DIR = DATA_DIR
        if hasattr(obj, "DEPTH_PATH"):
            obj.DEPTH_PATH = DATA_DIR / "gebco_station_depth.csv"
        if hasattr(obj, "COOPS_PATH"):
            obj.COOPS_PATH = DATA_DIR / "noaa_coops_met_station_hourly.csv"
        for name in ("v2", "v3", "v4", "final", "final4", "causal", "rolling", "p104", "priority1"):
            child = getattr(obj, name, None)
            if child is not None:
                visit(child, depth + 1)

    visit(p134)
    p134.configure_data_dir(DATA_DIR)


def valid_origin_mask(data: dict, station_ids: list[str], window: int, horizon: int) -> np.ndarray:
    water = pd.read_csv(DATA_DIR / "water_tide_residual_long.csv", parse_dates=["datetime"])
    water["station_id"] = water["station_id"].astype(str)
    matrix = water.pivot(index="datetime", columns="station_id", values="residual")
    times = pd.to_datetime(data["arrays"]["time"])
    complete = matrix.reindex(times)[station_ids].notna().all(axis=1).to_numpy(dtype=bool)
    bad = (~complete).astype(np.int64)
    prefix = np.concatenate([[0], np.cumsum(bad)])
    valid = np.zeros(len(times), dtype=bool)
    for target in range(window, len(times) - horizon + 1):
        valid[target] = prefix[target + horizon] - prefix[target - window] == 0
    return valid


def filter_datasets(data: dict, station_ids: list[str], window: int, horizon: int) -> dict:
    valid = valid_origin_mask(data, station_ids, window, horizon)
    audit = []
    for name in ("single_train", "single_val", "single_test", "multi_train", "multi_val", "multi_test"):
        dataset = data[name]
        before = len(dataset.indices)
        dataset.indices = np.asarray([int(index) for index in dataset.indices if valid[int(index)]], dtype=np.int64)
        after = len(dataset.indices)
        if after == 0:
            raise RuntimeError(f"No valid forecast origins remain in {name}")
        audit.append({"dataset": name, "origins_before_missing_filter": before, "origins_after_missing_filter": after, "removed": before - after})
    data["missing_filter_audit"] = pd.DataFrame(audit)
    return data


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    return float(1.0 - np.sum((true - pred) ** 2) / max(denominator, 1e-12))


def score(true: np.ndarray, pred: np.ndarray, train_threshold: float) -> dict:
    terminal_error = pred[..., -1] - true[..., -1]
    mask = np.abs(true) >= train_threshold
    return {
        "seq_residual_R2": r2_score(true, pred),
        "last_residual_R2": r2_score(true[..., -1], pred[..., -1]),
        "last_residual_RMSE": float(np.sqrt(np.mean(terminal_error**2))),
        "last_residual_MAE": float(np.mean(np.abs(terminal_error))),
        "descriptive_train_q95_residual_R2": r2_score(true[mask], pred[mask]) if int(mask.sum()) >= 10 else np.nan,
        "descriptive_train_q95_n": int(mask.sum()),
        "train_abs_residual_q95": float(train_threshold),
    }


def per_lead_station_rows(seed: int | str, model: str, true: np.ndarray, pred: np.ndarray, station_ids: list[str]):
    lead_rows = []
    station_rows = []
    for lead in range(true.shape[-1]):
        lead_rows.append({"seed": seed, "model": model, "lead_hour": lead + 1, "residual_R2": r2_score(true[..., lead], pred[..., lead]), "RMSE": float(np.sqrt(np.mean((true[..., lead] - pred[..., lead]) ** 2)))})
    for station_index, station_id in enumerate(station_ids):
        station_rows.append({"seed": seed, "model": model, "station_id": station_id, "sequence_residual_R2": r2_score(true[:, station_index], pred[:, station_index]), "lead24_residual_R2": r2_score(true[:, station_index, -1], pred[:, station_index, -1])})
    return lead_rows, station_rows


def design_matrix(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, targets, tides = [], [], []
    for target in dataset.indices:
        target = int(target)
        history = dataset.x_scaled[target - dataset.window:target]
        rows.append(np.concatenate([
            history[:, :, 0].reshape(-1),
            history[-1].reshape(-1),
            history.mean(axis=0).reshape(-1),
        ]))
        targets.append(dataset.residual[target:target + dataset.horizon].T.reshape(-1))
        tides.append(dataset.tide[target:target + dataset.horizon].T)
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64), np.asarray(tides, dtype=np.float64)


def run_varx(data: dict, station_ids: list[str], config: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("VARX: building frozen train/validation/test design matrices", flush=True)
    train_x, train_y, _ = design_matrix(data["single_train"])
    val_x, val_y, _ = design_matrix(data["single_val"])
    test_x, test_y, test_tide = design_matrix(data["single_test"])
    print(f"VARX: train={train_x.shape}, validation={val_x.shape}, test={test_x.shape}", flush=True)
    rows = []
    best_alpha, best_mse = None, float("inf")
    for alpha in config["models"]["varx_ridge"]["alpha_grid"]:
        print(f"VARX: fitting validation candidate alpha={alpha}", flush=True)
        model = Ridge(alpha=float(alpha), solver="lsqr").fit(train_x, train_y)
        mse = float(np.mean((model.predict(val_x) - val_y) ** 2))
        rows.append({"alpha": float(alpha), "validation_sequence_MSE": mse})
        if mse < best_mse:
            best_alpha, best_mse = float(alpha), mse
    model = Ridge(alpha=best_alpha, solver="lsqr").fit(train_x, train_y)
    nodes = len(station_ids)
    horizon = data["single_test"].horizon
    true = test_y.reshape(-1, nodes, horizon).astype(np.float32)
    pred = model.predict(test_x).reshape(true.shape).astype(np.float32)
    train_threshold = float(np.quantile(np.abs(data["arrays"]["residual"][:p134.rolling.time_index(data["arrays"]["time"], "2025-01-01")]), 0.95))
    metrics = {"seed": "deterministic", "model": "varx_ridge", "best_alpha": best_alpha, "best_validation_MSE": best_mse, **score(true, pred, train_threshold)}
    pd.DataFrame(rows).to_csv(output_dir / "alpha_selection.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    indices = np.asarray(data["single_test"].indices, dtype=np.int64)
    times = pd.to_datetime(data["arrays"]["time"])[indices].to_numpy(dtype="datetime64[ns]")
    np.savez_compressed(output_dir / "predictions.npz", pred_residual=pred, true_residual=true, target_tide=test_tide.astype(np.float32), target_origin_time=times, station_ids=np.asarray(station_ids))
    np.savez_compressed(output_dir / "frozen_coefficients.npz", coef=model.coef_, intercept=model.intercept_, alpha=np.asarray([best_alpha]), station_ids=np.asarray(station_ids))
    lead, station = per_lead_station_rows("deterministic", "varx_ridge", true, pred, station_ids)
    pd.DataFrame(lead).to_csv(output_dir / "per_lead.csv", index=False)
    pd.DataFrame(station).to_csv(output_dir / "per_station.csv", index=False)
    return metrics


def make_training_args(cli, config: dict) -> argparse.Namespace:
    model = config["models"]["eta_only_fs_gwn"]
    multi = config["models"]["multistate_fs_gwn"]
    return argparse.Namespace(
        output_dir=str(cli.output_dir), data_dir=str(DATA_DIR),
        fold_train_end="2025-01-01", fold_val_end="2025-07-01", fold_test_end="2026-01-01",
        untouched_holdout=True, horizon=24, window=24, train_stride=8,
        fixed_graph_type="distance", hidden_dim=int(model["hidden_dim"]), diffusion_steps=int(model["diffusion_steps"]),
        gwn_blocks=int(model["blocks"]), dropout=float(model["dropout"]), batch_size=cli.batch_size,
        epochs=cli.epochs, patience=cli.patience, min_delta=1e-5, lr=5e-4,
        weight_decay=1e-5, grad_clip=1.0, aux_weight=float(multi["aux_weight"]),
        last_step_weight=float(multi["last_step_weight"]), physics_lambda=0.0,
        physics_warmup_epochs=8, physics_ramp_epochs=14, physics_lr_mult=0.5,
        ode_coef_l2=1e-5, physics_forcing_mode="last_input", print_every=5,
        epoch_checkpoint_every=1, cpu_threads=cli.cpu_threads,
        extreme_quantile=0.90, event_quantile=0.95, resume=cli.resume,
    )


def run_neural_seed(seed: int, args: argparse.Namespace, config: dict, station_ids: list[str], output_dir: Path, prepared_data: dict | None = None) -> list[dict]:
    seed_dir = output_dir / f"seed_{seed}"
    final_metrics = seed_dir / "metrics.csv"
    if args.resume and final_metrics.exists() and (seed_dir / "predictions.npz").exists():
        return pd.read_csv(final_metrics).to_dict("records")
    seed_dir.mkdir(parents=True, exist_ok=True)
    data = prepared_data
    if data is None:
        data = p134.rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
        data = filter_datasets(data, station_ids, args.window, args.horizon)
    data["missing_filter_audit"].to_csv(seed_dir / "missing_origin_filter_audit.csv", index=False)
    experts = p134.train_experts(seed, data, args, torch.device("cuda" if torch.cuda.is_available() else "cpu"), seed_dir)
    weights = np.full(args.horizon, float(config["models"]["hs_dt_gwn"]["lead_1_to_23_multistate_weight"]), dtype=np.float64)
    weights[-1] = float(config["models"]["hs_dt_gwn"]["lead_24_multistate_weight"])
    hsdt = experts["eta"] + weights[None, None, :] * (experts["multi"] - experts["eta"])
    train_end = p134.rolling.time_index(data["arrays"]["time"], args.fold_train_end)
    train_threshold = float(np.quantile(np.abs(data["arrays"]["residual"][:train_end]), 0.95))
    predictions = {"gwn_eta_only": experts["eta"], "gwn_multistate_no_physics": experts["multi"], "hs_dt_gwn": hsdt}
    rows, lead_rows, station_rows = [], [], []
    for model, pred in predictions.items():
        rows.append({"seed": seed, "model": model, **score(experts["true"], pred, train_threshold)})
        lead, station = per_lead_station_rows(seed, model, experts["true"], pred, station_ids)
        lead_rows.extend(lead)
        station_rows.extend(station)
    pd.DataFrame(rows).to_csv(final_metrics, index=False)
    pd.DataFrame(lead_rows).to_csv(seed_dir / "per_lead.csv", index=False)
    pd.DataFrame(station_rows).to_csv(seed_dir / "per_station.csv", index=False)
    indices = np.asarray(data["single_test"].indices, dtype=np.int64)
    times = pd.to_datetime(data["arrays"]["time"])[indices].to_numpy(dtype="datetime64[ns]")
    np.savez_compressed(seed_dir / "predictions.npz", **predictions, true_residual=experts["true"], target_tide=experts["tide"], target_origin_time=times, hsdt_multistate_weights=weights, station_ids=np.asarray(station_ids))
    return rows


def merge_results(output_dir: Path, seeds: list[int]) -> None:
    runs, leads, stations = [], [], []
    varx = output_dir / "varx" / "metrics.csv"
    if varx.exists():
        runs.append(pd.read_csv(varx))
        leads.append(pd.read_csv(output_dir / "varx" / "per_lead.csv"))
        stations.append(pd.read_csv(output_dir / "varx" / "per_station.csv"))
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        if (seed_dir / "metrics.csv").exists():
            runs.append(pd.read_csv(seed_dir / "metrics.csv"))
            leads.append(pd.read_csv(seed_dir / "per_lead.csv"))
            stations.append(pd.read_csv(seed_dir / "per_station.csv"))
    if not runs:
        raise RuntimeError("No completed results to merge")
    all_runs = pd.concat(runs, ignore_index=True)
    all_runs.to_csv(output_dir / "all_runs.csv", index=False)
    pd.concat(leads, ignore_index=True).to_csv(output_dir / "per_lead_all.csv", index=False)
    pd.concat(stations, ignore_index=True).to_csv(output_dir / "per_station_all.csv", index=False)
    neural = all_runs[all_runs["model"] != "varx_ridge"]
    metrics = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "descriptive_train_q95_residual_R2"]
    summary = neural.groupby("model")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(value) for value in column if value) for column in summary.columns.to_flat_index()]
    if (all_runs["model"] == "varx_ridge").any():
        varx_row = all_runs[all_runs["model"] == "varx_ridge"][["model", *metrics]].copy()
        for metric in metrics:
            varx_row[f"{metric}_mean"] = varx_row[metric]
            varx_row[f"{metric}_std"] = 0.0
            varx_row[f"{metric}_count"] = 1
        summary = pd.concat([summary, varx_row[summary.columns]], ignore_index=True)
    summary.to_csv(output_dir / "mean_std.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    parser.add_argument("--models", choices=["all", "varx", "neural"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 2025, 3407])
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    cli = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    station_frame = pd.read_csv(DATA_DIR / "station_order.csv", dtype={"station_id": str})
    station_ids = station_frame["station_id"].tolist()
    minimum = int(config["station_exclusion_rule"]["minimum_retained_stations"])
    if len(station_ids) < minimum:
        raise RuntimeError(f"Only {len(station_ids)} retained stations; frozen minimum={minimum}")
    patch_data_modules(station_ids)
    print(f"Configured frozen external region with {len(station_ids)} stations", flush=True)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "protocol_sha256": sha256(CONFIG_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "station_ids": station_ids,
        "evidence_status": "untouched_spatial_external_confirmation",
        "test_target_used_for_selection": False
    }
    (cli.output_dir / "frozen_run_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if cli.mode == "merge":
        merge_results(cli.output_dir, cli.seeds)
        return
    args = make_training_args(cli, config)
    print("Building strict-causal 34-class feature arrays", flush=True)
    base_data = p134.rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    base_data = filter_datasets(base_data, station_ids, args.window, args.horizon)
    print(base_data["missing_filter_audit"].to_string(index=False), flush=True)
    base_data["missing_filter_audit"].to_csv(cli.output_dir / "missing_origin_filter_audit.csv", index=False)
    if cli.models in ("all", "varx"):
        run_varx(base_data, station_ids, config, cli.output_dir / "varx")
    if cli.models in ("all", "neural"):
        rows = []
        for seed in cli.seeds:
            rows.extend(run_neural_seed(seed, args, config, station_ids, cli.output_dir, prepared_data=base_data))
            pd.DataFrame(rows).to_csv(cli.output_dir / "neural_runs_progress.csv", index=False)
    merge_results(cli.output_dir, cli.seeds)


if __name__ == "__main__":
    main()
