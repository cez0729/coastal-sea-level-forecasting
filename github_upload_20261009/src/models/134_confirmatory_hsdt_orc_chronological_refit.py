from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 2024, 2025, 3407]
CONFIGS = [
    "gwn_eta_only",
    "gwn_multistate_no_physics",
    "hs_dt_gwn",
    "hsdt_zero_adapter",
    "hsdt_persistence_adapter",
    "orc_hs_dt_gwn",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p104 = load_module("confirmatory_p104", HERE / "104_priority2_physics_graph_wavenet.py")
rolling = load_module("confirmatory_rolling", HERE / "96_rolling_origin_validation.py")
adapter_impl = load_module("confirmatory_adapter", HERE / "109_ode_prior_gated_multistate_gwn.py")
final4 = p104.final4
priority1 = p104.priority1
v2 = p104.v2


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def configure_data_dir(value: str | Path) -> Path:
    data_dir = project_path(value)
    required = [
        "water_tide_residual_long.csv",
        "era5_station_hourly.csv",
        "surface_currents_station_daily.csv",
        "wave_direction_speed_station_3hourly.csv",
    ]
    missing = [str(data_dir / name) for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Missing processed holdout inputs:\n" + "\n".join(missing))
    rolling.causal.v2.DATA_DIR = data_dir
    rolling.causal.COOPS_PATH = data_dir / "noaa_coops_met_station_hourly.csv"
    return data_dir


def make_gwn(data: dict, args, num_states: int) -> torch.nn.Module:
    if num_states == 1:
        return priority1.GraphWaveNetForecaster(
            input_dim=data["feats"],
            adj=data["graph_priors"][args.fixed_graph_type],
            hidden_dim=args.hidden_dim,
            horizon=args.horizon,
            diffusion_steps=args.diffusion_steps,
            blocks=args.gwn_blocks,
            dropout=args.dropout,
        )
    return p104.GraphWaveNetMultistate(
        input_dim=data["feats"],
        adj=data["graph_priors"][args.fixed_graph_type],
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        num_states=num_states,
        diffusion_steps=args.diffusion_steps,
        blocks=args.gwn_blocks,
        dropout=args.dropout,
    )


def checkpoint_payload(model, data: dict, metadata: dict) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
        "feature_cols": data["feature_cols"],
        "physics_cols": data["physics_cols"],
        "graph_priors": data["graph_priors"],
        "state_scale": data["state_scale"],
        "delta_scale": data["delta_scale"],
        "x_scaler_state": data["x_scaler_state"],
        "physics_scaler_state": data["physics_scaler_state"],
    }


def training_metadata(seed: int, args, model: str) -> dict:
    return {
        "seed": seed,
        "model": model,
        "train_end_exclusive": args.fold_train_end,
        "validation_end_exclusive": args.fold_val_end,
        "test_end_exclusive": args.fold_test_end,
        "end_to_end_chronological_refit": True,
        "strict_causal_preprocessing": True,
        "future_residual_used_as_input": False,
        "physics_forcing_mode": "last_input",
        "hsdt_weights_frozen": "0.5 for leads 1-23; 1.0 multistate weight at lead 24",
        "orc_correction_scale_frozen": 1.0,
        "untouched_holdout": bool(args.untouched_holdout),
    }


def train_experts(seed: int, data: dict, args, device: torch.device, seed_dir: Path):
    eta_dir = seed_dir / "gwn_eta_only"
    eta_dir.mkdir(parents=True, exist_ok=True)
    p104.set_reproducible(seed, args.cpu_threads)
    eta_model = make_gwn(data, args, 1).to(device)
    eta_train = p104.make_loader(data["single_train"], args, True, seed)
    eta_val = p104.make_loader(data["single_val"], args, False, seed)
    eta_test = p104.make_loader(data["single_test"], args, False, seed)
    eta_history, eta_best, eta_timing = p104.train_eta_only(
        eta_model, eta_train, eta_val, args, device, eta_dir
    )
    eta_pred, eta_true, tide, eta_inference = p104.predict_eta(eta_model, eta_test, device)
    eta_history.to_csv(eta_dir / "training_log.csv", index=False)
    torch.save(
        checkpoint_payload(
            eta_model,
            data,
            {
                **training_metadata(seed, args, "gwn_eta_only"),
                "best_val_eta_loss": eta_best,
                **eta_timing,
            },
        ),
        eta_dir / "best_checkpoint.pt",
    )

    multi_dir = seed_dir / "gwn_multistate_no_physics"
    multi_dir.mkdir(parents=True, exist_ok=True)
    p104.set_reproducible(seed, args.cpu_threads)
    multi_model = make_gwn(data, args, 4).to(device)
    training_ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    multi_train = p104.make_loader(data["multi_train"], args, True, seed)
    multi_val = p104.make_loader(data["multi_val"], args, False, seed)
    multi_test = p104.make_loader(data["multi_test"], args, False, seed)
    multi_history, multi_best, multi_timing = p104.train_multistate(
        multi_model,
        training_ode,
        data,
        multi_train,
        multi_val,
        args,
        device,
        False,
        multi_dir,
    )
    multi_pred_states, multi_true_states, multi_tide, multi_inference = p104.predict_multistate(
        multi_model, multi_test, device
    )
    if not np.allclose(eta_true, multi_true_states[..., 0], atol=1e-6, rtol=0.0):
        raise RuntimeError("Eta-only and multistate test targets are not aligned")
    if not np.allclose(tide, multi_tide, atol=1e-6, rtol=0.0):
        raise RuntimeError("Eta-only and multistate tide arrays are not aligned")
    multi_history.to_csv(multi_dir / "training_log.csv", index=False)
    multi_metadata = {
        **training_metadata(seed, args, "gwn_multistate_no_physics"),
        "best_val_eta_loss": multi_best,
        **multi_timing,
    }
    torch.save(
        checkpoint_payload(multi_model, data, multi_metadata),
        multi_dir / "best_checkpoint.pt",
    )
    return {
        "eta_model": eta_model,
        "multi_model": multi_model,
        "eta": eta_pred.astype(np.float64),
        "multi": multi_pred_states[..., 0].astype(np.float64),
        "true": eta_true.astype(np.float64),
        "true_states": multi_true_states.astype(np.float64),
        "tide": tide.astype(np.float64),
        "timing": {
            "eta_training_seconds": eta_timing["training_seconds"],
            "eta_inference_seconds": eta_inference,
            "multi_training_seconds": multi_timing["training_seconds"],
            "multi_inference_seconds": multi_inference,
        },
    }


def train_adapter(
    seed: int,
    prior_mode: str,
    base_state: dict,
    data: dict,
    args,
    device: torch.device,
    seed_dir: Path,
) -> dict:
    run_dir = seed_dir / f"adapter_{prior_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter_args = copy.copy(args)
    adapter_args.epochs = args.adapter_epochs
    adapter_args.patience = args.adapter_patience
    adapter_args.lr = args.adapter_lr
    adapter_impl.set_seed(seed, args.cpu_threads)
    base = make_gwn(data, args, 4).to(device)
    base.load_state_dict(base_state)
    base.eval()
    ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    model = adapter_impl.ODEPriorGatedModel(
        base,
        ode,
        args.horizon,
        args.initial_gate,
        args.refiner_hidden,
        prior_mode,
    ).to(device)
    model.ode_adj = torch.tensor(
        data["graph_priors"][args.fixed_graph_type], dtype=torch.float32, device=device
    )
    train_loader = adapter_impl.make_loader(data["multi_train"], adapter_args, True, seed)
    val_loader = adapter_impl.make_loader(data["multi_val"], adapter_args, False, seed)
    test_loader = adapter_impl.make_loader(data["multi_test"], adapter_args, False, seed)
    history, best_val, seconds = adapter_impl.train_one(
        model, data, train_loader, val_loader, adapter_args, device, run_dir
    )
    outputs = adapter_impl.predict_all(model, test_loader, device)
    metadata = {
        **training_metadata(seed, args, f"adapter_{prior_mode}"),
        "prior_mode": prior_mode,
        "frozen_backbone_eval_during_adapter_training": True,
        "best_val_eta_loss": best_val,
        "training_seconds": seconds,
        "gate_values": model.gate_values().tolist(),
    }
    torch.save(checkpoint_payload(model, data, metadata), run_dir / "best_checkpoint.pt")
    return {
        "correction": outputs["final"][..., 0].astype(np.float64)
        - outputs["base"][..., 0].astype(np.float64),
        "base": outputs["base"][..., 0].astype(np.float64),
        "training_seconds": seconds,
        "gate": model.gate_values(),
    }


def training_thresholds(data: dict, args) -> np.ndarray:
    train_end = rolling.time_index(data["arrays"]["time"], args.fold_train_end)
    return np.quantile(data["arrays"]["residual"][:train_end], args.event_quantile, axis=0)


def score(true: np.ndarray, pred: np.ndarray, tide: np.ndarray, thresholds: np.ndarray) -> dict:
    metrics = final4.summarize_single(true, pred, tide)
    metrics.update(p104.operational_event_metrics(true, pred, thresholds))
    return metrics


def run_seed(seed: int, args, device: torch.device) -> list[dict]:
    output_dir = project_path(args.output_dir)
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    final_metrics = seed_dir / "metrics.csv"
    final_predictions = seed_dir / "predictions.npz"
    if args.resume and final_metrics.exists() and final_predictions.exists():
        return pd.read_csv(final_metrics).to_dict("records")

    data = rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    experts = train_experts(seed, data, args, device, seed_dir)
    weights = np.full(args.horizon, 0.5, dtype=np.float64)
    weights[-1] = 1.0
    hsdt = experts["eta"] + weights[None, None, :] * (experts["multi"] - experts["eta"])

    base_state = {
        key: value.detach().cpu().clone()
        for key, value in experts["multi_model"].state_dict().items()
    }
    adapters = {
        mode: train_adapter(seed, mode, base_state, data, args, device, seed_dir)
        for mode in ("learned_ode", "zero", "persistence")
    }
    for mode, payload in adapters.items():
        mismatch = float(np.abs(payload["base"] - experts["multi"]).max())
        if mismatch > 1e-6:
            raise RuntimeError(f"Frozen backbone mismatch for {mode}: {mismatch}")

    predictions = {
        "gwn_eta_only": experts["eta"],
        "gwn_multistate_no_physics": experts["multi"],
        "hs_dt_gwn": hsdt,
        "hsdt_zero_adapter": hsdt + adapters["zero"]["correction"],
        "hsdt_persistence_adapter": hsdt + adapters["persistence"]["correction"],
        "orc_hs_dt_gwn": hsdt + adapters["learned_ode"]["correction"],
    }
    thresholds = training_thresholds(data, args)
    rows = []
    for config, pred in predictions.items():
        rows.append(
            {
                "seed": seed,
                "config": config,
                **training_metadata(seed, args, config),
                **score(experts["true"], pred, experts["tide"], thresholds),
            }
        )
    pd.DataFrame(rows).to_csv(final_metrics, index=False)
    target_indices = np.asarray(data["single_test"].indices, dtype=np.int64)
    target_times = pd.to_datetime(data["arrays"]["time"])[target_indices].to_numpy(
        dtype="datetime64[ns]"
    )
    np.savez_compressed(
        final_predictions,
        **{name: pred for name, pred in predictions.items()},
        true_residual=experts["true"],
        target_tide=experts["tide"],
        target_origin_time=target_times,
        hsdt_multistate_weights=weights,
        station_ids=np.asarray(v2.STATION_IDS),
    )
    (seed_dir / "timing.json").write_text(
        json.dumps(
            {
                **experts["timing"],
                **{
                    f"adapter_{mode}_training_seconds": payload["training_seconds"]
                    for mode, payload in adapters.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"seed={seed} HS-DT seq={rows[2]['seq_residual_R2']:.4f} "
        f"terminal={rows[2]['last_residual_R2']:.4f}; "
        f"ORC seq={rows[-1]['seq_residual_R2']:.4f} "
        f"terminal={rows[-1]['last_residual_R2']:.4f}"
    )
    return rows


def exact_paired_rows(all_runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "event_CSI",
        "event_PR_AUC",
    ]
    comparisons = [
        ("hsdt_minus_eta", "hs_dt_gwn", "gwn_eta_only"),
        ("hsdt_minus_multistate", "hs_dt_gwn", "gwn_multistate_no_physics"),
        ("orc_minus_hsdt", "orc_hs_dt_gwn", "hs_dt_gwn"),
        ("orc_minus_zero", "orc_hs_dt_gwn", "hsdt_zero_adapter"),
        ("orc_minus_persistence", "orc_hs_dt_gwn", "hsdt_persistence_adapter"),
    ]
    pivot = all_runs.pivot(index="seed", columns="config", values=metrics)
    rows = []
    for name, candidate, baseline in comparisons:
        for metric in metrics:
            delta = pivot[(metric, candidate)] - pivot[(metric, baseline)]
            improvement = -delta if metric.endswith(("RMSE", "MAE")) else delta
            rows.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "mean_improvement": float(improvement.mean()),
                    "std_improvement": float(improvement.std(ddof=1)),
                    "wins": int((improvement > 0).sum()),
                    "count": int(improvement.notna().sum()),
                    "wilcoxon_greater_p": p104.exact_wilcoxon_greater(
                        improvement.to_numpy()
                    ),
                }
            )
    return pd.DataFrame(rows)


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    true_flat = np.asarray(true, dtype=np.float64).reshape(-1)
    pred_flat = np.asarray(pred, dtype=np.float64).reshape(-1)
    denominator = float(np.sum((true_flat - true_flat.mean()) ** 2))
    if denominator <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((true_flat - pred_flat) ** 2)) / denominator


def load_prediction_bundles(output_dir: Path, seeds: list[int]) -> list[dict]:
    bundles = []
    reference_true = None
    reference_times = None
    for seed in seeds:
        path = output_dir / f"seed_{seed}" / "predictions.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing seed predictions: {path}")
        with np.load(path, allow_pickle=False) as payload:
            bundle = {key: payload[key] for key in payload.files}
        if reference_true is None:
            reference_true = bundle["true_residual"]
            reference_times = bundle["target_origin_time"]
        else:
            if not np.array_equal(reference_times, bundle["target_origin_time"]):
                raise RuntimeError(f"Target timestamps differ for seed {seed}")
            if not np.allclose(reference_true, bundle["true_residual"], atol=0.0, rtol=0.0):
                raise RuntimeError(f"Targets differ for seed {seed}")
        bundles.append(bundle)
    return bundles


def write_detailed_diagnostics(bundles: list[dict], seeds: list[int], output_dir: Path) -> None:
    lead_rows = []
    station_rows = []
    station_ids = [str(value) for value in bundles[0]["station_ids"].tolist()]
    for seed, bundle in zip(seeds, bundles):
        true = bundle["true_residual"].astype(np.float64)
        for config in CONFIGS:
            pred = bundle[config].astype(np.float64)
            for lead in range(true.shape[-1]):
                error = pred[..., lead] - true[..., lead]
                lead_rows.append(
                    {
                        "seed": seed,
                        "config": config,
                        "lead_hour": lead + 1,
                        "residual_R2": r2_score(true[..., lead], pred[..., lead]),
                        "residual_RMSE": float(np.sqrt(np.mean(error**2))),
                        "residual_MAE": float(np.mean(np.abs(error))),
                    }
                )
            for station_index, station_id in enumerate(station_ids):
                error = pred[:, station_index, -1] - true[:, station_index, -1]
                station_rows.append(
                    {
                        "seed": seed,
                        "config": config,
                        "station_id": station_id,
                        "sequence_residual_R2": r2_score(
                            true[:, station_index, :], pred[:, station_index, :]
                        ),
                        "lead24_residual_R2": r2_score(
                            true[:, station_index, -1], pred[:, station_index, -1]
                        ),
                        "lead24_residual_RMSE": float(np.sqrt(np.mean(error**2))),
                    }
                )
    lead = pd.DataFrame(lead_rows)
    station = pd.DataFrame(station_rows)
    lead.to_csv(output_dir / "per_lead_by_seed.csv", index=False)
    station.to_csv(output_dir / "per_station_by_seed.csv", index=False)
    lead.groupby(["config", "lead_hour"])[
        ["residual_R2", "residual_RMSE", "residual_MAE"]
    ].agg(["mean", "std", "count"]).to_csv(output_dir / "per_lead_mean_std.csv")
    station.groupby(["config", "station_id"])[
        ["sequence_residual_R2", "lead24_residual_R2", "lead24_residual_RMSE"]
    ].agg(["mean", "std", "count"]).to_csv(output_dir / "per_station_mean_std.csv")


def moving_block_indices(
    rng: np.random.Generator, sample_count: int, block_size: int
) -> np.ndarray:
    block_size = min(max(1, int(block_size)), sample_count)
    block_count = int(np.ceil(sample_count / block_size))
    starts = rng.integers(0, sample_count - block_size + 1, size=block_count)
    indices = np.concatenate(
        [np.arange(start, start + block_size, dtype=np.int64) for start in starts]
    )
    return indices[:sample_count]


def block_bootstrap_comparisons(
    bundles: list[dict], args
) -> pd.DataFrame:
    comparisons = [
        ("hsdt_minus_eta", "hs_dt_gwn", "gwn_eta_only"),
        ("hsdt_minus_multistate", "hs_dt_gwn", "gwn_multistate_no_physics"),
        ("orc_minus_hsdt", "orc_hs_dt_gwn", "hs_dt_gwn"),
        ("orc_minus_zero", "orc_hs_dt_gwn", "hsdt_zero_adapter"),
        ("orc_minus_persistence", "orc_hs_dt_gwn", "hsdt_persistence_adapter"),
    ]
    rng = np.random.default_rng(20260728)
    rows = []
    for metric, horizon_slice in [("sequence_R2", slice(None)), ("lead24_R2", -1)]:
        true = bundles[0]["true_residual"][..., horizon_slice].astype(np.float64)
        sample_count = true.shape[0]
        flattened = true.reshape(sample_count, -1)
        per_sample_sum = flattened.sum(axis=1)
        per_sample_sum_sq = (flattened**2).sum(axis=1)
        values_per_sample = flattened.shape[1]
        for name, candidate, baseline in comparisons:
            error_advantage = []
            for bundle in bundles:
                candidate_error = (
                    bundle[candidate][..., horizon_slice].astype(np.float64) - true
                ).reshape(sample_count, -1)
                baseline_error = (
                    bundle[baseline][..., horizon_slice].astype(np.float64) - true
                ).reshape(sample_count, -1)
                error_advantage.append(
                    (baseline_error**2).sum(axis=1) - (candidate_error**2).sum(axis=1)
                )
            error_advantage = np.stack(error_advantage, axis=0)
            full_sst = float(
                per_sample_sum_sq.sum()
                - per_sample_sum.sum() ** 2 / (sample_count * values_per_sample)
            )
            point_deltas = error_advantage.sum(axis=1) / full_sst
            replicates = np.empty(args.bootstrap_replicates, dtype=np.float64)
            for replicate in range(args.bootstrap_replicates):
                indices = moving_block_indices(
                    rng, sample_count, args.bootstrap_block_hours
                )
                sampled_seeds = rng.integers(0, len(bundles), size=len(bundles))
                total_sum = float(per_sample_sum[indices].sum())
                total_sum_sq = float(per_sample_sum_sq[indices].sum())
                sst = total_sum_sq - total_sum**2 / (len(indices) * values_per_sample)
                seed_deltas = error_advantage[sampled_seeds][:, indices].sum(axis=1) / sst
                replicates[replicate] = float(seed_deltas.mean())
            rows.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "point_mean_delta": float(point_deltas.mean()),
                    "ci95_low": float(np.quantile(replicates, 0.025)),
                    "ci95_high": float(np.quantile(replicates, 0.975)),
                    "bootstrap_probability_positive": float(np.mean(replicates > 0.0)),
                    "block_hours": int(args.bootstrap_block_hours),
                    "replicates": int(args.bootstrap_replicates),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    bootstrap: pd.DataFrame,
    args,
    output_dir: Path,
) -> None:
    lookup = summary.set_index("config")
    paired_lookup = paired.set_index(["comparison", "metric"])
    bootstrap_lookup = bootstrap.set_index(["comparison", "metric"])
    claim = (
        "未触碰的独立时间留出验证"
        if args.untouched_holdout
        else "端到端按时间重训回测；该时期在模型设计过程中已被查看，因此不是未触碰独立留出集"
    )
    hs_eta_boot = bootstrap_lookup.loc[("hsdt_minus_eta", "sequence_R2")]
    hs_multi_boot = bootstrap_lookup.loc[("hsdt_minus_multistate", "sequence_R2")]
    orc_hs_boot = bootstrap_lookup.loc[("orc_minus_hsdt", "sequence_R2")]
    hs_trajectory_supported = bool(
        hs_eta_boot["ci95_low"] > 0.0 and hs_multi_boot["ci95_low"] > 0.0
    )
    orc_trajectory_supported = bool(orc_hs_boot["ci95_low"] > 0.0)
    model_labels = {
        "gwn_eta_only": "Eta-only FS-GWN",
        "gwn_multistate_no_physics": "Multistate FS-GWN (no physics)",
        "hs_dt_gwn": "HS-DT-GWN",
        "hsdt_zero_adapter": "HS-DT + zero adapter",
        "hsdt_persistence_adapter": "HS-DT + persistence adapter",
        "orc_hs_dt_gwn": "ORC-HS-DT-GWN",
    }
    lines = [
        "# HS-DT-GWN / ORC-HS-DT-GWN 按时间重训验证",
        "",
        f"证据等级：**{claim}**。",
        "",
        f"- 训练期结束：{args.fold_train_end}",
        f"- 验证期结束：{args.fold_val_end}",
        f"- 测试期结束：{args.fold_test_end}",
        "- 每个seed都从头训练Eta-only和Multistate两个专家，不读取旧预测结果。",
        "- HS-DT融合权重和ORC修正比例在测试评估前冻结。",
        "- learned-ODE与相同refiner/gate的zero和persistence控制比较；learned-ODE额外训练小型ODE模块，因此三者并非总参数完全相同。",
        "",
        "| 模型 | Sequence residual R2 | Lead-24 residual R2 | Lead-24 RMSE (m) | 描述性q95 R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        row = lookup.loc[config]
        lines.append(
            f"| {model_labels[config]} | {row['seq_residual_R2_mean']:.6f} +/- {row['seq_residual_R2_std']:.6f} "
            f"| {row['last_residual_R2_mean']:.6f} +/- {row['last_residual_R2_std']:.6f} "
            f"| {row['last_residual_RMSE_mean']:.6f} +/- {row['last_residual_RMSE_std']:.6f} "
            f"| {row['extreme_abs_q95_residual_R2_mean']:.6f} +/- {row['extreme_abs_q95_residual_R2_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 五seed配对证据",
            "",
            "| 比较 | 指标 | 平均改善 | 获胜seed | 单侧精确p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in paired[
        paired["metric"].isin(["seq_residual_R2", "last_residual_R2"])
    ].iterrows():
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['mean_improvement']:.6f} "
            f"| {int(row['wins'])}/{int(row['count'])} | {row['wilcoxon_greater_p']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"## 分层移动块bootstrap（{args.bootstrap_block_hours}小时块）",
            "",
            "| 比较 | 指标 | 平均差值 | 95% CI | P(delta > 0) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in bootstrap.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['point_mean_delta']:.6f} "
            f"| [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] "
            f"| {row['bootstrap_probability_positive']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 论文准入判断",
            "",
            f"- HS-DT轨迹改善是否在本次重训回测中得到支持：**{'是' if hs_trajectory_supported else '否'}**。",
            f"- ORC相对HS-DT的轨迹改善是否得到支持：**{'是' if orc_trajectory_supported else '否'}**。",
            f"- HS-DT相对Eta-only的Lead-24平均变化为{paired_lookup.loc[('hsdt_minus_eta', 'last_residual_R2'), 'mean_improvement']:.6f}，"
            f"仅{int(paired_lookup.loc[('hsdt_minus_eta', 'last_residual_R2'), 'wins'])}/5 seed获胜，不能声称终点稳定提升。",
            f"- ORC相对HS-DT的Lead-24平均变化为{paired_lookup.loc[('orc_minus_hsdt', 'last_residual_R2'), 'mean_improvement']:.6f}，"
            "当前结果不支持把ORC作为HS-DT的稳健终点升级。",
            "- HS-DT可以作为轨迹融合的稳健性实验写入补充材料，但本测试期已在研究中被查看，不能称为真正独立holdout。",
            "- ORC仅保留为探索性尾部/事件修正候选，不进入当前投稿主模型。",
            "- 只有当`untouched_holdout=true`且数据清单证明测试期在模型设计时不可见，才能升级为确认性独立验证主张。",
        ]
    )
    (output_dir / "CONFIRMATORY_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    lookup = summary.set_index("config")
    labels = ["Eta", "Multi", "HS-DT", "Zero", "Persist", "ORC"]
    colors = ["#596780", "#2B7A68", "#B45A4A", "#8A8A8A", "#C98743", "#2878B5"]
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    for axis, metric, title in [
        (axes[0], "seq_residual_R2", "Trajectory residual $R^2$"),
        (axes[1], "last_residual_R2", "Lead-24 residual $R^2$"),
        (axes[2], "last_residual_RMSE", "Lead-24 RMSE (m)"),
    ]:
        means = [lookup.loc[config, f"{metric}_mean"] for config in CONFIGS]
        stds = [lookup.loc[config, f"{metric}_std"] for config in CONFIGS]
        axis.bar(np.arange(len(CONFIGS)), means, yerr=stds, color=colors, capsize=3)
        axis.set_xticks(np.arange(len(CONFIGS)), labels, rotation=20, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle("End-to-end chronological refit of HS-DT and ORC-HS-DT")
    figure.tight_layout()
    figure.savefig(output_dir / "confirmatory_hsdt_orc_summary.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def merge(args) -> None:
    output_dir = project_path(args.output_dir)
    files = [output_dir / f"seed_{seed}" / "metrics.csv" for seed in args.seeds]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing seed metrics:\n" + "\n".join(missing))
    all_runs = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    all_runs.to_csv(output_dir / "all_runs.csv", index=False)
    metrics = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "last_residual_MAE",
        "extreme_abs_q95_residual_R2",
        "event_CSI",
        "event_PR_AUC",
    ]
    summary = all_runs.groupby("config")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part)
        for column in summary.columns.to_flat_index()
    ]
    summary.to_csv(output_dir / "mean_std.csv", index=False)
    paired = exact_paired_rows(all_runs)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    bundles = load_prediction_bundles(output_dir, args.seeds)
    write_detailed_diagnostics(bundles, args.seeds, output_dir)
    bootstrap = block_bootstrap_comparisons(bundles, args)
    bootstrap.to_csv(output_dir / "block_bootstrap_comparisons.csv", index=False)
    plot_summary(summary, output_dir)
    write_report(summary, paired, bootstrap, args, output_dir)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(bootstrap.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end chronological refit validation for HS-DT-GWN and ORC-HS-DT-GWN."
    )
    parser.add_argument("--mode", choices=["run", "merge"], default="run")
    parser.add_argument("--output-dir", default="results/confirmatory_hsdt_orc_refit_2025_h2")
    parser.add_argument("--data-dir", default="data/processed_multiyear_2023_2025")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--fold-train-end", default="2025-01-01")
    parser.add_argument("--fold-val-end", default="2025-07-01")
    parser.add_argument("--fold-test-end", default="2026-01-01")
    parser.add_argument(
        "--untouched-holdout", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", choices=["identity", "distance", "corr"], default="distance")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--gwn-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--adapter-epochs", type=int, default=12)
    parser.add_argument("--adapter-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--adapter-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.08)
    parser.add_argument("--last-step-weight", type=float, default=0.20)
    parser.add_argument("--physics-lambda", type=float, default=0.0002)
    parser.add_argument("--physics-warmup-epochs", type=int, default=8)
    parser.add_argument("--physics-ramp-epochs", type=int, default=14)
    parser.add_argument("--physics-lr-mult", type=float, default=0.5)
    parser.add_argument("--ode-coef-l2", type=float, default=1e-5)
    parser.add_argument("--prior-loss-weight", type=float, default=0.05)
    parser.add_argument("--gate-smooth-weight", type=float, default=0.01)
    parser.add_argument("--ode-reg-weight", type=float, default=1e-5)
    parser.add_argument("--initial-gate", type=float, default=0.80)
    parser.add_argument("--refiner-hidden", type=int, default=128)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--bootstrap-block-hours", type=int, default=168)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--physics-forcing-mode", choices=["last_input"], default="last_input")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--epoch-checkpoint-every", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = configure_data_dir(args.data_dir)
    config = {
        **vars(args),
        "data_dir": str(data_dir.relative_to(ROOT) if data_dir.is_relative_to(ROOT) else data_dir),
        "validation_label": (
            "untouched_independent_chronological_holdout"
            if args.untouched_holdout
            else "end_to_end_chronological_refit_not_untouched"
        ),
        "frozen_backbone_eval_during_adapter_training": True,
    }
    if args.mode == "merge":
        (output_dir / "final_experiment_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        merge(args)
        return
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}; seeds={args.seeds}; data={data_dir}")
    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "run_config.json").write_text(
            json.dumps({**config, "seeds": [seed]}, indent=2), encoding="utf-8"
        )
        run_seed(seed, args, device)
    merge(args)


if __name__ == "__main__":
    main()
