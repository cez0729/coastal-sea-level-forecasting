from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS = ROOT / "results" / "priority12_physics_graph_wavenet"
DEFAULT_OUT = ROOT / "results" / "publication_clean_evidence"
SEEDS = [42, 123, 2024, 2025, 3407]
MODELS = ["persistence", "gwn_eta_only", "gwn_multistate_no_physics", "gwn_multistate_physics"]
LABELS = {
    "persistence": "Persistence",
    "gwn_eta_only": "FS-GWN eta-only",
    "gwn_multistate_no_physics": "FS-GWN multistate",
    "gwn_multistate_physics": "FS-GWN multistate + physics",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def r2_rmse_mae(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    error = pred - true
    sse = float(np.sum(error**2))
    sst = float(np.sum((true - true.mean()) ** 2))
    return {
        "R2": 1.0 - sse / max(sst, 1e-12),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "count": int(true.size),
    }


def prediction_path(source: Path, seed: int, config: str) -> Path:
    return source / f"seed_{seed}" / "horizon_24h" / config / "predictions.npz"


def load_predictions(source: Path, seeds: list[int]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    by_model: dict[str, list[np.ndarray]] = {
        model: [] for model in MODELS if model != "persistence"
    }
    reference_true = None
    reference_tide = None
    for seed in seeds:
        for model in by_model:
            path = prediction_path(source, seed, model)
            with np.load(path) as payload:
                if model == "gwn_eta_only":
                    pred = payload["pred_residual"]
                    true = payload["true_residual"]
                else:
                    pred = payload["pred_states"][..., 0]
                    true = payload["true_states"][..., 0]
                tide = payload["target_tide"]
            if reference_true is None:
                reference_true = true.astype(np.float64)
                reference_tide = tide.astype(np.float64)
            if not np.allclose(true, reference_true, atol=1e-6, rtol=0.0):
                raise RuntimeError(f"Target mismatch: {path}")
            if not np.allclose(tide, reference_tide, atol=1e-6, rtol=0.0):
                raise RuntimeError(f"Tide mismatch: {path}")
            by_model[model].append(pred.astype(np.float64))
    assert reference_true is not None and reference_tide is not None
    return {model: np.stack(values) for model, values in by_model.items()}, reference_true, reference_tide


def build_aligned_context(true: np.ndarray):
    final4 = load_module("final4_clean_evidence", ROOT / "数据整理" / "78_final_four_models_enhanced_data.py")
    args = SimpleNamespace(
        train_ratio=0.70,
        val_ratio=0.15,
        window=24,
        train_stride=8,
        physics_forcing_mode="last_input",
        extreme_quantile=0.90,
    )
    data = final4.build_enhanced_data(args, 24, add_ode_prior=False)
    dataset = data["single_test"]
    if len(dataset) != true.shape[0]:
        raise RuntimeError(f"Test alignment mismatch: dataset={len(dataset)}, predictions={true.shape[0]}")
    indices = dataset.indices.astype(np.int64)
    arrays = data["arrays"]
    origin_times = pd.to_datetime(np.asarray(arrays["time"])[indices])
    terminal_times = pd.to_datetime(np.asarray(arrays["time"])[indices + 23])
    persistence = np.stack([dataset.residual[int(index) - 1] for index in indices]).astype(np.float64)
    persistence = np.repeat(persistence[:, :, None], 24, axis=2)
    train_end = int(len(arrays["residual"]) * args.train_ratio)
    train_residual = np.asarray(arrays["residual"][:train_end], dtype=np.float64)
    return persistence, train_residual, origin_times, terminal_times


def main_metrics(predictions: dict[str, np.ndarray], true: np.ndarray, tide: np.ndarray) -> pd.DataFrame:
    rows = []
    for model, pred_by_seed in predictions.items():
        for seed_index in range(pred_by_seed.shape[0]):
            pred = pred_by_seed[seed_index]
            for target, yt, yp in [
                ("residual_sequence", true, pred),
                ("residual_terminal", true[..., -1], pred[..., -1]),
                ("total_level_sequence", true + tide, pred + tide),
                ("total_level_terminal", true[..., -1] + tide[..., -1], pred[..., -1] + tide[..., -1]),
            ]:
                rows.append({"model": model, "seed_index": seed_index, "target": target, **r2_rmse_mae(yt, yp)})
    return pd.DataFrame(rows)


def train_defined_strata(
    predictions: dict[str, np.ndarray], true: np.ndarray, train_residual: np.ndarray
) -> pd.DataFrame:
    q05 = np.quantile(train_residual, 0.05, axis=0)
    q95 = np.quantile(train_residual, 0.95, axis=0)
    abs_q95 = np.quantile(np.abs(train_residual), 0.95, axis=0)
    terminal = true[..., -1]
    masks = {
        "all": np.ones_like(terminal, dtype=bool),
        "train_q05_negative": terminal <= q05[None, :],
        "train_q95_positive": terminal >= q95[None, :],
        "train_abs_q95": np.abs(terminal) >= abs_q95[None, :],
        "central_90pct": (terminal > q05[None, :]) & (terminal < q95[None, :]),
    }
    rows = []
    for model, pred_by_seed in predictions.items():
        for seed_index, pred in enumerate(pred_by_seed):
            pred_terminal = pred[..., -1]
            for stratum, mask in masks.items():
                rows.append(
                    {
                        "model": model,
                        "seed_index": seed_index,
                        "stratum": stratum,
                        **r2_rmse_mae(terminal[mask], pred_terminal[mask]),
                    }
                )
    return pd.DataFrame(rows)


def monthly_metrics(predictions: dict[str, np.ndarray], true: np.ndarray, terminal_times: pd.DatetimeIndex) -> pd.DataFrame:
    month_values = terminal_times.to_period("M").astype(str)
    rows = []
    for model, pred_by_seed in predictions.items():
        for seed_index, pred in enumerate(pred_by_seed):
            for month in sorted(set(month_values)):
                mask = np.asarray(month_values == month)
                rows.append(
                    {
                        "model": model,
                        "seed_index": seed_index,
                        "month": month,
                        **r2_rmse_mae(true[mask, :, -1], pred[mask, :, -1]),
                    }
                )
    return pd.DataFrame(rows)


def station_effects(predictions: dict[str, np.ndarray], true: np.ndarray, station_ids: list[str]) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("multistate_minus_eta", "gwn_multistate_no_physics", "gwn_eta_only"),
        ("physics_minus_no_physics", "gwn_multistate_physics", "gwn_multistate_no_physics"),
    ]
    for name, candidate, baseline in comparisons:
        for station_index, station_id in enumerate(station_ids):
            deltas = []
            for seed_index in range(predictions[candidate].shape[0]):
                candidate_metric = r2_rmse_mae(
                    true[:, station_index, -1], predictions[candidate][seed_index, :, station_index, -1]
                )
                baseline_metric = r2_rmse_mae(
                    true[:, station_index, -1], predictions[baseline][seed_index, :, station_index, -1]
                )
                deltas.append(candidate_metric["R2"] - baseline_metric["R2"])
            rows.append(
                {
                    "comparison": name,
                    "station_id": station_id,
                    "R2_delta_mean": float(np.mean(deltas)),
                    "R2_delta_std": float(np.std(deltas, ddof=1)),
                    "seed_wins": int(np.sum(np.asarray(deltas) > 0)),
                    "seed_count": len(deltas),
                }
            )
    return pd.DataFrame(rows)


def moving_block_indices(rng: np.random.Generator, n: int, block_length: int) -> np.ndarray:
    blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]


def block_bootstrap(
    predictions: dict[str, np.ndarray], true: np.ndarray, replicates: int, block_length: int, seed: int
) -> pd.DataFrame:
    comparisons = [
        ("multistate_minus_eta", "gwn_multistate_no_physics", "gwn_eta_only"),
        ("physics_minus_no_physics", "gwn_multistate_physics", "gwn_multistate_no_physics"),
        ("multistate_minus_persistence", "gwn_multistate_no_physics", "persistence"),
    ]
    rng = np.random.default_rng(seed)
    rows = []
    for scope, yt, lead_slice in [
        ("sequence", true, (..., slice(None))),
        ("terminal", true[..., -1], (..., -1)),
    ]:
        y_by_origin = yt.reshape(yt.shape[0], -1)
        y_sum = y_by_origin.sum(axis=1)
        y_sumsq = (y_by_origin**2).sum(axis=1)
        y_count = y_by_origin.shape[1]
        for comparison, candidate, baseline in comparisons:
            candidate_error = predictions[candidate][lead_slice] - yt[None, ...]
            baseline_error = predictions[baseline][lead_slice] - yt[None, ...]
            candidate_sse = np.mean((candidate_error.reshape(candidate_error.shape[0], yt.shape[0], -1) ** 2).sum(axis=2), axis=0)
            baseline_sse = np.mean((baseline_error.reshape(baseline_error.shape[0], yt.shape[0], -1) ** 2).sum(axis=2), axis=0)
            observed_sst = float(y_sumsq.sum() - y_sum.sum() ** 2 / (len(y_sum) * y_count))
            observed = float((baseline_sse.sum() - candidate_sse.sum()) / observed_sst)
            samples = np.empty(replicates, dtype=np.float64)
            for index in range(replicates):
                selected = moving_block_indices(rng, len(y_sum), block_length)
                total_count = len(selected) * y_count
                sst = float(y_sumsq[selected].sum() - y_sum[selected].sum() ** 2 / total_count)
                samples[index] = float((baseline_sse[selected].sum() - candidate_sse[selected].sum()) / max(sst, 1e-12))
            rows.append(
                {
                    "comparison": comparison,
                    "scope": scope,
                    "block_length_origins": block_length,
                    "replicates": replicates,
                    "observed_R2_delta": observed,
                    "bootstrap_mean": float(samples.mean()),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "probability_delta_gt_zero": float(np.mean(samples > 0)),
                }
            )
    return pd.DataFrame(rows)


def plot_clean_evidence(main: pd.DataFrame, strata: pd.DataFrame, bootstrap: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        "persistence": "#767676",
        "gwn_eta_only": "#2878B5",
        "gwn_multistate_no_physics": "#2B7A68",
        "gwn_multistate_physics": "#B45A4A",
    }
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    residual = main[main["target"].isin(["residual_sequence", "residual_terminal"])]
    summary = residual.groupby(["model", "target"])["R2"].agg(["mean", "std"])
    x = np.arange(2)
    width = 0.19
    for offset, model in enumerate(MODELS):
        means = [summary.loc[(model, target), "mean"] for target in ["residual_sequence", "residual_terminal"]]
        stds = [summary.loc[(model, target), "std"] for target in ["residual_sequence", "residual_terminal"]]
        axes[0].bar(x + (offset - 1.5) * width, means, width, yerr=stds, color=colors[model], label=LABELS[model], capsize=2)
    axes[0].set_xticks(x, ["24-h trajectory", "Lead 24"])
    axes[0].set_ylabel("Residual $R^2$")
    axes[0].set_title("Matched five-seed benchmark")
    axes[0].grid(axis="y", alpha=0.2)

    tail = strata[strata["stratum"].isin(["central_90pct", "train_q05_negative", "train_q95_positive"])]
    tail_summary = tail.groupby(["model", "stratum"])["RMSE"].mean()
    strata_order = ["central_90pct", "train_q05_negative", "train_q95_positive"]
    for offset, model in enumerate(MODELS[1:]):
        values = [tail_summary.loc[(model, stratum)] for stratum in strata_order]
        axes[1].bar(np.arange(3) + (offset - 1) * 0.25, values, 0.25, color=colors[model], label=LABELS[model])
    axes[1].set_xticks(np.arange(3), ["Central 90%", "Low tail", "High tail"])
    axes[1].set_ylabel("Lead-24 RMSE (m)")
    axes[1].set_title("Training-defined residual strata")
    axes[1].grid(axis="y", alpha=0.2)

    boot = bootstrap[
        (bootstrap["scope"] == "terminal")
        & (bootstrap["comparison"].isin(["multistate_minus_eta", "physics_minus_no_physics"]))
    ].copy()
    y = np.arange(len(boot))
    err_low = boot["observed_R2_delta"] - boot["ci95_low"]
    err_high = boot["ci95_high"] - boot["observed_R2_delta"]
    axes[2].errorbar(boot["observed_R2_delta"], y, xerr=np.vstack([err_low, err_high]), fmt="o", color="#303030", capsize=3)
    axes[2].axvline(0.0, color="#888888", linewidth=1)
    axes[2].set_yticks(y, ["Multistate - eta", "Physics - no physics"])
    axes[2].set_xlabel("Lead-24 $R^2$ difference")
    axes[2].set_title("168-h moving-block bootstrap")
    axes[2].grid(axis="x", alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    figure.savefig(output_dir / "clean_evidence_summary.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_lead_skill(source: Path, output_dir: Path) -> None:
    data = pd.read_csv(source / "priority2_physics_gwn_per_horizon.csv")
    summary = data.groupby(["config", "lead_hour"])["residual_R2"].agg(["mean", "std"]).reset_index()
    colors = {
        "gwn_eta_only": "#2878B5",
        "gwn_multistate_no_physics": "#2B7A68",
        "gwn_multistate_physics": "#B45A4A",
    }
    figure, axis = plt.subplots(figsize=(9.2, 5.0))
    for model in ["gwn_eta_only", "gwn_multistate_no_physics", "gwn_multistate_physics"]:
        subset = summary[summary["config"] == model]
        x = subset["lead_hour"].to_numpy()
        mean = subset["mean"].to_numpy()
        std = subset["std"].to_numpy()
        axis.plot(x, mean, color=colors[model], linewidth=2.2, label=LABELS[model])
        axis.fill_between(x, mean - std, mean + std, color=colors[model], alpha=0.12)
    axis.set_xlabel("Forecast lead (h)")
    axis.set_ylabel("Residual $R^2$")
    axis.set_xlim(1, 24)
    axis.set_title("Lead-dependent skill under matched FS-GWN training")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "matched_fs_gwn_per_horizon.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def write_report(
    main: pd.DataFrame,
    strata: pd.DataFrame,
    monthly: pd.DataFrame,
    station: pd.DataFrame,
    bootstrap: pd.DataFrame,
    output_dir: Path,
) -> None:
    metric_summary = main.groupby(["model", "target"])[["R2", "RMSE", "MAE"]].agg(["mean", "std"])
    terminal_boot = bootstrap[bootstrap["scope"] == "terminal"].set_index("comparison")
    station_aux = station[station["comparison"] == "multistate_minus_eta"]
    station_phys = station[station["comparison"] == "physics_minus_no_physics"]
    month_r2 = monthly.groupby(["model", "month"])["R2"].mean().unstack(0)
    month_aux_wins = int((month_r2["gwn_multistate_no_physics"] > month_r2["gwn_eta_only"]).sum())
    month_phys_wins = int((month_r2["gwn_multistate_physics"] > month_r2["gwn_multistate_no_physics"]).sum())
    tail_summary = strata.groupby(["model", "stratum"])[["RMSE", "MAE", "bias"]].mean()

    def value(model: str, target: str, metric: str) -> float:
        return float(metric_summary.loc[(model, target), (metric, "mean")])

    lines = [
        "# 投稿精简证据补充实验",
        "",
        "本分析只使用锁定的五种子 fixed-support Graph WaveNet 预测，不重新选择模型或超参数。",
        "",
        "## 核心结果",
        "",
        f"- Eta-only FS-GWN: sequence R2={value('gwn_eta_only', 'residual_sequence', 'R2'):.6f}, lead-24 R2={value('gwn_eta_only', 'residual_terminal', 'R2'):.6f}。",
        f"- Multistate FS-GWN: sequence R2={value('gwn_multistate_no_physics', 'residual_sequence', 'R2'):.6f}, lead-24 R2={value('gwn_multistate_no_physics', 'residual_terminal', 'R2'):.6f}。",
        f"- Multistate + physics: sequence R2={value('gwn_multistate_physics', 'residual_sequence', 'R2'):.6f}, lead-24 R2={value('gwn_multistate_physics', 'residual_terminal', 'R2'):.6f}。",
        "",
        "## 新增稳健性证据",
        "",
        f"- 168小时移动块bootstrap中，multistate相对eta-only的lead-24 R2差值为 {terminal_boot.loc['multistate_minus_eta', 'observed_R2_delta']:.6f}，95% CI [{terminal_boot.loc['multistate_minus_eta', 'ci95_low']:.6f}, {terminal_boot.loc['multistate_minus_eta', 'ci95_high']:.6f}]。",
        f"- Physics相对no-physics的lead-24 R2差值为 {terminal_boot.loc['physics_minus_no_physics', 'observed_R2_delta']:.6f}，95% CI [{terminal_boot.loc['physics_minus_no_physics', 'ci95_low']:.6f}, {terminal_boot.loc['physics_minus_no_physics', 'ci95_high']:.6f}]。",
        f"- Multistate终点提升在 {int((station_aux['R2_delta_mean'] > 0).sum())}/7 个站点平均为正；physics净效应在 {int((station_phys['R2_delta_mean'] > 0).sum())}/7 个站点平均为正。",
        f"- 按测试期自然月分组，multistate在 {month_aux_wins}/{month_r2.shape[0]} 个月优于eta-only；physics在 {month_phys_wins}/{month_r2.shape[0]} 个月优于no-physics。",
        f"- 训练期阈值定义的高残差尾部，eta-only/multistate/physics的lead-24 RMSE分别为 {tail_summary.loc[('gwn_eta_only', 'train_q95_positive'), 'RMSE']:.6f}/{tail_summary.loc[('gwn_multistate_no_physics', 'train_q95_positive'), 'RMSE']:.6f}/{tail_summary.loc[('gwn_multistate_physics', 'train_q95_positive'), 'RMSE']:.6f} m。",
        f"- 训练期阈值定义的低残差尾部，对应RMSE分别为 {tail_summary.loc[('gwn_eta_only', 'train_q05_negative'), 'RMSE']:.6f}/{tail_summary.loc[('gwn_multistate_no_physics', 'train_q05_negative'), 'RMSE']:.6f}/{tail_summary.loc[('gwn_multistate_physics', 'train_q05_negative'), 'RMSE']:.6f} m。",
        "",
        "## 论文使用边界",
        "",
        "- Graph WaveNet必须写成 fixed-support Graph WaveNet variant（FS-GWN），不能暗示实现了原论文的adaptive adjacency。",
        "- 训练期阈值分层可替代test-defined q95作为主要尾部证据；test-defined q95仅可留在补充材料。",
        "- 总水位指标只用于说明加回天文潮后的应用重建，不作为模型主贡献指标。",
        "- 这些是锁定预测上的补充评估，不是新的独立chronological holdout。",
    ]
    (output_dir / "CLEAN_EVIDENCE_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    source = Path(args.source_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, true, tide = load_predictions(source, args.seeds)
    persistence, train_residual, origin_times, terminal_times = build_aligned_context(true)
    predictions["persistence"] = np.repeat(persistence[None, ...], len(args.seeds), axis=0)

    main = main_metrics(predictions, true, tide)
    strata = train_defined_strata(predictions, true, train_residual)
    monthly = monthly_metrics(predictions, true, terminal_times)
    station_ids = [str(value) for value in pd.read_csv(source / "station_meta_used.csv")["station_id"].tolist()]
    station = station_effects(predictions, true, station_ids)
    bootstrap = block_bootstrap(predictions, true, args.bootstrap_replicates, args.block_length, args.random_seed)

    main.to_csv(output_dir / "main_metrics_all_seeds.csv", index=False)
    main.groupby(["model", "target"])[["R2", "RMSE", "MAE", "bias"]].agg(["mean", "std"]).to_csv(output_dir / "main_metrics_mean_std.csv")
    strata.to_csv(output_dir / "train_defined_strata_all_seeds.csv", index=False)
    strata.groupby(["model", "stratum"])[["R2", "RMSE", "MAE", "bias", "count"]].agg(["mean", "std"]).to_csv(output_dir / "train_defined_strata_mean_std.csv")
    monthly.to_csv(output_dir / "monthly_terminal_metrics_all_seeds.csv", index=False)
    monthly.groupby(["model", "month"])[["R2", "RMSE", "MAE", "bias"]].agg(["mean", "std"]).to_csv(output_dir / "monthly_terminal_metrics_mean_std.csv")
    station.to_csv(output_dir / "station_paired_effects.csv", index=False)
    bootstrap.to_csv(output_dir / "moving_block_bootstrap.csv", index=False)
    pd.DataFrame({"forecast_origin": origin_times, "lead24_target_time": terminal_times}).to_csv(output_dir / "test_time_alignment.csv", index=False)
    plot_clean_evidence(main, strata, bootstrap, output_dir)
    plot_lead_skill(source, output_dir)
    write_report(main, strata, monthly, station, bootstrap, output_dir)
    (output_dir / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean publication evidence analysis for sea-level residual forecasting.")
    parser.add_argument("--source-results", default=str(SOURCE_RESULTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=168)
    parser.add_argument("--random-seed", type=int, default=20260727)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
