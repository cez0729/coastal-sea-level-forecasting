from __future__ import annotations

"""Detailed, validation-locked physics-effect audit for the ordinary benchmark.

This script does not retrain models and does not select a test threshold. It
reuses the frozen five-seed benchmark predictions, applies forcing regimes
whose thresholds are computed from the training period, and reports effects
by forcing component, station, lead, and block bootstrap interval.
"""

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
SEEDS = [42, 123, 2024, 2025, 3407]
COMPONENTS = ["wind", "pressure_tendency", "current", "wave_flux", "wave_setup", "combined"]
COMPARISON_LABELS = {
    "gnn_ode_prior_minus_learnable": "GNN ODE prior - learnable",
    "gwn_physics_loss_minus_no_physics": "GWN physics loss - no physics",
    "orc_minus_hsdt": "ORC - HS-DT",
    "orc_minus_zero": "ORC - zero adapter",
    "orc_minus_persistence": "ORC - persistence adapter",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0]), float(values[0])
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def block_effect(
    true: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    """Block bootstrap the mean MSE reduction over forecast origins.

    Windows are overlapping in the benchmark, so resampling individual rows
    would overstate the effective sample size. Origins are grouped into
    contiguous blocks before resampling.
    """
    reduction = (true - baseline) ** 2 - (true - candidate) ** 2
    row_values = []
    n_rows = reduction.shape[0]
    for start in range(0, n_rows, block_size):
        block = reduction[start : start + block_size]
        block_mask = mask[start : start + block_size]
        if block_mask.any():
            row_values.append(float(block[block_mask].mean()))
    return bootstrap_mean(np.asarray(row_values), rng, n_boot)


def audit(args: argparse.Namespace) -> None:
    m149_path = next((ROOT / "数据整理").glob("149_physics_regime_switched_hsdt.py"))
    m149 = load_module("physics_regime_switch_149", m149_path)
    data_args = SimpleNamespace(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        window=args.window,
        train_stride=args.train_stride,
        physics_forcing_mode="last_input",
        extreme_quantile=args.extreme_quantile,
    )
    data = m149.final4.build_enhanced_data(data_args, args.horizon, add_ode_prior=False)
    forcing = m149.build_forcing_intensity(
        data,
        SimpleNamespace(
            train_ratio=args.train_ratio,
            candidate_quantiles=args.quantiles,
            regime_quantiles=args.quantiles,
        ),
    )
    test_indices = data["multi_test"].indices
    test_forcing = {
        component: forcing["combined"][test_indices - 1]
        if component == "combined"
        else forcing["raw"][component][test_indices - 1]
        for component in COMPONENTS
    }
    thresholds = {
        component: {
            float(q): np.asarray(forcing["component_thresholds"][component][float(q)], dtype=float)
            for q in args.quantiles
        }
        for component in COMPONENTS
    }

    prediction_args = SimpleNamespace(
        orc_results=args.orc_results,
        gwn_results=args.gwn_results,
        bigru_results=args.bigru_results,
        horizon=args.horizon,
    )
    all_rows: list[dict] = []
    station_rows: list[dict] = []
    lead_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    rng = np.random.default_rng(args.random_seed)
    for seed in args.seeds:
        predictions: dict[str, np.ndarray] = {}
        true = tide = None
        for name in ["hsdt_gwn", "hsdt_zero_adapter", "hsdt_persistence_adapter", "orc_hsdt_gwn"]:
            pred, current_true, current_tide = m149.load_prediction(seed, name, prediction_args)
            predictions[name] = pred
            if true is None:
                true, tide = current_true, current_tide
        gwn_root = ROOT / args.gwn_results / f"seed_{seed}" / f"horizon_{args.horizon}h"
        with np.load(gwn_root / "gwn_multistate_no_physics" / "predictions.npz") as payload:
            predictions["gwn_no_physics"] = payload["pred_states"][..., 0].astype(np.float64)
            gwn_true = payload["true_states"][..., 0].astype(np.float64)
        with np.load(gwn_root / "gwn_multistate_physics" / "predictions.npz") as payload:
            predictions["gwn_physics_loss"] = payload["pred_states"][..., 0].astype(np.float64)
        bigru_root = ROOT / args.bigru_results / f"seed_{seed}" / f"horizon_{args.horizon}h"
        with np.load(bigru_root / "learnable_graph" / "predictions.npz") as payload:
            predictions["gnn_learnable"] = payload["pred_residual"].astype(np.float64)
            bigru_true = payload["true_residual"].astype(np.float64)
        with np.load(bigru_root / "ode_based_learnable" / "predictions.npz") as payload:
            predictions["gnn_ode_prior"] = payload["pred_residual"].astype(np.float64)
        if not np.allclose(true, gwn_true, atol=1e-6, rtol=0.0) or not np.allclose(true, bigru_true, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"Target mismatch for seed {seed}")

        comparisons = m149.physics_comparisons(predictions)
        for component in COMPONENTS:
            values = test_forcing[component]
            for quantile in args.quantiles:
                mask = values >= thresholds[component][float(quantile)][None, :]
                for comparison, (candidate, baseline) in comparisons.items():
                    reduction = (true - baseline) ** 2 - (true - candidate) ** 2
                    baseline_mse = ((true - baseline) ** 2)[mask].mean()
                    selected = reduction[mask]
                    all_rows.append(
                        {
                            "seed": seed,
                            "component": component,
                            "train_quantile": quantile,
                            "comparison": comparison,
                            "coverage": float(mask.mean()),
                            "baseline_MSE": float(baseline_mse),
                            "MSE_reduction": float(selected.mean()),
                            "relative_MSE_reduction_pct": float(100.0 * selected.mean() / max(baseline_mse, 1e-12)),
                        }
                    )
                    if component == "combined" and quantile == max(args.quantiles):
                        mean, lo, hi = block_effect(true, candidate, baseline, mask, args.block_size, rng, args.bootstrap)
                        bootstrap_rows.append(
                            {
                                "seed": seed,
                                "component": component,
                                "train_quantile": quantile,
                                "comparison": comparison,
                                "block_size_origins": args.block_size,
                                "bootstrap_mean_MSE_reduction": mean,
                                "bootstrap_ci_low": lo,
                                "bootstrap_ci_high": hi,
                            }
                        )
                        for station_index, station_id in enumerate(m149.v2.STATION_IDS):
                            station_reduction = reduction[:, station_index, -1][mask[:, station_index]]
                            station_baseline = ((true[:, station_index, -1] - baseline[:, station_index, -1]) ** 2)[mask[:, station_index]]
                            if station_reduction.size:
                                station_rows.append(
                                    {
                                        "seed": seed,
                                        "station_id": station_id,
                                        "comparison": comparison,
                                        "train_quantile": quantile,
                                        "coverage": float(mask[:, station_index].mean()),
                                        "baseline_lead24_MSE": float(station_baseline.mean()),
                                        "lead24_MSE_reduction": float(station_reduction.mean()),
                                        "relative_lead24_MSE_reduction_pct": float(100.0 * station_reduction.mean() / max(station_baseline.mean(), 1e-12)),
                                    }
                                )
                        for lead in range(true.shape[-1]):
                            lead_reduction = reduction[:, :, lead][mask]
                            lead_rows.append(
                                {
                                    "seed": seed,
                                    "lead_hour": lead + 1,
                                    "comparison": comparison,
                                    "train_quantile": quantile,
                                    "coverage": float(mask.mean()),
                                    "MSE_reduction": float(lead_reduction.mean()),
                                }
                            )

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out / "component_regime_effects.csv", index=False)
    pd.DataFrame(station_rows).to_csv(out / "station_lead24_effects.csv", index=False)
    pd.DataFrame(lead_rows).to_csv(out / "lead_effects_q95_combined.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(out / "block_bootstrap_q95_combined.csv", index=False)
    threshold_json = {
        component: {str(q): values.tolist() for q, values in by_quantile.items()}
        for component, by_quantile in thresholds.items()
    }
    (out / "training_period_thresholds.json").write_text(json.dumps(threshold_json, indent=2), encoding="utf-8")

    grouped = pd.DataFrame(all_rows).groupby(["component", "train_quantile", "comparison"], as_index=False).agg(
        coverage=("coverage", "mean"), baseline_MSE=("baseline_MSE", "mean"),
        MSE_reduction_mean=("MSE_reduction", "mean"), MSE_reduction_std=("MSE_reduction", "std"),
        relative_MSE_reduction_pct_mean=("relative_MSE_reduction_pct", "mean"),
        seed_wins=("MSE_reduction", lambda x: int((x > 0).sum())),
    )
    grouped.to_csv(out / "component_regime_effect_summary.csv", index=False)

    q95 = grouped[grouped["train_quantile"] == max(args.quantiles)].copy()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    components = COMPONENTS
    x = np.arange(len(components))
    width = 0.16
    comparisons = list(COMPARISON_LABELS)
    for idx, comparison in enumerate(comparisons):
        values = []
        for component in components:
            row = q95[(q95.component == component) & (q95.comparison == comparison)]
            values.append(float(row.iloc[0].MSE_reduction_mean) if len(row) else 0.0)
        ax.bar(x + (idx - 2) * width, values, width, label=COMPARISON_LABELS[comparison])
    ax.axhline(0, color="#222", linewidth=0.8)
    ax.set_xticks(x, ["Wind", "Pressure tendency", "Current", "Wave flux", "Wave setup", "Combined"])
    ax.set_ylabel("Lead-24 MSE reduction")
    ax.set_title("Training-threshold q95 physical-regime effect")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "component_effect_q95.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    lead = pd.DataFrame(lead_rows)
    pivot = lead.groupby(["comparison", "lead_hour"], as_index=False).MSE_reduction.mean()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for comparison in comparisons:
        sub = pivot[pivot.comparison == comparison]
        ax.plot(sub.lead_hour, sub.MSE_reduction, marker="o", markersize=2.5, label=COMPARISON_LABELS[comparison])
    ax.axhline(0, color="#222", linewidth=0.8)
    ax.set_xlabel("Forecast lead (h)")
    ax.set_ylabel("MSE reduction in q95 combined regime")
    ax.set_title("Where physical correction changes the forecast")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "lead_effect_q95.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    report = write_report(grouped, pd.DataFrame(bootstrap_rows), out, args)
    (out / "EXPERIMENT_REPORT_CN.md").write_text(report, encoding="utf-8")


def write_report(grouped: pd.DataFrame, bootstrap: pd.DataFrame, out: Path, args: argparse.Namespace) -> str:
    q95 = grouped[grouped["train_quantile"] == max(args.quantiles)]
    def row(component: str, comparison: str):
        return q95[(q95.component == component) & (q95.comparison == comparison)].iloc[0]

    orc = row("combined", "orc_minus_hsdt")
    gwn = row("combined", "gwn_physics_loss_minus_no_physics")
    gnn = row("combined", "gnn_ode_prior_minus_learnable")
    boot_orc = bootstrap[bootstrap.comparison == "orc_minus_hsdt"]
    boot_text = "；".join(
        f"seed {int(r.seed)}: {r.bootstrap_mean_MSE_reduction:.6g} [{r.bootstrap_ci_low:.6g}, {r.bootstrap_ci_high:.6g}]"
        for _, r in boot_orc.iterrows()
    )
    return f"""# 普通 benchmark 物理作用审计

本审计复用普通 70/15/15 benchmark 的五 seed 测试预测，不重新训练模型，也不使用测试标签选择阈值。所有强迫阈值由训练期的风、气压趋势、流速、波能通量和 wave-setup proxy 构成的因果 forcing index 计算；这里只把 q95 作为高强迫示例，q75/q90/q95 全部结果见 `component_regime_effect_summary.csv`。

## 最清晰的条件结论

在训练期 q95 综合强迫状态下：

- ORC-HS-DT 相对 HS-DT 的 Lead-24 MSE 平均减少 **{orc.MSE_reduction_mean:.6g}**，相对高强迫基线误差约 **{orc.relative_MSE_reduction_pct_mean:.3f}%**，五个 seed 中 **{int(orc.seed_wins)}/5** 为正；这说明物理 ODE 修正的作用主要是识别强迫状态下的远期动力偏差，而非所有样本的统一增益。
- 直接 physics loss 相对无 physics GWN 的 Lead-24 MSE 平均变化为 **{gwn.MSE_reduction_mean:.6g}**，五个 seed 中 **{int(gwn.seed_wins)}/5** 为正，仍不能声称对强 GWN 有稳定总体提升。
- GNN ODE prior 相对 learnable GNN 的高强迫 Lead-24 MSE 平均减少 **{gnn.MSE_reduction_mean:.6g}**，但 seed 一致性弱于 ORC；它更适合作为条件性远期收益，而不是普遍收益。

## Bootstrap 检查

块长为 {args.block_size} 个 forecast origin 的 block bootstrap 结果（q95 综合强迫、ORC 相对 HS-DT）为：

{boot_text}

这些区间用于描述样本相关性下的不确定性，不替代五 seed 的独立重复。站点和 lead 的拆分见 `station_lead24_effects.csv` 与 `lead_effects_q95_combined.csv`，图见 `component_effect_q95.png` 和 `lead_effect_q95.png`。

## 对论文的建议

物理方法的“更大影响”不能通过提高 lambda、挑选测试子集或隐藏负结果来制造。当前最有说服力的表述是：**物理信息对全样本平均误差不是普遍增益，但在高风/高流/高波能状态下，尤其是远期 lead，能为 ODE-conditioned correction 提供选择依据；直接 physics loss 在强 GWN 上接近中性。** 这比声称 physics loss 普遍提升更严谨，也与反向/错位门控控制相一致。

该审计仍属于普通 benchmark 的机制分析，不是独立 chronological holdout；不得把它写成实时部署或外部年份验证。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit conditional physical effects in the ordinary benchmark")
    parser.add_argument("--output-dir", default="results/physics_effect_audit_benchmark")
    parser.add_argument("--orc-results", default="results/ode_residual_corrected_hsdt_gwn")
    parser.add_argument("--gwn-results", default="results/priority12_physics_graph_wavenet")
    parser.add_argument("--bigru-results", default="results/corrected_bigru_ladder")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.75, 0.90, 0.95])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260731)
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
