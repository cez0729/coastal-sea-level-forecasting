"""Compile architecture-conditional physics utility without retraining models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "results" / "architecture_conditional_physics_utility_20260811"

STRICT_META = {
    "GNN physics loss minus baseline": {
        "backbone": "GNN-BiGRU",
        "supervision": "multistate",
        "injection": "direct_physics_loss",
        "attribution_class": "matched_physics_increment",
        "matched_capacity_control": True,
        "capacity_confounded": False,
    },
    "GWN frozen ODE minus baseline": {
        "backbone": "Graph WaveNet",
        "supervision": "multistate",
        "injection": "frozen_backbone_ode_gate",
        "attribution_class": "capacity_confounded_candidate",
        "matched_capacity_control": False,
        "capacity_confounded": True,
    },
    "GWN joint ODE minus baseline": {
        "backbone": "Graph WaveNet",
        "supervision": "multistate",
        "injection": "joint_ode_gate_finetune",
        "attribution_class": "capacity_confounded_candidate",
        "matched_capacity_control": False,
        "capacity_confounded": True,
    },
    "GWN joint ODE minus capacity control": {
        "backbone": "Graph WaveNet",
        "supervision": "multistate",
        "injection": "joint_ode_gate_finetune",
        "attribution_class": "matched_physics_increment",
        "matched_capacity_control": True,
        "capacity_confounded": False,
    },
    "HS-DT capacity minus baseline": {
        "backbone": "HS-DT-GWN",
        "supervision": "dual_expert",
        "injection": "no_prior_finetune",
        "attribution_class": "capacity_only_control",
        "matched_capacity_control": False,
        "capacity_confounded": True,
    },
    "HS-DT dual ODE minus baseline": {
        "backbone": "HS-DT-GWN",
        "supervision": "dual_expert",
        "injection": "dual_ode_gate_finetune",
        "attribution_class": "capacity_confounded_candidate",
        "matched_capacity_control": False,
        "capacity_confounded": True,
    },
    "HS-DT dual ODE minus capacity control": {
        "backbone": "HS-DT-GWN",
        "supervision": "dual_expert",
        "injection": "dual_ode_gate_finetune",
        "attribution_class": "matched_physics_increment",
        "matched_capacity_control": True,
        "capacity_confounded": False,
    },
    "Direct physics loss on ODE HS-DT": {
        "backbone": "HS-DT-GWN",
        "supervision": "dual_expert",
        "injection": "direct_loss_after_ode",
        "attribution_class": "matched_physics_increment",
        "matched_capacity_control": True,
        "capacity_confounded": False,
    },
}

RETRO_META = {
    "gnn_ode_prior_minus_learnable": {
        "backbone": "GNN-BiGRU",
        "supervision": "eta",
        "injection": "ode_prior_features",
        "attribution_class": "capacity_confounded_broad_ablation",
        "matched_capacity_control": False,
        "capacity_confounded": True,
    },
    "gwn_physics_loss_minus_no_physics": {
        "backbone": "Graph WaveNet",
        "supervision": "multistate",
        "injection": "direct_physics_loss",
        "attribution_class": "matched_physics_increment",
        "matched_capacity_control": True,
        "capacity_confounded": False,
    },
    "orc_minus_hsdt": {
        "backbone": "HS-DT-GWN",
        "supervision": "dual_expert",
        "injection": "ode_residual_adapter",
        "attribution_class": "capacity_confounded_candidate",
        "matched_capacity_control": False,
        "capacity_confounded": True,
    },
    "orc_minus_zero": {
        "backbone": "HS-DT-GWN",
        "supervision": "dual_expert",
        "injection": "ode_residual_adapter",
        "attribution_class": "matched_physics_increment",
        "matched_capacity_control": True,
        "capacity_confounded": False,
    },
    "orc_minus_persistence": {
        "backbone": "HS-DT-GWN",
        "supervision": "dual_expert",
        "injection": "learned_ode_vs_persistence_adapter",
        "attribution_class": "matched_mechanism_comparison",
        "matched_capacity_control": True,
        "capacity_confounded": False,
    },
}

METRIC_LABELS = {
    "seq_residual_R2": "Sequence R2",
    "last_residual_R2": "Lead-24 R2",
    "extreme_abs_q95_residual_R2": "Descriptive q95 R2",
    "event_PR_AUC": "Event PR-AUC",
}


def strict_verdict(row: pd.Series) -> str:
    if row["attribution_class"] == "capacity_only_control":
        return "capacity_control_not_physics"
    if row["mean_delta"] <= 0:
        return "nonpositive"
    if row["matched_capacity_control"] and row["wilcoxon_p_greater"] <= 0.05:
        return "matched_positive_supported"
    if row["matched_capacity_control"]:
        return "matched_positive_inconclusive"
    return "positive_but_capacity_confounded"


def compile_strict() -> pd.DataFrame:
    path = ROOT / "results" / "paper_physics_factorial_2025_h2" / "paired_effects.csv"
    frame = pd.read_csv(path)
    metadata = pd.DataFrame.from_dict(STRICT_META, orient="index").rename_axis("comparison").reset_index()
    merged = frame.merge(metadata, on="comparison", validate="many_to_one")
    merged.insert(0, "protocol", "strict_2025_h2_chronological_refit")
    merged["metric_label"] = merged["metric"].map(METRIC_LABELS)
    merged["effect_definition"] = "enhanced_minus_baseline_metric; positive_is_better"
    merged["verdict"] = merged.apply(strict_verdict, axis=1)
    return merged


def compile_regime() -> pd.DataFrame:
    path = ROOT / "results" / "physics_effect_audit_benchmark" / "component_regime_effect_summary.csv"
    frame = pd.read_csv(path)
    metadata = pd.DataFrame.from_dict(RETRO_META, orient="index").rename_axis("comparison").reset_index()
    merged = frame.merge(metadata, on="comparison", validate="many_to_one")
    merged.insert(0, "protocol", "retrospective_70_15_15_aligned_forcing")
    merged["effect_definition"] = "baseline_minus_candidate_MSE; positive_is_better"
    merged["regime_support"] = np.select(
        [
            (merged["MSE_reduction_mean"] > 0) & (merged["seed_wins"] == 5),
            merged["MSE_reduction_mean"] > 0,
        ],
        ["positive_5_of_5", "positive_inconsistent"],
        default="nonpositive",
    )
    return merged


def compile_leads() -> pd.DataFrame:
    path = ROOT / "results" / "physics_effect_audit_benchmark" / "lead_effects_q95_combined.csv"
    frame = pd.read_csv(path)
    metadata = pd.DataFrame.from_dict(RETRO_META, orient="index").rename_axis("comparison").reset_index()
    merged = frame.merge(metadata, on="comparison", validate="many_to_one")
    merged.insert(0, "protocol", "retrospective_70_15_15_aligned_forcing")
    return merged


def plot_utility(strict: pd.DataFrame, regime: pd.DataFrame, leads: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 13.0), constrained_layout=True)

    strict_order = list(STRICT_META)
    matrix = strict.pivot(index="comparison", columns="metric", values="mean_delta").loc[
        strict_order, list(METRIC_LABELS)
    ]
    vmax = float(np.abs(matrix.to_numpy()).max())
    image = axes[0].imshow(
        matrix.to_numpy(), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), aspect="auto"
    )
    axes[0].set_xticks(range(len(METRIC_LABELS)), [METRIC_LABELS[m] for m in METRIC_LABELS], rotation=18, ha="right")
    axes[0].set_yticks(range(len(strict_order)), strict_order, fontsize=8)
    axes[0].set_title("a  Strict 2025 H2 metric deltas (positive is better)", loc="left", weight="bold")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            axes[0].text(col, row, f"{value:+.4f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(value) > 0.55 * vmax else "black")
    fig.colorbar(image, ax=axes[0], fraction=0.025, pad=0.02, label="Metric delta")

    q95 = regime[(regime["component"] == "combined") & np.isclose(regime["train_quantile"], 0.95)].copy()
    q95 = q95.set_index("comparison").loc[list(RETRO_META)].reset_index()
    colors = ["#B04A5A" if value < 0 else "#2D7D6D" for value in q95["relative_MSE_reduction_pct_mean"]]
    axes[1].barh(np.arange(len(q95)), q95["relative_MSE_reduction_pct_mean"], color=colors)
    axes[1].axvline(0.0, color="#222222", linewidth=0.9)
    axes[1].set_yticks(np.arange(len(q95)), q95["comparison"], fontsize=8)
    axes[1].set_xlabel("Relative MSE reduction (%)")
    axes[1].set_title("b  Retrospective high-forcing q95 regime (all leads)", loc="left", weight="bold")
    axes[1].grid(axis="x", alpha=0.2)
    x_min = float(q95["relative_MSE_reduction_pct_mean"].min())
    x_max = float(q95["relative_MSE_reduction_pct_mean"].max())
    axes[1].set_xlim(x_min - 0.35, x_max + 0.45)
    for index, row in q95.iterrows():
        value = row["relative_MSE_reduction_pct_mean"]
        offset = 0.02 if value >= 0 else -0.02
        axes[1].text(value + offset, index, f"{value:+.2f}% ({int(row['seed_wins'])}/5)", va="center", fontsize=8,
                     ha="left" if value >= 0 else "right")

    lead_mean = leads.groupby(["comparison", "lead_hour"], as_index=False)["MSE_reduction"].mean()
    colors_by_comparison = {
        "gnn_ode_prior_minus_learnable": "#6B7280",
        "gwn_physics_loss_minus_no_physics": "#C17D11",
        "orc_minus_hsdt": "#2A6FBB",
        "orc_minus_zero": "#2D7D6D",
        "orc_minus_persistence": "#8B5A9F",
    }
    for comparison in RETRO_META:
        part = lead_mean[lead_mean["comparison"] == comparison]
        axes[2].plot(part["lead_hour"], part["MSE_reduction"], linewidth=1.8,
                     color=colors_by_comparison[comparison], label=comparison)
    axes[2].axhline(0.0, color="#222222", linewidth=0.9)
    axes[2].set_xlim(1, 24)
    axes[2].set_xlabel("Forecast lead (h)")
    axes[2].set_ylabel("MSE reduction")
    axes[2].set_title("c  Retrospective high-forcing q95 effect by lead", loc="left", weight="bold")
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(output / "architecture_conditional_physics_utility.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    strict = compile_strict()
    regime = compile_regime()
    leads = compile_leads()
    strict.to_csv(output / "strict_physics_utility_matrix.csv", index=False)
    regime.to_csv(output / "retrospective_regime_utility_matrix.csv", index=False)
    leads.to_csv(output / "retrospective_q95_per_lead_utility.csv", index=False)
    plot_utility(strict, regime, leads, output)

    matched = strict[strict["attribution_class"] == "matched_physics_increment"]
    significant = matched[(matched["mean_delta"] > 0) & (matched["wilcoxon_p_greater"] <= 0.05)]
    q95 = regime[(regime["component"] == "combined") & np.isclose(regime["train_quantile"], 0.95)]
    decision = {
        "overall": "physics_utility_is_architecture_and_injection_conditional_not_universal",
        "strict_matched_positive_significant_rows": int(len(significant)),
        "strict_matched_rows": int(len(matched)),
        "strict_direct_loss_gnn_sequence_delta": float(strict.loc[
            (strict["comparison"] == "GNN physics loss minus baseline") & (strict["metric"] == "seq_residual_R2"), "mean_delta"
        ].iloc[0]),
        "strict_direct_loss_gwn_terminal_delta_retrospective_reference": -0.000232,
        "strict_capacity_matched_gwn_ode_sequence_delta": float(strict.loc[
            (strict["comparison"] == "GWN joint ODE minus capacity control") & (strict["metric"] == "seq_residual_R2"), "mean_delta"
        ].iloc[0]),
        "strict_capacity_matched_hsdt_ode_sequence_delta": float(strict.loc[
            (strict["comparison"] == "HS-DT dual ODE minus capacity control") & (strict["metric"] == "seq_residual_R2"), "mean_delta"
        ].iloc[0]),
        "retrospective_q95_positive_5_of_5_comparisons": q95.loc[
            (q95["MSE_reduction_mean"] > 0) & (q95["seed_wins"] == 5), "comparison"
        ].tolist(),
        "evidence_boundary": [
            "strict_2025_h2_is_chronological_refit_not_untouched_holdout",
            "retrospective_regime_results_are_not_mixed_with_strict_results",
            "capacity_confounded_candidates_are_not_attributed_to_physics_only",
        ],
    }
    (output / "PHYSICS_UTILITY_DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# 跨架构物理效用审计

## 结论

物理信息的收益不是统一的模型增益，而是由骨干、监督方式、注入位置和评价目标共同决定的边际效用。严格 2025 H2 中，{len(matched)} 个匹配归因的“比较-指标”组合里，满足正向且单侧精确 Wilcoxon p<=0.05 的组合为 {len(significant)} 个；因此当前证据不支持“physics 普遍提升连续水位精度”。

## 严格时间协议

- GNN-BiGRU 直接 physics loss：Sequence R2 变化 {decision['strict_direct_loss_gnn_sequence_delta']:+.6f}，不支持稳定提升。
- GWN joint ODE 相对同容量 no-prior control：Sequence R2 变化 {decision['strict_capacity_matched_gwn_ode_sequence_delta']:+.6f}，但五 seed 统计不显著，Lead-24 为负。
- HS-DT dual ODE 相对同容量 no-prior control：Sequence R2 变化 {decision['strict_capacity_matched_hsdt_ode_sequence_delta']:+.6f}，统计不显著，Lead-24 为负。
- ODE-HS-DT 上继续加 direct physics loss 的净增量接近零。

## 普通 retrospective 高强迫分层

q95 combined forcing 下，5/5 seed 正向的比较为：{', '.join(decision['retrospective_q95_positive_5_of_5_comparisons']) or '无'}。其中 ORC 相对原 HS-DT 的结果包含额外 adapter 容量，只能作为条件性候选；相对 zero/persistence 的匹配控制才可讨论 ODE 信号本身，而且增量很小。

## 论文写法

可以写成：简化物理信息在不同骨干上呈现明显的条件依赖；直接 physical residual loss 在强 GWN 中被架构能力和多状态监督吸收，而 ODE-conditioned adapter 在高强迫远期样本中提供小幅信号，但尚未形成跨协议、跨指标稳定的 physics-only 精度提升。

不能写成：physics loss 或 ODE prior 普遍提高所有模型；也不能把联合微调或 adapter 的总增益全部归因于物理信息。
"""
    (output / "EXPERIMENT_REPORT_CN.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
