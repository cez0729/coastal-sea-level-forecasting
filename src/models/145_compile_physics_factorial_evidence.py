from __future__ import annotations

"""Compile matched H2 physics evidence across GNN, GWN, and HS-DT."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_physics_factorial_2025_h2"
METRICS = [
    "seq_residual_R2",
    "last_residual_R2",
    "extreme_abs_q95_residual_R2",
    "event_PR_AUC",
]


def load(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


def select(frame: pd.DataFrame, config: str, label: str, family: str, mechanism: str) -> pd.DataFrame:
    chosen = frame.loc[frame["config"] == config, ["seed", *METRICS]].copy()
    chosen.insert(1, "model", label)
    chosen.insert(2, "family", family)
    chosen.insert(3, "mechanism", mechanism)
    return chosen


def paired(frame: pd.DataFrame, enhanced: str, baseline: str, comparison: str) -> list[dict]:
    rows = []
    left = frame.loc[frame["model"] == enhanced].set_index("seed")
    right = frame.loc[frame["model"] == baseline].set_index("seed")
    common = left.index.intersection(right.index)
    for metric in METRICS:
        delta = left.loc[common, metric].to_numpy() - right.loc[common, metric].to_numpy()
        pvalue = float(wilcoxon(delta, alternative="greater", method="exact").pvalue)
        rows.append(
            {
                "comparison": comparison,
                "enhanced": enhanced,
                "baseline": baseline,
                "metric": metric,
                "mean_delta": float(delta.mean()),
                "wins": int((delta > 0).sum()),
                "seed_count": int(len(delta)),
                "wilcoxon_p_greater": pvalue,
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bigru = load("results/strict_causal_physical_bigru_refit_2025_h2/all_runs.csv")
    frozen = load("results/formal_gwn_ode_physics_factorial_2025_h2/all_runs.csv")
    gwn_no_prior = load("results/multistate_no_prior_control_scored_2025_h2/all_runs.csv")
    expert = load("results/hsdt_expert_physics_conditioned/formal/all_runs.csv")
    no_prior = load("results/hsdt_no_prior_finetune_control/formal/all_runs.csv")
    plus_loss = load("results/hsdt_multistate_prior_physics_loss_active/formal/all_runs.csv")

    frames = [
        select(bigru, "gnn_bigru_no_physics", "GNN-BiGRU baseline", "GNN-BiGRU", "none"),
        select(bigru, "gnn_bigru_physics", "GNN-BiGRU + physics loss", "GNN-BiGRU", "physics_loss"),
        select(frozen, "baseline", "Multistate GWN baseline", "GWN", "none"),
        select(frozen, "ode_frozen", "GWN + frozen causal ODE", "GWN", "ode_prior"),
        select(expert, "multistate_causal_ode_expert", "GWN + joint causal ODE", "GWN", "ode_prior_and_finetune"),
        select(gwn_no_prior, "multistate_no_prior_finetune", "GWN + no-prior fine-tune", "GWN", "capacity_control"),
        select(expert, "hsdt_baseline", "HS-DT baseline", "HS-DT", "none"),
        select(no_prior, "hsdt_no_prior_finetune_control", "HS-DT + no-prior fine-tune", "HS-DT", "capacity_control"),
        select(expert, "hsdt_both_causal_ode_experts", "HS-DT + dual causal ODE", "HS-DT", "ode_prior_and_finetune"),
        select(expert, "hsdt_multistate_causal_ode_expert", "HS-DT + multistate causal ODE", "HS-DT", "ode_prior_and_finetune"),
        select(plus_loss, "hsdt_multistate_prior_plus_physics_loss", "HS-DT + multistate ODE + physics loss", "HS-DT", "ode_prior_and_physics_loss"),
    ]
    models = pd.concat(frames, ignore_index=True)
    models.to_csv(OUT / "all_model_runs.csv", index=False)
    means = models.groupby(["family", "model", "mechanism"], as_index=False)[METRICS].agg(["mean", "std", "count"])
    means.to_csv(OUT / "model_means.csv")

    comparisons = []
    for enhanced, baseline, name in [
        ("GNN-BiGRU + physics loss", "GNN-BiGRU baseline", "GNN physics loss minus baseline"),
        ("GWN + frozen causal ODE", "Multistate GWN baseline", "GWN frozen ODE minus baseline"),
        ("GWN + joint causal ODE", "Multistate GWN baseline", "GWN joint ODE minus baseline"),
        ("GWN + joint causal ODE", "GWN + no-prior fine-tune", "GWN joint ODE minus capacity control"),
        ("HS-DT + no-prior fine-tune", "HS-DT baseline", "HS-DT capacity minus baseline"),
        ("HS-DT + dual causal ODE", "HS-DT baseline", "HS-DT dual ODE minus baseline"),
        ("HS-DT + dual causal ODE", "HS-DT + no-prior fine-tune", "HS-DT dual ODE minus capacity control"),
        ("HS-DT + multistate ODE + physics loss", "HS-DT + multistate causal ODE", "Direct physics loss on ODE HS-DT"),
    ]:
        comparisons.extend(paired(models, enhanced, baseline, name))
    effects = pd.DataFrame(comparisons)
    effects.to_csv(OUT / "paired_effects.csv", index=False)

    labels = {
        "seq_residual_R2": "Sequence R2",
        "last_residual_R2": "Lead-24 R2",
        "extreme_abs_q95_residual_R2": "q95 R2",
        "event_PR_AUC": "Event PR-AUC",
    }
    plot_order = [
        "GNN physics loss minus baseline",
        "GWN frozen ODE minus baseline",
        "GWN joint ODE minus baseline",
        "HS-DT capacity minus baseline",
        "HS-DT dual ODE minus baseline",
        "HS-DT dual ODE minus capacity control",
        "GWN joint ODE minus capacity control",
        "Direct physics loss on ODE HS-DT",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=False)
    colors = ["#5B6573", "#2F7D6D", "#168AAD", "#B07A32", "#7B5EA7", "#A14B5A", "#3B6A8C", "#8B5A2B"]
    for axis, metric in zip(axes.flat, METRICS):
        part = effects.loc[effects["metric"] == metric].set_index("comparison").loc[plot_order]
        values = part["mean_delta"].to_numpy()
        axis.barh(np.arange(len(plot_order)), values, color=colors)
        axis.axvline(0.0, color="#222222", linewidth=0.9)
        axis.set_yticks(np.arange(len(plot_order)))
        axis.set_yticklabels(plot_order, fontsize=8)
        axis.set_title(labels[metric])
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Matched 2025 H2 effects of physics information and capacity controls", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "physics_effects_by_backbone.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    report = """# 2025 H2 物理信息跨骨干证据矩阵

## 统一协议

- 训练：2023--2024；验证选择：2025 H1；评估：2025 H2 chronological backtest。
- 五个配对 seed：42、123、2024、2025、3407。
- 严格因果预处理；未来 residual 不作为输入。
- 2025 H2 已在模型开发中被查看，因此不是 untouched independent holdout。

## 新增冻结 GWN ODE 对照

冻结 multistate GWN、只训练 causal ODE prior 和 horizon gate 后，相对纯 GWN：

- Sequence R2：+0.003492，3/5 seed，单侧精确 Wilcoxon p=0.3125；
- Lead-24 R2：-0.004967，2/5 seed；
- q95 R2：-0.008799，1/5 seed；
- Event PR-AUC：+0.007400，4/5 seed，p=0.09375。

这说明冻结主干时，ODE prior 只对事件排序表现出弱趋势，不能称为稳定精度提升。

## 可投稿的核心结论

1. GNN-BiGRU 上直接 physics loss 在锁定回测中轻微降低 sequence 和 Lead-24。
2. GWN 上冻结 ODE prior 不稳定；联合微调后提升更大，但同轮数 no-prior fine-tuning control 也能获得大部分收益。
3. HS-DT + dual ODE 相对冻结 HS-DT 改善整体轨迹和事件指标，但没有超过 no-prior control 的 Lead-24。
4. 在 ODE-enhanced HS-DT 上继续加入 direct physics loss，净增量接近零。
5. 因此最可信的创新不是“physics loss 普遍有效”，而是“物理信息的注入位置、骨干容量和预测目标共同决定收益”。

## 物理相关候选模型

当前最适合保留的候选是 horizon-conditioned causal ODE expert injection。它在单体 multistate GWN 中给出最高的描述性 q95 R2（0.549979），但相对匹配 no-prior fine-tuning control 的 q95 差值仅 +0.005584、未显著；在 HS-DT 中改善 sequence 和事件 PR-AUC。它只能称为 physics-conditioned candidate，而不能称为已独立确认的最终最强模型；投稿前仍需冻结结构后使用模型开发期间不可见的新年份或外部站点验证。
"""
    (OUT / "EXPERIMENT_MATRIX_CN.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
