from __future__ import annotations

"""Build ordinary retrospective-benchmark tables and physics attribution figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_only_submission_evidence"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bigru = pd.read_csv(ROOT / "results" / "corrected_bigru_ladder" / "merged" / "corrected_bigru_ladder_mean_std.csv")
    gwn = pd.read_csv(ROOT / "results" / "priority12_physics_graph_wavenet" / "priority2_physics_gwn_mean_std.csv")
    hsdt = pd.read_csv(ROOT / "results" / "horizon_specialized_dual_task_gwn" / "mean_std.csv")
    orc = pd.read_csv(ROOT / "results" / "ode_residual_corrected_hsdt_gwn" / "mean_std.csv")

    rows = []
    for _, row in bigru.iterrows():
        rows.append({
            "model": row["model_name"], "family": "GNN-BiGRU",
            "sequence_R2_mean": row["seq_residual_R2_mean"], "sequence_R2_std": row["seq_residual_R2_std"],
            "lead24_R2_mean": row["last_residual_R2_mean"], "lead24_R2_std": row["last_residual_R2_std"],
            "q95_R2_mean": np.nan, "q95_R2_std": np.nan,
        })
    gwn_names = {
        "gwn_eta_only": "Eta-only FS-GWN",
        "gwn_multistate_no_physics": "Multistate FS-GWN",
        "gwn_multistate_physics": "Multistate FS-GWN + physics loss",
    }
    for _, row in gwn.iterrows():
        rows.append({
            "model": gwn_names[row["config"]], "family": "FS-GWN",
            "sequence_R2_mean": row["seq_residual_R2_mean"], "sequence_R2_std": row["seq_residual_R2_std"],
            "lead24_R2_mean": row["last_residual_R2_mean"], "lead24_R2_std": row["last_residual_R2_std"],
            "q95_R2_mean": row["extreme_abs_q95_residual_R2_mean"], "q95_R2_std": row["extreme_abs_q95_residual_R2_std"],
        })
    hsdt_row = hsdt.loc[hsdt["config"] == "horizon_specialized_dual_task_gwn"].iloc[0]
    rows.append({
        "model": "HS-DT-GWN", "family": "Dual expert",
        "sequence_R2_mean": hsdt_row["seq_residual_R2_mean"], "sequence_R2_std": hsdt_row["seq_residual_R2_std"],
        "lead24_R2_mean": hsdt_row["last_residual_R2_mean"], "lead24_R2_std": hsdt_row["last_residual_R2_std"],
        "q95_R2_mean": hsdt_row["extreme_abs_q95_residual_R2_mean"], "q95_R2_std": hsdt_row["extreme_abs_q95_residual_R2_std"],
    })
    orc_names = {
        "hsdt_physics_correction": "HS-DT + physics correction",
        "hsdt_zero_adapter": "HS-DT + zero adapter",
        "hsdt_persistence_adapter": "HS-DT + persistence adapter",
        "orc_hsdt_gwn": "ORC-HS-DT-GWN",
    }
    for key, label in orc_names.items():
        row = orc.loc[orc["model"] == key].iloc[0]
        rows.append({
            "model": label, "family": "HS-DT adapter",
            "sequence_R2_mean": row["seq_residual_R2_mean"], "sequence_R2_std": row["seq_residual_R2_std"],
            "lead24_R2_mean": row["last_residual_R2_mean"], "lead24_R2_std": row["last_residual_R2_std"],
            "q95_R2_mean": row["extreme_abs_q95_residual_R2_mean"], "q95_R2_std": row["extreme_abs_q95_residual_R2_std"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "benchmark_model_hierarchy.csv", index=False)

    main_order = [
        "GNN-BiGRU", "Learnable-graph GNN-BiGRU", "ODE-based learnable GNN-BiGRU",
        "Physical-loss GNN-BiGRU", "Eta-only FS-GWN", "Multistate FS-GWN",
        "HS-DT-GWN", "ORC-HS-DT-GWN",
    ]
    main = summary.set_index("model").loc[main_order]
    colors = ["#697382", "#697382", "#697382", "#8A6A6A", "#2F7D6D", "#2F7D6D", "#526B8C", "#7B5EA7"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, metric, title in [
        (axes[0], "sequence_R2", "24-h trajectory R2"),
        (axes[1], "lead24_R2", "Lead-24 R2"),
    ]:
        values = main[f"{metric}_mean"].to_numpy()
        errors = main[f"{metric}_std"].to_numpy()
        x = np.arange(len(main))
        axis.bar(x, values, yerr=errors, capsize=3, color=colors)
        axis.set_xticks(x)
        axis.set_xticklabels(["Fixed GNN", "Learnable GNN", "GNN+ODE prior", "GNN physical", "Eta GWN", "Multi GWN", "HS-DT", "ORC-HS-DT"], rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Retrospective benchmark hierarchy: dual expert > FS-GWN > GNN-BiGRU", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "benchmark_model_hierarchy.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    values = summary.set_index("model")
    comparisons = [
        ("GNN ODE prior - learnable GNN", "ODE-based learnable GNN-BiGRU", "Learnable-graph GNN-BiGRU"),
        ("GWN physics loss - no physics", "Multistate FS-GWN + physics loss", "Multistate FS-GWN"),
        ("HS-DT physics correction - HS-DT", "HS-DT + physics correction", "HS-DT-GWN"),
        ("ORC - HS-DT", "ORC-HS-DT-GWN", "HS-DT-GWN"),
        ("ORC - zero adapter", "ORC-HS-DT-GWN", "HS-DT + zero adapter"),
        ("ORC - persistence adapter", "ORC-HS-DT-GWN", "HS-DT + persistence adapter"),
    ]
    effects = []
    for label, enhanced, baseline in comparisons:
        effects.append({
            "comparison": label,
            "sequence_R2_delta": values.loc[enhanced, "sequence_R2_mean"] - values.loc[baseline, "sequence_R2_mean"],
            "lead24_R2_delta": values.loc[enhanced, "lead24_R2_mean"] - values.loc[baseline, "lead24_R2_mean"],
            "q95_R2_delta": values.loc[enhanced, "q95_R2_mean"] - values.loc[baseline, "q95_R2_mean"],
        })
    effects = pd.DataFrame(effects)
    effects.to_csv(OUT / "benchmark_physics_attribution_deltas.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    y = np.arange(len(effects))
    effect_colors = ["#697382", "#8A6A6A", "#8A6A6A", "#7B5EA7", "#526B8C", "#B07A32"]
    for axis, metric, title in [
        (axes[0], "sequence_R2_delta", "Sequence R2 effect"),
        (axes[1], "lead24_R2_delta", "Lead-24 R2 effect"),
    ]:
        axis.barh(y, effects[metric], color=effect_colors)
        axis.axvline(0, color="#222222", linewidth=0.9)
        axis.set_yticks(y)
        axis.set_yticklabels(effects["comparison"], fontsize=8)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Physics attribution requires matched adapter and loss controls", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "benchmark_physics_attribution.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    table = main.reset_index()[["model", "sequence_R2_mean", "sequence_R2_std", "lead24_R2_mean", "lead24_R2_std", "q95_R2_mean", "q95_R2_std"]]
    lines = [
        r"\begin{table}[t]", r"\centering", r"\caption{Five-seed retrospective benchmark hierarchy.}",
        r"\label{tab:benchmark_hierarchy}", r"\small", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lccc}",
        r"\toprule", r"Model & Trajectory $R^2$ & Lead-24 $R^2$ & Descriptive $q_{0.95}$ $R^2$ \\", r"\midrule",
    ]
    for _, row in table.iterrows():
        q95 = "--" if pd.isna(row["q95_R2_mean"]) else f"${row['q95_R2_mean']:.4f}\\pm{row['q95_R2_std']:.4f}$"
        lines.append(
            f"{row['model']} & ${row['sequence_R2_mean']:.4f}\\pm{row['sequence_R2_std']:.4f}$ & "
            f"${row['lead24_R2_mean']:.4f}\\pm{row['lead24_R2_std']:.4f}$ & {q95} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    (OUT / "benchmark_hierarchy_table.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
