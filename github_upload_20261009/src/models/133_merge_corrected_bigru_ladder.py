from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "corrected_bigru_ladder"
DEFAULT_OUTPUT = DEFAULT_INPUT / "merged"
SEEDS = [42, 123, 2024, 2025, 3407]
ORDER = ["gnn_bigru", "learnable_graph", "ode_based_learnable", "physical_loss"]
LABELS = {
    "gnn_bigru": "Fixed-graph GNN-BiGRU",
    "learnable_graph": "Learnable-graph GNN-BiGRU",
    "ode_based_learnable": "ODE-prior GNN-BiGRU",
    "physical_loss": "Physical-loss GNN-BiGRU",
}
COLORS = {
    "gnn_bigru": "#6B7280",
    "learnable_graph": "#2878B5",
    "ode_based_learnable": "#2B7A68",
    "physical_loss": "#B45A4A",
}
STATION_ORDER_FILE = ROOT / "data" / "processed_multiyear_2023_2025" / "station_order.csv"


def collect(input_dir: Path, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        path = input_dir / f"seed_{seed}" / "final_four_models_enhanced_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if set(frame["model_key"]) != set(ORDER):
            raise RuntimeError(f"Incomplete model set in {path}: {frame['model_key'].tolist()}")
        if not (frame["bigru_state_extraction"] == "cat_top_layer_forward_backward_h_n").all():
            raise RuntimeError(f"Uncorrected BiGRU provenance in {path}")
        frame["seed"] = seed
        frame["source_file"] = str(path.relative_to(ROOT))
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def exact_wilcoxon_greater(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if len(values) == 0:
        return np.nan
    from scipy.stats import wilcoxon

    return float(wilcoxon(values, alternative="greater", method="exact").pvalue)


def summarize(all_runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "last_residual_MAE"]
    summary = all_runs.groupby(["model_key", "model_name"])[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(part) for part in column if part) for column in summary.columns.to_flat_index()]

    comparisons = [
        ("learnable_minus_fixed", "learnable_graph", "gnn_bigru"),
        ("ode_prior_minus_learnable", "ode_based_learnable", "learnable_graph"),
        ("physical_config_minus_ode_prior", "physical_loss", "ode_based_learnable"),
        ("physical_config_minus_fixed", "physical_loss", "gnn_bigru"),
    ]
    pivot = all_runs.pivot(index="seed", columns="model_key", values=metrics)
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
                    "wilcoxon_greater_p": exact_wilcoxon_greater(improvement.to_numpy()),
                }
            )
    return summary, pd.DataFrame(rows)


def plot(all_runs: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.6, 4.8))
    metrics = [
        ("seq_residual_R2", "Trajectory residual $R^2$", True),
        ("last_residual_R2", "Lead-24 residual $R^2$", True),
        ("last_residual_RMSE", "Lead-24 RMSE (m)", False),
    ]
    x = np.arange(len(ORDER))
    for axis, (metric, title, higher_better) in zip(axes, metrics):
        means = all_runs.groupby("model_key")[metric].mean().reindex(ORDER)
        stds = all_runs.groupby("model_key")[metric].std().reindex(ORDER)
        axis.bar(x, means, yerr=stds, color=[COLORS[key] for key in ORDER], capsize=3, alpha=0.92)
        for model_index, model in enumerate(ORDER):
            values = all_runs[all_runs["model_key"] == model].sort_values("seed")[metric].to_numpy()
            jitter = np.linspace(-0.08, 0.08, len(values))
            axis.scatter(model_index + jitter, values, s=18, color="#202020", zorder=3)
        axis.set_xticks(x, ["Fixed", "Learnable", "ODE-prior", "Physical"], rotation=18, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        if not higher_better:
            axis.set_ylim(bottom=0)
    figure.suptitle("Corrected five-seed GNN-BiGRU attribution ladder", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_dir / "corrected_bigru_ladder_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def load_eta_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        if "pred_residual" in archive:
            return archive["pred_residual"], archive["true_residual"]
        if "pred_states" in archive:
            return archive["pred_states"][..., 0], archive["true_states"][..., 0]
    raise RuntimeError(f"No eta prediction arrays in {path}")


def plot_prediction_example(input_dir: Path, seeds: list[int], output_dir: Path) -> None:
    predictions: dict[str, list[np.ndarray]] = {model: [] for model in ORDER}
    reference_true = None
    for seed in seeds:
        for model in ORDER:
            path = input_dir / f"seed_{seed}" / "horizon_24h" / model / "predictions.npz"
            pred, true = load_eta_predictions(path)
            if reference_true is None:
                reference_true = true
            elif reference_true.shape != true.shape or not np.allclose(reference_true, true, atol=1e-6):
                raise RuntimeError(f"Test targets are not aligned in {path}")
            predictions[model].append(pred)

    assert reference_true is not None
    origin_index = int(np.abs(reference_true).max(axis=(1, 2)).argmax())
    station_names = pd.read_csv(STATION_ORDER_FILE)["station_name"].tolist()
    if len(station_names) != reference_true.shape[1]:
        raise RuntimeError("Station metadata and prediction arrays have different node counts")

    lead = np.arange(1, reference_true.shape[2] + 1)
    figure, axes = plt.subplots(4, 2, figsize=(13.2, 12.2), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for station_index, station_name in enumerate(station_names):
        axis = axes_flat[station_index]
        axis.plot(
            lead,
            reference_true[origin_index, station_index],
            color="#151515",
            linewidth=2.2,
            label="Observed",
            zorder=5,
        )
        for model in ORDER:
            seed_mean = np.mean(
                [pred[origin_index, station_index] for pred in predictions[model]],
                axis=0,
            )
            axis.plot(lead, seed_mean, color=COLORS[model], linewidth=1.45, label=LABELS[model])
        axis.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.55)
        axis.set_title(station_name)
        axis.grid(alpha=0.18)
        axis.set_xlim(1, reference_true.shape[2])
        axis.set_xticks([1, 6, 12, 18, 24])

    legend_axis = axes_flat[-1]
    legend_axis.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc="center", frameon=False, fontsize=10)
    figure.supxlabel("Forecast lead (h)")
    figure.supylabel("Non-tidal residual (m)")
    figure.suptitle("Observed and five-seed mean predictions at a high-amplitude test origin", fontsize=14)
    figure.tight_layout(rect=(0.03, 0.03, 1.0, 0.97))
    figure.savefig(output_dir / "corrected_bigru_ladder_prediction_example.png", dpi=240, bbox_inches="tight")
    plt.close(figure)
    (output_dir / "prediction_example_selection.json").write_text(
        json.dumps(
            {
                "selection_rule": "test origin with maximum absolute observed residual across stations and 24 leads",
                "origin_index_zero_based": origin_index,
                "used_for_model_selection_or_metrics": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_report(summary: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> None:
    lookup = summary.set_index("model_key")
    lines = [
        "# 修正BiGRU后的五种子四模型阶梯",
        "",
        "所有模型均使用顶部双向GRU层的最终forward/backward隐藏状态拼接。旧结果不得与本表混用。",
        "",
        "| Model | Sequence R2 | Lead-24 R2 | Lead-24 RMSE (m) |",
        "|---|---:|---:|---:|",
    ]
    for model in ORDER:
        row = lookup.loc[model]
        lines.append(
            f"| {LABELS[model]} | {row['seq_residual_R2_mean']:.6f} +/- {row['seq_residual_R2_std']:.6f} "
            f"| {row['last_residual_R2_mean']:.6f} +/- {row['last_residual_R2_std']:.6f} "
            f"| {row['last_residual_RMSE_mean']:.6f} +/- {row['last_residual_RMSE_std']:.6f} |"
        )
    lead24 = paired[paired["metric"] == "last_residual_R2"].set_index("comparison")
    comparison_labels = {
        "learnable_minus_fixed": "Learnable minus Fixed",
        "ode_prior_minus_learnable": "ODE-prior minus Learnable",
        "physical_config_minus_ode_prior": "Physical configuration minus ODE-prior",
        "physical_config_minus_fixed": "Physical configuration minus Fixed",
    }
    lines.extend(
        [
            "",
            "## Lead-24 paired changes",
            "",
            "| Comparison | Mean R2 improvement | Wins | One-sided exact Wilcoxon p |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparison, label in comparison_labels.items():
        row = lead24.loc[comparison]
        lines.append(
            f"| {label} | {row['mean_improvement']:.6f} | {int(row['wins'])}/{int(row['count'])} "
            f"| {row['wilcoxon_greater_p']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## 归因边界",
            "",
            "Physical-loss配置同时包含learnable graph、multistate targets、terminal weighting和physics penalty，",
            "因此它相对Fixed或ODE-prior的差值不是physics-only贡献。Physics-only结论仍以matched FS-GWN实验为准。",
            "七站点轨迹图选择真实残差绝对幅度最大的测试origin，仅用于定性展示，不参与模型选择或指标计算。",
        ]
    )
    (output_dir / "CORRECTED_BIGRU_LADDER_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def write_latex_table(summary: pd.DataFrame, output_dir: Path) -> None:
    lookup = summary.set_index("model_key")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Corrected five-seed GNN--BiGRU diagnostic ladder. Values are mean $\pm$ standard deviation.}",
        r"\label{tab:corrected_bigru}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Configuration & Trajectory $R^2$ & Lead-24 $R^2$ & Lead-24 RMSE (m) \\",
        r"\midrule",
    ]
    for model in ORDER:
        row = lookup.loc[model]
        lines.append(
            f"{LABELS[model]} & ${row['seq_residual_R2_mean']:.4f}\\pm{row['seq_residual_R2_std']:.4f}$ "
            f"& ${row['last_residual_R2_mean']:.4f}\\pm{row['last_residual_R2_std']:.4f}$ "
            f"& ${row['last_residual_RMSE_mean']:.4f}\\pm{row['last_residual_RMSE_std']:.4f}$ \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\par\vspace{2pt}\footnotesize\raggedright Note: all recurrent representations concatenate the final forward and backward top-layer GRU states. The Physical-loss row also changes graph learning, auxiliary targets, and terminal weighting; its contrast with the other rows is not physics-only.",
            r"\end{table}",
        ]
    )
    (output_dir / "corrected_bigru_ladder_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge corrected five-seed GNN-BiGRU ladder runs.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_runs = collect(input_dir, args.seeds)
    summary, paired = summarize(all_runs)
    all_runs.to_csv(output_dir / "corrected_bigru_ladder_all_runs.csv", index=False)
    summary.to_csv(output_dir / "corrected_bigru_ladder_mean_std.csv", index=False)
    paired.to_csv(output_dir / "corrected_bigru_ladder_paired.csv", index=False)
    plot(all_runs, output_dir)
    plot_prediction_example(input_dir, args.seeds, output_dir)
    write_report(summary, paired, output_dir)
    write_latex_table(summary, output_dir)
    (output_dir / "merge_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
