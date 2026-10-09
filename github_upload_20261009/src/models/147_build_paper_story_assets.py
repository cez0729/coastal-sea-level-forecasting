from __future__ import annotations

"""Build publication-facing story figures from the locked H2 evidence."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_story_2025_h2"


def flat_means(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    metrics = ["seq_residual_R2", "last_residual_R2", "extreme_abs_q95_residual_R2", "event_PR_AUC"]
    for model in models:
        subset = frame.loc[frame["model"] == model]
        row = {"model": model}
        for metric in metrics:
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_std"] = float(subset[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    models = pd.read_csv(ROOT / "results" / "paper_physics_factorial_2025_h2" / "all_model_runs.csv")
    selected = [
        "GNN-BiGRU baseline",
        "GNN-BiGRU + physics loss",
        "Multistate GWN baseline",
        "GWN + frozen causal ODE",
        "GWN + no-prior fine-tune",
        "GWN + joint causal ODE",
        "HS-DT baseline",
        "HS-DT + no-prior fine-tune",
        "HS-DT + dual causal ODE",
    ]
    summary = flat_means(models, selected)
    summary.to_csv(OUT / "h2_story_model_table.csv", index=False)

    short = {
        "GNN-BiGRU baseline": "GNN",
        "GNN-BiGRU + physics loss": "GNN+loss",
        "Multistate GWN baseline": "GWN",
        "GWN + frozen causal ODE": "GWN+ODE\n(frozen)",
        "GWN + no-prior fine-tune": "GWN\ncapacity",
        "GWN + joint causal ODE": "GWN+ODE\n(joint)",
        "HS-DT baseline": "HS-DT",
        "HS-DT + no-prior fine-tune": "HS-DT\ncapacity",
        "HS-DT + dual causal ODE": "HS-DT+ODE",
    }
    colors = ["#5B6573", "#7A5A5A", "#2F7D6D", "#458F80", "#B07A32", "#168AAD", "#4C5E7A", "#B07A32", "#7B5EA7"]
    metric_specs = [
        ("seq_residual_R2", "Sequence R2"),
        ("last_residual_R2", "Lead-24 R2"),
        ("extreme_abs_q95_residual_R2", "q95 R2"),
        ("event_PR_AUC", "Event PR-AUC"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    x = np.arange(len(summary))
    for axis, (metric, title) in zip(axes.flat, metric_specs):
        axis.bar(x, summary[f"{metric}_mean"], yerr=summary[f"{metric}_std"], capsize=2.5, color=colors)
        axis.set_xticks(x)
        axis.set_xticklabels([short[name] for name in summary["model"]], rotation=34, ha="right", fontsize=8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Locked 2025 H2 chronology: architecture, capacity, and physics-conditioned candidates", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "h2_model_story_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    lead = pd.read_csv(ROOT / "results" / "hsdt_expert_physics_conditioned" / "formal_analysis" / "per_lead_mean_std.csv", header=[0, 1])
    lead.columns = ["config", "lead_hour", "r2_mean", "r2_std", "r2_count", "rmse_mean", "rmse_std", "rmse_count"]
    station = pd.read_csv(ROOT / "results" / "hsdt_expert_physics_conditioned" / "formal_analysis" / "per_station_mean_std.csv", header=[0, 1])
    station.columns = [
        "config", "station_id", "sequence_mean", "sequence_std", "sequence_count",
        "lead24_mean", "lead24_std", "lead24_count", "rmse_mean", "rmse_std", "rmse_count",
    ]
    baseline_lead = lead.loc[lead["config"] == "hsdt_baseline"]
    ode_lead = lead.loc[lead["config"] == "hsdt_both_causal_ode_experts"]
    lead_delta = ode_lead.set_index("lead_hour")["r2_mean"] - baseline_lead.set_index("lead_hour")["r2_mean"]
    lead_delta.to_csv(OUT / "hsdt_dual_ode_per_lead_sequence_delta.csv", header=["sequence_R2_delta"])
    station_base = station.loc[station["config"] == "hsdt_baseline"].set_index("station_id")
    station_ode = station.loc[station["config"] == "hsdt_both_causal_ode_experts"].set_index("station_id")
    station_delta = station_ode["sequence_mean"] - station_base["sequence_mean"]
    station_delta.to_csv(OUT / "hsdt_dual_ode_station_sequence_delta.csv", header=["sequence_R2_delta"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    axes[0].plot(baseline_lead["lead_hour"], baseline_lead["r2_mean"], label="HS-DT baseline", color="#4C5E7A", linewidth=2)
    axes[0].plot(ode_lead["lead_hour"], ode_lead["r2_mean"], label="HS-DT + dual ODE", color="#7B5EA7", linewidth=2)
    axes[0].set_xlabel("Forecast lead (h)")
    axes[0].set_ylabel("Residual R2")
    axes[0].set_title("Lead-wise trajectory skill")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    ids = station_delta.index.astype(str)
    axes[1].bar(ids, station_delta.to_numpy(), color=["#7B5EA7" if value >= 0 else "#A14B5A" for value in station_delta])
    axes[1].axhline(0, color="#222222", linewidth=0.9)
    axes[1].set_xlabel("Station ID")
    axes[1].set_ylabel("Sequence R2 delta")
    axes[1].set_title("Dual ODE minus HS-DT by station")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "hsdt_dual_ode_robustness.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
