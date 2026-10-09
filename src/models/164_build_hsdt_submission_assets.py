from __future__ import annotations

"""Build publication figures for the independently validated HS-DT paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "publication_final" / "overleaf_hsdt_independent_validation_submission_20260810" / "figures"
SEEDS = [42, 123, 2024, 2025, 3407]
STATIONS = ["New London", "Montauk", "Kings Point", "The Battery", "Sandy Hook", "Atlantic City", "Cape May"]
COLORS = {
    "eta": "#397A8A",
    "multi": "#D28B37",
    "hsdt": "#2D6A4F",
    "physics": "#A23B4A",
    "control": "#666666",
    "gold": "#C7A12A",
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )


def add_box(ax, xy, width, height, text, face, edge="#333333", fontsize=9, weight="normal"):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)
    return patch


def arrow(ax, start, end, color="#444444", style="-|>", width=1.2):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, linewidth=width, color=color))


def model_framework() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_box(ax, (0.02, 0.36), 0.16, 0.28, "Past 24 h\n7 stations x 34 inputs", "#E9F1F3", weight="bold")
    add_box(ax, (0.25, 0.61), 0.18, 0.22, "Eta-only\nFS-GWN expert", "#D9E9ED", edge=COLORS["eta"], weight="bold")
    add_box(ax, (0.25, 0.17), 0.18, 0.22, "Multistate\nFS-GWN expert", "#F5E5CF", edge=COLORS["multi"], weight="bold")
    add_box(ax, (0.51, 0.38), 0.20, 0.25, "Horizon-specialized fusion\nLeads 1-23: 0.5/0.5\nLead 24: multistate", "#DDEFE4", edge=COLORS["hsdt"], weight="bold")
    add_box(ax, (0.78, 0.58), 0.19, 0.20, "Continuous head\n24-h residual trajectory", "#DDEFE4", edge=COLORS["hsdt"], weight="bold")
    add_box(ax, (0.50, 0.05), 0.21, 0.19, "Causal ODE prior +\nlast-input forcing +\nexpert discrepancy", "#F4E0E4", edge=COLORS["physics"])
    add_box(ax, (0.78, 0.15), 0.19, 0.20, "Physics event head\nEvent-risk ranking", "#F4E0E4", edge=COLORS["physics"], weight="bold")
    arrow(ax, (0.18, 0.54), (0.25, 0.72), COLORS["eta"])
    arrow(ax, (0.18, 0.46), (0.25, 0.28), COLORS["multi"])
    arrow(ax, (0.43, 0.72), (0.51, 0.56), COLORS["eta"])
    arrow(ax, (0.43, 0.28), (0.51, 0.44), COLORS["multi"])
    arrow(ax, (0.71, 0.55), (0.78, 0.68), COLORS["hsdt"])
    arrow(ax, (0.18, 0.39), (0.50, 0.14), COLORS["physics"])
    arrow(ax, (0.71, 0.14), (0.78, 0.25), COLORS["physics"])
    ax.text(0.5, 0.96, "HS-DT-GWN with task-specific use of physical information", ha="center", va="top", fontsize=12, weight="bold")
    ax.text(0.875, 0.49, "Separate outputs; metrics are not mixed", ha="center", color="#555555", fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "hsdt_dual_head_framework.png", bbox_inches="tight")
    plt.close(fig)


def independent_validation() -> None:
    root = RESULTS / "formal_hsdt_independent_validation_2026"
    runs = pd.read_csv(root / "all_runs.csv")
    boot = pd.read_csv(root / "block_bootstrap_summary.csv")
    models = ["gwn_eta_only", "gwn_multistate_no_physics", "hs_dt_gwn"]
    labels = ["Eta-only", "Multistate", "HS-DT"]
    colors = [COLORS["eta"], COLORS["multi"], COLORS["hsdt"]]
    periods = ["2026_h1", "2026_july"]
    period_labels = ["2026 H1", "2026 July"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    x = np.arange(len(periods))
    width = 0.23
    for index, (model, label, color) in enumerate(zip(models, labels, colors)):
        means, stds = [], []
        for period in periods:
            values = runs[(runs["period"] == period) & (runs["model"] == model)]["seq_residual_R2"]
            means.append(values.mean())
            stds.append(values.std(ddof=1))
        axes[0].bar(x + (index - 1) * width, means, width, yerr=stds, capsize=3, color=color, alpha=0.9, label=label)
    axes[0].set_xticks(x, period_labels)
    axes[0].set_ylabel("Sequence residual $R^2$")
    axes[0].set_ylim(0.35, 0.57)
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[0].set_title("a  Independent chronological performance", loc="left", weight="bold")

    subset = boot[(boot["baseline"] == "multi") & (boot["metric"] == "sequence")].set_index("period")
    means = [subset.loc[p, "mean_delta_r2"] for p in periods]
    low = [subset.loc[p, "ci95_low"] for p in periods]
    high = [subset.loc[p, "ci95_high"] for p in periods]
    err = np.asarray([[m - l for m, l in zip(means, low)], [h - m for m, h in zip(means, high)]])
    axes[1].axhline(0, color="#333333", linewidth=0.9)
    axes[1].errorbar(x, means, yerr=err, fmt="o", markersize=8, capsize=5, color=COLORS["hsdt"], linewidth=2)
    axes[1].set_xticks(x, period_labels)
    axes[1].set_ylabel(r"HS-DT minus Multistate, $\Delta R^2$")
    axes[1].set_ylim(-0.004, 0.034)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].set_title("b  168-h block-bootstrap evidence", loc="left", weight="bold")
    for idx, (mean, lo, hi) in enumerate(zip(means, low, high)):
        axes[1].text(idx, hi + 0.002, f"{mean:+.3f}\n[{lo:.3f}, {hi:.3f}]", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "independent_validation_summary.png", bbox_inches="tight")
    plt.close(fig)


def physics_task_evidence() -> None:
    h1 = pd.read_csv(RESULTS / "frozen_2026_h1_physics_reliability_gwn" / "paired_comparisons.csv")
    july = pd.read_csv(RESULTS / "frozen_2026_july_safety_gated_dual_head_gwn" / "paired_comparisons.csv")
    h1 = h1[h1["baseline"] == "validation_locked_ridge_no_physics"].set_index("metric")
    july = july.set_index("metric")
    continuous_metrics = ["seq_residual_R2", "last_residual_R2", "extreme_abs_q95_residual_R2"]
    labels = ["Sequence", "Lead 24", "Absolute q95"]
    h1_values = [h1.loc[m, "mean_improvement"] for m in continuous_metrics]
    july_values = [july.loc[m, "mean_improvement"] for m in continuous_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.15), gridspec_kw={"width_ratios": [1.35, 1.0]})
    x = np.arange(len(labels))
    width = 0.34
    axes[0].axhline(0, color="#333333", linewidth=0.9)
    axes[0].bar(x - width / 2, h1_values, width, color=COLORS["physics"], alpha=0.78, label="H1 physics correction")
    axes[0].bar(x + width / 2, july_values, width, color=COLORS["gold"], alpha=0.88, label="July safety gate")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel(r"Improvement over matched no-physics, $\Delta R^2$")
    axes[0].set_title("a  Continuous correction does not transfer", loc="left", weight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.22)

    event_values = [h1.loc["event_PR_AUC", "mean_improvement"], july.loc["event_PR_AUC", "mean_improvement"]]
    event_stds = [h1.loc["event_PR_AUC", "std_improvement"], july.loc["event_PR_AUC", "std_improvement"]]
    axes[1].axhline(0, color="#333333", linewidth=0.9)
    axes[1].bar([0, 1], event_values, yerr=event_stds, capsize=4, color=[COLORS["physics"], COLORS["gold"]], alpha=0.88)
    axes[1].set_xticks([0, 1], ["2026 H1", "2026 July"])
    axes[1].set_ylabel("Event PR-AUC improvement")
    axes[1].set_title("b  Physics event ranking remains positive", loc="left", weight="bold")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].set_ylim(0.0, 0.0305)
    for idx, value in enumerate(event_values):
        y = value + max(event_stds[idx], 0.0003) + 0.0005
        axes[1].text(idx, min(y, 0.0282), f"{value:+.4f}\n5/5 seeds", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "physics_task_specific_evidence.png", bbox_inches="tight")
    plt.close(fig)


def july_prediction_case() -> None:
    root = RESULTS / "formal_hsdt_independent_validation_2026"
    true = None
    hsdt, multi, times = [], [], None
    for seed in SEEDS:
        with np.load(root / f"seed_{seed}_july_base_predictions.npz") as payload:
            if true is None:
                true = payload["true_residual"]
                times = payload["forecast_start"]
            hsdt.append(payload["pred_hsdt"])
            multi.append(payload["pred_multi"])
    assert true is not None and times is not None
    hsdt_mean = np.mean(hsdt, axis=0)
    multi_mean = np.mean(multi, axis=0)
    origin = int(np.argmax(np.max(np.abs(true), axis=(1, 2))))
    leads = np.arange(1, true.shape[-1] + 1)

    fig, axes = plt.subplots(4, 2, figsize=(10.8, 9.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for station, ax in enumerate(axes[:7]):
        ax.plot(leads, true[origin, station], color="#202020", linewidth=1.8, label="Observed")
        ax.plot(leads, multi_mean[origin, station], color=COLORS["multi"], linewidth=1.35, label="Multistate")
        ax.plot(leads, hsdt_mean[origin, station], color=COLORS["hsdt"], linewidth=1.35, label="HS-DT")
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.set_title(STATIONS[station], loc="left", fontsize=9, weight="bold")
        ax.grid(alpha=0.18)
    axes[7].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[7].legend(handles, labels, loc="center", frameon=False, fontsize=10)
    for ax in axes[6:7]:
        ax.set_xlabel("Forecast lead (h)")
    fig.supylabel("Non-tidal residual (m)", x=0.02, fontsize=10)
    timestamp = pd.Timestamp(times[origin]).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"Frozen July holdout example selected by observed amplitude only\nForecast origin: {timestamp}", y=0.995, fontsize=11, weight="bold")
    fig.tight_layout(rect=[0.03, 0.02, 1, 0.96])
    fig.savefig(OUT / "july_prediction_example.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    model_framework()
    independent_validation()
    physics_task_evidence()
    july_prediction_case()
    for path in sorted(OUT.glob("*.png")):
        print(f"{path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
