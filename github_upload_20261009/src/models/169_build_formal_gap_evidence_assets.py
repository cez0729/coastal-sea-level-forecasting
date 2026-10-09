from __future__ import annotations

"""Build figures and a machine-readable summary for the formal gap experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "publication_final" / "overleaf_hsdt_independent_validation_submission_20260810" / "figures"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 160, "savefig.dpi": 320,
    })


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"MISSING {path}")
        return None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        print(f"ERROR {path}: {exc}")
        return None
    print(f"READ {path} ({len(frame)} rows)")
    return frame


def strong_baseline_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    hsdt = read_csv(RESULTS / "horizon_specialized_dual_task_gwn" / "mean_std.csv")
    if hsdt is not None:
        for _, row in hsdt.iterrows():
            label = {"gwn_eta_only": "Eta-only FS-GWN", "gwn_multistate_no_physics": "Multistate FS-GWN", "horizon_specialized_dual_task_gwn": "HS-DT-GWN", "dual_task_equal_ensemble": "HS-DT-GWN"}.get(str(row["config"]), str(row["config"]))
            rows.append({"model": label, "sequence_R2": float(row["seq_residual_R2_mean"]), "lead24_R2": float(row["last_residual_R2_mean"]), "source": "retrospective 2025 H2"})
    varx = read_csv(RESULTS / "formal_varx_ridge_20260811" / "mean_std.csv")
    if varx is not None:
        for _, row in varx.iterrows():
            rows.append({"model": "VARX-Ridge", "sequence_R2": float(row["seq_residual_R2_mean"]), "lead24_R2": float(row["last_residual_R2_mean"]), "source": "retrospective 2025 H2"})
    att = read_csv(RESULTS / "formal_attention_20260811" / "mean_std.csv")
    if att is not None:
        for _, row in att.iterrows():
            rows.append({"model": "Compact ST-Attention", "sequence_R2": float(row["seq_residual_R2_mean"]), "lead24_R2": float(row["last_residual_R2_mean"]), "source": "retrospective 2025 H2"})
    # Use the consolidated five-seed file. The older final/ directory can
    # contain only a continuation subset and must not define the benchmark.
    adaptive_values: list[tuple[float, float]] = []
    adaptive_all = read_csv(RESULTS / "formal_adaptive_gwn_20260811" / "all_runs.csv")
    if adaptive_all is not None and len(adaptive_all):
        if "stage" in adaptive_all.columns:
            adaptive_all = adaptive_all[adaptive_all["stage"].astype(str).str.lower().eq("final")]
        for _, row in adaptive_all.iterrows():
            seq_col = "test_seq_residual_R2" if "test_seq_residual_R2" in adaptive_all.columns else "seq_residual_R2"
            last_col = "test_last_residual_R2" if "test_last_residual_R2" in adaptive_all.columns else "last_residual_R2"
            adaptive_values.append((float(row[seq_col]), float(row[last_col])))
    if adaptive_values:
        rows.append({"model": "Adaptive-support MixHop GWN", "sequence_R2": np.mean([x[0] for x in adaptive_values]), "lead24_R2": np.mean([x[1] for x in adaptive_values]), "source": f"retrospective 2025 H2 ({len(adaptive_values)} seeds)"})
    summary = pd.DataFrame(rows).drop_duplicates(subset=["model"], keep="last")
    if len(summary):
        summary.to_csv(OUT.parent / "formal_gap_model_summary.csv", index=False)
    return summary


def strong_baseline_plot(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    order = ["VARX-Ridge", "Compact ST-Attention", "Adaptive-support MixHop GWN", "Eta-only FS-GWN", "Multistate FS-GWN", "HS-DT-GWN"]
    summary = summary.set_index("model").reindex([x for x in order if x in summary["model"].tolist()]).dropna().reset_index()
    colors = ["#6C757D", "#8E6C8A", "#457B9D", "#397A8A", "#D28B37", "#2D6A4F"][: len(summary)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(summary))
    for ax, col, title in zip(axes, ["sequence_R2", "lead24_R2"], ["Sequence residual $R^2$", "Lead-24 residual $R^2$"]):
        bars = ax.bar(x, summary[col], color=colors, alpha=0.9)
        ax.set_xticks(x, summary["model"], rotation=35, ha="right")
        ax.set_ylim(0, 0.86)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, summary[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("a  Unified retrospective comparison", loc="left", weight="bold")
    axes[1].set_title("b  Terminal horizon", loc="left", weight="bold")
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "formal_strong_baseline_comparison.png", bbox_inches="tight")
    plt.close(fig)


def per_lead_station_plots() -> None:
    lead = read_csv(RESULTS / "horizon_specialized_dual_task_gwn" / "per_lead_metrics.csv")
    if lead is not None and len(lead):
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        for config, group in lead.groupby("config"):
            label = {"gwn_eta_only": "Eta-only", "gwn_multistate_no_physics": "Multistate", "dual_task_equal_ensemble": "HS-DT"}.get(str(config), str(config))
            ax.plot(group["lead_hour"], group["R2"], marker="o", markersize=2.5, linewidth=1.2, label=label)
        ax.set_xlabel("Forecast lead (h)"); ax.set_ylabel("Residual $R^2$"); ax.set_title("Per-horizon performance on the retrospective benchmark", loc="left", weight="bold"); ax.grid(alpha=0.2); ax.legend(frameon=False, ncol=3)
        fig.tight_layout(); fig.savefig(OUT / "formal_per_lead_comparison.png", bbox_inches="tight"); plt.close(fig)
    station = read_csv(RESULTS / "horizon_specialized_dual_task_gwn" / "per_station_terminal_metrics.csv")
    if station is not None and len(station):
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        pivot = station.pivot_table(index="station_id", columns="config", values="R2", aggfunc="mean")
        configs = [c for c in ["gwn_eta_only", "gwn_multistate_no_physics", "dual_task_equal_ensemble"] if c in pivot.columns]
        x = np.arange(len(pivot)); width = 0.8 / max(1, len(configs))
        for idx, config in enumerate(configs):
            label = {"gwn_eta_only": "Eta-only", "gwn_multistate_no_physics": "Multistate", "dual_task_equal_ensemble": "HS-DT"}.get(config, config)
            ax.bar(x + (idx - (len(configs)-1)/2)*width, pivot[config], width, label=label)
        ax.set_xticks(x, pivot.index.astype(str), rotation=35, ha="right"); ax.set_ylabel("Lead-24 residual $R^2$"); ax.set_title("Per-station terminal performance", loc="left", weight="bold"); ax.grid(axis="y", alpha=0.2); ax.legend(frameon=False, ncol=3)
        fig.tight_layout(); fig.savefig(OUT / "formal_per_station_terminal_comparison.png", bbox_inches="tight"); plt.close(fig)


def factorial_plot() -> None:
    frame = read_csv(RESULTS / "formal_multistate_factorial_20260811" / "all_runs.csv")
    if frame is None:
        frame = read_csv(RESULTS / "formal_multistate_factorial_20260811" / "priority2_physics_gwn_all_runs.csv")
    terminal = read_csv(RESULTS / "formal_multistate_factorial_terminal_20260811" / "priority2_physics_gwn_all_runs.csv")
    if terminal is not None and not terminal.empty:
        frame = pd.concat([frame, terminal], ignore_index=True) if frame is not None else terminal
    if frame is None or frame.empty:
        return
    rows = []
    for config, group in frame.groupby("config"):
        seq_col = "seq_residual_R2" if "seq_residual_R2" in group else "test_seq_residual_R2"
        last_col = "last_residual_R2" if "last_residual_R2" in group else "test_last_residual_R2"
        rows.append((str(config), group[seq_col].astype(float).mean(), group[last_col].astype(float).mean()))
    vals = pd.DataFrame(rows, columns=["config", "sequence_R2", "lead24_R2"])
    vals.to_csv(OUT.parent / "formal_factorial_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.2)); x = np.arange(len(vals)); width = 0.36
    ax.bar(x-width/2, vals.sequence_R2, width, label="Sequence $R^2$", color="#397A8A")
    ax.bar(x+width/2, vals.lead24_R2, width, label="Lead-24 $R^2$", color="#D28B37")
    ax.set_xticks(x, vals.config, rotation=25, ha="right"); ax.set_ylim(0, 0.8); ax.set_ylabel("Residual $R^2$"); ax.set_title("2 x 2 multistate objective ablation", loc="left", weight="bold"); ax.grid(axis="y", alpha=0.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "formal_multistate_factorial_ablation.png", bbox_inches="tight"); plt.close(fig)


def external_plot() -> None:
    frame = read_csv(RESULTS / "external_node_transfer_20260811" / "all_runs.csv")
    if frame is None:
        frame = read_csv(RESULTS / "external_node_transfer_20260811" / "all_runs_partial.csv")
    if frame is None or frame.empty:
        return
    cols = [("original7_R2", "Original 7 stations"), ("external2_R2", "Two external stations"), ("persistence_original7_R2", "Original persistence"), ("persistence_external2_R2", "External persistence")]
    means = [frame[c].astype(float).mean() for c, _ in cols]
    fig, ax = plt.subplots(figsize=(7.5, 4.1)); bars = ax.bar(np.arange(len(cols)), means, color=["#457B9D", "#2D6A4F", "#A8DADC", "#B7B7A4"])
    ax.set_xticks(np.arange(len(cols)), [label for _, label in cols], rotation=25, ha="right"); ax.set_ylabel("Residual $R^2$"); ax.set_ylim(0, 0.75); ax.set_title("Zero-target spatial transfer diagnostic", loc="left", weight="bold"); ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, means): ax.text(bar.get_x()+bar.get_width()/2, value+0.015, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "external_node_transfer_diagnostic.png", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    setup(); summary = strong_baseline_summary(); strong_baseline_plot(summary); per_lead_station_plots(); factorial_plot(); external_plot()
    print(summary.to_string(index=False) if len(summary) else "No formal baseline results available yet")


if __name__ == "__main__":
    main()
