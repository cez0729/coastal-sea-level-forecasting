"""Build publication assets for specialization and temporal hierarchy results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "results" / "paper_mainline_assets_20260811"
MODELS = ["varx_ridge", "gwn_eta_only", "gwn_multistate_no_physics", "hs_dt_gwn"]
COLORS = {
    "varx_ridge": "#111111",
    "gwn_eta_only": "#2A6FBB",
    "gwn_multistate_no_physics": "#D98C10",
    "hs_dt_gwn": "#388E3C",
}
LABELS = {
    "varx_ridge": "VARX-Ridge",
    "gwn_eta_only": "Eta-only GWN",
    "gwn_multistate_no_physics": "Multistate GWN",
    "hs_dt_gwn": "HS-DT-GWN",
}
STATIONS = ["New London", "Montauk", "Kings Point", "The Battery", "Sandy Hook", "Atlantic City", "Cape May"]
STATION_IDS = [8461490, 8510560, 8516945, 8518750, 8531680, 8534720, 8536110]

STRICT_ORDER = [
    "GNN physics loss minus baseline",
    "GWN frozen ODE minus baseline",
    "GWN joint ODE minus baseline",
    "GWN joint ODE minus capacity control",
    "HS-DT capacity minus baseline",
    "HS-DT dual ODE minus baseline",
    "HS-DT dual ODE minus capacity control",
    "Direct physics loss on ODE HS-DT",
]
# Event PR-AUC is retained in the complete audit tables, but it is not part of
# the manuscript mainline because this study does not define an operational
# surge-event task. The main figure therefore shows only continuous metrics.
STRICT_METRICS = ["seq_residual_R2", "last_residual_R2", "extreme_abs_q95_residual_R2"]


def hierarchy_curves(output: Path) -> None:
    strict = pd.read_csv(ROOT / "results" / "strict_varx_deep_robustness_audit_2025_h2" / "per_lead_mean_std.csv")
    strict_rows = []
    for model in ("gwn_eta_only", "gwn_multistate_no_physics", "hs_dt_gwn"):
        part = strict[strict["model"] == model]
        for _, row in part.iterrows():
            strict_rows.append({"period": "2025_h2", "model": model, "lead": int(row["lead"]), "mean": row["deep_R2_mean"], "std": row["deep_R2_std"]})
    varx = strict.groupby("lead", as_index=False)["varx_R2_mean"].mean()
    for _, row in varx.iterrows():
        strict_rows.append({"period": "2025_h2", "model": "varx_ridge", "lead": int(row["lead"]), "mean": row["varx_R2_mean"], "std": 0.0})
    later = pd.read_csv(ROOT / "results" / "frozen_varx_2026_diagnostic_20260811" / "per_lead_mean_std.csv")
    later = later[["period", "model", "lead", "mean", "std"]]
    curves = pd.concat([pd.DataFrame(strict_rows), later], ignore_index=True)
    curves.to_csv(output / "temporal_hierarchy_per_lead.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), sharey=False)
    titles = {
        "2025_h2": "2025 H2 chronological refit",
        "2026_h1": "2026 H1 diagnostic overlay",
        "2026_july": "2026 July diagnostic overlay",
    }
    for axis, period in zip(axes, titles):
        for model in MODELS:
            part = curves[(curves["period"] == period) & (curves["model"] == model)]
            axis.plot(part["lead"], part["mean"], color=COLORS[model], linewidth=2.5 if model == "varx_ridge" else 1.9, label=LABELS[model])
        axis.axhline(0.0, color="#555555", linewidth=0.7, alpha=0.5)
        axis.set_title(titles[period], fontsize=10, weight="bold")
        axis.set_xlabel("Forecast lead (h)")
        axis.set_xlim(1, 24)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Residual R2")
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Model hierarchy is horizon- and period-dependent", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output / "model_hierarchy_temporal_regimes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def station_horizon_heatmap(output: Path) -> None:
    source = pd.read_csv(ROOT / "results" / "horizon_specialization_complementarity_20260811" / "complementarity_by_seed_station_lead.csv")
    source = source[source["protocol"] == "strict_2025_h2"].copy()
    source["HSI"] = source["multistate_R2"] - source["eta_R2"]
    summary = source.groupby(["station_id", "lead"], as_index=False).agg(
        HSI_mean=("HSI", "mean"),
        HSI_std=("HSI", "std"),
        error_correlation_mean=("error_correlation", "mean"),
        eta_win_probability_mean=("eta_win_probability", "mean"),
    )
    summary.to_csv(output / "station_horizon_specialization.csv", index=False)
    matrix = summary.pivot(index="station_id", columns="lead", values="HSI_mean").reindex(STATION_IDS)
    vmax = float(np.nanmax(np.abs(matrix.to_numpy())))
    fig, ax = plt.subplots(figsize=(11.8, 4.4))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_yticks(range(len(STATIONS)), STATIONS)
    ax.set_xticks(range(0, 24, 2), range(1, 25, 2))
    ax.set_xlabel("Forecast lead (h)")
    ax.set_title("Station-by-horizon supervision specialization, strict 2025 H2", weight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("HSI = R2(multistate) - R2(eta-only)")
    fig.tight_layout()
    fig.savefig(output / "station_horizon_specialization_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def physics_mainline(output: Path) -> None:
    strict = pd.read_csv(ROOT / "results" / "architecture_conditional_physics_utility_20260811" / "strict_physics_utility_matrix.csv")
    regime = pd.read_csv(ROOT / "results" / "architecture_conditional_physics_utility_20260811" / "retrospective_regime_utility_matrix.csv")
    matrix = strict.pivot(index="comparison", columns="metric", values="mean_delta").loc[STRICT_ORDER, STRICT_METRICS]
    vmax = float(np.abs(matrix.to_numpy()).max())
    q95 = regime[(regime["component"] == "combined") & np.isclose(regime["train_quantile"], 0.95)].copy()
    order = ["gnn_ode_prior_minus_learnable", "gwn_physics_loss_minus_no_physics", "orc_minus_hsdt", "orc_minus_zero", "orc_minus_persistence"]
    q95 = q95.set_index("comparison").loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [1.65, 1.0]})
    image = axes[0].imshow(matrix.to_numpy(), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), aspect="auto")
    axes[0].set_xticks(range(len(STRICT_METRICS)), ["Sequence", "Lead-24", "q95"])
    axes[0].set_yticks(range(len(STRICT_ORDER)), STRICT_ORDER, fontsize=7)
    axes[0].set_title("a  Strict 2025 H2 metric deltas", loc="left", weight="bold")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            axes[0].text(col, row, f"{value:+.4f}", ha="center", va="center", fontsize=6.5,
                         color="white" if abs(value) > 0.55 * vmax else "black")
    fig.colorbar(image, ax=axes[0], fraction=0.03, pad=0.02, label="Metric delta")

    values = q95["relative_MSE_reduction_pct_mean"].to_numpy()
    axes[1].barh(np.arange(len(q95)), values, color=["#2D7D6D" if value > 0 else "#B04A5A" for value in values])
    axes[1].axvline(0.0, color="#222222", linewidth=0.8)
    axes[1].set_yticks(np.arange(len(q95)), q95["comparison"], fontsize=7)
    axes[1].set_xlim(float(values.min()) - 0.35, float(values.max()) + 0.45)
    axes[1].set_xlabel("Relative MSE reduction (%)")
    axes[1].set_title("b  Retrospective high-forcing q95", loc="left", weight="bold")
    axes[1].grid(axis="x", alpha=0.2)
    for index, row in q95.iterrows():
        value = row["relative_MSE_reduction_pct_mean"]
        x_position = value + 0.03
        axes[1].text(x_position, index, f"{value:+.2f}% ({int(row['seed_wins'])}/5)",
                     va="center", fontsize=7, ha="left",
                     color="white" if value < -0.4 else "black")
    fig.tight_layout()
    fig.savefig(output / "physics_utility_mainline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hierarchy_curves(output)
    station_horizon_heatmap(output)
    physics_mainline(output)
    print(output)


if __name__ == "__main__":
    main()
