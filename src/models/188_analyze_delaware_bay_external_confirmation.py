"""Analyze the preregistered Delaware Bay external confirmation without tuning."""
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
RESULTS = ROOT / "results" / "delaware_bay_external_confirmation_20260812"
ASSETS = ROOT / "results" / "delaware_bay_external_confirmation_20260812" / "paper_assets"
SEEDS = [42, 123, 2024, 2025, 3407]


def write_latex_rows(summary: pd.DataFrame, decision: dict, assets: Path) -> None:
    labels = {
        "varx_ridge": "VARX--Ridge",
        "gwn_eta_only": "Eta-only FS-GWN",
        "gwn_multistate_no_physics": "Multistate FS-GWN",
        "hs_dt_gwn": "HS-DT-GWN",
    }
    order = ["varx_ridge", "gwn_eta_only", "gwn_multistate_no_physics", "hs_dt_gwn"]
    metric_columns = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE"]
    model_rows = []
    for model in order:
        row = summary.loc[summary["model"] == model].iloc[0]
        values = []
        for metric in metric_columns:
            mean = float(row[f"{metric}_mean"])
            if model == "varx_ridge":
                values.append(f"{mean:.4f}")
            else:
                std = float(row[f"{metric}_std"])
                values.append(f"{mean:.4f} $\\pm$ {std:.4f}")
        model_rows.append(f"{labels[model]} & " + " & ".join(values) + r" \\")
    (assets / "external_region_model_rows.tex").write_text("\n".join(model_rows) + "\n", encoding="utf-8")

    status = lambda value: "Pass" if value else "Fail"
    criteria_rows = [
        "Supervision structure & "
        + status(decision["supervision_structure_replicated"])
        + f" & leads {decision['eta_winning_leads']} eta/{decision['multistate_winning_leads']} multi; "
        + f"stations {decision['eta_winning_stations']} eta/{decision['multistate_winning_stations']} multi"
        + r" \\",
        "HS-DT complementarity & "
        + status(decision["hsdt_complementarity_replicated"])
        + f" & $\\Delta R^2$ vs eta {decision['hsdt_minus_eta_sequence_R2']:+.4f}; "
        + f"vs multi {decision['hsdt_minus_multistate_sequence_R2']:+.4f}"
        + r" \\",
        "Linear--deep hierarchy & "
        + status(decision["linear_deep_hierarchy_pattern_replicated"])
        + f" & sequence $\\Delta R^2$ {decision['HS_DT_minus_VARX_sequence_R2']:+.4f}; "
        + f"Lead-24 {decision['lead24_HS_DT_minus_VARX_R2']:+.4f}"
        + r" \\",
    ]
    (assets / "external_confirmation_criteria_rows.tex").write_text("\n".join(criteria_rows) + "\n", encoding="utf-8")


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    denominator = np.sum((true - np.mean(true)) ** 2)
    return float(1.0 - np.sum((true - pred) ** 2) / max(float(denominator), 1e-12))


def load_predictions(results: Path):
    bundles = []
    for seed in SEEDS:
        data = np.load(results / f"seed_{seed}" / "predictions.npz")
        bundles.append({key: data[key].astype(np.float64) for key in ("gwn_eta_only", "gwn_multistate_no_physics", "hs_dt_gwn", "true_residual")})
    varx_npz = np.load(results / "varx" / "predictions.npz")
    varx = {"pred": varx_npz["pred_residual"].astype(np.float64), "true": varx_npz["true_residual"].astype(np.float64)}
    station_ids = np.load(results / f"seed_{SEEDS[0]}" / "predictions.npz")["station_ids"].astype(str)
    return bundles, varx, station_ids


def block_bootstrap_delta(bundles: list[dict], candidate: str, baseline: str, block: int, replicates: int, seed: int = 20260812):
    rng = np.random.default_rng(seed)
    true = bundles[0]["true_residual"]
    n = true.shape[0]
    blocks_needed = int(np.ceil(n / block))
    starts = np.arange(max(1, n - block + 1))
    # Pre-aggregate over station and lead dimensions. Selecting forecast
    # origins from these sufficient statistics is algebraically identical to
    # indexing the full arrays, but avoids copying ~1 million values per draw.
    elements_per_origin = int(np.prod(true.shape[1:]))
    true_sum = true.reshape(n, -1).sum(axis=1)
    true_sum_sq = np.square(true.reshape(n, -1)).sum(axis=1)
    candidate_sse = np.stack([
        np.square(payload[candidate] - true).reshape(n, -1).sum(axis=1)
        for payload in bundles
    ])
    baseline_sse = np.stack([
        np.square(payload[baseline] - true).reshape(n, -1).sum(axis=1)
        for payload in bundles
    ])
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = np.concatenate([(np.arange(start, start + block) % n) for start in rng.choice(starts, size=blocks_needed, replace=True)])[:n]
        selected_seeds = rng.integers(0, len(bundles), size=len(bundles))
        sample_sum = float(true_sum[indices].sum())
        sample_sum_sq = float(true_sum_sq[indices].sum())
        sample_count = len(indices) * elements_per_origin
        denominator = sample_sum_sq - sample_sum * sample_sum / sample_count
        deltas = [
            (baseline_sse[int(seed_index), indices].sum() - candidate_sse[int(seed_index), indices].sum())
            / max(float(denominator), 1e-12)
            for seed_index in selected_seeds
        ]
        values[replicate] = float(np.mean(deltas))
    point = np.mean([r2(payload["true_residual"], payload[candidate]) - r2(payload["true_residual"], payload[baseline]) for payload in bundles])
    return {"point_delta_R2": float(point), "ci95_low": float(np.quantile(values, 0.025)), "ci95_high": float(np.quantile(values, 0.975)), "probability_positive": float(np.mean(values > 0)), "block_hours": block, "replicates": replicates}


def specialization(bundles: list[dict], station_ids: np.ndarray):
    eta = np.mean(np.stack([bundle["gwn_eta_only"] for bundle in bundles]), axis=0)
    multi = np.mean(np.stack([bundle["gwn_multistate_no_physics"] for bundle in bundles]), axis=0)
    true = bundles[0]["true_residual"]
    heatmap = np.empty((len(station_ids), true.shape[-1]))
    lead_rows = []
    for lead in range(true.shape[-1]):
        eta_r2 = r2(true[..., lead], eta[..., lead])
        multi_r2 = r2(true[..., lead], multi[..., lead])
        lead_rows.append({"lead_hour": lead + 1, "eta_R2": eta_r2, "multistate_R2": multi_r2, "multistate_minus_eta_R2": multi_r2 - eta_r2, "winner": "multistate" if multi_r2 > eta_r2 else "eta"})
        for station in range(len(station_ids)):
            heatmap[station, lead] = np.mean((eta[:, station, lead] - true[:, station, lead]) ** 2) - np.mean((multi[:, station, lead] - true[:, station, lead]) ** 2)
    station_rows = []
    for station, station_id in enumerate(station_ids):
        eta_r2 = r2(true[:, station], eta[:, station])
        multi_r2 = r2(true[:, station], multi[:, station])
        station_rows.append({"station_id": station_id, "eta_sequence_R2": eta_r2, "multistate_sequence_R2": multi_r2, "multistate_minus_eta_R2": multi_r2 - eta_r2, "winner": "multistate" if multi_r2 > eta_r2 else "eta"})
    return pd.DataFrame(lead_rows), pd.DataFrame(station_rows), heatmap


def hierarchy(bundles: list[dict], varx: dict):
    hsdt = np.mean(np.stack([bundle["hs_dt_gwn"] for bundle in bundles]), axis=0)
    true = bundles[0]["true_residual"]
    if not np.array_equal(true.astype(np.float32), varx["true"].astype(np.float32)):
        raise RuntimeError("VARX and neural truth arrays are not aligned")
    rows = []
    for lead in range(true.shape[-1]):
        varx_r2 = r2(true[..., lead], varx["pred"][..., lead])
        hsdt_r2 = r2(true[..., lead], hsdt[..., lead])
        rows.append({"lead_hour": lead + 1, "VARX_R2": varx_r2, "HS_DT_R2": hsdt_r2, "HS_DT_minus_VARX_R2": hsdt_r2 - varx_r2})
    return pd.DataFrame(rows), {"VARX_sequence_R2": r2(true, varx["pred"]), "HS_DT_sequence_R2": r2(true, hsdt), "HS_DT_minus_VARX_sequence_R2": r2(true, hsdt) - r2(true, varx["pred"])}


def make_figure(summary: pd.DataFrame, lead_specialization: pd.DataFrame, hierarchy_lead: pd.DataFrame, heatmap: np.ndarray, stations: pd.DataFrame, output: Path):
    station_ids = stations["station_id"].astype(str).to_numpy()
    fig = plt.figure(figsize=(13.2, 8.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1], hspace=0.34, wspace=0.30)
    ax = fig.add_subplot(grid[0, 0])
    order = ["varx_ridge", "gwn_eta_only", "gwn_multistate_no_physics", "hs_dt_gwn"]
    labels = ["VARX", "Eta FS-GWN", "Multi FS-GWN", "HS-DT"]
    values = [float(summary.loc[summary["model"] == model, "seq_residual_R2_mean"].iloc[0]) for model in order]
    ax.bar(np.arange(4), values, color=["#4B5563", "#4878A8", "#D98B3A", "#2E7D62"])
    ax.set_xticks(np.arange(4), labels, rotation=18, ha="right")
    ax.set_ylabel("Sequence residual $R^2$")
    ax.set_title("(a) Prospectively locked 10-station region")
    ax.grid(axis="y", alpha=0.22)

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(hierarchy_lead["lead_hour"], hierarchy_lead["VARX_R2"], color="#4B5563", linewidth=2.2, label="VARX")
    ax.plot(hierarchy_lead["lead_hour"], hierarchy_lead["HS_DT_R2"], color="#2E7D62", linewidth=2.2, label="HS-DT")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Lead hour")
    ax.set_ylabel("Residual $R^2$")
    ax.set_title("(b) Lead-dependent linear/deep hierarchy")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)

    ax = fig.add_subplot(grid[1, 0])
    ax.plot(stations["lon"], stations["lat"], color="#9CA3AF", linewidth=1.2, zorder=1)
    ax.scatter(stations["lon"], stations["lat"], s=48, color="#2E7D62", edgecolor="white", linewidth=0.7, zorder=2)
    for row in stations.itertuples(index=False):
        label = str(row.station_name).replace(", Delaware River", "")
        ax.annotate(label, (row.lon, row.lat), xytext=(4, 2), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("(c) Prospectively locked Delaware Bay--River graph")
    ax.grid(alpha=0.20)

    ax = fig.add_subplot(grid[1, 1])
    limit = float(np.nanpercentile(np.abs(heatmap), 95))
    image = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(24), np.arange(1, 25))
    ax.set_yticks(np.arange(len(station_ids)), station_ids)
    ax.set_xlabel("Lead hour")
    ax.set_ylabel("NOAA station")
    ax.set_title("(d) Supervision-associated MSE difference")
    fig.colorbar(image, ax=ax, pad=0.015, label="Eta MSE - Multistate MSE")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    assets = args.results / "paper_assets"
    assets.mkdir(parents=True, exist_ok=True)
    bundles, varx, station_ids = load_predictions(args.results)
    all_runs = pd.read_csv(args.results / "all_runs.csv")
    metrics = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "descriptive_train_q95_residual_R2"]
    neural_summary = all_runs[all_runs["model"] != "varx_ridge"].groupby("model")[metrics].agg(["mean", "std", "count"]).reset_index()
    neural_summary.columns = ["_".join(str(value) for value in column if value) for column in neural_summary.columns.to_flat_index()]
    varx_run = all_runs[all_runs["model"] == "varx_ridge"].iloc[0]
    varx_summary = {"model": "varx_ridge"}
    for metric in metrics:
        varx_summary[f"{metric}_mean"] = float(varx_run[metric])
        varx_summary[f"{metric}_std"] = 0.0
        varx_summary[f"{metric}_count"] = 1
    summary = pd.concat([neural_summary, pd.DataFrame([varx_summary])], ignore_index=True)
    summary.to_csv(assets / "external_region_model_summary.csv", index=False)

    lead_specialization, station_specialization, heatmap = specialization(bundles, station_ids)
    lead_specialization.to_csv(assets / "external_supervision_per_lead.csv", index=False)
    station_specialization.to_csv(assets / "external_supervision_per_station.csv", index=False)
    hierarchy_lead, hierarchy_sequence = hierarchy(bundles, varx)
    hierarchy_lead.to_csv(assets / "external_varx_hsdt_per_lead.csv", index=False)

    hs_eta = block_bootstrap_delta(bundles, "hs_dt_gwn", "gwn_eta_only", 168, args.replicates)
    hs_multi = block_bootstrap_delta(bundles, "hs_dt_gwn", "gwn_multistate_no_physics", 168, args.replicates, seed=20260813)
    bootstrap = pd.DataFrame([{"comparison": "HS-DT - Eta", **hs_eta}, {"comparison": "HS-DT - Multistate", **hs_multi}])
    bootstrap.to_csv(assets / "external_hsdt_block_bootstrap.csv", index=False)

    eta_leads = int((lead_specialization["winner"] == "eta").sum())
    multi_leads = int((lead_specialization["winner"] == "multistate").sum())
    eta_stations = int((station_specialization["winner"] == "eta").sum())
    multi_stations = int((station_specialization["winner"] == "multistate").sum())
    supervision_ok = eta_leads > 0 and multi_leads > 0 and eta_stations > 0 and multi_stations > 0
    complementarity_ok = hs_eta["point_delta_R2"] > 0 and hs_multi["point_delta_R2"] > 0
    early_sign = float(hierarchy_lead.loc[hierarchy_lead["lead_hour"] <= 8, "HS_DT_minus_VARX_R2"].mean())
    lead24_sign = float(hierarchy_lead.loc[hierarchy_lead["lead_hour"] == 24, "HS_DT_minus_VARX_R2"].iloc[0])
    differs_from_original = hierarchy_sequence["HS_DT_minus_VARX_sequence_R2"] > 0
    hierarchy_ok = early_sign * lead24_sign < 0 or differs_from_original
    passed = int(supervision_ok) + int(complementarity_ok) + int(hierarchy_ok)
    decision = {
        "supervision_structure_replicated": bool(supervision_ok),
        "eta_winning_leads": eta_leads,
        "multistate_winning_leads": multi_leads,
        "eta_winning_stations": eta_stations,
        "multistate_winning_stations": multi_stations,
        "hsdt_complementarity_replicated": bool(complementarity_ok),
        "hsdt_minus_eta_sequence_R2": hs_eta["point_delta_R2"],
        "hsdt_minus_multistate_sequence_R2": hs_multi["point_delta_R2"],
        "linear_deep_hierarchy_pattern_replicated": bool(hierarchy_ok),
        **hierarchy_sequence,
        "early_lead_group": [1, 8],
        "early_group_definition_status": "fixed before external neural scoring but after the local protocol hash",
        "early_lead_HS_DT_minus_VARX_mean_R2": early_sign,
        "lead24_HS_DT_minus_VARX_R2": lead24_sign,
        "criteria_passed": passed,
        "criteria_total": 3,
        "external_confirmation_success": bool(passed >= 2),
        "report_unfavorable_results": True,
    }
    (assets / "EXTERNAL_CONFIRMATION_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_latex_rows(summary, decision, assets)
    stations = pd.read_csv(ROOT / "data" / "external_region_delaware_bay_2023_2025" / "processed" / "station_order.csv", dtype={"station_id": str})
    make_figure(summary, lead_specialization, hierarchy_lead, heatmap, stations, assets / "fig_external_region_confirmation.png")
    print(json.dumps(decision, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
