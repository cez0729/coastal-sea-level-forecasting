"""Quantify supervision-associated horizon specialization and complementarity."""
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
OUT_DEFAULT = ROOT / "results" / "horizon_specialization_complementarity_20260811"
SEEDS = [42, 123, 2024, 2025, 3407]


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    denominator = float(np.sum((true - true.mean()) ** 2))
    return float(1.0 - np.sum((true - pred) ** 2) / max(denominator, 1e-12))


def corr(left: np.ndarray, right: np.ndarray) -> float:
    left, right = left.reshape(-1), right.reshape(-1)
    if left.std() < 1e-12 or right.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def load_retro(seed: int):
    root = ROOT / "results" / "priority12_physics_graph_wavenet" / f"seed_{seed}" / "horizon_24h"
    eta_file = np.load(root / "gwn_eta_only" / "predictions.npz")
    multi_file = np.load(root / "gwn_multistate_no_physics" / "predictions.npz")
    eta = eta_file["pred_residual"].astype(np.float64)
    multi = multi_file["pred_states"][..., 0].astype(np.float64)
    true = eta_file["true_residual"].astype(np.float64)
    if not np.allclose(true, multi_file["true_states"][..., 0], atol=1e-7, rtol=0):
        raise RuntimeError(f"Retrospective targets do not align for seed {seed}")
    return {"eta": eta, "multi": multi, "true": true, "times": None, "stations": eta_file["station_ids"]}


def load_strict(seed: int):
    path = ROOT / "results" / "confirmatory_hsdt_orc_refit_2025_h2" / f"seed_{seed}" / "predictions.npz"
    saved = np.load(path)
    return {
        "eta": saved["gwn_eta_only"].astype(np.float64),
        "multi": saved["gwn_multistate_no_physics"].astype(np.float64),
        "true": saved["true_residual"].astype(np.float64),
        "times": saved["target_origin_time"],
        "stations": saved["station_ids"],
    }


def season_labels(times: np.ndarray) -> np.ndarray:
    months = pd.DatetimeIndex(times).month.to_numpy()
    labels = np.full(len(months), "winter", dtype=object)
    labels[np.isin(months, [3, 4, 5])] = "spring"
    labels[np.isin(months, [6, 7, 8])] = "summer"
    labels[np.isin(months, [9, 10, 11])] = "autumn"
    return labels


def analyze_bundle(protocol: str, seed: int, payload: dict):
    eta, multi, true = payload["eta"], payload["multi"], payload["true"]
    if eta.shape != multi.shape or eta.shape != true.shape:
        raise RuntimeError(f"Shape mismatch for {protocol}, seed {seed}")
    hsdt = 0.5 * (eta + multi)
    hsdt[..., -1] = multi[..., -1]
    models = {"eta_only": eta, "multistate": multi, "hs_dt": hsdt}
    model_rows = [{"protocol": protocol, "seed": seed, "model": name, "sequence_R2": r2(true, pred), "lead24_R2": r2(true[..., -1], pred[..., -1])} for name, pred in models.items()]
    lead_rows, station_rows, season_rows = [], [], []
    for lead in range(true.shape[-1]):
        eta_error = true[..., lead] - eta[..., lead]
        multi_error = true[..., lead] - multi[..., lead]
        eta_r2 = r2(true[..., lead], eta[..., lead])
        multi_r2 = r2(true[..., lead], multi[..., lead])
        fusion_r2 = r2(true[..., lead], hsdt[..., lead])
        lead_rows.append({
            "protocol": protocol, "seed": seed, "lead": lead + 1,
            "eta_R2": eta_r2, "multistate_R2": multi_r2,
            "HSI_multistate_minus_eta": multi_r2 - eta_r2,
            "error_correlation": corr(eta_error, multi_error),
            "eta_win_probability": float(np.mean(np.abs(eta_error) < np.abs(multi_error))),
            "multistate_win_probability": float(np.mean(np.abs(multi_error) < np.abs(eta_error))),
            "mean_abs_expert_disagreement": float(np.mean(np.abs(eta[..., lead] - multi[..., lead]))),
            "hsdt_R2": fusion_r2,
            "hsdt_gain_over_best_expert_R2": fusion_r2 - max(eta_r2, multi_r2),
        })
        for station, station_id in enumerate(payload["stations"]):
            se = eta_error[:, station]
            sm = multi_error[:, station]
            station_rows.append({
                "protocol": protocol, "seed": seed, "station_id": str(station_id), "lead": lead + 1,
                "eta_R2": r2(true[:, station, lead], eta[:, station, lead]),
                "multistate_R2": r2(true[:, station, lead], multi[:, station, lead]),
                "error_correlation": corr(se, sm),
                "eta_win_probability": float(np.mean(np.abs(se) < np.abs(sm))),
            })
        if payload["times"] is not None:
            labels = season_labels(payload["times"])
            for season in sorted(set(labels)):
                mask = labels == season
                season_rows.append({
                    "protocol": protocol, "seed": seed, "season": season, "lead": lead + 1,
                    "error_correlation": corr(eta_error[mask], multi_error[mask]),
                    "eta_win_probability": float(np.mean(np.abs(eta_error[mask]) < np.abs(multi_error[mask]))),
                    "HSI_multistate_minus_eta": r2(true[mask, :, lead], multi[mask, :, lead]) - r2(true[mask, :, lead], eta[mask, :, lead]),
                })
    return model_rows, lead_rows, station_rows, season_rows


def plot_protocol(summary: pd.DataFrame, protocol: str, out: Path):
    sub = summary[summary["protocol"] == protocol]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].plot(sub["lead"], sub["HSI_multistate_minus_eta_mean"], color="#2A6FBB", marker="o", markersize=3)
    axes[0].fill_between(sub["lead"], sub["HSI_multistate_minus_eta_mean"] - sub["HSI_multistate_minus_eta_std"], sub["HSI_multistate_minus_eta_mean"] + sub["HSI_multistate_minus_eta_std"], color="#2A6FBB", alpha=0.15)
    axes[0].set_title("Horizon specialization index")
    axes[0].set_ylabel("R2(multistate) - R2(eta-only)")
    axes[1].plot(sub["lead"], sub["error_correlation_mean"], color="#C44E52", marker="o", markersize=3)
    axes[1].set_ylim(0, 1.02); axes[1].set_title("Expert error correlation")
    axes[2].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axes[2].plot(sub["lead"], sub["eta_win_probability_mean"], color="#388E3C", marker="o", markersize=3, label="Eta-only wins")
    axes[2].plot(sub["lead"], sub["multistate_win_probability_mean"], color="#D98C10", marker="o", markersize=3, label="Multistate wins")
    axes[2].set_ylim(0.35, 0.65); axes[2].set_title("Per-element expert win probability"); axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Forecast lead (h)"); ax.set_xlim(1, 24); ax.grid(alpha=0.2)
    fig.suptitle(f"Supervision-associated specialization and complementarity: {protocol}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / f"specialization_{protocol}.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    model_rows, lead_rows, station_rows, season_rows = [], [], [], []
    for protocol, loader in (("retrospective", load_retro), ("strict_2025_h2", load_strict)):
        for seed in args.seeds:
            parts = analyze_bundle(protocol, seed, loader(seed))
            model_rows.extend(parts[0]); lead_rows.extend(parts[1]); station_rows.extend(parts[2]); season_rows.extend(parts[3])
    models = pd.DataFrame(model_rows); leads = pd.DataFrame(lead_rows); stations = pd.DataFrame(station_rows); seasons = pd.DataFrame(season_rows)
    models.to_csv(out / "model_metrics_by_seed.csv", index=False)
    leads.to_csv(out / "specialization_by_seed_lead.csv", index=False)
    stations.to_csv(out / "complementarity_by_seed_station_lead.csv", index=False)
    seasons.to_csv(out / "complementarity_by_seed_season_lead.csv", index=False)
    lead_summary = leads.groupby(["protocol", "lead"]).agg({
        "eta_R2": ["mean", "std"], "multistate_R2": ["mean", "std"],
        "HSI_multistate_minus_eta": ["mean", "std"], "error_correlation": ["mean", "std"],
        "eta_win_probability": ["mean", "std"], "multistate_win_probability": ["mean", "std"],
        "mean_abs_expert_disagreement": ["mean", "std"], "hsdt_gain_over_best_expert_R2": ["mean", "std"],
    }).reset_index()
    lead_summary.columns = ["_".join(x for x in col if x) for col in lead_summary.columns.to_flat_index()]
    lead_summary.to_csv(out / "specialization_mean_std.csv", index=False)
    for protocol in lead_summary["protocol"].unique():
        plot_protocol(lead_summary, protocol, out)

    strict_leads = lead_summary[lead_summary["protocol"] == "strict_2025_h2"]
    strict_models = models[models["protocol"] == "strict_2025_h2"].pivot(index="seed", columns="model", values="sequence_R2")
    positive = int((strict_leads["HSI_multistate_minus_eta_mean"] > 0).sum())
    negative = int((strict_leads["HSI_multistate_minus_eta_mean"] < 0).sum())
    eta_gain = strict_models["hs_dt"] - strict_models["eta_only"]
    multi_gain = strict_models["hs_dt"] - strict_models["multistate"]
    supported = bool(
        positive > 0 and negative > 0
        and strict_leads["error_correlation_mean"].mean() < 0.98
        and int((eta_gain > 0).sum()) >= 4
        and int((multi_gain > 0).sum()) >= 4
    )
    decision = {
        "status": "HSI_COMPLEMENTARITY_SUPPORTED" if supported else "HSI_MECHANISM_NOT_CONFIRMED",
        "strict_positive_HSI_leads": positive,
        "strict_negative_HSI_leads": negative,
        "strict_mean_error_correlation": float(strict_leads["error_correlation_mean"].mean()),
        "strict_error_correlation_range": [float(strict_leads["error_correlation_mean"].min()), float(strict_leads["error_correlation_mean"].max())],
        "hsdt_sequence_wins_vs_eta": int((eta_gain > 0).sum()),
        "hsdt_sequence_wins_vs_multistate": int((multi_gain > 0).sum()),
        "hsdt_mean_sequence_gain_vs_eta": float(eta_gain.mean()),
        "hsdt_mean_sequence_gain_vs_multistate": float(multi_gain.mean()),
    }
    (out / "SPECIALIZATION_DECISION.json").write_text(json.dumps(decision, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
