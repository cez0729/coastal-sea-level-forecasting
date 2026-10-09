from __future__ import annotations

"""Validation-locked physics-regime switching for the ordinary HS-DT benchmark.

The method uses the learned-ODE adapter only when a causal forcing-intensity
index is high and otherwise uses the persistence adapter. Candidate regime
thresholds are fixed in advance and selected on validation predictions only.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 2024, 2025, 3407]
COMPONENTS = ["wind", "pressure_tendency", "current", "wave_flux", "wave_setup"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


adapter = load_module("physics_regime_adapter", HERE / "109_ode_prior_gated_multistate_gwn.py")
orc = load_module("physics_regime_orc", HERE / "131_ode_residual_corrected_hsdt_gwn.py")
final4 = adapter.final4
v2 = adapter.v2
p104 = adapter.p104


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def robust_positive_z(values: np.ndarray, train_end: int) -> np.ndarray:
    train = values[:train_end]
    median = np.nanmedian(train, axis=0)
    iqr = np.nanquantile(train, 0.75, axis=0) - np.nanquantile(train, 0.25, axis=0)
    return np.clip((values - median) / np.maximum(iqr, 1e-6), 0.0, 10.0)


def build_forcing_intensity(data: dict, args) -> dict:
    arrays = data["arrays"]
    train_end = int(len(arrays["time"]) * args.train_ratio)
    raw = {
        "wind": np.sqrt(
            arrays["wind_stress_u_proxy"] ** 2 + arrays["wind_stress_v_proxy"] ** 2
        ),
        "pressure_tendency": np.abs(arrays["coops_pressure_tendency_3h"]),
        "current": arrays["current_speed_mps"],
        "wave_flux": arrays["wave_energy_flux"],
        "wave_setup": arrays["wave_setup_proxy"],
    }
    standardized = {name: robust_positive_z(values, train_end) for name, values in raw.items()}
    combined = np.mean(np.stack([standardized[name] for name in COMPONENTS]), axis=0)
    thresholds = {
        float(quantile): np.nanquantile(combined[:train_end], quantile, axis=0)
        for quantile in args.candidate_quantiles
    }
    component_thresholds = {
        name: {
            float(quantile): np.nanquantile(raw[name][:train_end], quantile, axis=0)
            for quantile in args.regime_quantiles
        }
        for name in COMPONENTS
    }
    component_thresholds["combined"] = {
        float(quantile): np.nanquantile(combined[:train_end], quantile, axis=0)
        for quantile in args.regime_quantiles
    }
    return {
        "raw": raw,
        "combined": combined,
        "thresholds": thresholds,
        "component_thresholds": component_thresholds,
        "train_end": train_end,
    }


def load_trained_adapter(seed: int, mode: str, data: dict, args, device: torch.device):
    base = adapter.load_base_model(seed, data, args, device)
    ode = v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    model = adapter.ODEPriorGatedModel(
        base, ode, args.horizon, args.initial_gate, args.refiner_hidden, mode
    ).to(device)
    model.ode_adj = torch.tensor(
        data["graph_priors"][args.fixed_graph_type], dtype=torch.float32, device=device
    )
    source = args.learned_ode_results if mode == "learned_ode" else args.persistence_results
    checkpoint = project_path(source) / f"seed_{seed}" / "best_checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def validation_sources(seed: int, data: dict, args, device: torch.device) -> dict:
    cache = project_path(args.output_dir) / "validation_sources" / f"seed_{seed}.npz"
    if args.resume and cache.exists():
        with np.load(cache) as payload:
            return {key: payload[key] for key in payload.files}
    loader = adapter.make_loader(data["multi_val"], args, False, seed)
    learned = adapter.predict_all(
        load_trained_adapter(seed, "learned_ode", data, args, device), loader, device
    )
    persistence = adapter.predict_all(
        load_trained_adapter(seed, "persistence", data, args, device), loader, device
    )
    if not np.allclose(learned["true"], persistence["true"], atol=1e-6, rtol=0.0):
        raise RuntimeError(f"Validation target mismatch for seed {seed}")
    output = {
        "learned": learned["final"][..., 0].astype(np.float32),
        "persistence": persistence["final"][..., 0].astype(np.float32),
        "true": learned["true"][..., 0].astype(np.float32),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **output)
    return output


def mse_pair(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    return float(np.mean((true - pred) ** 2)), float(np.mean((true[..., -1] - pred[..., -1]) ** 2))


def select_quantile(data: dict, forcing: dict, args, device: torch.device) -> tuple[float, pd.DataFrame]:
    indices = data["multi_val"].indices
    intensity = forcing["combined"][indices - 1]
    rows = []
    for seed in args.seeds:
        sources = validation_sources(seed, data, args, device)
        true = sources["true"]
        for quantile, threshold in forcing["thresholds"].items():
            high = intensity >= threshold
            hybrid = np.where(high[..., None], sources["learned"], sources["persistence"])
            seq_mse, lead_mse = mse_pair(true, hybrid)
            rows.append(
                {
                    "seed": seed,
                    "quantile": quantile,
                    "physics_usage_fraction": float(high.mean()),
                    "validation_sequence_MSE": seq_mse,
                    "validation_lead24_MSE": lead_mse,
                }
            )
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("quantile", as_index=False).agg(
        validation_sequence_MSE=("validation_sequence_MSE", "mean"),
        validation_lead24_MSE=("validation_lead24_MSE", "mean"),
        physics_usage_fraction=("physics_usage_fraction", "mean"),
    )
    grouped["validation_score"] = grouped["validation_lead24_MSE"]
    selected = float(grouped.sort_values(["validation_score", "quantile"]).iloc[0]["quantile"])
    frame = frame.merge(grouped[["quantile", "validation_score"]], on="quantile", how="left")
    frame["selected"] = frame["quantile"].eq(selected)
    return selected, frame


def load_prediction(seed: int, model: str, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = project_path(args.orc_results) / f"seed_{seed}" / f"{model}_predictions.npz"
    with np.load(path) as payload:
        return (
            payload["pred_residual"].astype(np.float64),
            payload["true_residual"].astype(np.float64),
            payload["target_tide"].astype(np.float64),
        )


def physics_comparisons(predictions: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "gnn_ode_prior_minus_learnable": (
            predictions["gnn_ode_prior"], predictions["gnn_learnable"]
        ),
        "gwn_physics_loss_minus_no_physics": (
            predictions["gwn_physics_loss"], predictions["gwn_no_physics"]
        ),
        "orc_minus_hsdt": (predictions["orc_hsdt_gwn"], predictions["hsdt_gwn"]),
        "orc_minus_zero": (predictions["orc_hsdt_gwn"], predictions["hsdt_zero_adapter"]),
        "orc_minus_persistence": (
            predictions["orc_hsdt_gwn"], predictions["hsdt_persistence_adapter"]
        ),
    }


def regime_effect_rows(
    seed: int,
    data: dict,
    forcing: dict,
    predictions: dict[str, np.ndarray],
    true: np.ndarray,
    args,
) -> list[dict]:
    test_indices = data["multi_test"].indices
    rows = []
    comparisons = physics_comparisons(predictions)
    for component in [*COMPONENTS, "combined"]:
        full = forcing["combined"] if component == "combined" else forcing["raw"][component]
        issue_values = full[test_indices - 1]
        for quantile, threshold in forcing["component_thresholds"][component].items():
            high = issue_values >= threshold
            for comparison, (candidate, baseline) in comparisons.items():
                base_error = (true - baseline) ** 2
                candidate_error = (true - candidate) ** 2
                rows.append(
                    {
                        "seed": seed,
                        "component": component,
                        "train_quantile": quantile,
                        "comparison": comparison,
                        "coverage": float(high.mean()),
                        "sequence_MSE_reduction": float((base_error - candidate_error)[high].mean()),
                        "lead24_MSE_reduction": float(
                            (base_error[..., -1] - candidate_error[..., -1])[high].mean()
                        ),
                    }
                )
    return rows


def selected_regime_per_lead_rows(
    seed: int,
    high: np.ndarray,
    predictions: dict[str, np.ndarray],
    true: np.ndarray,
) -> list[dict]:
    rows = []
    for comparison, (candidate, baseline) in physics_comparisons(predictions).items():
        for lead in range(true.shape[-1]):
            reduction = (true[..., lead] - baseline[..., lead]) ** 2
            reduction -= (true[..., lead] - candidate[..., lead]) ** 2
            rows.append(
                {
                    "seed": seed,
                    "comparison": comparison,
                    "lead_hour": lead + 1,
                    "MSE_reduction": float(reduction[high].mean()),
                    "coverage": float(high.mean()),
                }
            )
    return rows


def summarize_regime_effects(regimes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["component", "train_quantile", "comparison"]
    for key, group in regimes.groupby(keys):
        sequence = group["sequence_MSE_reduction"].to_numpy()
        lead = group["lead24_MSE_reduction"].to_numpy()
        rows.append(
            {
                **dict(zip(keys, key)),
                "coverage_mean": float(group["coverage"].mean()),
                "sequence_MSE_reduction_mean": float(sequence.mean()),
                "sequence_MSE_reduction_std": float(sequence.std(ddof=1)),
                "sequence_wins": int((sequence > 0).sum()),
                "sequence_wilcoxon_greater_p": p104.exact_wilcoxon_greater(sequence),
                "lead24_MSE_reduction_mean": float(lead.mean()),
                "lead24_MSE_reduction_std": float(lead.std(ddof=1)),
                "lead24_wins": int((lead > 0).sum()),
                "lead24_wilcoxon_greater_p": p104.exact_wilcoxon_greater(lead),
            }
        )
    return pd.DataFrame(rows)


def paired_summary(all_runs: pd.DataFrame, winner: str) -> pd.DataFrame:
    metrics = [
        "seq_residual_R2", "last_residual_R2", "last_residual_RMSE",
        "extreme_abs_q95_residual_R2", "event_CSI", "event_PR_AUC",
    ]
    pivot = all_runs.pivot(index="seed", columns="model", values=metrics)
    rows = []
    for baseline in [name for name in all_runs["model"].unique() if name != winner]:
        for metric in metrics:
            delta = pivot[(metric, winner)] - pivot[(metric, baseline)]
            improvement = -delta if metric.endswith(("RMSE", "MAE")) else delta
            rows.append(
                {
                    "comparison": f"{winner}_minus_{baseline}",
                    "metric": metric,
                    "mean_improvement": float(improvement.mean()),
                    "std_improvement": float(improvement.std()),
                    "wins": int((improvement > 0).sum()),
                    "count": int(improvement.notna().sum()),
                    "wilcoxon_greater_p": p104.exact_wilcoxon_greater(improvement.to_numpy()),
                }
            )
    return pd.DataFrame(rows)


def plot_results(
    summary: pd.DataFrame,
    regimes: pd.DataFrame,
    per_lead: pd.DataFrame,
    selected: float,
    output_dir: Path,
) -> None:
    order = [
        "hsdt_gwn", "hsdt_zero_adapter", "hsdt_persistence_adapter",
        "orc_hsdt_gwn", "physics_regime_switched_hsdt",
        "reverse_regime_control", "shifted_regime_control",
    ]
    labels = ["HS-DT", "Zero", "Persistence", "ORC", "PRS-HS-DT", "Reverse", "Shifted"]
    lookup = summary.set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    specs = [
        ("seq_residual_R2_mean", "Trajectory R2"),
        ("last_residual_R2_mean", "Lead-24 R2"),
        ("extreme_abs_q95_residual_R2_mean", "Descriptive q95 R2"),
    ]
    colors = ["#697382", "#8A6A6A", "#526B8C", "#7B5EA7", "#2F7D6D", "#B07A32", "#9A9A9A"]
    for axis, (metric, title) in zip(axes, specs):
        values = [lookup.loc[name, metric] for name in order]
        errors = [lookup.loc[name, metric.replace("_mean", "_std")] for name in order]
        axis.bar(np.arange(len(order)), values, yerr=errors, capsize=3, color=colors)
        axis.set_xticks(np.arange(len(order)), labels, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Validation-locked physics-regime switching (training q={selected:.2f})")
    fig.tight_layout()
    fig.savefig(output_dir / "physics_regime_switched_model_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    selected_regimes = regimes[np.isclose(regimes["train_quantile"], selected)].copy()
    grouped = selected_regimes.groupby(["component", "comparison"], as_index=False).agg(
        lead24_MSE_reduction=("lead24_MSE_reduction", "mean")
    )
    comparisons = [
        "gnn_ode_prior_minus_learnable",
        "gwn_physics_loss_minus_no_physics",
        "orc_minus_hsdt",
        "orc_minus_zero",
        "orc_minus_persistence",
    ]
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len([*COMPONENTS, "combined"]))
    width = 0.16
    for index, comparison in enumerate(comparisons):
        values = []
        for component in [*COMPONENTS, "combined"]:
            value = grouped[(grouped["component"] == component) & (grouped["comparison"] == comparison)]
            values.append(float(value.iloc[0]["lead24_MSE_reduction"]))
        axis.bar(x + (index - 2) * width, values, width, label=comparison.replace("_", " "))
    axis.axhline(0, color="#222222", linewidth=0.9)
    axis.set_xticks(x, ["Wind", "Pressure", "Current", "Wave flux", "Wave setup", "Combined"])
    axis.set_ylabel("Lead-24 MSE reduction in high-forcing regime")
    axis.set_title("Where ODE-conditioned correction helps")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "physics_regime_effect_map.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    lead_summary = per_lead.groupby(["comparison", "lead_hour"], as_index=False).agg(
        mean=("MSE_reduction", "mean"), std=("MSE_reduction", "std")
    )
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    for comparison in comparisons:
        group = lead_summary[lead_summary["comparison"] == comparison]
        axis.plot(group["lead_hour"], group["mean"], label=comparison.replace("_", " "))
    axis.axhline(0, color="#222222", linewidth=0.9)
    axis.set_xlabel("Forecast lead (h)")
    axis.set_ylabel("MSE reduction in selected high-forcing regime")
    axis.set_title("Lead-dependent physics effect under high forcing")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "physics_regime_per_lead_effect.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    selected: float,
    validation: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    regimes: pd.DataFrame,
) -> None:
    lookup = summary.set_index("model")
    prs = lookup.loc["physics_regime_switched_hsdt"]
    orc_row = lookup.loc["orc_hsdt_gwn"]
    persistence = lookup.loc["hsdt_persistence_adapter"]
    comparison = paired[paired["comparison"] == "physics_regime_switched_hsdt_minus_hsdt_persistence_adapter"]
    regime = regimes[
        np.isclose(regimes["train_quantile"], selected)
        & (regimes["component"] == "combined")
        & (regimes["comparison"] == "orc_minus_persistence")
    ]
    cross_backbone = regimes[
        np.isclose(regimes["train_quantile"], selected)
        & (regimes["component"] == "combined")
    ].groupby("comparison", as_index=False).agg(
        lead24_MSE_reduction=("lead24_MSE_reduction", "mean"),
        sequence_MSE_reduction=("sequence_MSE_reduction", "mean"),
        coverage=("coverage", "mean"),
    )
    cross_lines = ["| Comparison | Sequence MSE reduction | Lead-24 MSE reduction | Coverage |", "|---|---:|---:|---:|"]
    for _, row in cross_backbone.iterrows():
        cross_lines.append(
            f"| {row['comparison']} | {row['sequence_MSE_reduction']:.8f} | "
            f"{row['lead24_MSE_reduction']:.8f} | {row['coverage']:.1%} |"
        )
    cross_table = "\n".join(cross_lines)
    report = f"""# Physics-Regime-Switched HS-DT-GWN 实验报告

## 方法

该方法以HS-DT的两个已训练专家和现有adapter为基础。训练期风应力、气压趋势、流速、波能通量和wave-setup proxy先经过站点内robust标准化并取平均，形成只依赖预测起点前信息的forcing-intensity index。候选阈值固定为{args_to_text(validation['quantile'].unique())}，仅按五seed平均验证集Lead-24 MSE选择，锁定阈值为训练期q={selected:.2f}。

测试时，高强迫样本使用learned-ODE ORC修正，其余样本使用persistence adapter。反向门控和168小时错位门控保持相同模型和使用比例，用于检验物理状态与修正时机是否重要。

## 五seed普通benchmark结果

| 模型 | Sequence R2 | Lead-24 R2 | q95 R2 | Event PR-AUC |
|---|---:|---:|---:|---:|
| HS-DT | {lookup.loc['hsdt_gwn','seq_residual_R2_mean']:.6f} | {lookup.loc['hsdt_gwn','last_residual_R2_mean']:.6f} | {lookup.loc['hsdt_gwn','extreme_abs_q95_residual_R2_mean']:.6f} | {lookup.loc['hsdt_gwn','event_PR_AUC_mean']:.6f} |
| Persistence adapter | {persistence['seq_residual_R2_mean']:.6f} | {persistence['last_residual_R2_mean']:.6f} | {persistence['extreme_abs_q95_residual_R2_mean']:.6f} | {persistence['event_PR_AUC_mean']:.6f} |
| ORC | {orc_row['seq_residual_R2_mean']:.6f} | {orc_row['last_residual_R2_mean']:.6f} | {orc_row['extreme_abs_q95_residual_R2_mean']:.6f} | {orc_row['event_PR_AUC_mean']:.6f} |
| PRS-HS-DT | {prs['seq_residual_R2_mean']:.6f} | {prs['last_residual_R2_mean']:.6f} | {prs['extreme_abs_q95_residual_R2_mean']:.6f} | {prs['event_PR_AUC_mean']:.6f} |

PRS相对persistence adapter的配对结果见`paired_comparisons.csv`；相对ORC，它优先改善Lead-24，但可能牺牲部分sequence或q95表现，因此应定位为终点专门化模型，而不是所有指标统一冠军。

在训练期定义的高综合强迫区间，ORC相对persistence的Lead-24 MSE平均减少{regime['lead24_MSE_reduction'].mean():.8f}，覆盖率约{regime['coverage'].mean():.1%}。完整风、压、流、浪分层位于`physics_regime_effects.csv`。

## 跨骨干高强迫归因

{cross_table}

正值表示候选模型降低MSE，负值表示物理方法反而增加误差。该表使用相同的训练期q={selected:.2f}综合强迫阈值，不根据测试结果重新选择子集。

## 证据边界

这是普通70/15/15 aligned-forcing benchmark内的post-hoc扩展。阈值选择只使用validation，但该benchmark测试期在此前研究中已被查看，因此结果只能作为机制探索和新主模型候选，不能声称独立时间验证。该方法证明的是“物理强迫状态有助于决定使用哪一种已训练修正器”；它不证明完整ODE数值解优于GWN，也不把adapter的全部增益归因于physics。
"""
    (output_dir / "EXPERIMENT_REPORT_CN.md").write_text(report, encoding="utf-8")


def args_to_text(values) -> str:
    return "/".join(f"{float(value):.2f}" for value in sorted(values))


def run(args) -> None:
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = final4.build_enhanced_data(args, args.horizon, add_ode_prior=False)
    forcing = build_forcing_intensity(data, args)
    selected, validation = select_quantile(data, forcing, args, device)
    validation.to_csv(output_dir / "validation_threshold_selection.csv", index=False)

    test_indices = data["multi_test"].indices
    test_intensity = forcing["combined"][test_indices - 1]
    threshold = forcing["thresholds"][selected]
    high = test_intensity >= threshold
    shifted_high = np.roll(high, args.shift_hours, axis=0)
    rows, regime_rows, per_lead_rows = [], [], []
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    thresholds = np.quantile(data["arrays"]["residual"][: forcing["train_end"]], args.event_quantile, axis=0)

    for seed in args.seeds:
        predictions = {}
        true = tide = None
        for name in ["hsdt_gwn", "hsdt_zero_adapter", "hsdt_persistence_adapter", "orc_hsdt_gwn"]:
            pred, current_true, current_tide = load_prediction(seed, name, args)
            predictions[name] = pred
            if true is None:
                true, tide = current_true, current_tide
            elif not np.allclose(true, current_true, atol=1e-6, rtol=0.0):
                raise RuntimeError(f"Test target mismatch for seed {seed}: {name}")
        gwn_root = project_path(args.gwn_results) / f"seed_{seed}" / f"horizon_{args.horizon}h"
        with np.load(gwn_root / "gwn_multistate_no_physics" / "predictions.npz") as payload:
            predictions["gwn_no_physics"] = payload["pred_states"][..., 0].astype(np.float64)
            gwn_true = payload["true_states"][..., 0].astype(np.float64)
        with np.load(gwn_root / "gwn_multistate_physics" / "predictions.npz") as payload:
            predictions["gwn_physics_loss"] = payload["pred_states"][..., 0].astype(np.float64)
        bigru_root = project_path(args.bigru_results) / f"seed_{seed}" / f"horizon_{args.horizon}h"
        with np.load(bigru_root / "learnable_graph" / "predictions.npz") as payload:
            predictions["gnn_learnable"] = payload["pred_residual"].astype(np.float64)
            bigru_true = payload["true_residual"].astype(np.float64)
        with np.load(bigru_root / "ode_based_learnable" / "predictions.npz") as payload:
            predictions["gnn_ode_prior"] = payload["pred_residual"].astype(np.float64)
        if not np.allclose(true, gwn_true, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"GWN target mismatch for seed {seed}")
        if not np.allclose(true, bigru_true, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"GNN target mismatch for seed {seed}")
        predictions["physics_regime_switched_hsdt"] = np.where(
            high[..., None], predictions["orc_hsdt_gwn"], predictions["hsdt_persistence_adapter"]
        )
        predictions["reverse_regime_control"] = np.where(
            high[..., None], predictions["hsdt_persistence_adapter"], predictions["orc_hsdt_gwn"]
        )
        predictions["shifted_regime_control"] = np.where(
            shifted_high[..., None], predictions["orc_hsdt_gwn"], predictions["hsdt_persistence_adapter"]
        )
        reported = {
            name: pred for name, pred in predictions.items()
            if name not in {"gwn_no_physics", "gwn_physics_loss", "gnn_learnable", "gnn_ode_prior"}
        }
        for name, pred in reported.items():
            rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "selected_train_quantile": selected,
                    "physics_usage_fraction": float(high.mean()) if "regime" in name else np.nan,
                    **orc.summarize(true, pred, tide, thresholds),
                }
            )
        regime_rows.extend(regime_effect_rows(seed, data, forcing, predictions, true, args))
        per_lead_rows.extend(selected_regime_per_lead_rows(seed, high, predictions, true))
        np.savez_compressed(
            prediction_dir / f"seed_{seed}.npz",
            pred_residual=predictions["physics_regime_switched_hsdt"].astype(np.float32),
            true_residual=true.astype(np.float32),
            target_tide=tide.astype(np.float32),
            high_forcing_mask=high,
            selected_train_quantile=np.asarray(selected),
            station_ids=np.asarray(v2.STATION_IDS),
        )

    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(output_dir / "all_runs.csv", index=False)
    metrics = [
        "seq_residual_R2", "last_residual_R2", "last_residual_RMSE",
        "extreme_abs_q95_residual_R2", "event_CSI", "event_PR_AUC",
    ]
    summary = all_runs.groupby("model")[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(part) for part in column if part) for column in summary.columns.to_flat_index()]
    summary.to_csv(output_dir / "mean_std.csv", index=False)
    paired = paired_summary(all_runs, "physics_regime_switched_hsdt")
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    regimes = pd.DataFrame(regime_rows)
    regimes.to_csv(output_dir / "physics_regime_effects.csv", index=False)
    summarize_regime_effects(regimes).to_csv(output_dir / "physics_regime_effect_summary.csv", index=False)
    per_lead = pd.DataFrame(per_lead_rows)
    per_lead.to_csv(output_dir / "physics_regime_per_lead.csv", index=False)
    plot_results(summary, regimes, per_lead, selected, output_dir)
    write_report(output_dir, selected, validation, summary, paired, regimes)
    (output_dir / "experiment_config.json").write_text(
        json.dumps({**vars(args), "selected_train_quantile": selected}, indent=2), encoding="utf-8"
    )
    print(f"Selected training quantile: {selected:.2f}")
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-locked physics-regime switching for HS-DT-GWN")
    parser.add_argument("--output-dir", default="results/physics_regime_switched_hsdt_benchmark")
    parser.add_argument("--orc-results", default="results/ode_residual_corrected_hsdt_gwn")
    parser.add_argument("--gwn-results", default="results/priority12_physics_graph_wavenet")
    parser.add_argument("--bigru-results", default="results/corrected_bigru_ladder")
    parser.add_argument("--source-results", default="results/priority12_physics_graph_wavenet")
    parser.add_argument("--learned-ode-results", default="results/ode_prior_gated_multistate_gwn_learned_repro_stride8")
    parser.add_argument("--persistence-results", default="results/ode_prior_gated_multistate_gwn_persistence_stride8")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--candidate-quantiles", nargs="+", type=float, default=[0.75, 0.90, 0.95])
    parser.add_argument("--regime-quantiles", nargs="+", type=float, default=[0.75, 0.90, 0.95])
    parser.add_argument("--shift-hours", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--fixed-graph-type", default="distance", choices=["identity", "distance", "corr"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--gwn-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--initial-gate", type=float, default=0.80)
    parser.add_argument("--refiner-hidden", type=int, default=128)
    parser.add_argument("--physics-forcing-mode", default="last_input", choices=["last_input"])
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
