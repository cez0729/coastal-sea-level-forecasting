from __future__ import annotations

"""Publication diagnostics for causal physics-conditioned HS-DT experts."""

import argparse
import copy
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


m140 = load_module("hsdt_expert_physics_141", HERE / "140_hsdt_expert_physics_conditioned.py")

CONFIGS = [
    "hsdt_baseline",
    "hsdt_eta_causal_ode_expert",
    "hsdt_multistate_causal_ode_expert",
    "hsdt_both_causal_ode_experts",
]
COMPARISONS = [
    ("eta_expert_minus_hsdt", "hsdt_eta_causal_ode_expert", "hsdt_baseline"),
    ("multistate_expert_minus_hsdt", "hsdt_multistate_causal_ode_expert", "hsdt_baseline"),
    ("dual_expert_minus_hsdt", "hsdt_both_causal_ode_experts", "hsdt_baseline"),
    ("dual_minus_multistate_expert", "hsdt_both_causal_ode_experts", "hsdt_multistate_causal_ode_expert"),
]


def r2(true, pred):
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    denom = np.sum((true - true.mean()) ** 2)
    return float("nan") if denom <= 0 else 1.0 - float(np.sum((true - pred) ** 2)) / float(denom)


def load_enhanced(kind, base, data, args, seed, device):
    states = 1 if kind == "eta" else 4
    ode = m140.v2.MultistatePhysicsODE(data["nodes"], len(data["physics_cols"]), 4).to(device)
    adjacency = torch.tensor(data["graph_priors"][args.fixed_graph_type], dtype=torch.float32, device=device)
    model = m140.PhysicsConditionedExpert(
        copy.deepcopy(base), ode, adjacency, args.horizon, states, args.initial_gate
    ).to(device)
    path = ROOT / args.output_dir / "formal" / f"seed_{seed}" / f"{kind}_causal_ode_expert" / "best_checkpoint.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
    model.eval()
    return model


def make_bundle(seed, args, device):
    data = m140.rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    eta = m140.load_base(data, args, seed, 1, device)
    multi = m140.load_base(data, args, seed, 4, device)
    eta_phys = load_enhanced("eta", eta, data, args, seed, device)
    multi_phys = load_enhanced("multistate", multi, data, args, seed, device)
    eta_pred, true, tide = m140.score_predictions(eta, data, args, seed, device, "formal", 1)
    multi_pred = m140.score_predictions(multi, data, args, seed, device, "formal", 4)[0]
    eta_phys_pred = m140.score_predictions(eta_phys, data, args, seed, device, "formal", 1)[0]
    multi_phys_pred = m140.score_predictions(multi_phys, data, args, seed, device, "formal", 4)[0]
    weight = np.full(args.horizon, 0.5, dtype=np.float64)
    weight[-1] = 1.0
    predictions = {
        "hsdt_baseline": eta_pred + weight[None, None, :] * (multi_pred - eta_pred),
        "hsdt_eta_causal_ode_expert": eta_phys_pred + weight[None, None, :] * (multi_pred - eta_phys_pred),
        "hsdt_multistate_causal_ode_expert": eta_pred + weight[None, None, :] * (multi_phys_pred - eta_pred),
        "hsdt_both_causal_ode_experts": eta_phys_pred + weight[None, None, :] * (multi_phys_pred - eta_phys_pred),
    }
    if np.max(np.abs(predictions["hsdt_baseline"][..., -1] - multi_pred[..., -1])) > 1e-6:
        raise RuntimeError("HS-DT baseline lead-24 does not equal multistate expert")
    if np.max(np.abs(predictions["hsdt_both_causal_ode_experts"][..., -1] - multi_phys_pred[..., -1])) > 1e-6:
        raise RuntimeError("Dual-expert lead-24 does not equal enhanced multistate expert")
    indices = data["multi_test"].indices
    bundle = {
        "true_residual": true.astype(np.float32),
        "tide": tide.astype(np.float32),
        "target_origin_time": np.asarray(data["arrays"]["time"])[indices].astype("datetime64[ns]"),
        "station_ids": np.asarray(m140.v2.STATION_IDS, dtype="U"),
        **{name: value.astype(np.float32) for name, value in predictions.items()},
    }
    return bundle


def paired_tests(all_runs):
    metrics = ["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2", "event_PR_AUC"]
    pivot = all_runs.pivot(index="seed", columns="config", values=metrics)
    rows = []
    for name, candidate, baseline in COMPARISONS:
        for metric in metrics:
            raw = pivot[(metric, candidate)] - pivot[(metric, baseline)]
            improvement = -raw if metric.endswith(("RMSE", "MAE")) else raw
            rows.append({
                "comparison": name,
                "metric": metric,
                "mean_improvement": float(improvement.mean()),
                "std_improvement": float(improvement.std(ddof=1)),
                "wins": int((improvement > 0).sum()),
                "count": int(improvement.notna().sum()),
                "wilcoxon_greater_p": m140.p104.exact_wilcoxon_greater(improvement.to_numpy()),
            })
    return pd.DataFrame(rows)


def diagnostics(bundles, seeds, output_dir):
    lead_rows, station_rows = [], []
    stations = bundles[0]["station_ids"].tolist()
    for seed, bundle in zip(seeds, bundles):
        true = bundle["true_residual"].astype(np.float64)
        for config in CONFIGS:
            pred = bundle[config].astype(np.float64)
            for lead in range(true.shape[-1]):
                error = pred[..., lead] - true[..., lead]
                lead_rows.append({"seed": seed, "config": config, "lead_hour": lead + 1,
                                  "residual_R2": r2(true[..., lead], pred[..., lead]),
                                  "residual_RMSE": float(np.sqrt(np.mean(error ** 2)))})
            for index, station in enumerate(stations):
                error = pred[:, index, -1] - true[:, index, -1]
                station_rows.append({"seed": seed, "config": config, "station_id": station,
                                     "sequence_residual_R2": r2(true[:, index, :], pred[:, index, :]),
                                     "lead24_residual_R2": r2(true[:, index, -1], pred[:, index, -1]),
                                     "lead24_residual_RMSE": float(np.sqrt(np.mean(error ** 2)))})
    leads = pd.DataFrame(lead_rows)
    stations = pd.DataFrame(station_rows)
    leads.to_csv(output_dir / "per_lead_by_seed.csv", index=False)
    stations.to_csv(output_dir / "per_station_by_seed.csv", index=False)
    leads.groupby(["config", "lead_hour"])[["residual_R2", "residual_RMSE"]].agg(["mean", "std", "count"]).to_csv(output_dir / "per_lead_mean_std.csv")
    stations.groupby(["config", "station_id"])[["sequence_residual_R2", "lead24_residual_R2", "lead24_residual_RMSE"]].agg(["mean", "std", "count"]).to_csv(output_dir / "per_station_mean_std.csv")
    return leads, stations


def moving_blocks(rng, count, block):
    block = min(max(1, block), count)
    starts = rng.integers(0, count - block + 1, size=int(np.ceil(count / block)))
    return np.concatenate([np.arange(start, start + block) for start in starts])[:count]


def bootstrap(bundles, args):
    rng = np.random.default_rng(20260731)
    rows = []
    for metric, horizon in (("sequence_R2", slice(None)), ("lead24_R2", -1)):
        true = bundles[0]["true_residual"][..., horizon].astype(np.float64)
        count = true.shape[0]
        flat = true.reshape(count, -1)
        sums, sums2, width = flat.sum(1), (flat ** 2).sum(1), flat.shape[1]
        for name, candidate, baseline in COMPARISONS:
            advantages = []
            for bundle in bundles:
                ce = (bundle[candidate][..., horizon].astype(np.float64) - true).reshape(count, -1)
                be = (bundle[baseline][..., horizon].astype(np.float64) - true).reshape(count, -1)
                advantages.append((be ** 2).sum(1) - (ce ** 2).sum(1))
            advantages = np.stack(advantages)
            full_sst = sums2.sum() - sums.sum() ** 2 / (count * width)
            point = advantages.sum(1) / full_sst
            reps = np.empty(args.bootstrap_replicates)
            for index in range(args.bootstrap_replicates):
                sample = moving_blocks(rng, count, args.bootstrap_block_hours)
                sampled_seeds = rng.integers(0, len(bundles), size=len(bundles))
                sst = sums2[sample].sum() - sums[sample].sum() ** 2 / (len(sample) * width)
                reps[index] = np.mean(advantages[sampled_seeds][:, sample].sum(1) / sst)
            rows.append({"comparison": name, "metric": metric, "point_mean_delta": float(point.mean()),
                         "ci95_low": float(np.quantile(reps, 0.025)), "ci95_high": float(np.quantile(reps, 0.975)),
                         "probability_positive": float(np.mean(reps > 0)), "block_hours": args.bootstrap_block_hours,
                         "replicates": args.bootstrap_replicates})
    return pd.DataFrame(rows)


def make_figures(all_runs, leads, output_dir):
    labels = {"hsdt_baseline": "HS-DT", "hsdt_eta_causal_ode_expert": "Eta expert + prior",
              "hsdt_multistate_causal_ode_expert": "Multistate expert + prior",
              "hsdt_both_causal_ode_experts": "Both experts + prior",
              "hsdt_no_prior_finetune_control": "No-prior fine-tune"}
    plot_configs = [config for config in CONFIGS + ["hsdt_no_prior_finetune_control"] if config in set(all_runs["config"])]
    metrics = [("seq_residual_R2", "Sequence R2"), ("last_residual_R2", "Lead-24 R2"),
               ("extreme_abs_q95_residual_R2", "Descriptive q95 R2")]
    summary = all_runs.groupby("config").agg({m: ["mean", "std"] for m, _ in metrics})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    colors = ["#607D8B", "#2E86AB", "#4C956C", "#D97706"]
    for ax, (metric, title) in zip(axes, metrics):
        means = [summary.loc[c, (metric, "mean")] for c in plot_configs]
        stds = [summary.loc[c, (metric, "std")] for c in plot_configs]
        ax.bar(range(len(plot_configs)), means, yerr=stds, capsize=3, color=(colors + ["#B56576"])[:len(plot_configs)])
        ax.set_xticks(range(len(plot_configs)), [labels[c] for c in plot_configs], rotation=22, ha="right")
        ax.set_title(title); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "physics_injection_main_comparison.png", dpi=260)
    plt.close(fig)

    mean_lead = leads.groupby(["config", "lead_hour"])["residual_R2"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    lead_configs = [c for c in plot_configs if c != "hsdt_baseline" and c in mean_lead.columns]
    for config, color in zip(lead_configs, (colors + ["#B56576"])[1:]):
        ax.plot(mean_lead.index, mean_lead[config] - mean_lead["hsdt_baseline"], label=labels[config], color=color)
    ax.axhline(0, color="#333333", lw=1); ax.set_xlabel("Forecast lead (h)"); ax.set_ylabel("R2 difference vs HS-DT")
    ax.legend(frameon=False); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output_dir / "physics_injection_per_lead_delta.png", dpi=260); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/hsdt_expert_physics_conditioned")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 2025, 3407])
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-block-hours", type=int, default=168)
    args, _ = parser.parse_known_args()
    base = m140.parse_args()
    for key, value in vars(base).items():
        if not hasattr(args, key): setattr(args, key, value)
    output = ROOT / args.output_dir / "formal_analysis"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundles = []
    for seed in args.seeds:
        bundle = make_bundle(seed, args, device); bundles.append(bundle)
        seed_dir = output / f"seed_{seed}"; seed_dir.mkdir(exist_ok=True)
        np.savez_compressed(seed_dir / "predictions.npz", **bundle)
    all_runs = pd.read_csv(ROOT / args.output_dir / "formal" / "all_runs.csv")
    control_path = ROOT / "results" / "hsdt_no_prior_finetune_control" / "formal" / "all_runs.csv"
    if control_path.exists():
        control = pd.read_csv(control_path)
        control = control[control["config"] == "hsdt_no_prior_finetune_control"]
        all_runs = pd.concat([all_runs, control], ignore_index=True)
    all_runs.groupby("config")[["seq_residual_R2", "last_residual_R2", "last_residual_RMSE", "extreme_abs_q95_residual_R2", "event_PR_AUC"]].agg(["mean", "std", "count"]).to_csv(output / "mean_std.csv")
    paired_tests(all_runs).to_csv(output / "paired_tests.csv", index=False)
    leads, _ = diagnostics(bundles, args.seeds, output)
    bootstrap(bundles, args).to_csv(output / "block_bootstrap.csv", index=False)
    make_figures(all_runs, leads, output)
    (output / "EVIDENCE_STATUS_CN.md").write_text(
        "本目录仅使用严格因果预处理和2023--2024训练、2025 H1选择、2025 H2回测。2025 H2曾在开发中查看，因此是chronological refit backtest，不是untouched holdout。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
