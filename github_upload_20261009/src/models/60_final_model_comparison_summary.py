import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Utilities
# ============================================================
def find_file(outputs_root: Path, exact_relative_paths, glob_patterns):
    """
    Try exact paths first, then glob patterns.
    Return Path or None.
    """

    for rel in exact_relative_paths:
        p = outputs_root / rel
        if p.exists():
            return p

    candidates = []

    for pattern in glob_patterns:
        candidates.extend(outputs_root.glob(pattern))

    candidates = [p for p in candidates if p.exists() and p.is_file()]

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: len(str(x)))
    return candidates[0]


def safe_get(row, col, default=np.nan):
    if col in row.index:
        return row[col]
    return default


def safe_col(df, col, default=np.nan):
    if col in df.columns:
        return df[col]
    return default


def percent_improvement(old, new):
    """
    Positive means new is better.
    """

    if pd.isna(old) or pd.isna(new):
        return np.nan

    if abs(old) < 1e-12:
        return np.nan

    return (old - new) / old * 100.0


def make_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# Normalization functions
# ============================================================
def normalize_standard_metrics(
    df,
    model_family,
    model_name,
    source_name,
    source_path,
):
    """
    For CSVs that already contain:
    seq_residual_RMSE
    seq_residual_R2
    last_residual_RMSE
    last_residual_R2
    """

    rows = []

    for _, r in df.iterrows():
        row = {
            "model_family": model_family,
            "model_name": model_name,
            "source_name": source_name,
            "source_path": str(source_path),

            "horizon": safe_get(r, "horizon"),
            "target_mode": safe_get(r, "target_mode", "multi"),
            "feature_group": safe_get(r, "feature_group", ""),
            "graph_type": safe_get(r, "graph_type", ""),
            "window": safe_get(r, "window", np.nan),
            "num_features": safe_get(r, "num_features", np.nan),

            "learned_w_identity": safe_get(r, "learned_w_identity", np.nan),
            "learned_w_distance": safe_get(r, "learned_w_distance", np.nan),
            "learned_w_corr": safe_get(r, "learned_w_corr", np.nan),

            "seq_residual_RMSE": safe_get(r, "seq_residual_RMSE", np.nan),
            "seq_residual_R2": safe_get(r, "seq_residual_R2", np.nan),
            "last_residual_RMSE": safe_get(r, "last_residual_RMSE", np.nan),
            "last_residual_R2": safe_get(r, "last_residual_R2", np.nan),

            "seq_sea_level_RMSE": safe_get(r, "seq_sea_level_RMSE", np.nan),
            "seq_sea_level_R2": safe_get(r, "seq_sea_level_R2", np.nan),
            "last_sea_level_RMSE": safe_get(r, "last_sea_level_RMSE", np.nan),
            "last_sea_level_R2": safe_get(r, "last_sea_level_R2", np.nan),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def normalize_physics_ode(df, source_path):
    """
    For 58_physics_multi_ode_baselines.py output:
    physics_ode_metrics.csv
    """

    rows = []

    for _, r in df.iterrows():
        model_name = safe_get(r, "model_name", "physics_ode")

        row = {
            "model_family": "Physics ODE baseline",
            "model_name": model_name,
            "source_name": "58_physics_multi_ode_baselines",
            "source_path": str(source_path),

            "horizon": safe_get(r, "horizon"),
            "target_mode": "recursive_ode",
            "feature_group": "physics_terms",
            "graph_type": model_name,
            "window": np.nan,
            "num_features": np.nan,

            "learned_w_identity": np.nan,
            "learned_w_distance": np.nan,
            "learned_w_corr": np.nan,

            "seq_residual_RMSE": safe_get(r, "seq_residual_RMSE", np.nan),
            "seq_residual_R2": safe_get(r, "seq_residual_R2", np.nan),
            "last_residual_RMSE": safe_get(r, "last_residual_RMSE", np.nan),
            "last_residual_R2": safe_get(r, "last_residual_R2", np.nan),

            "seq_sea_level_RMSE": safe_get(r, "seq_sea_level_RMSE", np.nan),
            "seq_sea_level_R2": safe_get(r, "seq_sea_level_R2", np.nan),
            "last_sea_level_RMSE": safe_get(r, "last_sea_level_RMSE", np.nan),
            "last_sea_level_R2": safe_get(r, "last_sea_level_R2", np.nan),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def normalize_physics_guided(df, source_path):
    """
    For 59_physics_guided_learnable_graph_gnn_bigru.py output:
    physics_guided_metrics.csv

    This file contains both:
    physics_* columns
    final_* columns

    We convert them into two rows:
    1. Physics ODE used inside hybrid
    2. Physics-guided learnable graph GNN
    """

    rows = []

    for _, r in df.iterrows():
        base_info = {
            "horizon": safe_get(r, "horizon"),
            "target_mode": "multi",
            "feature_group": safe_get(r, "feature_group", ""),
            "graph_type": "physics_guided",
            "window": safe_get(r, "window", np.nan),
            "num_features": safe_get(r, "num_features", np.nan),

            "learned_w_identity": safe_get(r, "learned_w_identity", np.nan),
            "learned_w_distance": safe_get(r, "learned_w_distance", np.nan),
            "learned_w_corr": safe_get(r, "learned_w_corr", np.nan),
        }

        physics_row = {
            "model_family": "Physics ODE in hybrid",
            "model_name": "Physics ODE used before GNN correction",
            "source_name": "59_physics_guided_learnable_graph_gnn_bigru",
            "source_path": str(source_path),
            **base_info,

            "seq_residual_RMSE": safe_get(r, "physics_seq_residual_RMSE", np.nan),
            "seq_residual_R2": safe_get(r, "physics_seq_residual_R2", np.nan),
            "last_residual_RMSE": safe_get(r, "physics_last_residual_RMSE", np.nan),
            "last_residual_R2": safe_get(r, "physics_last_residual_R2", np.nan),

            "seq_sea_level_RMSE": safe_get(r, "physics_seq_sea_level_RMSE", np.nan),
            "seq_sea_level_R2": safe_get(r, "physics_seq_sea_level_R2", np.nan),
            "last_sea_level_RMSE": safe_get(r, "physics_last_sea_level_RMSE", np.nan),
            "last_sea_level_R2": safe_get(r, "physics_last_sea_level_R2", np.nan),
        }

        final_row = {
            "model_family": "Physics-guided learnable graph GNN-BiGRU",
            "model_name": "Physics prediction + GNN correction",
            "source_name": "59_physics_guided_learnable_graph_gnn_bigru",
            "source_path": str(source_path),
            **base_info,

            "seq_residual_RMSE": safe_get(r, "final_seq_residual_RMSE", np.nan),
            "seq_residual_R2": safe_get(r, "final_seq_residual_R2", np.nan),
            "last_residual_RMSE": safe_get(r, "final_last_residual_RMSE", np.nan),
            "last_residual_R2": safe_get(r, "final_last_residual_R2", np.nan),

            "seq_sea_level_RMSE": safe_get(r, "final_seq_sea_level_RMSE", np.nan),
            "seq_sea_level_R2": safe_get(r, "final_seq_sea_level_R2", np.nan),
            "last_sea_level_RMSE": safe_get(r, "final_last_sea_level_RMSE", np.nan),
            "last_sea_level_R2": safe_get(r, "final_last_sea_level_R2", np.nan),
        }

        rows.append(physics_row)
        rows.append(final_row)

    return pd.DataFrame(rows)


# ============================================================
# Load all available result files
# ============================================================
def load_all_results(outputs_root: Path, args):
    loaded = []
    missing = []

    # 55 final selected pure GNN
    final_selected_path = Path(args.final_selected) if args.final_selected else find_file(
        outputs_root,
        exact_relative_paths=[
            "gnn_bigru_final_selected_config/final_selected_metrics.csv",
        ],
        glob_patterns=[
            "**/final_selected_metrics.csv",
            "**/*final*selected*metrics*.csv",
        ],
    )

    if final_selected_path and final_selected_path.exists():
        df = pd.read_csv(final_selected_path)
        loaded.append(
            normalize_standard_metrics(
                df=df,
                model_family="Pure GNN-BiGRU selected config",
                model_name="Horizon-specific pure GNN-BiGRU",
                source_name="55_GNN_BiGRU_final_selected_config",
                source_path=final_selected_path,
            )
        )
        print(f"Loaded pure GNN selected config: {final_selected_path}")
    else:
        missing.append("55 final_selected_metrics.csv")

    # 56 learnable graph
    learnable_path = Path(args.learnable_graph) if args.learnable_graph else find_file(
        outputs_root,
        exact_relative_paths=[
            "gnn_bigru_learnable_graph_fusion_v2/learnable_graph_metrics.csv",
            "gnn_bigru_learnable_graph_fusion/learnable_graph_metrics.csv",
        ],
        glob_patterns=[
            "**/learnable_graph_metrics.csv",
        ],
    )

    if learnable_path and learnable_path.exists():
        df = pd.read_csv(learnable_path)
        loaded.append(
            normalize_standard_metrics(
                df=df,
                model_family="Learnable graph GNN-BiGRU",
                model_name="Softmax graph fusion GNN-BiGRU",
                source_name="56_GNN_BiGRU_learnable_graph_fusion",
                source_path=learnable_path,
            )
        )
        print(f"Loaded learnable graph GNN: {learnable_path}")
    else:
        missing.append("56 learnable_graph_metrics.csv")

    # 57 prior-guided learnable graph
    prior_path = Path(args.prior_learnable_graph) if args.prior_learnable_graph else find_file(
        outputs_root,
        exact_relative_paths=[
            "gnn_bigru_prior_learnable_graph_fusion/prior_learnable_graph_metrics.csv",
        ],
        glob_patterns=[
            "**/prior_learnable_graph_metrics.csv",
        ],
    )

    if prior_path and prior_path.exists():
        df = pd.read_csv(prior_path)
        loaded.append(
            normalize_standard_metrics(
                df=df,
                model_family="Prior-guided learnable graph GNN-BiGRU",
                model_name="Prior-initialized graph fusion GNN-BiGRU",
                source_name="57_GNN_BiGRU_prior_learnable_graph_fusion",
                source_path=prior_path,
            )
        )
        print(f"Loaded prior-guided learnable graph GNN: {prior_path}")
    else:
        missing.append("57 prior_learnable_graph_metrics.csv")

    # 58 physics ODE baselines
    physics_ode_path = Path(args.physics_ode) if args.physics_ode else find_file(
        outputs_root,
        exact_relative_paths=[
            "physics_multi_ode_baselines/physics_ode_metrics.csv",
        ],
        glob_patterns=[
            "**/physics_ode_metrics.csv",
        ],
    )

    if physics_ode_path and physics_ode_path.exists():
        df = pd.read_csv(physics_ode_path)
        loaded.append(normalize_physics_ode(df=df, source_path=physics_ode_path))
        print(f"Loaded physics ODE baselines: {physics_ode_path}")
    else:
        missing.append("58 physics_ode_metrics.csv")

    # 59 physics-guided hybrid
    physics_guided_path = Path(args.physics_guided) if args.physics_guided else find_file(
        outputs_root,
        exact_relative_paths=[
            "physics_guided_learnable_graph_gnn_bigru/physics_guided_metrics.csv",
        ],
        glob_patterns=[
            "**/physics_guided_metrics.csv",
        ],
    )

    if physics_guided_path and physics_guided_path.exists():
        df = pd.read_csv(physics_guided_path)
        loaded.append(normalize_physics_guided(df=df, source_path=physics_guided_path))
        print(f"Loaded physics-guided GNN: {physics_guided_path}")
    else:
        missing.append("59 physics_guided_metrics.csv")

    if not loaded:
        raise FileNotFoundError(
            "没有找到任何可汇总的结果文件。请检查 outputs 文件夹，或者用命令行参数指定 CSV 路径。"
        )

    all_results = pd.concat(loaded, ignore_index=True)

    all_results["horizon"] = pd.to_numeric(all_results["horizon"], errors="coerce")
    all_results = all_results.dropna(subset=["horizon"])
    all_results["horizon"] = all_results["horizon"].astype(int)

    metric_cols = [
        "seq_residual_RMSE",
        "seq_residual_R2",
        "last_residual_RMSE",
        "last_residual_R2",
        "seq_sea_level_RMSE",
        "seq_sea_level_R2",
        "last_sea_level_RMSE",
        "last_sea_level_R2",
        "learned_w_identity",
        "learned_w_distance",
        "learned_w_corr",
    ]

    for col in metric_cols:
        all_results[col] = pd.to_numeric(all_results[col], errors="coerce")

    print("\nMissing files:")
    for item in missing:
        print(f"  - {item}")

    return all_results


# ============================================================
# Summary tables
# ============================================================
def make_best_tables(all_results: pd.DataFrame):
    valid = all_results.dropna(subset=["last_residual_RMSE"]).copy()

    best_by_last = (
        valid
        .sort_values(["horizon", "last_residual_RMSE"])
        .groupby("horizon", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    valid_seq = all_results.dropna(subset=["seq_residual_RMSE"]).copy()

    best_by_seq = (
        valid_seq
        .sort_values(["horizon", "seq_residual_RMSE"])
        .groupby("horizon", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    return best_by_last, best_by_seq


def make_recommended_strategy(all_results: pd.DataFrame):
    """
    Scientific recommendation based on your current experimental conclusion:

    1h:
        Pure GNN or learnable graph GNN is preferred because short-term residual
        is dominated by local persistence / identity graph.

    12h, 24h:
        Physics-guided GNN is preferred because physical ODE + GNN correction
        improves medium/long horizon prediction.
    """

    rows = []

    for horizon in sorted(all_results["horizon"].unique()):
        subset = all_results[all_results["horizon"] == horizon].copy()

        chosen = None
        reason = ""

        if horizon == 1:
            candidates = subset[
                subset["model_family"].isin(
                    [
                        "Learnable graph GNN-BiGRU",
                        "Pure GNN-BiGRU selected config",
                        "Prior-guided learnable graph GNN-BiGRU",
                    ]
                )
            ].dropna(subset=["last_residual_RMSE"])

            if len(candidates) > 0:
                chosen = candidates.sort_values("last_residual_RMSE").iloc[0]
                reason = (
                    "1h residual is mainly controlled by local persistence; "
                    "pure or learnable GNN is preferred over physics-guided hybrid."
                )

        elif horizon in [12, 24]:
            candidates = subset[
                subset["model_family"] == "Physics-guided learnable graph GNN-BiGRU"
            ].dropna(subset=["last_residual_RMSE"])

            if len(candidates) > 0:
                chosen = candidates.sort_values("last_residual_RMSE").iloc[0]
                reason = (
                    f"{horizon}h is a medium/long horizon; physics ODE provides "
                    "physical trajectory and GNN corrects its residual error."
                )

        if chosen is None:
            candidates = subset.dropna(subset=["last_residual_RMSE"])
            if len(candidates) > 0:
                chosen = candidates.sort_values("last_residual_RMSE").iloc[0]
                reason = "Fallback: selected by lowest last-step residual RMSE."

        if chosen is not None:
            row = chosen.to_dict()
            row["recommendation_reason"] = reason
            rows.append(row)

    return pd.DataFrame(rows)


def make_physics_improvement_table(all_results: pd.DataFrame):
    """
    Compare physics ODE inside hybrid vs final physics-guided GNN.
    """

    rows = []

    for horizon in sorted(all_results["horizon"].unique()):
        subset = all_results[all_results["horizon"] == horizon]

        physics = subset[subset["model_family"] == "Physics ODE in hybrid"]
        final = subset[subset["model_family"] == "Physics-guided learnable graph GNN-BiGRU"]

        if len(physics) == 0 or len(final) == 0:
            continue

        physics = physics.iloc[0]
        final = final.iloc[0]

        row = {
            "horizon": horizon,

            "physics_seq_residual_RMSE": physics["seq_residual_RMSE"],
            "final_seq_residual_RMSE": final["seq_residual_RMSE"],
            "seq_RMSE_improvement_percent": percent_improvement(
                physics["seq_residual_RMSE"],
                final["seq_residual_RMSE"],
            ),

            "physics_last_residual_RMSE": physics["last_residual_RMSE"],
            "final_last_residual_RMSE": final["last_residual_RMSE"],
            "last_RMSE_improvement_percent": percent_improvement(
                physics["last_residual_RMSE"],
                final["last_residual_RMSE"],
            ),

            "physics_seq_residual_R2": physics["seq_residual_R2"],
            "final_seq_residual_R2": final["seq_residual_R2"],
            "seq_R2_gain": final["seq_residual_R2"] - physics["seq_residual_R2"],

            "physics_last_residual_R2": physics["last_residual_R2"],
            "final_last_residual_R2": final["last_residual_R2"],
            "last_R2_gain": final["last_residual_R2"] - physics["last_residual_R2"],

            "learned_w_identity": final["learned_w_identity"],
            "learned_w_distance": final["learned_w_distance"],
            "learned_w_corr": final["learned_w_corr"],
        }

        rows.append(row)

    return pd.DataFrame(rows)


def make_key_model_table(all_results: pd.DataFrame):
    """
    Keep only the most important rows for PPT/report.
    """

    keep_families = [
        "Pure GNN-BiGRU selected config",
        "Learnable graph GNN-BiGRU",
        "Prior-guided learnable graph GNN-BiGRU",
        "Physics ODE in hybrid",
        "Physics-guided learnable graph GNN-BiGRU",
    ]

    key = all_results[all_results["model_family"].isin(keep_families)].copy()

    # Also include the best standalone physics ODE baseline from 58 for each horizon.
    ode = all_results[all_results["model_family"] == "Physics ODE baseline"].copy()

    if len(ode) > 0:
        best_ode = (
            ode.dropna(subset=["last_residual_RMSE"])
            .sort_values(["horizon", "last_residual_RMSE"])
            .groupby("horizon", as_index=False)
            .head(1)
        )

        best_ode = best_ode.copy()
        best_ode["model_family"] = "Best standalone Physics ODE baseline"

        key = pd.concat([key, best_ode], ignore_index=True)

    key = key.sort_values(["horizon", "model_family", "last_residual_RMSE"])

    return key


# ============================================================
# Plotting
# ============================================================
def save_bar_plot(df, metric_col, title, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过画图。可以运行：python -m pip install matplotlib")
        return

    plot_df = df.dropna(subset=[metric_col]).copy()

    if len(plot_df) == 0:
        print(f"No data to plot for {metric_col}")
        return

    # Use shorter labels.
    label_map = {
        "Pure GNN-BiGRU selected config": "Pure GNN",
        "Learnable graph GNN-BiGRU": "Learnable GNN",
        "Prior-guided learnable graph GNN-BiGRU": "Prior GNN",
        "Physics ODE in hybrid": "Physics ODE",
        "Physics-guided learnable graph GNN-BiGRU": "Physics-guided GNN",
        "Best standalone Physics ODE baseline": "Best ODE",
    }

    plot_df["label"] = plot_df["model_family"].map(label_map).fillna(plot_df["model_family"])
    plot_df["horizon_label"] = plot_df["horizon"].astype(str) + "h"

    horizons = sorted(plot_df["horizon"].unique())
    labels = list(dict.fromkeys(plot_df["label"].tolist()))

    x = np.arange(len(horizons))
    width = 0.8 / max(1, len(labels))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, label in enumerate(labels):
        values = []

        for h in horizons:
            sub = plot_df[(plot_df["horizon"] == h) & (plot_df["label"] == label)]

            if len(sub) == 0:
                values.append(np.nan)
            else:
                values.append(float(sub.iloc[0][metric_col]))

        ax.bar(x + (i - len(labels) / 2) * width + width / 2, values, width, label=label)

    ax.set_title(title)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(metric_col)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}h" for h in horizons])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root outputs folder containing all experiment result folders.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/final_model_comparison",
        help="Where to save final comparison tables and plots.",
    )

    # Optional manual paths if automatic search fails.
    parser.add_argument("--final-selected", default="")
    parser.add_argument("--learnable-graph", default="")
    parser.add_argument("--prior-learnable-graph", default="")
    parser.add_argument("--physics-ode", default="")
    parser.add_argument("--physics-guided", default="")

    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    output_dir = make_output_dir(Path(args.output_dir))

    print("=" * 100)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("=" * 100)
    print(f"outputs_root: {outputs_root.resolve()}")
    print(f"output_dir  : {output_dir.resolve()}")

    all_results = load_all_results(outputs_root=outputs_root, args=args)

    # Save full long table.
    all_results_path = output_dir / "all_model_comparison_long.csv"
    all_results.to_csv(all_results_path, index=False)
    print(f"\nSaved: {all_results_path}")

    # Key model table.
    key_models = make_key_model_table(all_results)
    key_models_path = output_dir / "key_model_comparison_for_ppt.csv"
    key_models.to_csv(key_models_path, index=False)
    print(f"Saved: {key_models_path}")

    # Best tables.
    best_by_last, best_by_seq = make_best_tables(all_results)

    best_by_last_path = output_dir / "best_model_by_horizon_last_residual_rmse.csv"
    best_by_seq_path = output_dir / "best_model_by_horizon_seq_residual_rmse.csv"

    best_by_last.to_csv(best_by_last_path, index=False)
    best_by_seq.to_csv(best_by_seq_path, index=False)

    print(f"Saved: {best_by_last_path}")
    print(f"Saved: {best_by_seq_path}")

    # Recommended strategy.
    recommended = make_recommended_strategy(all_results)
    recommended_path = output_dir / "recommended_horizon_specific_strategy.csv"
    recommended.to_csv(recommended_path, index=False)
    print(f"Saved: {recommended_path}")

    # Physics improvement.
    physics_improvement = make_physics_improvement_table(all_results)
    physics_improvement_path = output_dir / "physics_to_hybrid_improvement.csv"
    physics_improvement.to_csv(physics_improvement_path, index=False)
    print(f"Saved: {physics_improvement_path}")

    # Graph weights table.
    graph_weight_cols = [
        "horizon",
        "model_family",
        "feature_group",
        "learned_w_identity",
        "learned_w_distance",
        "learned_w_corr",
        "seq_residual_RMSE",
        "last_residual_RMSE",
    ]

    graph_weights = all_results[
        all_results["learned_w_identity"].notna()
    ][graph_weight_cols].copy()

    graph_weights_path = output_dir / "learned_graph_weights_summary.csv"
    graph_weights.to_csv(graph_weights_path, index=False)
    print(f"Saved: {graph_weights_path}")

    # Save plots.
    save_bar_plot(
        df=key_models,
        metric_col="seq_residual_RMSE",
        title="Sequence residual RMSE by model",
        output_path=output_dir / "comparison_seq_residual_RMSE.png",
    )

    save_bar_plot(
        df=key_models,
        metric_col="last_residual_RMSE",
        title="Last-step residual RMSE by model",
        output_path=output_dir / "comparison_last_residual_RMSE.png",
    )

    # Console display.
    display_cols = [
        "horizon",
        "model_family",
        "model_name",
        "feature_group",
        "seq_residual_RMSE",
        "seq_residual_R2",
        "last_residual_RMSE",
        "last_residual_R2",
        "learned_w_identity",
        "learned_w_distance",
        "learned_w_corr",
    ]

    print("\n" + "=" * 100)
    print("KEY MODEL COMPARISON")
    print("=" * 100)
    print(key_models[display_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("RECOMMENDED HORIZON-SPECIFIC STRATEGY")
    print("=" * 100)

    rec_cols = [
        "horizon",
        "model_family",
        "model_name",
        "feature_group",
        "seq_residual_RMSE",
        "seq_residual_R2",
        "last_residual_RMSE",
        "last_residual_R2",
        "recommendation_reason",
    ]

    if len(recommended) > 0:
        print(recommended[rec_cols].to_string(index=False))
    else:
        print("No recommended strategy generated.")

    print("\n" + "=" * 100)
    print("PHYSICS ODE -> PHYSICS-GUIDED GNN IMPROVEMENT")
    print("=" * 100)

    if len(physics_improvement) > 0:
        print(physics_improvement.to_string(index=False))
    else:
        print("No physics-guided comparison available.")

    print("\nFinished.")
    print(f"Final comparison files are saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()