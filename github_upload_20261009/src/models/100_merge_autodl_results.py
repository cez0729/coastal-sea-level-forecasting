from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


def seed_from_path(path: Path) -> int:
    matches = re.findall(r"seed[_]?(\d+)", str(path))
    if not matches:
        raise ValueError(f"Cannot identify seed from {path}")
    return int(matches[-1])


def concat_csv(paths: list[Path], add_seed: bool = False) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if add_seed and "seed" not in df.columns:
            df["seed"] = seed_from_path(path)
        df["source_file"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_mean_std(df: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    available = [metric for metric in metrics if metric in df.columns]
    if df.empty or not available:
        return pd.DataFrame()
    out = df.groupby(groups)[available].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(x for x in col if x) for col in out.columns.to_flat_index()]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge distributed AutoDL experiment outputs.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/merged_publication")
    parser.add_argument("--run-evaluation", action="store_true")
    args = parser.parse_args()

    root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "extreme_abs_q95_residual_RMSE",
    ]

    lambda_df = concat_csv(sorted(root.glob("stage1_lambda/**/priority_top3_metrics.csv")))
    if not lambda_df.empty:
        lambda_df.to_csv(out / "lambda_search_all.csv", index=False)
        locked = lambda_df.sort_values("best_val_score", ascending=True).iloc[[0]]
        locked.to_csv(out / "validation_locked_lambda.csv", index=False)
        (out / "validation_locked_lambda.txt").write_text(
            f"{float(locked.iloc[0]['physics_lambda_max']):g}\n", encoding="utf-8"
        )

    final_df = concat_csv(
        sorted(root.glob("final/seed_*/final_four_models_enhanced_metrics.csv")), add_seed=True
    )
    if not final_df.empty:
        final_df.to_csv(out / "final_four_models_all_runs.csv", index=False)
        aggregate_mean_std(final_df, ["horizon", "model_name"], metrics).to_csv(
            out / "final_four_models_mean_std.csv", index=False
        )

    ablation_df = concat_csv(sorted(root.glob("ablation/seed_*/**/v3_ablation_metrics.csv")), add_seed=True)
    if not ablation_df.empty:
        ablation_df.to_csv(out / "physics_ablation_all_runs.csv", index=False)
        aggregate_mean_std(ablation_df, ["horizon", "mode"], metrics).to_csv(
            out / "physics_ablation_mean_std.csv", index=False
        )

    rolling_df = concat_csv(sorted(root.glob("rolling/seed_*/rolling_origin_metrics.csv")))
    if not rolling_df.empty:
        rolling_df.to_csv(out / "rolling_origin_all_runs.csv", index=False)
        aggregate_mean_std(rolling_df, ["fold", "model_name"], metrics).to_csv(
            out / "rolling_origin_mean_std.csv", index=False
        )

    baseline_df = concat_csv(sorted(root.glob("baselines/**/simple_baselines_metrics.csv")))
    if not baseline_df.empty:
        baseline_df = baseline_df.drop_duplicates(["horizon", "model_key"], keep="last")
        baseline_df.to_csv(out / "simple_baselines_all.csv", index=False)

    if args.run_evaluation:
        final_root = root / "final"
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-X",
                "utf8",
                "scripts/94_event_based_extreme_evaluation.py",
                "--input-root",
                str(final_root),
                "--output-dir",
                str(out / "event_metrics"),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-X",
                "utf8",
                "scripts/95_block_bootstrap_significance.py",
                "--input-root",
                str(final_root),
                "--block-hours",
                "168",
                "--bootstrap-n",
                "5000",
                "--output-dir",
                str(out / "block_bootstrap"),
            ],
            check=True,
        )

    print(f"Merged results written to {out.resolve()}")


if __name__ == "__main__":
    main()
