from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT87 = Path(__file__).resolve().parent / "87_cycle_consistent_gnn_bigru.py"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "cycle_paper_longrun_experiments"

EXISTING_FOUR_MODEL = (
    Path(__file__).resolve().parent
    / "outputs"
    / "final_validation_tasks"
    / "task2_all_models_horizons_seed42"
    / "final_four_models_enhanced_key_metrics.csv"
)
EXISTING_MULTI_24H = (
    Path(__file__).resolve().parent
    / "outputs"
    / "journal_longrun_experiments"
    / "multiseed_24h"
    / "task_1_multiseed_mean_std.csv"
)
EXISTING_SIMPLE_BASELINES = (
    Path(__file__).resolve().parent
    / "outputs"
    / "simple_baselines_for_paper"
    / "simple_baselines_key_metrics.csv"
)


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(cmd))
    print(f"Log: {log_path}")
    with log_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        code = proc.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows found._"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in out.to_numpy()]
    return "\n".join([header, sep, *rows])


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def collect_cycle_horizon_metrics(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "cycle_horizons_multiseed" / "cycle_consistent_mean_std.csv"
    df = read_csv_if_exists(path)
    if df.empty:
        return df
    df = df.copy()
    df["experiment_group"] = "Cycle horizons, multi-seed"
    return df


def collect_cycle_lambda_metrics(output_dir: Path) -> pd.DataFrame:
    frames = []
    lambda_root = output_dir / "cycle_lambda_ablation_24h"
    for child in sorted(lambda_root.glob("lambda_*")):
        path = child / "cycle_consistent_mean_std.csv"
        df = read_csv_if_exists(path)
        if df.empty:
            continue
        df = df.copy()
        try:
            df["cycle_lambda"] = float(child.name.replace("lambda_", "").replace("p", "."))
        except ValueError:
            df["cycle_lambda"] = child.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(lambda_root / "cycle_lambda_ablation_mean_std.csv", index=False)
    return out


def make_combined_tables(output_dir: Path) -> None:
    summary_dir = output_dir / "paper_summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    cycle_h = collect_cycle_horizon_metrics(output_dir)
    cycle_lambda = collect_cycle_lambda_metrics(output_dir)
    four = read_csv_if_exists(EXISTING_FOUR_MODEL)
    multi24 = read_csv_if_exists(EXISTING_MULTI_24H)
    simple = read_csv_if_exists(EXISTING_SIMPLE_BASELINES)

    if not cycle_h.empty:
        keep = [
            "horizon",
            "model_name",
            "seq_residual_R2_mean",
            "seq_residual_R2_std",
            "last_residual_R2_mean",
            "last_residual_R2_std",
            "last_residual_RMSE_mean",
            "last_residual_RMSE_std",
            "extreme_abs_q95_residual_R2_mean",
            "extreme_abs_q95_residual_R2_std",
            "last_sea_level_R2_mean",
            "last_sea_level_R2_std",
        ]
        keep = [c for c in keep if c in cycle_h.columns]
        cycle_h[keep].to_csv(summary_dir / "cycle_horizons_multiseed_table.csv", index=False)

    if not cycle_lambda.empty:
        keep = [
            "cycle_lambda",
            "horizon",
            "model_name",
            "last_residual_R2_mean",
            "last_residual_R2_std",
            "last_residual_RMSE_mean",
            "extreme_abs_q95_residual_R2_mean",
            "extreme_abs_q95_residual_R2_std",
        ]
        keep = [c for c in keep if c in cycle_lambda.columns]
        cycle_lambda[keep].sort_values(["horizon", "cycle_lambda"]).to_csv(
            summary_dir / "cycle_lambda_ablation_table.csv", index=False
        )

    comparison_rows = []
    if not multi24.empty:
        for _, row in multi24.iterrows():
            comparison_rows.append(
                {
                    "horizon": row.get("horizon"),
                    "model_name": row.get("model_name"),
                    "result_source": "existing 24h multi-seed long run",
                    "last_residual_R2_mean": row.get("last_residual_R2_mean"),
                    "last_residual_R2_std": row.get("last_residual_R2_std"),
                    "last_residual_RMSE_mean": row.get("last_residual_RMSE_mean"),
                    "extreme_abs_q95_residual_R2_mean": row.get("extreme_abs_q95_residual_R2_mean"),
                    "extreme_abs_q95_residual_R2_std": row.get("extreme_abs_q95_residual_R2_std"),
                }
            )
    if not cycle_h.empty:
        cycle24 = cycle_h[cycle_h["horizon"] == 24]
        for _, row in cycle24.iterrows():
            comparison_rows.append(
                {
                    "horizon": row.get("horizon"),
                    "model_name": row.get("model_name"),
                    "result_source": "new Cycle 24h multi-seed long run",
                    "last_residual_R2_mean": row.get("last_residual_R2_mean"),
                    "last_residual_R2_std": row.get("last_residual_R2_std"),
                    "last_residual_RMSE_mean": row.get("last_residual_RMSE_mean"),
                    "extreme_abs_q95_residual_R2_mean": row.get("extreme_abs_q95_residual_R2_mean"),
                    "extreme_abs_q95_residual_R2_std": row.get("extreme_abs_q95_residual_R2_std"),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    if not comparison.empty:
        comparison.to_csv(summary_dir / "main_24h_multiseed_comparison_with_cycle.csv", index=False)

    if not four.empty and not cycle_h.empty:
        cycle_seedless = cycle_h.copy()
        rename = {
            "seq_residual_R2_mean": "seq_residual_R2",
            "last_residual_R2_mean": "last_residual_R2",
            "last_residual_RMSE_mean": "last_residual_RMSE",
            "extreme_abs_q95_residual_R2_mean": "extreme_abs_q95_residual_R2",
            "last_sea_level_R2_mean": "last_sea_level_R2",
        }
        cycle_seedless = cycle_seedless.rename(columns=rename)
        cycle_seedless = cycle_seedless[
            [
                c
                for c in [
                    "horizon",
                    "model_name",
                    "seq_residual_R2",
                    "last_residual_R2",
                    "last_residual_RMSE",
                    "extreme_abs_q95_residual_R2",
                    "last_sea_level_R2",
                ]
                if c in cycle_seedless.columns
            ]
        ]
        combined = pd.concat([four[cycle_seedless.columns], cycle_seedless], ignore_index=True)
        combined.sort_values(["horizon", "last_residual_R2"], ascending=[True, False]).to_csv(
            summary_dir / "five_model_horizon_comparison_seed42_plus_cycle_mean.csv", index=False
        )

    lines = [
        "# Cycle Paper Long-Run Experiment Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Purpose",
        "",
        "This package adds the Cycle-consistent GNN-BiGRU experiments needed for paper-level evidence:",
        "multi-horizon evaluation, multi-seed stability, and cycle-loss weight ablation.",
        "",
    ]
    if not cycle_h.empty:
        lines += [
            "## Cycle Multi-Horizon Multi-Seed Results",
            "",
            markdown_table(
                cycle_h[
                    [
                        c
                        for c in [
                            "horizon",
                            "model_name",
                            "last_residual_R2_mean",
                            "last_residual_R2_std",
                            "last_residual_RMSE_mean",
                            "extreme_abs_q95_residual_R2_mean",
                            "extreme_abs_q95_residual_R2_std",
                        ]
                        if c in cycle_h.columns
                    ]
                ]
            ),
            "",
        ]
    if not comparison.empty:
        lines += [
            "## Main 24h Multi-Seed Comparison",
            "",
            markdown_table(comparison),
            "",
        ]
    if not cycle_lambda.empty:
        lines += [
            "## Cycle Lambda Ablation",
            "",
            markdown_table(
                cycle_lambda[
                    [
                        c
                        for c in [
                            "cycle_lambda",
                            "horizon",
                            "last_residual_R2_mean",
                            "last_residual_R2_std",
                            "extreme_abs_q95_residual_R2_mean",
                        ]
                        if c in cycle_lambda.columns
                    ]
                ].sort_values(["horizon", "cycle_lambda"])
            ),
            "",
        ]
    if not simple.empty:
        lines += [
            "## Existing Simple Baseline File",
            "",
            f"Source: `{EXISTING_SIMPLE_BASELINES}`",
            "",
        ]
    lines += [
        "## Paper Interpretation",
        "",
        "- If Cycle improves over plain GNN-BiGRU mainly at 24h, describe it as a long-horizon trajectory-consistency baseline.",
        "- If Physical-loss remains stronger on extreme q95 R2, keep Physical-loss GNN-BiGRU as the main model.",
        "- The lambda ablation should be used to show that the cycle branch is helpful only at an appropriate small weight.",
        "- Do not claim Cycle uses physical information; it is a pure data-driven consistency model.",
        "",
    ]
    (summary_dir / "cycle_paper_longrun_summary.md").write_text("\n".join(lines), encoding="utf-8")


def lambda_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-level Cycle-consistent GNN-BiGRU long experiments")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--horizons", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--cycle-lambda", type=float, default=0.04)
    parser.add_argument("--trend-lambda", type=float, default=0.15)
    parser.add_argument("--last-step-weight", type=float, default=0.35)
    parser.add_argument("--lambda-ablation-values", type=float, nargs="+", default=[0.0, 0.02, 0.04, 0.08])
    parser.add_argument("--lambda-ablation-horizon", type=int, default=24)
    parser.add_argument("--graph-mode", choices=["fixed", "learnable"], default="learnable")
    parser.add_argument("--skip-horizons", action="store_true")
    parser.add_argument("--skip-lambda-ablation", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        make_combined_tables(output_dir)
        print(f"Summary regenerated: {output_dir / 'paper_summary_tables' / 'cycle_paper_longrun_summary.md'}")
        return

    commands: list[tuple[str, list[str]]] = []
    if not args.skip_horizons:
        commands.append(
            (
                "cycle_horizons_multiseed",
                [
                    sys.executable,
                    "-u",
                    "-X",
                    "utf8",
                    str(SCRIPT87),
                    "--horizons",
                    *[str(h) for h in args.horizons],
                    "--seeds",
                    *[str(seed) for seed in args.seeds],
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--batch-size",
                    str(args.batch_size),
                    "--train-stride",
                    str(args.train_stride),
                    "--graph-mode",
                    args.graph_mode,
                    "--cycle-lambda",
                    str(args.cycle_lambda),
                    "--trend-lambda",
                    str(args.trend_lambda),
                    "--last-step-weight",
                    str(args.last_step_weight),
                    "--output-dir",
                    str(output_dir / "cycle_horizons_multiseed"),
                ],
            )
        )

    if not args.skip_lambda_ablation:
        for value in args.lambda_ablation_values:
            commands.append(
                (
                    f"cycle_lambda_ablation_24h/lambda_{lambda_label(value)}",
                    [
                        sys.executable,
                        "-u",
                        "-X",
                        "utf8",
                        str(SCRIPT87),
                        "--horizons",
                        str(args.lambda_ablation_horizon),
                        "--seeds",
                        *[str(seed) for seed in args.seeds],
                        "--epochs",
                        str(args.epochs),
                        "--patience",
                        str(args.patience),
                        "--batch-size",
                        str(args.batch_size),
                        "--train-stride",
                        str(args.train_stride),
                        "--graph-mode",
                        args.graph_mode,
                        "--cycle-lambda",
                        str(value),
                        "--trend-lambda",
                        str(args.trend_lambda),
                        "--last-step-weight",
                        str(args.last_step_weight),
                        "--output-dir",
                        str(output_dir / "cycle_lambda_ablation_24h" / f"lambda_{lambda_label(value)}"),
                    ],
                )
            )

    manifest = output_dir / "command_manifest.txt"
    manifest.write_text("\n\n".join(" ".join(cmd) for _, cmd in commands), encoding="utf-8")
    print(f"Command manifest written to: {manifest}")

    if args.dry_run:
        print("\nDry run only. Commands:")
        for name, cmd in commands:
            print(f"\n[{name}]\n" + " ".join(cmd))
        return

    for name, cmd in commands:
        run_command(cmd, ROOT, output_dir / "logs" / f"{name.replace('/', '_')}.log")

    make_combined_tables(output_dir)
    print("\nFinished Cycle paper long-run experiments.")
    print(f"Saved to: {output_dir}")
    print(f"Summary: {output_dir / 'paper_summary_tables' / 'cycle_paper_longrun_summary.md'}")


if __name__ == "__main__":
    main()
