from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT87 = Path(__file__).resolve().parent / "87_cycle_consistent_gnn_bigru.py"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "cycle_component_ablation"


COMPONENT_CONFIGS = [
    ("no_cycle_no_trend", 0.0, 0.0),
    ("trend_only", 0.0, 0.15),
    ("cycle_only", 0.02, 0.0),
    ("cycle_plus_trend", 0.02, 0.15),
    ("default_cycle_plus_trend", 0.04, 0.15),
]


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


def summarize(output_dir: Path) -> None:
    rows = []
    for name, cycle_lambda, trend_lambda in COMPONENT_CONFIGS:
        path = output_dir / name / "cycle_consistent_mean_std.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["component_config"] = name
        df["cycle_lambda"] = cycle_lambda
        df["trend_lambda"] = trend_lambda
        rows.append(df)

    summary_dir = output_dir / "paper_summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        (summary_dir / "cycle_component_ablation_summary.md").write_text(
            "# Cycle Component Ablation Summary\n\nNo completed component runs found.\n",
            encoding="utf-8",
        )
        return

    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(summary_dir / "cycle_component_ablation_mean_std.csv", index=False)

    keep = [
        "component_config",
        "cycle_lambda",
        "trend_lambda",
        "horizon",
        "last_residual_R2_mean",
        "last_residual_R2_std",
        "last_residual_RMSE_mean",
        "extreme_abs_q95_residual_R2_mean",
        "extreme_abs_q95_residual_R2_std",
        "seq_residual_R2_mean",
    ]
    keep = [c for c in keep if c in all_df.columns]
    table = all_df[keep].sort_values(["horizon", "last_residual_R2_mean"], ascending=[True, False])
    table.to_csv(summary_dir / "cycle_component_ablation_table.csv", index=False)

    lines = [
        "# Cycle Component Ablation Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Purpose",
        "",
        "This ablation separates the effect of the latent cycle-consistency loss from the trend-shape loss.",
        "",
        "## Results",
        "",
        markdown_table(table),
        "",
        "## Interpretation Guide",
        "",
        "- If `trend_only` is close to `cycle_plus_trend`, the gain mainly comes from trajectory-shape regularization.",
        "- If `cycle_only` improves over `no_cycle_no_trend`, the latent cycle mechanism has independent value.",
        "- If `cycle_plus_trend` is best, the two consistency losses are complementary.",
        "- Keep the Physical-loss model as the main model if it remains stronger on extreme q95 R2.",
        "",
    ]
    (summary_dir / "cycle_component_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cycle component ablation experiments")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--graph-mode", choices=["fixed", "learnable"], default="learnable")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        summarize(output_dir)
        print(f"Summary regenerated: {output_dir / 'paper_summary_tables' / 'cycle_component_ablation_summary.md'}")
        return

    commands: list[tuple[str, list[str]]] = []
    for name, cycle_lambda, trend_lambda in COMPONENT_CONFIGS:
        commands.append(
            (
                name,
                [
                    sys.executable,
                    "-u",
                    "-X",
                    "utf8",
                    str(SCRIPT87),
                    "--horizons",
                    str(args.horizon),
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
                    str(cycle_lambda),
                    "--trend-lambda",
                    str(trend_lambda),
                    "--last-step-weight",
                    "0.35",
                    "--output-dir",
                    str(output_dir / name),
                ],
            )
        )

    (output_dir / "command_manifest.txt").write_text(
        "\n\n".join(" ".join(cmd) for _, cmd in commands),
        encoding="utf-8",
    )
    if args.dry_run:
        for name, cmd in commands:
            print(f"\n[{name}]\n" + " ".join(cmd))
        return

    for name, cmd in commands:
        run_command(cmd, ROOT, output_dir / "logs" / f"{name}.log")

    summarize(output_dir)
    print("\nFinished Cycle component ablation.")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
