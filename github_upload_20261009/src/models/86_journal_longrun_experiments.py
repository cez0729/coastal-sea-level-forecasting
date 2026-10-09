from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT78 = Path(__file__).resolve().parent / "78_final_four_models_enhanced_data.py"
SCRIPT79 = Path(__file__).resolve().parent / "79_run_final_validation_tasks.py"
SCRIPT84 = Path(__file__).resolve().parent / "84_causal_preprocessing_robustness.py"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "journal_longrun_experiments"


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


def read_key_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def markdown_table(df: pd.DataFrame) -> str:
    """Small markdown table writer to avoid requiring the optional tabulate package."""

    if df.empty:
        return "_No rows found._"
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in text_df.to_numpy()]
    return "\n".join([header, sep, *rows])


def summarize_outputs(output_dir: Path, args) -> None:
    lines: list[str] = []
    lines.append("# Journal Long-Run Experiment Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This long-run package is designed to verify whether the paper's main conclusion "
        "is stable under longer training, multi-seed evaluation, and strict causal preprocessing."
    )
    lines.append("")

    multi_path = output_dir / "multiseed_24h" / "task_1_multiseed_mean_std.csv"
    if multi_path.exists():
        df = pd.read_csv(multi_path)
        lines.append("## 24h Multi-Seed Long Run")
        lines.append("")
        lines.append(f"Source: `{multi_path}`")
        lines.append("")
        keep = [
            "horizon",
            "model_name",
            "last_residual_R2_mean",
            "last_residual_R2_std",
            "last_residual_RMSE_mean",
            "extreme_abs_q95_residual_R2_mean",
            "extreme_abs_q95_residual_R2_std",
        ]
        keep = [c for c in keep if c in df.columns]
        lines.append(markdown_table(df[keep]))
        lines.append("")

    causal_path = output_dir / "causal_24h" / "causal_final_key_metrics.csv"
    if causal_path.exists():
        df = pd.read_csv(causal_path)
        lines.append("## 24h Strict Causal Preprocessing Long Run")
        lines.append("")
        lines.append(f"Source: `{causal_path}`")
        lines.append("")
        keep = [
            "horizon",
            "model_name",
            "seq_residual_R2",
            "last_residual_R2",
            "last_residual_RMSE",
            "extreme_abs_q95_residual_R2",
            "last_sea_level_R2",
        ]
        keep = [c for c in keep if c in df.columns]
        lines.append(markdown_table(df[keep]))
        lines.append("")

    full_path = output_dir / "full_four_models_6_12_24" / "final_four_models_enhanced_key_metrics.csv"
    if full_path.exists():
        df = pd.read_csv(full_path)
        lines.append("## Full Four-Model Long Run")
        lines.append("")
        lines.append(f"Source: `{full_path}`")
        lines.append("")
        keep = [
            "horizon",
            "model_name",
            "seq_residual_R2",
            "last_residual_R2",
            "last_residual_RMSE",
            "extreme_abs_q95_residual_R2",
            "last_sea_level_R2",
        ]
        keep = [c for c in keep if c in df.columns]
        lines.append(markdown_table(df[keep]))
        lines.append("")

    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- Use this as a final robustness check, not as a new model idea.")
    lines.append("- The key question is whether Physical-loss GNN-BiGRU remains better than GNN-BiGRU.")
    lines.append("- If longer training does not improve scores much, that is still useful: it supports the current early-stopping setup.")
    lines.append("- If causal preprocessing lowers absolute scores but preserves the physical-loss gain, that is strong evidence against leakage-driven conclusions.")
    lines.append("")
    (output_dir / "journal_longrun_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output_dir: Path, commands: list[list[str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "command_manifest.txt"
    manifest.write_text("\n\n".join(" ".join(cmd) for cmd in commands), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run journal-level long experiments")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--include-full-four-models", action="store_true")
    parser.add_argument("--skip-multiseed", action="store_true")
    parser.add_argument("--skip-causal", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        summarize_outputs(output_dir, args)
        print(f"Summary regenerated: {output_dir / 'journal_longrun_summary.md'}")
        return

    commands: list[tuple[str, list[str]]] = []
    if not args.skip_multiseed:
        commands.append(
            (
                "multiseed_24h",
                [
                    sys.executable,
                    "-u",
                    str(SCRIPT79),
                    "--skip-task2",
                    "--task1-horizon",
                    "24",
                    "--task1-seeds",
                    *[str(seed) for seed in args.seeds],
                    "--task1-models",
                    "gnn_bigru",
                    "physical_loss",
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--batch-size",
                    str(args.batch_size),
                    "--train-stride",
                    str(args.train_stride),
                    "--output-dir",
                    str(output_dir / "multiseed_24h"),
                ],
            )
        )

    if not args.skip_causal:
        commands.append(
            (
                "causal_24h",
                [
                    sys.executable,
                    "-u",
                    str(SCRIPT84),
                    "--horizons",
                    "24",
                    "--models",
                    "gnn_bigru",
                    "physical_loss",
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--batch-size",
                    str(args.batch_size),
                    "--train-stride",
                    str(args.train_stride),
                    "--output-dir",
                    str(output_dir / "causal_24h"),
                ],
            )
        )

    if args.include_full_four_models:
        commands.append(
            (
                "full_four_models_6_12_24",
                [
                    sys.executable,
                    "-u",
                    str(SCRIPT78),
                    "--horizons",
                    "6",
                    "12",
                    "24",
                    "--models",
                    "gnn_bigru",
                    "learnable_graph",
                    "ode_based_learnable",
                    "physical_loss",
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--batch-size",
                    str(args.batch_size),
                    "--train-stride",
                    str(args.train_stride),
                    "--seed",
                    "42",
                    "--output-dir",
                    str(output_dir / "full_four_models_6_12_24"),
                ],
            )
        )

    write_manifest(output_dir, [cmd for _, cmd in commands])
    print(f"Command manifest written to: {output_dir / 'command_manifest.txt'}")

    if args.dry_run:
        print("\nDry run only. Commands:")
        for name, cmd in commands:
            print(f"\n[{name}]")
            print(" ".join(cmd))
        return

    for name, cmd in commands:
        run_command(cmd, ROOT, output_dir / "logs" / f"{name}.log")

    summarize_outputs(output_dir, args)
    print(f"\nLong-run experiments finished. Summary: {output_dir / 'journal_longrun_summary.md'}")


if __name__ == "__main__":
    main()
