from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT76 = Path(__file__).resolve().parent / "76_enhanced_forcing_physics_loss_ablation_v3.py"
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "physics_ablation_24h_5seed_matched"


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("\n" + "=" * 100)
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def summarize(output_dir: Path) -> None:
    frames = []
    for metrics_path in sorted(output_dir.glob("seed_*/v3_ablation_metrics.csv")):
        seed = int(metrics_path.parent.name.replace("seed_", ""))
        df = pd.read_csv(metrics_path)
        df["seed"] = seed
        frames.append(df)
    if not frames:
        print("No completed seed metrics found.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(output_dir / "physics_ablation_all_seed_metrics.csv", index=False)
    metric_cols = [
        "seq_residual_R2",
        "last_residual_R2",
        "last_residual_RMSE",
        "extreme_abs_q95_residual_R2",
        "extreme_abs_q95_residual_RMSE",
    ]
    metric_cols = [c for c in metric_cols if c in all_df.columns]
    agg = all_df.groupby(["horizon", "mode"])[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(x for x in col if x) for col in agg.columns.to_flat_index()]
    agg.to_csv(output_dir / "physics_ablation_mean_std.csv", index=False)

    baseline = agg[agg["mode"] == "no_physics"].set_index("horizon")
    gains = []
    for _, row in agg.iterrows():
        horizon = row["horizon"]
        if horizon not in baseline.index:
            continue
        base = baseline.loc[horizon]
        gains.append(
            {
                "horizon": horizon,
                "mode": row["mode"],
                "last_R2_gain_vs_multistate_no_physics": row.get("last_residual_R2_mean", float("nan"))
                - base.get("last_residual_R2_mean", float("nan")),
                "q95_R2_gain_vs_multistate_no_physics": row.get("extreme_abs_q95_residual_R2_mean", float("nan"))
                - base.get("extreme_abs_q95_residual_R2_mean", float("nan")),
            }
        )
    pd.DataFrame(gains).to_csv(output_dir / "physics_ablation_incremental_gains.csv", index=False)
    print("\nPhysics ablation summary:")
    print(agg.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched multi-seed physics-state and physics-loss ablation.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024, 2025, 3407])
    parser.add_argument("--horizons", type=int, nargs="+", default=[24])
    parser.add_argument("--modes", nargs="+", default=["no_physics", "eta_only", "eta_uv", "eta_uvW"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--physics-lambda-max", type=float, default=0.0003)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(output_dir)
        return

    commands = []
    for seed in args.seeds:
        cmd = [
            sys.executable,
            "-u",
            "-X",
            "utf8",
            str(SCRIPT76),
            "--horizons",
            *[str(h) for h in args.horizons],
            "--modes",
            *args.modes,
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--train-stride",
            str(args.train_stride),
            "--physics-lambda-max",
            str(args.physics_lambda_max),
            "--physics-forcing-mode",
            "last_input",
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir / f"seed_{seed}"),
        ]
        commands.append(cmd)
        run_command(cmd, args.dry_run)

    (output_dir / "command_manifest.txt").write_text(
        "\n\n".join(" ".join(cmd) for cmd in commands), encoding="utf-8"
    )
    if not args.dry_run:
        summarize(output_dir)


if __name__ == "__main__":
    main()
