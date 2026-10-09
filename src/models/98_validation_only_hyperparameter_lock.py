from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "outputs"
    / "priority_top3_convincing_experiments_v2"
    / "priority_top3_metrics.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock physics hyperparameters using validation score only.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default="数据整理/outputs/validation_only_locked_config")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024, 2025, 3407])
    args = parser.parse_args()

    source = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source)
    required = {"horizon", "best_val_score", "physics_lambda_max"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    candidates = df[(df["horizon"] == args.horizon) & df["best_val_score"].notna()].copy()
    if candidates.empty:
        raise RuntimeError(f"No validation candidates for horizon={args.horizon}")
    locked = candidates.sort_values("best_val_score", ascending=True).iloc[0]
    locked.to_frame().T.to_csv(output_dir / "locked_physics_config.csv", index=False)

    lam = float(locked["physics_lambda_max"])
    command = (
        '.\\数据整理\\.venv\\Scripts\\python.exe -u -X utf8 '
        '"数据整理\\79_run_final_validation_tasks.py" --skip-task2 '
        f'--task1-horizon {args.horizon} '
        f'--task1-seeds {" ".join(str(x) for x in args.seeds)} '
        '--task1-models gnn_bigru learnable_graph ode_based_learnable physical_loss '
        '--epochs 120 --patience 20 --batch-size 256 --train-stride 8 '
        f'--physics-lambda-max {lam:g} '
        '--output-dir "数据整理/outputs/final_four_models_24h_5seed_locked"'
    )
    (output_dir / "run_locked_test_command.ps1.txt").write_text(command + "\n", encoding="utf-8")

    print("Validation-only locked configuration:")
    print(locked.to_string())
    print("\nRun this command for the final locked test:")
    print(command)
    print("\nDo not choose the final configuration using test R2/RMSE columns.")


if __name__ == "__main__":
    main()
