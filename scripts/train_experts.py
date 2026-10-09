from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


ROOT = REPO_ROOT
SCRIPTS = ROOT / "scripts"
BASE_RESULTS = ROOT / "results" / "tier_a_graph_wavenet"
HSDT_RESULTS = ROOT / "results" / "tier_a_hs_dt_gwn"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the locked retrospective Tier-A GWN experts and fixed HS-DT-GWN."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    resume_flag = "--no-resume" if args.no_resume else "--resume"
    run(
        [
            sys.executable,
            str(REPO_ROOT / 'src/training/train_graph_experts.py'),
            "--mode",
            "run",
            "--output-dir",
            str(BASE_RESULTS),
            "--seeds",
            *map(str, args.seeds),
            "--configs",
            "gwn_eta_only",
            "gwn_multistate_no_physics",
            "--horizon",
            "24",
            "--window",
            "24",
            "--train-ratio",
            "0.70",
            "--val-ratio",
            "0.15",
            "--train-stride",
            "8",
            "--fixed-graph-type",
            "distance",
            "--hidden-dim",
            "64",
            "--diffusion-steps",
            "2",
            "--gwn-blocks",
            "6",
            "--dropout",
            "0.15",
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--lr",
            "0.0005",
            "--weight-decay",
            "0.00001",
            "--grad-clip",
            "1.0",
            "--aux-weight",
            "0.08",
            "--last-step-weight",
            "0.20",
            "--physics-lambda",
            "0.0002",
            "--physics-forcing-mode",
            "last_input",
            "--cpu-threads",
            str(args.cpu_threads),
            "--no-reuse-priority1-eta",
            resume_flag,
        ]
    )

    run(
        [
            sys.executable,
            str(REPO_ROOT / 'src/models/ensemble/hsdt.py'),
            "--mode",
            "run",
            "--source-results",
            str(BASE_RESULTS),
            "--output-dir",
            str(HSDT_RESULTS),
            "--seeds",
            *map(str, args.seeds),
            "--horizon",
            "24",
            "--window",
            "24",
            "--train-ratio",
            "0.70",
            "--val-ratio",
            "0.15",
        ]
    )

    print(f"Base-model results: {BASE_RESULTS}")
    print(f"HS-DT results:      {HSDT_RESULTS}")


if __name__ == "__main__":
    main()
