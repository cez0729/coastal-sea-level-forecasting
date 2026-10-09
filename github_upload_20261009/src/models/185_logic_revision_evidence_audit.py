"""Build the manuscript logic audit and freeze prospective confirmation assets."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "results" / "manuscript_logic_revision_20260811"
SEEDS = [42, 123, 2024, 2025, 3407]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_claim_ledger(output: Path) -> pd.DataFrame:
    specialization = json.loads(
        (ROOT / "results" / "horizon_specialization_complementarity_20260811" / "SPECIALIZATION_DECISION.json").read_text(
            encoding="utf-8"
        )
    )
    h2_boot = pd.read_csv(
        ROOT / "results" / "confirmatory_hsdt_orc_refit_2025_h2" / "block_bootstrap_comparisons.csv"
    )
    later_paired = pd.read_csv(
        ROOT / "results" / "formal_hsdt_independent_validation_2026" / "paired_comparisons.csv"
    )
    later_boot = pd.read_csv(
        ROOT / "results" / "formal_hsdt_independent_validation_2026" / "block_bootstrap_summary.csv"
    )
    physics = pd.read_csv(
        ROOT / "results" / "architecture_conditional_physics_utility_20260811" / "strict_physics_utility_matrix.csv"
    )
    temporal = pd.read_csv(
        ROOT / "results" / "frozen_varx_2026_diagnostic_20260811" / "model_summary.csv"
    )

    rows: list[dict] = [
        {
            "claim_id": "expert_structure",
            "evidence_status": "tier_b_refit_backtest",
            "effect": (
                f"eta leads={specialization['strict_negative_HSI_leads']}; "
                f"multistate leads={specialization['strict_positive_HSI_leads']}; "
                f"mean error correlation={specialization['strict_mean_error_correlation']:.6f}"
            ),
            "primary_uncertainty": "five-seed lead/station variation",
            "allowed_wording": "supervision-associated, modest, horizon-varying and station-dependent",
            "forbidden_wording": "causal supervision effect or uniform late-horizon specialization",
        }
    ]

    h2 = h2_boot[
        (h2_boot["comparison"] == "hsdt_minus_multistate") & (h2_boot["metric"] == "sequence_R2")
    ].iloc[0]
    rows.append(
        {
            "claim_id": "hsdt_sequence_2025_h2",
            "evidence_status": "tier_b_refit_backtest",
            "effect": f"delta R2={h2['point_mean_delta']:+.6f}",
            "primary_uncertainty": f"168-h block CI [{h2['ci95_low']:.6f}, {h2['ci95_high']:.6f}]",
            "allowed_wording": "positive sequence complementarity in the refit backtest",
            "forbidden_wording": "untouched holdout confirmation",
        }
    )
    for period in ("2026_h1", "2026_july"):
        paired = later_paired[
            (later_paired["period"] == period)
            & (later_paired["candidate"] == "hs_dt_gwn")
            & (later_paired["baseline"] == "gwn_multistate_no_physics")
            & (later_paired["metric"] == "seq_residual_R2")
        ].iloc[0]
        boot = later_boot[
            (later_boot["period"] == period)
            & (later_boot["baseline"] == "multi")
            & (later_boot["metric"] == "sequence")
        ].iloc[0]
        rows.append(
            {
                "claim_id": f"hsdt_sequence_{period}",
                "evidence_status": "tier_c_frozen_deep_evaluation",
                "effect": f"paired mean delta R2={paired['mean_improvement']:+.6f}; wins={int(paired['wins'])}/5",
                "primary_uncertainty": f"168-h block CI [{boot['ci95_low']:.6f}, {boot['ci95_high']:.6f}]",
                "allowed_wording": "period-consistent frozen deep-expert sequence gain",
                "forbidden_wording": "universal best model or extra Lead-24 gain",
            }
        )

    matched = physics[
        (physics["attribution_class"] == "matched_physics_increment")
        & (physics["metric"].isin(["seq_residual_R2", "last_residual_R2", "extreme_abs_q95_residual_R2"]))
    ]
    max_seq = matched[matched["metric"] == "seq_residual_R2"].sort_values("mean_delta", ascending=False).iloc[0]
    lead24_positive = int(
        ((matched["metric"] == "last_residual_R2") & (matched["mean_delta"] > 0)).sum()
    )
    rows.append(
        {
            "claim_id": "matched_physics_utility",
            "evidence_status": "tier_b_refit_backtest_matched_attribution",
            "effect": (
                f"largest matched sequence delta={max_seq['mean_delta']:+.6f}; "
                f"positive Lead-24 rows={lead24_positive}/{int((matched['metric'] == 'last_residual_R2').sum())}"
            ),
            "primary_uncertainty": "effect size, seed direction, and matched-capacity design; exact p is descriptive",
            "allowed_wording": "small, mixed, architecture- and task-conditional marginal utility",
            "forbidden_wording": "physics explains joint-model gains or the 0.6153 terminal score",
        }
    )

    # VARX was added after the 2026 targets were viewed; this row deliberately
    # records the diagnostic status rather than upgrading it to confirmation.
    later = temporal[temporal["period"].isin(["2026_h1", "2026_july"])]
    rows.append(
        {
            "claim_id": "linear_deep_hierarchy",
            "evidence_status": "tier_b_plus_tier_c_diagnostic_overlay",
            "effect": "; ".join(
                f"{period}: VARX={group.loc[group['model']=='varx_ridge','seq_residual_R2'].iloc[0]:.4f}, "
                f"HS-DT={group.loc[group['model']=='hs_dt_gwn','seq_residual_R2'].iloc[0]:.4f}"
                for period, group in later.groupby("period")
            ),
            "primary_uncertainty": "period-wise lead curves; 2026 VARX is diagnostic-only",
            "allowed_wording": "observed period- and horizon-dependent hierarchy change",
            "forbidden_wording": "independently confirmed temporal reversal",
        }
    )
    ledger = pd.DataFrame(rows)
    ledger.to_csv(output / "claim_evidence_ledger.csv", index=False)
    return ledger


def freeze_varx(output: Path) -> tuple[Path, dict]:
    p180 = load_module("p180_logic_freeze", HERE / "180_strict_varx_chronological_validation.py")
    args = argparse.Namespace(
        data_dir="data/processed_multiyear_2023_2025",
        fold_train_end="2025-01-01",
        fold_val_end="2025-07-01",
        fold_test_end="2026-01-01",
        window=24,
        horizon=24,
        train_stride=8,
        physics_forcing_mode="last_input",
        extreme_quantile=0.90,
    )
    p180.p134.configure_data_dir(args.data_dir)
    data = p180.p134.rolling.build_fold_data(args, args.horizon, add_ode_prior=False)
    train_x, train_y, _ = p180.p170.design_matrix(data["single_train"])
    test_x, _, _ = p180.p170.design_matrix(data["single_test"])
    model = Ridge(alpha=100.0, solver="lsqr").fit(train_x, train_y)
    pred = model.predict(test_x).reshape(-1, 7, 24).astype(np.float32)
    reference_path = ROOT / "results" / "strict_varx_chronological_validation_2025_h2" / "strict_varx_predictions.npz"
    with np.load(reference_path, allow_pickle=False) as payload:
        reference = payload["pred_residual"]
    max_abs_difference = float(np.max(np.abs(pred - reference)))
    if max_abs_difference > 1e-5:
        raise RuntimeError(f"Frozen VARX reconstruction mismatch: {max_abs_difference}")
    artifact = output / "frozen_varx_2023_2024_alpha100.npz"
    np.savez_compressed(
        artifact,
        coef=model.coef_.astype(np.float64),
        intercept=np.asarray(model.intercept_, dtype=np.float64),
        alpha=np.asarray([100.0], dtype=np.float64),
        n_features_in=np.asarray([model.n_features_in_], dtype=np.int64),
    )
    return artifact, {
        "alpha": 100.0,
        "solver": "lsqr",
        "training_end_exclusive": "2025-01-01",
        "selection_period_end_exclusive": "2025-07-01",
        "window_hours": 24,
        "horizon_hours": 24,
        "reconstruction_max_abs_difference": max_abs_difference,
    }


def build_confirmation_manifest(output: Path, varx_artifact: Path, varx_metadata: dict) -> dict:
    files = [
        HERE / "134_confirmatory_hsdt_orc_chronological_refit.py",
        HERE / "179_horizon_specialization_complementarity.py",
        HERE / "180_strict_varx_chronological_validation.py",
        HERE / "183_frozen_varx_2026_diagnostic.py",
        varx_artifact,
    ]
    for seed in SEEDS:
        root = ROOT / "results" / "confirmatory_hsdt_orc_refit_2025_h2" / f"seed_{seed}"
        files.extend(
            [
                root / "gwn_eta_only" / "best_checkpoint.pt",
                root / "gwn_multistate_no_physics" / "best_checkpoint.pt",
            ]
        )
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing files: {missing}")
    artifacts = [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    manifest = {
        "manifest_version": "2026-08-11-logic-revision-1",
        "purpose": "prospective untouched temporal or regional confirmation; not a completed validation",
        "current_evidence_boundary": {
            "2025_h2": "chronological_refit_backtest_already_viewed",
            "2026_deep": "frozen_evaluation",
            "2026_varx": "diagnostic_only_added_after_target_viewing",
        },
        "hs_dt_rule": {
            "lead_1_to_23": "0.5 * eta_only + 0.5 * multistate",
            "lead_24": "multistate",
            "lead_24_extra_gain_claim_allowed": False,
            "provenance": "formulated after Tier-A benchmark; frozen before later evaluation runs",
        },
        "varx": varx_metadata,
        "preprocessing_lock": {
            "scalers": "fit on training period only",
            "missing_values": "past-dependent causal filling",
            "graph": "distance support for refit checkpoints; any regional remapping must be declared before scoring",
            "future_residual_as_input": False,
        },
        "primary_metrics": ["sequence_residual_R2", "lead24_residual_R2", "lead24_RMSE"],
        "descriptive_metrics": ["absolute_q95_residual_R2"],
        "excluded_from_mainline": ["event_PR_AUC_without_operational_event_definition"],
        "acceptance_criteria": [
            "eta-only and multistate errors remain highly related but not identical across horizon/station",
            "HS-DT sequence delta versus both experts is reported with 168-h block uncertainty without requiring positivity",
            "VARX/deep ranking is evaluated by lead and period without target-selected gates or masks",
        ],
        "artifacts": artifacts,
    }
    (output / "prospective_confirmation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ledger = build_claim_ledger(output)
    varx_artifact, varx_metadata = freeze_varx(output)
    manifest = build_confirmation_manifest(output, varx_artifact, varx_metadata)
    decision = {
        "title_wording": "supervision-associated_not_supervision-induced",
        "event_metric_mainline": False,
        "primary_statistical_support": "effect_size_direction_consistency_and_168h_block_bootstrap",
        "claim_count": int(len(ledger)),
        "frozen_artifact_count": int(len(manifest["artifacts"])),
        "external_confirmation_completed": False,
    }
    (output / "LOGIC_REVISION_DECISION.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
