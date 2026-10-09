# Release verification

Verified locally on 2026-10-09:

- Python syntax compilation for public source, scripts, and tests.
- Seven synthetic invariant tests passed.
- Expert-training and C4 CLI help imported successfully.
- The result viewer read the saved CSVs and rendered its comparison figure.
- Missing-data preflight correctly refused full training without the five required processed inputs.
- With identical state dictionaries and synthetic inputs, renamed Eta GWN and Multistate GWN outputs matched the original Tier-A implementations exactly (maximum absolute difference 0.0).
- C4 state-dict keys and checked outputs (mean, expert predictions, gate, correction, and log scale) matched the original aligned implementation exactly.

The BatchNorm invariant calls `keep_frozen_experts_eval` after `model.train()`, as the original training loop does. Freezing gradients alone is not sufficient to freeze running statistics.

These checks validate code organization and selected model invariants. They are not a new data audit, model retraining, or empirical generalization test. GitHub Actions is configured for Python 3.11; that hosted run has not yet occurred in this release.
