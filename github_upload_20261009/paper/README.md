# Overleaf external-confirmation manuscript package

This package contains the revised attribution-and-transfer manuscript.

## Compile

1. Upload the complete directory to Overleaf.
2. Set `main.tex` as the main document.
3. Use pdfLaTeX and BibTeX.

## Evidence boundaries

- Retrospective aligned-forcing results and strict chronological results are reported separately.
- 2025 H2 is a chronological refit backtest that was viewed during development.
- The GWN/HS-DT 2026 evaluations were frozen; the later-added VARX overlay is diagnostic-only.
- Tier D is a prospectively locked spatial confirmation on ten non-overlapping Delaware Bay--River stations. It uses the same 2025 H2 calendar period as the original-region refit, so it is not a later-period temporal holdout or third-party preregistration.
- Two of the three locked Tier-D criteria pass. Expert complementarity and lead-dependent VARX/deep ordering replicate; the original-region supervision-specialization pattern does not.
- External HS-DT improves the mean sequence point estimate over Multistate GWN by 0.00453, but its 168-h block interval crosses zero. VARX-Ridge remains the strongest external sequence model.
- HS-DT is the deep sequence model, not an additional lead-24 improvement over Multistate GWN.
- Physics effects are attributed only against matched or same-capacity controls.
- Five-seed tests are descriptive; effect size, direction consistency, and 168-h block bootstrap carry the main statistical argument.
- Event PR-AUC is excluded from the mainline because no operational event task is defined.
- The descriptive q95 subset is not an operational storm-surge threshold.
- Smoke, partial, obsolete lambda=0.0003, and superseded PDF results are excluded.

`supplementary_results/claim_evidence_ledger.csv` records allowed and forbidden wording for each central claim. `supplementary_results/external_confirmation_protocol.json` is the immutable protocol copy used for Tier D. Its historical machine label `untouched_spatial_external_confirmation` means that the second-region targets had not been downloaded or scored before the local hash lock; it must not be expanded into a claim of public preregistration, third-party independence, or later-period temporal validation. The older `prospective_confirmation_manifest.json` concerns the prior original-region temporal plan and is not Tier-D evidence.

## Main conclusion

The evidence supports low-complexity expert complementarity and horizon-dependent linear/deep model ordering across two non-overlapping station networks. The exact original-region supervision pattern does not transfer, and physical-prior effects remain small and conditional. A publicly preregistered later-period or hydrographically more distant confirmation remains outstanding.
